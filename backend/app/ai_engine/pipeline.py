import cv2
import base64
import json
import requests
from typing import Dict, Any

OLLAMA_URL = "http://localhost:11434/api/chat"

RULES_PATH = "backend/app/knowledge_base/rules_kz.json"
try:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        RULES_DB = json.load(f)
except Exception:
    RULES_DB = {}

def extract_frames_for_vlm(video_path: str, fps_sample: float = 0.2) -> list:
    """Реже берем кадры (например, 1 кадр в 5 секунд), чтобы не переполнять контекст токенов"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frames_data = []
    frame_interval = max(int(fps / fps_sample), 1)
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            timestamp_sec = round(current_frame / fps, 1)
            
            # Уменьшаем разрешение кадров до 640x360, чтобы экономить токены и видеопамять!
            frame_resized = cv2.resize(frame, (640, 360))
            
            annotated_frame = frame_resized.copy()
            cv2.putText(
                annotated_frame, 
                f"Time: {timestamp_sec}s", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
            
            _, buffer = cv2.imencode(".jpeg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            b64_annotated = base64.b64encode(buffer).decode("utf-8")
            
            # Чистый оригинальный кадр для evidence_frames в хорошем качестве
            _, clean_buffer = cv2.imencode(".jpeg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_clean = base64.b64encode(clean_buffer).decode("utf-8")
            
            frames_data.append({
                "timestamp_sec": timestamp_sec,
                "annotated_b64": b64_annotated,
                "clean_b64": b64_clean
            })
            
        current_frame += 1
        
    cap.release()
    return frames_data

def analyze_video_with_qwen(video_path: str) -> Dict[str, Any]:
    # Берем кадры с частотой раз в 5 секунд (fps_sample=0.2), чтобы ролик поместился в контекст
    frames = extract_frames_for_vlm(video_path, fps_sample=0.2)
    if not frames:
        return {"verdict": "no_violation", "violations": []}

    # Берем максимум 8 кадров на весь ролик
    batch_frames = frames[:8] 
    images_b64 = [f["annotated_b64"] for f in batch_frames]

    prompt = """
Ты — профессиональный эксперт по банковской кассовой дисциплине и комплаенсу ForteBank. 
Проанализируй кадры с камеры наблюдения на кассе (видео снято сверху).
Проверь их на наличие нарушений правил НБРК:
1. SAFE_AND_STORAGE_RULES: Открытый сейф, оставленный ключ, деньги не убраны в сейф, смешивание денег.
2. DOOR_AND_ACCESS_CONTROL: Дверь открыта, посторонние лица, два клиента одновременно.
3. CASH_COUNT_AND_AUTHENTICITY: Выдача неполной суммы, прием денег без пересчета.
4. ID_VERIFICATION_MISSING: Прием ценностей без удостоверения личности.
5. PERSONAL_ITEMS_AND_FOOD: Личные вещи или еда/кофе на рабочем столе.
6. INCASSATION_BAG_UNATTENDED: Инкассаторская сумка без присмотра.
7. WORKSTATION_AND_EQUIPMENT_SECURITY: Незаблокированный ноутбук при уходе.

Таймкоды указаны на изображениях (Time: Хs).
Верни ответ строго в формате JSON без лишнего текста:
{
  "verdict": "violation" или "no_violation",
  "violations": [
    {
      "rule_code": "код правила из списка выше или none",
      "timecode_sec": точное число секунд из штампа,
      "description": "описание нарушения"
    }
  ]
}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "qwen2.5vl:3b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images_b64
                    }
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 16384  # Увеличиваем контекстное окно Ollama до 16К токенов!
                }
            },
            timeout=180
        )
        
        resp_data = response.json()
        print("Ollama raw response:", resp_data)
        
        message_obj = resp_data.get("message", {})
        result_content = message_obj.get("content", "{}")
        
        parsed_res = json.loads(result_content)
        
        final_violations = []
        for v in parsed_res.get("violations", []):
            rule_code = v.get("rule_code")
            rule_info = RULES_DB.get(rule_code, {})
            
            target_ts = v.get("timecode_sec", 0.0)
            matched_frame = next((f for f in frames if abs(f["timestamp_sec"] - target_ts) < 2.0), frames[0])
            
            violation_type = rule_info.get("type", "Нарушение кассовой дисциплины")
            normative = rule_info.get("normative_source", "Постановление Правления НБРК № 231 (adilet.zan.kz)")
            
            final_violations.append({
                "type": violation_type,
                "timecode_sec": target_ts,
                "description": f"{v.get('description', '')}. Нормативный источник: {normative}",
                "evidence_frames": [
                    {"timestamp_sec": matched_frame["timestamp_sec"], "image_base64": matched_frame["clean_b64"]}
                ]
            })

        return {
            "verdict": parsed_res.get("verdict", "no_violation"),
            "violations": final_violations
        }
        
    except Exception as e:
        print(f"Error in VLM pipeline: {e}")
        return {"verdict": "no_violation", "violations": []}