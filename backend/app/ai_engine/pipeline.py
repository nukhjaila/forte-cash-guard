import os
import re
import cv2
import base64
import json
import numpy as np
import requests
from typing import Dict, Any, List

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base", "rules_kz.json")

try:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        RULES_DB = json.load(f)
except Exception as e:
    print(f"Warning: Could not load rules from {RULES_PATH}: {e}")
    RULES_DB = {}


def make_grid_2x2(frames_subset: List[Dict[str, Any]]) -> str:
    """Склеивает 4 кадра в легкий коллаж 2x2 (672x392)."""
    cell_w, cell_h = 336, 196
    cells = []
    for item in frames_subset:
        cell = cv2.resize(item["raw_frame"], (cell_w, cell_h))
        cv2.rectangle(cell, (4, 4), (145, 28), (0, 0, 0), -1)
        cv2.putText(
            cell,
            f"T={item['timestamp_sec']}s",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )
        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (255, 255, 255), 1)
        cells.append(cell)

    while len(cells) < 4:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    top = np.hstack([cells[0], cells[1]])
    bottom = np.hstack([cells[2], cells[3]])
    grid = np.vstack([top, bottom])

    _, buf = cv2.imencode(".jpeg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf).decode("utf-8")


def extract_key_frames(video_path: str, total_samples: int = 8) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total_frames <= 0:
        cap.release()
        return []

    start_f = int(total_frames * 0.05)
    end_f = max(int(total_frames * 0.95), start_f + 1)
    indices = np.linspace(start_f, min(end_f, total_frames - 1), total_samples, dtype=int)

    frames_data = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp_sec = round(int(frame_idx) / fps, 1)
        _, clean_buffer = cv2.imencode(".jpeg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_clean = base64.b64encode(clean_buffer).decode("utf-8")

        frames_data.append({
            "timestamp_sec": timestamp_sec,
            "raw_frame": frame,
            "clean_b64": b64_clean
        })

    cap.release()
    return frames_data


PROMPT_TEXT = """Look at this 2x2 image of a bank cash desk (each panel shows T=Xs).
Normal cashier work (counting cash at desk, using laptop while seated, opening safe while putting cash inside, serving 1 client) is NOT a violation.

Check if there is an OBVIOUS violation from this list:
- SAFE_AND_STORAGE_RULES (cashier left room leaving money/stamps on desk; safe door left open after leaving; key left in safe lock; mixing two clients' money)
- DOOR_AND_ACCESS_CONTROL (room entrance door left open; two clients inside at once; stranger peeking in open door)
- CASH_COUNT_AND_AUTHENTICITY (client walked out while cashier counts money; putting money from bag to safe without counting)
- ID_VERIFICATION_MISSING (accepting cash or bag without checking ID card)
- PERSONAL_ITEMS_AND_FOOD (drinking coffee cup near cash; personal clutter on desk)
- INCASSATION_BAG_UNATTENDED (cashier walked out leaving incassation bag unattended)
- WORKSTATION_AND_EQUIPMENT_SECURITY (cashier walked out leaving laptop unlocked)

Reply in EXACTLY 4 short lines (no JSON, no markdown):
VERDICT: no_violation OR violation
RULE: NONE OR one code from list above
TIME: number of seconds from T=Xs
DESC: одно короткое предложение на русском языке, что видно на кадре"""


def parse_four_lines(text: str) -> Dict[str, Any]:
    verdict = "no_violation"
    rule_code = "NONE"
    time_sec = 0.0
    desc = ""

    for line in text.splitlines():
        line_clean = line.strip()
        if line_clean.upper().startswith("VERDICT:"):
            val = line_clean.split(":", 1)[1].strip().lower()
            if "violation" in val and "no_" not in val:
                verdict = "violation"
        elif line_clean.upper().startswith("RULE:"):
            val = line_clean.split(":", 1)[1].strip().upper()
            for known_code in RULES_DB.keys():
                if known_code in val:
                    rule_code = known_code
                    break
        elif line_clean.upper().startswith("TIME:"):
            val = line_clean.split(":", 1)[1].strip()
            nums = re.findall(r"\d+(?:\.\d+)?", val)
            if nums:
                time_sec = float(nums[0])
        elif line_clean.upper().startswith("DESC:"):
            desc = line_clean.split(":", 1)[1].strip()

    return {
        "verdict": verdict,
        "rule_code": rule_code,
        "timecode_sec": time_sec,
        "description_ru": desc
    }


def analyze_grid_with_ollama(grid_b64: str) -> Dict[str, Any]:
    # Используем /api/generate БЕЗ параметра format, чтобы избежать бага token repeat limit в Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": PROMPT_TEXT,
            "images": [grid_b64],
            "stream": False,
            "options": {
                "temperature": 0.4,
                "repeat_penalty": 1.3,
                "repeat_last_n": 64,
                "top_k": 20,
                "top_p": 0.85,
                "num_ctx": 4096,
                "num_predict": 120
            }
        },
        timeout=120
    )
    resp_data = response.json()
    if "error" in resp_data:
        print("Ollama error:", resp_data["error"])
        return {}

    raw_content = resp_data.get("response", "").strip()
    print("Ollama output:\n", raw_content)
    return parse_four_lines(raw_content)


def analyze_video_with_qwen(video_path: str) -> Dict[str, Any]:
    frames = extract_key_frames(video_path, total_samples=8)
    if not frames:
        raise ValueError("Не удалось извлечь кадры из видеофайла.")

    halves = [frames[:4], frames[4:]]
    best_violation = None

    for subset in halves:
        if not subset:
            continue
        grid_b64 = make_grid_2x2(subset)
        res = analyze_grid_with_ollama(grid_b64)

        if res.get("verdict") == "violation" and res.get("rule_code", "NONE") != "NONE":
            best_violation = res
            break

    if not best_violation:
        return {
            "verdict": "no_violation",
            "violations": []
        }

    rule_code = best_violation["rule_code"]
    rule_info = RULES_DB.get(rule_code, {})
    target_ts = float(best_violation.get("timecode_sec") or frames[0]["timestamp_sec"])

    matched_frame = min(frames, key=lambda f: abs(f["timestamp_sec"] - target_ts))
    violation_type = rule_info.get("type", f"Нарушение кассовой дисциплины ({rule_code})")
    normative = rule_info.get("normative_source", "Постановление Правления НБРК № 231 (adilet.zan.kz)")
    desc_ru = best_violation.get("description_ru") or "Зафиксировано визуальное нарушение кассовой дисциплины"

    return {
        "verdict": "violation",
        "violations": [
            {
                "type": violation_type,
                "timecode_sec": matched_frame["timestamp_sec"],
                "description": f"{desc_ru}. Нормативный источник: {normative}",
                "evidence_frames": [
                    {
                        "timestamp_sec": matched_frame["timestamp_sec"],
                        "image_base64": matched_frame["clean_b64"]
                    }
                ]
            }
        ]
    }
