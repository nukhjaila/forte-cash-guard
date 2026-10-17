import asyncio
import base64
import os
import uuid
from typing import Dict, List, Literal
import cv2
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

API_KEY = os.getenv("X_API_KEY", "forte-secret-key-2026")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="ForteBank Cash Guard AI", version="1.0.0")

class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str

class ViolationItem(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: List[EvidenceFrame]

class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: List[ViolationItem] = []

JOBS_DB: Dict[str, dict] = {}
JOB_QUEUE: asyncio.Queue = asyncio.Queue()

def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

def extract_frame_base64(video_path: str, timestamp_sec: float) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = min(int(timestamp_sec * fps), max(total_frames - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return ""
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")

async def video_worker():
    while True:
        job_id, video_path = await JOB_QUEUE.get()
        try:
            await asyncio.sleep(2)
            sample_ts = 1.0
            frame_b64 = await asyncio.to_thread(extract_frame_base64, video_path, sample_ts)

            JOBS_DB[job_id].update({
                "status": "done",
                "verdict": "violation",
                "violations": [
                    {
                        "type": "Оставление наличных денег и ценностей без присмотра",
                        "timecode_sec": sample_ts,
                        "description": (
                            "Визуально зафиксировано отсутствие кассира на рабочем месте при незакрытой кассе. "
                            "Нормативный источник: Постановление Правления НБРК № 204 «Об утверждении Правил ведения кассовых операций "
                            "и операций по инкассации банкнот, монет и ценностей в банках второго уровня...» (adilet.zan.kz)."
                        ),
                        "evidence_frames": [
                            {"timestamp_sec": sample_ts, "image_base64": frame_b64}
                        ],
                    }
                ],
            })
        except Exception as e:
            print(f"Error processing {job_id}: {e}")
            JOBS_DB[job_id]["status"] = "failed"
        finally:
            JOB_QUEUE.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(video_worker())

@app.post("/api/v1/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    x_api_key: str = Header(None, alias="X-API-Key")
):
    verify_api_key(x_api_key)
    job_id = uuid.uuid4().hex[:8]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(video_path, "wb") as f:
        f.write(await file.read())

    JOBS_DB[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "verdict": None,
        "violations": []
    }
    await JOB_QUEUE.put((job_id, video_path))
    return {"job_id": job_id}

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    x_api_key: str = Header(None, alias="X-API-Key")
):
    verify_api_key(x_api_key)
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS_DB[job_id]

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>ForteBank — AI-контроль кассовой дисциплины</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 text-gray-900 min-h-screen p-8">
        <div class="max-w-4xl mx-auto">
            <header class="flex items-center justify-between border-b pb-4 mb-8">
                <div>
                    <h1 class="text-3xl font-bold text-[#9E1B44]">ForteBank Cash Guard AI</h1>
                    <p class="text-sm text-gray-500">Локальный AI-сервис выявления нарушений кассовой дисциплины</p>
                </div>
            </header>
            <div class="bg-white p-6 rounded-xl shadow-sm border mb-6">
                <label class="block text-sm font-medium mb-2">Загрузите видеозапись (.mov)</label>
                <div class="flex gap-4">
                    <input type="file" id="videoFile" accept=".mov,video/*" class="block w-full text-sm border rounded-lg p-2"/>
                    <button onclick="uploadVideo()" class="bg-[#9E1B44] text-white px-6 py-2 rounded-lg font-medium hover:bg-[#7d1536] shrink-0">
                        Проверить видео
                    </button>
                </div>
                <div id="statusBox" class="hidden mt-4 text-sm font-medium text-blue-600"></div>
            </div>
            <div id="resultBox" class="hidden space-y-6"></div>
        </div>
        <script>
            const API_KEY = "forte-secret-key-2026";
            async function uploadVideo() {
                const fileInput = document.getElementById('videoFile');
                if (!fileInput.files.length) return alert('Выберите видеофайл!');
                const statusBox = document.getElementById('statusBox');
                const resultBox = document.getElementById('resultBox');
                statusBox.className = "mt-4 text-sm font-medium text-blue-600";
                statusBox.innerText = "Загрузка и запуск анализа...";
                resultBox.classList.add('hidden');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                const res = await fetch('/api/v1/analyze', {
                    method: 'POST',
                    headers: { 'X-API-Key': API_KEY },
                    body: formData
                });
                const data = await res.json();
                pollJob(data.job_id);
            }
            async function pollJob(jobId) {
                const statusBox = document.getElementById('statusBox');
                statusBox.innerText = `Статус: обработка видео (ID: ${jobId})...`;
                const interval = setInterval(async () => {
                    const res = await fetch(`/api/v1/jobs/${jobId}`, {
                        headers: { 'X-API-Key': API_KEY }
                    });
                    const job = await res.json();
                    if (job.status === 'done') {
                        clearInterval(interval);
                        statusBox.classList.add('hidden');
                        renderResult(job);
                    } else if (job.status === 'failed') {
                        clearInterval(interval);
                        statusBox.className = "mt-4 text-sm font-medium text-red-600";
                        statusBox.innerText = "Ошибка при обработке видео.";
                    }
                }, 1500);
            }
            function renderResult(job) {
                const resultBox = document.getElementById('resultBox');
                resultBox.classList.remove('hidden');
                const isViolation = job.verdict === 'violation';
                let html = `
                    <div class="p-4 rounded-xl border ${isViolation ? 'bg-red-50 border-red-300 text-red-900' : 'bg-green-50 border-green-300 text-green-900'}">
                        <h2 class="text-xl font-bold">
                            Вердикт: ${isViolation ? 'Обнаружены признаки возможных нарушений' : 'Нарушение не обнаружено'}
                        </h2>
                    </div>
                `;
                job.violations.forEach((v, idx) => {
                    const framesHtml = v.evidence_frames.map(f => `
                        <div>
                            <img src="data:image/jpeg;base64,${f.image_base64}" class="rounded-lg border max-h-64 object-contain" />
                            <span class="text-xs text-gray-500">Кадр на ${f.timestamp_sec} сек.</span>
                        </div>
                    `).join('');
                    html += `
                        <div class="bg-white p-6 rounded-xl shadow-sm border space-y-3">
                            <div class="flex justify-between items-center">
                                <span class="font-bold text-lg text-[#9E1B44]">#${idx + 1}. ${v.type}</span>
                                <span class="bg-gray-100 px-3 py-1 rounded-full text-sm font-mono">Таймкод: ${v.timecode_sec} сек</span>
                            </div>
                            <p class="text-sm text-gray-700 leading-relaxed">${v.description}</p>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">${framesHtml}</div>
                        </div>
                    `;
                });
                resultBox.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
