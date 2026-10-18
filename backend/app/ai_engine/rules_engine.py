"""Detector + rules: finds cashier-discipline violations in a video, fully local.

Pipeline:
  1. sample the video at SAMPLE_FPS, run the fine-tuned RF-DETR detector on the frames
  2. derive simple facts per frame (is the cashier at the desk, is money visible, is a door open ...)
  3. rules turn facts that last long enough into events (type, timecode, description, evidence frames)

All numbers you may want to tune are in the CONFIG section. Put this file in backend/app/ai_engine/.
"""
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repo root (backend/app/ai_engine -> repo)
KB_PATH = HERE.parent / "knowledge_base" / "rules_kz.json"
CKPT_NAMES = ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]
CACHE_DIR = ROOT / "data" / "cache"

# ----------------------------------------------------------------------------- CONFIG
SAMPLE_FPS = 5.0           # frames per second given to the detector
PREDICT_THR = 0.25         # detector threshold (per-class thresholds below are applied on top)
DEFAULT_CONF = 0.40
CONF = {"person": 0.45, "key": 0.30, "id_card": 0.35, "personal_item": 0.35,
        "safe_open": 0.45, "door_open": 0.45}
MAX_GAP = 1.0              # seconds: short detector dropouts inside an event are ignored
CASHIER_MIN_OVERLAP = 0.25  # a person covering this share of the laptop (or counter) area = cashier at desk
MIN_PERSON_AREA = 0.004     # ignore person boxes smaller than this share of the frame
MAX_EVENTS = 5

# id: rule settings. min_dur = seconds the situation must last. Set enabled False to switch a rule off.
RULES = {
    "door_open_start": dict(enabled=True, min_dur=2.0, first_sec=25.0, code="DOOR_AND_ACCESS_CONTROL",
                            type="Дверь кассового помещения открыта при начале работы",
                            what="В начале работы дверь кассового помещения открыта", draw=["door_open", "person"]),
    "safe_open_long": dict(enabled=True, min_dur=6.0, code="SAFE_AND_STORAGE_RULES",
                           type="Дверца сейфа остаётся открытой",
                           what="Дверца сейфа открыта", draw=["safe_open", "person"]),
    "key_left": dict(enabled=True, min_dur=4.0, code="SAFE_AND_STORAGE_RULES",
                     type="Ключ оставлен в замке сейфа",
                     what="Ключ находится у замка сейфа", draw=["key", "safe_open", "safe_closed"]),
    "money_unattended": dict(enabled=True, min_dur=3.0, code="SAFE_AND_STORAGE_RULES",
                             type="Наличные деньги вне сейфа без присмотра кассира",
                             what="Наличные деньги лежат вне сейфа, а кассира у рабочего места нет",
                             draw=["money", "person"]),
    "bag_unattended": dict(enabled=True, min_dur=3.0, code="INCASSATION_BAG_UNATTENDED",
                           type="Инкассаторская сумка без присмотра",
                           what="Сумка лежит без присмотра, кассира у рабочего места нет", draw=["bag", "person"]),
    "too_many_people": dict(enabled=True, min_dur=3.0, code="DOOR_AND_ACCESS_CONTROL",
                            type="Одновременно трое и более человек в кассовом помещении",
                            what="В кадре одновременно три и более человека", draw=["person"]),
    "drink_on_desk": dict(enabled=True, min_dur=2.0, code="PERSONAL_ITEMS_AND_FOOD",
                          type="Напиток или посуда на рабочем месте",
                          what="На рабочем месте находится чашка или напиток", draw=["cup"]),
    "personal_items_on_desk": dict(enabled=True, min_dur=3.0, code="PERSONAL_ITEMS_AND_FOOD",
                                   type="Личные вещи на рабочем месте",
                                   what="На рабочем месте находятся личные вещи", draw=["personal_item"]),
}
# --------------------------------------------------------------------------------------

_MODEL = {"obj": None, "path": None}
_KB = None


def kb():
    global _KB
    if _KB is None:
        try:
            _KB = json.load(open(KB_PATH, encoding="utf-8"))
        except Exception:
            _KB = {}
    return _KB


def find_checkpoint():
    env = os.getenv("MODEL_CKPT")
    if env and Path(env).exists():
        return Path(env)
    for n in CKPT_NAMES:
        p = ROOT / "models" / "assist" / n
        if p.exists():
            return p
    raise FileNotFoundError("no detector checkpoint: expected models/assist/checkpoint_best_total.pth "
                            "(or set MODEL_CKPT)")


def get_model():
    if _MODEL["obj"] is None:
        from rfdetr import RFDETR
        p = find_checkpoint()
        _MODEL["obj"] = RFDETR.from_checkpoint(str(p))
        _MODEL["path"] = p
    return _MODEL["obj"], _MODEL["path"]


# ----------------------------------------------------------------------------- detection
def detect_video(video_path, use_cache=False):
    """Run the detector on sampled frames. Returns dict(fps, w, h, frames=[{i, t, d=[[cls, score, x1,y1,x2,y2]]}])."""
    video_path = Path(video_path)
    model, ckpt = get_model()
    cache = None
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"{video_path.stem}_{int(ckpt.stat().st_mtime)}.json"
        if cache.exists():
            return json.load(open(cache, encoding="utf-8"))
    names = list(model.class_names)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / SAMPLE_FPS)), 1)
    frames, batch, meta = [], [], []
    size = [0, 0]

    def flush():
        if not batch:
            return
        res = model.predict(batch, threshold=PREDICT_THR)
        if not isinstance(res, list):
            res = [res]
        for (idx, t), det in zip(meta, res):
            dets = []
            if det is not None and det.confidence is not None and len(det.confidence):
                for box, score, cid in zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist()):
                    if 0 <= int(cid) < len(names):
                        dets.append([names[int(cid)], round(float(score), 3)] + [round(float(v), 1) for v in box])
            frames.append({"i": idx, "t": round(t, 3), "d": dets})
        batch.clear()
        meta.clear()

    idx = 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                size = [frame.shape[1], frame.shape[0]]
                batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                meta.append((idx, idx / fps))
                if len(batch) == 8:
                    flush()
        idx += 1
    flush()
    cap.release()
    if not frames:
        raise ValueError("no frames could be read from the video")
    out = {"fps": fps, "w": size[0], "h": size[1], "frames": frames}
    if cache is not None:
        json.dump(out, open(cache, "w", encoding="utf-8"))
    return out


# ----------------------------------------------------------------------------- geometry helpers
def area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def expand(b, fx, fy):
    w, h = b[2] - b[0], b[3] - b[1]
    return [b[0] - fx * w, b[1] - fy * h, b[2] + fx * w, b[3] + fy * h]


def median_box(boxes):
    return [float(v) for v in np.median(np.array(boxes), axis=0)] if len(boxes) >= 3 else None


def runs(times, cond, min_dur, max_gap=MAX_GAP):
    """(start, end) intervals where cond is true, bridging gaps up to max_gap, lasting >= min_dur seconds."""
    out, start, last = [], None, None
    for t, c in zip(times, cond):
        if not c:
            continue
        if start is None:
            start = t
        elif t - last > max_gap:
            out.append((start, last))
            start = t
        last = t
    if start is not None:
        out.append((start, last))
    return [(a, b) for a, b in out if b - a >= min_dur - 1e-9]


# ----------------------------------------------------------------------------- facts + rules
def good(d):
    return d[1] >= CONF.get(d[0], DEFAULT_CONF)


def build_facts(frames, W, H):
    dets = [[d for d in f["d"] if good(d)] for f in frames]

    def boxes(i, cls):
        return [d[2:6] for d in dets[i] if d[0] == cls]

    all_boxes = lambda classes: [d[2:6] for ds in dets for d in ds if d[0] in classes]
    desk = median_box(all_boxes({"laptop"})) or median_box(all_boxes({"money_counter_machine"}))
    safe = median_box(all_boxes({"safe_open", "safe_closed"}))
    frame_area = float(W * H) if W and H else 1.0

    n = len(frames)
    F = {k: [False] * n for k in ["door_open", "safe_open", "key_near_safe", "money", "bag", "cup", "personal",
                                  "many_people", "cashier"]}
    for i in range(n):
        persons = [b for b in boxes(i, "person") if area(b) / frame_area >= MIN_PERSON_AREA]
        F["many_people"][i] = len(persons) >= 3
        if desk is not None:
            F["cashier"][i] = any(inter(b, desk) / max(area(desk), 1) >= CASHIER_MIN_OVERLAP for b in persons)
        safe_open_boxes = boxes(i, "safe_open")
        F["door_open"][i] = bool(boxes(i, "door_open"))
        F["safe_open"][i] = bool(safe_open_boxes)
        money = [b for b in boxes(i, "money") if not any(inside(center(b), s) for s in safe_open_boxes)]
        F["money"][i] = bool(money)
        F["bag"][i] = bool(boxes(i, "bag"))
        F["cup"][i] = bool(boxes(i, "cup"))
        F["personal"][i] = bool(boxes(i, "personal_item"))
        keys = boxes(i, "key")
        if keys:
            zone = expand(safe, 0.3, 0.3) if safe else None
            F["key_near_safe"][i] = any(zone is None or inside(center(k), zone) for k in keys)
    return F, desk is not None, dets


def rule_conditions(F, times, have_desk):
    absent = [not c for c in F["cashier"]]
    cond = {
        "door_open_start": [d and t <= RULES["door_open_start"]["first_sec"] for d, t in zip(F["door_open"], times)],
        "safe_open_long": F["safe_open"],
        "key_left": F["key_near_safe"],
        "money_unattended": [m and a for m, a in zip(F["money"], absent)] if have_desk else None,
        "bag_unattended": [b and a for b, a in zip(F["bag"], absent)] if have_desk else None,
        "too_many_people": F["many_people"],
        "drink_on_desk": F["cup"],
        "personal_items_on_desk": F["personal"],
    }
    return cond


def encode_frame(video_path, idx, boxes):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return ""
    for cls, score, x1, y1, x2, y2 in boxes:
        color = tuple(int(c) for c in np.random.RandomState(abs(hash(cls)) % 2 ** 31).randint(60, 255, 3))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(frame, f"{cls} {score:.2f}", (int(x1), max(int(y1) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf).decode("utf-8") if ok else ""


def describe(rule_id, a, b):
    r = RULES[rule_id]
    span = f"{a:.1f} с" if b - a < 0.5 else f"с {a:.1f} по {b:.1f} с"
    text = (f"{r['what']} ({span}). Это может указывать на возможное нарушение; "
            f"вывод требует проверки человеком.")
    src = kb().get(r["code"], {}).get("normative_source")
    if src:
        text += f" Нормативный источник: {src}"
    return text


def analyze_frames(det, video_path=None, with_frames=True):
    frames, W, H, fps = det["frames"], det["w"], det["h"], det["fps"]
    times = [f["t"] for f in frames]
    F, have_desk, dets = build_facts(frames, W, H)
    cond = rule_conditions(F, times, have_desk)
    events = []
    for rule_id, c in cond.items():
        cfg = RULES[rule_id]
        if not cfg["enabled"] or c is None:
            continue
        for a, b in runs(times, c, cfg["min_dur"]):
            events.append({"rule": rule_id, "start": a, "end": b})
    events.sort(key=lambda e: e["start"])
    if len(events) > MAX_EVENTS:
        events = sorted(sorted(events, key=lambda e: e["start"] - e["end"])[:MAX_EVENTS], key=lambda e: e["start"])

    violations = []
    for e in events:
        cfg = RULES[e["rule"]]
        item = {"rule": e["rule"], "type": cfg["type"], "timecode_sec": round(e["start"], 1),
                "end_sec": round(e["end"], 1), "description": describe(e["rule"], e["start"], e["end"]),
                "evidence_frames": []}
        if with_frames and video_path:
            picks = [e["start"]]
            if e["end"] - e["start"] >= 1.5:
                picks.append((e["start"] + e["end"]) / 2)
            for t in picks:
                k = int(np.argmin([abs(x - t) for x in times]))
                draw = [d for d in dets[k] if d[0] in cfg["draw"]]
                img = encode_frame(video_path, frames[k]["i"], draw)
                if img:
                    item["evidence_frames"].append({"timestamp_sec": round(times[k], 1), "image_base64": img})
        violations.append(item)
    return {"verdict": "violation" if violations else "no_violation", "violations": violations}


def analyze_video(video_path, use_cache=False, with_frames=True):
    det = detect_video(video_path, use_cache=use_cache)
    return analyze_frames(det, video_path=video_path, with_frames=with_frames)