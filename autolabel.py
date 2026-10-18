"""Auto-label train frames with Grounding DINO (zero-shot) and write COCO json.

Run from the repo root, after step1_setup.py and after the frames exist in data/frames:
    pip install transformers torch pillow
    python autolabel.py --step 3

Output:
    data/labeling/images/*.jpg           frames to fix by hand (train videos only)
    data/labeling/auto_coco.json         COCO 1.0 boxes (import into the labeling tool)
Only TRAIN videos from data/split.json are used. Holdout videos are never touched.
Frames kept: every --step-th frame, plus all frames within +-5 s of the xlsx violation second.
It can be stopped and started again: finished frames are skipped.
"""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
FRAMES = ROOT / "data" / "frames"
OUT_DIR = ROOT / "data" / "labeling"
IMG_DIR = OUT_DIR / "images"
OUT_JSON = OUT_DIR / "auto_coco.json"
PARTIAL = OUT_DIR / "auto_partial.jsonl"
FPS = 3  # must match the ffmpeg fps used in step 3

CLASSES = ["person", "money", "safe_open", "safe_closed", "key", "id_card", "laptop",
           "bag", "cup", "personal_item", "journal", "stamp", "door_open", "door_closed"]

PHRASES = {
    "person": ["person"],
    "money": ["banknotes", "stack of cash", "money", "cash"],
    "safe_open": [],
    "safe_closed": ["safe", "metal safe", "safe box"],
    "key": ["key", "keys"],
    "id_card": ["id card"],
    "laptop": ["laptop"],
    "bag": ["bag"],
    "cup": ["cup"],
    "personal_item": ["smartphone", "wallet"],
    "journal": ["notebook"],
    "stamp": ["stamp"],
    "door_open": ["open door"],
    "door_closed": ["door"],
}
PHRASE_TO_CLASS = {p: c for c, ps in PHRASES.items() for p in ps}
PHRASES_BY_LEN = sorted(PHRASE_TO_CLASS, key=len, reverse=True)
PROMPT = ". ".join(PHRASE_TO_CLASS) + "."


def norm(s: str) -> str:
    return s.replace("Х", "X").replace("х", "x").upper()


def load_split_and_labels():
    split = json.load(open(ROOT / "data" / "split.json", encoding="utf-8"))
    train = set(split["train"])
    events = {}
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["verdict"] == "violation" and r["time_sec"]:
                events[r["video"]] = float(r["time_sec"])
    return train, events


def pick_frames(train, events, step):
    chosen = []
    for p in sorted(FRAMES.glob("*.jpg")):
        m = re.match(r"^(.*)_(\d+)$", p.stem)
        if not m:
            continue
        video, idx = norm(m.group(1)), int(m.group(2))
        if video not in train:
            continue
        t = (idx - 1) / FPS
        near_event = video in events and abs(t - events[video]) <= 5
        if idx % step == 0 or near_event:
            chosen.append(p)
    return chosen


def class_of(label: str):
    label = label.lower().strip()
    for ph in PHRASES_BY_LEN:
        if ph in label:
            return PHRASE_TO_CLASS[ph]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=3, help="keep every N-th frame (3 = 1 per second)")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    args = ap.parse_args()

    if not FRAMES.exists() or not any(FRAMES.glob("*.jpg")):
        sys.exit("no frames in data/frames: run the ffmpeg step first")
    train, events = load_split_and_labels()
    frames = pick_frames(train, events, args.step)
    if not frames:
        sys.exit("no train frames found: frame names must look like X-1_00001.jpg")
    print(f"{len(frames)} frames to label")

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if PARTIAL.exists():
        for line in open(PARTIAL, encoding="utf-8"):
            rec = json.loads(line)
            done[rec["file"]] = rec["dets"]

    with open(PARTIAL, "a", encoding="utf-8") as out:
        for i, p in enumerate(frames, 1):
            if p.name in done:
                continue
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, text=PROMPT, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            kwargs = dict(text_threshold=args.text_thr, target_sizes=[img.size[::-1]])
            try:
                res = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=args.box_thr, **kwargs)[0]
            except TypeError:
                res = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=args.box_thr, **kwargs)[0]
            labels = res.get("text_labels", res.get("labels"))
            dets = []
            for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
                cls = class_of(str(lab))
                if cls is None:
                    continue
                x1, y1, x2, y2 = box
                dets.append({"cls": cls, "score": round(score, 3),
                             "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]})
            out.write(json.dumps({"file": p.name, "w": img.size[0], "h": img.size[1], "dets": dets}) + "\n")
            out.flush()
            done[p.name] = dets
            shutil.copy2(p, IMG_DIR / p.name)
            if i % 50 == 0:
                print(f"{i}/{len(frames)}")

    # build COCO from the partial file (so a restart still gives a full json)
    sizes = {}
    for line in open(PARTIAL, encoding="utf-8"):
        rec = json.loads(line)
        sizes[rec["file"]] = (rec["w"], rec["h"])
    images, annotations = [], []
    ann_id = 1
    for img_id, p in enumerate(frames, 1):
        if p.name not in done:
            continue
        w, h = sizes[p.name]
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        for d in done[p.name]:
            x, y, bw, bh = d["bbox"]
            annotations.append({"id": ann_id, "image_id": img_id,
                                "category_id": CLASSES.index(d["cls"]) + 1,
                                "bbox": d["bbox"], "area": round(bw * bh, 1), "iscrowd": 0})
            ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    json.dump(coco, open(OUT_JSON, "w", encoding="utf-8"))
    per_class = {c: 0 for c in CLASSES}
    for a in annotations:
        per_class[CLASSES[a["category_id"] - 1]] += 1
    print(f"wrote {OUT_JSON.relative_to(ROOT)}: {len(images)} images, {len(annotations)} boxes")
    print("boxes per class:", per_class)


if __name__ == "__main__":
    main()