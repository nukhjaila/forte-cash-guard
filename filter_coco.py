"""Clean the auto-labels and pick a small review set (about 300 frames).

Run from the repo root after autolabel.py:
    python filter_coco.py --per-video 10

Reads : data/labeling/auto_partial.jsonl, eval/labels.csv
Writes: data/labeling/review/images/*.jpg   frames to fix by hand
        data/labeling/review/coco.json      cleaned boxes for those frames

Cleaning rules (there is one safe and one door in every scene):
  - safe_*  : keep only the single best box per frame (class kept as the model gave it)
  - door_*  : keep only the single best box per frame
  - key     : keep only the single best box per frame
  - journal, id_card, stamp: dropped (the model confuses them, draw them by hand if needed)
  - other classes: keep boxes above a per-class score threshold
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
LAB = ROOT / "data" / "labeling"
FPS = 3

CLASSES = ["person", "money", "safe_open", "safe_closed", "key", "id_card", "laptop",
           "bag", "cup", "personal_item", "journal", "stamp", "door_open", "door_closed"]
DROP = {"journal", "id_card", "stamp"}
TOP1_GROUPS = {"safe": {"safe_open", "safe_closed"}, "door": {"door_open", "door_closed"}, "key": {"key"}}
MIN_SCORE = {"person": 0.35, "money": 0.30, "laptop": 0.40, "bag": 0.40, "cup": 0.40,
             "personal_item": 0.30, "safe_open": 0.30, "safe_closed": 0.30,
             "door_open": 0.30, "door_closed": 0.30, "key": 0.25}


def norm(s):
    return s.replace("Х", "X").replace("х", "x").upper()


def clean(dets):
    out = [d for d in dets if d["cls"] not in DROP and d["score"] >= MIN_SCORE.get(d["cls"], 0.35)]
    keep = []
    for group in TOP1_GROUPS.values():
        members = [d for d in out if d["cls"] in group]
        if members:
            keep.append(max(members, key=lambda d: d["score"]))
    in_groups = set().union(*TOP1_GROUPS.values())
    keep += [d for d in out if d["cls"] not in in_groups]
    return keep


def frame_info(name):
    m = re.match(r"^(.*)_(\d+)\.jpg$", name)
    return norm(m.group(1)), (int(m.group(2)) - 1) / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=10)
    args = ap.parse_args()

    events = {}
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["verdict"] == "violation" and r["time_sec"]:
                events[r["video"]] = float(r["time_sec"])

    by_video = {}
    for line in open(LAB / "auto_partial.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        video, t = frame_info(rec["file"])
        by_video.setdefault(video, []).append((t, rec))

    chosen = []
    for video, items in sorted(by_video.items()):
        items.sort(key=lambda x: x[0])
        picked = {}
        if video in events:
            near = sorted(items, key=lambda x: abs(x[0] - events[video]))[:4]
            for t, rec in near:
                picked[rec["file"]] = rec
        n = args.per_video - len(picked)
        if n > 0 and items:
            step = max(len(items) / n, 1)
            for k in range(n):
                t, rec = items[min(int(k * step), len(items) - 1)]
                picked.setdefault(rec["file"], rec)
        chosen += list(picked.values())

    out_img = LAB / "review" / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    images, annotations, ann_id = [], [], 1
    counts = {c: 0 for c in CLASSES}
    for img_id, rec in enumerate(chosen, 1):
        src = LAB / "images" / rec["file"]
        if src.exists():
            shutil.copy2(src, out_img / rec["file"])
        images.append({"id": img_id, "file_name": rec["file"], "width": rec["w"], "height": rec["h"]})
        for d in clean(rec["dets"]):
            x, y, w, h = d["bbox"]
            annotations.append({"id": ann_id, "image_id": img_id,
                                "category_id": CLASSES.index(d["cls"]) + 1,
                                "bbox": d["bbox"], "area": round(w * h, 1), "iscrowd": 0})
            counts[d["cls"]] += 1
            ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    json.dump(coco, open(LAB / "review" / "coco.json", "w", encoding="utf-8"))
    print(f"review set: {len(images)} images, {len(annotations)} boxes")
    print("boxes per class:", counts)


if __name__ == "__main__":
    main()