"""Train a quick detector on the frames you already corrected, then pre-label the rest.

Run from the repo root.

1) In CVAT: Actions -> Export task dataset -> COCO 1.0 (untick "Save images").
   Put the zip (or the instances_default.json inside it) anywhere.

2) Train on the corrected frames. ":77" means "only the first 77 frames of that file are corrected":
       python train_assist.py train --coco ~/Downloads/task_1.zip:77 --size small --epochs 30

3) Pre-label the remaining frames (not corrected yet):
       python train_assist.py predict --first-n 77 --thr 0.35
   Output: data/labeling/part2/images/ and data/labeling/part2/coco.json
   In CVAT create a NEW task from part2/images (same project) and upload part2/coco.json (COCO 1.0).
   (Do not upload into the old task: CVAT replaces the existing annotations of a task.)

4) After you fix part 2, export both tasks and train the final model on everything:
       python train_assist.py train --coco task_1.zip:77 --coco task_2.zip --size small --epochs 60
   (task_1.zip:77 = the 77 frames you corrected in task 1. If you finish all frames of task 1, leave out ":77".)
"""
import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path.cwd()
LAB = ROOT / "data" / "labeling"
IMAGE_DIRS = [LAB / "review" / "images", LAB / "part2" / "images", LAB / "images"]
DS = ROOT / "data" / "rfdetr_ds"
OUT = ROOT / "models" / "assist"
CKPT_NAMES = ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]


def parse_coco_arg(arg):
    m = re.match(r"^(.*):(\d+)$", arg)
    if m:
        return Path(m.group(1)).expanduser(), int(m.group(2))
    return Path(arg).expanduser(), None


def read_json(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".json")]
            pick = [n for n in names if n.endswith("instances_default.json")] or names
            if not pick:
                sys.exit(f"no json inside {path}")
            return json.loads(z.read(pick[0]).decode("utf-8"))
    return json.load(open(path, encoding="utf-8"))


def read_cvat_xml(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            pick = [n for n in names if n.endswith("annotations.xml")] or names
            if not pick:
                return None
            return z.read(pick[0])
    if path.suffix.lower() == ".xml":
        return path.read_bytes()
    return None


def has_json(path: Path):
    if path.suffix.lower() != ".zip":
        return path.suffix.lower() == ".json"
    with zipfile.ZipFile(path) as z:
        return any(n.endswith(".json") for n in z.namelist())


def load_frames_xml(raw: bytes, first_n):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(raw)
    label_names = []
    for lab in root.iter("label"):
        nm = lab.find("name")
        if nm is not None and nm.text and nm.text not in label_names:
            label_names.append(nm.text)
    imgs = sorted(root.iter("image"), key=lambda i: Path(i.get("name")).name)
    if first_n:
        imgs = imgs[:first_n]
    frames = []
    for im in imgs:
        boxes = []
        for b in im.findall("box"):
            x1, y1, x2, y2 = (float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            if x2 - x1 > 1 and y2 - y1 > 1:
                name = b.get("label")
                if name not in label_names:
                    label_names.append(name)
                boxes.append({"name": name, "bbox": [x1, y1, x2 - x1, y2 - y1]})
        frames.append({"file": Path(im.get("name")).name, "w": int(im.get("width")), "h": int(im.get("height")),
                       "boxes": boxes})
    return frames, label_names


def load_frames(path: Path, first_n):
    if not has_json(path):
        raw = read_cvat_xml(path)
        if raw is not None:
            return load_frames_xml(raw, first_n)
    data = read_json(path)
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    anns = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    imgs = sorted(data["images"], key=lambda i: Path(i["file_name"]).name)
    if first_n:
        imgs = imgs[:first_n]
    frames = []
    for im in imgs:
        boxes = []
        for a in anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w > 1 and h > 1:
                boxes.append({"name": cat_name[a["category_id"]], "bbox": [x, y, w, h]})
        frames.append({"file": Path(im["file_name"]).name, "w": im["width"], "h": im["height"], "boxes": boxes})
    return frames, [c["name"] for c in data["categories"]]


def find_image(name):
    for d in IMAGE_DIRS:
        if (d / name).exists():
            return d / name
    return None


def video_of(name):
    return name.rsplit("_", 1)[0]


def build_dataset(frames, class_names):
    if DS.exists():
        shutil.rmtree(DS)
    videos = sorted({video_of(f["file"]) for f in frames})
    valid_videos = set(videos[::6]) if len(videos) >= 3 else set(videos[-1:])
    split = {"train": [], "valid": []}
    for f in frames:
        split["valid" if video_of(f["file"]) in valid_videos else "train"].append(f)
    if not split["train"] or not split["valid"]:
        half = max(len(frames) // 6, 1)
        split = {"train": frames[half:], "valid": frames[:half]}
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    cid = {n: i + 1 for i, n in enumerate(class_names)}
    missing = 0
    for part, items in split.items():
        d = DS / part
        d.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for f in items:
            src = find_image(f["file"])
            if src is None:
                missing += 1
                continue
            shutil.copy2(src, d / f["file"])
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": f["file"], "width": f["w"], "height": f["h"]})
            for b in f["boxes"]:
                annotations.append({"id": len(annotations) + 1, "image_id": img_id, "category_id": cid[b["name"]],
                                    "bbox": b["bbox"], "area": round(b["bbox"][2] * b["bbox"][3], 1), "iscrowd": 0})
        json.dump({"images": images, "annotations": annotations, "categories": cats},
                  open(d / "_annotations.coco.json", "w", encoding="utf-8"))
        print(f"{part}: {len(images)} images, {len(annotations)} boxes")
    if missing:
        print(f"WARNING: {missing} frames skipped: image file not found in {[str(p) for p in IMAGE_DIRS]}")
    return split


def cmd_train(args):
    frames, class_names = [], []
    for arg in args.coco:
        path, n = parse_coco_arg(arg)
        fr, names = load_frames(path, n)
        frames += fr
        for nm in names:
            if nm not in class_names:
                class_names.append(nm)
        print(f"{path.name}: {len(fr)} frames")
    seen, uniq = set(), []
    for f in frames:
        if f["file"] not in seen:
            seen.add(f["file"])
            uniq.append(f)
    frames = uniq
    counts = {n: 0 for n in class_names}
    for f in frames:
        for b in f["boxes"]:
            counts[b["name"]] += 1
    print("boxes per class:", counts)
    build_dataset(frames, class_names)
    if args.prepare_only:
        return
    from rfdetr import RFDETRNano, RFDETRSmall
    cls = {"nano": RFDETRNano, "small": RFDETRSmall}[args.size]
    model = cls()
    OUT.mkdir(parents=True, exist_ok=True)
    model.train(dataset_dir=str(DS), epochs=args.epochs, batch_size=args.batch, grad_accum_steps=max(16 // args.batch, 1),
                output_dir=str(OUT))
    print("done. checkpoint in", OUT)


def cmd_predict(args):
    from PIL import Image
    from rfdetr import RFDETR

    ckpt = next((OUT / n for n in CKPT_NAMES if (OUT / n).exists()), None)
    if ckpt is None:
        sys.exit(f"no checkpoint in {OUT}: run the train command first")
    review = read_json(LAB / "review" / "coco.json") if (LAB / "review" / "coco.json").exists() else None
    if review is None:
        sys.exit("data/labeling/review/coco.json not found")
    names = sorted(Path(i["file_name"]).name for i in review["images"])
    rest = names[args.first_n:]
    print(f"{len(rest)} frames to pre-label (after the first {args.first_n})")

    model = RFDETR.from_checkpoint(str(ckpt))
    class_names = list(model.class_names)
    out_img = LAB / "part2" / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for k, name in enumerate(rest, 1):
        src = find_image(name)
        if src is None:
            continue
        shutil.copy2(src, out_img / name)
        im = Image.open(src).convert("RGB")
        det = model.predict(im, threshold=args.thr)
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": name, "width": im.size[0], "height": im.size[1]})
        found = []
        for box, score, cid in zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist()):
            if 0 <= int(cid) < len(class_names):
                found.append((class_names[int(cid)], float(score), box))
        keep = []
        for prefix in ("safe_", "door_"):  # one safe and one door per scene
            group = [f for f in found if f[0].startswith(prefix)]
            if group:
                keep.append(max(group, key=lambda f: f[1]))
        keep += [f for f in found if not f[0].startswith(("safe_", "door_"))]
        for cname, score, (x1, y1, x2, y2) in keep:
            annotations.append({"id": len(annotations) + 1, "image_id": img_id,
                                "category_id": class_names.index(cname) + 1,
                                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                                "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0})
        if k % 25 == 0:
            print(f"{k}/{len(rest)}")
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]}
    json.dump(coco, open(LAB / "part2" / "coco.json", "w", encoding="utf-8"))
    print(f"wrote data/labeling/part2/coco.json: {len(images)} images, {len(annotations)} boxes")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--coco", action="append", required=True, help="CVAT COCO export (zip or json), optional :N")
    t.add_argument("--size", choices=["nano", "small"], default="small")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--prepare-only", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--first-n", type=int, required=True, help="frames already corrected (taken from the start)")
    p.add_argument("--thr", type=float, default=0.35)
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    main()