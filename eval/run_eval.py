"""Score the detector+rules on labeled videos.   Run from repo root:
    python eval/run_eval.py --split holdout          (default; videos never trained on)
    python eval/run_eval.py --split train|all --tol 4
First run is slow (detector); later runs reuse data/cache, so tuning thresholds in rules_engine.py is fast.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from backend.app.ai_engine.rules_engine import analyze_video  # noqa: E402


def norm(s):
    return s.replace("Х", "X").replace("х", "x").upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout", choices=["holdout", "train", "all"])
    ap.add_argument("--tol", type=float, default=4.0, help="seconds: event counts as hit if this close to label")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    split = json.load(open(ROOT / "data" / "split.json", encoding="utf-8"))
    names = split["train"] + split["holdout"] if a.split == "all" else split[a.split]
    names = {norm(n) for n in names}
    labels = defaultdict(list)
    verdict = {}
    for r in csv.DictReader(open(ROOT / "eval" / "labels.csv", encoding="utf-8-sig")):
        v = norm(r["video"])
        verdict[v] = r["verdict"]
        if r["verdict"] == "violation" and r["time_sec"]:
            labels[v].append(float(r["time_sec"]))
    vids = {norm(p.stem): p for p in (ROOT / "data" / "videos").iterdir() if p.suffix.lower() in (".mov", ".mp4")}

    tp = fn = fp = tn = hit_t = 0
    rules = Counter()
    for v in sorted(names):
        if v not in vids or v not in verdict:
            print(f"skip {v}: video or label missing")
            continue
        res = analyze_video(vids[v], use_cache=not a.no_cache, with_frames=False)
        ev = res["violations"]
        for e in ev:
            rules[e["rule"]] += 1
        truth = verdict[v] == "violation"
        pred = bool(ev)
        tag = "TP" if truth and pred else "FN" if truth else "FP" if pred else "TN"
        tp += tag == "TP"; fn += tag == "FN"; fp += tag == "FP"; tn += tag == "TN"
        near = ""
        if truth and labels[v]:
            err = min(abs(e["timecode_sec"] - t) for e in ev for t in labels[v]) if ev else None
            if err is not None and err <= a.tol:
                hit_t += 1
            near = f" label={labels[v]} best_err={None if err is None else round(err, 1)}"
        print(f"{tag} {v}: " + ", ".join(f"{e['rule']}@{e['timecode_sec']}" for e in ev) + near)
    n = tp + fn + fp + tn
    print(f"\nvideos {n} | TP {tp} FN {fn} FP {fp} TN {tn}")
    if n:
        print(f"accuracy {(tp + tn) / n:.2f} | recall {tp / max(tp + fn, 1):.2f} | precision {tp / max(tp + fp, 1):.2f}"
              f" | time hits (±{a.tol}s) {hit_t}/{tp + fn}")
    print("rule firings:", dict(rules))


if __name__ == "__main__":
    main()