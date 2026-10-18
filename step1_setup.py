"""Step 1 setup for forte-cash-guard.

Run from the repo root (the folder that contains backend/):
    python step1_setup.py "path/to/table.xlsx"

What it does:
 1. creates data/ models/ src/ eval/ folders
 2. reads the xlsx, normalizes Cyrillic Kh to Latin X, writes eval/labels.csv
 3. writes data/split.json (holdout fixed below, train = every other video)
 4. patches backend/app/main.py and backend/app/ai_engine/pipeline.py:
    - API key comes from the X_API_KEY environment variable (no hardcoded key)
    - the web page asks for the key instead of containing it
    - pipeline errors raise, so the job becomes "failed" (not "no_violation")
    - rules_kz.json is found by an absolute path
 5. writes a clean backend/requirements.txt (old one saved once as
    backend/requirements.conda-dump.txt)
It is safe to run twice.
"""
import csv
import json
import re
import sys
from pathlib import Path

import openpyxl

HOLDOUT = ["X-17", "X-20", "X-33", "X-34", "X-38", "X-46",
           "X-3", "X-12", "X-21", "X-35", "X-45", "X-48"]

ROOT = Path.cwd()
MAIN_PY = ROOT / "backend" / "app" / "main.py"
PIPELINE_PY = ROOT / "backend" / "app" / "ai_engine" / "pipeline.py"
REQ_TXT = ROOT / "backend" / "requirements.txt"

REQUIREMENTS = """fastapi
uvicorn
python-multipart
opencv-python
requests
pydantic
openpyxl
"""


def norm_name(s) -> str:
    s = str(s).strip()
    s = s.replace("Х", "X").replace("х", "x")  # Cyrillic Kh -> Latin X
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\.mov$", "", s, flags=re.I)
    return s.upper().replace(" ", "")


def parse_time(s):
    if s is None:
        return ""
    s = str(s).lower()
    m = re.search(r"(\d+)\s*мин", s)
    sec = re.search(r"(\d+)\s*сек", s)
    if not m and not sec:
        return ""
    return (int(m.group(1)) * 60 if m else 0) + (int(sec.group(1)) if sec else 0)


def make_folders():
    for d in ["data/videos", "data/frames", "data/labels", "models", "src", "eval"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("folders ready")


def read_labels(xlsx_path):
    ws = openpyxl.load_workbook(xlsx_path).active
    rows = []
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i == 0 or not r or not r[0]:
            continue
        viol = str(r[2]).strip().lower() == "нарушение"
        rows.append({
            "video": norm_name(r[0]),
            "verdict": "violation" if viol else "no_violation",
            "time_sec": parse_time(r[1]) if viol else "",
            "description": (r[3] or "").strip() if len(r) > 3 and r[3] else "",
        })
    return rows


def write_labels_and_split(rows):
    with open(ROOT / "eval" / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "verdict", "time_sec", "description"])
        w.writeheader()
        w.writerows(rows)
    names = [r["video"] for r in rows]
    missing = [h for h in HOLDOUT if h not in names]
    assert not missing, f"holdout not in xlsx: {missing}"
    train = [n for n in names if n not in HOLDOUT]
    with open(ROOT / "data" / "split.json", "w", encoding="utf-8") as f:
        json.dump({"train": train, "holdout": HOLDOUT}, f, indent=2)
    hv = sum(1 for r in rows if r["video"] in HOLDOUT and r["verdict"] == "violation")
    print(f"labels: {len(rows)} videos | train {len(train)} | holdout {len(HOLDOUT)} "
          f"({hv} violation, {len(HOLDOUT) - hv} clean)")
    return names


def check_videos(names):
    vids = sorted((ROOT / "data" / "videos").glob("*.mov"))
    if not vids:
        print("data/videos is empty: copy the .mov files there, then run this script again")
        return
    stems = {norm_name(v.stem): v.name for v in vids}
    no_file = [n for n in names if n not in stems]
    no_label = [s for s in stems if s not in names]
    print(f"videos found: {len(vids)}")
    if no_file:
        print("in xlsx but no .mov file:", ", ".join(no_file))
    if no_label:
        print(".mov file but not in xlsx:", ", ".join(no_label))


def patch(path: Path, pairs):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    changed = False
    for old, new in pairs:
        if new in text:
            continue
        assert old in text, f"{path.name}: pattern not found:\n{old}"
        text = text.replace(old, new, 1)
        changed = True
    if crlf:
        text = text.replace("\n", "\r\n")
    if changed:
        path.write_bytes(text.encode("utf-8"))
        print(f"patched {path.relative_to(ROOT)}")
    else:
        print(f"already patched {path.relative_to(ROOT)}")


MAIN_PAIRS = [
    (
        'API_KEY = os.getenv("X_API_KEY", "forte-secret-key-2026")\n',
        'API_KEY = os.getenv("X_API_KEY")\n'
        'if not API_KEY:\n'
        '    raise RuntimeError("Set the X_API_KEY environment variable before starting the server")\n',
    ),
    (
        'const API_KEY = "forte-secret-key-2026";',
        "const API_KEY = sessionStorage.getItem('apiKey') || prompt('API key:') || '';\n"
        "            sessionStorage.setItem('apiKey', API_KEY);",
    ),
    (
        "                const data = await res.json();\n"
        "                pollJob(data.job_id);",
        "                if (!res.ok) {\n"
        "                    statusBox.className = \"mt-4 text-sm font-medium text-red-600\";\n"
        "                    statusBox.innerText = \"Ошибка загрузки: неверный API-ключ или сервер недоступен.\";\n"
        "                    sessionStorage.removeItem('apiKey');\n"
        "                    return;\n"
        "                }\n"
        "                const data = await res.json();\n"
        "                pollJob(data.job_id);",
    ),
]

def patch_pipeline(path: Path):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    orig = text
    warn = []

    if not re.search(r"^import os\s*$", text, flags=re.M):
        text = "import os\n" + text

    rules_new = ('RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '
                 '"..", "knowledge_base", "rules_kz.json")')
    if re.search(r"^RULES_PATH\s*=", text, flags=re.M):
        text = re.sub(r"^RULES_PATH\s*=.*$", lambda m: rules_new, text, count=1, flags=re.M)
    else:
        warn.append("RULES_PATH line not found: make sure rules_kz.json is opened by an absolute path")

    text, n = re.subn(
        r'(print\(f"Error in VLM pipeline: \{e\}"\))\s*\n(\s*)return \{"verdict": "no_violation", "violations": \[\]\}',
        lambda m: m.group(1) + "\n" + m.group(2) + "raise", text, count=1)
    if n == 0 and "Error in VLM pipeline" in text and "\n        raise" not in text:
        warn.append("final except block not patched: make it 'raise' instead of returning no_violation")

    text, n = re.subn(
        r'if not frames:\s*\n(\s*)return \{"verdict": "no_violation", "violations": \[\]\}',
        lambda m: 'if not frames:\n' + m.group(1) + 'raise ValueError("no frames could be read from the video")',
        text, count=1)

    if crlf:
        text = text.replace("\n", "\r\n")
    if text != orig.replace("\n", "\r\n" if crlf else "\n"):
        path.write_bytes(text.encode("utf-8"))
        print(f"patched {path.relative_to(ROOT)}")
    else:
        print(f"already patched {path.relative_to(ROOT)}")
    for w in warn:
        print("WARNING:", w)


def write_requirements():
    if REQ_TXT.exists():
        old = REQ_TXT.read_text(encoding="utf-8", errors="ignore")
        if old.strip() == REQUIREMENTS.strip():
            print("requirements.txt already clean")
            return
        backup = REQ_TXT.with_name("requirements.conda-dump.txt")
        if not backup.exists():
            backup.write_text(old, encoding="utf-8")
    REQ_TXT.write_text(REQUIREMENTS, encoding="utf-8")
    print("wrote backend/requirements.txt")


def find_xlsx():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    found = sorted(ROOT.glob("*.xlsx"))
    if not found:
        sys.exit('usage: python step1_setup.py "path/to/table.xlsx"')
    return found[0]


def main():
    for p in (MAIN_PY, PIPELINE_PY):
        if not p.exists():
            sys.exit(f"{p} not found: run this script from the repo root (the folder with backend/)")
    make_folders()
    names = write_labels_and_split(read_labels(find_xlsx()))
    check_videos(names)
    patch(MAIN_PY, MAIN_PAIRS)
    patch_pipeline(PIPELINE_PY)
    write_requirements()
    print("done")


if __name__ == "__main__":
    main()