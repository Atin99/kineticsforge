"""Extract trained checkpoints from Kaggle results zip files.

Usage: python scripts/extract_checkpoints.py

Scans for results*.zip in common download locations and extracts
*_best.pt / *_final.pt files into checkpoints/trained/
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints" / "trained"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    Path.home() / "Downloads",
    ROOT.parent / "kaggle_deploy",
    ROOT.parent / "kaggle_deploy" / "acct1_zip",
    ROOT.parent / "kaggle_deploy" / "acct2_zip",
    ROOT.parent / "kaggle_deploy" / "acct3_zip",
]

CHECKPOINT_PATTERNS = ["_best.pt", "_final.pt", "_resume.pt", "tracker.json"]

def extract_all():
    found = 0
    for search_dir in SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.iterdir()):
            if f.suffix == ".zip" and "result" in f.name.lower():
                print(f"Scanning: {f}")
                try:
                    with zipfile.ZipFile(f, 'r') as zf:
                        for name in zf.namelist():
                            base = os.path.basename(name)
                            if any(base.endswith(p) for p in CHECKPOINT_PATTERNS):
                                target = CKPT_DIR / base
                                if target.exists():
                                    print(f"  SKIP (exists): {base}")
                                    continue
                                with zf.open(name) as src, open(target, 'wb') as dst:
                                    shutil.copyfileobj(src, dst)
                                size_kb = target.stat().st_size / 1024
                                print(f"  EXTRACTED: {base} ({size_kb:.0f} KB)")
                                found += 1
                except (zipfile.BadZipFile, Exception) as e:
                    print(f"  ERROR: {e}")

    # Also scan for loose .pt files in kaggle_deploy subdirs
    for search_dir in SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for f in search_dir.rglob("*.pt"):
            base = f.name
            if any(base.endswith(p) for p in CHECKPOINT_PATTERNS[:2]):
                target = CKPT_DIR / base
                if not target.exists():
                    shutil.copy2(f, target)
                    print(f"  COPIED: {base} from {f.parent}")
                    found += 1

    print(f"\nTotal checkpoints extracted: {found}")
    print(f"Checkpoint directory: {CKPT_DIR}")
    if CKPT_DIR.exists():
        for f in sorted(CKPT_DIR.iterdir()):
            print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")

if __name__ == "__main__":
    extract_all()
