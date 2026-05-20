"""Extract trained checkpoints from Kaggle results zip files.

Usage: python scripts/extract_checkpoints.py

Scans the local Downloads folder and the in-repo Kaggle deployment folders,
extracts checkpoint artifacts into checkpoints/trained/, and writes a manifest
so the API/UI can show exactly which result zip produced each model file.
"""
import hashlib
import json
import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints" / "trained"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = CKPT_DIR / "checkpoint_manifest.json"

SEARCH_DIRS = [
    Path.home() / "Downloads",
    ROOT / "kaggle_deploy",
    ROOT / "kaggle_deploy_2",
    ROOT / "kaggle_deploy_3",
    ROOT / "kaggle_deploy" / "acct1_zip",
    ROOT / "kaggle_deploy" / "acct2_zip",
    ROOT / "kaggle_deploy" / "acct3_zip",
    ROOT / "kaggle_deploy_2" / "acct1_zip",
    ROOT / "kaggle_deploy_2" / "acct2_zip",
    ROOT / "kaggle_deploy_3" / "acct_zip",
]

CHECKPOINT_PATTERNS = ["_best.pt", "_final.pt", "_resume.pt", "tracker.json"]

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_results_zip(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".zip" and (
        "result" in name or "checkpoint" in name or "allmodel" in name or name.startswith("kf-")
    )


def _candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    seen = set()
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        for path in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
            key = str(path.resolve()).lower()
            if key not in seen:
                dirs.append(path)
                seen.add(key)
    return dirs


def _result_number(path: Path) -> int:
    import re

    match = re.search(r"results\s*\((\d+)\)", path.name.lower())
    if match:
        return int(match.group(1))
    if path.name.lower() == "results.zip":
        return 0
    return -1


def _prefer(existing: Optional[Dict], candidate: Dict) -> Dict:
    if existing is None:
        return candidate
    existing_score = (existing["source_mtime"], _result_number(Path(existing["source"])), existing["size"])
    candidate_score = (candidate["source_mtime"], _result_number(Path(candidate["source"])), candidate["size"])
    return candidate if candidate_score >= existing_score else existing


def _zip_candidates(paths: Iterable[Path]) -> Dict[str, Dict]:
    chosen: Dict[str, Dict] = {}
    for archive in sorted((p for p in paths if _is_results_zip(p)), key=lambda p: (p.stat().st_mtime, _result_number(p))):
        print(f"Scanning: {archive}")
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                for member in zf.namelist():
                    base = os.path.basename(member)
                    if not base or not any(base.endswith(p) for p in CHECKPOINT_PATTERNS):
                        continue
                    info = zf.getinfo(member)
                    chosen[base] = _prefer(chosen.get(base), {
                        "kind": "zip",
                        "base": base,
                        "member": member,
                        "source": str(archive),
                        "source_name": archive.name,
                        "source_mtime": archive.stat().st_mtime,
                        "size": info.file_size,
                    })
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"  ERROR: {exc}")
    return chosen


def _loose_candidates(paths: Iterable[Path], chosen: Dict[str, Dict]) -> Dict[str, Dict]:
    for directory in paths:
        for path in directory.glob("*.pt"):
            base = path.name
            if not any(base.endswith(p) for p in CHECKPOINT_PATTERNS[:3]):
                continue
            chosen[base] = _prefer(chosen.get(base), {
                "kind": "file",
                "base": base,
                "member": base,
                "source": str(path),
                "source_name": path.parent.name,
                "source_mtime": path.stat().st_mtime,
                "size": path.stat().st_size,
            })
    return chosen


def _write_candidate(candidate: Dict) -> Dict:
    target = CKPT_DIR / candidate["base"]
    if candidate["kind"] == "zip":
        with zipfile.ZipFile(candidate["source"], "r") as zf:
            with zf.open(candidate["member"]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(candidate["source"], target)
    stat = target.stat()
    return {
        "file": candidate["base"],
        "path": str(target.relative_to(ROOT)),
        "bytes": stat.st_size,
        "sha256": _sha256(target),
        "source_name": candidate["source_name"],
        "source_member": candidate["member"],
        "source_mtime": candidate["source_mtime"],
    }


def extract_all():
    dirs = _candidate_dirs()
    zip_paths = []
    for directory in dirs:
        zip_paths.extend(directory.glob("*.zip"))

    chosen = _zip_candidates(zip_paths)
    chosen = _loose_candidates(dirs, chosen)

    datetime = __import__("datetime")
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "checkpoint_dir": str(CKPT_DIR.relative_to(ROOT)),
        "files": [],
        "source_zips": [],
    }

    source_zip_map: Dict[str, Dict] = {}
    written = 0
    for base in sorted(chosen):
        item = _write_candidate(chosen[base])
        manifest["files"].append(item)
        if str(chosen[base]["source"]).lower().endswith(".zip"):
            source_zip_map[str(chosen[base]["source"])] = {
                "name": item["source_name"],
                "mtime": item["source_mtime"],
            }
        written += 1
        print(f"  READY: {base} ({item['bytes'] / 1024:.0f} KB) from {item['source_name']}")

    manifest["source_zips"] = sorted(source_zip_map.values(), key=lambda x: (x["mtime"], x["name"]))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nTotal checkpoint artifacts ready: {written}")
    print(f"Checkpoint directory: {CKPT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    if CKPT_DIR.exists():
        for f in sorted(CKPT_DIR.glob("*.pt")):
            print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")

if __name__ == "__main__":
    extract_all()
