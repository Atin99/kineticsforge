"""Create a deployment checkpoint bundle from checkpoints/trained/.

The bundle is for Render/docker/on-prem pilots where .pt files are not kept in
git. Upload the generated zip to private object storage and set
KF_CHECKPOINT_BUNDLE_URL on the deployed service.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints" / "trained"
OUT_DIR = ROOT / "artifacts"
OUT = OUT_DIR / "kineticsforge-checkpoints.zip"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle() -> Path:
    if not CKPT_DIR.exists():
        raise SystemExit("checkpoints/trained does not exist. Run scripts/extract_checkpoints.py first.")
    files = sorted(
        p for p in CKPT_DIR.iterdir()
        if p.is_file() and (p.name.endswith(("_best.pt", "_final.pt", "_resume.pt")) or p.name in {"checkpoint_manifest.json", "tracker.json"})
    )
    if not any(p.suffix == ".pt" for p in files):
        raise SystemExit("No checkpoint weights found. Run scripts/extract_checkpoints.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, f"checkpoints/trained/{path.name}")
    sidecar = OUT.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "bundle": OUT.name,
        "bytes": OUT.stat().st_size,
        "sha256": _sha256(OUT),
        "files": len(files),
        "weights": sum(1 for p in files if p.suffix == ".pt"),
    }, indent=2), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    path = bundle()
    print(f"Created {path}")
    print(f"SHA256 {_sha256(path)}")
