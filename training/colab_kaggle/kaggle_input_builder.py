import argparse
import fnmatch
import hashlib
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "runs",
}


DEFAULT_EXCLUDE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.log",
    "*.zip",
    "*.manifest.json",
    "*.pt",
    "*.pth",
    "*.pdf",
    "*.html",
    "raw_documents/*",
}


DEFAULT_INCLUDE_ROOTS = [
    "core",
    "modules",
    "training",
    "data",
    "business",
    "validation",
    "dashboard",
    "README.md",
    "requirements.txt",
    "setup.py",
    "KINETICSFORGE_ARCHITECTURE_MANUAL.md",
]


@dataclass
class ZipEntry:
    path: str
    bytes: int
    sha256: str


@dataclass
class KaggleInputManifest:
    schema: str = "kineticsforge.kaggle_input.v2"
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    project_name: str = "kineticsforge_v2"
    file_count: int = 0
    total_bytes: int = 0
    entries: List[ZipEntry] = field(default_factory=list)
    run_commands: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


class KaggleInputBuilder:
    def __init__(
        self,
        project_root: Path,
        include_roots: Optional[Sequence[str]] = None,
        exclude_dirs: Optional[Sequence[str]] = None,
        exclude_patterns: Optional[Sequence[str]] = None,
        max_file_mb: float = 64.0,
    ):
        self.project_root = project_root.resolve()
        self.include_roots = list(include_roots or DEFAULT_INCLUDE_ROOTS)
        self.exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)
        self.exclude_patterns = set(exclude_patterns or DEFAULT_EXCLUDE_PATTERNS)
        self.max_file_bytes = int(max_file_mb * 1024 * 1024)

    def iter_files(self) -> Iterable[Path]:
        for root_name in self.include_roots:
            root = self.project_root / root_name
            if not root.exists():
                continue
            if root.is_file():
                if self._include_file(root):
                    yield root
                continue
            for path in root.rglob("*"):
                if path.is_file() and self._include_file(path):
                    yield path

    def _include_file(self, path: Path) -> bool:
        rel = self._rel(path)
        parts = set(path.relative_to(self.project_root).parts[:-1])
        if parts & self.exclude_dirs:
            return False
        if path.stat().st_size > self.max_file_bytes:
            return False
        rel_posix = rel.replace("\\", "/")
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(path.name, pattern):
                return False
        if rel_posix.startswith("data/synthetic/cathode/") and not self._small_synthetic_keep(path, "cathode_hyper_", 12):
            return False
        if rel_posix.startswith("data/synthetic/bms/") and not self._small_synthetic_keep(path, "bms_hyper_", 8):
            return False
        return True

    def _small_synthetic_keep(self, path: Path, prefix: str, keep: int) -> bool:
        name = path.name
        if not name.endswith(".npz"):
            return True
        if prefix not in name:
            return True
        digits = "".join(ch for ch in name if ch.isdigit())
        if not digits:
            return True
        return int(digits[-4:]) < keep

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.project_root)).replace(os.sep, "/")

    def build(self, out_zip: Path, write_manifest: bool = True) -> KaggleInputManifest:
        out_zip = out_zip.resolve()
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        manifest = KaggleInputManifest(
            run_commands=[
                "python -m validation.v2_readiness_report --project-root .",
                "python -m training.colab_kaggle.industrial_training_pipeline --project-root . --task all --profile smoke",
                "python -m training.colab_kaggle.kaggle_colab_train --task all --profile quick --device auto",
            ],
            notes=[
                "This zip intentionally excludes model checkpoints and raw PDFs/HTML.",
                "Large foundation data should be regenerated or attached as a separate Kaggle dataset if needed.",
                "Use the one-cell notebook markdown in training/colab_kaggle/notebooks.",
            ],
        )
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(self.iter_files(), key=lambda p: self._rel(p)):
                rel = self._rel(path)
                digest = sha256_file(path)
                size = path.stat().st_size
                zf.write(path, rel)
                manifest.entries.append(ZipEntry(path=rel, bytes=int(size), sha256=digest))
                manifest.file_count += 1
                manifest.total_bytes += int(size)
            manifest_payload = json.dumps(asdict(manifest), indent=2)
            zf.writestr("training/colab_kaggle/input_manifest/kaggle_input_manifest_v2.json", manifest_payload)
        if write_manifest:
            manifest_path = out_zip.with_suffix(".manifest.json")
            manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
        return manifest


def write_one_cell(path: Path, zip_name: str = "kineticsforge_v2_input.zip") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = f"""# KineticsForge V2 Kaggle one-cell run
# Attach `{zip_name}` as a Kaggle Dataset input, then run this cell.
import os, sys, json, shutil, subprocess, zipfile, pathlib, textwrap

ROOT = pathlib.Path("/kaggle/working/kineticsforge_v2")
INPUTS = list(pathlib.Path("/kaggle/input").rglob("{zip_name}"))
if not INPUTS:
    raise FileNotFoundError("Attach the KineticsForge V2 input zip as a Kaggle Dataset.")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(INPUTS[0], "r") as zf:
    zf.extractall(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

def run(cmd):
    print("\\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

run([sys.executable, "-m", "pip", "install", "-q", "-r", "training/colab_kaggle/requirements_kaggle.txt"])
run([sys.executable, "-m", "validation.v2_readiness_report", "--project-root", ".", "--out", "data/cache/v2_readiness_report.json"])
run([sys.executable, "-m", "training.colab_kaggle.industrial_training_pipeline", "--project-root", ".", "--task", "all", "--profile", "smoke"])
run([sys.executable, "-m", "training.colab_kaggle.kaggle_colab_data_bootstrap", "--profile", "smoke"])
run([sys.executable, "-m", "training.colab_kaggle.kaggle_colab_train", "--task", "all", "--profile", "quick", "--device", "auto"])

print("\\nArtifacts:")
for p in [
    "data/cache/v2_readiness_report.json",
    "training/colab_kaggle/runs",
    "checkpoints/metrics.json",
]:
    print(" -", ROOT / p)
"""
    path.write_text(cell, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="training/colab_kaggle/input_manifest/kineticsforge_v2_input.zip")
    parser.add_argument("--max-file-mb", type=float, default=64.0)
    parser.add_argument("--write-cell", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.out
    builder = KaggleInputBuilder(root, max_file_mb=args.max_file_mb)
    manifest = builder.build(out)
    if args.write_cell:
        write_one_cell(root / "training" / "colab_kaggle" / "notebooks" / "KINETICSFORGE_V2_ONE_CELL.py", out.name)
    print(json.dumps({"zip": str(out), "files": manifest.file_count, "bytes": manifest.total_bytes}, indent=2))


if __name__ == "__main__":
    main()
