import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def run_command(root: Path, args: List[str], timeout: int | None = None) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=str(root),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "args": [sys.executable, *args],
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


class BootstrapRunner:
    def __init__(self, root: Path, profile: str, seed: int, allow_download: bool, max_real_gb: float):
        self.root = root
        self.profile = profile
        self.seed = seed
        self.allow_download = allow_download
        self.max_real_gb = max_real_gb
        self.events: List[Dict[str, Any]] = []

    def exists(self, relative: str) -> bool:
        return (self.root / relative).exists()

    def execute(self, name: str, args: List[str], timeout: int | None = None, required: bool = True) -> None:
        event = {"name": name, "args": args, "required": required}
        result = run_command(self.root, args, timeout=timeout)
        event.update(result)
        self.events.append(event)
        if required and result["returncode"] != 0:
            raise RuntimeError(f"{name} failed with code {result['returncode']}\n{result['stderr'][-4000:]}")

    def write_catalog(self) -> None:
        self.execute(
            "write_real_source_catalog",
            ["-m", "data.real_data_catalog", "--write-catalog", "--dry-run", "--max-gb", str(self.max_real_gb)],
            timeout=120,
            required=False,
        )

    def acquire_real_data(self) -> None:
        if self.exists("data/real/batterylife_processed_v10/NA-ion.zip"):
            return
        if not self.allow_download:
            self.events.append(
                {
                    "name": "download_real_data",
                    "skipped": True,
                    "reason": "allow_download_false",
                    "needed_path": str(self.root / "data/real/batterylife_processed_v10/NA-ion.zip"),
                }
            )
            return
        self.execute(
            "download_real_naion",
            ["-m", "data.real_data_catalog", "--write-catalog", "--source", "batterylife_processed_v10", "--max-gb", str(self.max_real_gb)],
            timeout=1800,
            required=True,
        )

    def normalize_real_data(self) -> None:
        if not self.exists("data/real/batterylife_processed_v10/NA-ion.zip"):
            return
        self.execute(
            "normalize_real_naion",
            ["-m", "data.normalize_real_data", "--max-points-per-cycle", "20"],
            timeout=1200,
            required=True,
        )
        self.execute(
            "assemble_real_dataset",
            ["-m", "data.assemble_real_dataset"],
            timeout=300,
            required=True,
        )

    def scrape_literature(self, max_papers: int, max_documents: int) -> None:
        if self.exists("data/real/scraped/literature_measurements.parquet"):
            self.execute("curate_existing_literature", ["-m", "data.curate_literature_data"], timeout=240, required=False)
            return
        self.execute(
            "scrape_open_literature",
            [
                "-m",
                "data.literature_scraper",
                "--max-papers",
                str(max_papers),
                "--max-documents",
                str(max_documents),
                "--mailto",
                "kineticsforge.kaggle@example.com",
            ],
            timeout=1800,
            required=False,
        )
        self.execute("curate_literature", ["-m", "data.curate_literature_data"], timeout=240, required=False)

    def generate_synthetic(self) -> None:
        self.execute(
            "generate_hyper_data",
            ["-m", "data.hyper_data_pipeline", "--profile", self.profile, "--seed", str(self.seed)],
            timeout=1800,
            required=True,
        )
        self.execute(
            "validate_hyper_data",
            ["-m", "data.validate_hyper_data", "--profile", self.profile],
            timeout=300,
            required=True,
        )

    def run(self, scrape: bool, max_papers: int, max_documents: int) -> Dict[str, Any]:
        self.write_catalog()
        self.acquire_real_data()
        self.normalize_real_data()
        if scrape:
            self.scrape_literature(max_papers, max_documents)
        self.generate_synthetic()
        report = {
            "project_root": str(self.root),
            "profile": self.profile,
            "seed": self.seed,
            "allow_download": self.allow_download,
            "events": self.events,
        }
        out = self.root / "training" / "colab_kaggle" / "bootstrap_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(project_root_from_file()))
    parser.add_argument("--profile", choices=["smoke", "foundation", "hyper"], default="foundation")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-real-gb", type=float, default=0.45)
    parser.add_argument("--scrape", action="store_true")
    parser.add_argument("--max-papers", type=int, default=120)
    parser.add_argument("--max-documents", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = BootstrapRunner(
        root=Path(args.project_root).resolve(),
        profile=args.profile,
        seed=args.seed,
        allow_download=args.allow_download,
        max_real_gb=args.max_real_gb,
    )
    report = runner.run(scrape=args.scrape, max_papers=args.max_papers, max_documents=args.max_documents)
    print(json.dumps({"status": "complete", "events": len(report["events"]), "profile": args.profile}, indent=2))


if __name__ == "__main__":
    main()

