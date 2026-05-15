import argparse
import json
from pathlib import Path

from data.dataset_contracts import write_validation_report
from data.hyper_data_pipeline import PROFILES, HyperDatasetValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated KineticsForge hyper datasets.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="foundation")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    profile = PROFILES[args.profile]
    validator = HyperDatasetValidator(root, profile)
    result = validator.run()
    report_path = root / "cache" / f"data_quality_report_{profile.name}.json"
    write_validation_report(report_path, result["issues"], result["metrics"])
    print(json.dumps({"status": "pass" if not any(i.severity == "error" for i in result["issues"]) else "fail", "report": str(report_path), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

