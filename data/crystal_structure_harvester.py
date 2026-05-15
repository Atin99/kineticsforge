import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import requests

from data.dataset_contracts import ensure_dir, sha256_file, utc_now, write_json


NA_ION_FORMULAS = (
    "Na-Mn-O",
    "Na-Fe-O",
    "Na-Mn-Fe-O",
    "Na-V-P-O",
    "Na-Ti-O",
    "Na-Ni-Mn-O",
)


class MaterialsProjectHarvester:
    def __init__(self, api_key: str, out_dir: Path):
        self.api_key = api_key
        self.out_dir = ensure_dir(out_dir)
        self.session = requests.Session()
        self.session.headers.update({"X-API-KEY": api_key, "User-Agent": "KineticsForge crystal harvester"})

    def search(self, chemsys: str, limit: int = 100) -> List[Dict[str, object]]:
        url = "https://api.materialsproject.org/materials/summary/"
        params = {
            "chemsys": chemsys,
            "is_stable": "False",
            "_fields": "material_id,formula_pretty,formation_energy_per_atom,energy_above_hull,band_gap,symmetry,structure",
            "_limit": min(limit, 500),
        }
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def write_records(self, records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        rows = []
        for record in records:
            material_id = str(record.get("material_id") or "")
            if not material_id:
                continue
            path = self.out_dir / f"{material_id}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            rows.append({
                "material_id": material_id,
                "formula": record.get("formula_pretty", ""),
                "path": str(path),
                "sha256": sha256_file(path),
                "source": "materials_project",
                "retrieved_at": utc_now(),
            })
        return rows

    def run(self, chemsys_values: Iterable[str], per_system: int = 100) -> Dict[str, object]:
        all_rows = []
        for chemsys in chemsys_values:
            rows = self.write_records(self.search(chemsys, limit=per_system))
            all_rows.extend(rows)
            time.sleep(0.25)
        manifest = {
            "created_at": utc_now(),
            "source": "materials_project",
            "records": len(all_rows),
            "rows": all_rows,
            "legal_access": "Requires user-provided Materials Project API key.",
        }
        write_json(self.out_dir / "crystal_structure_manifest.json", manifest)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/real/crystal_structures/materials_project")
    parser.add_argument("--api-key", default=os.getenv("MP_API_KEY", ""))
    parser.add_argument("--chemsys", action="append", default=None)
    parser.add_argument("--per-system", type=int, default=100)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("MP_API_KEY or --api-key is required for legal Materials Project access.")
    harvester = MaterialsProjectHarvester(args.api_key, Path(args.out_dir))
    manifest = harvester.run(args.chemsys or NA_ION_FORMULAS, per_system=args.per_system)
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
