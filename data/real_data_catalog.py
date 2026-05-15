import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from data.dataset_contracts import ensure_dir, sha256_file, source_catalog, utc_now, write_json


ZENODO_RECORDS: Dict[str, Dict[str, str]] = {
    "batterylife_processed_v10": {
        "record_id": "18646655",
        "record_url": "https://zenodo.org/records/18646655",
        "api_url": "https://zenodo.org/api/records/18646655",
    }
}

DIRECT_SOURCE_FILES: Dict[str, Dict[str, object]] = {
    "nasa_pcoe_battery_aging": {
        "record_url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "files": [
            {
                "key": "NASA_5_Battery_Data_Set.zip",
                "size": 209_708_670,
                "checksum": "",
                "download_url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
            }
        ],
    },
    "isu_ilcc_battery_aging": {
        "record_url": "https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234",
        "files": [
            {
                "key": "Valid_cells.csv",
                "size": 1_727,
                "checksum": "",
                "download_url": "https://ndownloader.figshare.com/files/43754763",
            },
            {
                "key": "process_data.py",
                "size": 9_769,
                "checksum": "",
                "download_url": "https://ndownloader.figshare.com/files/43754835",
            },
            {
                "key": "README_V2.0.pdf",
                "size": 640_953,
                "checksum": "",
                "download_url": "https://ndownloader.figshare.com/files/43754898",
            },
            {
                "key": "capacity_fade.zip",
                "size": 140_256,
                "checksum": "",
                "download_url": "https://ndownloader.figshare.com/files/43755582",
            },
            {
                "key": "Q_interpolated.zip",
                "size": 47_797_453,
                "checksum": "",
                "download_url": "https://ndownloader.figshare.com/files/43755588",
            },
            {
                "key": "RPT_json.zip",
                "size": 929_406_930,
                "checksum": "",
                "download_url": "https://ndownloader.figshare.com/files/43756491",
            },
        ],
    }
}


class RealDataCatalog:
    def __init__(self, root: Path):
        self.root = root
        self.real_dir = ensure_dir(root / "real")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "KineticsForge/0.3 free-research-data-pipeline"})

    def write_catalog(self) -> Path:
        path = self.real_dir / "source_catalog.json"
        write_json(path, {"created_at": utc_now(), "sources": [s.__dict__ for s in source_catalog()]})
        return path

    def zenodo_files(self, source_id: str) -> List[Dict[str, object]]:
        record = ZENODO_RECORDS[source_id]
        try:
            resp = self.session.get(record["api_url"], timeout=45)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return self.fallback_zenodo_files(source_id)
        files = []
        for item in payload.get("files", []):
            key = item.get("key") or item.get("filename")
            links = item.get("links", {})
            url = links.get("self") or links.get("content") or f"{record['record_url']}/files/{key}?download=1"
            files.append(
                {
                    "key": key,
                    "size": int(item.get("size", 0)),
                    "checksum": item.get("checksum", ""),
                    "download_url": url,
                }
            )
        return files

    def source_files(self, source_id: str) -> List[Dict[str, object]]:
        if source_id in ZENODO_RECORDS:
            return self.zenodo_files(source_id)
        if source_id in DIRECT_SOURCE_FILES:
            return [dict(item) for item in DIRECT_SOURCE_FILES[source_id]["files"]]
        return []

    def fallback_zenodo_files(self, source_id: str) -> List[Dict[str, object]]:
        if source_id != "batterylife_processed_v10":
            return []
        base = ZENODO_RECORDS[source_id]["record_url"]
        known = {
            "CALB.zip": ("020155f525a9dff91df3696906b43589", 13_200_000),
            "NA-ion.zip": ("bf0a03ac84c74f87a02e203cfc1f9ebf", 289_300_000),
            "Life labels.zip": ("6a75015c69c66bde1d831c12deaa5792", 12_600),
            "READMEs.zip": ("f1b28ff26d2cbb1e81455518be9b0e23", 17_200),
            "CALCE.zip": ("4cbd6bcec6387739c89bfaa9914f184c", 88_900_000),
            "SNL.zip": ("900a5bb283ffb0b3255da618118510b7", 115_200_000),
            "HNEI.zip": ("27d009bbb908f04e90ecd9a145d81b62", 43_100_000),
            "MICH.zip": ("cc34ea7ed8edc6419cb30757548ca3da", 120_300_000),
            "MICH_EXP.zip": ("e267051a90f0fc02f8e6701b9f3ecc58", 67_600_000),
            "UL_PUR.zip": ("65551018b3d67d96eda724552a0360bd", 6_000_000),
            "XJTU.zip": ("3fb532b9bc88dc4fe3d73f305a673f8c", 397_000_000),
        }
        return [
            {"key": key, "size": size, "checksum": f"md5:{md5}", "download_url": f"{base}/files/{key}?download=1"}
            for key, (md5, size) in known.items()
        ]

    def plan(self, source_id: str, requested_files: Optional[Iterable[str]] = None, max_gb: float = 1.0, all_files: bool = False) -> Dict[str, object]:
        files = self.source_files(source_id)
        wanted = set(requested_files or [])
        selected = []
        total = 0
        default_batterylife = {"NA-ion.zip", "Life labels.zip", "READMEs.zip"}
        for item in files:
            if wanted and item["key"] not in wanted:
                continue
            if source_id == "batterylife_processed_v10" and not wanted and not all_files and item["key"] not in default_batterylife:
                continue
            if total + int(item["size"]) > max_gb * 1024**3:
                continue
            selected.append(item)
            total += int(item["size"])
        return {"source_id": source_id, "max_gb": max_gb, "total_bytes": total, "files": selected}

    def download_plan(self, plan: Dict[str, object], dry_run: bool = False) -> Path:
        source_id = str(plan["source_id"])
        out_dir = ensure_dir(self.real_dir / source_id)
        events = []
        for item in plan["files"]:
            name = str(item["key"])
            dest = out_dir / name
            event = {"file": name, "url": item["download_url"], "expected_size": item["size"], "path": str(dest), "status": "planned"}
            if dry_run:
                events.append(event)
                continue
            if dest.exists() and dest.stat().st_size > 0:
                event["status"] = "exists"
                event["bytes"] = dest.stat().st_size
                event["sha256"] = sha256_file(dest)
                events.append(event)
                continue
            with self.session.get(str(item["download_url"]), stream=True, timeout=180) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            event["status"] = "downloaded"
            event["bytes"] = dest.stat().st_size
            event["sha256"] = sha256_file(dest)
            events.append(event)
            time.sleep(0.5)
        log_path = out_dir / f"download_log_{int(time.time())}.json"
        write_json(log_path, {"created_at": utc_now(), "plan": plan, "events": events})
        return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Catalog and download free real battery data sources.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--source", default="batterylife_processed_v10", choices=sorted(set(ZENODO_RECORDS) | set(DIRECT_SOURCE_FILES)))
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--max-gb", type=float, default=1.0)
    parser.add_argument("--all-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-catalog", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = RealDataCatalog(Path(args.root).resolve())
    if args.write_catalog:
        catalog.write_catalog()
    plan = catalog.plan(args.source, args.files, args.max_gb, all_files=args.all_files)
    log_path = catalog.download_plan(plan, dry_run=args.dry_run)
    print(json.dumps({"plan": plan, "log": str(log_path), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
