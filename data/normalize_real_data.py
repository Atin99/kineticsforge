import argparse
import json
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.dataset_contracts import describe_array, ensure_dir, sha256_file, utc_now, write_json


BATTERYLIFE_SOURCE = {
    "source_id": "batterylife_processed_v10",
    "source_url": "https://zenodo.org/records/18646655",
    "raw_source_url": "https://zenodo.org/records/14904364",
    "paper": "BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction",
    "license_note": "CC BY 4.0 on the Zenodo processed record; cite BatteryLife and the original source dataset.",
}


BATTERYLIFE_DATASET_META: Dict[str, Dict[str, str]] = {
    "NA-ion": {
        "source_id": "batterylife_processed_v10_naion",
        "chemistry_family": "commercial Na-ion 18650",
    },
    "UL_PUR": {
        "source_id": "batterylife_processed_v10_ul_pur",
        "chemistry_family": "commercial Li-ion NCA/graphite 18650",
    },
}


class BatteryLifeProcessedNormalizer:
    """Normalize BatteryLife v10 uniform pickle archives without inventing fields."""

    def __init__(
        self,
        root: Path,
        archives: Optional[Iterable[str]] = None,
        max_points_per_cycle: int = 24,
    ):
        self.root = root
        self.real_dir = ensure_dir(root / "real")
        self.source_dir = ensure_dir(self.real_dir / "batterylife_processed_v10")
        self.out_dir = ensure_dir(self.real_dir / "normalized")
        self.max_points_per_cycle = max_points_per_cycle
        self.labels_zip = self.source_dir / "Life labels.zip"
        self.readme_zip = self.source_dir / "READMEs.zip"
        self.archives = self.resolve_archives(archives)

    def resolve_archives(self, archives: Optional[Iterable[str]]) -> List[Path]:
        if archives:
            out = [self.source_dir / name for name in archives]
        else:
            out = [
                path
                for path in sorted(self.source_dir.glob("*.zip"))
                if path.name not in {"Life labels.zip", "READMEs.zip"}
            ]
        missing = [str(path) for path in out if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing BatteryLife archive(s): {missing}")
        return out

    @staticmethod
    def dataset_key(archive: Path) -> str:
        return archive.stem

    @staticmethod
    def archive_slug(dataset_key: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", dataset_key.lower()).strip("_")

    def shard_paths(self, dataset_key: str) -> Tuple[Path, Path]:
        slug = self.archive_slug(dataset_key)
        return (
            self.out_dir / f"batterylife_processed_{slug}_cycle_summary.parquet",
            self.out_dir / f"batterylife_processed_{slug}_timeseries_sample.parquet",
        )

    @staticmethod
    def label_candidates(dataset_key: str) -> List[str]:
        variants = [
            dataset_key,
            dataset_key.replace("_", "-"),
            dataset_key.replace("-", "_"),
        ]
        return [f"Life labels/{variant}_labels.json" for variant in dict.fromkeys(variants)]

    def labels(self, dataset_key: str) -> Dict[str, int]:
        if not self.labels_zip.exists():
            return {}
        with zipfile.ZipFile(self.labels_zip) as z:
            names = set(z.namelist())
            for candidate in self.label_candidates(dataset_key):
                if candidate in names:
                    data = z.read(candidate)
                    return {str(k): int(v) for k, v in json.loads(data.decode("utf-8")).items()}
        return {}

    def readme(self, dataset_key: str) -> str:
        if not self.readme_zip.exists():
            return ""
        candidate = f"READMEs/{dataset_key}_README.md"
        with zipfile.ZipFile(self.readme_zip) as z:
            if candidate in z.namelist():
                return z.read(candidate).decode("utf-8", errors="replace")
        return ""

    def source_for(self, dataset_key: str) -> Dict[str, str]:
        meta = BATTERYLIFE_DATASET_META.get(dataset_key, {})
        slug = re.sub(r"[^a-z0-9]+", "_", dataset_key.lower()).strip("_")
        return {
            **BATTERYLIFE_SOURCE,
            "source_id": meta.get("source_id", f"batterylife_processed_v10_{slug}"),
            "chemistry_family": meta.get("chemistry_family", ""),
            "dataset_key": dataset_key,
        }

    def file_temperature(self, name: str, data: Dict[str, Any]) -> Optional[float]:
        candidates = re.findall(r"[-_](?:T)?(-?\d{1,3})C(?:[_-]|$)", name)
        if candidates:
            return float(candidates[-1])
        simple = re.findall(r"[-_](25|30|35|40|45|50)(?:[_-]|$)", name)
        if simple:
            return float(simple[-1])
        desc = " ".join([str(data.get("description") or ""), str(data.get("reference") or "")])
        m = re.search(r"(-?\d{1,3})\s*(?:deg|degrees|°|C)", desc, re.I)
        if m:
            return float(m.group(1))
        return None

    def protocol_rate(self, data: Dict[str, Any], key: str) -> Optional[float]:
        proto = data.get(key) or []
        if isinstance(proto, list) and proto:
            rate = proto[0].get("rate_in_C") if isinstance(proto[0], dict) else None
            if rate is not None:
                try:
                    return float(rate)
                except (TypeError, ValueError):
                    return None
        return None

    def protocol_label(self, data: Dict[str, Any], key: str) -> Optional[str]:
        proto = data.get(key)
        if not proto:
            return None
        if isinstance(proto, list) and proto:
            first = proto[0]
            if isinstance(first, dict):
                rate = first.get("rate_in_C")
                if isinstance(rate, str):
                    return rate
                return None
            return str(first)
        return str(proto)

    @staticmethod
    def arr(values: Any) -> np.ndarray:
        if values is None:
            return np.array([], dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    @staticmethod
    def safe_stat(values: np.ndarray, fn: str) -> Optional[float]:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        if fn == "min":
            return float(np.min(values))
        if fn == "max":
            return float(np.max(values))
        if fn == "mean":
            return float(np.mean(values))
        if fn == "median":
            return float(np.median(values))
        if fn == "std":
            return float(np.std(values))
        return None

    @staticmethod
    def energy_wh(voltage: np.ndarray, current: np.ndarray, time_s: np.ndarray, sign: str) -> Optional[float]:
        if len(voltage) < 2 or len(current) < 2 or len(time_s) < 2:
            return None
        n = min(len(voltage), len(current), len(time_s))
        v = voltage[:n]
        i = current[:n]
        t = time_s[:n]
        mask = i > 0 if sign == "charge" else i < 0
        if np.sum(mask) < 2:
            return 0.0
        order = np.argsort(t[mask])
        power_w = np.abs(v[mask][order] * i[mask][order])
        seconds = t[mask][order]
        return float(np.trapezoid(power_w, seconds) / 3600.0)

    @staticmethod
    def sample_indices(n: int, max_points: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        if n <= max_points:
            return np.arange(n, dtype=int)
        return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))

    def normalize_cell(
        self,
        archive: Path,
        source_file: str,
        data: Dict[str, Any],
        life_labels: Dict[str, int],
        zip_sha256: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        dataset_key = self.dataset_key(archive)
        source = self.source_for(dataset_key)
        cell_id = str(data.get("cell_id") or Path(source_file).stem)
        basename = Path(source_file).name
        cycle_life = life_labels.get(basename)
        temp_c = self.file_temperature(basename, data)
        charge_c = self.protocol_rate(data, "charge_protocol")
        discharge_c = self.protocol_rate(data, "discharge_protocol")
        charge_protocol_label = self.protocol_label(data, "charge_protocol")
        discharge_protocol_label = self.protocol_label(data, "discharge_protocol")
        nominal_capacity = float(data.get("nominal_capacity_in_Ah") or 1.0)
        metadata = {
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "raw_source_url": source["raw_source_url"],
            "citation": source["paper"],
            "license_note": source["license_note"],
            "dataset_key": dataset_key,
            "chemistry_family": source.get("chemistry_family") or None,
            "source_file": source_file,
            "source_archive": str(archive),
            "source_archive_sha256": zip_sha256,
            "cell_id": cell_id,
            "form_factor": data.get("form_factor"),
            "battery_type": "Na-ion" if dataset_key == "NA-ion" else "Li-ion",
            "anode_material": data.get("anode_material"),
            "cathode_material": data.get("cathode_material"),
            "electrolyte_material": data.get("electrolyte_material"),
            "nominal_capacity_Ah": nominal_capacity,
            "operation_temperature_C": temp_c,
            "charge_rate_C": charge_c,
            "discharge_rate_C": discharge_c,
            "charge_protocol_label": charge_protocol_label,
            "discharge_protocol_label": discharge_protocol_label,
            "cycle_life_label": cycle_life,
            "max_voltage_limit_V": data.get("max_voltage_limit_in_V"),
            "min_voltage_limit_V": data.get("min_voltage_limit_in_V"),
            "max_current_limit_A": data.get("max_current_limit_in_A"),
            "min_current_limit_A": data.get("min_current_limit_in_A"),
            "depth_of_charge": data.get("depth_of_charge"),
            "depth_of_discharge": data.get("depth_of_discharge"),
            "already_spent_cycles": data.get("already_spent_cycles"),
            "schema_version": "batterylife-uniform-v2",
        }
        cycle_rows: List[Dict[str, Any]] = []
        timeseries_rows: List[Dict[str, Any]] = []
        cycle_data = data.get("cycle_data") or []
        for cycle_idx, cycle in enumerate(cycle_data):
            if not isinstance(cycle, dict):
                continue
            current = self.arr(cycle.get("current_in_A"))
            voltage = self.arr(cycle.get("voltage_in_V"))
            q_charge = self.arr(cycle.get("charge_capacity_in_Ah"))
            q_discharge = self.arr(cycle.get("discharge_capacity_in_Ah"))
            time_s = self.arr(cycle.get("time_in_s"))
            temp = self.arr(cycle.get("temperature_in_C"))
            r_int = self.arr(cycle.get("internal_resistance_in_ohm"))
            n = min(len(current), len(voltage), len(q_charge), len(q_discharge), len(time_s))
            if n <= 0:
                continue
            current = current[:n]
            voltage = voltage[:n]
            q_charge = q_charge[:n]
            q_discharge = q_discharge[:n]
            time_s = time_s[:n]
            cycle_number = int(cycle.get("cycle_number") or (cycle_idx + 1))
            qd_max = self.safe_stat(q_discharge, "max")
            qc_max = self.safe_stat(q_charge, "max")
            qd_norm = (qd_max / nominal_capacity) if qd_max is not None and nominal_capacity else None
            qc_norm = (qc_max / nominal_capacity) if qc_max is not None and nominal_capacity else None
            charge_wh = self.energy_wh(voltage, current, time_s, "charge")
            discharge_wh = self.energy_wh(voltage, current, time_s, "discharge")
            cycle_rows.append(
                {
                    **metadata,
                    "cycle_number": cycle_number,
                    "sample_count": int(n),
                    "duration_s": float(np.nanmax(time_s) - np.nanmin(time_s)) if len(time_s) else None,
                    "charge_capacity_Ah": qc_max,
                    "discharge_capacity_Ah": qd_max,
                    "normalized_charge_capacity": qc_norm,
                    "normalized_discharge_capacity": qd_norm,
                    "coulombic_efficiency": (qd_max / qc_max) if qd_max is not None and qc_max and qc_max > 1e-9 else None,
                    "charge_energy_Wh": charge_wh,
                    "discharge_energy_Wh": discharge_wh,
                    "energy_efficiency": (discharge_wh / charge_wh) if charge_wh and charge_wh > 1e-9 and discharge_wh is not None else None,
                    "voltage_min_V": self.safe_stat(voltage, "min"),
                    "voltage_max_V": self.safe_stat(voltage, "max"),
                    "voltage_mean_V": self.safe_stat(voltage, "mean"),
                    "current_mean_A": self.safe_stat(current, "mean"),
                    "current_abs_mean_A": self.safe_stat(np.abs(current), "mean"),
                    "current_abs_max_A": self.safe_stat(np.abs(current), "max"),
                    "temperature_mean_C": self.safe_stat(temp, "mean") if len(temp) else temp_c,
                    "internal_resistance_mean_ohm": self.safe_stat(r_int, "mean") if len(r_int) else None,
                    "row_kind": "cycle_summary",
                    "created_at": utc_now(),
                }
            )
            idxs = self.sample_indices(n, self.max_points_per_cycle)
            for point_idx in idxs:
                timeseries_rows.append(
                    {
                        **metadata,
                        "cycle_number": cycle_number,
                        "point_index": int(point_idx),
                        "time_s": float(time_s[point_idx]),
                        "current_A": float(current[point_idx]),
                        "voltage_V": float(voltage[point_idx]),
                        "charge_capacity_Ah": float(q_charge[point_idx]),
                        "discharge_capacity_Ah": float(q_discharge[point_idx]),
                        "normalized_charge_capacity": float(q_charge[point_idx] / nominal_capacity) if nominal_capacity else None,
                        "normalized_discharge_capacity": float(q_discharge[point_idx] / nominal_capacity) if nominal_capacity else None,
                        "temperature_C": float(temp[point_idx]) if len(temp) > point_idx else temp_c,
                        "internal_resistance_ohm": float(r_int[point_idx]) if len(r_int) > point_idx else None,
                        "row_kind": "timeseries_sample",
                        "created_at": utc_now(),
                    }
                )
        return cycle_rows, timeseries_rows

    def run(
        self,
        max_cells_per_archive: Optional[int] = None,
        write_combined: bool = True,
        report_suffix: str = "",
    ) -> Dict[str, Any]:
        cycle_shards: List[Path] = []
        timeseries_shards: List[Path] = []
        archive_reports: List[Dict[str, Any]] = []
        for archive in self.archives:
            dataset_key = self.dataset_key(archive)
            labels = self.labels(dataset_key)
            zip_sha = sha256_file(archive)
            processed = 0
            archive_cycle_rows: List[Dict[str, Any]] = []
            archive_timeseries_rows: List[Dict[str, Any]] = []
            with zipfile.ZipFile(archive) as z:
                names = sorted(n for n in z.namelist() if n.endswith(".pkl"))
                if max_cells_per_archive is not None:
                    names = names[:max_cells_per_archive]
                for name in names:
                    with z.open(name) as member:
                        data = pickle.load(member)
                    if not isinstance(data, dict):
                        continue
                    cycle_rows, timeseries_rows = self.normalize_cell(archive, name, data, labels, zip_sha)
                    archive_cycle_rows.extend(cycle_rows)
                    archive_timeseries_rows.extend(timeseries_rows)
                    processed += 1

            shard_cycle_path, shard_ts_path = self.shard_paths(dataset_key)
            if archive_cycle_rows:
                pd.DataFrame(archive_cycle_rows).to_parquet(shard_cycle_path, index=False)
                cycle_shards.append(shard_cycle_path)
            if archive_timeseries_rows:
                pd.DataFrame(archive_timeseries_rows).to_parquet(shard_ts_path, index=False)
                timeseries_shards.append(shard_ts_path)
            archive_reports.append(
                {
                    "dataset_key": dataset_key,
                    "source_archive": str(archive),
                    "source_archive_sha256": zip_sha,
                    "cells_processed": processed,
                    "life_label_count": len(labels),
                    "cycle_rows": len(archive_cycle_rows),
                    "timeseries_rows": len(archive_timeseries_rows),
                    "cycle_shard": str(shard_cycle_path) if archive_cycle_rows else "",
                    "timeseries_shard": str(shard_ts_path) if archive_timeseries_rows else "",
                    "readme_available": bool(self.readme(dataset_key)),
                }
            )

        if write_combined:
            cycle_df = pd.concat([pd.read_parquet(path) for path in cycle_shards], ignore_index=True, sort=False) if cycle_shards else pd.DataFrame()
            ts_df = pd.concat([pd.read_parquet(path) for path in timeseries_shards], ignore_index=True, sort=False) if timeseries_shards else pd.DataFrame()
        else:
            cycle_df = pd.DataFrame()
            ts_df = pd.DataFrame()
        cycle_path = self.out_dir / "batterylife_processed_cycle_summary.parquet"
        ts_path = self.out_dir / "batterylife_processed_timeseries_sample.parquet"
        if write_combined and not cycle_df.empty:
            cycle_df.to_parquet(cycle_path, index=False)
        if write_combined and not ts_df.empty:
            ts_df.to_parquet(ts_path, index=False)
        cycle_row_count = int(len(cycle_df)) if write_combined else int(sum(item["cycle_rows"] for item in archive_reports))
        timeseries_row_count = int(len(ts_df)) if write_combined else int(sum(item["timeseries_rows"] for item in archive_reports))
        report = {
            "created_at": utc_now(),
            "source": BATTERYLIFE_SOURCE,
            "archives": archive_reports,
            "cells_processed": int(sum(item["cells_processed"] for item in archive_reports)),
            "cycle_rows": cycle_row_count,
            "timeseries_rows": timeseries_row_count,
            "total_real_rows": int(cycle_row_count + timeseries_row_count),
            "paths": {
                "combined_written": write_combined,
                "cycle_summary": str(cycle_path) if write_combined else "",
                "timeseries_sample": str(ts_path) if write_combined else "",
                "cycle_shards": [str(path) for path in cycle_shards],
                "timeseries_shards": [str(path) for path in timeseries_shards],
                "readme_zip": str(self.readme_zip),
                "labels_zip": str(self.labels_zip),
            },
            "metrics": {
                "dataset_keys": sorted(cycle_df["dataset_key"].dropna().unique().tolist()) if "dataset_key" in cycle_df else sorted(item["dataset_key"] for item in archive_reports),
                "cell_count": int(cycle_df["cell_id"].nunique()) if "cell_id" in cycle_df else int(sum(item["cells_processed"] for item in archive_reports)),
                "normalized_discharge_capacity": describe_array(cycle_df["normalized_discharge_capacity"].dropna().to_numpy()) if "normalized_discharge_capacity" in cycle_df else {},
                "cycle_life_label": describe_array(cycle_df["cycle_life_label"].dropna().to_numpy()) if "cycle_life_label" in cycle_df else {},
                "voltage_V": describe_array(ts_df["voltage_V"].dropna().to_numpy()) if "voltage_V" in ts_df else {},
                "current_A": describe_array(ts_df["current_A"].dropna().to_numpy()) if "current_A" in ts_df else {},
            },
        }
        safe_suffix = self.archive_slug(report_suffix) if report_suffix else ""
        report_name = (
            f"batterylife_processed_{safe_suffix}_normalization_report.json"
            if safe_suffix
            else "batterylife_processed_normalization_report.json"
        )
        write_json(self.out_dir / report_name, report)
        return report


class BatteryLifeNAIonNormalizer(BatteryLifeProcessedNormalizer):
    def __init__(self, root: Path, max_points_per_cycle: int = 24):
        super().__init__(
            root,
            archives=["NA-ion.zip"],
            max_points_per_cycle=max_points_per_cycle,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize downloaded BatteryLife v10 real data into provenance-rich parquet rows.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--archives", nargs="*", default=None, help="Specific BatteryLife archive names. Defaults to every downloaded data zip.")
    parser.add_argument("--max-cells-per-archive", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None, help="Deprecated alias for --max-cells-per-archive.")
    parser.add_argument("--max-points-per-cycle", type=int, default=24)
    parser.add_argument("--skip-combined", action="store_true", help="Write per-archive shards and report only; do not overwrite combined parquet files.")
    parser.add_argument("--report-suffix", default="", help="Optional suffix for the normalization report file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_cells = args.max_cells_per_archive if args.max_cells_per_archive is not None else args.max_cells
    report = BatteryLifeProcessedNormalizer(
        Path(args.root).resolve(),
        archives=args.archives,
        max_points_per_cycle=args.max_points_per_cycle,
    ).run(max_cells_per_archive=max_cells, write_combined=not args.skip_combined, report_suffix=args.report_suffix)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
