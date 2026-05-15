import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

from data.dataset_contracts import describe_array, ensure_dir, sha256_file, utc_now, write_json


NASA_PCOE_SOURCE = {
    "source_id": "nasa_pcoe_battery_aging",
    "source_url": "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
    "download_url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
    "landing_url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
    "citation": 'B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.',
    "license_note": "Public NASA Prognostics Data Repository dataset; acknowledge NASA PCoE and data donors.",
}


class NASAPCoENormalizer:
    def __init__(self, root: Path, max_points_per_operation: int = 24):
        self.root = root
        self.real_dir = ensure_dir(root / "real")
        self.source_dir = ensure_dir(self.real_dir / "nasa_pcoe_battery_aging")
        self.out_dir = ensure_dir(self.real_dir / "normalized")
        self.archive = self.source_dir / "NASA_5_Battery_Data_Set.zip"
        self.max_points_per_operation = max_points_per_operation
        self.nominal_capacity_Ah = 2.0

    @staticmethod
    def sha256_bytes(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def arr(values: Any, dtype: Any = np.float64) -> np.ndarray:
        if values is None:
            return np.array([], dtype=dtype)
        try:
            arr = np.asarray(values, dtype=dtype)
        except Exception:
            return np.array([], dtype=dtype)
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr.reshape(-1)

    @staticmethod
    def scalar(values: Any) -> Optional[float]:
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr[0])

    @staticmethod
    def safe_stat(values: np.ndarray, fn: str) -> Optional[float]:
        values = np.asarray(values, dtype=float)
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
    def safe_complex_abs(values: Any) -> np.ndarray:
        if values is None:
            return np.array([], dtype=float)
        try:
            return np.abs(np.asarray(values).reshape(-1)).astype(float)
        except Exception:
            return np.array([], dtype=float)

    @staticmethod
    def sample_indices(n: int, max_points: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        if n <= max_points:
            return np.arange(n, dtype=int)
        return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))

    @staticmethod
    def date_vector_to_iso(values: Any) -> Optional[str]:
        try:
            vec = np.asarray(values, dtype=float).reshape(-1)
            year, month, day, hour, minute = [int(v) for v in vec[:5]]
            second_float = float(vec[5]) if len(vec) > 5 else 0.0
            second = int(second_float)
            microsecond = int(round((second_float - second) * 1_000_000))
            return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc).isoformat()
        except Exception:
            return None

    def energy_wh(self, voltage: np.ndarray, current: np.ndarray, time_s: np.ndarray, operation_type: str) -> Optional[float]:
        if len(voltage) < 2 or len(current) < 2 or len(time_s) < 2:
            return None
        n = min(len(voltage), len(current), len(time_s))
        v = voltage[:n]
        i = current[:n]
        t = time_s[:n]
        if operation_type == "charge":
            mask = i > 0
        elif operation_type == "discharge":
            mask = i < 0
        else:
            mask = np.isfinite(i)
        if np.sum(mask) < 2:
            mask = np.isfinite(i) & np.isfinite(v) & np.isfinite(t)
        if np.sum(mask) < 2:
            return None
        order = np.argsort(t[mask])
        power_w = np.abs(v[mask][order] * i[mask][order])
        seconds = t[mask][order]
        return float(np.trapezoid(power_w, seconds) / 3600.0)

    def metadata(self, cell_id: str, nested_archive: str, source_file: str, mat_sha256: str) -> Dict[str, Any]:
        return {
            "source_id": NASA_PCOE_SOURCE["source_id"],
            "source_url": NASA_PCOE_SOURCE["source_url"],
            "raw_source_url": NASA_PCOE_SOURCE["download_url"],
            "landing_url": NASA_PCOE_SOURCE["landing_url"],
            "citation": NASA_PCOE_SOURCE["citation"],
            "license_note": NASA_PCOE_SOURCE["license_note"],
            "dataset_key": "NASA_PCoE",
            "source_archive": str(self.archive),
            "source_archive_sha256": sha256_file(self.archive),
            "nested_archive": nested_archive,
            "source_file": source_file,
            "source_file_sha256": mat_sha256,
            "cell_id": cell_id,
            "battery_type": "Li-ion",
            "form_factor": "18650",
            "nominal_capacity_Ah": self.nominal_capacity_Ah,
            "schema_version": "nasa-pcoe-v1",
        }

    def collect_mat_files(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        if not self.archive.exists():
            raise FileNotFoundError(f"Missing {self.archive}. Run data.real_data_catalog for nasa_pcoe_battery_aging first.")
        selected: Dict[str, Dict[str, Any]] = {}
        conflicts: List[Dict[str, Any]] = []
        readmes: List[str] = []
        with zipfile.ZipFile(self.archive) as outer:
            for nested_name in sorted(n for n in outer.namelist() if n.endswith(".zip")):
                nested_bytes = outer.read(nested_name)
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                    for name in nested.namelist():
                        if name.lower().endswith("readme.txt"):
                            readmes.append(f"{nested_name}:{name}")
                            continue
                        if not name.lower().endswith(".mat"):
                            continue
                        payload = nested.read(name)
                        cell_id = Path(name).stem
                        item = {
                            "cell_id": cell_id,
                            "nested_archive": nested_name,
                            "source_file": name,
                            "payload": payload,
                            "bytes": len(payload),
                            "sha256": self.sha256_bytes(payload),
                        }
                        prior = selected.get(cell_id)
                        if prior is None:
                            selected[cell_id] = item
                        elif prior["sha256"] != item["sha256"]:
                            conflicts.append(
                                {
                                    "cell_id": cell_id,
                                    "kept_nested_archive": prior["nested_archive"],
                                    "candidate_nested_archive": item["nested_archive"],
                                    "kept_bytes": prior["bytes"],
                                    "candidate_bytes": item["bytes"],
                                }
                            )
                            if item["bytes"] > prior["bytes"]:
                                selected[cell_id] = item
        return list(selected.values()), conflicts, readmes

    def normalize_cell(self, item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        mat = sio.loadmat(io.BytesIO(item["payload"]), squeeze_me=True, struct_as_record=False)
        cell_id = str(item["cell_id"])
        obj = mat[cell_id]
        cycles = np.asarray(obj.cycle).reshape(-1)
        meta = self.metadata(cell_id, str(item["nested_archive"]), str(item["source_file"]), str(item["sha256"]))
        cycle_rows: List[Dict[str, Any]] = []
        timeseries_rows: List[Dict[str, Any]] = []
        impedance_rows: List[Dict[str, Any]] = []
        discharge_index = 0
        charge_index = 0
        impedance_index = 0
        for operation_index, operation in enumerate(cycles):
            operation_type = str(getattr(operation, "type", "")).lower()
            data = getattr(operation, "data", None)
            ambient_temperature = self.scalar(getattr(operation, "ambient_temperature", None))
            started_at = self.date_vector_to_iso(getattr(operation, "time", None))
            operation_meta = {
                **meta,
                "operation_index": int(operation_index),
                "operation_type": operation_type,
                "operation_started_at": started_at,
                "operation_temperature_C": ambient_temperature,
            }
            if operation_type in {"charge", "discharge"} and data is not None:
                voltage = self.arr(getattr(data, "Voltage_measured", None))
                current = self.arr(getattr(data, "Current_measured", None))
                temp = self.arr(getattr(data, "Temperature_measured", None))
                time_s = self.arr(getattr(data, "Time", None))
                n = min(len(voltage), len(current), len(time_s))
                if n <= 0:
                    continue
                voltage = voltage[:n]
                current = current[:n]
                time_s = time_s[:n]
                capacity = self.scalar(getattr(data, "Capacity", None)) if operation_type == "discharge" else None
                normalized_capacity = capacity / self.nominal_capacity_Ah if capacity is not None else None
                if operation_type == "discharge":
                    discharge_index += 1
                    cycle_number = discharge_index
                    cycle_rows.append(
                        {
                            **operation_meta,
                            "cycle_number": int(cycle_number),
                            "sample_count": int(n),
                            "duration_s": float(np.nanmax(time_s) - np.nanmin(time_s)) if len(time_s) else None,
                            "discharge_capacity_Ah": capacity,
                            "normalized_discharge_capacity": normalized_capacity,
                            "charge_capacity_Ah": None,
                            "normalized_charge_capacity": None,
                            "coulombic_efficiency": None,
                            "discharge_energy_Wh": self.energy_wh(voltage, current, time_s, "discharge"),
                            "charge_energy_Wh": None,
                            "energy_efficiency": None,
                            "voltage_min_V": self.safe_stat(voltage, "min"),
                            "voltage_max_V": self.safe_stat(voltage, "max"),
                            "voltage_mean_V": self.safe_stat(voltage, "mean"),
                            "current_mean_A": self.safe_stat(current, "mean"),
                            "current_abs_mean_A": self.safe_stat(np.abs(current), "mean"),
                            "current_abs_max_A": self.safe_stat(np.abs(current), "max"),
                            "temperature_mean_C": self.safe_stat(temp, "mean") if len(temp) else ambient_temperature,
                            "row_kind": "cycle_summary",
                            "created_at": utc_now(),
                        }
                    )
                else:
                    charge_index += 1
                    cycle_number = charge_index
                for point_idx in self.sample_indices(n, self.max_points_per_operation):
                    timeseries_rows.append(
                        {
                            **operation_meta,
                            "cycle_number": int(cycle_number),
                            "point_index": int(point_idx),
                            "time_s": float(time_s[point_idx]),
                            "current_A": float(current[point_idx]),
                            "voltage_V": float(voltage[point_idx]),
                            "temperature_C": float(temp[point_idx]) if len(temp) > point_idx else ambient_temperature,
                            "discharge_capacity_Ah": capacity,
                            "normalized_discharge_capacity": normalized_capacity,
                            "row_kind": "timeseries_sample",
                            "created_at": utc_now(),
                        }
                    )
            elif operation_type == "impedance" and data is not None:
                impedance_index += 1
                battery_impedance = self.safe_complex_abs(getattr(data, "Battery_impedance", None))
                rectified_impedance = self.safe_complex_abs(getattr(data, "Rectified_Impedance", None))
                impedance_rows.append(
                    {
                        **operation_meta,
                        "impedance_index": int(impedance_index),
                        "cycle_number": int(discharge_index) if discharge_index else None,
                        "Re_ohm": self.scalar(getattr(data, "Re", None)),
                        "Rct_ohm": self.scalar(getattr(data, "Rct", None)),
                        "battery_impedance_abs_mean_ohm": self.safe_stat(battery_impedance, "mean"),
                        "battery_impedance_abs_min_ohm": self.safe_stat(battery_impedance, "min"),
                        "battery_impedance_abs_max_ohm": self.safe_stat(battery_impedance, "max"),
                        "rectified_impedance_abs_mean_ohm": self.safe_stat(rectified_impedance, "mean"),
                        "rectified_impedance_abs_min_ohm": self.safe_stat(rectified_impedance, "min"),
                        "rectified_impedance_abs_max_ohm": self.safe_stat(rectified_impedance, "max"),
                        "sample_count": int(max(len(battery_impedance), len(rectified_impedance))),
                        "row_kind": "impedance_summary",
                        "created_at": utc_now(),
                    }
                )
        return cycle_rows, timeseries_rows, impedance_rows

    def run(self, max_cells: Optional[int] = None) -> Dict[str, Any]:
        mat_files, conflicts, readmes = self.collect_mat_files()
        mat_files = sorted(mat_files, key=lambda item: item["cell_id"])
        if max_cells is not None:
            mat_files = mat_files[:max_cells]
        all_cycle_rows: List[Dict[str, Any]] = []
        all_timeseries_rows: List[Dict[str, Any]] = []
        all_impedance_rows: List[Dict[str, Any]] = []
        cell_reports: List[Dict[str, Any]] = []
        for item in mat_files:
            cycle_rows, timeseries_rows, impedance_rows = self.normalize_cell(item)
            all_cycle_rows.extend(cycle_rows)
            all_timeseries_rows.extend(timeseries_rows)
            all_impedance_rows.extend(impedance_rows)
            cell_reports.append(
                {
                    "cell_id": item["cell_id"],
                    "nested_archive": item["nested_archive"],
                    "source_file": item["source_file"],
                    "source_file_sha256": item["sha256"],
                    "cycle_rows": len(cycle_rows),
                    "timeseries_rows": len(timeseries_rows),
                    "impedance_rows": len(impedance_rows),
                }
            )
        cycle_df = pd.DataFrame(all_cycle_rows)
        ts_df = pd.DataFrame(all_timeseries_rows)
        imp_df = pd.DataFrame(all_impedance_rows)
        cycle_path = self.out_dir / "nasa_pcoe_cycle_summary.parquet"
        ts_path = self.out_dir / "nasa_pcoe_timeseries_sample.parquet"
        imp_path = self.out_dir / "nasa_pcoe_impedance_summary.parquet"
        if not cycle_df.empty:
            cycle_df.to_parquet(cycle_path, index=False)
        if not ts_df.empty:
            ts_df.to_parquet(ts_path, index=False)
        if not imp_df.empty:
            imp_df.to_parquet(imp_path, index=False)
        report = {
            "created_at": utc_now(),
            "source": NASA_PCOE_SOURCE,
            "source_archive": str(self.archive),
            "source_archive_sha256": sha256_file(self.archive),
            "cells_processed": len(cell_reports),
            "mat_conflicts": conflicts,
            "readme_files_seen": readmes,
            "cycle_rows": int(len(cycle_df)),
            "timeseries_rows": int(len(ts_df)),
            "impedance_rows": int(len(imp_df)),
            "total_real_rows": int(len(cycle_df) + len(ts_df) + len(imp_df)),
            "paths": {
                "cycle_summary": str(cycle_path),
                "timeseries_sample": str(ts_path),
                "impedance_summary": str(imp_path),
            },
            "cells": cell_reports,
            "metrics": {
                "normalized_discharge_capacity": describe_array(cycle_df["normalized_discharge_capacity"].dropna().to_numpy()) if "normalized_discharge_capacity" in cycle_df else {},
                "voltage_V": describe_array(ts_df["voltage_V"].dropna().to_numpy()) if "voltage_V" in ts_df else {},
                "current_A": describe_array(ts_df["current_A"].dropna().to_numpy()) if "current_A" in ts_df else {},
                "Re_ohm": describe_array(imp_df["Re_ohm"].dropna().to_numpy()) if "Re_ohm" in imp_df else {},
                "Rct_ohm": describe_array(imp_df["Rct_ohm"].dropna().to_numpy()) if "Rct_ohm" in imp_df else {},
            },
        }
        write_json(self.out_dir / "nasa_pcoe_normalization_report.json", report)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize NASA PCoE Li-ion battery aging MATLAB data into provenance-rich parquet rows.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-points-per-operation", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = NASAPCoENormalizer(
        Path(args.root).resolve(),
        max_points_per_operation=args.max_points_per_operation,
    ).run(max_cells=args.max_cells)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
