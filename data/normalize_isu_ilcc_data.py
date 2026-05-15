import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.dataset_contracts import describe_array, ensure_dir, sha256_file, utc_now, write_json


ISU_ILCC_SOURCE = {
    "source_id": "isu_ilcc_battery_aging",
    "source_url": "https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234",
    "raw_source_url": "https://api.figshare.com/v2/articles/22582234",
    "download_urls": {
        "capacity_fade.zip": "https://ndownloader.figshare.com/files/43755582",
        "Q_interpolated.zip": "https://ndownloader.figshare.com/files/43755588",
        "RPT_json.zip": "https://ndownloader.figshare.com/files/43756491",
        "Valid_cells.csv": "https://ndownloader.figshare.com/files/43754763",
        "README_V2.0.pdf": "https://ndownloader.figshare.com/files/43754898",
        "process_data.py": "https://ndownloader.figshare.com/files/43754835",
    },
    "citation": "ISU-ILCC Battery Aging Dataset, Iowa State University / Iowa Lakes Community College / UConn REIL.",
    "license_note": "Figshare public dataset record lists CC BY 4.0; cite the dataset and source investigators.",
}


class ISUILCCNormalizer:
    """Normalize public ISU-ILCC battery aging derived files with explicit provenance."""

    def __init__(self, root: Path, max_voltage_points: int = 96):
        self.root = root
        self.real_dir = ensure_dir(root / "real")
        self.source_dir = ensure_dir(self.real_dir / "isu_ilcc_battery_aging")
        self.out_dir = ensure_dir(self.real_dir / "normalized")
        self.max_voltage_points = max_voltage_points
        self.capacity_zip = self.source_dir / "capacity_fade.zip"
        self.q_zip = self.source_dir / "Q_interpolated.zip"
        self.valid_cells_path = self.source_dir / "Valid_cells.csv"

    @staticmethod
    def sha256_bytes(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def sample_indices(n: int, max_points: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        if n <= max_points:
            return np.arange(n, dtype=int)
        return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))

    @staticmethod
    def cell_id_from_member(member: str) -> str:
        return Path(member).stem

    @staticmethod
    def release_from_member(member: str) -> str:
        parts = Path(member).parts
        for part in parts:
            if part.startswith("Release"):
                return part
        return ""

    def valid_cells(self) -> set[str]:
        if not self.valid_cells_path.exists():
            return set()
        df = pd.read_csv(self.valid_cells_path)
        if "Cell" not in df:
            return set()
        return {str(cell) for cell in df["Cell"].dropna().tolist()}

    def base_metadata(self, archive: Path, member: str, member_sha256: str) -> Dict[str, Any]:
        cell_id = self.cell_id_from_member(member)
        return {
            "source_id": ISU_ILCC_SOURCE["source_id"],
            "source_url": ISU_ILCC_SOURCE["source_url"],
            "raw_source_url": ISU_ILCC_SOURCE["download_urls"].get(archive.name, ISU_ILCC_SOURCE["raw_source_url"]),
            "citation": ISU_ILCC_SOURCE["citation"],
            "license_note": ISU_ILCC_SOURCE["license_note"],
            "dataset_key": "ISU_ILCC",
            "source_archive": str(archive),
            "source_archive_sha256": sha256_file(archive),
            "source_file": member,
            "source_file_sha256": member_sha256,
            "cell_id": cell_id,
            "release": self.release_from_member(member),
            "battery_type": "Li-ion polymer",
            "schema_version": "isu-ilcc-v1",
        }

    def normalize_capacity_fade(self) -> Tuple[pd.DataFrame, Dict[str, float], List[Dict[str, Any]]]:
        if not self.capacity_zip.exists():
            raise FileNotFoundError(f"Missing {self.capacity_zip}")
        rows: List[Dict[str, Any]] = []
        reports: List[Dict[str, Any]] = []
        initial_capacity: Dict[str, float] = {}
        valid = self.valid_cells()
        with zipfile.ZipFile(self.capacity_zip) as z:
            members = sorted(name for name in z.namelist() if name.lower().endswith(".csv"))
            for member in members:
                payload = z.read(member)
                member_sha = self.sha256_bytes(payload)
                cell_id = self.cell_id_from_member(member)
                meta = self.base_metadata(self.capacity_zip, member, member_sha)
                df = pd.read_csv(io.BytesIO(payload))
                if "Time" not in df or "Capacity" not in df:
                    reports.append({"source_file": member, "status": "skipped_schema", "columns": list(df.columns)})
                    continue
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Capacity"])
                finite_capacity = pd.to_numeric(df["Capacity"], errors="coerce").dropna()
                finite_capacity = finite_capacity[finite_capacity > 0]
                first_capacity = float(finite_capacity.iloc[0]) if not finite_capacity.empty else np.nan
                if np.isfinite(first_capacity) and first_capacity > 0:
                    initial_capacity[cell_id] = first_capacity
                for rpt_index, item in df.reset_index(drop=True).iterrows():
                    capacity = pd.to_numeric(pd.Series([item.get("Capacity")]), errors="coerce").iloc[0]
                    time_weeks = pd.to_numeric(pd.Series([item.get("Time")]), errors="coerce").iloc[0]
                    if not np.isfinite(capacity):
                        continue
                    rows.append(
                        {
                            **meta,
                            "is_valid_cell": cell_id in valid if valid else None,
                            "cycle_number": int(rpt_index),
                            "rpt_index": int(rpt_index),
                            "time_on_test_weeks": float(time_weeks) if np.isfinite(time_weeks) else None,
                            "discharge_capacity_Ah": float(capacity),
                            "initial_discharge_capacity_Ah": first_capacity if np.isfinite(first_capacity) else None,
                            "normalized_discharge_capacity": float(capacity / first_capacity) if np.isfinite(first_capacity) and first_capacity > 0 else None,
                            "cycle_number_note": "ISU-ILCC capacity_fade rows are RPT measurement index, not raw cycling cycle count.",
                            "row_kind": "rpt_capacity_summary",
                            "created_at": utc_now(),
                        }
                    )
                reports.append({"source_file": member, "status": "normalized", "rows": int(len(df)), "cell_id": cell_id})
        return pd.DataFrame(rows), initial_capacity, reports

    def normalize_q_interpolated(self, initial_capacity: Dict[str, float]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        if not self.q_zip.exists():
            raise FileNotFoundError(f"Missing {self.q_zip}")
        frames: List[pd.DataFrame] = []
        reports: List[Dict[str, Any]] = []
        valid = self.valid_cells()
        with zipfile.ZipFile(self.q_zip) as z:
            members = sorted(name for name in z.namelist() if name.lower().endswith(".csv"))
            for member in members:
                payload = z.read(member)
                member_sha = self.sha256_bytes(payload)
                cell_id = self.cell_id_from_member(member)
                meta = self.base_metadata(self.q_zip, member, member_sha)
                df = pd.read_csv(io.BytesIO(payload), header=None)
                if df.empty:
                    reports.append({"source_file": member, "status": "skipped_empty", "cell_id": cell_id})
                    continue
                df = df.apply(pd.to_numeric, errors="coerce")
                voltage_grid = np.linspace(3.0, 4.18, df.shape[0])
                selected = self.sample_indices(df.shape[0], self.max_voltage_points)
                sampled = df.iloc[selected, :].copy()
                sampled["voltage_index"] = selected
                long_df = sampled.melt(
                    id_vars="voltage_index",
                    var_name="rpt_index",
                    value_name="interpolated_discharge_q_Ah",
                )
                long_df = long_df.dropna(subset=["interpolated_discharge_q_Ah"])
                if long_df.empty:
                    reports.append({"source_file": member, "status": "skipped_no_finite_q", "cell_id": cell_id})
                    continue
                first_capacity = initial_capacity.get(cell_id)
                long_df["source_id"] = meta["source_id"]
                long_df["source_url"] = meta["source_url"]
                long_df["raw_source_url"] = meta["raw_source_url"]
                long_df["citation"] = meta["citation"]
                long_df["license_note"] = meta["license_note"]
                long_df["dataset_key"] = meta["dataset_key"]
                long_df["source_archive"] = meta["source_archive"]
                long_df["source_archive_sha256"] = meta["source_archive_sha256"]
                long_df["source_file"] = meta["source_file"]
                long_df["source_file_sha256"] = meta["source_file_sha256"]
                long_df["cell_id"] = meta["cell_id"]
                long_df["release"] = meta["release"]
                long_df["battery_type"] = meta["battery_type"]
                long_df["schema_version"] = meta["schema_version"]
                long_df["is_valid_cell"] = cell_id in valid if valid else None
                long_df["rpt_index"] = long_df["rpt_index"].astype(int)
                long_df["cycle_number"] = long_df["rpt_index"]
                long_df["point_index"] = long_df["voltage_index"].astype(int)
                long_df["voltage_V"] = voltage_grid[long_df["voltage_index"].to_numpy(dtype=int)]
                long_df["time_s"] = np.nan
                long_df["current_A"] = np.nan
                long_df["discharge_capacity_Ah"] = long_df["interpolated_discharge_q_Ah"].astype(float)
                long_df["normalized_discharge_capacity"] = (
                    long_df["discharge_capacity_Ah"] / first_capacity
                    if first_capacity is not None and np.isfinite(first_capacity) and first_capacity > 0
                    else np.nan
                )
                long_df["operation_type"] = "qv_discharge_c5_interpolated"
                long_df["voltage_grid_min_V"] = 3.0
                long_df["voltage_grid_max_V"] = 4.18
                long_df["cycle_number_note"] = "ISU-ILCC Q_interpolated columns are RPT measurement index, not raw cycling cycle count."
                long_df["row_kind"] = "q_interpolated_voltage_sample"
                long_df["created_at"] = utc_now()
                frames.append(long_df)
                reports.append(
                    {
                        "source_file": member,
                        "status": "normalized",
                        "cell_id": cell_id,
                        "voltage_points_in_file": int(df.shape[0]),
                        "rpt_columns": int(df.shape[1]),
                        "sampled_rows": int(len(long_df)),
                    }
                )
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(), reports

    def run(self) -> Dict[str, Any]:
        capacity_df, initial_capacity, capacity_reports = self.normalize_capacity_fade()
        q_df, q_reports = self.normalize_q_interpolated(initial_capacity)
        capacity_path = self.out_dir / "isu_ilcc_capacity_summary.parquet"
        q_path = self.out_dir / "isu_ilcc_q_interpolated_timeseries.parquet"
        if not capacity_df.empty:
            capacity_df.to_parquet(capacity_path, index=False)
        if not q_df.empty:
            q_df.to_parquet(q_path, index=False)
        report = {
            "created_at": utc_now(),
            "source": ISU_ILCC_SOURCE,
            "max_voltage_points": self.max_voltage_points,
            "capacity_rows": int(len(capacity_df)),
            "q_interpolated_rows": int(len(q_df)),
            "total_real_rows": int(len(capacity_df) + len(q_df)),
            "cell_count": int(capacity_df["cell_id"].nunique()) if not capacity_df.empty else 0,
            "paths": {
                "capacity_summary": str(capacity_path),
                "q_interpolated_timeseries": str(q_path),
                "capacity_archive": str(self.capacity_zip),
                "q_interpolated_archive": str(self.q_zip),
                "valid_cells": str(self.valid_cells_path),
            },
            "metrics": {
                "discharge_capacity_Ah": describe_array(capacity_df["discharge_capacity_Ah"].dropna().to_numpy()) if "discharge_capacity_Ah" in capacity_df else {},
                "normalized_discharge_capacity": describe_array(capacity_df["normalized_discharge_capacity"].dropna().to_numpy()) if "normalized_discharge_capacity" in capacity_df else {},
                "q_interpolated_Ah": describe_array(q_df["interpolated_discharge_q_Ah"].dropna().to_numpy()) if "interpolated_discharge_q_Ah" in q_df else {},
                "voltage_V": describe_array(q_df["voltage_V"].dropna().to_numpy()) if "voltage_V" in q_df else {},
            },
            "capacity_files": capacity_reports,
            "q_interpolated_files": q_reports,
        }
        write_json(self.out_dir / "isu_ilcc_normalization_report.json", report)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize public ISU-ILCC battery aging data into provenance-rich parquet rows.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--max-voltage-points", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ISUILCCNormalizer(Path(args.root).resolve(), max_voltage_points=args.max_voltage_points).run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
