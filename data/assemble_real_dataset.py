import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data.dataset_contracts import describe_array, ensure_dir, sha256_file, utc_now, write_json


class RealDatasetAssembler:
    def __init__(self, root: Path):
        self.root = root
        self.real_dir = ensure_dir(root / "real")
        self.normalized_dir = ensure_dir(self.real_dir / "normalized")
        self.scraped_dir = ensure_dir(self.real_dir / "scraped")
        self.out_dir = ensure_dir(self.real_dir / "assembled")

    def read_optional(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.DataFrame()

    def add_provenance(self, df: pd.DataFrame, path: Path, real_domain: str, data_class: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["real_domain"] = real_domain
        df["real_data_class"] = data_class
        df["provenance_path"] = str(path)
        df["provenance_sha256"] = sha256_file(path)
        return df

    def normalized_frames(self, paths: List[Path], real_domain: str, data_class: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for path in paths:
            df = self.add_provenance(self.read_optional(path), path, real_domain, data_class)
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    def battery_life_paths(self, kind: str) -> List[Path]:
        shard_paths = sorted(self.normalized_dir.glob(f"batterylife_processed_*_{kind}.parquet"))
        if shard_paths:
            return list(shard_paths)
        combined = self.normalized_dir / f"batterylife_processed_{kind}.parquet"
        fallback = self.normalized_dir / f"batterylife_naion_{kind}.parquet"
        paths: List[Path] = []
        if combined.exists():
            paths.append(combined)
        elif fallback.exists():
            paths.append(fallback)
        return paths

    @staticmethod
    def dedupe_rows(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        subset = [key for key in keys if key in df.columns]
        if not subset:
            return df
        return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    def battery_life_cycle_summary(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        df = self.normalized_frames(
            self.battery_life_paths("cycle_summary"),
            "cell_cycle_summary",
            "premade_public_dataset_normalized",
        )
        if not df.empty:
            frames.append(
                self.dedupe_rows(
                    df,
                    ["source_id", "source_archive_sha256", "source_file", "cell_id", "cycle_number"],
                )
            )
        nasa_path = self.normalized_dir / "nasa_pcoe_cycle_summary.parquet"
        nasa_df = self.add_provenance(
            self.read_optional(nasa_path),
            nasa_path,
            "cell_cycle_summary",
            "raw_public_dataset_normalized",
        )
        if not nasa_df.empty:
            frames.append(nasa_df)
        isu_path = self.normalized_dir / "isu_ilcc_capacity_summary.parquet"
        isu_df = self.add_provenance(
            self.read_optional(isu_path),
            isu_path,
            "rpt_capacity_summary",
            "raw_public_dataset_normalized",
        )
        if not isu_df.empty:
            frames.append(isu_df)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    def battery_life_timeseries(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        df = self.normalized_frames(
            self.battery_life_paths("timeseries_sample"),
            "cell_timeseries_sample",
            "premade_public_dataset_normalized",
        )
        if not df.empty:
            frames.append(
                self.dedupe_rows(
                    df,
                    ["source_id", "source_archive_sha256", "source_file", "cell_id", "cycle_number", "point_index"],
                )
            )
        nasa_path = self.normalized_dir / "nasa_pcoe_timeseries_sample.parquet"
        nasa_df = self.add_provenance(
            self.read_optional(nasa_path),
            nasa_path,
            "cell_timeseries_sample",
            "raw_public_dataset_normalized",
        )
        if not nasa_df.empty:
            frames.append(nasa_df)
        isu_path = self.normalized_dir / "isu_ilcc_q_interpolated_timeseries.parquet"
        isu_df = self.add_provenance(
            self.read_optional(isu_path),
            isu_path,
            "cell_qv_interpolated_sample",
            "raw_public_dataset_normalized",
        )
        if not isu_df.empty:
            frames.append(isu_df)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    def timeseries_sources(self) -> List[tuple[Path, str, str]]:
        sources: List[tuple[Path, str, str]] = [
            (path, "cell_timeseries_sample", "premade_public_dataset_normalized")
            for path in self.battery_life_paths("timeseries_sample")
        ]
        nasa_path = self.normalized_dir / "nasa_pcoe_timeseries_sample.parquet"
        if nasa_path.exists():
            sources.append((nasa_path, "cell_timeseries_sample", "raw_public_dataset_normalized"))
        isu_path = self.normalized_dir / "isu_ilcc_q_interpolated_timeseries.parquet"
        if isu_path.exists():
            sources.append((isu_path, "cell_qv_interpolated_sample", "raw_public_dataset_normalized"))
        return sources

    def impedance_measurements(self) -> pd.DataFrame:
        path = self.normalized_dir / "nasa_pcoe_impedance_summary.parquet"
        df = self.read_optional(path)
        return self.add_provenance(df, path, "cell_impedance_summary", "raw_public_dataset_normalized")

    def literature_measurements(self) -> pd.DataFrame:
        path = self.scraped_dir / "curated_literature_measurements.parquet"
        df = self.read_optional(path)
        if df.empty:
            return df
        df = df.copy()
        df["real_domain"] = "literature_measurement"
        df["real_data_class"] = "open_literature_scraped_curated"
        df["provenance_path"] = str(path)
        df["provenance_sha256"] = sha256_file(path)
        return df

    def row_index(self, cycle_df: pd.DataFrame, ts_df: pd.DataFrame, lit_df: pd.DataFrame, impedance_df: pd.DataFrame) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        if not cycle_df.empty:
            frames.append(
                pd.DataFrame(
                    {
                        "row_id": [f"cycle:{i}" for i in range(len(cycle_df))],
                        "real_domain": cycle_df["real_domain"].to_numpy(),
                        "real_data_class": cycle_df["real_data_class"].to_numpy(),
                        "source_id": cycle_df["source_id"].to_numpy(),
                        "source_url": cycle_df["source_url"].to_numpy(),
                        "citation": cycle_df["citation"].to_numpy(),
                        "cell_id": cycle_df["cell_id"].to_numpy(),
                        "cycle_number": cycle_df["cycle_number"].to_numpy(),
                        "primary_value": cycle_df["normalized_discharge_capacity"].to_numpy(),
                        "primary_unit": "fraction_nominal_capacity",
                        "provenance_path": cycle_df["provenance_path"].to_numpy(),
                        "provenance_sha256": cycle_df["provenance_sha256"].to_numpy(),
                    }
                )
            )
        if not ts_df.empty:
            frames.append(
                pd.DataFrame(
                    {
                        "row_id": [f"timeseries:{i}" for i in range(len(ts_df))],
                        "real_domain": ts_df["real_domain"].to_numpy(),
                        "real_data_class": ts_df["real_data_class"].to_numpy(),
                        "source_id": ts_df["source_id"].to_numpy(),
                        "source_url": ts_df["source_url"].to_numpy(),
                        "citation": ts_df["citation"].to_numpy(),
                        "cell_id": ts_df["cell_id"].to_numpy(),
                        "cycle_number": ts_df["cycle_number"].to_numpy(),
                        "primary_value": ts_df["voltage_V"].to_numpy(),
                        "primary_unit": "V",
                        "provenance_path": ts_df["provenance_path"].to_numpy(),
                        "provenance_sha256": ts_df["provenance_sha256"].to_numpy(),
                    }
                )
            )
        if not lit_df.empty:
            accepted = lit_df[lit_df.get("curation_status", "") == "accepted"].copy()
            if not accepted.empty:
                frames.append(
                    pd.DataFrame(
                        {
                            "row_id": [f"literature:{i}" for i in range(len(accepted))],
                            "real_domain": accepted["real_domain"].to_numpy(),
                            "real_data_class": accepted["real_data_class"].to_numpy(),
                            "source_id": "open_literature_scrape",
                            "source_url": accepted["source_url"].to_numpy(),
                            "citation": accepted["title"].to_numpy(),
                            "cell_id": "",
                            "cycle_number": accepted["cycle_count"].to_numpy(),
                            "primary_value": accepted["value"].to_numpy(),
                            "primary_unit": accepted["unit"].to_numpy(),
                            "provenance_path": accepted["provenance_path"].to_numpy(),
                            "provenance_sha256": accepted["provenance_sha256"].to_numpy(),
                        }
                    )
                )
        if not impedance_df.empty:
            primary = impedance_df["Re_ohm"] if "Re_ohm" in impedance_df else impedance_df.get("Rct_ohm", pd.Series(np.nan, index=impedance_df.index))
            frames.append(
                pd.DataFrame(
                    {
                        "row_id": [f"impedance:{i}" for i in range(len(impedance_df))],
                        "real_domain": impedance_df["real_domain"].to_numpy(),
                        "real_data_class": impedance_df["real_data_class"].to_numpy(),
                        "source_id": impedance_df["source_id"].to_numpy(),
                        "source_url": impedance_df["source_url"].to_numpy(),
                        "citation": impedance_df["citation"].to_numpy(),
                        "cell_id": impedance_df["cell_id"].to_numpy(),
                        "cycle_number": impedance_df["cycle_number"].to_numpy() if "cycle_number" in impedance_df else np.nan,
                        "primary_value": primary.to_numpy(),
                        "primary_unit": "ohm_Re",
                        "provenance_path": impedance_df["provenance_path"].to_numpy(),
                        "provenance_sha256": impedance_df["provenance_sha256"].to_numpy(),
                    }
                )
            )
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["assembled_at"] = utc_now()
        return out

    def write_partitioned_parquet(
        self,
        df: pd.DataFrame,
        out_dir: Path,
        rows_per_part: int = 250_000,
        start_part: int = 0,
        clean: bool = True,
    ) -> List[Path]:
        ensure_dir(out_dir)
        if clean:
            for old in out_dir.glob("*.parquet"):
                old.unlink()
        parts: List[Path] = []
        for offset, start in enumerate(range(0, len(df), rows_per_part)):
            part_idx = start_part + offset
            part = out_dir / f"part-{part_idx:04d}.parquet"
            df.iloc[start : start + rows_per_part].to_parquet(part, index=False)
            parts.append(part)
        return parts

    @staticmethod
    def update_stream_stats(stats: Dict[str, Any], values: pd.Series) -> None:
        arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if arr.size == 0:
            return
        stats["count"] = int(stats.get("count", 0) + arr.size)
        stats["sum"] = float(stats.get("sum", 0.0) + float(np.sum(arr)))
        stats["min"] = float(np.min(arr)) if "min" not in stats else float(min(stats["min"], float(np.min(arr))))
        stats["max"] = float(np.max(arr)) if "max" not in stats else float(max(stats["max"], float(np.max(arr))))

    @staticmethod
    def finalize_stream_stats(stats: Dict[str, Any]) -> Dict[str, float]:
        count = int(stats.get("count", 0))
        if count == 0:
            return {"count": 0, "min": np.nan, "mean": np.nan, "max": np.nan}
        return {
            "count": count,
            "min": float(stats["min"]),
            "mean": float(stats["sum"] / count),
            "max": float(stats["max"]),
        }

    def timeseries_index(self, df: pd.DataFrame, row_start: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": [f"timeseries:{row_start + i}" for i in range(len(df))],
                "real_domain": df["real_domain"].to_numpy(),
                "real_data_class": df["real_data_class"].to_numpy(),
                "source_id": df["source_id"].to_numpy(),
                "source_url": df["source_url"].to_numpy(),
                "citation": df["citation"].to_numpy(),
                "cell_id": df["cell_id"].to_numpy(),
                "cycle_number": df["cycle_number"].to_numpy(),
                "primary_value": df["voltage_V"].to_numpy(),
                "primary_unit": "V",
                "provenance_path": df["provenance_path"].to_numpy(),
                "provenance_sha256": df["provenance_sha256"].to_numpy(),
            }
        )

    def write_timeseries_stream(
        self,
        index_dir: Path,
        index_part_start: int,
        rows_per_part: int = 250_000,
    ) -> tuple[List[Path], List[Path], int, Dict[str, Dict[str, float]]]:
        ts_dir = ensure_dir(self.out_dir / "real_timeseries_sample_parts")
        for old in ts_dir.glob("*.parquet"):
            old.unlink()
        ts_parts: List[Path] = []
        index_parts: List[Path] = []
        ts_part_idx = 0
        index_part_idx = index_part_start
        row_start = 0
        voltage_stats: Dict[str, Any] = {}
        current_stats: Dict[str, Any] = {}
        for path, real_domain, data_class in self.timeseries_sources():
            df = self.add_provenance(self.read_optional(path), path, real_domain, data_class)
            if df.empty:
                continue
            for start in range(0, len(df), rows_per_part):
                chunk = df.iloc[start : start + rows_per_part].copy()
                ts_part = ts_dir / f"part-{ts_part_idx:04d}.parquet"
                chunk.to_parquet(ts_part, index=False)
                ts_parts.append(ts_part)
                idx = self.timeseries_index(chunk, row_start)
                index_part = index_dir / f"part-{index_part_idx:04d}.parquet"
                idx.to_parquet(index_part, index=False)
                index_parts.append(index_part)
                self.update_stream_stats(voltage_stats, chunk["voltage_V"])
                if "current_A" in chunk:
                    self.update_stream_stats(current_stats, chunk["current_A"])
                row_start += len(chunk)
                ts_part_idx += 1
                index_part_idx += 1
        return (
            ts_parts,
            index_parts,
            row_start,
            {
                "voltage_V": self.finalize_stream_stats(voltage_stats),
                "current_A": self.finalize_stream_stats(current_stats),
            },
        )

    def manifest(
        self,
        cycle_df: pd.DataFrame,
        ts_df: pd.DataFrame,
        lit_df: pd.DataFrame,
        impedance_df: pd.DataFrame,
        index_df: pd.DataFrame,
        timeseries_parts: List[Path] | None = None,
        index_parts: List[Path] | None = None,
        timeseries_rows_override: int | None = None,
        index_rows_override: int | None = None,
        timeseries_metrics: Dict[str, Dict[str, float]] | None = None,
    ) -> Dict[str, Any]:
        accepted_lit = lit_df[lit_df.get("curation_status", "") == "accepted"] if not lit_df.empty else pd.DataFrame()
        timeseries_path = self.out_dir / "real_timeseries_sample.parquet"
        timeseries_parts = timeseries_parts or []
        index_path = self.out_dir / "real_master_index.parquet"
        index_parts = index_parts or []
        timeseries_rows = int(timeseries_rows_override) if timeseries_rows_override is not None else int(len(ts_df))
        index_rows = int(index_rows_override) if index_rows_override is not None else int(len(index_df))
        metrics = {
            "cycle_rows": int(len(cycle_df)),
            "timeseries_rows": timeseries_rows,
            "impedance_rows": int(len(impedance_df)),
            "literature_rows_total": int(len(lit_df)),
            "literature_rows_accepted": int(len(accepted_lit)),
            "real_index_rows": index_rows,
            "total_real_rows": int(len(cycle_df) + timeseries_rows + len(impedance_df) + len(accepted_lit)),
            "cell_count": int(cycle_df["cell_id"].nunique()) if not cycle_df.empty else 0,
            "source_file_count": int(cycle_df["source_file"].nunique()) if not cycle_df.empty and "source_file" in cycle_df else 0,
            "dataset_key_count": int(cycle_df["dataset_key"].nunique()) if not cycle_df.empty and "dataset_key" in cycle_df else 0,
            "dataset_keys": sorted(cycle_df["dataset_key"].dropna().unique().tolist()) if not cycle_df.empty and "dataset_key" in cycle_df else [],
            "source_ids": sorted(cycle_df["source_id"].dropna().unique().tolist()) if not cycle_df.empty and "source_id" in cycle_df else [],
        }
        if not cycle_df.empty:
            metrics["normalized_discharge_capacity"] = describe_array(cycle_df["normalized_discharge_capacity"].dropna().to_numpy())
            metrics["cycle_life_label"] = describe_array(cycle_df["cycle_life_label"].dropna().to_numpy())
        if not ts_df.empty:
            metrics["voltage_V"] = describe_array(ts_df["voltage_V"].dropna().to_numpy())
            metrics["current_A"] = describe_array(ts_df["current_A"].dropna().to_numpy())
        elif timeseries_metrics:
            metrics.update(timeseries_metrics)
        if not accepted_lit.empty:
            metrics["literature_confidence"] = describe_array(accepted_lit["confidence"].dropna().to_numpy())
        if not impedance_df.empty:
            metrics["Re_ohm"] = describe_array(impedance_df["Re_ohm"].dropna().to_numpy()) if "Re_ohm" in impedance_df else {}
            metrics["Rct_ohm"] = describe_array(impedance_df["Rct_ohm"].dropna().to_numpy()) if "Rct_ohm" in impedance_df else {}
        status = "pass" if metrics["total_real_rows"] >= 100000 and metrics["literature_rows_accepted"] >= 10 else "needs_more_data"
        return {
            "created_at": utc_now(),
            "status": status,
            "quality_floor": {
                "minimum_real_rows": 100000,
                "minimum_accepted_literature_rows": 10,
                "provenance_required": True,
            },
            "metrics": metrics,
            "paths": {
                "cycle_summary": str(self.out_dir / "real_cycle_summary.parquet"),
                "timeseries_sample": str(timeseries_path) if not timeseries_parts else "",
                "timeseries_sample_parts_dir": str(self.out_dir / "real_timeseries_sample_parts") if timeseries_parts else "",
                "timeseries_sample_parts": [str(path) for path in timeseries_parts],
                "impedance_summary": str(self.out_dir / "real_impedance_summary.parquet"),
                "literature_measurements": str(self.out_dir / "real_literature_measurements.parquet"),
                "real_master_index": str(index_path) if not index_parts else "",
                "real_master_index_parts_dir": str(self.out_dir / "real_master_index_parts") if index_parts else "",
                "real_master_index_parts": [str(path) for path in index_parts],
            },
        }

    def run(self) -> Dict[str, Any]:
        cycle_df = self.battery_life_cycle_summary()
        ts_df = pd.DataFrame()
        lit_df = self.literature_measurements()
        impedance_df = self.impedance_measurements()
        base_index_df = self.row_index(cycle_df, ts_df, lit_df, impedance_df)
        index_df = pd.DataFrame()
        cycle_path = self.out_dir / "real_cycle_summary.parquet"
        ts_path = self.out_dir / "real_timeseries_sample.parquet"
        impedance_path = self.out_dir / "real_impedance_summary.parquet"
        lit_path = self.out_dir / "real_literature_measurements.parquet"
        index_path = self.out_dir / "real_master_index.parquet"
        ts_parts: List[Path] = []
        index_parts: List[Path] = []
        index_dir = ensure_dir(self.out_dir / "real_master_index_parts")
        for old in index_dir.glob("*.parquet"):
            old.unlink()
        if not cycle_df.empty:
            cycle_df.to_parquet(cycle_path, index=False)
        if not impedance_df.empty:
            impedance_df.to_parquet(impedance_path, index=False)
        if not lit_df.empty:
            lit_df.to_parquet(lit_path, index=False)
        if not base_index_df.empty:
            index_parts.extend(self.write_partitioned_parquet(base_index_df, index_dir, clean=False))
        ts_index_start = len(index_parts)
        ts_parts, ts_index_parts, ts_rows, ts_metrics = self.write_timeseries_stream(index_dir, ts_index_start)
        index_parts.extend(ts_index_parts)
        index_rows = int(len(base_index_df) + ts_rows)
        manifest = self.manifest(
            cycle_df,
            ts_df,
            lit_df,
            impedance_df,
            index_df,
            timeseries_parts=ts_parts,
            index_parts=index_parts,
            timeseries_rows_override=ts_rows,
            index_rows_override=index_rows,
            timeseries_metrics=ts_metrics,
        )
        for key, value in list(manifest["paths"].items()):
            if isinstance(value, str) and value:
                path = Path(value)
                if path.exists() and path.is_file():
                    manifest["paths"][f"{key}_sha256"] = sha256_file(path)
        write_json(self.out_dir / "real_dataset_manifest.json", manifest)
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble normalized real KineticsForge data into a provenance-rich master index.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = RealDatasetAssembler(Path(args.root).resolve()).run()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
