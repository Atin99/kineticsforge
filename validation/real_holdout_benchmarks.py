import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def stable_cell_split(cell_ids: np.ndarray, val_fraction: float = 0.20) -> Tuple[set, set]:
    unique = sorted(str(c) for c in pd.Series(cell_ids).dropna().unique())
    holdout = set()
    train = set()
    threshold = int(10000 * val_fraction)
    for cell_id in unique:
        digest = hashlib.sha256(cell_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
        if bucket < threshold:
            holdout.add(cell_id)
        else:
            train.add(cell_id)
    if not holdout and unique:
        holdout.add(unique[-1])
        train.discard(unique[-1])
    if not train and unique:
        train.add(unique[0])
        holdout.discard(unique[0])
    return train, holdout


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    cycle = pd.to_numeric(df.get("cycle_number"), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["cycle"] = cycle
    out["sqrt_cycle"] = np.sqrt(cycle + 1.0)
    out["log_cycle"] = np.log1p(cycle)
    for col, default in [
        ("operation_temperature_C", 25.0),
        ("charge_rate_C", 0.5),
        ("discharge_rate_C", 0.5),
        ("nominal_capacity_Ah", 1.0),
        ("max_voltage_limit_V", 4.0),
        ("min_voltage_limit_V", 2.0),
    ]:
        values = pd.to_numeric(df.get(col), errors="coerce")
        out[col] = values.fillna(float(values.median()) if values.notna().any() else default)
    out["voltage_window_V"] = out["max_voltage_limit_V"] - out["min_voltage_limit_V"]
    out["abs_current_proxy"] = out["nominal_capacity_Ah"] * (out["charge_rate_C"].abs() + out["discharge_rate_C"].abs()) / 2.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(x_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def ridge_predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    return x_aug @ coef


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"mae_fraction": math.nan, "rmse_fraction": math.nan, "mape_percent": math.nan, "r2": math.nan, "n": 0.0}
    residual = y_true - y_pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae_fraction": float(np.mean(np.abs(residual))),
        "rmse_fraction": float(np.sqrt(np.mean(residual ** 2))),
        "mape_percent": float(np.mean(np.abs(residual) / np.maximum(np.abs(y_true), 1e-9)) * 100.0),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "n": float(y_true.size),
    }


def prediction_interval_report(train_residual: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    train_residual = train_residual[np.isfinite(train_residual)]
    if train_residual.size < 20:
        return {"coverage_90": math.nan, "mean_width_fraction": math.nan}
    lo, hi = np.quantile(train_residual, [0.05, 0.95])
    lower = y_pred + lo
    upper = y_pred + hi
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(mask):
        return {"coverage_90": math.nan, "mean_width_fraction": math.nan}
    return {
        "coverage_90": float(np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask]))),
        "mean_width_fraction": float(np.mean(upper[mask] - lower[mask])),
    }


def run_real_holdout(project_root: Path) -> Dict[str, Any]:
    path = project_root / "data" / "real" / "assembled" / "real_cycle_summary.parquet"
    if not path.exists():
        return {"status": "missing_data", "reason": str(path)}
    df = pd.read_parquet(path)
    required = {"cell_id", "cycle_number", "normalized_discharge_capacity"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {"status": "invalid_schema", "missing_columns": missing}
    df = df.copy()
    df["cell_id"] = df["cell_id"].astype(str)
    y = pd.to_numeric(df["normalized_discharge_capacity"], errors="coerce")
    valid = y.notna() & y.between(0.0, 1.2)
    df = df.loc[valid].reset_index(drop=True)
    train_cells, holdout_cells = stable_cell_split(df["cell_id"].to_numpy(), val_fraction=0.20)
    train_mask = df["cell_id"].isin(train_cells).to_numpy()
    holdout_mask = df["cell_id"].isin(holdout_cells).to_numpy()
    train_df = df.loc[train_mask].reset_index(drop=True)
    holdout_df = df.loc[holdout_mask].reset_index(drop=True)
    if len(train_df) < 100 or len(holdout_df) < 50:
        return {
            "status": "needs_more_real_split_data",
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "train_cells": int(len(train_cells)),
            "holdout_cells": int(len(holdout_cells)),
        }
    train_features = feature_frame(train_df)
    holdout_features = feature_frame(holdout_df)
    x_train_raw = train_features.to_numpy(dtype=float)
    x_holdout_raw = holdout_features.to_numpy(dtype=float)
    x_train, x_holdout, _, _ = standardize(x_train_raw, x_holdout_raw)
    y_train = train_df["normalized_discharge_capacity"].to_numpy(dtype=float)
    y_holdout = holdout_df["normalized_discharge_capacity"].to_numpy(dtype=float)
    coef = ridge_fit(x_train, y_train, alpha=2e-2)
    train_pred = np.clip(ridge_predict(x_train, coef), 0.0, 1.2)
    holdout_pred = np.clip(ridge_predict(x_holdout, coef), 0.0, 1.2)
    holdout_metrics = metrics(y_holdout, holdout_pred)
    train_metrics = metrics(y_train, train_pred)
    interval = prediction_interval_report(y_train - train_pred, y_holdout, holdout_pred)
    quality = "real_holdout_baseline_pass"
    if not math.isfinite(holdout_metrics["mae_fraction"]):
        quality = "invalid_metrics"
    elif holdout_metrics["mae_fraction"] > 0.08 or interval.get("coverage_90", 0.0) < 0.75:
        quality = "needs_model_improvement"
    return {
        "status": "pass" if quality != "invalid_metrics" else "fail",
        "quality": quality,
        "target": "normalized_discharge_capacity",
        "split": "whole_cell_hash_holdout",
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_cells": int(len(train_cells)),
        "holdout_cells": int(len(holdout_cells)),
        "features": list(train_features.columns),
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "prediction_interval": interval,
        "data_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real whole-cell holdout benchmarks on normalized public battery data.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="data/cache/real_holdout_benchmark.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    payload = run_real_holdout(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload.get("status"), "quality": payload.get("quality", ""), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
