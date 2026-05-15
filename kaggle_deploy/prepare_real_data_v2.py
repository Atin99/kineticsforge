"""
KineticsForge V4 — Full Real Data Pipeline
Uses ALL 16.6M rows: timeseries + cycle summaries + impedance + ISU-ILCC
Extracts per-cycle waveform features from voltage/current/temp timeseries
"""
import pandas as pd
import numpy as np
import os, json, shutil, glob
from pathlib import Path

ROOT = Path(r"c:\project 5\kineticsforge_v2_work\data\real")
NORM = ROOT / "normalized"
ASSEMBLED = ROOT / "assembled"
OUT = Path(r"c:\project 5\kaggle_deploy\phase2_data_v2")
os.makedirs(OUT, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# STEP 1: Load ALL cycle summaries (185K rows, 547 cells)
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading cycle summaries...")
cycle_df = pd.read_parquet(ASSEMBLED / "real_cycle_summary.parquet")
print(f"  cycle_summary: {len(cycle_df):,} rows, {cycle_df['cell_id'].nunique()} cells")

# ──────────────────────────────────────────────────────────────
# STEP 2: Load ALL timeseries data (~12M rows) and extract
#         per-cycle waveform features
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Loading ALL timeseries and extracting per-cycle features...")

ts_files = sorted(glob.glob(str(NORM / "*timeseries*.parquet")))
print(f"  found {len(ts_files)} timeseries parquet files")

all_ts_features = []
total_ts_rows = 0

for tf in ts_files:
    fname = os.path.basename(tf)
    print(f"  processing {fname}...", end=" ")
    ts = pd.read_parquet(tf)
    total_ts_rows += len(ts)
    
    # Group by cell_id + cycle_number to extract waveform features
    needed_cols = ["cell_id", "cycle_number"]
    feat_cols = {}
    for c in ["voltage_V", "current_A", "temperature_C", "normalized_discharge_capacity",
              "charge_capacity_Ah", "discharge_capacity_Ah", "internal_resistance_ohm"]:
        if c in ts.columns:
            feat_cols[c] = c
    
    if not feat_cols or "cell_id" not in ts.columns:
        print(f"skipped (missing columns)")
        continue
    
    groups = ts.groupby(["cell_id", "cycle_number"], observed=True)
    
    features = []
    for (cid, cyc), grp in groups:
        row = {"cell_id": cid, "cycle_number": cyc, "ts_points": len(grp)}
        
        if "voltage_V" in grp.columns:
            v = grp["voltage_V"].dropna()
            if len(v) > 0:
                row["v_mean"] = float(v.mean())
                row["v_std"] = float(v.std()) if len(v) > 1 else 0.0
                row["v_min"] = float(v.min())
                row["v_max"] = float(v.max())
                row["v_range"] = row["v_max"] - row["v_min"]
                # Voltage curve shape features
                if len(v) >= 4:
                    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
                    row["v_iqr"] = q3 - q1
                    row["v_skew"] = float(v.skew()) if len(v) > 2 else 0.0
                    # Slope of voltage curve (linear fit)
                    x = np.arange(len(v), dtype=np.float32)
                    if len(v) > 1:
                        row["v_slope"] = float(np.polyfit(x, v.values, 1)[0])
                    
        if "current_A" in grp.columns:
            i_col = grp["current_A"].dropna()
            if len(i_col) > 0:
                row["i_mean"] = float(i_col.mean())
                row["i_std"] = float(i_col.std()) if len(i_col) > 1 else 0.0
                row["i_abs_mean"] = float(i_col.abs().mean())
                row["i_max"] = float(i_col.max())
                row["i_min"] = float(i_col.min())
        
        if "temperature_C" in grp.columns:
            t_col = grp["temperature_C"].dropna()
            if len(t_col) > 0:
                row["temp_mean"] = float(t_col.mean())
                row["temp_max"] = float(t_col.max())
                row["temp_range"] = float(t_col.max() - t_col.min())
        
        if "internal_resistance_ohm" in grp.columns:
            r_col = grp["internal_resistance_ohm"].dropna()
            if len(r_col) > 0:
                row["ir_mean"] = float(r_col.mean())
        
        if "normalized_discharge_capacity" in grp.columns:
            nd = grp["normalized_discharge_capacity"].dropna()
            if len(nd) > 0:
                row["ndc_mean"] = float(nd.mean())
                row["ndc_end"] = float(nd.iloc[-1])
        
        features.append(row)
    
    feat_df = pd.DataFrame(features)
    all_ts_features.append(feat_df)
    print(f"{len(feat_df):,} cycle-features extracted")

ts_feat_df = pd.concat(all_ts_features, ignore_index=True) if all_ts_features else pd.DataFrame()
print(f"\n  TOTAL timeseries processed: {total_ts_rows:,} rows -> {len(ts_feat_df):,} cycle-features")

# ──────────────────────────────────────────────────────────────
# STEP 3: Load ISU-ILCC Q-interpolated data (523K rows)
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Loading ISU-ILCC Q-interpolated timeseries...")
isu_path = NORM / "isu_ilcc_q_interpolated_timeseries.parquet"
if isu_path.exists():
    isu = pd.read_parquet(isu_path)
    print(f"  ISU-ILCC: {len(isu):,} rows, {isu['cell_id'].nunique()} cells")
    isu_groups = isu.groupby(["cell_id", "cycle_number"], observed=True)
    isu_feats = []
    for (cid, cyc), grp in isu_groups:
        row = {"cell_id": cid, "cycle_number": cyc, "ts_points": len(grp)}
        if "voltage_V" in grp.columns:
            v = grp["voltage_V"].dropna()
            if len(v) > 0:
                row["v_mean"] = float(v.mean())
                row["v_std"] = float(v.std()) if len(v) > 1 else 0.0
                row["v_min"] = float(v.min())
                row["v_max"] = float(v.max())
                row["v_range"] = row["v_max"] - row["v_min"]
        if "interpolated_discharge_q_Ah" in grp.columns:
            q = grp["interpolated_discharge_q_Ah"].dropna()
            if len(q) > 0:
                row["q_mean"] = float(q.mean())
                row["q_max"] = float(q.max())
        if "normalized_discharge_capacity" in grp.columns:
            nd = grp["normalized_discharge_capacity"].dropna()
            if len(nd) > 0:
                row["ndc_mean"] = float(nd.mean())
                row["ndc_end"] = float(nd.iloc[-1])
        isu_feats.append(row)
    isu_feat_df = pd.DataFrame(isu_feats)
    ts_feat_df = pd.concat([ts_feat_df, isu_feat_df], ignore_index=True)
    print(f"  added {len(isu_feat_df):,} ISU-ILCC cycle-features -> total {len(ts_feat_df):,}")

# ──────────────────────────────────────────────────────────────
# STEP 4: Load impedance data (1,956 rows)
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Loading impedance data...")
imp_path = NORM / "nasa_pcoe_impedance_summary.parquet"
imp_df = pd.read_parquet(imp_path) if imp_path.exists() else pd.DataFrame()
if not imp_df.empty:
    print(f"  impedance: {len(imp_df):,} rows, {imp_df['cell_id'].nunique()} cells")
    imp_feats = imp_df.groupby("cell_id").agg(
        re_mean=("Re_ohm", "mean"), re_std=("Re_ohm", "std"),
        re_trend=("Re_ohm", lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x) > 1 else 0),
        rct_mean=("Rct_ohm", "mean"), rct_std=("Rct_ohm", "std"),
        n_imp=("Re_ohm", "count")
    ).reset_index()
    print(f"  impedance cell features: {len(imp_feats)} cells")

# ──────────────────────────────────────────────────────────────
# STEP 5: Merge everything into rich per-cell datasets
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Merging all data sources...")

# Merge timeseries features with cycle summaries
keep_cycle = ["cell_id", "dataset_key", "cycle_number", "normalized_discharge_capacity",
    "operation_temperature_C", "charge_rate_C", "discharge_rate_C",
    "cathode_material", "anode_material", "nominal_capacity_Ah",
    "coulombic_efficiency", "voltage_min_V", "voltage_max_V",
    "internal_resistance_mean_ohm", "cycle_life_label"]
cycle_slim = cycle_df[[c for c in keep_cycle if c in cycle_df.columns]].copy()

# Join timeseries features to cycle data
if not ts_feat_df.empty:
    merged = cycle_slim.merge(ts_feat_df, on=["cell_id", "cycle_number"], how="left")
    print(f"  merged cycle+timeseries: {len(merged):,} rows")
else:
    merged = cycle_slim
    print(f"  no timeseries features to merge")

# Add impedance features per cell
if not imp_df.empty:
    merged = merged.merge(imp_feats, on="cell_id", how="left")
    print(f"  added impedance features: {merged['re_mean'].notna().sum():,} rows have impedance")

# Encode cathode materials
cat_map = {}
if "cathode_material" in merged.columns:
    for i, cat in enumerate(merged["cathode_material"].dropna().unique()):
        cat_map[cat] = i
    merged["cathode_id"] = merged["cathode_material"].map(cat_map).fillna(-1).astype(int)

print(f"  final merged dataset: {len(merged):,} rows, {merged.columns.tolist()}")

# ──────────────────────────────────────────────────────────────
# STEP 6: Build per-cell NPZ files with ALL features
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: Building per-cell NPZ files...")

FEATURE_COLS = ["v_mean","v_std","v_min","v_max","v_range","v_iqr","v_skew","v_slope",
    "i_mean","i_std","i_abs_mean","i_max","i_min",
    "temp_mean","temp_max","temp_range","ir_mean","ndc_mean","ndc_end",
    "coulombic_efficiency","voltage_min_V","voltage_max_V","internal_resistance_mean_ohm",
    "re_mean","re_std","re_trend","rct_mean","rct_std"]
FEATURE_COLS = [c for c in FEATURE_COLS if c in merged.columns]
print(f"  using {len(FEATURE_COLS)} waveform features per cycle")

MAX_SEQ = 600
cells = merged.groupby("cell_id")
cell_list = []
for cid, grp in cells:
    grp = grp.sort_values("cycle_number")
    cap = grp["normalized_discharge_capacity"].values.astype(np.float32)
    cyc = grp["cycle_number"].values.astype(np.float32)
    if len(cap) < 5:
        continue
    
    # Rich per-cycle features
    feats = grp[FEATURE_COLS].fillna(0).values.astype(np.float32)
    
    # Operating conditions
    temp = grp["operation_temperature_C"].median()
    temp = float(temp) if pd.notna(temp) else 25.0
    cr = float(grp["charge_rate_C"].median()) if "charge_rate_C" in grp and grp["charge_rate_C"].notna().any() else 1.0
    dr = float(grp["discharge_rate_C"].median()) if "discharge_rate_C" in grp and grp["discharge_rate_C"].notna().any() else 1.0
    cat_id = int(grp["cathode_id"].iloc[0]) if "cathode_id" in grp.columns else -1
    nom = float(grp["nominal_capacity_Ah"].median()) if "nominal_capacity_Ah" in grp and grp["nominal_capacity_Ah"].notna().any() else 1.0
    ds = grp["dataset_key"].iloc[0] if "dataset_key" in grp.columns else "unknown"
    cl = float(grp["cycle_life_label"].iloc[0]) if "cycle_life_label" in grp.columns and grp["cycle_life_label"].notna().any() else -1
    
    cell_list.append({
        "cell_id": cid, "dataset": ds, "n_cycles": len(cap),
        "capacity": cap[:MAX_SEQ], "cycles": cyc[:MAX_SEQ], "features": feats[:MAX_SEQ],
        "temp_C": temp, "charge_rate": cr, "discharge_rate": dr,
        "cathode_id": cat_id, "nominal_Ah": nom, "cycle_life": cl,
    })

print(f"  valid cells: {len(cell_list)}")

# ──────────────────────────────────────────────────────────────
# STEP 7: Train/Val/Test split
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7: Splitting data...")
np.random.seed(42)
test_ds = ["CALCE", "NASA_PCoE"]
train_cells, test_cells = [], []
for c in cell_list:
    (test_cells if c["dataset"] in test_ds else train_cells).append(c)
np.random.shuffle(train_cells)
n_val = max(1, len(train_cells) // 8)
val_cells = train_cells[:n_val]
train_cells = train_cells[n_val:]
print(f"  train={len(train_cells)} val={len(val_cells)} test={len(test_cells)}")

# ──────────────────────────────────────────────────────────────
# STEP 8: Save NPZ files
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 8: Saving NPZ files...")

n_feat = len(FEATURE_COLS)

def pack_cells(cells, prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate(cells):
        cap = c["capacity"]
        cyc = c["cycles"]
        feats = c["features"]
        cond = np.array([c["temp_C"], c["charge_rate"], c["discharge_rate"],
            float(c["cathode_id"]), c["nominal_Ah"]], dtype=np.float32)
        np.savez_compressed(os.path.join(out_dir, f"{prefix}_{i:04d}.npz"),
            capacity=cap, cycles=cyc, features=feats, conditions=cond,
            cell_id=str(c["cell_id"]), dataset=str(c["dataset"]),
            cycle_life=c["cycle_life"])
    return len(cells)

for sub in ["train","val","test"]:
    d = OUT / sub
    if d.exists(): shutil.rmtree(d)

n_tr = pack_cells(train_cells, "train", OUT / "train")
n_va = pack_cells(val_cells, "val", OUT / "val")
n_te = pack_cells(test_cells, "test", OUT / "test")

# Also save impedance data separately
if not imp_df.empty:
    imp_cols = [c for c in ["cell_id","cycle_number","Re_ohm","Rct_ohm",
        "operation_temperature_C","battery_impedance_abs_mean_ohm"] if c in imp_df.columns]
    imp_df[imp_cols].to_parquet(OUT / "impedance.parquet", index=False)

# Save timeseries features for direct use
ts_feat_df.to_parquet(OUT / "timeseries_features.parquet", index=False)
print(f"  saved timeseries_features.parquet: {len(ts_feat_df):,} rows")

# ──────────────────────────────────────────────────────────────
# STEP 9: Create metadata
# ──────────────────────────────────────────────────────────────
meta = {
    "version": "v2_full_pipeline",
    "train_cells": n_tr, "val_cells": n_va, "test_cells": n_te,
    "max_seq": MAX_SEQ, "cond_dim": 5, "n_features": n_feat,
    "feature_names": FEATURE_COLS,
    "cond_names": ["temp_C","charge_rate","discharge_rate","cathode_id","nominal_Ah"],
    "cathode_map": cat_map, "test_datasets": test_ds,
    "total_cycle_rows": int(sum(c["n_cycles"] for c in cell_list)),
    "total_timeseries_rows_processed": total_ts_rows,
    "total_ts_cycle_features": len(ts_feat_df),
    "impedance_rows": len(imp_df) if not imp_df.empty else 0,
    "data_sources": {
        "cycle_summaries": "185K rows from 11 datasets",
        "timeseries": f"{total_ts_rows:,} rows → {len(ts_feat_df):,} per-cycle features",
        "impedance": f"{len(imp_df):,} EIS measurements" if not imp_df.empty else "none",
        "isu_ilcc": "523K Q-interpolated voltage curves",
    }
}
with open(OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ──────────────────────────────────────────────────────────────
# STEP 10: Create Kaggle ZIPs
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 10: Creating Kaggle ZIPs...")

ZIP_BASE = Path(r"c:\project 5\kaggle_deploy")
for acct in ["acct1", "acct2", "acct3"]:
    zname = f"kineticsforge-realdata-{acct}"
    zdir = ZIP_BASE / zname
    if zdir.exists(): shutil.rmtree(zdir)
    os.makedirs(zdir)
    shutil.copy(OUT / "meta.json", zdir / "meta.json")
    for sub in ["train", "val", "test"]:
        shutil.copytree(OUT / sub, zdir / sub)
    if (OUT / "impedance.parquet").exists():
        shutil.copy(OUT / "impedance.parquet", zdir / "impedance.parquet")
    if (OUT / "timeseries_features.parquet").exists():
        shutil.copy(OUT / "timeseries_features.parquet", zdir / "timeseries_features.parquet")
    
    shutil.make_archive(str(ZIP_BASE / zname), "zip", zdir)
    sz = os.path.getsize(str(ZIP_BASE / zname) + ".zip")
    print(f"  {zname}.zip: {sz/1024/1024:.1f} MB")

print("=" * 60)
print(json.dumps(meta, indent=2))
print("=" * 60)
print("DONE! Upload the 3 ZIPs to Kaggle.")
