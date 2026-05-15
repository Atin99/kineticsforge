import pandas as pd
import numpy as np
import os
import json
import hashlib
from pathlib import Path

ROOT = Path(r"c:\project 5\kineticsforge_v2_work\data\real\assembled")
OUT = Path(r"c:\project 5\kaggle_deploy\phase2_data")
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(ROOT / "real_cycle_summary.parquet")
print(f"loaded {len(df)} rows, {df['cell_id'].nunique()} cells")

keep = ["cell_id","dataset_key","cycle_number","normalized_discharge_capacity",
    "operation_temperature_C","charge_rate_C","discharge_rate_C",
    "cathode_material","anode_material","nominal_capacity_Ah",
    "coulombic_efficiency","voltage_min_V","voltage_max_V",
    "internal_resistance_mean_ohm","cycle_life_label"]
df = df[[c for c in keep if c in df.columns]].copy()
df = df.dropna(subset=["normalized_discharge_capacity","cycle_number","cell_id"])
df = df.sort_values(["cell_id","cycle_number"]).reset_index(drop=True)

cat_map = {}
for i, cat in enumerate(df["cathode_material"].dropna().unique()):
    cat_map[cat] = i
df["cathode_id"] = df["cathode_material"].map(cat_map).fillna(-1).astype(int)

cells = df.groupby("cell_id")
cell_list = []
for cid, grp in cells:
    grp = grp.sort_values("cycle_number")
    cap = grp["normalized_discharge_capacity"].values.astype(np.float32)
    cyc = grp["cycle_number"].values.astype(np.float32)
    if len(cap) < 5:
        continue
    temp = grp["operation_temperature_C"].median()
    temp = float(temp) if pd.notna(temp) else 25.0
    cr = grp["charge_rate_C"].median()
    cr = float(cr) if pd.notna(cr) else 1.0
    dr = grp["discharge_rate_C"].median()
    dr = float(dr) if pd.notna(dr) else 1.0
    cat_id = int(grp["cathode_id"].iloc[0])
    nom = grp["nominal_capacity_Ah"].median()
    nom = float(nom) if pd.notna(nom) else 1.0
    ds = grp["dataset_key"].iloc[0]
    cell_list.append({
        "cell_id": cid, "dataset": ds, "n_cycles": len(cap),
        "capacity": cap, "cycles": cyc,
        "temp_C": temp, "charge_rate": cr, "discharge_rate": dr,
        "cathode_id": cat_id, "nominal_Ah": nom,
    })

print(f"valid cells: {len(cell_list)}")

np.random.seed(42)
datasets = list(set(c["dataset"] for c in cell_list))
test_ds = ["CALCE","NASA_PCoE"]
val_cells = []
train_cells = []
test_cells = []
for c in cell_list:
    if c["dataset"] in test_ds:
        test_cells.append(c)
    else:
        train_cells.append(c)

np.random.shuffle(train_cells)
n_val = max(1, len(train_cells) // 8)
val_cells = train_cells[:n_val]
train_cells = train_cells[n_val:]

print(f"train={len(train_cells)} val={len(val_cells)} test={len(test_cells)}")
print(f"test datasets: {test_ds}")

MAX_SEQ = 600

def pack_cells(cells, prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate(cells):
        cap = c["capacity"][:MAX_SEQ]
        cyc = c["cycles"][:MAX_SEQ]
        cond = np.array([c["temp_C"], c["charge_rate"], c["discharge_rate"],
            float(c["cathode_id"]), c["nominal_Ah"]], dtype=np.float32)
        np.savez_compressed(os.path.join(out_dir, f"{prefix}_{i:04d}.npz"),
            capacity=cap, cycles=cyc, conditions=cond,
            cell_id=c["cell_id"], dataset=c["dataset"])
    return len(cells)

train_dir = OUT / "train"
val_dir = OUT / "val"
test_dir = OUT / "test"
n_tr = pack_cells(train_cells, "train", train_dir)
n_va = pack_cells(val_cells, "val", val_dir)
n_te = pack_cells(test_cells, "test", test_dir)

meta = {
    "train_cells": n_tr, "val_cells": n_va, "test_cells": n_te,
    "max_seq": MAX_SEQ, "cond_dim": 5,
    "cond_names": ["temp_C","charge_rate","discharge_rate","cathode_id","nominal_Ah"],
    "cathode_map": cat_map, "test_datasets": test_ds,
    "total_cycle_rows": int(sum(c["n_cycles"] for c in cell_list)),
}
with open(OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))

import shutil
zip1 = Path(r"c:\project 5\kaggle_deploy\kineticsforge-realdata-acct1")
zip2 = Path(r"c:\project 5\kaggle_deploy\kineticsforge-realdata-acct2")
zip3 = Path(r"c:\project 5\kaggle_deploy\kineticsforge-realdata-acct3")

for zd in [zip1, zip2, zip3]:
    if zd.exists(): shutil.rmtree(zd)
    os.makedirs(zd)
    shutil.copy(OUT / "meta.json", zd / "meta.json")
    for sub in ["train","val","test"]:
        dst = zd / sub
        shutil.copytree(OUT / sub, dst)

imp_path = ROOT / "real_impedance_summary.parquet"
if imp_path.exists():
    imp = pd.read_parquet(imp_path)
    imp_cols = [c for c in ["cell_id","cycle_number","Re_ohm","Rct_ohm",
        "source_id","operation_temperature_C"] if c in imp.columns]
    imp[imp_cols].to_parquet(zip2 / "impedance.parquet", index=False)
    imp[imp_cols].to_parquet(zip3 / "impedance.parquet", index=False)
    print(f"impedance: {len(imp)} rows")

print("creating ZIPs...")
for name, src in [("kineticsforge-realdata-acct1", zip1),
                  ("kineticsforge-realdata-acct2", zip2),
                  ("kineticsforge-realdata-acct3", zip3)]:
    shutil.make_archive(str(Path(r"c:\project 5\kaggle_deploy") / name), "zip", src)
    sz = os.path.getsize(str(Path(r"c:\project 5\kaggle_deploy") / name) + ".zip")
    print(f"  {name}.zip: {sz/1024:.0f} KB")

print("DONE. Upload these 3 ZIPs to Kaggle as datasets.")
