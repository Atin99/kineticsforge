"""KineticsForge V5 — Account 1: M11 ElectrolyteHealth + M12 Replenishability
Uses REAL data. Upload kf-m11m12-data.zip as Kaggle dataset, paste this, GPU on, Run All.
"""
import os, sys, time, gc, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

def log(msg): print(msg, flush=True)

log("[BOOT] Starting KF-V5 Account 1...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("/kaggle/working/checkpoints"); OUT.mkdir(exist_ok=True)
# Delete stale checkpoints from previous failed runs
for old in OUT.glob("*.pt"):
    log(f"  Deleting stale checkpoint: {old.name}"); old.unlink()
EPOCHS = 200; BS = 128; LR = 3e-4

# ── AUTO-FIND DATA (recursive scan) ──
def find_data():
    base = "/kaggle/input"
    log(f"Scanning {base}/ for train/*.npz ...")
    for r, dirs, files in os.walk(base):
        if os.path.basename(r) == "train":
            npzs = [f for f in files if f.endswith(".npz")]
            if npzs:
                parent = os.path.dirname(r)
                log(f"FOUND: {parent} ({len(npzs)} train files)")
                return Path(parent)
    raise FileNotFoundError("No train/*.npz found! Upload the data zip as Kaggle dataset.")

DATA = find_data()
meta = json.loads((DATA / "meta.json").read_text()) if (DATA / "meta.json").exists() else {}
log(f"[KF-V5] Device={DEVICE} Data={DATA}")

# ═══ M11: ElectrolyteHealth ═══
class ElectrolyteHealthModel(nn.Module):
    def __init__(self, in_dim=7, hd=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,hd),nn.GELU(),nn.LayerNorm(hd),nn.Linear(hd,hd),nn.GELU(),nn.Dropout(0.05),nn.Linear(hd,48))
        self.deg_head = nn.Linear(48, 1)
        self.plat_head = nn.Linear(48, 1)
        self.cr_head = nn.Sequential(nn.Linear(48,16),nn.GELU(),nn.Linear(16,1),nn.Softplus())
    def forward(self, x):
        z = self.enc(x)
        return self.deg_head(z).squeeze(-1), self.plat_head(z).squeeze(-1), torch.clamp(self.cr_head(z).squeeze(-1),0.05,5.0)

# ═══ M12: Replenishability ═══
class ReplenishabilityModel(nn.Module):
    def __init__(self, hist_dim=20, feat_dim=10, hd=64):
        super().__init__()
        self.he = nn.Sequential(nn.Linear(hist_dim,hd),nn.GELU(),nn.Linear(hd,32))
        self.fe = nn.Sequential(nn.Linear(feat_dim,hd),nn.GELU(),nn.LayerNorm(hd),nn.Linear(hd,32))
        self.head = nn.Sequential(nn.Linear(64,48),nn.GELU(),nn.Dropout(0.1),nn.Linear(48,2))
    def forward(self, hist, feats):
        o = self.head(torch.cat([self.he(hist), self.fe(feats)], dim=-1))
        return o[:, 0], o[:, 1]

# ═══ REAL DATA LOADING ═══
def load_cells(split_dir):
    cells = []
    for f in sorted(glob.glob(str(split_dir / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cells.append({"capacity": d["capacity"].astype(np.float32), "features": d["features"].astype(np.float32),
                       "conditions": d["conditions"].astype(np.float32), "cycle_life": float(d.get("cycle_life", -1))})
    return cells

def build_eis_dataset():
    imp_path = DATA / "impedance.parquet"
    if not imp_path.exists():
        log("  WARNING: No impedance.parquet"); return None, None
    imp = pd.read_parquet(imp_path)
    # Drop rows with NaN in critical columns
    for c in ["Re_ohm", "Rct_ohm"]:
        if c in imp.columns: imp = imp.dropna(subset=[c])
    log(f"  Loaded {len(imp)} real EIS measurements from {imp['cell_id'].nunique()} cells")
    samples = []
    for cid, grp in imp.groupby("cell_id"):
        grp = grp.sort_values("cycle_number"); n = len(grp)
        if n < 2: continue
        re_init = float(grp["Re_ohm"].iloc[0])
        for i, (_, row) in enumerate(grp.iterrows()):
            re = float(row["Re_ohm"]); rct = float(row["Rct_ohm"])
            if np.isnan(re) or np.isnan(rct): continue
            imp_abs = float(row.get("battery_impedance_abs_mean_ohm", 0)) if "battery_impedance_abs_mean_ohm" in row.index and pd.notna(row.get("battery_impedance_abs_mean_ohm")) else 0.0
            temp = float(row.get("operation_temperature_C", 25)) if pd.notna(row.get("operation_temperature_C")) else 25.0
            cyc_frac = float(row.get("cycle_number", i)) / max(float(grp["cycle_number"].max()), 1)
            feats = [max(0,re), max(0,rct), max(0,re-re_init), max(0,imp_abs), temp/50.0, 0.5, cyc_frac]
            deg = i / max(n-1, 1)
            plat = min(1.0, max(0.0, deg*0.5 + max(0, (25-temp))*0.02))
            safe_cr = max(0.1, 2.5 - 2.0*deg)
            samples.append((feats, [deg, plat, safe_cr]))
    if not samples: return None, None
    X = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    Y = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    # Z-score normalize features (prevents NaN from scale mismatch)
    mu = X.mean(dim=0, keepdim=True); std = X.std(dim=0, keepdim=True).clamp(min=1e-8)
    X = (X - mu) / std
    X = torch.nan_to_num(X, 0.0)  # safety net
    log(f"  Built {len(X)} EIS samples, feature ranges: min={X.min():.2f} max={X.max():.2f}")
    return X, Y

def build_replenish_dataset(cells, window=20, lookahead=20):
    """Build M12 data using REAL fade deceleration as target.
    Target: fade_ratio = fade_after / fade_before
    <1 means fade is slowing (recoverable), >1 means accelerating (not recoverable)
    We predict fade_ratio clamped to [0, 2] and normalized to [0, 1].
    """
    hists, feats_all, targets = [], [], []
    for cell in cells:
        cap = cell["capacity"]; cond = cell["conditions"]
        if len(cap) < window + lookahead + 5: continue
        for start in range(0, len(cap)-window-lookahead, max(1, len(cap)//15)):
            end = start + window
            if end + lookahead > len(cap): break
            hist = cap[start:end].copy()
            if hist[0] > 0: hist = hist / hist[0]
            fb = float(cap[start]-cap[end]) / window          # fade rate before
            fa = float(cap[end]-cap[end+lookahead]) / lookahead  # fade rate after
            if abs(fb) < 1e-7: continue  # skip flat regions
            # fade_ratio: <1 = decelerating (good), >1 = accelerating (bad)
            fade_ratio = fa / (fb + 1e-8)
            fade_ratio_norm = min(1.0, max(0.0, fade_ratio / 2.0))  # normalize [0,2] -> [0,1]
            # Capacity gain proxy: how much capacity was "recovered" relative to expected
            expected_end = cap[end] - fb * lookahead
            actual_end = cap[end + lookahead]
            gain = max(0, min(0.15, float(actual_end - expected_end)))
            soh = float(cap[end])
            curv = float(np.mean(np.diff(np.diff(hist)))) if len(hist)>=3 else 0.0
            feat = np.array([soh, fb, fa, curv, float(np.std(np.diff(hist))),
                             cond[0]/50, cond[1], cond[2], start/max(len(cap),1),
                             cell["cycle_life"]/2000 if cell["cycle_life"]>0 else 0.5], dtype=np.float32)
            feat = np.nan_to_num(feat, 0.0)
            hists.append(hist); feats_all.append(feat)
            targets.append([fade_ratio_norm, gain])
    log(f"  Built {len(hists)} replenishability samples")
    tgt = np.array(targets, dtype=np.float32)
    log(f"  fade_ratio_norm: mean={tgt[:,0].mean():.3f} std={tgt[:,0].std():.3f} min={tgt[:,0].min():.3f} max={tgt[:,0].max():.3f}")
    return (torch.tensor(np.array(hists),dtype=torch.float32), torch.tensor(np.array(feats_all),dtype=torch.float32),
            torch.tensor(tgt[:,0],dtype=torch.float32), torch.tensor(tgt[:,1],dtype=torch.float32))

# ═══ TRAINING ═══
def train_loop(name, model, tdl, vdl, lfn, epochs=EPOCHS):
    model=model.to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); sc=GradScaler("cuda"); best=1e9
    res=OUT/f"{name}_resume.pt"; s0=0
    if res.exists():
        ck=torch.load(res,map_location=DEVICE,weights_only=False); model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); s0=ck.get("epoch",0); best=ck.get("best",1e9)
    t0=time.time()
    for ep in range(s0,epochs):
        if time.time()-t0>6.5*3600: log(f"  ⏱ Time limit ep {ep}"); break
        model.train(); tl=0
        for b in tdl:
            b=[x.to(DEVICE) for x in b]; opt.zero_grad(set_to_none=True)
            with autocast("cuda"): l=lfn(model,b)
            if not torch.isfinite(l): continue
            sc.scale(l).backward(); sc.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            sc.step(opt); sc.update(); tl+=l.item()
        sch.step(); model.eval(); vl=0
        with torch.no_grad():
            for b in vdl:
                b=[x.to(DEVICE) for x in b]
                with autocast("cuda"): vl+=lfn(model,b).item()
        vl/=max(len(vdl),1)
        if vl<best: best=vl; torch.save({"model":model.state_dict(),"epoch":ep,"best":best},OUT/f"{name}_best.pt")
        if ep%5==0: torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"epoch":ep+1,"best":best},res)
        if ep%10==0: log(f"  [{name}] ep {ep}/{epochs} v={vl:.5f} best={best:.5f} [{time.time()-t0:.0f}s]")
    torch.save({"model":model.state_dict()},OUT/f"{name}_final.pt")
    log(f"  ✓ {name} best={best:.5f} {time.time()-t0:.0f}s"); gc.collect(); torch.cuda.empty_cache()

# ═══ MAIN ═══
t0=time.time()
train_cells = load_cells(DATA/"train"); val_cells = load_cells(DATA/"val")
log(f"Loaded {len(train_cells)} train + {len(val_cells)} val cells")

log("\n══ M11 ElectrolyteHealth ══")
X,Y = build_eis_dataset()
if X is not None and len(X)>50:
    sp=int(0.85*len(X)); idx=torch.randperm(len(X))
    def eis_loss(m,b):
        d,p,c=m(b[0])
        l1 = F.mse_loss(torch.sigmoid(d), b[1][:,0])  # degradation
        l2 = F.binary_cross_entropy_with_logits(p, b[1][:,1])  # plating
        l3 = F.mse_loss(c, b[1][:,2])  # safe C-rate
        loss = l1 + l2 + 0.5*l3
        return loss
    train_loop("electrolyte_health",ElectrolyteHealthModel(in_dim=X.shape[1]),
               DataLoader(TensorDataset(X[idx[:sp]],Y[idx[:sp]]),batch_size=BS,shuffle=True),
               DataLoader(TensorDataset(X[idx[sp:]],Y[idx[sp:]]),batch_size=BS),eis_loss)
del X,Y; gc.collect()

log("\n══ M12 Replenishability ══")
Ht,Ft,rpt,rgt = build_replenish_dataset(train_cells)
Hv,Fv,rpv,rgv = build_replenish_dataset(val_cells)
def rep_loss(m,b):
    # Both outputs are regression targets now (not BCE)
    fr_pred, gain_pred = m(b[0],b[1])
    return F.mse_loss(fr_pred, b[2]) + 2.0*F.mse_loss(gain_pred, b[3])
train_loop("replenishability",ReplenishabilityModel(feat_dim=Ft.shape[1]),
           DataLoader(TensorDataset(Ht,Ft,rpt,rgt),batch_size=BS,shuffle=True),
           DataLoader(TensorDataset(Hv,Fv,rpv,rgv),batch_size=BS),rep_loss)

log(f"\nDone — {(time.time()-t0)/3600:.2f}h, Checkpoints: {[f.name for f in OUT.glob('*.pt')]}")
