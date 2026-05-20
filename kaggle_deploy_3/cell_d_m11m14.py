"""KineticsForge V5 MEGA — Cell D: M11 ElectrolyteHealth, M12 Replenishability, M13 ChemIdentifier, M14 FormationProtocol
Run AFTER Cell C. Uses same dataset + impedance.parquet. Continues tracker.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd, json, time, os, glob, gc
def log(m): print(m, flush=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/kaggle/working/checkpoints"; os.makedirs(CKPT, exist_ok=True)
t0 = time.time(); TL = 29.0*3600.0
EPOCHS = 200; BS = 128; LR = 3e-4; COND_DIM = 5; N_FEAT = 27

def find_data_root():
    base = "/kaggle/input"
    for r, dirs, files in os.walk(base):
        if os.path.basename(r) == "train":
            npzs = [f for f in files if f.endswith(".npz")]
            if npzs: return Path(os.path.dirname(r))
    raise FileNotFoundError("No train/*.npz found!")
from pathlib import Path
DATA = find_data_root()

def load_cells(split_dir):
    cells = []
    for f in sorted(glob.glob(str(split_dir / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cells.append({"capacity": d["capacity"].astype(np.float32), "features": d["features"].astype(np.float32),
                       "conditions": d["conditions"].astype(np.float32), "cycle_life": float(d.get("cycle_life", -1))})
    return cells

def load_tracker():
    for p in [os.path.join(CKPT,"tracker.json"), os.path.join(DATA,"tracker.json")]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return {"done":[],"current":None,"epoch":0,"best":999}
def save_tracker(t):
    with open(os.path.join(CKPT,"tracker.json"),"w") as f: json.dump(t,f)
def time_ok(): return (time.time()-t0) < TL

def train_loop(name, model, tdl, vdl, lfn, epochs=EPOCHS):
    tracker = load_tracker()
    if name in tracker["done"]: log(f"SKIP {name} (done)"); return True
    
    model=model.to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); sc=torch.amp.GradScaler("cuda")
    best = tracker["best"] if tracker["current"]==name else 1e9
    s0 = tracker["epoch"] if tracker["current"]==name else 0
    res=Path(CKPT)/f"{name}_resume.pt"
    if not res.exists(): res = DATA/f"{name}_resume.pt"
    if res.exists():
        ck=torch.load(res,map_location=DEVICE,weights_only=False); model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); s0=ck.get("epoch",0); best=ck.get("best",1e9)
        if "sched" in ck: sch.load_state_dict(ck["sched"])
        log(f"  resumed {name} from ep {s0}")
    tracker["current"]=name; save_tracker(tracker)
    total_b = len(tdl)
    for ep in range(s0,epochs):
        if not time_ok():
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sch.state_dict(),"epoch":ep,"best":best}, Path(CKPT)/f"{name}_resume.pt")
            tracker["epoch"]=ep; tracker["best"]=best; save_tracker(tracker); return False
        model.train(); tl=0
        for b in tdl:
            b=[x.to(DEVICE) for x in b]; opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"): l=lfn(model,b)
            if not torch.isfinite(l): continue
            sc.scale(l).backward(); sc.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            sc.step(opt); sc.update(); tl+=l.item()
        sch.step(); model.eval(); vl=0
        with torch.no_grad():
            for b in vdl:
                b=[x.to(DEVICE) for x in b]
                with torch.amp.autocast("cuda"): vl+=lfn(model,b).item()
        vl/=max(len(vdl),1)
        if vl<best: best=vl; torch.save({"model":model.state_dict(),"epoch":ep,"best":best},Path(CKPT)/f"{name}_best.pt")
        if ep%5==0: torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sch.state_dict(),"epoch":ep+1,"best":best},Path(CKPT)/f"{name}_resume.pt")
        if ep%10==0: log(f"  [{name}] ep {ep}/{epochs} v={vl:.5f} best={best:.5f} [{time.time()-t0:.0f}s]")
        tracker["epoch"]=ep+1; tracker["best"]=best; save_tracker(tracker)
    torch.save({"model":model.state_dict()},Path(CKPT)/f"{name}_final.pt")
    tracker["done"].append(name); tracker["current"]=None; save_tracker(tracker)
    log(f"  ✓ {name} best={best:.5f}"); return True

# ── M11 ElectrolyteHealth ──
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

def build_eis_dataset():
    imp_path = DATA / "impedance.parquet"
    if not imp_path.exists(): log("  WARNING: No impedance.parquet"); return None, None
    imp = pd.read_parquet(imp_path)
    for c in ["Re_ohm", "Rct_ohm"]:
        if c in imp.columns: imp = imp.dropna(subset=[c])
    log(f"  Loaded {len(imp)} real EIS measurements")
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
            deg = i / max(n-1, 1); plat = min(1.0, max(0.0, deg*0.5 + max(0, (25-temp))*0.02)); safe_cr = max(0.1, 2.5 - 2.0*deg)
            samples.append((feats, [deg, plat, safe_cr]))
    if not samples: return None, None
    X = torch.tensor([s[0] for s in samples], dtype=torch.float32); Y = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    mu = X.mean(dim=0, keepdim=True); std = X.std(dim=0, keepdim=True).clamp(min=1e-8)
    X = (X - mu) / std; X = torch.nan_to_num(X, 0.0)
    return X, Y

# ── M12 Replenishability ──
class ReplenishabilityModel(nn.Module):
    def __init__(self, hist_dim=20, feat_dim=10, hd=64):
        super().__init__()
        self.he = nn.Sequential(nn.Linear(hist_dim,hd),nn.GELU(),nn.Linear(hd,32))
        self.fe = nn.Sequential(nn.Linear(feat_dim,hd),nn.GELU(),nn.LayerNorm(hd),nn.Linear(hd,32))
        self.head = nn.Sequential(nn.Linear(64,48),nn.GELU(),nn.Dropout(0.1),nn.Linear(48,2))
    def forward(self, hist, feats):
        o = self.head(torch.cat([self.he(hist), self.fe(feats)], dim=-1))
        return o[:, 0], o[:, 1]

def build_replenish_dataset(cells, window=20, lookahead=20):
    hists, feats_all, targets = [], [], []
    for cell in cells:
        cap = cell["capacity"]; cond = cell["conditions"]
        if len(cap) < window + lookahead + 5: continue
        for start in range(0, len(cap)-window-lookahead, max(1, len(cap)//15)):
            end = start + window
            if end + lookahead > len(cap): break
            hist = cap[start:end].copy()
            if hist[0] > 0: hist = hist / hist[0]
            fb = float(cap[start]-cap[end]) / window
            fa = float(cap[end]-cap[end+lookahead]) / lookahead
            if abs(fb) < 1e-7: continue
            fade_ratio = fa / (fb + 1e-8); fade_ratio_norm = min(1.0, max(0.0, fade_ratio / 2.0))
            expected_end = cap[end] - fb * lookahead; actual_end = cap[end + lookahead]
            gain = max(0, min(0.15, float(actual_end - expected_end)))
            soh = float(cap[end])
            curv = float(np.mean(np.diff(np.diff(hist)))) if len(hist)>=3 else 0.0
            feat = np.array([soh, fb, fa, curv, float(np.std(np.diff(hist))),
                             cond[0]/50, cond[1], cond[2], start/max(len(cap),1),
                             cell["cycle_life"]/2000 if cell["cycle_life"]>0 else 0.5], dtype=np.float32)
            hists.append(hist); feats_all.append(np.nan_to_num(feat, 0.0)); targets.append([fade_ratio_norm, gain])
    tgt = np.array(targets, dtype=np.float32)
    return (torch.tensor(np.array(hists),dtype=torch.float32), torch.tensor(np.array(feats_all),dtype=torch.float32),
            torch.tensor(tgt[:,0],dtype=torch.float32), torch.tensor(tgt[:,1],dtype=torch.float32))

# ── M13 ChemIdentifier ──
class ChemIdentifier(nn.Module):
    def __init__(self, in_dim=27, n_classes=9, hd=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(16))
        self.head = nn.Sequential(nn.Linear(32*16+4,hd),nn.GELU(),nn.LayerNorm(hd),nn.Dropout(0.15),nn.Linear(hd,64),nn.GELU(),nn.Linear(64,n_classes))
    def forward(self, feats, cond):
        return self.head(torch.cat([self.conv(feats.unsqueeze(1)).flatten(1),cond],dim=-1))

def build_chem_dataset(cells, n_feat=27):
    all_f, all_c, all_l = [], [], []; skipped = 0
    for cell in cells:
        cat_id = int(cell["conditions"][3])
        if cat_id < 0: skipped += 1; continue
        feats = cell["features"]; cond = cell["conditions"]; n_cyc = len(feats); window = min(10, n_cyc)
        for start in range(0, n_cyc-window+1, max(1, window//2)):
            avg = np.nan_to_num(np.nanmean(feats[start:start+window], axis=0), 0).astype(np.float32)
            if len(avg)<n_feat: avg=np.pad(avg,(0,n_feat-len(avg)))
            else: avg=avg[:n_feat]
            c = np.array([cond[0]/50,cond[1],cond[2],cond[4]], dtype=np.float32)
            all_f.append(avg); all_c.append(c); all_l.append(cat_id)
    labels = np.array(all_l); unique = sorted(np.unique(labels)); lmap = {old:new for new,old in enumerate(unique)}
    remapped = np.array([lmap[l] for l in all_l], dtype=np.int64)
    return (torch.tensor(np.array(all_f),dtype=torch.float32), torch.tensor(np.array(all_c),dtype=torch.float32),
            torch.tensor(remapped,dtype=torch.long), len(unique))

# ── M14 FormationProtocol ──
class FormationProtocolModel(nn.Module):
    def __init__(self, in_dim=27, cond_dim=5, hd=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,hd),nn.GELU(),nn.LayerNorm(hd),nn.Linear(hd,64),nn.GELU())
        self.cond_enc = nn.Sequential(nn.Linear(cond_dim,32),nn.GELU(),nn.Linear(32,32))
        self.head = nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,3))
    def forward(self, feats, cond):
        out = self.head(torch.cat([self.enc(feats),self.cond_enc(cond)],dim=-1))
        return out[:,0], out[:,1], out[:,2]

def build_formation_dataset(cells, early=10, n_feat=27):
    all_f, all_c, all_t = [], [], []
    for cell in cells:
        cap=cell["capacity"]; feats=cell["features"]; cond=cell["conditions"]; cl=cell["cycle_life"]
        if len(cap)<early+5 or cl<=0: continue
        avg = np.nan_to_num(np.nanmean(feats[:early],axis=0),0).astype(np.float32)
        if len(avg)<n_feat: avg=np.pad(avg,(0,n_feat-len(avg)))
        else: avg=avg[:n_feat]
        life = min(cl/2000,1.0); robust = min(1.0,life*float(cond[2]))
        sei = min(1.0,max(0.0,float(cap[min(early,len(cap)-1)]/max(cap[0],0.01))))
        all_f.append(avg); all_c.append(cond); all_t.append([life,robust,sei])
    if not all_f: return None,None,None
    return (torch.tensor(np.array(all_f),dtype=torch.float32), torch.tensor(np.array(all_c),dtype=torch.float32),
            torch.tensor(np.array(all_t),dtype=torch.float32))

# ── MAIN RUN ──
train_cells = load_cells(DATA/"train"); val_cells = load_cells(DATA/"val")
log(f"Loaded {len(train_cells)} train + {len(val_cells)} val cells")

log("\n══ M11 ElectrolyteHealth ══")
X,Y = build_eis_dataset()
if X is not None and len(X)>50:
    sp=int(0.85*len(X)); idx=torch.randperm(len(X))
    def eis_loss(m,b):
        d,p,c=m(b[0]); l1 = F.mse_loss(torch.sigmoid(d), b[1][:,0])
        l2 = F.binary_cross_entropy_with_logits(p, b[1][:,1]); l3 = F.mse_loss(c, b[1][:,2])
        return l1 + l2 + 0.5*l3
    from torch.utils.data import DataLoader, TensorDataset
    train_loop("electrolyte_health",ElectrolyteHealthModel(in_dim=X.shape[1]),
               DataLoader(TensorDataset(X[idx[:sp]],Y[idx[:sp]]),batch_size=BS,shuffle=True),
               DataLoader(TensorDataset(X[idx[sp:]],Y[idx[sp:]]),batch_size=BS),eis_loss)
del X,Y; gc.collect()

log("\n══ M12 Replenishability ══")
Ht,Ft,rpt,rgt = build_replenish_dataset(train_cells)
Hv,Fv,rpv,rgv = build_replenish_dataset(val_cells)
def rep_loss(m,b):
    fr_pred, gain_pred = m(b[0],b[1]); return F.mse_loss(fr_pred, b[2]) + 2.0*F.mse_loss(gain_pred, b[3])
train_loop("replenishability",ReplenishabilityModel(feat_dim=Ft.shape[1]),
           DataLoader(TensorDataset(Ht,Ft,rpt,rgt),batch_size=BS,shuffle=True),
           DataLoader(TensorDataset(Hv,Fv,rpv,rgv),batch_size=BS),rep_loss)
del Ht,Ft,rpt,rgt,Hv,Fv,rpv,rgv; gc.collect()

log("\n══ M13 ChemIdentifier ══")
Ftr,Ctr,Ltr,nc = build_chem_dataset(train_cells)
Fva,Cva,Lva,_ = build_chem_dataset(val_cells)
def chem_loss(m,b): return F.cross_entropy(m(b[0],b[1]),b[2])
train_loop("chem_identifier",ChemIdentifier(in_dim=Ftr.shape[1],n_classes=nc),
           DataLoader(TensorDataset(Ftr,Ctr,Ltr),batch_size=BS,shuffle=True),
           DataLoader(TensorDataset(Fva,Cva,Lva),batch_size=BS),chem_loss)
del Ftr,Ctr,Ltr,Fva,Cva,Lva; gc.collect()

log("\n══ M14 FormationProtocol ══")
ft=build_formation_dataset(train_cells); fv=build_formation_dataset(val_cells)
if ft[0] is not None and fv[0] is not None:
    def form_loss(m,b):
        l,c,s=m(b[0],b[1]); return F.mse_loss(l,b[2][:,0])+F.mse_loss(c,b[2][:,1])+F.mse_loss(s,b[2][:,2])
    train_loop("formation_protocol",FormationProtocolModel(in_dim=ft[0].shape[1]),
               DataLoader(TensorDataset(*ft),batch_size=BS,shuffle=True),
               DataLoader(TensorDataset(*fv),batch_size=BS),form_loss)

log(f"\nALL CELLS COMPLETED — Checkpoints: {[f.name for f in Path(CKPT).glob('*.pt')]}")
