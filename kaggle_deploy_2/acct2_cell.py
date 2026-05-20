"""KineticsForge V5 — Account 2: M13 ChemIdentifier + M14 FormationProtocol
Uses REAL data. Upload kf-m13m14-data.zip as Kaggle dataset, paste this, GPU on, Run All.
"""
import os, time, gc, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("/kaggle/working/checkpoints"); OUT.mkdir(exist_ok=True)
EPOCHS = 200; BS = 128; LR = 3e-4; COND_DIM = 5

def find_data():
    base = "/kaggle/input"
    print(f"Scanning {base}/ for train/*.npz ...")
    for r, dirs, files in os.walk(base):
        if os.path.basename(r) == "train":
            npzs = [f for f in files if f.endswith(".npz")]
            if npzs:
                parent = os.path.dirname(r)
                print(f"FOUND: {parent} ({len(npzs)} train files)")
                return Path(parent)
    raise FileNotFoundError("No train/*.npz found! Upload data zip as Kaggle dataset.")

DATA = find_data()
meta = json.loads((DATA / "meta.json").read_text()) if (DATA / "meta.json").exists() else {}
print(f"[KF-V5] Device={DEVICE} Data={DATA} Cathodes={meta.get('cathode_map',{})}")

# ═══ M13: ChemIdentifier ═══
class ChemIdentifier(nn.Module):
    def __init__(self, in_dim=27, n_classes=9, hd=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(16))
        self.head = nn.Sequential(nn.Linear(32*16+4,hd),nn.GELU(),nn.LayerNorm(hd),nn.Dropout(0.15),nn.Linear(hd,64),nn.GELU(),nn.Linear(64,n_classes))
    def forward(self, feats, cond):
        return self.head(torch.cat([self.conv(feats.unsqueeze(1)).flatten(1),cond],dim=-1))

# ═══ M14: FormationProtocol ═══
class FormationProtocolModel(nn.Module):
    def __init__(self, in_dim=27, cond_dim=5, hd=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,hd),nn.GELU(),nn.LayerNorm(hd),nn.Linear(hd,64),nn.GELU())
        self.cond_enc = nn.Sequential(nn.Linear(cond_dim,32),nn.GELU(),nn.Linear(32,32))
        self.head = nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,3))
    def forward(self, feats, cond):
        out = self.head(torch.cat([self.enc(feats),self.cond_enc(cond)],dim=-1))
        return out[:,0], out[:,1], out[:,2]

# ═══ REAL DATA ═══
def load_cells(split_dir):
    cells = []
    for f in sorted(glob.glob(str(split_dir / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cells.append({"capacity": d["capacity"].astype(np.float32), "features": d["features"].astype(np.float32),
                       "conditions": d["conditions"].astype(np.float32), "cycle_life": float(d.get("cycle_life", -1))})
    return cells

def build_chem_dataset(cells, n_feat=27):
    all_f, all_c, all_l = [], [], []; skipped = 0
    for cell in cells:
        cat_id = int(cell["conditions"][3])
        if cat_id < 0: skipped += 1; continue
        feats = cell["features"]; cond = cell["conditions"]; n_cyc = len(feats)
        window = min(10, n_cyc)
        for start in range(0, n_cyc-window+1, max(1, window//2)):
            avg = np.nan_to_num(np.nanmean(feats[start:start+window], axis=0), 0).astype(np.float32)
            if len(avg)<n_feat: avg=np.pad(avg,(0,n_feat-len(avg)))
            else: avg=avg[:n_feat]
            c = np.array([cond[0]/50,cond[1],cond[2],cond[4]], dtype=np.float32)
            all_f.append(avg); all_c.append(c); all_l.append(cat_id)
    labels = np.array(all_l); unique = sorted(np.unique(labels))
    lmap = {old:new for new,old in enumerate(unique)}
    remapped = np.array([lmap[l] for l in all_l], dtype=np.int64)
    print(f"  {len(all_f)} samples, {len(unique)} classes, skipped {skipped} unknown")
    return (torch.tensor(np.array(all_f),dtype=torch.float32), torch.tensor(np.array(all_c),dtype=torch.float32),
            torch.tensor(remapped,dtype=torch.long), len(unique))

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
    print(f"  {len(all_f)} formation samples with cycle_life labels")
    if not all_f: return None,None,None
    return (torch.tensor(np.array(all_f),dtype=torch.float32), torch.tensor(np.array(all_c),dtype=torch.float32),
            torch.tensor(np.array(all_t),dtype=torch.float32))

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
        if time.time()-t0>6.5*3600: print(f"  ⏱ Time limit ep {ep}"); break
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
        if ep%10==0: print(f"  [{name}] ep {ep}/{epochs} v={vl:.5f} best={best:.5f} [{time.time()-t0:.0f}s]")
    torch.save({"model":model.state_dict()},OUT/f"{name}_final.pt")
    print(f"  ✓ {name} best={best:.5f} {time.time()-t0:.0f}s"); gc.collect(); torch.cuda.empty_cache()

# ═══ MAIN ═══
t0=time.time()
train_cells = load_cells(DATA/"train"); val_cells = load_cells(DATA/"val")
print(f"Loaded {len(train_cells)} train + {len(val_cells)} val cells")

print("\n══ M13 ChemIdentifier ══")
Ftr,Ctr,Ltr,nc = build_chem_dataset(train_cells)
Fva,Cva,Lva,_ = build_chem_dataset(val_cells)
def chem_loss(m,b): return F.cross_entropy(m(b[0],b[1]),b[2])
train_loop("chem_identifier",ChemIdentifier(in_dim=Ftr.shape[1],n_classes=nc),
           DataLoader(TensorDataset(Ftr,Ctr,Ltr),batch_size=BS,shuffle=True),
           DataLoader(TensorDataset(Fva,Cva,Lva),batch_size=BS),chem_loss)
del Ftr,Ctr,Ltr,Fva,Cva,Lva; gc.collect()

print("\n══ M14 FormationProtocol ══")
ft=build_formation_dataset(train_cells); fv=build_formation_dataset(val_cells)
if ft[0] is not None and fv[0] is not None:
    def form_loss(m,b):
        l,c,s=m(b[0],b[1]); return F.mse_loss(l,b[2][:,0])+F.mse_loss(c,b[2][:,1])+F.mse_loss(s,b[2][:,2])
    train_loop("formation_protocol",FormationProtocolModel(in_dim=ft[0].shape[1]),
               DataLoader(TensorDataset(*ft),batch_size=BS,shuffle=True),
               DataLoader(TensorDataset(*fv),batch_size=BS),form_loss)

print(f"\nDone — {(time.time()-t0)/3600:.2f}h, Checkpoints: {[f.name for f in OUT.glob('*.pt')]}")
