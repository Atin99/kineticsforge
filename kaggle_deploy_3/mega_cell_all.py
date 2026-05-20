"""KineticsForge V5 MEGA — Cell A: M1 CathodeUDE, M2 SOH, M3 CycleLife, M4 FadeRate
Upload kf-allmodels-data.zip as Kaggle dataset. Run cells A→B→C sequentially.
30hr budget. Real data only. All fixes applied (corrected physics).
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd, json, time, os, glob, sys, gc
def log(m): print(m, flush=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/kaggle/working/checkpoints"; os.makedirs(CKPT, exist_ok=True)
import shutil
for r, dirs, files in os.walk("/kaggle/input"):
    if "tracker.json" in files:
        log(f"Found previous checkpoints in {r}. Restoring to {CKPT}...")
        for f in files:
            if f.endswith(".pt") or f.endswith(".json"):
                dst = os.path.join(CKPT, f)
                if not os.path.exists(dst): shutil.copy2(os.path.join(r, f), dst)
        break

t0 = time.time(); TL = 29.0*3600.0
LOG_EVERY = 5; MAX_SEQ = 600; N_FEAT = 27; COND_DIM = 5; WINDOW = 20; N_CELLS_SIM = 4; WINDOW_30 = 30

def find_data_root():
    base = "/kaggle/input"
    log(f"Scanning {base}/ for train/*.npz ...")
    for r, dirs, files in os.walk(base):
        if os.path.basename(r) == "train":
            npzs = [f for f in files if f.endswith(".npz")]
            if npzs:
                parent = os.path.dirname(r)
                log(f"FOUND: {parent} ({len(npzs)} train files)")
                return parent
    raise FileNotFoundError("No train/*.npz found! Upload data zip.")
INPUT = find_data_root()
log(f"[KF-V5 MEGA] Device={DEVICE} Data={INPUT}")

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, d, max_len=600):
        self.data = []
        for f in sorted(glob.glob(os.path.join(d,"*.npz"))):
            z = np.load(f, allow_pickle=True)
            cap = z["capacity"].astype(np.float32)
            if len(cap)<5: continue
            cond = z["conditions"].astype(np.float32)
            feats = z["features"].astype(np.float32) if "features" in z else np.zeros((len(cap),N_FEAT),dtype=np.float32)
            cl = float(z["cycle_life"]) if "cycle_life" in z else -1
            L = min(len(cap), max_len); pad = max_len - L
            self.data.append({"cap": torch.from_numpy(np.pad(cap[:L],(0,pad),constant_values=cap[min(L,len(cap))-1])),
                "feats": torch.from_numpy(np.pad(feats[:L],((0,pad),(0,0)),constant_values=0)),
                "cond": torch.from_numpy(cond), "len": L, "cl": cl})
        log(f"  loaded {len(self.data)} cells from {d}")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; return d["cond"], d["cap"], d["feats"], d["len"], d["cl"]

train_ds = CellDataset(os.path.join(INPUT,"train"))
val_ds = CellDataset(os.path.join(INPUT,"val"))

def load_tracker():
    p1 = os.path.join(CKPT,"tracker.json")
    if os.path.exists(p1):
        with open(p1) as f: return json.load(f)
    return {"done":[],"current":None,"epoch":0,"best":999}
def save_tracker(t):
    with open(os.path.join(CKPT,"tracker.json"),"w") as f: json.dump(t,f)

log("Restoration check complete.")
tracker_test = load_tracker()
log(f"Tracker state: done={tracker_test['done']}, current={tracker_test['current']}, epoch={tracker_test['epoch']}")

def time_ok(): return (time.time()-t0) < TL

def train_model(name, model, train_loader, val_loader, epochs, lr, save_name):
    tracker = load_tracker()
    if name in tracker["done"]: log(f"SKIP {name} (done)"); return True
    start_ep = tracker["epoch"] if tracker["current"]==name else 0
    best = tracker["best"] if tracker["current"]==name else 999
    ckp = os.path.join(CKPT, f"{save_name}_resume.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE=="cuda")
    if os.path.exists(ckp):
        ck = torch.load(ckp, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if "scaler" in ck: scaler.load_state_dict(ck["scaler"])
        start_ep = ck.get("epoch",0); best = ck.get("best",999)
        log(f"  resumed {save_name} from epoch {start_ep}")
    tracker["current"] = name; save_tracker(tracker)
    total_b = len(train_loader)
    log(f"  training {name}: {epochs} ep, lr={lr}, b/ep={total_b}")
    for ep in range(start_ep, epochs):
        if not time_ok():
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":ep,"best":best}, os.path.join(CKPT,f"{save_name}_resume.pt"))
            tracker["epoch"]=ep; tracker["best"]=best; save_tracker(tracker)
            log(f"  TIME LIMIT {name} ep {ep}"); return False
        model.train(); ep_loss=0; nb=0
        for bi, batch in enumerate(train_loader):
            if not time_ok():
                torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":ep,"best":best}, os.path.join(CKPT,f"{save_name}_resume.pt"))
                tracker["epoch"]=ep; tracker["best"]=best; save_tracker(tracker); return False
            loss = model.training_step(batch)
            if loss.grad_fn is None: continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            bl = float(loss.detach()); ep_loss+=bl; nb+=1
            if (bi+1)%LOG_EVERY==0 or bi==total_b-1:
                log(json.dumps({"M":name,"e":ep+1,"b":f"{bi+1}/{total_b}","loss":round(bl,6),"avg":round(ep_loss/max(nb,1),6),"best":round(best,6),"min":round((time.time()-t0)/60,1)}))
        sched.step()
        if (ep+1)%10==0:
            model.eval(); vl=0; vn=0
            with torch.no_grad():
                for vb in val_loader: vl+=float(model.training_step(vb).detach()); vn+=1
            vm = vl/max(vn,1)
            log(json.dumps({"M":name,"EVAL":ep+1,"val":round(vm,6),"best":round(best,6)}))
            if vm<best: best=vm; torch.save(model.state_dict(), os.path.join(CKPT,f"{save_name}_best.pt")); log(f"  NEW BEST {vm:.6f}")
        if (ep+1)%25==0:
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":ep+1,"best":best}, os.path.join(CKPT,f"{save_name}_resume.pt"))
            tracker["epoch"]=ep+1; tracker["best"]=best; save_tracker(tracker)
    tracker["done"].append(name); tracker["current"]=None; tracker["epoch"]=0; tracker["best"]=999; save_tracker(tracker)
    torch.save(model.state_dict(), os.path.join(CKPT,f"{save_name}_final.pt"))
    log(f"  DONE {name} best={best:.6f}"); return True


from pathlib import Path
# ── M1: CathodeUDE ──

class CathodeUDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_embed = nn.Sequential(nn.Linear(COND_DIM,64),nn.GELU(),nn.Linear(64,48))
        self.feat_embed = nn.Sequential(nn.Linear(N_FEAT,64),nn.GELU(),nn.Linear(64,32))
        self.gate = nn.Sequential(nn.Linear(2+48+32+1,128),nn.GELU(),nn.Linear(128,2),nn.Sigmoid())
        self.neural = nn.Sequential(nn.Linear(2+48+32+1,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,2))
        self.sei_k = nn.Parameter(torch.tensor(-2.0))
    def forward(self, t, state, z, fz):
        Q,R = state[...,0:1], state[...,1:2]; tv = t*torch.ones_like(Q)
        inp = torch.cat([state,z,fz,tv],dim=-1); g = self.gate(inp); nn_out = self.neural(inp)
        sei_rate = torch.exp(self.sei_k) / (torch.sqrt(tv + 1e-4))
        dQ_phys = -sei_rate * Q
        dQ = g[...,0:1]*dQ_phys + (1-g[...,0:1])*nn_out[...,0:1]; dR = nn_out[...,1:2]
        return torch.cat([dQ,dR],dim=-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            z = self.cond_embed(cond)
            # CRITICAL FIX: Use only early features (first 10 cycles) to prevent data leakage!
            # Forecasting ODEs cannot see future EIS/features at step `st`.
            early_feats = feats[:, :min(10, feats.shape[1])].mean(dim=1)
            fz_static = self.feat_embed(early_feats)
            q0 = cap[:,0:1]; state = torch.cat([q0,torch.zeros_like(q0)],dim=-1)
            preds = [q0.squeeze(-1)]; H = min(cap.shape[1],MAX_SEQ)
            dt = 1.0 / max(H, 1)
            for st in range(1,H):
                ds = torch.clamp(self(float(st)/max(H,1),state,z,fz_static),-5.0,5.0)
                state = torch.nan_to_num(state+ds*dt,nan=0,posinf=2,neginf=-2)
                preds.append(state[...,0])
            pred = torch.stack(preds,dim=1); tgt = cap[:,:pred.shape[1]]
            loss = 0; cnt = 0
            for i in range(len(lengths)):
                L = min(int(lengths[i]),pred.shape[1])
                if L<2: continue
                loss = loss + torch.mean((pred[i,:L]-tgt[i,:L])**2); cnt+=1
            loss = loss/max(cnt,1)
            mono = torch.mean(torch.relu(pred[:,1:]-pred[:,:-1]+0.001)**2)
        return loss + 0.1*mono

# ── M2: SOH ──
class SOHModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW,128),nn.GELU(),nn.Linear(128,64))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT,64),nn.GELU(),nn.Linear(64,32))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM+1,32),nn.GELU(),nn.Linear(32,32))
        self.head = nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1),nn.Sigmoid())
    def forward(self, hist, feats, cond, cf):
        h = self.hist(hist); fe = self.feat_enc(feats); c = self.cond_enc(torch.cat([cond,cf.unsqueeze(-1)],dim=-1))
        return self.head(torch.cat([h,fe,c],dim=-1)).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < WINDOW+2: continue
                for j in range(WINDOW, L, max(1,(L-WINDOW)//10)):
                    h = cap[i,j-WINDOW:j]; f = feats[i,min(j,feats.shape[1]-1)]
                    soh = cap[i,j]; cf = torch.tensor(j/L,device=DEVICE,dtype=torch.float32)
                    p = self(h.unsqueeze(0),f.unsqueeze(0),cond[i:i+1],cf.unsqueeze(0))
                    losses.append((p-soh)**2)
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M3: CycleLife ──
class CycleLifeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Sequential(nn.Linear(min(100,MAX_SEQ),128),nn.GELU(),nn.Linear(128,64))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT,32),nn.GELU(),nn.Linear(32,32))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM,32),nn.GELU(),nn.Linear(32,32))
        self.head = nn.Sequential(nn.Linear(128,96),nn.GELU(),nn.Dropout(0.1),nn.Linear(96,4))
    def forward(self, early_cap, early_feat, cond):
        h = self.early(early_cap); f = self.feat_enc(early_feat); c = self.cond_enc(cond)
        return self.head(torch.cat([h,f,c],dim=-1))
    def training_step(self, batch):
        cond,cap,feats,lengths,cl = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                if float(cl[i]) < 0: continue
                L = min(int(lengths[i]),100)
                if L < 20: continue
                ec = cap[i,:100]; ef = feats[i,:min(feats.shape[1],N_FEAT)].mean(dim=0)
                life = float(cl[i])
                label = 0 if life<200 else (1 if life<500 else (2 if life<1500 else 3))
                p = self(ec.unsqueeze(0),ef.unsqueeze(0),cond[i:i+1])
                losses.append(nn.functional.cross_entropy(p,torch.tensor([label],device=DEVICE)))
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M4: FadeRate ──
class FadeRateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW,64),nn.GELU(),nn.Linear(64,32))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT,32),nn.GELU(),nn.Linear(32,16))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM,16),nn.GELU(),nn.Linear(16,16))
        self.head = nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self, hist, feat, cond):
        h = self.hist(hist); f = self.feat_enc(feat); c = self.cond_enc(cond)
        return self.head(torch.cat([h,f,c],dim=-1)).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < WINDOW+10: continue
                for j in range(WINDOW+5,L,max(1,(L-WINDOW)//8)):
                    h = cap[i,j-WINDOW:j]; f = feats[i,min(j,feats.shape[1]-1)]
                    fade = (cap[i,j-5]-cap[i,j])/5.0
                    p = self(h.unsqueeze(0),f.unsqueeze(0),cond[i:i+1])
                    losses.append((p-fade)**2)
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M5: BMS Pack Risk (TGN) ──

class PackTGN(nn.Module):
    def __init__(self, nd=7, ed=3, h=64, nc=N_CELLS_SIM):
        super().__init__()
        self.nc = nc
        self.msg = nn.Sequential(nn.Linear(nd*2+ed,h),nn.LeakyReLU(0.2),nn.Linear(h,32))
        self.upd = nn.Sequential(nn.Linear(nd+32,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,nd))
        edges = [[i,j] for i in range(nc) for j in range(nc) if i!=j]
        self.register_buffer("ei", torch.tensor(edges,dtype=torch.long).T)
        self.ea = nn.Parameter(torch.randn(len(edges),ed)*0.1)
        self.risk = nn.Sequential(nn.Linear(nd,64),nn.GELU(),nn.Dropout(0.05),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, t, x):
        r,c = self.ei; msgs = self.msg(torch.cat([x[r],x[c],self.ea],dim=-1))
        agg = torch.zeros(x.shape[0],32,device=x.device,dtype=msgs.dtype)
        agg.scatter_add_(0, c.unsqueeze(-1).expand_as(msgs), msgs)
        x2 = self.upd(torch.cat([x,agg],dim=-1))
        return x2, self.risk(x2).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []; GS = 40
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < 10: continue
                ns = torch.zeros(N_CELLS_SIM,7,device=DEVICE)
                for ci in range(N_CELLS_SIM):
                    ns[ci,0] = cap[i,0] + torch.randn(1,device=DEVICE)*0.02
                    ns[ci,2] = cond[i,0]/50.0; ns[ci,3] = 0.8
                indices = torch.linspace(0,L-1,GS,device=DEVICE).long()
                for si, iv in enumerate(indices):
                    ii = int(iv)
                    ns2, risk_p = self(float(ii)/max(L-1,1), ns)
                    ns = ns + 0.01*torch.clamp(ns2-ns,-5,5)
                    for ci in range(N_CELLS_SIM):
                        ns[ci,0] = cap[i,ii] + torch.randn(1,device=DEVICE)*0.02
                    risk_tgt = torch.clamp((1.0-cap[i,ii])*2.0, 0, 1).expand(N_CELLS_SIM)
                    losses.append(nn.functional.mse_loss(risk_p, risk_tgt))
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M6: RUL ──
class RULModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW,128),nn.GELU(),nn.Linear(128,64))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT,64),nn.GELU(),nn.Linear(64,32))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM+1,32),nn.GELU(),nn.Linear(32,32))
        self.head = nn.Sequential(nn.Linear(128,96),nn.GELU(),nn.Dropout(0.1),nn.Linear(96,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, hist, feat, cond, cf):
        h = self.hist(hist); f = self.feat_enc(feat); c = self.cond_enc(torch.cat([cond,cf.unsqueeze(-1)],dim=-1))
        return self.head(torch.cat([h,f,c],dim=-1)).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < WINDOW+5: continue
                for j in range(WINDOW, L, max(1,(L-WINDOW)//8)):
                    h = cap[i,j-WINDOW:j]; f = feats[i,min(j,feats.shape[1]-1)]
                    rul_frac = max(0,(L-j)/L); cf = torch.tensor(j/L,device=DEVICE,dtype=torch.float32)
                    p = self(h.unsqueeze(0),f.unsqueeze(0),cond[i:i+1],cf.unsqueeze(0))
                    losses.append((p-rul_frac)**2)
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M7: Anomaly Autoencoder ──
class AnomalyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(WINDOW+N_FEAT+COND_DIM,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,16))
        self.decoder = nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,128),nn.GELU(),nn.Linear(128,WINDOW+N_FEAT))
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z), z
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < WINDOW+2: continue
                for j in range(WINDOW, L, max(1,(L-WINDOW)//6)):
                    h = cap[i,j-WINDOW:j]; f = feats[i,min(j,feats.shape[1]-1)]
                    inp = torch.cat([h,f,cond[i]],dim=-1)
                    recon, _ = self(inp.unsqueeze(0)); tgt = torch.cat([h,f],dim=-1)
                    losses.append(nn.functional.mse_loss(recon.squeeze(0), tgt))
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M8: Joint SOH+RUL+Fade ──

class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW_30,192),nn.GELU(),nn.LayerNorm(192),nn.Linear(192,128),nn.GELU(),nn.Linear(128,96))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT,96),nn.GELU(),nn.Linear(96,64))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM+1,64),nn.GELU(),nn.Linear(64,48))
        trunk_dim = 96+64+48
        self.trunk = nn.Sequential(nn.Linear(trunk_dim,192),nn.GELU(),nn.LayerNorm(192),nn.Dropout(0.1),nn.Linear(192,128),nn.GELU())
        self.soh_head = nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        self.rul_head = nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        self.fade_head = nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
    def forward(self, hist, feat, cond, cf):
        h = self.hist(hist); f = self.feat_enc(feat); c = self.cond_enc(torch.cat([cond,cf.unsqueeze(-1)],dim=-1))
        z = self.trunk(torch.cat([h,f,c],dim=-1))
        return self.soh_head(z).squeeze(-1), self.rul_head(z).squeeze(-1), self.fade_head(z).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < WINDOW_30+10: continue
                for j in range(WINDOW_30+5, L, max(1,(L-WINDOW_30)//10)):
                    h = cap[i,j-WINDOW_30:j]; f = feats[i,min(j,feats.shape[1]-1)]
                    cf = torch.tensor(j/L,device=DEVICE,dtype=torch.float32)
                    soh_p, rul_p, fade_p = self(h.unsqueeze(0),f.unsqueeze(0),cond[i:i+1],cf.unsqueeze(0))
                    soh_l = (soh_p - cap[i,j])**2
                    rul_l = (rul_p - max(0,(L-j)/L))**2
                    fade_tgt = (cap[i,j-5]-cap[i,j])/5.0 if j>=5 else torch.tensor(0.0,device=DEVICE)
                    fade_l = (fade_p - fade_tgt)**2
                    losses.append(soh_l + 0.5*rul_l + 0.3*fade_l)
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M9: KneeDetect ──
class KneeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(50))
        self.fc = nn.Sequential(nn.Linear(32*50+COND_DIM,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, cap_seq, cond):
        x = self.conv(cap_seq.unsqueeze(1)); x = x.flatten(1)
        return self.fc(torch.cat([x,cond],dim=-1)).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < 30: continue
                c = cap[i,:L]; diffs = c[1:]-c[:-1]
                if len(diffs)<5: continue
                mean_d = diffs.mean(); std_d = diffs.std()+1e-8; worst = diffs.min()
                knee_score = torch.clamp((mean_d - worst)/std_d/3.0, 0, 1)
                p = self(cap[i,:MAX_SEQ].unsqueeze(0), cond[i:i+1])
                losses.append((p - knee_score.detach())**2)
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

# ── M10: ChemRank ──
class ChemRanker(nn.Module):
    def __init__(self, n_chem=10):
        super().__init__()
        self.chem_embed = nn.Embedding(n_chem, 32)
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM-1,32),nn.GELU(),nn.Linear(32,32))
        self.head = nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self, chem_id, cond_no_chem):
        ce = self.chem_embed(chem_id); co = self.cond_enc(cond_no_chem)
        return self.head(torch.cat([ce,co],dim=-1)).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,cl = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L<20: continue
                chem_id = int(cond[i,3])
                if chem_id < 0: chem_id = 0
                chem_t = torch.tensor([chem_id],device=DEVICE,dtype=torch.long)
                cond_no = torch.cat([cond[i,:3],cond[i,4:]]).unsqueeze(0)
                avg_cap = cap[i,:L].mean()
                p = self(chem_t, cond_no)
                losses.append((p - avg_cap)**2)
            if not losses: return torch.tensor(0.0, device=DEVICE)
        return torch.stack(losses).mean()

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


# ── MAIN RUN LOOP ──
BS = 8
train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BS,shuffle=True,num_workers=2,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=BS,shuffle=False,num_workers=0)
log(f"train={len(train_ds)} val={len(val_ds)} b/ep={len(train_loader)}")

tasks = [
    ("M1_CathodeUDE", lambda: CathodeUDE().to(DEVICE), 400, 3e-4, "cathode_ude"),
    ("M2_SOH", lambda: SOHModel().to(DEVICE), 300, 5e-4, "soh"),
    ("M3_CycleLife", lambda: CycleLifeModel().to(DEVICE), 250, 5e-4, "cycle_life"),
    ("M4_FadeRate", lambda: FadeRateModel().to(DEVICE), 250, 5e-4, "fade_rate"),
    ("M5_BMS_TGN", lambda: PackTGN().to(DEVICE), 300, 3e-4, "bms_tgn"),
    ("M6_RUL", lambda: RULModel().to(DEVICE), 300, 5e-4, "rul"),
    ("M7_Anomaly", lambda: AnomalyAE().to(DEVICE), 250, 5e-4, "anomaly_ae"),
    ("M8_Joint_SOH_RUL", lambda: JointModel().to(DEVICE), 500, 3e-4, "joint_soh_rul"),
    ("M9_KneeDetect", lambda: KneeDetector().to(DEVICE), 300, 5e-4, "knee_detect"),
    ("M10_ChemRank", lambda: ChemRanker().to(DEVICE), 250, 5e-4, "chem_rank"),
]
for name, mk, epochs, lr, sn in tasks:
    if not time_ok(): log(f"TIME LIMIT before {name}"); break
    log(f"\n{'='*60}\nSTARTING {name}\n{'='*60}")
    model = mk()
    if train_model(name, model, train_loader, val_loader, epochs, lr, sn) == False: break
    del model; gc.collect(); torch.cuda.empty_cache()

# M11-M14 uses different datasets, so we run them differently
DATA = Path(INPUT)
BS = 128

def load_cells(split_dir):
    cells = []
    for f in sorted(glob.glob(str(split_dir / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cells.append({"capacity": d["capacity"].astype(np.float32), "features": d["features"].astype(np.float32),
                       "conditions": d["conditions"].astype(np.float32), "cycle_life": float(d.get("cycle_life", -1))})
    return cells

def train_loop(name, model, tdl, vdl, lfn, epochs=200):
    tracker = load_tracker()
    if name in tracker["done"]: log(f"SKIP {name} (done)"); return True
    
    model=model.to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); sc=torch.amp.GradScaler("cuda")
    best = tracker["best"] if tracker["current"]==name else 1e9
    s0 = tracker["epoch"] if tracker["current"]==name else 0
    res=Path(CKPT)/f"{name}_resume.pt"
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
