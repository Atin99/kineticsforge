"""KineticsForge V5 MEGA — Cell B: M5 BMS_TGN, M6 RUL, M7 Anomaly
Run AFTER Cell A. Uses same dataset. Continues tracker from Cell A.
"""
import torch, torch.nn as nn, numpy as np, json, time, os, glob, gc
def log(m): print(m, flush=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/kaggle/working/checkpoints"; os.makedirs(CKPT, exist_ok=True)
t0 = time.time(); TL = 29.0*3600.0
LOG_EVERY = 5; MAX_SEQ = 600; N_FEAT = 27; COND_DIM = 5; WINDOW = 20; N_CELLS_SIM = 4

def find_data_root():
    base = "/kaggle/input"
    for r, dirs, files in os.walk(base):
        if os.path.basename(r) == "train":
            npzs = [f for f in files if f.endswith(".npz")]
            if npzs: parent = os.path.dirname(r); log(f"FOUND: {parent} ({len(npzs)} train files)"); return parent
    raise FileNotFoundError("No train/*.npz found!")
INPUT = find_data_root()

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
    for p in [os.path.join(CKPT,"tracker.json"), os.path.join(INPUT,"tracker.json")]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return {"done":[],"current":None,"epoch":0,"best":999}
def save_tracker(t):
    with open(os.path.join(CKPT,"tracker.json"),"w") as f: json.dump(t,f)
def time_ok(): return (time.time()-t0) < TL

def train_model(name, model, train_loader, val_loader, epochs, lr, save_name):
    tracker = load_tracker()
    if name in tracker["done"]: log(f"SKIP {name} (done)"); return True
    start_ep = tracker["epoch"] if tracker["current"]==name else 0
    best = tracker["best"] if tracker["current"]==name else 999
    ckp = os.path.join(CKPT, f"{save_name}_resume.pt")
    if not os.path.exists(ckp): ckp = os.path.join(INPUT, f"{save_name}_resume.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=80, T_mult=2)
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
                log(json.dumps({"M":name,"e":ep+1,"b":f"{bi+1}/{total_b}","loss":round(bl,6),"avg":round(ep_loss/max(nb,1),6),"min":round((time.time()-t0)/60,1)}))
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

# ── RUN ──
BS = 4
train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BS,shuffle=True,num_workers=2,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=BS,shuffle=False,num_workers=0)
log(f"train={len(train_ds)} val={len(val_ds)} b/ep={len(train_loader)}")

tasks = [
    ("M5_BMS_TGN", lambda: PackTGN().to(DEVICE), 500, 3e-4, "bms_tgn"),
    ("M6_RUL", lambda: RULModel().to(DEVICE), 600, 5e-4, "rul"),
    ("M7_Anomaly", lambda: AnomalyAE().to(DEVICE), 400, 5e-4, "anomaly_ae"),
]
for name, mk, epochs, lr, sn in tasks:
    if not time_ok(): log(f"TIME LIMIT before {name}"); break
    log(f"\n{'='*60}\nSTARTING {name}\n{'='*60}")
    model = mk()
    if train_model(name, model, train_loader, val_loader, epochs, lr, sn) == False: break
    del model; gc.collect(); torch.cuda.empty_cache()

tracker = load_tracker()
log(f"\nCell B done. Completed: {tracker['done']}")
for f in os.listdir(CKPT): log(f"  {f}: {os.path.getsize(os.path.join(CKPT,f))} bytes")
log(f"Time: {(time.time()-t0)/60:.1f} min")
