"""KineticsForge V5 MEGA — Cell A: M1 CathodeUDE, M2 SOH, M3 CycleLife, M4 FadeRate
Upload kf-allmodels-data.zip as Kaggle dataset. Run cells A→B→C sequentially.
30hr budget. Real data only. All fixes applied (corrected physics).
"""
import torch, torch.nn as nn, numpy as np, json, time, os, glob, sys, gc
def log(m): print(m, flush=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/kaggle/working/checkpoints"; os.makedirs(CKPT, exist_ok=True)
t0 = time.time(); TL = 29.0*3600.0
LOG_EVERY = 5; MAX_SEQ = 600; N_FEAT = 27; COND_DIM = 5; WINDOW = 20

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

# ── M1: CathodeUDE ──
class CathodeUDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_embed = nn.Sequential(nn.Linear(COND_DIM,64),nn.GELU(),nn.Linear(64,48))
        self.feat_embed = nn.Sequential(nn.Linear(N_FEAT,64),nn.GELU(),nn.Linear(64,32))
        self.gate = nn.Sequential(nn.Linear(2+48+32+1,128),nn.GELU(),nn.Linear(128,2),nn.Sigmoid())
        self.neural = nn.Sequential(nn.Linear(2+48+32+1,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,2))
        self.sei_k = nn.Parameter(torch.tensor(-9.0))
    def forward(self, t, state, z, fz):
        Q,R = state[...,0:1], state[...,1:2]; tv = t*torch.ones_like(Q)
        inp = torch.cat([state,z,fz,tv],dim=-1); g = self.gate(inp); nn_out = self.neural(inp)
        dQ_phys = -torch.exp(self.sei_k)*Q*(0.01+0.001*Q.abs())
        dQ = g[...,0:1]*dQ_phys + (1-g[...,0:1])*nn_out[...,0:1]; dR = nn_out[...,1:2]
        return torch.cat([dQ,dR],dim=-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            z = self.cond_embed(cond); fz_all = self.feat_embed(feats)
            q0 = cap[:,0:1]; state = torch.cat([q0,torch.zeros_like(q0)],dim=-1)
            preds = [q0.squeeze(-1)]; H = min(cap.shape[1],200)
            for st in range(1,H):
                fz = fz_all[:,min(st,fz_all.shape[1]-1)]
                ds = torch.clamp(self(float(st)/max(H,1),state,z,fz),-0.02,0.02)
                state = torch.nan_to_num(state+ds*0.05,nan=0,posinf=2,neginf=-2)
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

# ── RUN ──
BS = 8
train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BS,shuffle=True,num_workers=2,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=BS,shuffle=False,num_workers=0)
log(f"train={len(train_ds)} val={len(val_ds)} b/ep={len(train_loader)}")

tasks = [
    ("M1_CathodeUDE", lambda: CathodeUDE().to(DEVICE), 800, 3e-4, "cathode_ude"),
    ("M2_SOH", lambda: SOHModel().to(DEVICE), 600, 5e-4, "soh"),
    ("M3_CycleLife", lambda: CycleLifeModel().to(DEVICE), 400, 5e-4, "cycle_life"),
    ("M4_FadeRate", lambda: FadeRateModel().to(DEVICE), 400, 5e-4, "fade_rate"),
]
for name, mk, epochs, lr, sn in tasks:
    if not time_ok(): log(f"TIME LIMIT before {name}"); break
    log(f"\n{'='*60}\nSTARTING {name}\n{'='*60}")
    model = mk()
    if train_model(name, model, train_loader, val_loader, epochs, lr, sn) == False: break
    del model; gc.collect(); torch.cuda.empty_cache()

tracker = load_tracker()
log(f"\nCell A done. Completed: {tracker['done']}")
for f in os.listdir(CKPT): log(f"  {f}: {os.path.getsize(os.path.join(CKPT,f))} bytes")
log(f"Time: {(time.time()-t0)/60:.1f} min")
