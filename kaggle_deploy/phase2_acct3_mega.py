"""ACCT3 MEGA-CELL: Trains 3 models sequentially on real data
M8: Joint SOH+RUL+Fade (big multi-task model)
M9: Knee Point Detector (detect accelerated aging onset)
M10: Chemistry Performance Ranker (rank cathode materials)
Budget: 28 hrs
"""
import torch, torch.nn as nn, numpy as np, json, time, os, glob
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SLUG = "kineticsforge-realdata-acct3a"
INPUT = f"/kaggle/input/{SLUG}"
CKPT = "/kaggle/working/checkpoints"
os.makedirs(CKPT, exist_ok=True)
t0 = time.time(); TL = 11.0*3600.0
LOG_EVERY = 5; MAX_SEQ = 600; N_FEAT = 27; COND_DIM = 5; WINDOW = 30

def find_data_root():
    base = "/kaggle/input"
    print(f"Scanning {base}/ recursively...")
    for r, dirs, files in os.walk(base):
        depth = r.replace(base, "").count(os.sep)
        if depth < 4:
            indent = "  " * depth
            print(f"{indent}{os.path.basename(r)}/ ({len(files)} files, {len(dirs)} dirs)")
            if depth >= 2 and files:
                print(f"{indent}  files: {files[:5]}")
    for r, dirs, files in os.walk(base):
        if os.path.basename(r) == "train":
            npzs = [f for f in files if f.endswith(".npz")]
            if npzs:
                parent = os.path.dirname(r)
                print(f"FOUND: {parent} ({len(npzs)} train npz files)")
                return parent
    print("WARNING: No train/*.npz found anywhere!")
    return base
INPUT = find_data_root()
print(f"DATA ROOT: {INPUT}")
print(f"  contents: {os.listdir(INPUT)}")

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
            self.data.append({
                "cap": torch.from_numpy(np.pad(cap[:L],(0,pad),constant_values=cap[min(L,len(cap))-1])),
                "feats": torch.from_numpy(np.pad(feats[:L],((0,pad),(0,0)),constant_values=0)),
                "cond": torch.from_numpy(cond), "len": L, "cl": cl})
        print(f"  loaded {len(self.data)} cells from {d}")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        return d["cond"], d["cap"], d["feats"], d["len"], d["cl"]

train_ds = CellDataset(os.path.join(INPUT,"train"))
val_ds = CellDataset(os.path.join(INPUT,"val"))
test_ds = CellDataset(os.path.join(INPUT,"test"))

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
    if name in tracker["done"]:
        print(f"SKIP {name} (done)"); return True
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
        print(f"  resumed {save_name} from epoch {start_ep}")
    tracker["current"] = name; save_tracker(tracker)
    total_b = len(train_loader)
    print(f"  training {name}: {epochs} ep, lr={lr}, b/ep={total_b}")
    for ep in range(start_ep, epochs):
        if not time_ok():
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":ep,"best":best}, os.path.join(CKPT,f"{save_name}_resume.pt"))
            tracker["epoch"]=ep; tracker["best"]=best; save_tracker(tracker); return False
        model.train(); ep_loss=0; nb=0
        for bi, batch in enumerate(train_loader):
            if not time_ok():
                torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":ep,"best":best}, os.path.join(CKPT,f"{save_name}_resume.pt"))
                tracker["epoch"]=ep; tracker["best"]=best; save_tracker(tracker); return False
            loss = model.training_step(batch)
            if loss.grad_fn is None:
                continue  # skip batch with no valid samples (avoids GradScaler assertion)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            bl = float(loss.detach()); ep_loss+=bl; nb+=1
            if (bi+1)%LOG_EVERY==0 or bi==total_b-1:
                print(json.dumps({"M":name,"e":ep+1,"b":f"{bi+1}/{total_b}","loss":round(bl,6),"avg":round(ep_loss/max(nb,1),6),"min":round((time.time()-t0)/60,1)}))
        sched.step()
        if (ep+1)%10==0:
            model.eval(); vl=0; vn=0
            with torch.no_grad():
                for vb in val_loader: vl+=float(model.training_step(vb).detach()); vn+=1
            vm = vl/max(vn,1)
            print(json.dumps({"M":name,"EVAL":ep+1,"val":round(vm,6),"best":round(best,6)}))
            if vm<best: best=vm; torch.save(model.state_dict(), os.path.join(CKPT,f"{save_name}_best.pt")); print(f"  NEW BEST {vm:.6f}")
        if (ep+1)%25==0:
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":ep+1,"best":best}, os.path.join(CKPT,f"{save_name}_resume.pt"))
            tracker["epoch"]=ep+1; tracker["best"]=best; save_tracker(tracker)
    tracker["done"].append(name); tracker["current"]=None; tracker["epoch"]=0; tracker["best"]=999; save_tracker(tracker)
    torch.save(model.state_dict(), os.path.join(CKPT,f"{save_name}_final.pt")); print(f"  DONE {name} best={best:.6f}"); return True

# ── MODEL 8: JOINT SOH+RUL+FADE (BIG MULTI-TASK) ───────────
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW,192),nn.GELU(),nn.LayerNorm(192),nn.Linear(192,128),nn.GELU(),nn.Linear(128,96))
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
                if L < WINDOW+10: continue
                for j in range(WINDOW+5, L, max(1,(L-WINDOW)//10)):
                    h = cap[i,j-WINDOW:j]; f = feats[i,min(j,feats.shape[1]-1)]
                    cf = torch.tensor(j/L,device=DEVICE,dtype=torch.float32)
                    soh_p, rul_p, fade_p = self(h.unsqueeze(0),f.unsqueeze(0),cond[i:i+1],cf.unsqueeze(0))
                    soh_l = (soh_p - cap[i,j])**2
                    rul_l = (rul_p - max(0,(L-j)/L))**2
                    fade_tgt = (cap[i,j-5]-cap[i,j])/5.0 if j>=5 else torch.tensor(0.0,device=DEVICE)
                    fade_l = (fade_p - fade_tgt)**2
                    losses.append(soh_l + 0.5*rul_l + 0.3*fade_l)
            if not losses:
                return torch.tensor(0.0, device=DEVICE)  # no grad_fn → skipped by train loop
        return torch.stack(losses).mean()

# ── MODEL 9: KNEE POINT DETECTOR ────────────────────────────
class KneeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(50))
        self.fc = nn.Sequential(nn.Linear(32*50+COND_DIM,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, cap_seq, cond):
        x = self.conv(cap_seq.unsqueeze(1))
        x = x.flatten(1)
        return self.fc(torch.cat([x,cond],dim=-1)).squeeze(-1)
    def training_step(self, batch):
        cond,cap,feats,lengths,_ = [x.to(DEVICE) if isinstance(x,torch.Tensor) else x for x in batch]
        with torch.amp.autocast("cuda",enabled=DEVICE=="cuda"):
            losses = []
            for i in range(len(lengths)):
                L = min(int(lengths[i]),cap.shape[1])
                if L < 30: continue
                c = cap[i,:L]
                diffs = c[1:]-c[:-1]
                if len(diffs)<5: continue
                mean_d = diffs.mean(); std_d = diffs.std()+1e-8
                worst = diffs.min()
                knee_score = torch.clamp((mean_d - worst)/std_d/3.0, 0, 1)
                p = self(cap[i,:MAX_SEQ].unsqueeze(0), cond[i:i+1])
                losses.append((p - knee_score.detach())**2)
            if not losses:
                return torch.tensor(0.0, device=DEVICE)  # no grad_fn → skipped by train loop
        return torch.stack(losses).mean()

# ── MODEL 10: CHEMISTRY PERFORMANCE RANKER ──────────────────
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
            if not losses:
                return torch.tensor(0.0, device=DEVICE)  # no grad_fn → skipped by train loop
        return torch.stack(losses).mean()

# ── MAIN ─────────────────────────────────────────────────────
BS = 8
train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BS,shuffle=True,num_workers=2,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=BS,shuffle=False,num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=BS,shuffle=False,num_workers=0)
print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} b/ep={len(train_loader)}")

tasks = [
    ("M8_Joint_SOH_RUL", lambda: JointModel().to(DEVICE), 1000, 3e-4, "joint_soh_rul"),
    ("M9_KneeDetect", lambda: KneeDetector().to(DEVICE), 600, 5e-4, "knee_detect"),
    ("M10_ChemRank", lambda: ChemRanker().to(DEVICE), 400, 5e-4, "chem_rank"),
]
for name, mk, epochs, lr, sn in tasks:
    if not time_ok(): print(f"TIME LIMIT before {name}"); break
    print(f"\n{'='*60}\nSTARTING {name}\n{'='*60}")
    model = mk()
    if train_model(name, model, train_loader, val_loader, epochs, lr, sn) == False: break

# Final test
tracker = load_tracker()
print(f"\nCOMPLETED: {tracker['done']}")
if "M8_Joint_SOH_RUL" in tracker["done"]:
    bp = os.path.join(CKPT,"joint_soh_rul_best.pt")
    if os.path.exists(bp):
        jm = JointModel().to(DEVICE); jm.load_state_dict(torch.load(bp,map_location=DEVICE,weights_only=False))
        jm.eval(); soh_err=[]; rul_err=[]
        with torch.no_grad():
            for cond,cap,feats,lengths,_ in test_loader:
                cond,cap,feats = cond.to(DEVICE),cap.to(DEVICE),feats.to(DEVICE)
                for i in range(len(lengths)):
                    L=min(int(lengths[i]),cap.shape[1])
                    if L<WINDOW+5: continue
                    j=L-1
                    h=cap[i,j-WINDOW:j]; f=feats[i,min(j,feats.shape[1]-1)]
                    cf=torch.tensor(j/L,device=DEVICE,dtype=torch.float32)
                    sp,rp,_ = jm(h.unsqueeze(0),f.unsqueeze(0),cond[i:i+1],cf.unsqueeze(0))
                    soh_err.append(abs(float(sp)-float(cap[i,j])))
                    rul_err.append(abs(float(rp)-max(0,(L-j)/L)))
        if soh_err:
            print(json.dumps({"TEST":True,"soh_mae":round(np.mean(soh_err),6),"rul_mae":round(np.mean(rul_err),6),"n":len(soh_err)}))

for f in os.listdir(CKPT): print(f"  {f}: {os.path.getsize(os.path.join(CKPT,f))} bytes")
print(f"Total: {(time.time()-t0)/60:.1f} min")
