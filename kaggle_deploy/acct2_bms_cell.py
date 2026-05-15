import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import glob

TOTAL_EPOCHS = 400
BATCH_SIZE = 3
LR = 5e-4
SAVE_EVERY = 20
SEQ_LEN = 360
N_SYNTH = 80
DURATION_STEPS = 900
N_CELLS = 8
DATASET_SLUG = "kineticsforge-acct2"
INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
WORK_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
DATA_DIR = os.path.join(WORK_DIR, "synth_bms")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 5
SAVE_MID = 15
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def gen_bms_data(n, out_dir, dur, nc):
    rng = np.random.RandomState(42)
    for idx in range(n):
        inj = idx < max(1, n//3)
        t = np.arange(dur, dtype=np.float32)
        ph = 2*np.pi*t/max(dur,1)
        I = (18*np.sin(ph*3)+5*np.sin(ph*17)).astype(np.float32)
        Ic = I[:,None]/max(nc,1)
        cm = rng.normal(1,0.04,size=(1,nc)).clip(0.82,1.15)
        Ta = (308+8*np.sin(ph-0.7)).astype(np.float32)
        V = (3.35+0.12*cm+0.025*rng.normal(size=(dur,nc))).astype(np.float32)
        R = (0.028/np.clip(cm,0.8,1.2)+0.0015*rng.normal(size=(dur,nc))).astype(np.float32)
        T = (Ta[:,None]+7*np.abs(Ic)*R+rng.normal(0,0.4,size=(dur,nc))).astype(np.float32)
        risk = (1/(1+np.exp(-(T-346)/6-(R-0.035)*60))).astype(np.float32)
        fc = -1; fs = np.full(nc, dur, dtype=np.int32)
        if inj:
            fc = int(rng.randint(0,nc)); onset = int(rng.randint(180,dur-2))
            ramp = np.linspace(0,1,dur-onset).astype(np.float32)
            mode = int(rng.randint(0,3))
            if mode==0: T[onset:,fc]+=70*ramp; risk[onset:,fc]=np.maximum(risk[onset:,fc],0.45+0.55*ramp)
            elif mode==1: R[onset:,fc]+=0.08*ramp; risk[onset:,fc]=np.maximum(risk[onset:,fc],0.35+0.65*ramp)
            else: T[onset:,fc]+=35*ramp; R[onset:,fc]+=0.045*ramp; risk[onset:,fc]=np.maximum(risk[onset:,fc],0.4+0.6*ramp)
            fs[fc] = onset
        np.savez_compressed(os.path.join(out_dir,f"bms_{idx:05d}.npz"),
            V=V,T=T,I=np.repeat(I[:,None],nc,axis=1).astype(np.float32),risk=risk,fail_cell=fc,failure_step=fs)
    return n

class BMSDataset(torch.utils.data.Dataset):
    def __init__(self, ddir, sl=360):
        self.data = []
        for f in sorted(glob.glob(os.path.join(ddir,"bms_*.npz"))):
            d = np.load(f, allow_pickle=True)
            ns = d["V"].shape[0]; stride = max(1,ns//sl)
            self.data.append({k:torch.from_numpy(d[k].astype(np.float32)[::stride][:sl]) for k in ["V","T","risk","I"]})
            self.data[-1]["fc"] = int(d["fail_cell"]) if "fail_cell" in d else -1
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; return d["V"],d["T"],d["I"],d["risk"],d["fc"]

class TGNFunction(nn.Module):
    def __init__(self, nd=7, ed=3, h=64):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(nd*2+ed,h),nn.LeakyReLU(0.2),nn.Linear(h,h),nn.LeakyReLU(0.2),nn.Linear(h,32))
        self.upd = nn.Sequential(nn.Linear(nd+32,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,nd))
        self.egru = nn.GRUCell(ed, h); self.eproj = nn.Linear(h, ed)
        edges = [[i,j] for i in range(N_CELLS) for j in range(N_CELLS) if abs(i-j)==1 or abs(i-j)==4]
        self.register_buffer("ei", torch.tensor(edges, dtype=torch.long).T)
        self.ea = nn.Parameter(torch.randn(len(edges), ed)*0.1)
        self.ed = nn.Parameter(torch.tensor(0.015))
    def forward(self, t, x, em=None):
        r,c = self.ei; msgs = self.msg(torch.cat([x[r],x[c],self.ea],dim=-1))
        if em is None: em = torch.zeros(self.ea.shape[0],self.egru.hidden_size,device=x.device)
        mo = self.egru(self.ea, em); me = self.eproj(mo)
        self.ea.data = self.ea.data + torch.exp(-torch.abs(self.ed)*t)*me*0.01
        agg = torch.zeros(x.shape[0],32,device=x.device,dtype=msgs.dtype)
        agg.scatter_add_(0, c.unsqueeze(-1).expand_as(msgs), msgs)
        return self.upd(torch.cat([x,agg],dim=-1)), mo

class RiskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7,96),nn.GELU(),nn.Dropout(0.05),nn.Linear(96,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x): return self.net(x)

def load_ckpt(odef, rh, opt, sched, scaler):
    for base in [CKPT_DIR, INPUT_DIR]:
        p = os.path.join(base, "bms_resume.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            odef.load_state_dict(ck["ode_fn"]); rh.load_state_dict(ck["risk_head"])
            opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
            if scaler and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
            e,bl,bi = ck.get("epoch",0),ck.get("best_loss",float("inf")),ck.get("batch_idx",0)
            print(f"RESUMED {p} | epoch={e} batch={bi} best={bl:.6f}"); return e,bl,bi
    return 0, float("inf"), 0

def save_ckpt(odef, rh, opt, sched, scaler, ep, bl, bi=0):
    p = {"ode_fn":odef.state_dict(),"risk_head":rh.state_dict(),"optimizer":opt.state_dict(),
         "scheduler":sched.state_dict(),"epoch":ep,"best_loss":bl,"batch_idx":bi}
    if scaler: p["scaler"] = scaler.state_dict()
    torch.save(p, os.path.join(CKPT_DIR,"bms_resume.pt"))
    torch.save({"ode_fn":odef.state_dict(),"risk_head":rh.state_dict()}, os.path.join(CKPT_DIR,"bms_graph_node.pt"))

existing = glob.glob(os.path.join(DATA_DIR,"bms_*.npz"))
if len(existing) < 10:
    print(f"generating {N_SYNTH} BMS files..."); print(f"generated {gen_bms_data(N_SYNTH,DATA_DIR,DURATION_STEPS,N_CELLS)}")

dataset = BMSDataset(DATA_DIR, sl=SEQ_LEN)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
total_b = len(loader)
print(f"dataset={len(dataset)} batches/epoch={total_b}")

odef = TGNFunction().to(DEVICE); rh = RiskHead().to(DEVICE)
params = list(odef.parameters()) + list(rh.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2)
amp_on = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

start_ep, best_loss, start_bi = load_ckpt(odef, rh, opt, sched, scaler)
print(f"start_ep={start_ep} start_bi={start_bi} best={best_loss:.6f} dev={DEVICE}")

t0 = time.time(); TL = 11.0*3600.0; gs = start_ep*total_b+start_bi
GRAPH_STEPS = 80

for ep in range(start_ep, TOTAL_EPOCHS):
    el = time.time()-t0
    if el > TL:
        print(f"TIME LIMIT epoch {ep}"); save_ckpt(odef,rh,opt,sched,scaler,ep,best_loss,0); break
    odef.train(); rh.train()
    ep_loss = 0.0; nb = 0
    skip = start_bi if ep == start_ep else 0
    for bi, (V,T,I,rtgt,fc) in enumerate(loader):
        if bi < skip: continue
        el = time.time()-t0
        if el > TL:
            print(f"TIME LIMIT mid e={ep} b={bi}"); save_ckpt(odef,rh,opt,sched,scaler,ep,best_loss,bi)
            print("SAVED. Download checkpoints/ to resume."); exit(0)
        V,T,I,rtgt = V.to(DEVICE),T.to(DEVICE),I.to(DEVICE),rtgt.to(DEVICE)
        ctx = torch.amp.autocast("cuda", enabled=amp_on) if amp_on else torch.no_grad()
        with ctx:
            bs,sl,nc = V.shape; loss = torch.tensor(0.0,device=DEVICE)
            indices = torch.linspace(0,sl-1,GRAPH_STEPS,device=DEVICE).long()
            sub_log_every = max(1, GRAPH_STEPS // 4)
            for b in range(bs):
                ns = torch.zeros(nc,7,device=DEVICE)
                ns[:,0]=V[b,0]; ns[:,1]=I[b,0]; ns[:,2]=T[b,0]; ns[:,3]=0.8; ns[:,4]=1e-9; ns[:,5]=0.01
                pr = None
                for si, iv in enumerate(indices):
                    ii = int(iv.detach().cpu())
                    ds, _ = odef(float(ii)/max(sl-1,1), ns)
                    ns = ns + 0.01*torch.clamp(ds,-10,10)
                    ns[:,0]=V[b,ii]; ns[:,1]=I[b,ii]; ns[:,2]=T[b,ii]
                    pred_r = rh(ns).squeeze(-1); tgt = rtgt[b,ii]
                    mse = torch.nn.functional.mse_loss(pred_r, tgt)
                    loss = loss + mse*(1+3*tgt.mean())
                    if pr is not None: loss = loss + 0.02*torch.mean(torch.relu(pr-pred_r-0.10)**2)
                    pr = pred_r
                    if (si+1) % sub_log_every == 0:
                        print(json.dumps({"e":ep+1,"b":f"{bi+1}/{total_b}","sample":b+1,
                            "gstep":f"{si+1}/{GRAPH_STEPS}","sub_mse":round(float(mse),6),
                            "min":round((time.time()-t0)/60,1)}))
            loss = loss / max(bs*GRAPH_STEPS,1)
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params,1.0); scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        bl = float(loss.detach().cpu()); ep_loss += bl; nb += 1; gs += 1
        if (bi+1) % LOG_EVERY == 0 or bi == total_b-1:
            print(json.dumps({"e":ep+1,"b":f"{bi+1}/{total_b}","bl":round(bl,6),
                "avg":round(ep_loss/max(nb,1),6),"best":round(best_loss,6),
                "gs":gs,"min":round((time.time()-t0)/60,1),
                "rem_hr":round(max(0,TL-(time.time()-t0))/3600,2)}))
        if (bi+1) % SAVE_MID == 0:
            save_ckpt(odef,rh,opt,sched,scaler,ep,best_loss,bi+1)
    sched.step()
    avg = ep_loss/max(nb,1)
    if avg < best_loss:
        best_loss = avg
        torch.save({"ode_fn":odef.state_dict(),"risk_head":rh.state_dict()}, os.path.join(CKPT_DIR,"bms_best.pt"))
    if (ep+1) % SAVE_EVERY == 0:
        save_ckpt(odef,rh,opt,sched,scaler,ep+1,best_loss,0)
    print(json.dumps({"EPOCH":ep+1,"avg":round(avg,6),"best":round(best_loss,6),
        "min":round((time.time()-t0)/60,1)}))

save_ckpt(odef,rh,opt,sched,scaler,TOTAL_EPOCHS,best_loss,0)
print(f"DONE best={best_loss:.6f}")
for f in os.listdir(CKPT_DIR): print(f"  {f}: {os.path.getsize(os.path.join(CKPT_DIR,f))} bytes")
