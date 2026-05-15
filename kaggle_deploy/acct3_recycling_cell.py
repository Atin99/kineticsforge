import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import glob

TOTAL_EPOCHS = 500
BATCH_SIZE = 48
LR = 7e-4
SAVE_EVERY = 25
N_COND = 270
N_SPECIES = 3
STEPS = 180
DATASET_SLUG = "kineticsforge-acct3"
INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
WORK_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
DATA_DIR = os.path.join(WORK_DIR, "synth_leaching")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY_STEP = 20
LOG_EVERY_EPOCH = 5
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def gen_leaching_data(out_dir, n_cond, steps):
    rng = np.random.RandomState(42)
    conds = []
    for t in [323.15,333.15,343.15,353.15,363.15]:
        for p in [0.5,1.0,1.5,2.0,2.5,3.0]:
            for c in [0.5,1.5,3.0]:
                for s in [10.0,50.0,100.0]:
                    conds.append([t,p,c,s])
    rng.shuffle(conds); conds = conds[:n_cond]
    all_c = np.array(conds, dtype=np.float32)
    all_t = np.zeros((len(conds), N_SPECIES, steps), dtype=np.float32)
    D0=np.array([1e-12,8e-13,2e-12]); Ea=np.array([0.35,0.35,0.35])
    n_av=np.array([2.0,1.8,2.5]); R=8.314; F=96485.0
    for ci, cond in enumerate(conds):
        T,pH,ca,r0 = cond; De = D0*np.exp(-Ea*F/(R*T))
        kA = np.exp(-np.array([0.05,0.04,0.06])*(T-333)+np.array([0.3,0.2,0.4])*pH)
        alpha = np.zeros(N_SPECIES)
        for st in range(steps):
            tv = float(st+1)
            da_sc = (3*De*ca*1000)/(r0**2*5000*np.clip((1-alpha)**(1/3),1e-6,None))
            da_av = kA*n_av*np.clip(tv,1e-6,None)**(n_av-1)*np.exp(-kA*tv**n_av)
            g = 1/(1+np.exp(-(T-343)/10+pH*0.5))
            alpha = np.clip(alpha + g*da_sc+(1-g)*da_av, 0, 1)
            all_t[ci,:,st] = np.clip(alpha+rng.normal(0,0.015,N_SPECIES), 0, 1)
    np.savez_compressed(os.path.join(out_dir,"leaching_grid.npz"),
        conditions=np.array([{"T":c[0],"pH":c[1],"c_acid":c[2],"r0":c[3]} for c in conds],dtype=object),
        alpha_trajectories=all_t)
    return len(conds)

class LeachingODEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avrami_net = nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,3))
        self.blend_net = nn.Sequential(nn.Linear(3,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1),nn.Sigmoid())
        self.D0 = nn.Parameter(torch.tensor([1e-12,8e-13,2e-12]))
        self.Ea = nn.Parameter(torch.tensor([0.35,0.35,0.35]))
        self.n_avrami = nn.Parameter(torch.tensor([2.0,1.8,2.5]))
    def forward(self, t, alpha, cond):
        T,pH,ca,r0 = cond[...,0],cond[...,1],cond[...,2],cond[...,3]
        De = self.D0*torch.exp(-self.Ea*96485/(8.314*T.unsqueeze(-1)))
        da_sc = (3*De*ca.unsqueeze(-1)*1000)/(r0.unsqueeze(-1)**2*5000*torch.clamp((1-alpha)**(1/3),min=1e-6))
        kA = torch.exp(self.avrami_net(torch.stack([T,pH],dim=-1)))
        n = torch.clamp(self.n_avrami,1,4)
        da_av = kA*n*torch.clamp(t,min=1e-6).unsqueeze(-1)**(n-1)*torch.exp(-kA*t.unsqueeze(-1)**n)
        g = self.blend_net(torch.stack([T,pH,r0],dim=-1))
        return g*da_sc + (1-g)*da_av

def load_ckpt(model, opt, sched, scaler):
    for base in [CKPT_DIR, INPUT_DIR]:
        p = os.path.join(base, "recycling_resume.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if scaler and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
            e,bl = ck.get("epoch",0),ck.get("best_loss",float("inf"))
            print(f"RESUMED {p} | epoch={e} best={bl:.6f}"); return e, bl
    return 0, float("inf")

def save_ckpt(model, opt, sched, scaler, ep, bl):
    p = {"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),
         "epoch":ep,"best_loss":bl}
    if scaler: p["scaler"] = scaler.state_dict()
    torch.save(p, os.path.join(CKPT_DIR,"recycling_resume.pt"))
    torch.save(model.state_dict(), os.path.join(CKPT_DIR,"recycling_ode.pt"))

gp = os.path.join(DATA_DIR, "leaching_grid.npz")
if not os.path.exists(gp):
    print(f"generating {N_COND} conditions..."); print(f"generated {gen_leaching_data(DATA_DIR,N_COND,STEPS)}")

data = np.load(gp, allow_pickle=True)
trajs = torch.from_numpy(data["alpha_trajectories"].astype(np.float32)).to(DEVICE)
cr = data["conditions"]
cl = []
for c in cr:
    if isinstance(c,dict): cl.append([c["T"],c["pH"],c["c_acid"],c["r0"]])
    else: cl.append([float(c["T"]),float(c["pH"]),float(c["c_acid"]),float(c["r0"])])
conds = torch.tensor(cl, dtype=torch.float32).to(DEVICE)
print(f"conditions={len(conds)} traj_shape={trajs.shape}")

model = LeachingODEModel().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
amp_on = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

start_ep, best_loss = load_ckpt(model, opt, sched, scaler)
print(f"start_ep={start_ep} best={best_loss:.6f} dev={DEVICE}")

t0 = time.time(); TL = 11.0*3600.0; nc = conds.shape[0]

for ep in range(start_ep, TOTAL_EPOCHS):
    el = time.time()-t0
    if el > TL:
        print(f"TIME LIMIT epoch {ep}"); save_ckpt(model,opt,sched,scaler,ep,best_loss); break
    model.train()
    idx = torch.randperm(nc, device=DEVICE)[:min(BATCH_SIZE, nc)]
    cb = conds[idx]; tb = trajs[idx]
    ctx = torch.amp.autocast("cuda", enabled=amp_on) if amp_on else torch.no_grad()
    with ctx:
        alpha = torch.zeros(len(idx), N_SPECIES, device=DEVICE)
        loss = torch.tensor(0.0, device=DEVICE)
        hz = tb.shape[2]
        for st in range(hz):
            tv = torch.tensor(float(st), device=DEVICE)
            da = torch.clamp(model(tv, alpha, cb), 0, 0.08)
            alpha = torch.clamp(alpha + da, 0, 1)
            loss = loss + torch.nn.functional.mse_loss(alpha, tb[:,:,st])
            if (st+1) % LOG_EVERY_STEP == 0:
                print(json.dumps({"e":ep+1,"step":f"{st+1}/{hz}",
                    "step_mse":round(float(torch.nn.functional.mse_loss(alpha.detach(),tb[:,:,st])),6),
                    "min":round((time.time()-t0)/60,1)}))
        loss = loss / hz
        total = loss + torch.mean(torch.relu(-alpha)**2)
    opt.zero_grad(set_to_none=True)
    if scaler:
        scaler.scale(total).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
    else:
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    sched.step()
    val = float(loss.detach().cpu())
    if val < best_loss:
        best_loss = val; torch.save(model.state_dict(), os.path.join(CKPT_DIR,"recycling_best.pt"))
    if (ep+1) % SAVE_EVERY == 0:
        save_ckpt(model,opt,sched,scaler,ep+1,best_loss)
    if (ep+1) % LOG_EVERY_EPOCH == 0 or ep == 0:
        print(json.dumps({"EPOCH":ep+1,"loss":round(val,6),"best":round(best_loss,6),
            "min":round((time.time()-t0)/60,1),"rem_hr":round(max(0,TL-(time.time()-t0))/3600,2)}))

save_ckpt(model,opt,sched,scaler,TOTAL_EPOCHS,best_loss)
print(f"DONE best={best_loss:.6f}")
for f in os.listdir(CKPT_DIR): print(f"  {f}: {os.path.getsize(os.path.join(CKPT_DIR,f))} bytes")
