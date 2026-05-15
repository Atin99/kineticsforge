import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import glob

TOTAL_EPOCHS = 600
BATCH_SIZE = 16
LR = 8e-4
SAVE_EVERY = 25
HORIZON = 160
N_SYNTH = 400
SYNTH_CYCLES = 500
TEMPS = [298.15, 313.15, 318.15, 323.15, 333.15]
DATASET_SLUG = "kineticsforge-acct1"
INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
WORK_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
DATA_DIR = os.path.join(WORK_DIR, "synth_cathode")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 10
SAVE_MID = 50
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def gen_cathode_data(n, out_dir, n_cycles):
    comps = []
    for na in np.linspace(0.88, 1.12, 6):
        for mn in np.linspace(0.2, 0.8, 12):
            fe = max(0.0, 1.0 - mn)
            for d in [0, 1, 2, 3]:
                frac = 0.05 if d > 0 else 0.0
                comps.append([na, mn, fe, d, frac])
    np.random.seed(42)
    np.random.shuffle(comps)
    comps = comps[:n]
    idx = 0
    for comp in comps:
        for temp in TEMPS:
            na, mn, fe, dop, frac = comp
            q0 = 120.0 + 40.0 * mn - 20.0 * fe + 15.0 * (dop == 1) + np.random.normal(0, 6)
            k_fade = 1e-4 * np.exp(-0.6 * 96485.0 / (8.314 * temp))
            cap = np.zeros(n_cycles + 1, dtype=np.float32)
            res = np.zeros(n_cycles + 1, dtype=np.float32)
            temparr = np.full(n_cycles + 1, temp, dtype=np.float32)
            cap[0] = q0
            res[0] = 0.02
            for c in range(1, n_cycles + 1):
                cap[c] = max(cap[c-1] + (-k_fade * cap[c-1] * (0.01 + 0.001 * cap[c-1]**2)), 0.0)
                res[c] = res[c-1] + 0.00002 * np.exp(k_fade * c * 0.01)
            cap += np.random.normal(0, 0.008 * cap, cap.shape).astype(np.float32)
            drop = np.random.rand(n_cycles + 1) > 0.04
            drop[0] = True
            np.savez(os.path.join(out_dir, f"cathode_{idx:05d}.npz"),
                cycles=np.arange(n_cycles+1)[drop].astype(np.float32),
                capacity=cap[drop], resistance=res[drop], temperature=temparr[drop],
                Na=np.float32(na), Mn=np.float32(mn), Fe=np.float32(fe),
                dopant=str(int(dop)), dopant_frac=np.float32(frac))
            idx += 1
    return idx

class CathodeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        raw = []
        dopmap = {"0":0,"1":1,"2":2,"3":3,"None":0,"Al":1,"Ti":2,"Mg":3}
        for f in sorted(glob.glob(os.path.join(data_dir, "cathode_*.npz"))):
            d = np.load(f, allow_pickle=True)
            cap = d["capacity"].astype(np.float32)
            m = ~np.isnan(cap)
            if m.sum() < 2: continue
            cap[~m] = np.interp(np.where(~m)[0], np.where(m)[0], cap[m])
            cv = np.array([float(d["Na"]),float(d["Mn"]),float(d["Fe"]),
                dopmap.get(str(d["dopant"]),0),float(d["dopant_frac"])], dtype=np.float32)
            raw.append({"cap":cap, "res":d["resistance"].astype(np.float32),
                "temp":d["temperature"].astype(np.float32), "comp":cv})
        self.max_len = max(len(r["cap"]) for r in raw) if raw else 1
        self.data = []
        for r in raw:
            L = len(r["cap"])
            pad = self.max_len - L
            self.data.append({
                "cap":torch.from_numpy(np.pad(r["cap"],(0,pad),constant_values=r["cap"][-1])),
                "res":torch.from_numpy(np.pad(r["res"],(0,pad),constant_values=r["res"][-1])),
                "temp":torch.from_numpy(np.pad(r["temp"],(0,pad),constant_values=r["temp"][-1])),
                "comp":torch.from_numpy(r["comp"]),
                "len":L})
        print(f"padded all sequences to {self.max_len}")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        return d["comp"], d["cap"], d["res"], d["temp"]

class UDECathodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp_embed = nn.Sequential(nn.Linear(5,64),nn.LeakyReLU(0.2),nn.Linear(64,128),nn.LeakyReLU(0.2),nn.Linear(128,48))
        self.sindy_coeffs = nn.Parameter(torch.zeros(12))
        self.physics_gate = nn.Sequential(nn.Linear(52,96),nn.GELU(),nn.Linear(96,3),nn.Sigmoid())
        self.neural_residual = nn.Sequential(nn.Linear(52,96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,3))
    def sindy_basis(self, Q, V, t):
        return torch.stack([torch.ones_like(Q),Q,V,Q**2,V**2,Q*V,Q**3,torch.sin(V),
            torch.exp(-Q/(Q.abs().max()+1e-6)),t,Q*t,V*t], dim=-1)
    def arrhenius_sei(self, Q, T_norm):
        k = 1e-4 * torch.exp(-0.6 / (8.314e-5 * (T_norm * 50.0 + 318.0)))
        return -k * Q * (0.01 + 0.001 * Q**2)
    def forward(self, t_s, state, z):
        Q,V,x = state[...,0:1],state[...,1:2],state[...,2:3]
        t_val = t_s * torch.ones_like(Q)
        sindy_dQ = torch.sum(self.sindy_basis(Q,V,t_val)*self.sindy_coeffs, dim=-1, keepdim=False)
        inp = torch.cat([state, z, t_val], dim=-1)
        gate = self.physics_gate(inp)
        nn_res = self.neural_residual(inp)
        dQ = gate[...,0:1]*(self.arrhenius_sei(Q,t_val)+sindy_dQ) + (1-gate[...,0:1])*nn_res[...,0:1]
        dV = nn_res[...,1:2]
        dx = -Q.abs()/(Q.abs()+1e-6)*0.001 + nn_res[...,2:3]
        return torch.cat([dQ,dV,dx], dim=-1)

def load_ckpt(model, opt, sched, scaler):
    for base in [CKPT_DIR, INPUT_DIR]:
        p = os.path.join(base, "cathode_resume.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if scaler and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
            e,bl,bi = ck.get("epoch",0), ck.get("best_loss",float("inf")), ck.get("batch_idx",0)
            print(f"RESUMED {p} | epoch={e} batch={bi} best={bl:.6f}")
            return e, bl, bi
    return 0, float("inf"), 0

def save_ckpt(model, opt, sched, scaler, epoch, best_loss, batch_idx=0):
    p = {"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),
         "epoch":epoch,"best_loss":best_loss,"batch_idx":batch_idx}
    if scaler: p["scaler"] = scaler.state_dict()
    torch.save(p, os.path.join(CKPT_DIR,"cathode_resume.pt"))
    torch.save(model.state_dict(), os.path.join(CKPT_DIR,"cathode_model.pt"))

existing = glob.glob(os.path.join(DATA_DIR, "cathode_*.npz"))
if len(existing) < 10:
    print(f"generating synthetic data...")
    print(f"generated {gen_cathode_data(N_SYNTH//len(TEMPS), DATA_DIR, SYNTH_CYCLES)} files")

dataset = CathodeDataset(DATA_DIR)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
total_b = len(loader)
print(f"dataset={len(dataset)} batches/epoch={total_b}")

model = UDECathodeModel().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
amp_on = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

start_ep, best_loss, start_bi = load_ckpt(model, opt, sched, scaler)
print(f"start_ep={start_ep} start_bi={start_bi} best={best_loss:.6f} dev={DEVICE}")

t0 = time.time()
TL = 11.0 * 3600.0
gs = start_ep * total_b + start_bi

for ep in range(start_ep, TOTAL_EPOCHS):
    el = time.time() - t0
    if el > TL:
        print(f"TIME LIMIT epoch {ep}"); save_ckpt(model,opt,sched,scaler,ep,best_loss,0); break
    model.train()
    ep_loss = 0.0; nb = 0
    skip = start_bi if ep == start_ep else 0
    for bi, (comp, cap, res, temp) in enumerate(loader):
        if bi < skip: continue
        el = time.time() - t0
        if el > TL:
            print(f"TIME LIMIT mid e={ep} b={bi}"); save_ckpt(model,opt,sched,scaler,ep,best_loss,bi)
            print("SAVED. Download checkpoints/ to resume."); exit(0)
        comp, cap = comp.to(DEVICE), cap.to(DEVICE)
        ctx = torch.amp.autocast("cuda", enabled=amp_on) if amp_on else torch.no_grad()
        with ctx:
            z = model.comp_embed(comp)
            qs = torch.clamp(cap[:,0:1], min=1.0)
            cn = cap / qs
            s = torch.cat([cn[:,0:1], torch.ones_like(cn[:,0:1])*0.75, torch.ones_like(cn[:,0:1])*0.5], dim=-1)
            preds = [cn[:,0]]
            h = min(cn.shape[1], HORIZON)
            for st in range(1, h):
                ds = torch.clamp(model(float(st)/max(h,1), s, z), -0.05, 0.05)
                s = torch.nan_to_num(s + ds*0.02, nan=0.0, posinf=2.0, neginf=-2.0)
                preds.append(s[...,0])
            pred = torch.stack(preds, dim=1)
            tgt = cn[:,:pred.shape[1]]
            lm = torch.mean((pred-tgt)**2)
            sm = torch.mean(torch.relu(pred[:,1:]-pred[:,:-1]+0.002)**2)
            loss = lm + 0.2*sm + 0.0005*torch.sum(torch.abs(model.sindy_coeffs))
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        bl = float(lm.detach().cpu()); ep_loss += bl; nb += 1; gs += 1
        if (bi+1) % LOG_EVERY == 0 or bi == total_b-1:
            print(json.dumps({"e":ep+1,"b":f"{bi+1}/{total_b}","bl":round(bl,6),
                "avg":round(ep_loss/max(nb,1),6),"best":round(best_loss,6),
                "gs":gs,"min":round((time.time()-t0)/60,1),
                "rem_hr":round(max(0,TL-(time.time()-t0))/3600,2)}))
        if (bi+1) % SAVE_MID == 0:
            save_ckpt(model,opt,sched,scaler,ep,best_loss,bi+1)
    sched.step()
    avg = ep_loss/max(nb,1)
    if avg < best_loss:
        best_loss = avg; torch.save(model.state_dict(), os.path.join(CKPT_DIR,"cathode_best.pt"))
    if (ep+1) % SAVE_EVERY == 0:
        save_ckpt(model,opt,sched,scaler,ep+1,best_loss,0)
    print(json.dumps({"EPOCH":ep+1,"avg":round(avg,6),"best":round(best_loss,6),
        "sindy":int((model.sindy_coeffs.abs()>0.01).sum().item()),"min":round((time.time()-t0)/60,1)}))

save_ckpt(model,opt,sched,scaler,TOTAL_EPOCHS,best_loss,0)
print(f"DONE best={best_loss:.6f}")
for f in os.listdir(CKPT_DIR): print(f"  {f}: {os.path.getsize(os.path.join(CKPT_DIR,f))} bytes")
