import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import glob

TOTAL_EPOCHS = 600
BATCH_SIZE = 4
LR = 3e-4
LR_FINE = 5e-5
WARMUP_EP = 80
SEQ_LEN = 400
COND_DIM = 5
N_CELLS_SIM = 4
DATASET_SLUG = "kineticsforge-realdata-acct2"
PRETRAINED_SLUG = "kineticsforge-acct2"
INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
PRETRAINED_DIR = f"/kaggle/input/{PRETRAINED_SLUG}"
WORK_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 5
os.makedirs(CKPT_DIR, exist_ok=True)

class RealBMSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_len=400):
        self.data = []
        for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            cap = d["capacity"].astype(np.float32)
            cond = d["conditions"].astype(np.float32)
            if len(cap) < 5:
                continue
            L = min(len(cap), max_len)
            cap_padded = np.pad(cap[:L], (0, max(0, max_len - L)), constant_values=cap[min(L, len(cap)) - 1])
            risk = np.zeros(max_len, dtype=np.float32)
            for i in range(L):
                risk[i] = max(0, 1.0 - cap_padded[i]) * 2.0
            risk = np.clip(risk, 0, 1)
            V_sim = np.zeros((max_len, N_CELLS_SIM), dtype=np.float32)
            T_sim = np.zeros((max_len, N_CELLS_SIM), dtype=np.float32)
            I_sim = np.zeros((max_len, N_CELLS_SIM), dtype=np.float32)
            for c_idx in range(N_CELLS_SIM):
                noise = np.random.normal(0, 0.02, max_len).astype(np.float32)
                V_sim[:, c_idx] = 3.2 + 0.8 * cap_padded + noise
                T_sim[:, c_idx] = cond[0] + np.random.normal(0, 2, max_len).astype(np.float32)
                I_sim[:, c_idx] = cond[1] * 2.0 + np.random.normal(0, 0.1, max_len).astype(np.float32)
            risk_2d = np.tile(risk[:, None], (1, N_CELLS_SIM)).astype(np.float32)
            self.data.append({"V": torch.from_numpy(V_sim), "T": torch.from_numpy(T_sim), "I": torch.from_numpy(I_sim), "risk": torch.from_numpy(risk_2d), "cond": torch.from_numpy(cond), "len": L})
        print(f"loaded {len(self.data)} cells as pseudo-pack from {data_dir}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        return d["V"], d["T"], d["I"], d["risk"], d["len"]

class PackTGN(nn.Module):
    def __init__(self, nd=7, ed=3, h=64, nc=4):
        super().__init__()
        self.nc = nc
        self.msg = nn.Sequential(nn.Linear(nd * 2 + ed, h), nn.LeakyReLU(0.2), nn.Linear(h, 32))
        self.upd = nn.Sequential(nn.Linear(nd + 32, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, nd))
        edges = [[i, j] for i in range(nc) for j in range(nc) if i != j]
        self.register_buffer("ei", torch.tensor(edges, dtype=torch.long).T)
        self.ea = nn.Parameter(torch.randn(len(edges), ed) * 0.1)
    def forward(self, t, x):
        r, c = self.ei
        msgs = self.msg(torch.cat([x[r], x[c], self.ea], dim=-1))
        agg = torch.zeros(x.shape[0], 32, device=x.device, dtype=msgs.dtype)
        agg.scatter_add_(0, c.unsqueeze(-1).expand_as(msgs), msgs)
        return self.upd(torch.cat([x, agg], dim=-1))

class RiskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 96), nn.GELU(), nn.Dropout(0.05), nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

def load_ckpt(tgn, rh, opt, sched, scaler):
    for base in [CKPT_DIR, INPUT_DIR]:
        p = os.path.join(base, "real_bms_resume.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            tgn.load_state_dict(ck["tgn"])
            rh.load_state_dict(ck["rh"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if scaler and "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            e, bl, bi = ck.get("epoch", 0), ck.get("best_val", float("inf")), ck.get("batch_idx", 0)
            print(f"RESUMED {p} | epoch={e} batch={bi} best_val={bl:.6f}")
            return e, bl, bi
    pre = os.path.join(PRETRAINED_DIR, "checkpoints", "bms_best.pt")
    if os.path.exists(pre):
        sd = torch.load(pre, map_location=DEVICE, weights_only=False)
        if "ode_fn" in sd:
            matched = 0
            for k, v in sd["ode_fn"].items():
                mk = k.replace("msg_net", "msg").replace("upd_net", "upd")
                if mk in dict(tgn.named_parameters()):
                    p = dict(tgn.named_parameters())[mk]
                    if p.shape == v.shape:
                        p.data.copy_(v)
                        matched += 1
            print(f"loaded pretrained TGN: {matched} params")
        if "risk_head" in sd:
            try:
                rh.load_state_dict(sd["risk_head"])
                print("loaded pretrained risk_head")
            except Exception:
                print("risk_head shape mismatch, skipping")
    return 0, float("inf"), 0

def save_ckpt(tgn, rh, opt, sched, scaler, ep, bv, bi=0):
    p = {"tgn": tgn.state_dict(), "rh": rh.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "epoch": ep, "best_val": bv, "batch_idx": bi}
    if scaler:
        p["scaler"] = scaler.state_dict()
    torch.save(p, os.path.join(CKPT_DIR, "real_bms_resume.pt"))
    torch.save({"tgn": tgn.state_dict(), "rh": rh.state_dict()}, os.path.join(CKPT_DIR, "real_bms_best.pt"))

train_ds = RealBMSDataset(os.path.join(INPUT_DIR, "train"))
val_ds = RealBMSDataset(os.path.join(INPUT_DIR, "val"))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
total_b = len(train_loader)
print(f"train={len(train_ds)} val={len(val_ds)} batches/ep={total_b}")

tgn = PackTGN(nc=N_CELLS_SIM).to(DEVICE)
rh = RiskHead().to(DEVICE)
params = list(tgn.parameters()) + list(rh.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=60, T_mult=2)
amp_on = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

start_ep, best_val, start_bi = load_ckpt(tgn, rh, opt, sched, scaler)
t0 = time.time()
TL = 11.0 * 3600.0
GRAPH_STEPS = 60

for ep in range(start_ep, TOTAL_EPOCHS):
    el = time.time() - t0
    if el > TL:
        save_ckpt(tgn, rh, opt, sched, scaler, ep, best_val, 0)
        print(f"TIME LIMIT epoch {ep}")
        break
    if ep == WARMUP_EP:
        for pg in opt.param_groups:
            pg["lr"] = LR_FINE
        print(f"LR -> {LR_FINE}")
    tgn.train()
    rh.train()
    ep_loss = 0.0
    nb = 0
    skip = start_bi if ep == start_ep else 0
    for bi, (V, T, I, rtgt, lengths) in enumerate(train_loader):
        if bi < skip:
            continue
        el = time.time() - t0
        if el > TL:
            save_ckpt(tgn, rh, opt, sched, scaler, ep, best_val, bi)
            print("SAVED mid-epoch")
            exit(0)
        V, T, I, rtgt = V.to(DEVICE), T.to(DEVICE), I.to(DEVICE), rtgt.to(DEVICE)
        ctx = torch.amp.autocast("cuda", enabled=amp_on) if amp_on else torch.no_grad()
        with ctx:
            bs, sl, nc = V.shape
            indices = torch.linspace(0, sl - 1, GRAPH_STEPS, device=DEVICE).long()
            loss = torch.tensor(0.0, device=DEVICE)
            for b in range(bs):
                L = min(int(lengths[b]), sl)
                ns = torch.zeros(nc, 7, device=DEVICE)
                ns[:, 0] = V[b, 0]
                ns[:, 1] = I[b, 0]
                ns[:, 2] = T[b, 0]
                ns[:, 3] = 0.8
                for si, iv in enumerate(indices):
                    ii = min(int(iv), L - 1)
                    ds = tgn(float(ii) / max(sl - 1, 1), ns)
                    ns = ns + 0.01 * torch.clamp(ds, -10, 10)
                    ns[:, 0] = V[b, ii]
                    ns[:, 1] = I[b, ii]
                    ns[:, 2] = T[b, ii]
                    pr = rh(ns).squeeze(-1)
                    tgt = rtgt[b, ii]
                    loss = loss + torch.nn.functional.mse_loss(pr, tgt) * (1 + 3 * tgt.mean())
            loss = loss / max(bs * GRAPH_STEPS, 1)
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        bl = float(loss.detach().cpu())
        ep_loss += bl
        nb += 1
        if (bi + 1) % LOG_EVERY == 0 or bi == total_b - 1:
            print(json.dumps({"e": ep + 1, "b": f"{bi+1}/{total_b}", "bl": round(bl, 6), "avg": round(ep_loss / max(nb, 1), 6), "best_val": round(best_val, 6), "min": round((time.time() - t0) / 60, 1), "rem_hr": round(max(0, TL - (time.time() - t0)) / 3600, 2)}))
    sched.step()
    if (ep + 1) % 20 == 0:
        save_ckpt(tgn, rh, opt, sched, scaler, ep + 1, best_val, 0)
    if (ep + 1) % 10 == 0:
        tgn.eval()
        rh.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for V, T, I, rtgt, lengths in val_loader:
                V, T, I, rtgt = V.to(DEVICE), T.to(DEVICE), I.to(DEVICE), rtgt.to(DEVICE)
                bs, sl, nc = V.shape
                indices = torch.linspace(0, sl - 1, GRAPH_STEPS, device=DEVICE).long()
                for b in range(bs):
                    ns = torch.zeros(nc, 7, device=DEVICE)
                    ns[:, 0] = V[b, 0]
                    ns[:, 1] = I[b, 0]
                    ns[:, 2] = T[b, 0]
                    ns[:, 3] = 0.8
                    for si, iv in enumerate(indices):
                        ii = int(iv)
                        ds = tgn(float(ii) / max(sl - 1, 1), ns)
                        ns = ns + 0.01 * torch.clamp(ds, -10, 10)
                        ns[:, 0] = V[b, ii]
                        ns[:, 1] = I[b, ii]
                        ns[:, 2] = T[b, ii]
                        pr = rh(ns).squeeze(-1)
                        val_loss += float(torch.nn.functional.mse_loss(pr, rtgt[b, ii]))
                        vn += 1
        vm = val_loss / max(vn, 1)
        print(json.dumps({"EVAL": ep + 1, "val_mse": round(vm, 6), "best_val": round(best_val, 6)}))
        if vm < best_val:
            best_val = vm
            torch.save({"tgn": tgn.state_dict(), "rh": rh.state_dict()}, os.path.join(CKPT_DIR, "real_bms_best.pt"))
            print(f"NEW BEST val={vm:.6f}")

save_ckpt(tgn, rh, opt, sched, scaler, TOTAL_EPOCHS, best_val, 0)
print(f"DONE best_val={best_val:.6f}")
for f in os.listdir(CKPT_DIR):
    print(f"  {f}: {os.path.getsize(os.path.join(CKPT_DIR, f))} bytes")
