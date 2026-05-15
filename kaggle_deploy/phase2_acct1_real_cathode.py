import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import glob

TOTAL_EPOCHS = 800
BATCH_SIZE = 8
LR_WARMUP = 3e-4
LR_FINETUNE = 5e-5
WARMUP_EPOCHS = 100
HORIZON = 200
COND_DIM = 5
DATASET_SLUG = "kineticsforge-realdata-acct1"
PRETRAINED_SLUG = "kineticsforge-acct1"
INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
PRETRAINED_DIR = f"/kaggle/input/{PRETRAINED_SLUG}"
WORK_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 5
SAVE_MID = 30
os.makedirs(CKPT_DIR, exist_ok=True)

class RealCellDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_len=600):
        self.data = []
        for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            cap = d["capacity"].astype(np.float32)
            cond = d["conditions"].astype(np.float32)
            if len(cap) < 5:
                continue
            pad = max_len - len(cap)
            if pad > 0:
                cap = np.pad(cap, (0, pad), constant_values=cap[-1])
            else:
                cap = cap[:max_len]
            self.data.append({"cap": torch.from_numpy(cap), "cond": torch.from_numpy(cond), "len": min(len(d["capacity"]), max_len)})
        print(f"loaded {len(self.data)} cells from {data_dir}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        return d["cond"], d["cap"], d["len"]

class RealUDEModel(nn.Module):
    def __init__(self, cond_dim=5, state_dim=2, hidden=128):
        super().__init__()
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.physics_gate = nn.Sequential(nn.Linear(state_dim + 64 + 1, hidden), nn.GELU(), nn.Linear(hidden, state_dim), nn.Sigmoid())
        self.neural_ode = nn.Sequential(nn.Linear(state_dim + 64 + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, state_dim))
        self.sei_k0 = nn.Parameter(torch.tensor(-9.0))
        self.sei_Ea = nn.Parameter(torch.tensor(0.6))
    def physics_dQ(self, Q, t_norm, cond_z):
        temp_C = cond_z[..., 0:1] * 0.01
        k = torch.exp(self.sei_k0) * torch.exp(-self.sei_Ea / (8.314e-5 * (temp_C * 100.0 + 298.0).clamp(min=250)))
        return -k * Q * (0.01 + 0.001 * Q.abs())
    def forward(self, t_s, state, z):
        Q = state[..., 0:1]
        R = state[..., 1:2]
        t_val = t_s * torch.ones_like(Q)
        inp = torch.cat([state, z, t_val], dim=-1)
        gate = self.physics_gate(inp)
        nn_out = self.neural_ode(inp)
        dQ_phys = self.physics_dQ(Q, t_val, z)
        dQ = gate[..., 0:1] * dQ_phys + (1.0 - gate[..., 0:1]) * nn_out[..., 0:1]
        dR = nn_out[..., 1:2]
        return torch.cat([dQ, dR], dim=-1)

def load_ckpt(model, opt, sched, scaler):
    for base in [CKPT_DIR, INPUT_DIR]:
        p = os.path.join(base, "real_cathode_resume.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if scaler and "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            e, bl, bi = ck.get("epoch", 0), ck.get("best_val", float("inf")), ck.get("batch_idx", 0)
            print(f"RESUMED {p} | epoch={e} batch={bi} best_val={bl:.6f}")
            return e, bl, bi
    pre = os.path.join(PRETRAINED_DIR, "checkpoints", "cathode_best.pt")
    if not os.path.exists(pre):
        pre = os.path.join(PRETRAINED_DIR, "checkpoints", "cathode_model.pt")
    if os.path.exists(pre):
        sd = torch.load(pre, map_location=DEVICE, weights_only=False)
        matched = 0
        for k, v in sd.items():
            mapped = k.replace("comp_embed", "cond_embed").replace("neural_residual", "neural_ode")
            if mapped in dict(model.named_parameters()) or mapped in dict(model.named_buffers()):
                try:
                    p = dict(model.named_parameters()).get(mapped) or dict(model.named_buffers()).get(mapped)
                    if p is not None and p.shape == v.shape:
                        p.data.copy_(v)
                        matched += 1
                except Exception:
                    pass
        print(f"loaded pretrained weights: {matched} params matched from {pre}")
    else:
        print("no pretrained weights found, training from scratch")
    return 0, float("inf"), 0

def save_ckpt(model, opt, sched, scaler, epoch, best_val, batch_idx=0):
    p = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "epoch": epoch, "best_val": best_val, "batch_idx": batch_idx}
    if scaler:
        p["scaler"] = scaler.state_dict()
    torch.save(p, os.path.join(CKPT_DIR, "real_cathode_resume.pt"))
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "real_cathode_model.pt"))

def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for cond, cap, lengths in loader:
            cond, cap = cond.to(device), cap.to(device)
            z = model.cond_embed(cond)
            q0 = cap[:, 0:1]
            state = torch.cat([q0, torch.zeros_like(q0)], dim=-1)
            preds = [q0.squeeze(-1)]
            h = min(cap.shape[1], HORIZON)
            for st in range(1, h):
                ds = torch.clamp(model(float(st) / max(h, 1), state, z), -0.02, 0.02)
                state = torch.nan_to_num(state + ds * 0.05, nan=0.0, posinf=2.0, neginf=-2.0)
                preds.append(state[..., 0])
            pred = torch.stack(preds, dim=1)
            tgt = cap[:, :pred.shape[1]]
            for i in range(len(lengths)):
                L = min(int(lengths[i]), pred.shape[1])
                if L < 2:
                    continue
                p_i = pred[i, :L]
                t_i = tgt[i, :L]
                total_mse += float(torch.mean((p_i - t_i) ** 2))
                total_mae += float(torch.mean(torch.abs(p_i - t_i)))
                n += 1
    return total_mse / max(n, 1), total_mae / max(n, 1)

train_ds = RealCellDataset(os.path.join(INPUT_DIR, "train"))
val_ds = RealCellDataset(os.path.join(INPUT_DIR, "val"))
test_ds = RealCellDataset(os.path.join(INPUT_DIR, "test"))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
total_b = len(train_loader)
print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} batches/ep={total_b}")

model = RealUDEModel(cond_dim=COND_DIM).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR_WARMUP, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
amp_on = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

start_ep, best_val, start_bi = load_ckpt(model, opt, sched, scaler)
print(f"start_ep={start_ep} best_val={best_val:.6f} dev={DEVICE}")

t0 = time.time()
TL = 11.0 * 3600.0
gs = start_ep * total_b + start_bi

for ep in range(start_ep, TOTAL_EPOCHS):
    el = time.time() - t0
    if el > TL:
        print(f"TIME LIMIT epoch {ep}")
        save_ckpt(model, opt, sched, scaler, ep, best_val, 0)
        break
    if ep == WARMUP_EPOCHS:
        for pg in opt.param_groups:
            pg["lr"] = LR_FINETUNE
        print(f"LR switched to {LR_FINETUNE} at epoch {WARMUP_EPOCHS}")
    model.train()
    ep_loss = 0.0
    nb = 0
    skip = start_bi if ep == start_ep else 0
    for bi, (cond, cap, lengths) in enumerate(train_loader):
        if bi < skip:
            continue
        el = time.time() - t0
        if el > TL:
            print(f"TIME LIMIT mid e={ep} b={bi}")
            save_ckpt(model, opt, sched, scaler, ep, best_val, bi)
            print("SAVED. Download checkpoints/ to resume.")
            exit(0)
        cond, cap = cond.to(DEVICE), cap.to(DEVICE)
        ctx = torch.amp.autocast("cuda", enabled=amp_on) if amp_on else torch.no_grad()
        with ctx:
            z = model.cond_embed(cond)
            q0 = cap[:, 0:1]
            state = torch.cat([q0, torch.zeros_like(q0)], dim=-1)
            preds = [q0.squeeze(-1)]
            h = min(cap.shape[1], HORIZON)
            for st in range(1, h):
                ds = torch.clamp(model(float(st) / max(h, 1), state, z), -0.02, 0.02)
                state = torch.nan_to_num(state + ds * 0.05, nan=0.0, posinf=2.0, neginf=-2.0)
                preds.append(state[..., 0])
            pred = torch.stack(preds, dim=1)
            tgt = cap[:, :pred.shape[1]]
            mask_loss = 0.0
            count = 0
            for i in range(len(lengths)):
                L = min(int(lengths[i]), pred.shape[1])
                if L < 2:
                    continue
                mask_loss = mask_loss + torch.mean((pred[i, :L] - tgt[i, :L]) ** 2)
                count += 1
            loss = mask_loss / max(count, 1)
            mono = torch.mean(torch.relu(pred[:, 1:] - pred[:, :-1] + 0.001) ** 2)
            loss = loss + 0.1 * mono
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bl = float(loss.detach().cpu())
        ep_loss += bl
        nb += 1
        gs += 1
        if (bi + 1) % LOG_EVERY == 0 or bi == total_b - 1:
            print(json.dumps({"e": ep + 1, "b": f"{bi+1}/{total_b}", "bl": round(bl, 6), "avg": round(ep_loss / max(nb, 1), 6), "best_val": round(best_val, 6), "gs": gs, "min": round((time.time() - t0) / 60, 1), "rem_hr": round(max(0, TL - (time.time() - t0)) / 3600, 2)}))
        if (bi + 1) % SAVE_MID == 0:
            save_ckpt(model, opt, sched, scaler, ep, best_val, bi + 1)
    sched.step()
    avg = ep_loss / max(nb, 1)
    if (ep + 1) % 10 == 0 or ep == 0:
        val_mse, val_mae = evaluate(model, val_loader, DEVICE)
        print(json.dumps({"EVAL": ep + 1, "train_avg": round(avg, 6), "val_mse": round(val_mse, 6), "val_mae": round(val_mae, 6), "best_val": round(best_val, 6), "min": round((time.time() - t0) / 60, 1)}))
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "real_cathode_best.pt"))
            print(f"NEW BEST val_mse={val_mse:.6f}")
    if (ep + 1) % 25 == 0:
        save_ckpt(model, opt, sched, scaler, ep + 1, best_val, 0)

print("running final test evaluation...")
test_mse, test_mae = evaluate(model, test_loader, DEVICE)
print(json.dumps({"FINAL_TEST": True, "test_mse": round(test_mse, 6), "test_mae": round(test_mae, 6), "test_datasets": ["CALCE", "NASA_PCoE"]}))
save_ckpt(model, opt, sched, scaler, TOTAL_EPOCHS, best_val, 0)
print(f"DONE best_val={best_val:.6f} test_mse={test_mse:.6f}")
for f in os.listdir(CKPT_DIR):
    print(f"  {f}: {os.path.getsize(os.path.join(CKPT_DIR, f))} bytes")
