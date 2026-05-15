import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import glob

TOTAL_EPOCHS = 1000
BATCH_SIZE = 16
LR = 5e-4
LR_FINE = 3e-5
WARMUP_EP = 150
COND_DIM = 5
WINDOW = 20
DATASET_SLUG = "kineticsforge-realdata-acct3"
INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
WORK_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 10
os.makedirs(CKPT_DIR, exist_ok=True)

class SOHDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, window=20):
        self.samples = []
        for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            cap = d["capacity"].astype(np.float32)
            cond = d["conditions"].astype(np.float32)
            if len(cap) < window + 5:
                continue
            for i in range(window, len(cap)):
                hist = cap[max(0, i - window):i]
                if len(hist) < window:
                    hist = np.pad(hist, (window - len(hist), 0), constant_values=hist[0])
                soh = cap[i]
                rul_frac = max(0, (len(cap) - i) / len(cap))
                fade_rate = (cap[max(0, i - 5)] - cap[i]) / max(5, 1) if i >= 5 else 0.0
                self.samples.append({
                    "hist": torch.from_numpy(hist),
                    "cond": torch.from_numpy(cond),
                    "soh": torch.tensor(soh, dtype=torch.float32),
                    "rul_frac": torch.tensor(rul_frac, dtype=torch.float32),
                    "fade_rate": torch.tensor(fade_rate, dtype=torch.float32),
                    "cycle_frac": torch.tensor(i / len(cap), dtype=torch.float32),
                })
        print(f"loaded {len(self.samples)} SOH samples from {data_dir}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        d = self.samples[i]
        return d["hist"], d["cond"], d["soh"], d["rul_frac"], d["fade_rate"], d["cycle_frac"]

class SOHModel(nn.Module):
    def __init__(self, window=20, cond_dim=5, hidden=128):
        super().__init__()
        self.hist_encoder = nn.Sequential(
            nn.Linear(window, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 64))
        self.cond_encoder = nn.Sequential(nn.Linear(cond_dim + 1, 32), nn.GELU(), nn.Linear(32, 32))
        self.soh_head = nn.Sequential(
            nn.Linear(96, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Sigmoid())
        self.rul_head = nn.Sequential(
            nn.Linear(96, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())
        self.fade_head = nn.Sequential(
            nn.Linear(96, 64), nn.GELU(),
            nn.Linear(64, 1))
    def forward(self, hist, cond, cycle_frac):
        h = self.hist_encoder(hist)
        c = self.cond_encoder(torch.cat([cond, cycle_frac.unsqueeze(-1)], dim=-1))
        z = torch.cat([h, c], dim=-1)
        soh = self.soh_head(z).squeeze(-1)
        rul = self.rul_head(z).squeeze(-1)
        fade = self.fade_head(z).squeeze(-1)
        return soh, rul, fade

def load_ckpt(model, opt, sched, scaler):
    for base in [CKPT_DIR, INPUT_DIR]:
        p = os.path.join(base, "soh_resume.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if scaler and "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            e, bl = ck.get("epoch", 0), ck.get("best_val", float("inf"))
            print(f"RESUMED {p} | epoch={e} best_val={bl:.6f}")
            return e, bl
    return 0, float("inf")

def save_ckpt(model, opt, sched, scaler, ep, bv):
    p = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "epoch": ep, "best_val": bv}
    if scaler:
        p["scaler"] = scaler.state_dict()
    torch.save(p, os.path.join(CKPT_DIR, "soh_resume.pt"))
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "soh_model.pt"))

train_ds = SOHDataset(os.path.join(INPUT_DIR, "train"), window=WINDOW)
val_ds = SOHDataset(os.path.join(INPUT_DIR, "val"), window=WINDOW)
test_ds = SOHDataset(os.path.join(INPUT_DIR, "test"), window=WINDOW)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
total_b = len(train_loader)
print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} batches/ep={total_b}")

model = SOHModel(window=WINDOW, cond_dim=COND_DIM).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
amp_on = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

start_ep, best_val = load_ckpt(model, opt, sched, scaler)
t0 = time.time()
TL = 11.0 * 3600.0

for ep in range(start_ep, TOTAL_EPOCHS):
    el = time.time() - t0
    if el > TL:
        save_ckpt(model, opt, sched, scaler, ep, best_val)
        print(f"TIME LIMIT epoch {ep}")
        break
    if ep == WARMUP_EP:
        for pg in opt.param_groups:
            pg["lr"] = LR_FINE
        print(f"LR -> {LR_FINE}")
    model.train()
    ep_loss = 0.0
    nb = 0
    for bi, (hist, cond, soh_tgt, rul_tgt, fade_tgt, cf) in enumerate(train_loader):
        el = time.time() - t0
        if el > TL:
            save_ckpt(model, opt, sched, scaler, ep, best_val)
            exit(0)
        hist, cond, soh_tgt, rul_tgt, fade_tgt, cf = hist.to(DEVICE), cond.to(DEVICE), soh_tgt.to(DEVICE), rul_tgt.to(DEVICE), fade_tgt.to(DEVICE), cf.to(DEVICE)
        ctx = torch.amp.autocast("cuda", enabled=amp_on) if amp_on else torch.no_grad()
        with ctx:
            soh_p, rul_p, fade_p = model(hist, cond, cf)
            loss_soh = torch.nn.functional.mse_loss(soh_p, soh_tgt)
            loss_rul = torch.nn.functional.mse_loss(rul_p, rul_tgt)
            loss_fade = torch.nn.functional.mse_loss(fade_p, fade_tgt)
            loss = loss_soh + 0.5 * loss_rul + 0.3 * loss_fade
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
        if (bi + 1) % LOG_EVERY == 0 or bi == total_b - 1:
            print(json.dumps({"e": ep + 1, "b": f"{bi+1}/{total_b}", "loss": round(bl, 6), "soh": round(float(loss_soh), 6), "rul": round(float(loss_rul), 6), "min": round((time.time() - t0) / 60, 1), "rem_hr": round(max(0, TL - (time.time() - t0)) / 3600, 2)}))
    sched.step()
    if (ep + 1) % 25 == 0:
        save_ckpt(model, opt, sched, scaler, ep + 1, best_val)
    if (ep + 1) % 10 == 0:
        model.eval()
        vm = 0.0
        vn = 0
        with torch.no_grad():
            for hist, cond, soh_tgt, rul_tgt, fade_tgt, cf in val_loader:
                hist, cond, soh_tgt, cf = hist.to(DEVICE), cond.to(DEVICE), soh_tgt.to(DEVICE), cf.to(DEVICE)
                soh_p, _, _ = model(hist, cond, cf)
                vm += float(torch.nn.functional.mse_loss(soh_p, soh_tgt))
                vn += 1
        vm = vm / max(vn, 1)
        print(json.dumps({"EVAL": ep + 1, "val_soh_mse": round(vm, 6), "best": round(best_val, 6)}))
        if vm < best_val:
            best_val = vm
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "soh_best.pt"))
            print(f"NEW BEST val_soh={vm:.6f}")

print("FINAL TEST...")
model.eval()
test_soh_err = []
test_rul_err = []
with torch.no_grad():
    for hist, cond, soh_tgt, rul_tgt, fade_tgt, cf in test_loader:
        hist, cond, soh_tgt, rul_tgt, cf = hist.to(DEVICE), cond.to(DEVICE), soh_tgt.to(DEVICE), rul_tgt.to(DEVICE), cf.to(DEVICE)
        soh_p, rul_p, _ = model(hist, cond, cf)
        test_soh_err.append(float(torch.mean(torch.abs(soh_p - soh_tgt))))
        test_rul_err.append(float(torch.mean(torch.abs(rul_p - rul_tgt))))
print(json.dumps({"FINAL_TEST": True, "soh_mae": round(np.mean(test_soh_err), 6), "rul_mae": round(np.mean(test_rul_err), 6), "test_datasets": ["CALCE", "NASA_PCoE"]}))
save_ckpt(model, opt, sched, scaler, TOTAL_EPOCHS, best_val)
print(f"DONE best_val={best_val:.6f}")
for f in os.listdir(CKPT_DIR):
    print(f"  {f}: {os.path.getsize(os.path.join(CKPT_DIR, f))} bytes")
