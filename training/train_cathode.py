import torch
import torch.nn as nn
import numpy as np
import os
import glob
import json
import time
from pathlib import Path


class CathodeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_files=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "cathode_*.npz")))
        if max_files:
            self.files = self.files[:max_files]
        self.data = []
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            capacity = d["capacity"].astype(np.float32)
            mask = ~np.isnan(capacity)
            if mask.sum() < 2:
                continue
            capacity[~mask] = np.interp(
                np.where(~mask)[0], np.where(mask)[0], capacity[mask]
            )
            dopmap = {"0": 0, "1": 1, "2": 2, "3": 3, "None": 0, "Al": 1, "Ti": 2, "Mg": 3}
            comp_vec = np.array(
                [
                    float(d["Na"]),
                    float(d["Mn"]),
                    float(d["Fe"]),
                    dopmap.get(str(d["dopant"]), 0),
                    float(d["dopant_frac"]),
                ],
                dtype=np.float32,
            )
            self.data.append(
                {
                    "capacity": torch.from_numpy(capacity),
                    "resistance": torch.from_numpy(d["resistance"].astype(np.float32)),
                    "temperature": torch.from_numpy(d["temperature"].astype(np.float32)),
                    "composition": torch.from_numpy(comp_vec),
                    "cycles": torch.from_numpy(d["cycles"].astype(np.float32)),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["composition"], d["capacity"], d["resistance"], d["temperature"]


class SINDyNODEModel(nn.Module):
    def __init__(self, comp_dim=5, state_dim=3, hidden=96, sindy_terms=12):
        super().__init__()
        self.comp_embed = nn.Sequential(
            nn.Linear(comp_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 48),
        )
        self.sindy_coeffs = nn.Parameter(torch.zeros(sindy_terms))
        self.physics_gate = nn.Sequential(
            nn.Linear(state_dim + 48 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
            nn.Sigmoid(),
        )
        self.neural_residual = nn.Sequential(
            nn.Linear(state_dim + 48 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
        )

    def sindy_basis(self, Q, V, t):
        return torch.stack(
            [
                torch.ones_like(Q),
                Q,
                V,
                Q ** 2,
                V ** 2,
                Q * V,
                Q ** 3,
                torch.sin(V),
                torch.exp(-Q / (Q.abs().max() + 1e-6)),
                t,
                Q * t,
                V * t,
            ],
            dim=-1,
        )

    def forward(self, t_scalar, state, z_comp):
        Q = state[..., 0:1]
        V = state[..., 1:2]
        x = state[..., 2:3]
        basis = self.sindy_basis(Q, V, t_scalar * torch.ones_like(Q))
        sindy_dQ = torch.sum(basis * self.sindy_coeffs, dim=-1, keepdim=True)
        nn_input = torch.cat([state, z_comp, t_scalar * torch.ones_like(Q)], dim=-1)
        gate = self.physics_gate(nn_input)
        nn_res = self.neural_residual(nn_input)
        dQ = gate[..., 0:1] * sindy_dQ + (1.0 - gate[..., 0:1]) * nn_res[..., 0:1]
        dV = nn_res[..., 1:2]
        dx = -Q.abs() / (Q.abs() + 1e-6) * 0.001 + nn_res[..., 2:3]
        return torch.cat([dQ, dV, dx], dim=-1)


class CathodeTrainer:
    def __init__(self, data_dir, lr=1e-3, device="cpu", checkpoint_dir="checkpoints"):
        self.device = torch.device(device)
        self.model = SINDyNODEModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        self.dataset = CathodeDataset(data_dir)
        self.best_loss = float("inf")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0

    def save_resume(self, epoch):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": self.best_loss,
            },
            self.checkpoint_dir / "cathode_resume.pt",
        )

    def load_resume(self):
        path = self.checkpoint_dir / "cathode_resume.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"]
            self.best_loss = ckpt["best_loss"]

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True)
        for comp, cap, res, temp in loader:
            comp, cap = comp.to(self.device), cap.to(self.device)
            z_comp = self.model.comp_embed(comp)
            q_scale = torch.clamp(cap[:, 0:1], min=1.0)
            cap_norm = cap / q_scale
            q0 = cap_norm[:, 0:1]
            v0 = torch.ones_like(q0) * 0.75
            x0 = torch.ones_like(q0) * 0.5
            state = torch.cat([q0, v0, x0], dim=-1)
            pred_caps = [q0.squeeze(-1)]
            h = min(cap_norm.shape[1], 160)
            for t in range(1, h):
                dstate = self.model(float(t) / max(h, 1), state, z_comp)
                dstate = torch.clamp(dstate, -0.05, 0.05)
                state = state + dstate * 0.02
                state = torch.nan_to_num(state, nan=0.0, posinf=2.0, neginf=-2.0)
                pred_caps.append(state[..., 0])
            pred = torch.stack(pred_caps, dim=1)
            target = cap_norm[:, : pred.shape[1]]
            loss_main = torch.mean((pred - target) ** 2)
            smooth = torch.mean(torch.relu(pred[:, 1:] - pred[:, :-1] + 0.002) ** 2)
            sindy_l1 = 0.0005 * torch.sum(torch.abs(self.model.sindy_coeffs))
            loss = loss_main + 0.2 * smooth + sindy_l1
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss_main.item()
        self.scheduler.step()
        avg = total_loss / max(len(loader), 1)
        if avg < self.best_loss:
            self.best_loss = avg
            torch.save(self.model.state_dict(), self.checkpoint_dir / "cathode_model.pt")
        return avg

    def train(self, epochs=500, resume=True):
        if resume:
            self.load_resume()
        for e in range(self.start_epoch, epochs):
            loss = self.train_epoch(e)
            if (e + 1) % 25 == 0:
                self.save_resume(e + 1)
            if (e + 1) % 50 == 0:
                active = int((self.model.sindy_coeffs.abs() > 0.01).sum().item())
                print(json.dumps({"e": e + 1, "loss": round(loss, 6), "best": round(self.best_loss, 6), "sindy": active}))
        self.save_resume(epochs)


if __name__ == "__main__":
    trainer = CathodeTrainer(
        data_dir="data/synthetic/cathode",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.train(epochs=500)
