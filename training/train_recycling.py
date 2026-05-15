import torch
import torch.nn as nn
import numpy as np
import os
import json
from pathlib import Path


class LeachingODEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avrami_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )
        self.blend_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.D0 = nn.Parameter(torch.tensor([1e-12, 8e-13, 2e-12]))
        self.Ea = nn.Parameter(torch.tensor([0.35, 0.35, 0.35]))
        self.n_avrami = nn.Parameter(torch.tensor([2.0, 1.8, 2.5]))

    def forward(self, t, alpha, conditions):
        T = conditions[..., 0]
        pH = conditions[..., 1]
        c_acid = conditions[..., 2]
        r0 = conditions[..., 3]
        R = 8.314
        F = 96485.0
        D_eff = self.D0 * torch.exp(-self.Ea * F / (R * T.unsqueeze(-1)))
        da_sc = (3.0 * D_eff * c_acid.unsqueeze(-1) * 1000.0) / (
            r0.unsqueeze(-1) ** 2 * 5000.0 * torch.clamp((1.0 - alpha) ** (1.0 / 3.0), min=1e-6)
        )
        k_A = torch.exp(self.avrami_net(torch.stack([T, pH], dim=-1)))
        n = torch.clamp(self.n_avrami, 1.0, 4.0)
        da_av = (
            k_A
            * n
            * torch.clamp(t, min=1e-6).unsqueeze(-1) ** (n - 1.0)
            * torch.exp(-k_A * t.unsqueeze(-1) ** n)
        )
        gamma = self.blend_net(torch.stack([T, pH, r0], dim=-1))
        return gamma * da_sc + (1.0 - gamma) * da_av


class RecyclingTrainer:
    def __init__(self, data_path, lr=1e-3, device="cpu", checkpoint_dir="checkpoints"):
        self.device = torch.device(device)
        self.model = LeachingODEModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        data = np.load(data_path, allow_pickle=True)
        self.trajectories = torch.from_numpy(data["alpha_trajectories"].astype(np.float32)).to(self.device)
        conditions_raw = data["conditions"]
        cond_list = []
        for c in conditions_raw:
            if isinstance(c, dict):
                cond_list.append([c["T"], c["pH"], c["c_acid"], c["r0"]])
            else:
                cond_list.append([float(c["T"]), float(c["pH"]), float(c["c_acid"]), float(c["r0"])])
        self.conditions = torch.tensor(cond_list, dtype=torch.float32).to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
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
            self.checkpoint_dir / "recycling_resume.pt",
        )

    def load_resume(self):
        path = self.checkpoint_dir / "recycling_resume.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"]
            self.best_loss = ckpt["best_loss"]

    def train(self, epochs=100, resume=True):
        if resume:
            self.load_resume()
        for e in range(self.start_epoch, epochs):
            self.model.train()
            n_cond = self.conditions.shape[0]
            idx = torch.randperm(n_cond, device=self.device)[:min(48, n_cond)]
            cond_batch = self.conditions[idx]
            target_batch = self.trajectories[idx]
            alpha = torch.zeros(len(idx), 3, device=self.device)
            loss = torch.tensor(0.0, device=self.device)
            horizon = target_batch.shape[2]
            for t_step in range(horizon):
                t_val = torch.tensor(float(t_step), device=self.device)
                da = torch.clamp(self.model(t_val, alpha, cond_batch), 0.0, 0.08)
                alpha = torch.clamp(alpha + da, 0.0, 1.0)
                loss = loss + nn.functional.mse_loss(alpha, target_batch[:, :, t_step])
            loss = loss / horizon
            mono = torch.mean(torch.relu(-alpha) ** 2)
            total = loss + mono
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            val = loss.item()
            if val < self.best_loss:
                self.best_loss = val
                torch.save(self.model.state_dict(), self.checkpoint_dir / "recycling_ode.pt")
            if (e + 1) % 25 == 0:
                self.save_resume(e + 1)
            if (e + 1) % 10 == 0:
                print(json.dumps({"e": e + 1, "loss": round(val, 6), "best": round(self.best_loss, 6)}))
        self.save_resume(epochs)


if __name__ == "__main__":
    trainer = RecyclingTrainer(
        data_path="data/synthetic/leaching/leaching_grid.npz",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.train(epochs=100)
