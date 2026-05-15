import torch
import torch.nn as nn
import numpy as np
import os
import glob
import json
from pathlib import Path


class BMSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len=500):
        self.files = sorted(glob.glob(os.path.join(data_dir, "bms_*.npz")))
        self.seq_len = seq_len
        self.data = []
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            V = d["V"].astype(np.float32)
            T = d["T"].astype(np.float32)
            risk = d["risk"].astype(np.float32)
            I = d["I"].astype(np.float32)
            n_steps = V.shape[0]
            stride = max(1, n_steps // seq_len)
            self.data.append(
                {
                    "V": torch.from_numpy(V[::stride][:seq_len]),
                    "T": torch.from_numpy(T[::stride][:seq_len]),
                    "risk": torch.from_numpy(risk[::stride][:seq_len]),
                    "I": torch.from_numpy(I[::stride][:seq_len]),
                    "failure_type": str(d["failure_type"]) if "failure_type" in d else "unknown",
                    "fail_cell": int(d["fail_cell"]) if "fail_cell" in d else -1,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["V"], d["T"], d["I"], d["risk"], d["fail_cell"]


class GraphNODEFunction(nn.Module):
    def __init__(self, node_dim=7, edge_dim=3, hidden=64):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 32),
        )
        self.upd_net = nn.Sequential(
            nn.Linear(node_dim + 32, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, node_dim),
        )
        edges = []
        for i in range(8):
            for j in range(8):
                if abs(i - j) == 1 or abs(i - j) == 4:
                    edges.append([i, j])
        self.register_buffer("edge_index", torch.tensor(edges, dtype=torch.long).T)
        self.edge_attr = nn.Parameter(torch.randn(self.edge_index.shape[1], edge_dim) * 0.1)

    def forward(self, t, x):
        row, col = self.edge_index
        src, dst = x[row], x[col]
        msg_in = torch.cat([src, dst, self.edge_attr], dim=-1)
        messages = self.msg_net(msg_in)
        aggr = torch.zeros(x.shape[0], 32, device=x.device)
        aggr.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)
        upd_in = torch.cat([x, aggr], dim=-1)
        return self.upd_net(upd_in)


class BMSTrainer:
    def __init__(self, data_dir, lr=5e-4, device="cpu", checkpoint_dir="checkpoints"):
        self.device = torch.device(device)
        self.ode_fn = GraphNODEFunction().to(self.device)
        self.risk_head = nn.Sequential(
            nn.Linear(7, 96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)
        self.params = list(self.ode_fn.parameters()) + list(self.risk_head.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=40, T_mult=2)
        self.dataset = BMSDataset(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.start_epoch = 0

    def save_resume(self, epoch):
        torch.save(
            {
                "ode_fn": self.ode_fn.state_dict(),
                "risk_head": self.risk_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": self.best_loss,
            },
            self.checkpoint_dir / "bms_resume.pt",
        )

    def load_resume(self):
        path = self.checkpoint_dir / "bms_resume.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.ode_fn.load_state_dict(ckpt["ode_fn"])
            self.risk_head.load_state_dict(ckpt["risk_head"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"]
            self.best_loss = ckpt["best_loss"]

    def train_epoch(self, epoch):
        self.ode_fn.train()
        self.risk_head.train()
        total_loss = 0
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=3, shuffle=True)
        for V, T, I, risk_target, fail_cell in loader:
            batch_size, seq_len, n_cells = V.shape
            loss = torch.tensor(0.0, device=self.device)
            steps = min(80, seq_len)
            indices = torch.linspace(0, seq_len - 1, steps, device=self.device).long()
            for b in range(batch_size):
                node_state = torch.zeros(n_cells, 7, device=self.device)
                node_state[:, 0] = V[b, 0].to(self.device)
                node_state[:, 1] = I[b, 0].to(self.device)
                node_state[:, 2] = T[b, 0].to(self.device)
                node_state[:, 3] = 0.8
                node_state[:, 4] = 1e-9
                node_state[:, 5] = 0.01
                node_state[:, 6] = 0.0
                prev_risk = None
                for idx_val in indices:
                    idx_int = int(idx_val.detach().cpu())
                    dstate = self.ode_fn(float(idx_int) / max(seq_len - 1, 1), node_state)
                    node_state = node_state + 0.01 * torch.clamp(dstate, -10.0, 10.0)
                    node_state[:, 0] = V[b, idx_int].to(self.device)
                    node_state[:, 1] = I[b, idx_int].to(self.device)
                    node_state[:, 2] = T[b, idx_int].to(self.device)
                    pred_risk = self.risk_head(node_state).squeeze(-1)
                    target = risk_target[b, idx_int].to(self.device)
                    mse = nn.functional.mse_loss(pred_risk, target)
                    weighted = mse * (1.0 + 3.0 * target.mean())
                    loss = loss + weighted
                    if prev_risk is not None:
                        loss = loss + 0.02 * torch.mean(torch.relu(prev_risk - pred_risk - 0.10) ** 2)
                    prev_risk = pred_risk
            loss = loss / max(batch_size * steps, 1)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        avg = total_loss / max(len(loader), 1)
        if avg < self.best_loss:
            self.best_loss = avg
            torch.save(
                {"ode_fn": self.ode_fn.state_dict(), "risk_head": self.risk_head.state_dict()},
                self.checkpoint_dir / "bms_graph_node.pt",
            )
        return avg

    def train(self, epochs=200, resume=True):
        if resume:
            self.load_resume()
        for e in range(self.start_epoch, epochs):
            loss = self.train_epoch(e)
            if (e + 1) % 20 == 0:
                self.save_resume(e + 1)
                print(json.dumps({"e": e + 1, "loss": round(loss, 6), "best": round(self.best_loss, 6)}))
        self.save_resume(epochs)


if __name__ == "__main__":
    trainer = BMSTrainer(
        data_dir="data/synthetic/bms",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.train(epochs=200)
