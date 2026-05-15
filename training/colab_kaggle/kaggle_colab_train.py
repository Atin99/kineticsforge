import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root_from_file()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from training.train_bms import BMSTrainer, BMSDataset
from training.train_cathode import CathodeTrainer, CathodeDataset
from training.train_recycling import RecyclingTrainer


@dataclass
class TrainProfile:
    name: str
    cathode_epochs: int
    bms_epochs: int
    recycling_epochs: int
    cathode_max_files: Optional[int]
    bms_seq_len: int
    batch_note: str


PROFILES = {
    "quick": TrainProfile("quick", 2, 2, 2, 32, 180, "fast smoke profile"),
    "standard": TrainProfile("standard", 20, 12, 20, 128, 360, "free GPU profile"),
    "deep": TrainProfile("deep", 80, 45, 60, None, 500, "long Kaggle session profile"),
}


class RunLedger:
    def __init__(self, root: Path, run_name: str):
        self.root = root
        self.run_name = run_name
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.events: List[Dict[str, Any]] = []
        self.out_dir = root / "training" / "colab_kaggle" / "runs" / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, payload: Dict[str, Any]) -> None:
        item = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": name}
        item.update(payload)
        self.events.append(item)
        self.flush()

    def flush(self) -> None:
        payload = {
            "run_name": self.run_name,
            "started_at": self.started_at,
            "events": self.events,
        }
        (self.out_dir / "run_ledger.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class EnvironmentReport:
    def __init__(self, root: Path):
        self.root = root

    def as_dict(self) -> Dict[str, Any]:
        cuda = torch.cuda.is_available()
        return {
            "project_root": str(self.root),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(cuda),
            "cuda_device_count": torch.cuda.device_count() if cuda else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda else "",
            "numpy": np.__version__,
            "working_directory": os.getcwd(),
        }


class DataAvailability:
    def __init__(self, root: Path):
        self.root = root

    def count_npz(self, relative: str, pattern: str) -> int:
        return len(list((self.root / relative).glob(pattern))) if (self.root / relative).exists() else 0

    def file_size(self, relative: str) -> int:
        path = self.root / relative
        return int(path.stat().st_size) if path.exists() else 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cathode_npz": self.count_npz("data/synthetic/cathode", "cathode_*.npz"),
            "bms_npz": self.count_npz("data/synthetic/bms", "bms_*.npz"),
            "leaching_grid_exists": (self.root / "data/synthetic/leaching/leaching_grid.npz").exists(),
            "real_manifest_exists": (self.root / "data/real/assembled/real_dataset_manifest.json").exists(),
            "foundation_manifest_size": self.file_size("data/cache/hyper_manifest_foundation.json"),
            "foundation_quality_size": self.file_size("data/cache/data_quality_report_foundation.json"),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


class CathodeKaggleTrainer(CathodeTrainer):
    def __init__(self, data_dir: str, max_files: Optional[int], lr: float, device: str, checkpoint_dir: Path):
        self.device = torch.device(device)
        self.model = __import__("training.train_cathode", fromlist=["SINDyNODEModel"]).SINDyNODEModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        self.dataset = CathodeDataset(data_dir, max_files=max_files)
        self.best_loss = float("inf")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.amp = MixedPrecisionWrapper(enabled=(self.device.type == "cuda"))

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True, num_workers=0)
        for comp, cap, res, temp in loader:
            comp = comp.to(self.device)
            cap = cap.to(self.device)
            with self.amp.forward_context():
                z_comp = self.model.comp_embed(comp)
                q_scale = torch.clamp(cap[:, 0:1], min=1.0)
                cap_norm = cap / q_scale
                q0 = cap_norm[:, 0:1]
                v0 = torch.ones_like(q0) * 0.75
                x0 = torch.ones_like(q0) * 0.5
                state = torch.cat([q0, v0, x0], dim=-1)
                pred_caps = [q0.squeeze(-1)]
                horizon = min(cap_norm.shape[1], 160)
                for step in range(1, horizon):
                    dstate = self.model(float(step) / max(horizon, 1), state, z_comp)
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
            self.amp.backward(loss, self.optimizer, self.model.parameters(), max_norm=1.0)
            total_loss += float(loss_main.detach().cpu())
        self.scheduler.step()
        avg = total_loss / max(len(loader), 1)
        if avg < self.best_loss:
            self.best_loss = avg
            torch.save(self.model.state_dict(), self.checkpoint_dir / "cathode_model.pt")
        return avg


class BMSKaggleTrainer(BMSTrainer):
    def __init__(self, data_dir: str, seq_len: int, lr: float, device: str, checkpoint_dir: Path):
        self.device = torch.device(device)
        self.ode_fn = __import__("training.train_bms", fromlist=["GraphNODEFunction"]).GraphNODEFunction().to(self.device)
        self.risk_head = torch.nn.Sequential(
            torch.nn.Linear(7, 96),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(96, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)
        params = list(self.ode_fn.parameters()) + list(self.risk_head.parameters())
        self.params = params
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.dataset = BMSDataset(data_dir, seq_len=seq_len)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.amp = MixedPrecisionWrapper(enabled=(self.device.type == "cuda"))

    def train_epoch(self, epoch: int) -> float:
        self.ode_fn.train()
        self.risk_head.train()
        total_loss = 0.0
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=3, shuffle=True, num_workers=0)
        for V, T, I, risk_target, fail_cell in loader:
            V = V.to(self.device)
            T = T.to(self.device)
            I = I.to(self.device)
            risk_target = risk_target.to(self.device)
            with self.amp.forward_context():
                batch_size, seq_len, n_cells = V.shape
                loss = torch.tensor(0.0, device=self.device)
                steps = min(80, seq_len)
                indices = torch.linspace(0, seq_len - 1, steps, device=self.device).long()
                for b in range(batch_size):
                    node_state = torch.zeros(n_cells, 7, device=self.device)
                    node_state[:, 0] = V[b, 0]
                    node_state[:, 1] = I[b, 0]
                    node_state[:, 2] = T[b, 0]
                    node_state[:, 3] = 0.8
                    node_state[:, 4] = 1e-9
                    node_state[:, 5] = 0.01
                    node_state[:, 6] = 0.0
                    previous_risk = None
                    for idx in indices:
                        idx_int = int(idx.detach().cpu())
                        dstate = self.ode_fn(float(idx_int) / max(seq_len - 1, 1), node_state)
                        node_state = node_state + 0.01 * torch.clamp(dstate, -10.0, 10.0)
                        node_state[:, 0] = V[b, idx_int]
                        node_state[:, 1] = I[b, idx_int]
                        node_state[:, 2] = T[b, idx_int]
                        pred_risk = self.risk_head(node_state).squeeze(-1)
                        target = risk_target[b, idx_int]
                        mse = torch.nn.functional.mse_loss(pred_risk, target)
                        weighted = mse * (1.0 + 3.0 * target.mean())
                        loss = loss + weighted
                        if previous_risk is not None:
                            loss = loss + 0.02 * torch.mean(torch.relu(previous_risk - pred_risk - 0.10) ** 2)
                        previous_risk = pred_risk
                loss = loss / max(batch_size * steps, 1)
            self.optimizer.zero_grad(set_to_none=True)
            self.amp.backward(loss, self.optimizer, self.params, max_norm=1.0)
            total_loss += float(loss.detach().cpu())
        self.scheduler.step()
        return total_loss / max(len(loader), 1)

    def train(self, epochs: int = 200) -> None:
        best = float("inf")
        for epoch in range(epochs):
            loss = self.train_epoch(epoch)
            if loss < best:
                best = loss
                torch.save({"ode_fn": self.ode_fn.state_dict(), "risk_head": self.risk_head.state_dict()}, self.checkpoint_dir / "bms_graph_node.pt")
            print(json.dumps({"task": "bms", "epoch": epoch + 1, "epochs": epochs, "loss": loss, "best": best}))


class RecyclingKaggleTrainer(RecyclingTrainer):
    def __init__(self, data_path: str, lr: float, device: str, checkpoint_dir: Path):
        super().__init__(data_path=data_path, lr=lr, device=device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.amp = MixedPrecisionWrapper(enabled=(self.device.type == "cuda"))

    def train(self, epochs: int = 100) -> None:
        best = float("inf")
        for epoch in range(epochs):
            self.model.train()
            n_cond = self.conditions.shape[0]
            idx = torch.randperm(n_cond, device=self.device)[: min(48, n_cond)]
            cond_batch = self.conditions[idx]
            target_batch = self.trajectories[idx]
            with self.amp.forward_context():
                alpha = torch.zeros(len(idx), 3, device=self.device)
                loss = torch.tensor(0.0, device=self.device)
                horizon = target_batch.shape[2]
                for t_step in range(horizon):
                    t_val = torch.tensor(float(t_step), device=self.device)
                    da = torch.clamp(self.model(t_val, alpha, cond_batch), 0.0, 0.08)
                    alpha = torch.clamp(alpha + da, 0.0, 1.0)
                    loss = loss + torch.nn.functional.mse_loss(alpha, target_batch[:, :, t_step])
                loss = loss / horizon
                monotonic_penalty = torch.mean(torch.relu(-alpha) ** 2)
                total = loss + monotonic_penalty
            self.optimizer.zero_grad(set_to_none=True)
            self.amp.backward(total, self.optimizer, self.model.parameters(), max_norm=1.0)
            value = float(loss.detach().cpu())
            if value < best:
                best = value
                torch.save(self.model.state_dict(), self.checkpoint_dir / "recycling_ode.pt")
            print(json.dumps({"task": "recycling", "epoch": epoch + 1, "epochs": epochs, "loss": value, "best": best}))


class MixedPrecisionWrapper:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled) if self.enabled else None

    def forward_context(self):
        if self.enabled:
            return torch.cuda.amp.autocast()
        import contextlib
        return contextlib.nullcontext()

    def backward(self, loss, optimizer, parameters, max_norm=1.0):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
            optimizer.step()


class MetricsCollector:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            diff = pred - target
            mae = torch.mean(torch.abs(diff)).item()
            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            mape = torch.mean(torch.abs(diff) / (torch.abs(target) + 1e-8)).item() * 100
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def log(self, task: str, epoch: int, metrics: Dict[str, float]):
        entry = {"task": task, "epoch": epoch, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        entry.update(metrics)
        self.records.append(entry)

    def save(self, path: Path):
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


class CheckpointMetadata:
    @staticmethod
    def collect(root: Path) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        manifest_path = root / "data" / "cache" / "hyper_manifest_foundation.json"
        if manifest_path.exists():
            content = manifest_path.read_bytes()
            meta["manifest_hash"] = hashlib.sha256(content).hexdigest()[:16]
        else:
            meta["manifest_hash"] = "missing"
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                                    capture_output=True, text=True, timeout=5)
            meta["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
            result2 = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                                     capture_output=True, text=True, timeout=5)
            meta["workspace_dirty"] = bool(result2.stdout.strip()) if result2.returncode == 0 else None
        except Exception:
            meta["git_commit"] = "unavailable"
            meta["workspace_dirty"] = None
        return meta


class StratifiedSplitter:
    def __init__(self, val_fraction: float = 0.15, seed: int = 42):
        self.val_frac = val_fraction
        self.rng = np.random.RandomState(seed)

    def split_by_key(self, keys: np.ndarray) -> tuple:
        unique = np.unique(keys)
        self.rng.shuffle(unique)
        n_val = max(1, int(len(unique) * self.val_frac))
        val_keys = set(unique[:n_val])
        train_idx = [i for i, k in enumerate(keys) if k not in val_keys]
        val_idx = [i for i, k in enumerate(keys) if k in val_keys]
        return np.array(train_idx), np.array(val_idx)


class TrainingOrchestrator:
    def __init__(self, root: Path, profile: TrainProfile, task: str, device: str, seed: int, checkpoint_dir: Optional[Path], resume: bool = False):
        self.root = root
        self.profile = profile
        self.task = task
        self.device = resolve_device(device)
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir or (root / "checkpoints")
        self.ledger = RunLedger(root, f"{task}_{profile.name}_{int(time.time())}")
        self.metrics = MetricsCollector()
        self.amp = MixedPrecisionWrapper(enabled=(self.device == "cuda"))
        self.resume = resume
        set_seed(seed)

    def verify(self) -> None:
        env = EnvironmentReport(self.root).as_dict()
        data = DataAvailability(self.root).as_dict()
        self.ledger.event("environment", env)
        self.ledger.event("data", data)
        missing = []
        if data["cathode_npz"] == 0:
            missing.append("data/synthetic/cathode/cathode_*.npz")
        if data["bms_npz"] == 0:
            missing.append("data/synthetic/bms/bms_*.npz")
        if not data["leaching_grid_exists"]:
            missing.append("data/synthetic/leaching/leaching_grid.npz")
        if missing:
            raise FileNotFoundError("Missing training data: " + ", ".join(missing))

    def train_cathode(self) -> None:
        trainer = CathodeKaggleTrainer(
            data_dir=str(self.root / "data/synthetic/cathode"),
            max_files=self.profile.cathode_max_files,
            lr=8e-4,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.ledger.event("cathode_start", {"epochs": self.profile.cathode_epochs, "files": len(trainer.dataset)})
        for epoch in range(self.profile.cathode_epochs):
            loss = trainer.train_epoch(epoch)
            self.ledger.event("cathode_epoch", {"epoch": epoch + 1, "loss": loss, "best_loss": trainer.best_loss})
            print(json.dumps({"task": "cathode", "epoch": epoch + 1, "epochs": self.profile.cathode_epochs, "loss": loss, "best": trainer.best_loss}))

    def train_bms(self) -> None:
        trainer = BMSKaggleTrainer(
            data_dir=str(self.root / "data/synthetic/bms"),
            seq_len=self.profile.bms_seq_len,
            lr=5e-4,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.ledger.event("bms_start", {"epochs": self.profile.bms_epochs, "files": len(trainer.dataset)})
        trainer.train(epochs=self.profile.bms_epochs)
        self.ledger.event("bms_end", {"epochs": self.profile.bms_epochs})

    def train_recycling(self) -> None:
        trainer = RecyclingKaggleTrainer(
            data_path=str(self.root / "data/synthetic/leaching/leaching_grid.npz"),
            lr=7e-4,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.ledger.event("recycling_start", {"epochs": self.profile.recycling_epochs, "conditions": int(trainer.conditions.shape[0])})
        trainer.train(epochs=self.profile.recycling_epochs)
        self.ledger.event("recycling_end", {"epochs": self.profile.recycling_epochs})

    def run(self) -> None:
        self.verify()
        tasks = ["cathode", "bms", "recycling"] if self.task == "all" else [self.task]
        for task in tasks:
            if task == "cathode":
                self.train_cathode()
            elif task == "bms":
                self.train_bms()
            elif task == "recycling":
                self.train_recycling()
            else:
                raise ValueError(task)
        ckpt_meta = CheckpointMetadata.collect(self.root)
        self.ledger.event("complete", {"task": self.task, "profile": self.profile.name,
                                        "checkpoint_dir": str(self.checkpoint_dir),
                                        "checkpoint_metadata": ckpt_meta})
        self.metrics.save(self.checkpoint_dir / "metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--task", choices=["cathode", "bms", "recycling", "all"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    profile = PROFILES[args.profile]
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    orchestrator = TrainingOrchestrator(root, profile, args.task, args.device, args.seed, checkpoint_dir, resume=args.resume)
    orchestrator.run()


if __name__ == "__main__":
    main()
