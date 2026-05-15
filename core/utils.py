import torch
import torch.nn as nn
import numpy as np
import os
import json
import logging
import time
import hashlib
from pathlib import Path
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'
SYNTH_DIR = PROJECT_ROOT / 'data' / 'synthetic'
REAL_DIR = PROJECT_ROOT / 'data' / 'real'

def ensure_dirs():
    for d in [CHECKPOINT_DIR, CACHE_DIR, SYNTH_DIR, REAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()


class CheckpointManager:
    def __init__(self, save_dir, model_name, max_keep=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_keep = max_keep
        self.history_file = self.save_dir / f'{model_name}_history.json'
        self.history = self._load_history()

    def _load_history(self):
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'best_loss': float('inf'), 'best_path': None}

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def save(self, model, optimizer, epoch, loss, metrics=None, scheduler=None):
        ckpt_name = f'{self.model_name}_epoch{epoch:04d}_loss{loss:.6f}.pt'
        ckpt_path = self.save_dir / ckpt_name

        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics or {},
            'timestamp': time.time()
        }
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(state, ckpt_path)
        self.history['checkpoints'].append({
            'path': str(ckpt_path),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics or {}
        })

        is_best = False
        if loss < self.history['best_loss']:
            self.history['best_loss'] = loss
            best_path = self.save_dir / f'{self.model_name}_best.pt'
            torch.save(state, best_path)
            self.history['best_path'] = str(best_path)
            is_best = True

        while len(self.history['checkpoints']) > self.max_keep:
            old = self.history['checkpoints'].pop(0)
            old_path = Path(old['path'])
            if old_path.exists() and str(old_path) != self.history.get('best_path'):
                old_path.unlink()

        self._save_history()
        return is_best

    def load_best(self, model, optimizer=None, scheduler=None):
        if self.history['best_path'] is None or not Path(self.history['best_path']).exists():
            log.warning(f"No best checkpoint found for {self.model_name}")
            return 0
        return self._load(self.history['best_path'], model, optimizer, scheduler)

    def load_latest(self, model, optimizer=None, scheduler=None):
        if not self.history['checkpoints']:
            log.warning(f"No checkpoints found for {self.model_name}")
            return 0
        latest = self.history['checkpoints'][-1]
        return self._load(latest['path'], model, optimizer, scheduler)

    def _load(self, path, model, optimizer=None, scheduler=None):
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        log.info(f"Loaded checkpoint from {path} (epoch {state['epoch']}, loss {state['loss']:.6f})")
        return state['epoch']


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class CosineAnnealingWarmRestarts:
    def __init__(self, optimizer, T_0, T_mult=2, eta_min=1e-6):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)

        for i, pg in enumerate(self.optimizer.param_groups):
            lr = self.eta_min + (self.base_lrs[i] - self.eta_min) * (
                1 + np.cos(np.pi * self.T_cur / self.T_i)
            ) / 2
            pg['lr'] = lr


class GradientClipper:
    def __init__(self, max_norm=1.0, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_history = []

    def clip(self, model):
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, norm_type=self.norm_type
        )
        self.grad_history.append(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)
        return total_norm

    def get_stats(self):
        if not self.grad_history:
            return {}
        arr = np.array(self.grad_history[-100:])
        return {
            'grad_norm_mean': float(arr.mean()),
            'grad_norm_max': float(arr.max()),
            'grad_norm_min': float(arr.min()),
            'grad_norm_std': float(arr.std()),
            'clipped_fraction': float(np.mean(arr > self.max_norm))
        }


class MetricsTracker:
    def __init__(self):
        self.metrics = OrderedDict()
        self.epoch_data = []

    def update(self, name, value, step=None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({'value': value, 'step': step, 'time': time.time()})

    def log_epoch(self, epoch, **kwargs):
        entry = {'epoch': epoch, **kwargs}
        self.epoch_data.append(entry)
        parts = [f"Epoch {epoch:04d}"]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")
        log.info(" | ".join(parts))

    def get_series(self, name):
        if name not in self.metrics:
            return [], []
        values = [m['value'] for m in self.metrics[name]]
        steps = [m['step'] for m in self.metrics[name]]
        return steps, values

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'metrics': {k: [{'value': m['value'], 'step': m['step']} for m in v]
                            for k, v in self.metrics.items()},
                'epochs': self.epoch_data
            }, f, indent=2)


def compute_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def model_summary(model, input_size=None):
    info = compute_parameter_count(model)
    log.info(f"Model: {model.__class__.__name__}")
    log.info(f"  Total parameters: {info['total']:,}")
    log.info(f"  Trainable: {info['trainable']:,}")
    log.info(f"  Frozen: {info['frozen']:,}")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        log.info(f"  {name}: {module.__class__.__name__} ({n_params:,} params)")
    return info


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        log.info("Using CPU")
    return device


class DataNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0) + 1e-8
            self.min_val = data.min(dim=0)[0]
            self.max_val = data.max(dim=0)[0]
        else:
            data = np.array(data)
            self.mean = torch.tensor(data.mean(axis=0), dtype=torch.float32)
            self.std = torch.tensor(data.std(axis=0) + 1e-8, dtype=torch.float32)
            self.min_val = torch.tensor(data.min(axis=0), dtype=torch.float32)
            self.max_val = torch.tensor(data.max(axis=0) + 1e-8, dtype=torch.float32)

    def normalize(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def denormalize(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def minmax_normalize(self, data):
        return (data - self.min_val.to(data.device)) / (self.max_val.to(data.device) - self.min_val.to(data.device) + 1e-8)

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'min_val': self.min_val, 'max_val': self.max_val}

    def load_state_dict(self, state):
        self.mean = state['mean']
        self.std = state['std']
        self.min_val = state['min_val']
        self.max_val = state['max_val']


class PhysicsInformedLoss:
    def __init__(self, data_weight=1.0, physics_weight=0.1, lyap_weight=0.05,
                 koopman_weight=0.1, sindy_weight=0.05, monotonicity_weight=0.1):
        self.w_data = data_weight
        self.w_phys = physics_weight
        self.w_lyap = lyap_weight
        self.w_koop = koopman_weight
        self.w_sindy = sindy_weight
        self.w_mono = monotonicity_weight

    def data_loss(self, pred, target):
        return torch.mean((pred - target)**2)

    def physics_consistency_loss(self, pred_trajectory, T, composition_params):
        capacity = pred_trajectory[..., 0]
        dQ = capacity[..., 1:] - capacity[..., :-1]
        violation = torch.relu(dQ)
        return torch.mean(violation)

    def monotonicity_loss(self, pred_capacity):
        dQ = pred_capacity[..., 1:] - pred_capacity[..., :-1]
        return torch.mean(torch.relu(dQ))

    def arrhenius_consistency_loss(self, k_values, temperatures, Ea_bounds=(0.3, 1.2)):
        if len(k_values) < 2:
            return torch.tensor(0.0)
        ln_k = torch.log(k_values + 1e-30)
        inv_T = 1.0 / temperatures
        if len(ln_k) > 2:
            slope = (ln_k[-1] - ln_k[0]) / (inv_T[-1] - inv_T[0] + 1e-30)
            Ea_estimated = -slope * 8.617e-5
            lower_violation = torch.relu(Ea_bounds[0] - Ea_estimated)
            upper_violation = torch.relu(Ea_estimated - Ea_bounds[1])
            return lower_violation + upper_violation
        return torch.tensor(0.0)

    def koopman_linearity_loss(self, encoder, decoder, K_matrix, z_t, z_t1):
        phi_t = encoder(z_t)
        phi_t1 = encoder(z_t1)
        K_phi_t = torch.matmul(phi_t, K_matrix.T)
        linearity = torch.mean((K_phi_t - phi_t1)**2)
        recon_t = decoder(phi_t)
        recon = torch.mean((recon_t - z_t)**2)
        return linearity + 0.5 * recon

    def combined_loss(self, pred, target, pred_trajectory=None, capacity=None,
                      encoder=None, decoder=None, K=None, z_t=None, z_t1=None,
                      lyap_loss_val=None, sindy_residual=None):
        total = self.w_data * self.data_loss(pred, target)

        if pred_trajectory is not None:
            total = total + self.w_phys * self.physics_consistency_loss(pred_trajectory, None, None)

        if capacity is not None:
            total = total + self.w_mono * self.monotonicity_loss(capacity)

        if encoder is not None and K is not None and z_t is not None:
            total = total + self.w_koop * self.koopman_linearity_loss(encoder, decoder, K, z_t, z_t1)

        if lyap_loss_val is not None:
            total = total + self.w_lyap * lyap_loss_val

        if sindy_residual is not None:
            total = total + self.w_sindy * torch.mean(sindy_residual**2)

        return total


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.target_lr * (self.step_count / self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr


class TrainingTimer:
    def __init__(self):
        self.start_time = None
        self.epoch_times = []

    def start(self):
        self.start_time = time.time()

    def epoch_end(self):
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed)
        self.start_time = time.time()
        return elapsed

    def estimate_remaining(self, current_epoch, total_epochs):
        if not self.epoch_times:
            return 0
        avg_time = np.mean(self.epoch_times[-10:])
        remaining = (total_epochs - current_epoch) * avg_time
        return remaining

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def hash_config(config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def count_nan_inf(tensor):
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    return n_nan, n_inf


def safe_divide(a, b, default=0.0):
    if isinstance(a, torch.Tensor):
        mask = torch.abs(b) < 1e-30
        result = a / (b + 1e-30)
        result[mask] = default
        return result
    else:
        if abs(b) < 1e-30:
            return default
        return a / b


class FeatureImportanceTracker:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.importances = {name: [] for name in feature_names}

    def update_from_gradients(self, model, x, y, loss_fn):
        x_req = x.clone().requires_grad_(True)
        pred = model(x_req)
        loss = loss_fn(pred, y)
        loss.backward()
        grad_importance = torch.abs(x_req.grad).mean(dim=0)
        for i, name in enumerate(self.feature_names):
            if i < grad_importance.shape[0]:
                self.importances[name].append(grad_importance[i].item())

    def get_rankings(self):
        avg_importance = {}
        for name in self.feature_names:
            if self.importances[name]:
                avg_importance[name] = np.mean(self.importances[name])
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features


class BatchGenerator:
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data) if isinstance(data, list) else data.shape[0]

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            if isinstance(self.data, list):
                yield [self.data[i] for i in batch_idx]
            elif isinstance(self.data, torch.Tensor):
                yield self.data[batch_idx]
            elif isinstance(self.data, np.ndarray):
                yield self.data[batch_idx]
            elif isinstance(self.data, tuple):
                yield tuple(d[batch_idx] for d in self.data)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class NumericalStabilityGuard:
    def __init__(self, model, check_interval=10):
        self.model = model
        self.check_interval = check_interval
        self.step_count = 0
        self.issues = []

    def check(self):
        self.step_count += 1
        if self.step_count % self.check_interval != 0:
            return True
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                self.issues.append(f"NaN in {name}")
                return False
            if torch.isinf(param).any():
                self.issues.append(f"Inf in {name}")
                return False
            if param.abs().max() > 1e6:
                self.issues.append(f"Very large values in {name}: {param.abs().max().item()}")
        return True

    def fix_nans(self):
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nn.init.xavier_normal_(param)
                log.warning(f"Reset {name} due to NaN/Inf")
