import torch
import torch.nn as nn
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


@dataclass
class DistributionalPrediction:
    mean: float
    variance: float
    lower_95: float
    upper_95: float
    source: str

    @classmethod
    def from_mean_variance(cls, mean: float, variance: float, source: str = "model") -> "DistributionalPrediction":
        var = max(float(variance), 0.0)
        width = 1.96 * math.sqrt(var)
        return cls(mean=float(mean), variance=var, lower_95=float(mean - width), upper_95=float(mean + width), source=source)


class DeepEnsemblePredictor(nn.Module):
    def __init__(self, base_model_fn, n_models: int = 5):
        super().__init__()
        self.models = nn.ModuleList([base_model_fn() for _ in range(n_models)])

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack([m(*args, **kwargs) for m in self.models])
        return preds.mean(dim=0), preds.var(dim=0)

    def predict_with_disagreement(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        preds = torch.stack([m(*args, **kwargs) for m in self.models])
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)
        pairwise_var = torch.mean(torch.stack([
            (preds[i] - preds[j]) ** 2
            for i in range(len(self.models)) for j in range(i + 1, len(self.models))
        ]), dim=0)
        return {"mean": mean, "variance": var, "epistemic": var, "disagreement": pairwise_var}


class MCDropoutEstimator(nn.Module):
    def __init__(self, model: nn.Module, n_samples: int = 30):
        super().__init__()
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self):
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self._enable_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(self.model(*args, **kwargs))
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.var(dim=0)

    def entropy(self, *args, **kwargs) -> torch.Tensor:
        mean, var = self.forward(*args, **kwargs)
        return 0.5 * torch.log(2 * math.pi * math.e * var + 1e-12)


class EvidentialRegression(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, 1)
        self.nu_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        gamma = self.gamma_head(h)
        nu = nn.functional.softplus(self.nu_head(h)) + 1e-6
        alpha = nn.functional.softplus(self.alpha_head(h)) + 1.0
        beta = nn.functional.softplus(self.beta_head(h)) + 1e-6
        return gamma, nu, alpha, beta

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        gamma, nu, alpha, beta = self.forward(x)
        aleatoric = beta / (alpha - 1.0 + 1e-6)
        epistemic = aleatoric / (nu + 1e-6)
        return {
            "mean": gamma.squeeze(-1),
            "aleatoric": aleatoric.squeeze(-1),
            "epistemic": epistemic.squeeze(-1),
            "total": (aleatoric + epistemic).squeeze(-1),
        }

    def nig_loss(self, x: torch.Tensor, y: torch.Tensor, lam: float = 0.01) -> torch.Tensor:
        gamma, nu, alpha, beta = self.forward(x)
        y = y.unsqueeze(-1) if y.dim() < gamma.dim() else y
        omega = 2.0 * beta * (1.0 + nu)
        nll = 0.5 * torch.log(math.pi / (nu + 1e-6)) \
              - alpha * torch.log(omega + 1e-6) \
              + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega + 1e-6) \
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        reg = lam * (2.0 * nu + alpha) * torch.abs(y - gamma)
        return torch.mean(nll + reg)


class ConformalPredictor:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.q_hat: float = 0.0

    def calibrate(self, y_pred: np.ndarray, y_true: np.ndarray):
        scores = np.abs(y_pred - y_true)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = float(np.quantile(scores, q_level))
        self.calibration_scores = scores

    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return y_pred - self.q_hat, y_pred + self.q_hat

    def adaptive_intervals(self, y_pred: np.ndarray, uncertainty: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.calibration_scores is None:
            return self.predict_intervals(y_pred)
        normalized_scores = self.calibration_scores / (uncertainty[:len(self.calibration_scores)] + 1e-8)
        n = len(normalized_scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        q_hat_norm = float(np.quantile(normalized_scores, q_level))
        width = q_hat_norm * uncertainty
        return y_pred - width, y_pred + width

    def coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        return float(np.mean((y_true >= lower) & (y_true <= upper)))


class CalibrationCurve:
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def compute(self, predicted_var: np.ndarray, squared_error: np.ndarray) -> Dict:
        sorted_idx = np.argsort(predicted_var)
        predicted_var = predicted_var[sorted_idx]
        squared_error = squared_error[sorted_idx]
        bin_edges = np.linspace(0, len(predicted_var), self.n_bins + 1, dtype=int)
        bin_predicted = []
        bin_observed = []
        for i in range(self.n_bins):
            start, end = bin_edges[i], bin_edges[i + 1]
            if end <= start:
                continue
            bin_predicted.append(float(np.mean(predicted_var[start:end])))
            bin_observed.append(float(np.mean(squared_error[start:end])))
        predicted_arr = np.array(bin_predicted)
        observed_arr = np.array(bin_observed)
        ece = float(np.mean(np.abs(predicted_arr - observed_arr))) if len(predicted_arr) > 0 else 999.0
        sharpness = float(np.mean(predicted_var))
        return {
            "bin_predicted": predicted_arr.tolist(),
            "bin_observed": observed_arr.tolist(),
            "ece": ece,
            "sharpness": sharpness,
        }


class UncertaintyDecomposer:
    @staticmethod
    def from_ensemble(predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = predictions.mean(dim=0)
        per_model_var = predictions.var(dim=-1) if predictions.dim() > 2 else torch.zeros_like(mean)
        aleatoric = per_model_var.mean(dim=0)
        epistemic = predictions.mean(dim=-1).var(dim=0) if predictions.dim() > 2 else predictions.var(dim=0)
        return {"mean": mean, "aleatoric": aleatoric, "epistemic": epistemic, "total": aleatoric + epistemic}

    @staticmethod
    def mean_variance_tuple(prediction: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = prediction.get("mean")
        variance = prediction.get("total", None)
        if variance is None:
            variance = prediction.get("variance", None)
        if variance is None:
            aleatoric = prediction.get("aleatoric", torch.zeros_like(mean))
            epistemic = prediction.get("epistemic", torch.zeros_like(mean))
            variance = aleatoric + epistemic
        return mean, torch.clamp(variance, min=0.0)


class UncertaintyPropagation:
    @staticmethod
    def scalar(mean: float, variance: float, source: str) -> Dict[str, float]:
        pred = DistributionalPrediction.from_mean_variance(mean, variance, source)
        return pred.__dict__

    @staticmethod
    def add_independent(*items: Tuple[float, float], source: str = "quadrature") -> Dict[str, float]:
        mean = sum(float(x[0]) for x in items)
        variance = sum(max(float(x[1]), 0.0) for x in items)
        return UncertaintyPropagation.scalar(mean, variance, source)

    @staticmethod
    def product(a_mean: float, a_var: float, b_mean: float, b_var: float, source: str = "first_order_delta") -> Dict[str, float]:
        mean = float(a_mean) * float(b_mean)
        variance = (float(b_mean) ** 2) * max(float(a_var), 0.0) + (float(a_mean) ** 2) * max(float(b_var), 0.0) + max(float(a_var), 0.0) * max(float(b_var), 0.0)
        return UncertaintyPropagation.scalar(mean, variance, source)

    @staticmethod
    def interval_from_relative(mean: float, relative_std: float, source: str) -> Dict[str, float]:
        variance = (abs(float(mean)) * max(float(relative_std), 0.0)) ** 2
        return UncertaintyPropagation.scalar(mean, variance, source)


class ActiveLearningSelector:
    def __init__(self, strategy: str = "max_uncertainty"):
        self.strategy = strategy

    def select(self, candidate_uncertainties: np.ndarray, n_select: int = 5) -> np.ndarray:
        if self.strategy == "max_uncertainty":
            return np.argsort(candidate_uncertainties)[-n_select:][::-1]
        elif self.strategy == "diverse_uncertain":
            selected = []
            remaining = set(range(len(candidate_uncertainties)))
            for _ in range(min(n_select, len(candidate_uncertainties))):
                best = max(remaining, key=lambda i: candidate_uncertainties[i])
                selected.append(best)
                remaining.discard(best)
                if not remaining:
                    break
                for r in list(remaining):
                    if abs(r - best) < 3:
                        remaining.discard(r)
            return np.array(selected)
        return np.argsort(candidate_uncertainties)[-n_select:][::-1]


class BayesianLastLayer(nn.Module):
    def __init__(self, feature_extractor: nn.Module, feature_dim: int, output_dim: int = 1):
        super().__init__()
        self.backbone = feature_extractor
        self.mu_w = nn.Parameter(torch.zeros(output_dim, feature_dim))
        self.log_sigma_w = nn.Parameter(torch.zeros(output_dim, feature_dim) - 3.0)
        self.mu_b = nn.Parameter(torch.zeros(output_dim))
        self.log_sigma_b = nn.Parameter(torch.zeros(output_dim) - 3.0)

    def forward(self, x: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.backbone(x)
        preds = []
        for _ in range(n_samples):
            w = self.mu_w + torch.exp(self.log_sigma_w) * torch.randn_like(self.mu_w)
            b = self.mu_b + torch.exp(self.log_sigma_b) * torch.randn_like(self.mu_b)
            preds.append(features @ w.T + b)
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.var(dim=0)

    def kl_divergence(self) -> torch.Tensor:
        sigma_w = torch.exp(self.log_sigma_w)
        kl_w = 0.5 * torch.sum(sigma_w ** 2 + self.mu_w ** 2 - 1.0 - 2.0 * self.log_sigma_w)
        sigma_b = torch.exp(self.log_sigma_b)
        kl_b = 0.5 * torch.sum(sigma_b ** 2 + self.mu_b ** 2 - 1.0 - 2.0 * self.log_sigma_b)
        return kl_w + kl_b
