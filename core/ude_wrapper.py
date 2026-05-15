from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn


TensorTerm = Callable[[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]


@dataclass
class UDEContribution:
    name: str
    rms: float
    fraction: float


class UniversalDifferentialEquation(nn.Module):
    def __init__(
        self,
        physics_terms: Optional[Dict[str, TensorTerm]] = None,
        residual_net: Optional[nn.Module] = None,
        residual_scale: float = 0.10,
    ):
        super().__init__()
        self.physics_terms = physics_terms or {}
        self.residual_net = residual_net
        self.logit_residual_mix = nn.Parameter(torch.logit(torch.tensor(float(residual_scale)).clamp(1e-4, 0.95)))

    def forward(self, t: torch.Tensor, state: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context = context or {}
        pieces: Dict[str, torch.Tensor] = {}
        total = torch.zeros_like(state)
        for name, term in self.physics_terms.items():
            value = term(t, state, context)
            pieces[name] = value
            total = total + value
        if self.residual_net is not None:
            residual = self.residual_net(state, context.get("z_comp"), t)
            mix = torch.sigmoid(self.logit_residual_mix)
            pieces["neural_residual"] = mix * residual
            total = total + pieces["neural_residual"]
        return total, pieces

    @staticmethod
    def contribution_report(pieces: Dict[str, torch.Tensor]) -> list[UDEContribution]:
        rms = {k: float(torch.sqrt(torch.mean(v.detach() ** 2)).cpu()) for k, v in pieces.items()}
        denom = sum(rms.values()) + 1e-12
        return [UDEContribution(name=k, rms=v, fraction=v / denom) for k, v in sorted(rms.items())]


class PhysicsResidualAudit:
    def __init__(self, min_physics_fraction: float = 0.50):
        self.min_physics_fraction = min_physics_fraction

    def evaluate(self, contributions: Iterable[UDEContribution]) -> Dict[str, object]:
        items = list(contributions)
        residual = sum(x.fraction for x in items if "residual" in x.name.lower() or "neural" in x.name.lower())
        physics = max(0.0, 1.0 - residual)
        return {
            "physics_fraction": float(physics),
            "residual_fraction": float(residual),
            "passed": bool(physics >= self.min_physics_fraction),
            "contributions": [x.__dict__ for x in items],
            "minimum_physics_fraction": float(self.min_physics_fraction),
        }
