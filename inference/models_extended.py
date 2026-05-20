"""Extended KineticsForge inference models (M11-M14).

These definitions intentionally mirror the Kaggle V5 training cells and
checkpoint key names. Earlier documentation proposed richer output heads, but
the shipped checkpoints in results (25).zip were trained with the compact
interfaces below, so inference must match them exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_FEAT = 27
COND_DIM = 5
WINDOW = 20


class ElectrolyteHealthModel(nn.Module):
    """M11: EIS-derived electrolyte degradation and plating risk.

    Input: [R_ohm, R_ct, R_sei, sigma_w, T_scaled, SOC, cycle_frac].
    Output: raw degradation logit, raw plating logit, safe C-rate.
    """

    def __init__(self, in_dim: int = 7, hd: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hd),
            nn.GELU(),
            nn.LayerNorm(hd),
            nn.Linear(hd, hd),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hd, 48),
        )
        self.deg_head = nn.Linear(48, 1)
        self.plat_head = nn.Linear(48, 1)
        self.cr_head = nn.Sequential(
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        degradation = self.deg_head(z).squeeze(-1)
        plating = self.plat_head(z).squeeze(-1)
        safe_crate = torch.clamp(self.cr_head(z).squeeze(-1), 0.05, 5.0)
        return degradation, plating, safe_crate


class ReplenishabilityModel(nn.Module):
    """M12: recovery/reformation score from a capacity window and 10 features."""

    def __init__(self, hist_dim: int = WINDOW, feat_dim: int = 10, hd: int = 64):
        super().__init__()
        self.he = nn.Sequential(nn.Linear(hist_dim, hd), nn.GELU(), nn.Linear(hd, 32))
        self.fe = nn.Sequential(nn.Linear(feat_dim, hd), nn.GELU(), nn.LayerNorm(hd), nn.Linear(hd, 32))
        self.head = nn.Sequential(nn.Linear(64, 48), nn.GELU(), nn.Dropout(0.1), nn.Linear(48, 2))

    def forward(self, hist: torch.Tensor, feats: torch.Tensor, *unused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(torch.cat([self.he(hist), self.fe(feats)], dim=-1))
        return out[:, 0], out[:, 1]


class ChemIdentifier(nn.Module):
    """M13: cathode chemistry classifier.

    The V5 checkpoint was trained on 27 averaged early-cycle features plus four
    condition values: [T/50, C-rate, DOD, form_factor_or_aux].
    """

    N_CLASSES = 8

    def __init__(self, in_dim: int = N_FEAT, n_classes: int = N_CLASSES, hd: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 16 + 4, hd),
            nn.GELU(),
            nn.LayerNorm(hd),
            nn.Dropout(0.15),
            nn.Linear(hd, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, feats: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.conv(feats.unsqueeze(1)).flatten(1)
        return self.head(torch.cat([z, cond], dim=-1))


class FormationProtocolModel(nn.Module):
    """M14: formation quality model.

    Output heads are normalized life index, robustness index, and SEI quality.
    Protocol suggestions are derived in the product layer from these scores.
    """

    def __init__(self, in_dim: int = N_FEAT, cond_dim: int = COND_DIM, hd: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hd), nn.GELU(), nn.LayerNorm(hd), nn.Linear(hd, 64), nn.GELU())
        self.cond_enc = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3))

    def forward(self, feats: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.head(torch.cat([self.enc(feats), self.cond_enc(cond)], dim=-1))
        return out[:, 0], out[:, 1], out[:, 2]


EXTENDED_MODEL_CLASSES = {
    "M11_ElectrolyteHealth": ElectrolyteHealthModel,
    "M12_Replenishability": ReplenishabilityModel,
    "M13_ChemIdentifier": ChemIdentifier,
    "M14_FormationProtocol": FormationProtocolModel,
}

EXTENDED_CHECKPOINT_NAMES = {
    "M11_ElectrolyteHealth": "electrolyte_health",
    "M12_Replenishability": "replenishability",
    "M13_ChemIdentifier": "chem_identifier",
    "M14_FormationProtocol": "formation_protocol",
}
