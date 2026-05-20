"""KineticsForge V4 — Unified model definitions for inference.

These mirror the architectures from the Kaggle training cells (phase2_acct1/2/3_mega.py)
so trained checkpoints can be loaded directly.
"""
import torch
import torch.nn as nn

N_FEAT = 27
COND_DIM = 5
WINDOW = 20
WINDOW_30 = 30
MAX_SEQ = 600
N_CELLS_SIM = 4


# ── M1: Cathode UDE ─────────────────────────────
class CathodeUDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_embed = nn.Sequential(nn.Linear(COND_DIM, 64), nn.GELU(), nn.Linear(64, 48))
        self.feat_embed = nn.Sequential(nn.Linear(N_FEAT, 64), nn.GELU(), nn.Linear(64, 32))
        self.gate = nn.Sequential(nn.Linear(2+48+32+1, 128), nn.GELU(), nn.Linear(128, 2), nn.Sigmoid())
        self.neural = nn.Sequential(nn.Linear(2+48+32+1, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))
        self.sei_k = nn.Parameter(torch.tensor(-9.0))

    def forward(self, t, state, z, fz):
        Q, R = state[..., 0:1], state[..., 1:2]
        tv = t * torch.ones_like(Q)
        inp = torch.cat([state, z, fz, tv], dim=-1)
        g = self.gate(inp)
        nn_out = self.neural(inp)
        dQ_phys = -torch.exp(self.sei_k) * Q * (0.01 + 0.001 * Q.abs())
        dQ = g[..., 0:1] * dQ_phys + (1 - g[..., 0:1]) * nn_out[..., 0:1]
        dR = nn_out[..., 1:2]
        return torch.cat([dQ, dR], dim=-1)


# ── M2: SOH Estimator ───────────────────────────
class SOHModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW, 128), nn.GELU(), nn.Linear(128, 64))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT, 64), nn.GELU(), nn.Linear(64, 32))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM+1, 32), nn.GELU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, hist, feats, cond, cf):
        h = self.hist(hist)
        fe = self.feat_enc(feats)
        c = self.cond_enc(torch.cat([cond, cf.unsqueeze(-1)], dim=-1))
        return self.head(torch.cat([h, fe, c], dim=-1)).squeeze(-1)


# ── M3: Cycle Life Classifier ───────────────────
class CycleLifeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Sequential(nn.Linear(min(100, MAX_SEQ), 128), nn.GELU(), nn.Linear(128, 64))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT, 32), nn.GELU(), nn.Linear(32, 32))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM, 32), nn.GELU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 4))

    def forward(self, early_cap, early_feat, cond):
        h = self.early(early_cap)
        f = self.feat_enc(early_feat)
        c = self.cond_enc(cond)
        return self.head(torch.cat([h, f, c], dim=-1))


# ── M4: Fade Rate Predictor ─────────────────────
class FadeRateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW, 64), nn.GELU(), nn.Linear(64, 32))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT, 32), nn.GELU(), nn.Linear(32, 16))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM, 16), nn.GELU(), nn.Linear(16, 16))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, hist, feat, cond):
        h = self.hist(hist)
        f = self.feat_enc(feat)
        c = self.cond_enc(cond)
        return self.head(torch.cat([h, f, c], dim=-1)).squeeze(-1)


# ── M5: BMS Pack TGN ────────────────────────────
class PackTGN(nn.Module):
    def __init__(self, nd=7, ed=3, h=64, nc=N_CELLS_SIM):
        super().__init__()
        self.nc = nc
        self.msg = nn.Sequential(nn.Linear(nd*2+ed, h), nn.LeakyReLU(0.2), nn.Linear(h, 32))
        self.upd = nn.Sequential(nn.Linear(nd+32, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, nd))
        edges = [[i, j] for i in range(nc) for j in range(nc) if i != j]
        self.register_buffer("ei", torch.tensor(edges, dtype=torch.long).T)
        self.ea = nn.Parameter(torch.randn(len(edges), ed) * 0.1)
        self.risk = nn.Sequential(nn.Linear(nd, 64), nn.GELU(), nn.Dropout(0.05), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, t, x):
        r, c = self.ei
        msgs = self.msg(torch.cat([x[r], x[c], self.ea], dim=-1))
        agg = torch.zeros(x.shape[0], 32, device=x.device, dtype=msgs.dtype)
        agg.scatter_add_(0, c.unsqueeze(-1).expand_as(msgs), msgs)
        x2 = self.upd(torch.cat([x, agg], dim=-1))
        return x2, self.risk(x2).squeeze(-1)


# ── M6: RUL Predictor ───────────────────────────
class RULModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW, 128), nn.GELU(), nn.Linear(128, 64))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT, 64), nn.GELU(), nn.Linear(64, 32))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM+1, 32), nn.GELU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, hist, feat, cond, cf):
        h = self.hist(hist)
        f = self.feat_enc(feat)
        c = self.cond_enc(torch.cat([cond, cf.unsqueeze(-1)], dim=-1))
        return self.head(torch.cat([h, f, c], dim=-1)).squeeze(-1)


# ── M7: Anomaly Autoencoder ──────────────────────
class AnomalyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(WINDOW+N_FEAT+COND_DIM, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, WINDOW+N_FEAT))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ── M8: Joint SOH+RUL+Fade ──────────────────────
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(WINDOW_30, 192), nn.GELU(), nn.LayerNorm(192), nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 96))
        self.feat_enc = nn.Sequential(nn.Linear(N_FEAT, 96), nn.GELU(), nn.Linear(96, 64))
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM+1, 64), nn.GELU(), nn.Linear(64, 48))
        trunk_dim = 96 + 64 + 48
        self.trunk = nn.Sequential(nn.Linear(trunk_dim, 192), nn.GELU(), nn.LayerNorm(192), nn.Dropout(0.1), nn.Linear(192, 128), nn.GELU())
        self.soh_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.rul_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.fade_head = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, hist, feat, cond, cf):
        h = self.hist(hist)
        f = self.feat_enc(feat)
        c = self.cond_enc(torch.cat([cond, cf.unsqueeze(-1)], dim=-1))
        z = self.trunk(torch.cat([h, f, c], dim=-1))
        return self.soh_head(z).squeeze(-1), self.rul_head(z).squeeze(-1), self.fade_head(z).squeeze(-1)


# ── M9: Knee Point Detector ─────────────────────
class KneeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(50))
        self.fc = nn.Sequential(nn.Linear(32*50+COND_DIM, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, cap_seq, cond):
        x = self.conv(cap_seq.unsqueeze(1))
        x = x.flatten(1)
        return self.fc(torch.cat([x, cond], dim=-1)).squeeze(-1)


# ── M10: Chemistry Performance Ranker ────────────
class ChemRanker(nn.Module):
    def __init__(self, n_chem=10):
        super().__init__()
        self.chem_embed = nn.Embedding(n_chem, 32)
        self.cond_enc = nn.Sequential(nn.Linear(COND_DIM-1, 32), nn.GELU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, chem_id, cond_no_chem):
        ce = self.chem_embed(chem_id)
        co = self.cond_enc(cond_no_chem)
        return self.head(torch.cat([ce, co], dim=-1)).squeeze(-1)


# ── EXTENDED MODELS (M11-M14) ────────────────────
# Import lazily to avoid circular deps if models_extended isn't present
try:
    from inference.models_extended import (
        ElectrolyteHealthModel, ReplenishabilityModel,
        ChemIdentifier, FormationProtocolModel,
    )
    _HAS_EXTENDED = True
except ImportError:
    _HAS_EXTENDED = False

# ── MODEL REGISTRY ───────────────────────────────
MODEL_CLASSES = {
    "M1_CathodeUDE": CathodeUDE,
    "M2_SOH": SOHModel,
    "M3_CycleLife": CycleLifeModel,
    "M4_FadeRate": FadeRateModel,
    "M5_BMS_TGN": PackTGN,
    "M6_RUL": RULModel,
    "M7_Anomaly": AnomalyAE,
    "M8_Joint_SOH_RUL": JointModel,
    "M9_KneeDetect": KneeDetector,
    "M10_ChemRank": ChemRanker,
}

if _HAS_EXTENDED:
    MODEL_CLASSES.update({
        "M11_ElectrolyteHealth": ElectrolyteHealthModel,
        "M12_Replenishability": ReplenishabilityModel,
        "M13_ChemIdentifier": ChemIdentifier,
        "M14_FormationProtocol": FormationProtocolModel,
    })

CHECKPOINT_NAMES = {
    "M1_CathodeUDE": "cathode_ude",
    "M2_SOH": "soh",
    "M3_CycleLife": "cycle_life",
    "M4_FadeRate": "fade_rate",
    "M5_BMS_TGN": "bms_tgn",
    "M6_RUL": "rul",
    "M7_Anomaly": "anomaly_ae",
    "M8_Joint_SOH_RUL": "joint_soh_rul",
    "M9_KneeDetect": "knee_detect",
    "M10_ChemRank": "chem_rank",
    "M11_ElectrolyteHealth": "electrolyte_health",
    "M12_Replenishability": "replenishability",
    "M13_ChemIdentifier": "chem_identifier",
    "M14_FormationProtocol": "formation_protocol",
}
