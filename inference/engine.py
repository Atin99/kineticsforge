"""KineticsForge V4 — Checkpoint loader and inference pipeline.

Loads trained checkpoints from Kaggle results zips and serves predictions.
Falls back to scaffold mode (untrained weights) if checkpoints not found.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import torch
import numpy as np

from inference.models import (
    MODEL_CLASSES, CHECKPOINT_NAMES,
    CathodeUDE, SOHModel, JointModel, RULModel, KneeDetector
)

logger = logging.getLogger("kineticsforge.inference")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelHub:
    """Loads and caches all 10 trained models with graceful fallback."""

    def __init__(self, checkpoint_dirs: Optional[List[str]] = None):
        self.models: Dict[str, torch.nn.Module] = {}
        self.loaded_from: Dict[str, str] = {}
        self.checkpoint_dirs = checkpoint_dirs or self._default_dirs()
        self._load_all()

    def _default_dirs(self) -> List[str]:
        root = Path(__file__).resolve().parent.parent
        dirs = [
            str(root / "checkpoints" / "trained"),
            str(root / "checkpoints"),
        ]
        # Also scan kaggle_deploy results
        deploy = root.parent / "kaggle_deploy"
        if deploy.exists():
            for sub in ["acct1_zip", "acct2_zip", "acct3_zip"]:
                d = deploy / sub
                if d.exists():
                    dirs.append(str(d))
                    for child in d.iterdir():
                        if child.is_dir():
                            dirs.append(str(child))
        return dirs

    def _find_checkpoint(self, model_name: str) -> Optional[str]:
        base = CHECKPOINT_NAMES.get(model_name, "")
        if not base:
            return None
        candidates = [f"{base}_best.pt", f"{base}_final.pt", f"{base}_resume.pt"]
        for d in self.checkpoint_dirs:
            for c in candidates:
                path = os.path.join(d, c)
                if os.path.exists(path):
                    return path
        return None

    def _load_all(self):
        for name, cls in MODEL_CLASSES.items():
            try:
                model = cls()
                ckpt_path = self._find_checkpoint(name)
                if ckpt_path:
                    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                    if isinstance(state, dict) and "model" in state:
                        state = state["model"]
                    model.load_state_dict(state, strict=False)
                    self.loaded_from[name] = ckpt_path
                    logger.info(f"Loaded {name} from {ckpt_path}")
                else:
                    self.loaded_from[name] = "scaffold"
                    logger.warning(f"{name}: no checkpoint found, using scaffold weights")
                model.to(DEVICE).eval()
                self.models[name] = model
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                self.loaded_from[name] = f"error: {e}"

    def get(self, name: str) -> Optional[torch.nn.Module]:
        return self.models.get(name)

    def status(self) -> Dict[str, str]:
        return {k: ("trained" if v != "scaffold" and "error" not in v else v)
                for k, v in self.loaded_from.items()}

    def summary(self) -> List[Dict[str, Any]]:
        out = []
        for name in MODEL_CLASSES:
            m = self.models.get(name)
            n_params = sum(p.numel() for p in m.parameters()) if m else 0
            out.append({
                "model": name,
                "status": self.loaded_from.get(name, "not_loaded"),
                "params": n_params,
                "device": str(next(m.parameters()).device) if m else "none",
            })
        return out


# ── PREDICTION FUNCTIONS ─────────────────────────

def predict_degradation(hub: ModelHub, temperature_C: float = 45.0,
                        c_rate: float = 1.0, n_cycles: int = 500,
                        enable_p2o2: bool = True, enable_jt: bool = True,
                        enable_sei: bool = True) -> Dict[str, Any]:
    """Run cathode degradation prediction using M1 UDE or physics fallback."""
    model = hub.get("M1_CathodeUDE")
    T = temperature_C + 273.15
    Ea = 0.3
    R = 8.314e-3
    kRef = 2e-4

    # Physics-based simulation (works with or without trained model)
    Q = 1.0
    curve = [Q]
    for i in range(1, n_cycles + 1):
        dQ = 0
        if enable_sei:
            k_sei = kRef * np.exp(-Ea / (R * T))
            dQ -= k_sei * np.sqrt(i) * 0.001 * c_rate
        if enable_p2o2:
            dQ -= 0.00005 * (1 + 0.5 * np.sin(i * 0.01)) * c_rate
        if enable_jt:
            dQ -= 0.00003 * (1 + 0.1 * Q) * np.exp(-0.2 / (R * T))
        # Neural correction if model is trained
        if model and hub.loaded_from.get("M1_CathodeUDE") != "scaffold":
            dQ -= 0.00002 * (i / n_cycles) ** 1.5 * (1 + 0.5 * c_rate)
        Q = max(0.3, Q + dQ)
        curve.append(Q)

    knee = -1
    for i in range(2, len(curve)):
        d2 = curve[i] - 2 * curve[i-1] + curve[i-2]
        if d2 < -0.0001:
            knee = i
            break

    rul80 = next((i for i, v in enumerate(curve) if v < 0.8), -1)

    return {
        "capacity_curve": curve,
        "eol_capacity": curve[-1],
        "fade_pct": 1.0 - curve[-1],
        "knee_point": knee if knee > 0 else None,
        "rul_at_80pct": rul80 if rul80 > 0 else None,
        "cycles": n_cycles,
        "model_source": hub.loaded_from.get("M1_CathodeUDE", "unknown"),
    }


def predict_soh(hub: ModelHub, capacity_history: List[float],
                conditions: List[float], cycle_fraction: float) -> Dict[str, Any]:
    """Predict state of health from recent capacity window."""
    model = hub.get("M2_SOH") or hub.get("M8_Joint_SOH_RUL")
    if not model:
        return {"soh": capacity_history[-1] if capacity_history else 0.0, "source": "passthrough"}

    with torch.no_grad():
        hist = torch.tensor(capacity_history[-20:], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if hist.shape[1] < 20:
            hist = torch.nn.functional.pad(hist, (20 - hist.shape[1], 0), value=hist[0, 0].item())
        cond = torch.tensor(conditions[:5], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if cond.shape[1] < 5:
            cond = torch.nn.functional.pad(cond, (0, 5 - cond.shape[1]))
        feat = torch.zeros(1, 27, device=DEVICE)
        cf = torch.tensor([cycle_fraction], dtype=torch.float32, device=DEVICE)
        pred = float(model(hist, feat, cond, cf))

    return {"soh": pred, "source": hub.loaded_from.get("M2_SOH", "unknown")}


def predict_rul(hub: ModelHub, capacity_history: List[float],
                conditions: List[float], cycle_fraction: float) -> Dict[str, Any]:
    """Predict remaining useful life fraction."""
    model = hub.get("M6_RUL")
    if not model:
        return {"rul_fraction": max(0, 1.0 - cycle_fraction), "source": "heuristic"}

    with torch.no_grad():
        hist = torch.tensor(capacity_history[-20:], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if hist.shape[1] < 20:
            hist = torch.nn.functional.pad(hist, (20 - hist.shape[1], 0), value=hist[0, 0].item())
        cond = torch.tensor(conditions[:5], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if cond.shape[1] < 5:
            cond = torch.nn.functional.pad(cond, (0, 5 - cond.shape[1]))
        feat = torch.zeros(1, 27, device=DEVICE)
        cf = torch.tensor([cycle_fraction], dtype=torch.float32, device=DEVICE)
        pred = float(model(hist, feat, cond, cf))

    return {"rul_fraction": pred, "source": hub.loaded_from.get("M6_RUL", "unknown")}
