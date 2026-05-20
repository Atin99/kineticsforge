"""KineticsForge V4 — Checkpoint loader and inference pipeline.

Loads trained checkpoints from Kaggle results zips and serves predictions.
Falls back to scaffold mode (untrained weights) if checkpoints not found.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch
import numpy as np

from inference.models import MODEL_CLASSES, CHECKPOINT_NAMES

logger = logging.getLogger("kineticsforge.inference")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelHub:
    """Loads and caches M1-M14 trained models with graceful fallback."""

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
        # Also scan in-repo Kaggle deployment outputs. Earlier code looked one
        # directory too high, so locally extracted result folders were invisible.
        for deploy in (root / "kaggle_deploy", root / "kaggle_deploy_2", root / "kaggle_deploy_3"):
            if not deploy.exists():
                continue
            dirs.append(str(deploy))
            for sub in ("acct1_zip", "acct2_zip", "acct3_zip", "acct_zip"):
                d = deploy / sub
                if not d.exists():
                    continue
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
                    # Try safe loading first (weights_only=True prevents arbitrary code execution)
                    try:
                        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                    except Exception:
                        # Fallback for legacy checkpoints that contain non-tensor objects
                        logger.warning(f"{name}: falling back to weights_only=False for {ckpt_path} — "
                                       "consider re-saving this checkpoint safely")
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

def _m1_forward_probe(model: torch.nn.Module, c_rate: float, temperature_C: float) -> Dict[str, float]:
    """Call the trained M1 UDE once and expose its derivative/gate output.

    This is intentionally a probe, not a fake capacity correction. The capacity
    trajectory below comes from the same production Na-ion physics used by the
    API. The probe proves the checkpointed M1 network is live and gives a
    bounded neural derivative readout for downstream runtimes.
    """
    with torch.no_grad():
        cond = torch.tensor([[temperature_C / 50.0, c_rate, 1.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
        feat = torch.zeros(1, 27, dtype=torch.float32, device=DEVICE)
        state = torch.tensor([[1.0, 0.04]], dtype=torch.float32, device=DEVICE)
        z = model.cond_embed(cond)
        fz = model.feat_embed(feat)
        t = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        deriv = model(t, state, z, fz)
        inp = torch.cat([state, z, fz, torch.zeros_like(state[:, :1])], dim=-1)
        gate = model.gate(inp)
    return {
        "dQ_dt": float(deriv[0, 0]),
        "dR_dt": float(deriv[0, 1]),
        "gate_physics": float(gate[0, 0]),
        "gate_neural": float(1.0 - gate[0, 0]),
    }


def predict_degradation(hub: ModelHub, temperature_C: float = 45.0,
                        c_rate: float = 1.0, n_cycles: int = 500,
                        enable_p2o2: bool = True, enable_jt: bool = True,
                        enable_sei: bool = True,
                        na: float = 1.02, mn: float = 0.52, fe: float = 0.43,
                        dopant_frac: float = 0.05) -> Dict[str, Any]:
    """Run degradation using the production Na-ion physics plus M1 probe.

    Older code used toy sine-wave and hardcoded polynomial terms. That is gone.
    The trajectory now delegates to serve_lite.simulate_degradation so the
    full/checkpoint path and API path share the same Na-ion physics.
    """
    from serve_lite import DegradationRequest, simulate_degradation

    req = DegradationRequest(
        temperature_C=temperature_C,
        c_rate=c_rate,
        cycles=n_cycles,
        enable_p2o2=enable_p2o2,
        enable_jt=enable_jt,
        enable_sei=enable_sei,
        enable_neural=False,
        na=na,
        mn=mn,
        fe=fe,
        dopant_frac=dopant_frac,
    )
    result = simulate_degradation(req)
    out = {
        "capacity_curve": result["curve_sampled"],
        "voltage_curve": result["voltage_sampled"],
        "eol_capacity": result["capacity_end"],
        "fade_pct": result["fade_pct"],
        "knee_point": result["knee_point"],
        "rul_at_80pct": result["rul_at_80pct"],
        "cycles": result["cycles"],
        "mechanisms": result["mechanisms"],
        "composition": result["composition"],
        "physics_source": "serve_lite.simulate_degradation",
        "model_source": hub.loaded_from.get("M1_CathodeUDE", "unknown"),
    }
    model = hub.get("M1_CathodeUDE")
    if model is not None and hub.loaded_from.get("M1_CathodeUDE") not in {"scaffold", None}:
        out["m1_forward_probe"] = _m1_forward_probe(model, c_rate=c_rate, temperature_C=temperature_C)
    return out


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


def _pad(values: List[float], width: int, fill: float = 0.0) -> List[float]:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        clean = [fill]
    if len(clean) >= width:
        return clean[-width:]
    return [clean[0]] * (width - len(clean)) + clean


def _as_feature_tensor(feature_vector: List[float], feature_mask: Optional[List[int]] = None) -> torch.Tensor:
    values = _pad(feature_vector[:27], 27)
    mask = _pad([float(x) for x in (feature_mask or [1] * 27)[:27]], 27, fill=0.0)
    return torch.tensor([v * m for v, m in zip(values, mask)], dtype=torch.float32, device=DEVICE).unsqueeze(0)


def _condition_tensor(conditions: Optional[List[float]], feature_vector: List[float]) -> torch.Tensor:
    # [temperature_scaled, c_rate, DOD, chemistry_id_or_aux, form_factor_or_aux]
    if conditions:
        vals = _pad(conditions[:5], 5)
    else:
        temp_c = feature_vector[5] if len(feature_vector) > 5 and np.isfinite(feature_vector[5]) else 25.0
        c_rate = feature_vector[6] if len(feature_vector) > 6 and np.isfinite(feature_vector[6]) else 1.0
        vals = [temp_c / 50.0, c_rate, 1.0, 0.0, 0.0]
    return torch.tensor(vals, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def predict_byod_features(
    hub: ModelHub,
    feature_vector: List[float],
    feature_mask: Optional[List[int]] = None,
    capacity_history: Optional[List[float]] = None,
    conditions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run checkpoint-backed BYOD inference on a canonical 27-feature vector.

    The lite FastAPI service intentionally avoids torch. This function is for
    the full Python path, offline pilots, notebooks, or GPU workers. Missing
    fields are masked before inference so absent EIS/cycler columns are not
    silently treated as real zeros.
    """
    feat = _as_feature_tensor(feature_vector, feature_mask)
    cond = _condition_tensor(conditions, feature_vector)
    hist_seed = feature_vector[14] if len(feature_vector) > 14 and np.isfinite(feature_vector[14]) else 1.0
    hist20 = torch.tensor(_pad(capacity_history or [hist_seed], 20, 1.0), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    hist30 = torch.tensor(_pad(capacity_history or [hist_seed], 30, 1.0), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    observed_cycles = feature_vector[15] if len(feature_vector) > 15 and np.isfinite(feature_vector[15]) else 0.0
    cycle_fraction = torch.tensor([min(max(float(observed_cycles) / 1000.0, 0.0), 1.0)], dtype=torch.float32, device=DEVICE)

    outputs: Dict[str, Any] = {
        "runtime": DEVICE,
        "models": {},
        "loaded_from": hub.loaded_from,
        "mask_present": int(sum(1 for x in (feature_mask or []) if x)),
    }

    def run(name: str, fn):
        model = hub.get(name)
        if model is None:
            outputs["models"][name] = {"status": "not_loaded"}
            return
        try:
            with torch.no_grad():
                outputs["models"][name] = fn(model)
                outputs["models"][name]["checkpoint"] = hub.loaded_from.get(name, "unknown")
        except Exception as exc:
            outputs["models"][name] = {"status": "error", "detail": str(exc)}

    def run_m1(model):
        z = model.cond_embed(cond)
        fz = model.feat_embed(feat)
        r0 = feature_vector[4] if len(feature_vector) > 4 and np.isfinite(feature_vector[4]) else 0.04
        state = torch.tensor([[float(hist20[0, -1]), float(r0)]], dtype=torch.float32, device=DEVICE)
        deriv = model(cycle_fraction[0], state, z, fz)
        inp = torch.cat([state, z, fz, cycle_fraction.reshape(1, 1)], dim=-1)
        gate = model.gate(inp)
        return {
            "dQ_dt": float(deriv[0, 0]),
            "dR_dt": float(deriv[0, 1]),
            "gate_physics": float(gate[0, 0]),
            "gate_neural": float(1.0 - gate[0, 0]),
            "projected_soh_500": float(np.clip(float(hist20[0, -1]) + float(deriv[0, 0]) * 0.5, 0.0, 1.05)),
            "status": "ok",
        }

    run("M1_CathodeUDE", run_m1)
    run("M2_SOH", lambda m: {"soh": float(m(hist20, feat, cond, cycle_fraction)), "status": "ok"})
    run("M4_FadeRate", lambda m: {"fade_rate": float(m(hist20, feat, cond)), "status": "ok"})
    run("M6_RUL", lambda m: {"rul_fraction": float(m(hist20, feat, cond, cycle_fraction)), "status": "ok"})
    run("M8_Joint_SOH_RUL", lambda m: (lambda y: {"soh": float(y[0]), "rul_fraction": float(y[1]), "fade_rate": float(y[2]), "status": "ok"})(m(hist30, feat, cond, cycle_fraction)))

    def run_m11(model):
        r_ohm = max(float(feature_vector[4]) if len(feature_vector) > 4 and np.isfinite(feature_vector[4]) else 0.04, 1e-4)
        temp_c = float(feature_vector[5]) if len(feature_vector) > 5 and np.isfinite(feature_vector[5]) else 25.0
        soh = float(feature_vector[14]) if len(feature_vector) > 14 and np.isfinite(feature_vector[14]) else 1.0
        eis = torch.tensor([[r_ohm, r_ohm * 1.8, max(0.002, (1.0 - soh) * 0.05), 0.01, temp_c / 50.0, 0.5, float(cycle_fraction[0])]], dtype=torch.float32, device=DEVICE)
        deg, plating, safe_crate = model(eis)
        return {
            "electrolyte_degradation": float(torch.sigmoid(deg)[0]),
            "sodium_plating_probability": float(torch.sigmoid(plating)[0]),
            "recommended_c_rate": float(safe_crate[0]),
            "status": "ok",
            "imputed_eis": True,
        }

    run("M11_ElectrolyteHealth", run_m11)
    run("M12_Replenishability", lambda m: (lambda y: {"recovery_probability": float(torch.sigmoid(y[0])[0]), "expected_recovery_fraction": float(y[1][0]), "status": "ok"})(m(hist20, feat[:, :10])))
    run("M13_ChemIdentifier", lambda m: (lambda logits: {"class_id": int(torch.argmax(logits, dim=-1)[0]), "confidence": float(torch.softmax(logits, dim=-1).max()), "status": "ok"})(m(feat, cond[:, :4])))
    run("M14_FormationProtocol", lambda m: (lambda y: {"life_index": float(torch.sigmoid(y[0])[0]), "robustness_index": float(torch.sigmoid(y[1])[0]), "sei_quality": float(torch.sigmoid(y[2])[0]), "status": "ok"})(m(feat, cond)))
    return outputs
