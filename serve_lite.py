"""KineticsForge lightweight server for Render free tier.

No PyTorch dependency. Uses compact numpy physics mirrors for the web
application and API endpoints.

Run: python serve_lite.py
"""
import os
import sys
import time
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PYDEPS = ROOT / "_pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ROOT / ".env")

try:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("pip install fastapi uvicorn numpy pydantic", file=sys.stderr)
    raise

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kineticsforge")

try:
    from api.chat_assistant import answer_chat
except Exception as exc:  # pragma: no cover - keeps the lite server bootable
    logger.warning("chat assistant unavailable: %s", exc)
    answer_chat = None

app = FastAPI(title="KineticsForge")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SEI_RATE_CALIBRATION = 1e10  # Converts the eV Arrhenius prefactor into the cycle-step loss scale used by the lite mirror.
SEI_PREF_DEFAULT = 5.0e-5
SEI_SQRT_COEFF = 0.048
GLOBAL_DEGRADATION_SCALE = 0.052
JT_LOSS_COEFF = 6.5e-3
DESOLV_LOSS_COEFF = 2.5e-4
BV_RATE_LOSS_COEFF = 1.2e-4
RESIDUAL_LOSS_COEFF = 1.0e-5
RECYCLING_MC_SAMPLES = 200


class DegradationRequest(BaseModel):
    temperature_C: float = 45.0
    c_rate: float = 1.0
    cycles: int = 500
    enable_p2o2: bool = True
    enable_jt: bool = True
    enable_sei: bool = True
    enable_neural: bool = True
    sei_k_scale: float = 1.0
    sei_ea_ev: float = 0.56
    p2_rate: float = 0.0028
    p2_soc_crit: float = 0.78
    jt_scale: float = 1.0
    bv_scale: float = 1.0
    stress_exponent: float = 0.55
    residual_scale: float = 1.0
    na: float = 1.02
    mn: float = 0.52
    fe: float = 0.43
    dopant_frac: float = 0.05


class BMSRequest(BaseModel):
    n_cells: int = 8
    duration_seconds: int = 120
    inject_fault: bool = True
    enable_eis: bool = True
    asymmetric_alert: bool = True
    cth_j_per_k: float = 95.0
    edge_k: float = 0.18
    cooling_h: float = 0.045
    load_scale: float = 1.0
    rct_gate: float = 0.043
    risk_threshold: float = 0.42
    ambient_C: float = 45.0


class RecyclingRequest(BaseModel):
    mass_kg: float = 100.0
    acid_molarity: float = 2.0
    temperature_C: float = 80.0
    monte_carlo: bool = True
    bayesian_update: bool = True
    leach_time_min: float = 120.0
    particle_um: float = 50.0
    acid_order: float = 0.95
    mn_ea_j_mol: float = 27000.0


class ScreenRequest(BaseModel):
    na: float = 1.0
    mn: float = 0.5
    fe: float = 0.5
    al_doped: bool = False
    ti_doped: bool = False
    upper_voltage: float = 4.10
    ehull_slope: float = 20.0
    w_capacity: float = 0.32
    w_stability: float = 0.32
    w_fade: float = 0.22
    w_cost: float = 0.14


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1200)
    section: str = "general"
    state: Optional[Dict[str, Any]] = None


class CathodeBatchRequest(BaseModel):
    n: int = 100
    temperature_K: float = 318.15


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def na_ion_terms(q: float, soc: float, comp: Dict[str, float], temp_K: float, req: Optional[DegradationRequest] = None) -> Dict[str, float]:
    k_b = 8.617e-5
    sei_scale = clamp(getattr(req, "sei_k_scale", 1.0), 0.0, 10.0)
    sei_ea = clamp(getattr(req, "sei_ea_ev", 0.56), 0.30, 0.95)
    p2_base = clamp(getattr(req, "p2_rate", 2.8e-3), 0.0, 0.025)
    p2_soc_base = clamp(getattr(req, "p2_soc_crit", 0.78), 0.45, 1.05)
    jt_scale = clamp(getattr(req, "jt_scale", 1.0), 0.0, 4.0)
    mn = clamp(comp.get("Mn", 0.52), 0.0, 1.5)
    fe = clamp(comp.get("Fe", 0.43), 0.0, 1.5)
    dop = clamp(comp.get("dopant_frac", 0.05), 0.0, 0.25)
    jt = clamp(
        mn
        * clamp(1.15 - soc, 0.0, 1.0)
        * math.exp(clamp((temp_K - 298.15) * 0.018, -4.0, 4.0))
        * math.exp(-0.45 * fe - 0.70 * dop),
        0.0,
        4.0,
    ) * jt_scale
    soc_crit = clamp(p2_soc_base - 0.09 * mn + 0.06 * fe + 0.18 * dop, 0.55, 0.95)
    p2_gate = sigmoid((soc - soc_crit) / 0.045)
    p2o2_rate = clamp(
        p2_base
        * p2_gate
        * math.exp(clamp((temp_K - 298.15) * 0.024 / 25.0, -3.0, 3.0))
        * (1.0 + 0.35 * jt),
        0.0,
        0.08,
    )
    barrier = 0.18 + 0.025 * mn - 0.014 * fe - 0.050 * dop
    desolv = clamp(
        math.exp(clamp(barrier / (k_b * temp_K + 1e-10), -2.0, 4.0))
        * (1.0 + 0.25 * max(0.0, soc - 0.85)),
        0.2,
        30.0,
    )
    beta = clamp(0.48 - 0.035 * math.log1p(desolv) + 0.025 * clamp(soc - 0.5, -0.5, 0.5), 0.25, 0.75)
    sei_rate = SEI_PREF_DEFAULT * sei_scale * math.exp(-sei_ea / (k_b * temp_K))
    return {"jt": jt, "p2o2_rate": p2o2_rate, "desolv": desolv, "beta": beta, "sei_rate": sei_rate, "soc_crit": soc_crit}


def simulate_degradation(req: DegradationRequest) -> Dict:
    temp_K = req.temperature_C + 273.15
    cycles = int(clamp(req.cycles, 50, 3000))
    comp = {
        "Na": clamp(req.na, 0.60, 1.20),
        "Mn": clamp(req.mn, 0.05, 0.95),
        "Fe": clamp(req.fe, 0.05, 0.95),
        "dopant_frac": clamp(req.dopant_frac, 0.0, 0.25),
    }
    q = 1.0
    stress = 0.6 + req.c_rate ** clamp(req.stress_exponent, 0.25, 3.0)
    curve = [q]
    voltage = [3.34]
    mechanisms = {"p2o2": 0.0, "jt": 0.0, "sei_desolv": 0.0, "rate_polarization": 0.0, "residual": 0.0}
    for i in range(1, cycles + 1):
        soh_window = clamp(q, 0.50, 1.0)
        soc_base = 0.78 + 0.04 * min(1.0, req.c_rate / 2.4)
        usable_soc = 0.62 + 0.38 * soh_window
        soc = clamp(0.55 + (soc_base - 0.55) * usable_soc + 0.022 * math.sin(i * 0.17) * usable_soc, 0.55, 0.98)
        terms = na_ion_terms(q, soc, comp, temp_K, req)
        scale = GLOBAL_DEGRADATION_SCALE * stress
        sqrt_increment = math.sqrt(i) - math.sqrt(i - 1)
        sei_loss = q * (terms["sei_rate"] * SEI_RATE_CALIBRATION + SEI_SQRT_COEFF * sqrt_increment) * scale if req.enable_sei else 0.0
        p2_loss = q * 0.65 * terms["p2o2_rate"] * scale if req.enable_p2o2 else 0.0
        jt_loss = q * JT_LOSS_COEFF * terms["jt"] * scale if req.enable_jt else 0.0
        desolv_loss = q * DESOLV_LOSS_COEFF * math.log1p(terms["desolv"]) * scale
        exchange_proxy = clamp(0.34 + 0.18 * comp["Fe"] - 0.08 * math.log1p(terms["desolv"]) + 0.04 * (1.0 - terms["beta"]), 0.08, 0.90)
        eta = math.asinh(req.c_rate / (2.0 * exchange_proxy))
        rate_stress = 1.0 + 0.20 * max(0.0, req.c_rate - 1.5) ** 2
        rate_loss = q * clamp(req.bv_scale, 0.0, 5.0) * BV_RATE_LOSS_COEFF * eta * eta * rate_stress * scale
        residual_loss = q * RESIDUAL_LOSS_COEFF * clamp(req.residual_scale, 0.0, 5.0) * sigmoid((i / cycles - 0.62) / 0.16) * (0.8 + 0.35 * req.c_rate) if req.enable_neural else 0.0
        q = clamp(q - sei_loss - p2_loss - jt_loss - desolv_loss - rate_loss - residual_loss, 0.25, 1.02)
        mechanisms["p2o2"] += p2_loss
        mechanisms["jt"] += jt_loss
        mechanisms["sei_desolv"] += sei_loss + desolv_loss
        mechanisms["rate_polarization"] += rate_loss
        mechanisms["residual"] += residual_loss
        v_degradation = mechanisms["p2o2"] * 0.15 + mechanisms["jt"] * 0.08 + mechanisms["sei_desolv"] * 0.05 + mechanisms["rate_polarization"] * 0.04
        voltage.append(clamp(3.34 - v_degradation, 2.40, 3.50))
        curve.append(q)
    knee = None
    for i in range(2, len(curve)):
        if curve[i] - 2 * curve[i - 1] + curve[i - 2] < -1.6e-5:
            knee = i
            break
    rul80 = next((i for i, v in enumerate(curve) if v < 0.8), None)
    step = max(1, len(curve) // 160)
    return {
        "capacity_start": 1.0,
        "capacity_end": round(curve[-1], 5),
        "fade_pct": round(1.0 - curve[-1], 5),
        "knee_point": knee,
        "rul_at_80pct": rul80,
        "cycles": cycles,
        "composition": comp,
        "curve_sampled": [round(curve[i], 5) for i in range(0, len(curve), step)],
        "voltage_sampled": [round(voltage[i], 5) for i in range(0, len(voltage), step)],
        "mechanisms": {k: round(v, 5) for k, v in mechanisms.items()},
    }


def build_neighbors(n: int) -> List[List[int]]:
    cols = n if n <= 8 else int(math.ceil(math.sqrt(n * 1.4)))
    rows = int(math.ceil(n / cols))
    pos = [(i % cols, i // cols) for i in range(n)]
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(pos[i][0] - pos[j][0])
            dy = abs(pos[i][1] - pos[j][1])
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def simulate_bms(req: BMSRequest) -> Dict:
    n = int(clamp(req.n_cells, 4, 32))
    duration = int(clamp(req.duration_seconds, 30, 600))
    steps = int(clamp(duration, 60, 240))
    dt = duration / steps
    ambient = clamp(req.ambient_C, 15.0, 70.0) + 273.15
    rng = np.random.default_rng()
    fault_cell = int(rng.integers(0, n)) if req.inject_fault else -1
    neighbors = build_neighbors(n)
    temp = ambient + rng.normal(0, 0.25, n)
    r0 = 0.033 * (1.0 + rng.normal(0, 0.025, n))
    sei = 0.010 + rng.random(n) * 0.002
    risk = np.zeros(n)
    histories = [[] for _ in range(n)]
    alerts = []
    threshold = clamp(req.risk_threshold if req.asymmetric_alert else max(req.risk_threshold, 0.55), 0.05, 0.95)
    failure_time = duration * 0.84
    for s in range(steps + 1):
        t = s * dt
        prev = temp.copy()
        raw = np.zeros(n)
        for i in range(n):
            fault_drive = sigmoid((t - duration * 0.46) / max(3.0, duration * 0.07)) ** 2 if i == fault_cell else 0.0
            arrh = math.exp(-0.28 / 8.617e-5 * (1.0 / prev[i] - 1.0 / ambient))
            sei[i] += dt * (1.0e-6 * arrh + fault_drive * 7.0e-5)
            r_int = r0[i] + 0.18 * sei[i] + fault_drive * 0.020
            q_ohm = clamp(req.load_scale, 0.05, 4.0) * (34.0 * r_int + fault_drive * 14.0)
            coupling = sum(clamp(req.edge_k, 0.0, 1.2) * (prev[j] - prev[i]) for j in neighbors[i])
            dtdt = (q_ohm + coupling - clamp(req.cooling_h, 0.0, 0.5) * (prev[i] - ambient)) / max(10.0, req.cth_j_per_k)
            temp[i] = clamp(prev[i] + dt * dtdt, 290.0, 390.0)
            r_sei = 0.006 + 0.080 * sei[i] + fault_drive * 0.010
            r_ct = 0.028 * math.exp(1800.0 * (1.0 / temp[i] - 1.0 / ambient)) * (1.0 + 3.5 * sei[i] + fault_drive * 1.6)
            temp_score = sigmoid((temp[i] - 333.15) / 4.5)
            slope_score = sigmoid((dtdt * 60.0 - 1.2) / 0.7)
            eis_score = sigmoid((r_ct + r_sei - clamp(req.rct_gate, 0.005, 0.20)) / 0.009) if req.enable_eis else 0.25 * temp_score
            neigh_score = np.mean([sigmoid((temp[j] - 333.15) / 5.0) for j in neighbors[i]]) if neighbors[i] else 0.0
            raw[i] = clamp(0.34 * temp_score + 0.21 * slope_score + 0.27 * eis_score + 0.18 * neigh_score, 0.0, 1.0)
            histories[i].append(float(raw[i]))
            hist = histories[i]
            def back(w: int) -> float:
                part = hist[max(0, len(hist) - w):]
                return float(np.mean(part)) if part else 0.0
            lookback = 0.40 * back(30) + 0.28 * back(60) + 0.20 * back(120) + 0.12 * back(240)
            risk[i] = clamp(0.78 * risk[i] + 0.22 * lookback, 0.0, 1.0)
        if s % max(4, steps // 16) == 0 or float(np.max(risk)) > threshold:
            max_cell = int(np.argmax(risk))
            if risk[max_cell] > threshold:
                alerts.append({"t": round(t, 1), "cell": max_cell, "risk": round(float(risk[max_cell]), 4), "lead_seconds": max(0, round(failure_time - t, 1))})
    return {
        "cells": n,
        "fault_cell": fault_cell,
        "thermal_equation": "Cth dT/dt = q + sum(kij(Tj-Ti)) - h(T-Ta)",
        "final_risks": {f"C{i}": round(float(risk[i]), 4) for i in range(n)},
        "final_temperature_C": {f"C{i}": round(float(temp[i] - 273.15), 2) for i in range(n)},
        "alerts": alerts[-30:],
        "max_risk": round(float(np.max(risk)), 4),
        "ambient_C": round(ambient - 273.15, 2),
    }


def score_composition(comp: Dict[str, float], temp_K: float = 318.15, knobs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    knobs = knobs or comp
    upper_voltage = clamp(float(knobs.get("upper_voltage", 4.10)), 3.60, 4.50)
    ehull_slope = max(1.0, float(knobs.get("ehull_slope", 20.0)))
    w_capacity = float(knobs.get("w_capacity", 0.32))
    w_stability = float(knobs.get("w_stability", 0.32))
    w_fade = float(knobs.get("w_fade", 0.22))
    w_cost = float(knobs.get("w_cost", 0.14))
    al = bool(comp.get("al_doped", False))
    ti = bool(comp.get("ti_doped", False))
    fade_mult, life_mult, cap_mult, vol_mult, rate_mult = (0.90, 1.10, 0.99, 0.90, 1.08) if ti else ((0.82, 1.18, 0.97, 0.85, 1.05) if al else (1, 1, 1, 1, 1))
    dop_frac = (0.04 if al else 0.0) + (0.03 if ti else 0.0)
    na, mn, fe = comp["Na"], comp["Mn"], comp["Fe"]
    q0 = (120.0 + 40.0 * mn - 20.0 * fe) * cap_mult * (1.0 - 0.5 * abs(na - 1.0))
    ea = 0.55 + 0.1 * mn - 0.03 * fe
    k_fade = 1e-4 * (1.0 + 0.2 * fe) * math.exp(-ea * 96485.0 / (8.314 * temp_K))
    jt = 1.0 + 0.3 * max(0.0, mn - 0.5)
    ss = 1.0 / (1.0 + math.exp(-8.0 * (0.5 - mn)))
    fe_stab = 0.9 + 0.2 * fe
    voltage_stress = 1.0 + 1.8 * sigmoid((upper_voltage - 4.05) / 0.08)
    fade_500 = clamp(1.0 - math.exp(-(k_fade * jt * fade_mult / fe_stab) * voltage_stress * 500.0 ** 1.15), 0.02, 0.48)
    cycle_life = 400.0 * life_mult / jt * (0.85 if mn > 0.6 else 1.0)
    avg_voltage = 3.3 + 0.2 * fe - 0.1 * mn
    energy_density = q0 * avg_voltage
    e_form = -4.2 - 0.6 * mn - 0.35 * fe - 0.4 * na - (0.048 if al else 0.0) - (0.054 if ti else 0.0)
    ehull = max(0.0, e_form - (-4.0 - 0.3 * mn - 0.2 * fe) + 0.05)
    phase_stab = 1.0 / (1.0 + math.exp(ehull_slope * (ehull - 0.05)))
    thermal_abuse = clamp((250.0 - 30.0 * max(0.0, mn - 0.5) + 15.0 * fe + (8.0 if al else 0.0) + (7.5 if ti else 0.0) - 180.0) / 120.0, 0.0, 1.0)
    oxygen_risk = clamp(0.22 + max(0.0, mn - 0.55) + max(0.0, 1.0 - na) * 0.8 + 0.24 * sigmoid((upper_voltage - 4.15) / 0.07) - (0.06 if al else 0.0) - (0.08 if ti else 0.0), 0.0, 1.0)
    mixing_risk = clamp(0.18 + abs(mn - fe) * 0.35 + max(0.0, 0.98 - na) * 1.2 + (0.03 if ti else -0.02 if al else 0.0), 0.0, 1.0)
    moisture = clamp(0.20 + max(0.0, na - 0.98) * 0.9 + max(0.0, 1.0 - na) * 2.2 + 0.24, 0.0, 1.0)
    jt_index = clamp((mn - 0.48) * 1.8 - (0.18 if ti else 0.0), 0.0, 1.0)
    defect_score = clamp(1.0 - (0.24 * oxygen_risk + 0.22 * mixing_risk + 0.20 * moisture + 0.24 * jt_index), 0.0, 1.0)
    cost_kg = na * 3.1 * 0.23 + mn * 2.4 * 0.55 + fe * 0.45 * 0.56 + dop_frac * (11.0 * 0.479 if ti else 2.7 * 0.27 if al else 0.0) + 2.5
    cost_kwh = cost_kg / max(energy_density / 1000.0, 0.01)
    stability = clamp(0.28 * (1.0 - fade_500) + 0.18 * ss * fe_stab + 0.18 * phase_stab + 0.16 * thermal_abuse + 0.20 * defect_score, 0.0, 1.0)
    mn_ox = 3.0 + (1.0 - clamp(mn, 0.0, 1.0)) * 0.5
    fe_ox = 3.0
    dop_charge = (0.04 * 3.0 if al else 0.0) + (0.03 * 4.0 if ti else 0.0)
    total_charge = na + mn * mn_ox + fe * fe_ox + dop_charge
    charge_balance_risk = clamp(abs(total_charge - 4.0) / 1.4, 0.0, 1.0)
    score = w_capacity * (q0 / 180.0) + w_stability * stability + w_fade * (1.0 - fade_500) + w_cost * max(0.0, 1.0 - cost_kwh / 200.0) - 0.08 * charge_balance_risk
    return {
        "capacity": round(q0, 3),
        "capacity_500": round(q0 * (1.0 - fade_500), 3),
        "fade_500": round(fade_500, 5),
        "cycle_life": round(cycle_life, 1),
        "voltage": round(avg_voltage, 4),
        "stability": round(stability, 4),
        "jt_index": round(jt_index, 4),
        "energy_density": round(energy_density, 3),
        "cost_usd_kwh": round(cost_kwh, 2),
        "oxygen_risk": round(oxygen_risk, 4),
        "charge_balance_risk": round(charge_balance_risk, 4),
        "score": round(score, 5),
    }


def screen_batch(n: int = 100, temp_K: float = 318.15, knobs: Optional[Dict[str, float]] = None) -> List[Dict]:
    out = []
    for na in np.linspace(0.84, 1.12, 8):
        for mn in np.linspace(0.20, 0.82, 16):
            for dop in ("none", "al", "ti"):
                fe = clamp(1.0 - mn - (0.04 if dop == "al" else 0.03 if dop == "ti" else 0.0), 0.12, 0.82)
                comp = {"Na": float(na), "Mn": float(mn), "Fe": float(fe), "al_doped": dop == "al", "ti_doped": dop == "ti"}
                prop = score_composition(comp, temp_K=temp_K, knobs=knobs)
                score = prop["score"]
                out.append({"composition": comp, "properties": prop, "score": round(score, 5)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[: max(1, min(n, len(out)))]


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def shrinking_core(k: float, t_min: float) -> float:
    y = clamp(k * t_min, 0.0, 0.995)
    return clamp(1.0 - (1.0 - y) ** 3, 0.0, 0.995)


def recycling_result(req: RecyclingRequest) -> Dict:
    acid_order = clamp(req.acid_order, 0.1, 2.5)
    particle_um = clamp(req.particle_um, 5.0, 500.0)
    elements = [
        {"n": "Mn", "wt": 0.22, "k0": 0.0038, "Ea": clamp(req.mn_ea_j_mol, 12000.0, 62000.0), "order": acid_order, "particle": particle_um, "prior": (8.8, 1.2)},
        {"n": "Fe", "wt": 0.11, "k0": 0.0029, "Ea": 30000.0, "order": max(0.1, acid_order - 0.10), "particle": particle_um * 1.10, "prior": (7.2, 2.8)},
        {"n": "Na", "wt": 0.05, "k0": 0.0062, "Ea": 19000.0, "order": max(0.1, acid_order - 0.40), "particle": particle_um * 0.90, "prior": (6.5, 3.5)},
        {"n": "Al", "wt": 0.04, "k0": 0.0011, "Ea": 36000.0, "order": acid_order + 0.10, "particle": particle_um * 1.30, "prior": None},
        {"n": "Cu", "wt": 0.015, "k0": 0.0007, "Ea": 34000.0, "order": max(0.1, acid_order - 0.15), "particle": particle_um * 1.40, "prior": None},
    ]
    temp_K = req.temperature_C + 273.15
    t_min = clamp(req.leach_time_min, 5.0, 360.0)
    recoveries = {}
    total = 0.0
    for el in elements:
        temp_factor = math.exp(-el["Ea"] / 8.314 * (1.0 / temp_K - 1.0 / 353.15))
        k = el["k0"] * req.acid_molarity ** el["order"] * temp_factor * (50.0 / el["particle"]) ** 0.35
        x = shrinking_core(k, t_min)
        if req.bayesian_update and el["prior"]:
            x = clamp(0.75 * x + 0.25 * beta_mean(*el["prior"]), 0.0, 0.995)
        mass = req.mass_kg * el["wt"] * x
        total += mass
        recoveries[el["n"]] = {"recovery_rate": round(x, 5), "mass_kg": round(mass, 3)}
    interval = None
    if req.monte_carlo:
        rng = np.random.default_rng()
        samples = []
        rates = np.array([recoveries[el["n"]]["recovery_rate"] for el in elements])
        wts = np.array([el["wt"] for el in elements])
        for _ in range(RECYCLING_MC_SAMPLES):
            feed_noise = np.clip(rng.normal(1.0, 0.08, len(elements)), 0.75, 1.25)
            assay_noise = np.clip(rng.normal(1.0, 0.025, len(elements)), 0.92, 1.08)
            samples.append(float(np.sum(req.mass_kg * wts * feed_noise * rates * assay_noise)))
        interval = {"p05_kg": round(float(np.percentile(samples, 5)), 3), "p95_kg": round(float(np.percentile(samples, 95)), 3)}
    acid_kg = req.acid_molarity * 0.098 * req.mass_kg * (t_min / 120.0) ** 0.12
    heat_kwh = max(0.0, req.temperature_C - 25.0) * req.mass_kg * 0.00116
    impurity_penalty = clamp(recoveries["Al"]["recovery_rate"] * 0.28 + recoveries["Cu"]["recovery_rate"] * 0.36, 0.0, 0.8)
    purity_proxy = clamp(0.94 - impurity_penalty * 0.18 + (recoveries["Mn"]["recovery_rate"] + recoveries["Fe"]["recovery_rate"] + recoveries["Na"]["recovery_rate"]) * 0.012, 0.70, 0.98)
    cost = acid_kg * 8.5 + heat_kwh * 8.0 + req.mass_kg * 150.0
    margin_proxy = total * 620.0 * purity_proxy - cost
    return {
        "feedstock_kg": req.mass_kg,
        "kinetics": "shrinking-core leaching",
        "recipe": {"time_min": round(t_min, 2), "particle_um": round(particle_um, 2), "acid_order": round(acid_order, 3)},
        "recoveries": recoveries,
        "total_recovered_kg": round(total, 3),
        "uncertainty_interval": interval,
        "product_purity_proxy": round(purity_proxy, 4),
        "margin_proxy_inr": round(margin_proxy, 2),
        "cost_estimate_inr": round(cost, 2),
        "priors": {"Mn": "Beta(8.8,1.2)", "Fe": "Beta(7.2,2.8)", "Na": "Beta(6.5,3.5)"},
    }


@app.get("/health")
def health():
    return {
        "status": "operational",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "lite numpy physics",
        "models": 10,
    }


@app.get("/api/models")
def api_models():
    names = [
        ("Cathode UDE", "runnable", 150000),
        ("SOH Estimator", "runnable", 45000),
        ("Cycle Life", "runnable", 35000),
        ("Fade Rate", "runnable", 20000),
        ("BMS Pack Graph", "label_gate", 80000),
        ("RUL Predictor", "runnable", 50000),
        ("Anomaly AE", "runnable", 30000),
        ("Joint SOH+RUL", "runnable", 120000),
        ("Knee Detector", "runnable", 60000),
        ("Chem Ranker", "runnable", 15000),
    ]
    return {"models": [{"name": n, "status": s, "params": p} for n, s, p in names]}


@app.post("/api/predict/degradation")
def api_degradation(req: DegradationRequest):
    return {"result": simulate_degradation(req), "provenance": {"model": "Na-ion UDE physics mirror", "claim": "simulation-backed"}}


@app.post("/api/simulate/bms")
def api_bms(req: BMSRequest):
    return simulate_bms(req)


@app.post("/api/optimize/recycling")
def api_recycling(req: RecyclingRequest):
    return recycling_result(req)


@app.post("/api/screen/cathode")
def api_screen(req: ScreenRequest):
    comp = {"Na": req.na, "Mn": req.mn, "Fe": req.fe, "al_doped": req.al_doped, "ti_doped": req.ti_doped}
    knobs = {
        "upper_voltage": req.upper_voltage,
        "ehull_slope": req.ehull_slope,
        "w_capacity": req.w_capacity,
        "w_stability": req.w_stability,
        "w_fade": req.w_fade,
        "w_cost": req.w_cost,
    }
    return {"composition": comp, "predicted": score_composition(comp, knobs=knobs), "candidates": screen_batch(24, knobs=knobs)}


@app.post("/predict/lifetime")
def predict_lifetime(req: DegradationRequest):
    return api_degradation(req)


@app.post("/alert/bms")
def alert_bms(req: BMSRequest):
    result = simulate_bms(req)
    return {"result": {"alert_fired": bool(result["alerts"]), "alerts": result["alerts"], "max_risk": result["max_risk"], "fault_cell": result["fault_cell"]}, "provenance": {"model": "thermal graph plus EIS risk mirror"}}


@app.post("/optimize/recycling")
def optimize_recycling(req: RecyclingRequest):
    return {"result": recycling_result(req), "provenance": {"model": "Bayesian shrinking-core recycling mirror"}}


@app.post("/cathode/screen")
def cathode_screen(req: CathodeBatchRequest):
    return {"count": min(req.n, 100), "results": screen_batch(req.n, req.temperature_K)}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if answer_chat is None:
        return {
            "answer": "Assistant module is unavailable, but the physics API is still running.",
            "source": "local_fallback",
            "memory": "off",
            "setup_required": True,
        }
    return answer_chat(req.question, section=req.section, state=req.state)


WEBAPP = ROOT / "webapp"
if WEBAPP.exists():
    @app.get("/")
    async def index():
        return FileResponse(str(WEBAPP / "index.html"))

    app.mount("/", StaticFiles(directory=str(WEBAPP)), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("serve_lite:app", host="127.0.0.1", port=port, log_level="info")
