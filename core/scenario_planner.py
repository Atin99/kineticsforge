"""Scenario Planner for KineticsForge

Provides a standalone function to run multiple BMS simulations with different
parameter sets and aggregate key metrics (max risk, fault cell, alert count,
confidence).  No dependency on serve_lite — the physics kernel is embedded here
so the module is importable as a library.
"""

import math
from typing import Any, Dict, List


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _mulberry32(seed: int):
    """Deterministic 32-bit PRNG (mirrors the frontend implementation)."""
    state = seed & 0xFFFFFFFF

    def _next():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (1 | state)) & 0xFFFFFFFF
        t = ((t + ((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return _next


def _seeded_gaussian(rng) -> float:
    import math as _m
    u = 1.0 - rng()
    v = 1.0 - rng()
    return _m.sqrt(-2.0 * _m.log(u)) * _m.cos(2.0 * _m.pi * v)


def _hist_back(h: list, w: int) -> float:
    frm = max(0, len(h) - w)
    s = sum(h[frm:])
    return s / max(1, len(h) - frm)


def simulate_bms(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single BMS thermal-graph simulation and return summary metrics.

    Accepts the same keys as the frontend ``BMSRequest``:
        n_cells, duration_seconds, inject_fault, enable_eis, asymmetric_alert,
        cth_j_per_k, edge_k, cooling_h, load_scale, rct_gate, risk_threshold,
        ambient_C, seed
    """
    n = int(params.get("n_cells", 8))
    duration = int(params.get("duration_seconds", 120))
    inject_fault = bool(params.get("inject_fault", True))
    use_eis = bool(params.get("enable_eis", True))
    use_asym = bool(params.get("asymmetric_alert", True))
    cth = max(10.0, float(params.get("cth_j_per_k", 95)))
    kedge = max(0.0, float(params.get("edge_k", 0.18)))
    cooling = max(0.0, float(params.get("cooling_h", 0.045)))
    load = max(0.05, float(params.get("load_scale", 1.0)))
    rct_gate = max(0.001, float(params.get("rct_gate", 0.043)))
    threshold = float(params.get("risk_threshold", 0.42))
    if not use_asym:
        threshold = max(0.48, threshold)
    ambient_K = float(params.get("ambient_C", 45)) + 273.15
    seed = int(params.get("seed", 42))

    rng = _mulberry32(seed)

    # Build simple linear topology
    neighbors: List[List[int]] = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if abs(a - b) == 1:
                neighbors[a].append(b)
                neighbors[b].append(a)

    fault_cell = int(rng() * n) if inject_fault else -1

    # Cell state
    temps = [ambient_K + _seeded_gaussian(rng) * 0.25 for _ in range(n)]
    r0s = [0.033 * (1 + _seeded_gaussian(rng) * 0.025) for _ in range(n)]
    seis = [0.010 + rng() * 0.002 for _ in range(n)]
    risks = [0.0] * n
    raw_hists: List[List[float]] = [[] for _ in range(n)]

    steps = _clamp(round(duration), 60, 240)
    dt = duration / steps
    alerts: List[Dict[str, Any]] = []
    max_risk = 0.0
    max_cell = 0

    for s in range(steps + 1):
        t = s * dt
        prev_temps = temps[:]
        for c in range(n):
            is_fault = c == fault_cell
            fault_drive = (_sigmoid((t - duration * 0.46) / max(3, duration * 0.07)) ** 2) if is_fault else 0.0
            arrh = math.exp(-0.28 / 8.617e-5 * (1 / temps[c] - 1 / ambient_K))
            seis[c] += dt * (1.0e-6 * arrh + fault_drive * 7.0e-5)
            r_int = r0s[c] + 0.18 * seis[c] + fault_drive * 0.020
            q_ohm = load * (34 * r_int + fault_drive * 14.0)
            coupling = sum(kedge * (prev_temps[j] - prev_temps[c]) for j in neighbors[c])
            d_t_dt = (q_ohm + coupling - cooling * (prev_temps[c] - ambient_K)) / cth
            temps[c] = _clamp(prev_temps[c] + dt * d_t_dt, 290, 390)

            r_sei = 0.006 + 0.080 * seis[c] + fault_drive * 0.010
            r_ct = 0.028 * math.exp(1800 * (1 / temps[c] - 1 / ambient_K)) * (1 + 3.5 * seis[c] + fault_drive * 1.6)

            temp_score = _sigmoid((temps[c] - 333.15) / 4.5)
            slope_score = _sigmoid((d_t_dt * 60 - 1.2) / 0.7)
            eis_score = _sigmoid((r_ct + r_sei - rct_gate) / 0.009) if use_eis else 0.25 * temp_score
            nbr_temp = 0.0
            if neighbors[c]:
                nbr_temp = sum(_sigmoid((prev_temps[j] - 273.15 - 60) / 5.0) for j in neighbors[c]) / len(neighbors[c])
            raw = _clamp(0.34 * temp_score + 0.21 * slope_score + 0.27 * eis_score + 0.18 * nbr_temp, 0, 1)
            raw_hists[c].append(raw)

            h = raw_hists[c]
            lookback = 0.40 * _hist_back(h, 30) + 0.28 * _hist_back(h, 60) + 0.20 * _hist_back(h, 120) + 0.12 * _hist_back(h, 240)
            risks[c] = _clamp(0.78 * risks[c] + 0.22 * lookback, 0, 1)

        cur_max_risk = max(risks)
        cur_max_cell = risks.index(cur_max_risk)
        if cur_max_risk > threshold:
            alerts.append({"t": round(t), "cell": cur_max_cell, "risk": round(cur_max_risk, 4)})
        max_risk = cur_max_risk
        max_cell = cur_max_cell

    return {
        "max_risk": round(max_risk, 4),
        "fault_cell": fault_cell,
        "max_cell": max_cell,
        "alert_count": len(alerts),
        "alerts": alerts[:20],
        "risk_threshold": threshold,
        "seed": seed,
        "n_cells": n,
        "duration_seconds": duration,
        "final_temps_C": [round(t - 273.15, 2) for t in temps],
    }


def compute_bms_confidence(result: Dict[str, Any]) -> tuple:
    """Compute a confidence score (0-1) for a BMS simulation result."""
    score = 0.5
    reasons = []
    if result.get("seed") is not None:
        score += 0.2
        reasons.append("deterministic seed present")
    max_risk = result.get("max_risk", 0.0)
    risk_score = max(0.0, 1.0 - max_risk)
    score += 0.4 * risk_score
    reasons.append(f"max risk {max_risk:.2f} gives {risk_score:.2f} contribution")
    alerts = result.get("alerts", [])
    alert_bonus = min(len(alerts) * 0.02, 0.1)
    score += alert_bonus
    reasons.append(f"{len(alerts)} alerts give +{alert_bonus:.2f}")
    threshold = result.get("risk_threshold", 0.42)
    if max_risk < threshold:
        margin = (threshold - max_risk) / threshold
        margin_bonus = 0.1 * margin
        score += margin_bonus
        reasons.append(f"margin to threshold {margin:.2f} adds +{margin_bonus:.2f}")
    score = max(0.0, min(1.0, score))
    return score, ", ".join(reasons)


def run_scenarios(params_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run a list of BMS parameter dictionaries through the simulation.

    Each dict should contain the fields accepted by ``BMSRequest`` (except
    ``seed`` which defaults to 42 if omitted). Returns a list of result dicts
    with the input params plus ``max_risk``, ``fault_cell``, ``alert_count``
    and ``confidence``.
    """
    results: List[Dict[str, Any]] = []
    for params in params_list:
        if "seed" not in params:
            params["seed"] = 42
        sim_res = simulate_bms(params)
        confidence, reason = compute_bms_confidence(sim_res)
        results.append({
            "params": params,
            "max_risk": sim_res.get("max_risk"),
            "fault_cell": sim_res.get("fault_cell"),
            "alert_count": sim_res.get("alert_count"),
            "alerts": sim_res.get("alerts", []),
            "final_temps_C": sim_res.get("final_temps_C"),
            "confidence": confidence,
            "confidence_reason": reason,
        })
    return results


if __name__ == "__main__":
    import json
    test_params = [
        {"ambient_C": 30, "duration_seconds": 120, "seed": 1},
        {"ambient_C": 45, "duration_seconds": 120, "seed": 2},
        {"ambient_C": 60, "duration_seconds": 120, "seed": 3},
    ]
    results = run_scenarios(test_params)
    for r in results:
        print(json.dumps({
            "ambient": r["params"]["ambient_C"],
            "max_risk": r["max_risk"],
            "alerts": r["alert_count"],
            "confidence": round(r["confidence"], 3),
        }, indent=2))
