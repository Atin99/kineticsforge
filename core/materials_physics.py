"""Data-backed Na-ion layered-oxide material scoring.

The public UI needs a fast local surrogate, but the surrogate should still
behave like a physics model: normalize impossible recipes, expose penalties,
and condition confidence on nearby local evidence instead of pretending every
slider setting is equally validated.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FARADAY_MAH_PER_MOL = 26801.0
K_B_EV = 8.617333262e-5

ELEMENT_MASS = {
    "Na": 22.9898,
    "Mn": 54.938,
    "Fe": 55.845,
    "Al": 26.9815,
    "Ti": 47.867,
    "Mg": 24.305,
    "Co": 58.933,
    "Ni": 58.693,
    "O": 15.999,
}

ELEMENT_COST_USD_KG = {
    "Na": 3.10,
    "Mn": 2.40,
    "Fe": 0.45,
    "Al": 2.70,
    "Ti": 11.00,
    "Mg": 2.80,
    "O": 0.0,
}

DOPANTS = {
    "Al": {
        "default_frac": 0.04,
        "valence": 3.0,
        "phase_stabilization": 0.34,
        "jt_suppression": 0.16,
        "p2_suppression": 0.42,
        "capacity_inactive": 1.0,
        "rate_bonus": 0.04,
    },
    "Ti": {
        "default_frac": 0.03,
        "valence": 4.0,
        "phase_stabilization": 0.28,
        "jt_suppression": 0.44,
        "p2_suppression": 0.30,
        "capacity_inactive": 1.0,
        "rate_bonus": 0.08,
    },
    "Mg": {
        "default_frac": 0.04,
        "valence": 2.0,
        "phase_stabilization": 0.18,
        "jt_suppression": 0.10,
        "p2_suppression": 0.15,
        "capacity_inactive": 1.0,
        "rate_bonus": 0.02,
    },
}

FALLBACK_EVIDENCE = [
    {
        "formula": "Na0.67Fe0.3Mn0.7O2",
        "Na": 0.67,
        "Mn": 0.70,
        "Fe": 0.30,
        "capacity_mAh_g": None,
        "retention": 0.826,
        "cycles": 300.0,
        "c_rate": 5.0,
        "source": "data/real/scraped/literature_evidence.jsonl",
    },
    {
        "formula": "Na0.75Co0.125Cu0.125Fe0.125Ni0.125Mn0.5O2",
        "Na": 0.75,
        "Mn": 0.50,
        "Fe": 0.125,
        "capacity_mAh_g": 70.0,
        "retention": 0.96,
        "cycles": 500.0,
        "c_rate": 1.0,
        "source": "data/real/scraped/literature_evidence.jsonl",
    },
    {
        "formula": "Na0.76Mn0.5Ni0.3Fe0.1Mg0.1O2",
        "Na": 0.76,
        "Mn": 0.50,
        "Fe": 0.10,
        "capacity_mAh_g": None,
        "retention": 0.80,
        "cycles": 700.0,
        "c_rate": None,
        "source": "data/real/scraped/literature_evidence.jsonl",
    },
]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def sigmoid(x: float) -> float:
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _num(value: Any, fallback: float) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else fallback
    except (TypeError, ValueError):
        return fallback


def _dopant_inputs(comp: Dict[str, Any]) -> Dict[str, float]:
    dopants: Dict[str, float] = {}
    explicit = comp.get("dopant")
    explicit_frac = _num(comp.get("dopant_frac"), 0.0)
    if explicit in DOPANTS:
        dopants[str(explicit)] = clamp(explicit_frac or DOPANTS[str(explicit)]["default_frac"], 0.0, 0.20)

    if comp.get("al") or comp.get("al_doped"):
        dopants["Al"] = clamp(_num(comp.get("al_frac"), DOPANTS["Al"]["default_frac"]), 0.0, 0.20)
    if comp.get("ti") or comp.get("ti_doped"):
        dopants["Ti"] = clamp(_num(comp.get("ti_frac"), DOPANTS["Ti"]["default_frac"]), 0.0, 0.20)
    if comp.get("mg") or comp.get("mg_doped"):
        dopants["Mg"] = clamp(_num(comp.get("mg_frac"), DOPANTS["Mg"]["default_frac"]), 0.0, 0.20)
    return {k: v for k, v in dopants.items() if v > 1e-8}


def normalize_composition(comp: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a nominal recipe onto one transition-metal site.

    The UI sliders are intentionally user-controlled. When the entered Mn + Fe
    + dopant amount is not one site, the model normalizes for physics and keeps
    the site-balance mismatch as an explicit penalty/confidence driver.
    """
    na = clamp(_num(comp.get("Na", comp.get("na", 1.0)), 1.0), 0.45, 1.30)
    mn_raw = clamp(_num(comp.get("Mn", comp.get("mn", 0.5)), 0.5), 0.0, 1.40)
    fe_raw = clamp(_num(comp.get("Fe", comp.get("fe", 0.5)), 0.5), 0.0, 1.40)
    dopants_raw = _dopant_inputs(comp)
    dop_total_raw = sum(dopants_raw.values())
    tm_total = mn_raw + fe_raw + dop_total_raw
    if tm_total <= 1e-10:
        mn_site, fe_site, dopants_site = 0.5, 0.5, {}
        tm_total = 1.0
    else:
        mn_site = mn_raw / tm_total
        fe_site = fe_raw / tm_total
        dopants_site = {k: v / tm_total for k, v in dopants_raw.items()}

    site_error = tm_total - 1.0
    return {
        "Na": na,
        "Mn_raw": mn_raw,
        "Fe_raw": fe_raw,
        "Mn": mn_site,
        "Fe": fe_site,
        "dopants": dopants_site,
        "dopant_total": sum(dopants_site.values()),
        "tm_total_raw": tm_total,
        "site_error": site_error,
        "site_error_abs": abs(site_error),
        "al_doped": dopants_site.get("Al", 0.0) > 0,
        "ti_doped": dopants_site.get("Ti", 0.0) > 0,
    }


def _formula_amounts(formula: str) -> Dict[str, float]:
    clean = (
        str(formula or "")
        .replace("−", "-")
        .replace("–", "-")
        .replace(" ", "")
    )
    if not clean or clean.lower() == "unknown":
        return {}
    amounts: Dict[str, float] = {}
    for el, amount in re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", clean):
        if el not in ELEMENT_MASS:
            continue
        value = float(amount) if amount else 1.0
        amounts[el] = amounts.get(el, 0.0) + value
    return amounts


def _anchor_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    formula = str(row.get("formula") or "")
    amounts = _formula_amounts(formula)
    if not amounts.get("Na") or not amounts.get("Mn") or not amounts.get("O"):
        return None
    title = str(row.get("title") or "").lower()
    if not any(token in title for token in ("layered", "p2", "o3", "sodium-ion", "sodium ion")):
        return None
    tm_elements = ["Mn", "Fe", "Al", "Ti", "Mg", "Co", "Ni"]
    tm_total = sum(amounts.get(el, 0.0) for el in tm_elements)
    if tm_total <= 0:
        return None
    prop = str(row.get("property_name") or "")
    retention = None
    capacity = None
    if prop == "capacity_retention":
        retention = clamp(_num(row.get("retention_percent", row.get("value")), 0.0) / 100.0, 0.0, 1.2)
    elif prop == "specific_capacity":
        capacity = _num(row.get("capacity_mAh_g", row.get("value")), 0.0)
        if capacity <= 0 or capacity > 260:
            return None
        rp = row.get("retention_percent")
        if rp is not None:
            retention = clamp(_num(rp, 0.0) / 100.0, 0.0, 1.2)
    else:
        return None

    return {
        "formula": formula,
        "Na": amounts.get("Na", 0.0),
        "Mn": amounts.get("Mn", 0.0) / tm_total,
        "Fe": amounts.get("Fe", 0.0) / tm_total,
        "Al": amounts.get("Al", 0.0) / tm_total,
        "Ti": amounts.get("Ti", 0.0) / tm_total,
        "Mg": amounts.get("Mg", 0.0) / tm_total,
        "capacity_mAh_g": capacity,
        "retention": retention,
        "cycles": row.get("cycle_count"),
        "c_rate": row.get("c_rate"),
        "source": "data/real/scraped/literature_evidence.jsonl",
        "title": row.get("title"),
    }


@lru_cache(maxsize=1)
def load_literature_anchors() -> Tuple[Dict[str, Any], ...]:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "real" / "scraped" / "literature_evidence.jsonl"
    anchors: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            anchor = _anchor_from_row(row)
            if anchor:
                anchors.append(anchor)

    if not anchors:
        anchors = list(FALLBACK_EVIDENCE)

    dedup: Dict[Tuple[str, Optional[float], Optional[float]], Dict[str, Any]] = {}
    for item in anchors:
        key = (str(item.get("formula")), item.get("cycles"), item.get("c_rate"))
        prev = dedup.get(key)
        if not prev:
            dedup[key] = item
            continue
        if item.get("retention") is not None and prev.get("retention") is None:
            prev["retention"] = item.get("retention")
        if item.get("capacity_mAh_g") is not None and prev.get("capacity_mAh_g") is None:
            prev["capacity_mAh_g"] = item.get("capacity_mAh_g")
    return tuple(dedup.values())


def nearest_evidence(norm: Dict[str, Any]) -> Dict[str, Any]:
    anchors = load_literature_anchors()
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    al = norm["dopants"].get("Al", 0.0)
    ti = norm["dopants"].get("Ti", 0.0)
    for anchor in anchors:
        d = math.sqrt(
            1.6 * (norm["Na"] - _num(anchor.get("Na"), norm["Na"])) ** 2
            + 1.2 * (norm["Mn"] - _num(anchor.get("Mn"), norm["Mn"])) ** 2
            + 1.0 * (norm["Fe"] - _num(anchor.get("Fe"), norm["Fe"])) ** 2
            + 0.8 * (al - _num(anchor.get("Al"), 0.0)) ** 2
            + 0.8 * (ti - _num(anchor.get("Ti"), 0.0)) ** 2
        )
        if best is None or d < best[0]:
            best = (d, anchor)

    if best is None:
        return {"distance": 1.0, "count": 0, "nearest_formula": None}
    return {
        "distance": round(best[0], 4),
        "count": len(anchors),
        "nearest_formula": best[1].get("formula"),
        "nearest_capacity_mAh_g": best[1].get("capacity_mAh_g"),
        "nearest_retention": best[1].get("retention"),
        "nearest_cycles": best[1].get("cycles"),
        "source": best[1].get("source"),
    }


def _molar_mass(norm: Dict[str, Any]) -> float:
    mass = norm["Na"] * ELEMENT_MASS["Na"] + norm["Mn"] * ELEMENT_MASS["Mn"] + norm["Fe"] * ELEMENT_MASS["Fe"]
    for el, frac in norm["dopants"].items():
        mass += frac * ELEMENT_MASS.get(el, 45.0)
    return mass + 2.0 * ELEMENT_MASS["O"]


def _dopant_strength(norm: Dict[str, Any], key: str) -> float:
    total = 0.0
    for el, frac in norm["dopants"].items():
        meta = DOPANTS.get(el, {})
        default = max(1e-6, _num(meta.get("default_frac"), 0.04))
        total += (frac / default) * _num(meta.get(key), 0.0)
    return clamp(total, 0.0, 1.0)


def simulate_retention_curve(fade_terms: Dict[str, float], cycles: int = 1200) -> Dict[str, Any]:
    beta = 0.72
    base_k = sum(max(0.0, v) for v in fade_terms.values())
    base_k = max(base_k, 1e-7)
    p2_frac = fade_terms.get("p2_o2", 0.0) / base_k
    jt_frac = fade_terms.get("jahn_teller", 0.0) / base_k
    knee_cycle = clamp((0.58 - 0.18 * p2_frac - 0.10 * jt_frac) * cycles, 80.0, cycles * 0.92)
    out_cycles: List[int] = []
    retention: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    step = max(1, cycles // 240)
    for n in range(0, cycles + 1, step):
        accel = 1.0
        if n > knee_cycle:
            accel += (p2_frac * 0.75 + jt_frac * 0.35) * ((n - knee_cycle) / max(1.0, cycles - knee_cycle)) ** 1.35
        loss = 1.0 - math.exp(-base_k * (max(0, n) ** beta) * accel)
        cap = clamp(1.0 - loss, 0.42, 1.01)
        band = clamp(0.012 + 0.055 * math.sqrt(max(0, n) / max(1, cycles)), 0.0, 0.12)
        out_cycles.append(n)
        retention.append(round(cap, 5))
        lo.append(round(clamp(cap - band, 0.35, 1.01), 5))
        hi.append(round(clamp(cap + band, 0.35, 1.04), 5))
    eol80 = None
    for cyc, cap in zip(out_cycles, retention):
        if cap <= 0.80:
            eol80 = cyc
            break
    return {
        "cycles": out_cycles,
        "retention": retention,
        "lo": lo,
        "hi": hi,
        "knee_cycle": int(round(knee_cycle)),
        "eol80_cycle": eol80,
    }


def score_composition(
    comp: Dict[str, Any],
    temp_K: float = 318.15,
    knobs: Optional[Dict[str, Any]] = None,
    include_curve: bool = True,
) -> Dict[str, Any]:
    knobs = knobs or {}
    temp_K = clamp(temp_K, 250.0, 390.0)
    norm = normalize_composition(comp)
    upper_v = clamp(_num(knobs.get("upper_voltage", knobs.get("upperV")), 4.10), 3.40, 4.55)
    ehull_slope = max(1.0, _num(knobs.get("ehull_slope", knobs.get("ehullSlope")), 20.0))
    weights = {
        "capacity": _num(knobs.get("w_capacity", knobs.get("wCap")), 0.32),
        "stability": _num(knobs.get("w_stability", knobs.get("wStab")), 0.32),
        "fade": _num(knobs.get("w_fade", knobs.get("wFade")), 0.22),
        "cost": _num(knobs.get("w_cost", knobs.get("wCost")), 0.14),
    }
    charge_penalty = max(0.0, _num(knobs.get("charge_penalty", knobs.get("chargePenalty")), 0.10))
    defect_penalty = max(0.0, _num(knobs.get("defect_penalty", knobs.get("defectPenalty")), 0.06))

    na, mn, fe = norm["Na"], norm["Mn"], norm["Fe"]
    dop_charge = sum(frac * DOPANTS.get(el, {}).get("valence", 3.0) for el, frac in norm["dopants"].items())
    mn_ox_raw = (4.0 - na - 3.0 * fe - dop_charge) / max(mn, 1e-6)
    mn_ox_error = max(0.0, 3.0 - mn_ox_raw, mn_ox_raw - 4.0)
    mn_ox = clamp(mn_ox_raw, 3.0, 4.0)
    mn3_fraction = clamp(4.0 - mn_ox, 0.0, 1.0)
    charge_balance_risk = clamp(0.65 * mn_ox_error + 0.90 * norm["site_error_abs"], 0.0, 1.0)

    phase_stabilization = _dopant_strength(norm, "phase_stabilization")
    jt_suppression = _dopant_strength(norm, "jt_suppression")
    p2_suppression = _dopant_strength(norm, "p2_suppression")

    na_mobility = clamp(1.0 - 1.15 * abs(na - 0.88) + 0.08 * sigmoid((na - 0.65) / 0.08), 0.22, 1.0)
    fe_redox_gate = sigmoid((upper_v - 4.02) / 0.10)
    mn_utilization = clamp(0.70 + 0.17 * sigmoid((upper_v - 3.72) / 0.12) - 0.20 * charge_balance_risk - 0.12 * norm["site_error_abs"], 0.15, 0.94)
    fe_utilization = clamp(0.14 + 0.58 * fe_redox_gate - 0.10 * charge_balance_risk, 0.05, 0.72)
    oxygen_redox_e = clamp((upper_v - 4.18) * 0.75, 0.0, 0.16) * mn * clamp(1.0 - na, 0.0, 0.45)
    mn_e = mn * mn3_fraction * mn_utilization
    fe_e = fe * fe_utilization
    extractable_na = clamp(na - 0.22 - 0.08 * norm["site_error_abs"], 0.0, 0.92)
    e_per_formula = min(extractable_na, mn_e + fe_e + oxygen_redox_e)
    utilization_penalty = clamp(0.94 * na_mobility - 0.20 * norm["site_error_abs"] - 0.10 * charge_balance_risk, 0.35, 0.98)
    q0 = FARADAY_MAH_PER_MOL * e_per_formula / max(1e-6, _molar_mass(norm)) * utilization_penalty

    voltage_terms = []
    if mn_e > 1e-8:
        voltage_terms.append((mn_e, 3.50 + 0.24 * (1.0 - mn3_fraction) + 0.05 * (upper_v - 4.0)))
    if fe_e > 1e-8:
        voltage_terms.append((fe_e, 3.18 + 0.08 * fe_redox_gate))
    if oxygen_redox_e > 1e-8:
        voltage_terms.append((oxygen_redox_e, 4.15))
    denom = sum(x[0] for x in voltage_terms) or 1.0
    avg_voltage = sum(w * v for w, v in voltage_terms) / denom + 0.06 * (0.85 - na)
    avg_voltage = clamp(avg_voltage, 2.60, min(upper_v, 4.35))
    energy_density = q0 * avg_voltage

    p2_crit_v = 4.04 + 0.12 * fe + 0.18 * phase_stabilization - 0.13 * mn3_fraction - 0.08 * max(0.0, 0.78 - na)
    na_p2_weight = 0.35 + (1.0 - 0.35) * sigmoid((0.86 - na) / 0.12)
    p2_o2_risk = clamp(sigmoid((upper_v - p2_crit_v) / 0.075) * na_p2_weight * (1.0 - 0.52 * p2_suppression), 0.0, 1.0)
    jt_index = clamp(mn * mn3_fraction * (1.0 - 0.55 * jt_suppression) * (1.0 + 0.18 * sigmoid((upper_v - 4.05) / 0.10)), 0.0, 1.0)
    oxygen_risk = clamp(
        0.12
        + 0.54 * p2_o2_risk
        + 0.34 * clamp(upper_v - 4.10, 0.0, 0.35) / 0.35
        + 0.30 * max(0.0, 0.76 - na)
        + 0.18 * mn * (1.0 - mn3_fraction)
        - 0.20 * phase_stabilization,
        0.0,
        1.0,
    )
    tm_mixing_risk = clamp(0.10 + 0.28 * abs(mn - fe) + 0.34 * norm["site_error_abs"] + 0.16 * max(0.0, 0.72 - na) - 0.10 * phase_stabilization, 0.0, 1.0)
    moisture_risk = clamp(0.16 + 0.62 * abs(na - 0.92) + 0.18 * norm["site_error_abs"], 0.0, 1.0)
    defect_score = clamp(1.0 - (0.26 * oxygen_risk + 0.22 * tm_mixing_risk + 0.18 * moisture_risk + 0.24 * jt_index + 0.22 * charge_balance_risk), 0.0, 1.0)

    ehull = clamp(
        0.020
        + 0.080 * max(0.0, 0.78 - na)
        + 0.050 * norm["site_error_abs"]
        + 0.038 * oxygen_risk
        + 0.030 * tm_mixing_risk
        + 0.018 * charge_balance_risk
        - 0.022 * phase_stabilization,
        0.0,
        0.26,
    )
    phase_stability = 1.0 / (1.0 + math.exp(clamp(ehull_slope * (ehull - 0.055), -40.0, 40.0)))
    thermal_onset_c = 238.0 - 42.0 * oxygen_risk - 24.0 * jt_index + 24.0 * phase_stabilization + 10.0 * fe
    thermal_abuse = clamp((thermal_onset_c - 170.0) / 125.0, 0.0, 1.0)
    rate_capability = clamp(0.45 + 0.42 * na_mobility + 0.12 * fe + 0.12 * _dopant_strength(norm, "rate_bonus") - 0.16 * charge_balance_risk, 0.05, 1.0)

    arrh = math.exp(clamp(-0.42 / K_B_EV * (1.0 / temp_K - 1.0 / 318.15), -3.0, 3.0))
    fade_terms = {
        "sei": 2.7e-4 * arrh * (0.75 + 0.50 * moisture_risk),
        "p2_o2": 7.5e-4 * p2_o2_risk * (0.75 + 0.35 * sigmoid((upper_v - 4.05) / 0.08)),
        "jahn_teller": 5.8e-4 * jt_index,
        "oxygen": 4.2e-4 * oxygen_risk,
        "rate": 2.6e-4 * (1.0 - rate_capability),
        "charge_site": 4.8e-4 * charge_balance_risk,
    }
    beta = 0.72
    fade_k = sum(fade_terms.values())
    fade_500 = clamp(1.0 - math.exp(-fade_k * (500.0 ** beta)), 0.002, 0.68)
    cycle_life = clamp((-math.log(0.80) / max(fade_k, 1e-7)) ** (1.0 / beta), 50.0, 5000.0)
    retention_curve = simulate_retention_curve(fade_terms, cycles=max(1000, int(min(2200, cycle_life * 1.25)))) if include_curve else None

    cost_kg = 2.50
    formula_mass = _molar_mass(norm)
    for el, frac in [("Na", norm["Na"]), ("Mn", mn), ("Fe", fe), ("O", 2.0)]:
        cost_kg += (frac * ELEMENT_MASS[el] / formula_mass) * ELEMENT_COST_USD_KG.get(el, 0.0)
    for el, frac in norm["dopants"].items():
        cost_kg += (frac * ELEMENT_MASS.get(el, 45.0) / formula_mass) * ELEMENT_COST_USD_KG.get(el, 6.0)
    cost_kwh = cost_kg / max(energy_density / 1000.0, 0.01)

    stability = clamp(
        0.24 * (1.0 - fade_500)
        + 0.22 * phase_stability
        + 0.18 * defect_score
        + 0.14 * thermal_abuse
        + 0.12 * rate_capability
        + 0.10 * (1.0 - charge_balance_risk),
        0.0,
        1.0,
    )

    evidence = nearest_evidence(norm)
    evidence_distance = _num(evidence.get("distance"), 1.0)
    confidence = clamp(
        0.84
        - 0.22 * min(evidence_distance, 1.5)
        - 0.28 * charge_balance_risk
        - 0.22 * norm["site_error_abs"]
        - 0.12 * oxygen_risk
        + (0.05 if evidence.get("count", 0) >= 8 else 0.0),
        0.18,
        0.92,
    )

    score = (
        weights["capacity"] * clamp(q0 / 180.0, 0.0, 1.25)
        + weights["stability"] * stability
        + weights["fade"] * (1.0 - fade_500)
        + weights["cost"] * clamp(1.0 - cost_kwh / 220.0, 0.0, 1.0)
        - charge_penalty * charge_balance_risk
        - defect_penalty * (1.0 - defect_score)
    )

    if p2_o2_risk > 0.62:
        phase_state = "P2->O2 transition risk"
    elif charge_balance_risk > 0.45:
        phase_state = "charge-compensated defect phase"
    elif phase_stability < 0.38:
        phase_state = "mixed/impurity phase risk"
    elif jt_index > 0.38:
        phase_state = "JT-distorted layered phase"
    else:
        phase_state = "P2 layered phase"

    lattice_spacing = clamp(5.48 + 0.22 * na - 0.16 * p2_o2_risk - 0.08 * norm["site_error_abs"] + 0.04 * phase_stabilization, 5.25, 5.82)
    radar = [
        {"label": "Capacity", "value": clamp(q0 / 180.0, 0.0, 1.0)},
        {"label": "Phase", "value": phase_stability},
        {"label": "Fade", "value": 1.0 - fade_500},
        {"label": "Cost", "value": clamp(1.0 - cost_kwh / 220.0, 0.0, 1.0)},
        {"label": "O2 Safe", "value": 1.0 - oxygen_risk},
        {"label": "Charge", "value": 1.0 - charge_balance_risk},
    ]

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
        "phase_stability": round(phase_stability, 4),
        "phase_state": phase_state,
        "ehull_ev_atom": round(ehull, 5),
        "p2_o2_risk": round(p2_o2_risk, 4),
        "mn_oxidation_state": round(mn_ox, 4),
        "mn3_fraction": round(mn3_fraction, 4),
        "site_balance_error": round(norm["site_error"], 5),
        "normalized_composition": {
            "Na": round(norm["Na"], 5),
            "Mn": round(norm["Mn"], 5),
            "Fe": round(norm["Fe"], 5),
            **{el: round(frac, 5) for el, frac in norm["dopants"].items()},
        },
        "lattice_spacing_A": round(lattice_spacing, 4),
        "thermal_onset_C": round(thermal_onset_c, 2),
        "rate_capability": round(rate_capability, 4),
        "defect_score": round(defect_score, 4),
        "tm_mixing_risk": round(tm_mixing_risk, 4),
        "moisture_risk": round(moisture_risk, 4),
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "mechanisms": {k: round(v, 8) for k, v in fade_terms.items()},
        "retention_curve": retention_curve,
        "radar": [{"label": r["label"], "value": round(r["value"], 4)} for r in radar],
        "model_basis": "charge-balance capacity + phase/fade surrogate calibrated against local literature evidence",
    }


def _candidate_grid() -> Iterable[Dict[str, Any]]:
    for na_i in range(13):
        na = 0.62 + na_i * 0.04
        for mn_i in range(16):
            mn = 0.20 + mn_i * 0.04
            for dop in ("none", "al", "ti", "al_ti"):
                al = dop in ("al", "al_ti")
                ti = dop in ("ti", "al_ti")
                dop_raw = (DOPANTS["Al"]["default_frac"] if al else 0.0) + (DOPANTS["Ti"]["default_frac"] if ti else 0.0)
                fe = clamp(1.0 - mn - dop_raw, 0.06, 0.82)
                yield {"Na": na, "Mn": mn, "Fe": fe, "al_doped": al, "ti_doped": ti}


def screen_batch(n: int = 100, temp_K: float = 318.15, knobs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for comp in _candidate_grid():
        prop = score_composition(comp, temp_K=temp_K, knobs=knobs, include_curve=False)
        compact = dict(prop)
        compact.pop("retention_curve", None)
        compact.pop("mechanisms", None)
        items.append({"composition": comp, "properties": compact, "score": prop["score"]})
    items.sort(key=lambda item: item["score"], reverse=True)
    return items[: max(1, min(int(n), len(items)))]
