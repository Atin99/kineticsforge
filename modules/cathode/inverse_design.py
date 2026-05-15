import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.india_context import IndiaOperatingContext
from modules.cathode.composition_sampler import get_100_compositions
from modules.cathode.screener import (
    CostModel,
    ElectrolyteCompatibility,
    PhaseStabilityCalculator,
    SynthesizabilityScore,
    ThermalAbuseScreener,
    _score_composition,
)


DOPANTS = [None, "Al", "Ti", "Mg", "V", "Cr"]


@dataclass
class DesignTarget:
    min_q0_mAh_g: float = 145.0
    max_fade_500: float = 0.10
    max_cost_inr_kwh: float = 7000.0
    min_phase_stability: float = 0.55
    min_defect_tolerance: float = 0.50
    min_synthesizability: float = 0.70
    min_thermal_onset_C: float = 220.0
    preferred_route: str = "coprecipitation"
    operating_temperature_K: float = 318.0
    capacity_weight: float = 0.25
    fade_weight: float = 0.28
    cost_weight: float = 0.14
    stability_weight: float = 0.16
    synthesis_weight: float = 0.10
    uncertainty_weight: float = 0.07


@dataclass
class DesignConstraint:
    name: str
    passed: bool
    margin: float
    value: float
    threshold: float
    direction: str


@dataclass
class CandidateExperiment:
    rank: int
    composition: Dict[str, Any]
    utility: float
    acquisition_score: float
    constraints: List[DesignConstraint]
    predicted_metrics: Dict[str, Any]
    rationale: List[str] = field(default_factory=list)
    next_measurements: List[str] = field(default_factory=list)
    kill_criteria: List[str] = field(default_factory=list)


def composition_vector(comp: Dict[str, Any]) -> np.ndarray:
    dopant = comp.get("dopant")
    dopant_index = float(DOPANTS.index(dopant)) / max(len(DOPANTS) - 1, 1) if dopant in DOPANTS else 0.0
    return np.array(
        [
            float(comp.get("Na", 1.0)),
            float(comp.get("Mn", 0.5)),
            float(comp.get("Fe", 0.5)),
            float(comp.get("dopant_frac", 0.0)),
            dopant_index,
        ],
        dtype=float,
    )


def normalize_comp(comp: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(comp)
    out["Na"] = float(np.clip(float(out.get("Na", 1.0)), 0.80, 1.15))
    mn = max(0.05, float(out.get("Mn", 0.5)))
    fe = max(0.05, float(out.get("Fe", 0.5)))
    dopant_frac = float(np.clip(float(out.get("dopant_frac", 0.0)), 0.0, 0.10))
    scale = max(mn + fe + dopant_frac, 1e-9)
    if scale > 1.02:
        mn = mn / scale
        fe = fe / scale
        dopant_frac = dopant_frac / scale
    out["Mn"] = float(np.clip(mn, 0.05, 0.90))
    out["Fe"] = float(np.clip(fe, 0.05, 0.90))
    out["dopant_frac"] = float(dopant_frac)
    if out.get("dopant") not in DOPANTS:
        out["dopant"] = None
    if out["dopant_frac"] < 0.005:
        out["dopant"] = None
        out["dopant_frac"] = 0.0
    return out


class InverseCathodeDesigner:
    def __init__(
        self,
        target: Optional[DesignTarget] = None,
        seed: int = 20260502,
        evaluator: Optional[Callable[[Dict[str, Any], float], Dict[str, Any]]] = None,
    ) -> None:
        self.target = target or DesignTarget()
        self.rng = np.random.RandomState(seed)
        self.cost_model = CostModel()
        self.phase_calc = PhaseStabilityCalculator()
        self.thermal = ThermalAbuseScreener()
        self.synthesis = SynthesizabilityScore()
        self.elyte = ElectrolyteCompatibility()
        self.india = IndiaOperatingContext.from_env()
        self.evaluator = evaluator or self._default_evaluator

    def _default_evaluator(self, comp: Dict[str, Any], temperature_K: float) -> Dict[str, Any]:
        return _score_composition(
            normalize_comp(comp),
            temperature_K,
            self.cost_model,
            self.phase_calc,
            self.thermal,
            self.synthesis,
            self.elyte,
            n_mc=40,
        )

    def candidate_pool(self, base: Optional[Sequence[Dict[str, Any]]] = None, n_jitter: int = 320) -> List[Dict[str, Any]]:
        pool = [normalize_comp(c) for c in (base or get_100_compositions())]
        for _ in range(n_jitter):
            parent = dict(pool[self.rng.randint(0, len(pool))])
            parent["Na"] += self.rng.normal(0.0, 0.035)
            parent["Mn"] += self.rng.normal(0.0, 0.055)
            parent["Fe"] += self.rng.normal(0.0, 0.055)
            if self.rng.rand() < 0.38:
                parent["dopant"] = DOPANTS[self.rng.randint(0, len(DOPANTS))]
                parent["dopant_frac"] = self.rng.uniform(0.015, 0.075) if parent["dopant"] else 0.0
            pool.append(normalize_comp(parent))
        return self._dedupe(pool)

    def _dedupe(self, pool: Sequence[Dict[str, Any]], decimals: int = 3) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for comp in pool:
            key = (
                round(float(comp["Na"]), decimals),
                round(float(comp["Mn"]), decimals),
                round(float(comp["Fe"]), decimals),
                comp.get("dopant"),
                round(float(comp.get("dopant_frac", 0.0)), decimals),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(comp)
        return out

    def evaluate(self, comp: Dict[str, Any]) -> Tuple[float, List[DesignConstraint], Dict[str, Any]]:
        metrics = self.evaluator(comp, self.target.operating_temperature_K)
        synth = metrics.get("synthesizability", {})
        route_score = float(synth.get(self.target.preferred_route, max([v for k, v in synth.items() if k != "best_route"] or [0.0])))
        constraints = [
            self._constraint("initial capacity", metrics["Q0"], self.target.min_q0_mAh_g, "min"),
            self._constraint("500 cycle fade", metrics["fade_500"], self.target.max_fade_500, "max"),
            self._constraint("cost per kWh INR", metrics.get("cost_inr_kwh", self.india.usd_to_rupees(metrics["cost_usd_kwh"])), self.target.max_cost_inr_kwh, "max"),
            self._constraint("phase stability", metrics["phase_stability"], self.target.min_phase_stability, "min"),
            self._constraint("defect tolerance", metrics.get("defect_tolerance_score", 0.0), self.target.min_defect_tolerance, "min"),
            self._constraint("preferred route score", route_score, self.target.min_synthesizability, "min"),
            self._constraint("thermal onset", metrics["thermal_abuse_onset_C"], self.target.min_thermal_onset_C, "min"),
        ]
        utility = self._utility(metrics, route_score)
        hard_penalty = sum(max(0.0, -c.margin) for c in constraints)
        utility = float(utility - 0.18 * hard_penalty)
        metrics["preferred_route_score"] = route_score
        metrics["target_violation_count"] = sum(0 if c.passed else 1 for c in constraints)
        return utility, constraints, metrics

    def _constraint(self, name: str, value: float, threshold: float, direction: str) -> DesignConstraint:
        if direction == "min":
            margin = (value - threshold) / max(abs(threshold), 1e-9)
            passed = value >= threshold
        else:
            margin = (threshold - value) / max(abs(threshold), 1e-9)
            passed = value <= threshold
        return DesignConstraint(name=name, passed=bool(passed), margin=float(margin), value=float(value), threshold=float(threshold), direction=direction)

    def _utility(self, metrics: Dict[str, Any], route_score: float) -> float:
        target = self.target
        q_term = self._softplus_ratio(metrics["Q0"], target.min_q0_mAh_g, maximize=True)
        fade_term = self._softplus_ratio(metrics["fade_500"], target.max_fade_500, maximize=False)
        cost_inr = metrics.get("cost_inr_kwh", self.india.usd_to_rupees(metrics["cost_usd_kwh"]))
        cost_term = self._softplus_ratio(cost_inr, target.max_cost_inr_kwh, maximize=False)
        stability = (
            0.35 * metrics["phase_stability"]
            + 0.35 * min(1.0, metrics["thermal_abuse_onset_C"] / max(target.min_thermal_onset_C, 1.0))
            + 0.30 * metrics.get("defect_tolerance_score", 0.0)
        )
        synth_term = route_score
        unc = float(metrics.get("uncertainty", 0.0))
        unc_term = 1.0 / (1.0 + unc / 16.0)
        return (
            target.capacity_weight * q_term
            + target.fade_weight * fade_term
            + target.cost_weight * cost_term
            + target.stability_weight * stability
            + target.synthesis_weight * synth_term
            + target.uncertainty_weight * unc_term
        )

    def _softplus_ratio(self, value: float, target: float, maximize: bool) -> float:
        ratio = value / max(target, 1e-9)
        if not maximize:
            ratio = target / max(value, 1e-9)
        return float(1.0 / (1.0 + math.exp(-4.0 * (ratio - 1.0))))

    def search(self, n_return: int = 10, n_jitter: int = 480) -> List[CandidateExperiment]:
        pool = self.candidate_pool(n_jitter=n_jitter)
        evaluated: List[Tuple[float, Dict[str, Any], List[DesignConstraint], Dict[str, Any]]] = []
        vectors = np.array([composition_vector(c) for c in pool])
        novelty = self._novelty_scores(vectors)
        for idx, comp in enumerate(pool):
            utility, constraints, metrics = self.evaluate(comp)
            acquisition = utility + 0.08 * novelty[idx] + 0.035 * min(float(metrics.get("uncertainty", 0.0)) / 20.0, 1.0)
            evaluated.append((acquisition, comp, constraints, metrics))
        evaluated.sort(key=lambda x: x[0], reverse=True)
        selected = self._diverse_select(evaluated, n_return)
        out: List[CandidateExperiment] = []
        for rank, item in enumerate(selected, 1):
            acquisition, comp, constraints, metrics = item
            utility, _, _ = self.evaluate(comp)
            out.append(
                CandidateExperiment(
                    rank=rank,
                    composition=comp,
                    utility=float(utility),
                    acquisition_score=float(acquisition),
                    constraints=constraints,
                    predicted_metrics=self._public_metrics(metrics),
                    rationale=self._rationale(comp, metrics, constraints),
                    next_measurements=self._measurements(metrics, constraints),
                    kill_criteria=self._kill_criteria(metrics),
                )
            )
        return out

    def _novelty_scores(self, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) < 2:
            return np.ones(len(vectors))
        dists = np.sqrt(((vectors[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=-1))
        dists[dists == 0] = np.inf
        nearest = np.min(dists, axis=1)
        return nearest / max(float(np.max(nearest[np.isfinite(nearest)])), 1e-9)

    def _diverse_select(
        self,
        evaluated: Sequence[Tuple[float, Dict[str, Any], List[DesignConstraint], Dict[str, Any]]],
        n_return: int,
    ) -> List[Tuple[float, Dict[str, Any], List[DesignConstraint], Dict[str, Any]]]:
        selected: List[Tuple[float, Dict[str, Any], List[DesignConstraint], Dict[str, Any]]] = []
        selected_vecs: List[np.ndarray] = []
        for item in evaluated:
            vec = composition_vector(item[1])
            if selected_vecs:
                min_dist = min(float(np.linalg.norm(vec - sv)) for sv in selected_vecs)
                if min_dist < 0.045 and len(selected) < max(3, n_return // 2):
                    continue
            selected.append(item)
            selected_vecs.append(vec)
            if len(selected) >= n_return:
                break
        return selected

    def _public_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        keep = [
            "Q0",
            "Q_500",
            "fade_500",
            "cycle_life",
            "score",
            "uncertainty",
            "phase_stability",
            "thermal_abuse_onset_C",
            "preferred_route_score",
            "cost_usd_kwh",
            "cost_inr_kwh",
            "cost_index_india",
            "avg_voltage",
            "energy_density",
            "electrolyte_compatibility",
            "defect_tolerance_score",
            "oxygen_redox_risk",
            "tm_mixing_risk",
            "moisture_sensitivity",
            "charge_balance_error",
        ]
        return {k: self._json_safe(metrics[k]) for k in keep if k in metrics}

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float):
            return float(value)
        if isinstance(value, int):
            return int(value)
        return value

    def _rationale(self, comp: Dict[str, Any], metrics: Dict[str, Any], constraints: Sequence[DesignConstraint]) -> List[str]:
        reasons: List[str] = []
        dopant = comp.get("dopant") or "undoped"
        reasons.append(f"{dopant} candidate balances Mn redox capacity with Fe structural stabilization.")
        if metrics.get("fade_500", 1.0) <= self.target.max_fade_500:
            reasons.append("Predicted 500-cycle fade clears the requested high-temperature target.")
        if metrics.get("phase_stability", 0.0) >= self.target.min_phase_stability:
            reasons.append("Phase stability score is high enough for a first-pass layered oxide experiment.")
        failed = [c.name for c in constraints if not c.passed]
        if failed:
            reasons.append("Watch items: " + ", ".join(failed) + ".")
        else:
            reasons.append("All current hard design gates pass in the surrogate audit.")
        return reasons

    def _measurements(self, metrics: Dict[str, Any], constraints: Sequence[DesignConstraint]) -> List[str]:
        measurements = [
            "XRD phase identification after calcination and after first electrochemical cycle",
            "ICP-OES or XRF confirmation of Na:Mn:Fe:dopant ratio",
            "Coin-cell C/10 formation, C/2 cycling to 100 cycles, impedance every 25 cycles",
            "45 C accelerated cycling for first 50 cycles to validate Arrhenius fade slope",
        ]
        if any(c.name == "thermal onset" and not c.passed for c in constraints):
            measurements.append("DSC/TGA thermal onset scan before scale-up")
        if metrics.get("uncertainty", 0.0) > 12.0:
            measurements.append("Duplicate synthesis batch to separate model uncertainty from process variance")
        return measurements

    def _kill_criteria(self, metrics: Dict[str, Any]) -> List[str]:
        return [
            "Reject if XRD shows dominant impurity phase above 8 percent relative intensity.",
            "Reject if first-cycle discharge capacity is below 85 percent of predicted Q0.",
            "Reject if capacity retention at 50 cycles is worse than the model's 95 percent lower bound.",
            "Reject if charge-transfer resistance doubles before cycle 25 at 45 C.",
        ]


def candidate_to_dict(candidate: CandidateExperiment) -> Dict[str, Any]:
    payload = asdict(candidate)
    payload["constraints"] = [asdict(c) for c in candidate.constraints]
    return payload


def run_inverse_design(out_path: Optional[Path] = None, n_return: int = 8) -> List[Dict[str, Any]]:
    designer = InverseCathodeDesigner()
    candidates = designer.search(n_return=n_return)
    payload = [candidate_to_dict(c) for c in candidates]
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/inverse_cathode_design_v2.json")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    payload = run_inverse_design(Path(args.out), n_return=args.n)
    print(json.dumps({"candidates": len(payload), "top": payload[0] if payload else None}, indent=2))


if __name__ == "__main__":
    main()
