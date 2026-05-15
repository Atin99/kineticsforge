import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.india_context import IndiaOperatingContext
from modules.cathode.screener import screen_compositions


@dataclass
class FeedstockAssay:
    mass_kg: float = 100.0
    mn_wt: float = 0.22
    fe_wt: float = 0.11
    na_wt: float = 0.05
    al_wt: float = 0.04
    cu_wt: float = 0.015
    moisture_wt: float = 0.03
    black_mass_price_inr_kg: float = 150.0


@dataclass
class LeachCondition:
    temperature_K: float
    pH: float
    acid_M: float
    time_min: float
    particle_um: float = 50.0


@dataclass
class RecoveredStream:
    mn_kg: float
    fe_kg: float
    na_kg: float
    impurity_kg: float
    purity: float
    recovery_score: float
    reagent_cost_inr: float
    heat_cost_inr: float
    reagent_cost_usd: float
    heat_cost_usd: float
    waste_index: float


@dataclass
class RecoveryPrior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return float(self.alpha / max(self.alpha + self.beta, 1e-12))

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return float((self.alpha * self.beta) / max(total * total * (total + 1.0), 1e-12))

    def update(self, observed_recovery: float, n_trials: float = 1.0) -> "RecoveryPrior":
        observed = float(np.clip(observed_recovery, 0.0, 1.0))
        trials = max(float(n_trials), 1e-9)
        return RecoveryPrior(alpha=self.alpha + observed * trials, beta=self.beta + (1.0 - observed) * trials)


@dataclass
class ClosedLoopPlan:
    target_composition: Dict[str, Any]
    condition: LeachCondition
    stream: RecoveredStream
    makeable_mass_kg: float
    limiting_element: str
    economics: Dict[str, float]
    validation_gates: List[str] = field(default_factory=list)
    operational_notes: List[str] = field(default_factory=list)


class ClosedLoopOptimizer:
    def __init__(self, feedstock: Optional[FeedstockAssay] = None, seed: int = 7):
        self.feedstock = feedstock or FeedstockAssay()
        self.rng = np.random.RandomState(seed)
        self.india = IndiaOperatingContext.from_env()
        self.recovery_priors = {
            "Mn": RecoveryPrior(alpha=8.8, beta=1.2),
            "Fe": RecoveryPrior(alpha=7.2, beta=2.8),
            "Na": RecoveryPrior(alpha=6.5, beta=3.5),
        }

    def update_from_outcome(self, element: str, observed_recovery: float, n_trials: float = 1.0) -> None:
        key = element.strip().title()
        if key not in self.recovery_priors:
            raise ValueError(f"Unsupported recovery prior element: {element}")
        self.recovery_priors[key] = self.recovery_priors[key].update(observed_recovery, n_trials=n_trials)

    def update_many_from_outcomes(self, outcomes: Dict[str, float], n_trials: float = 1.0) -> None:
        for element, recovery in outcomes.items():
            self.update_from_outcome(element, recovery, n_trials=n_trials)

    def prior_summary(self) -> Dict[str, Dict[str, float]]:
        return {k: {"alpha": v.alpha, "beta": v.beta, "mean": v.mean, "variance": v.variance} for k, v in self.recovery_priors.items()}

    def estimate_stream(self, condition: LeachCondition) -> RecoveredStream:
        f = self.feedstock
        temp_C = condition.temperature_K - 273.15
        temp_factor = np.clip((temp_C - 45.0) / 45.0, 0.0, 1.2)
        acidity_factor = np.clip((3.2 - condition.pH) / 3.0, 0.0, 1.2)
        conc_factor = np.clip(condition.acid_M / 2.5, 0.0, 1.3)
        time_factor = 1.0 - np.exp(-condition.time_min / 72.0)
        particle_factor = np.clip((90.0 / max(condition.particle_um, 5.0)) ** 0.35, 0.55, 1.35)
        base = temp_factor * acidity_factor * conc_factor * time_factor * particle_factor
        mn_model = float(np.clip(0.46 + 0.48 * base, 0.0, 0.96))
        fe_model = float(np.clip(0.38 + 0.42 * base - 0.05 * acidity_factor, 0.0, 0.90))
        na_model = float(np.clip(0.62 + 0.30 * acidity_factor * time_factor, 0.0, 0.98))
        mn_rec = float(np.clip(0.75 * mn_model + 0.25 * self.recovery_priors["Mn"].mean, 0.0, 0.98))
        fe_rec = float(np.clip(0.75 * fe_model + 0.25 * self.recovery_priors["Fe"].mean, 0.0, 0.95))
        na_rec = float(np.clip(0.75 * na_model + 0.25 * self.recovery_priors["Na"].mean, 0.0, 0.99))
        impurity_pull = float(np.clip(0.10 + 0.35 * acidity_factor + 0.12 * temp_factor, 0.0, 0.72))
        mn = f.mass_kg * f.mn_wt * mn_rec
        fe = f.mass_kg * f.fe_wt * fe_rec
        na = f.mass_kg * f.na_wt * na_rec
        impurity = f.mass_kg * (f.al_wt + f.cu_wt) * impurity_pull
        purity = (mn + fe + na) / max(mn + fe + na + impurity, 1e-9)
        acid_kg = condition.acid_M * 0.098 * f.mass_kg
        reagent_cost = acid_kg * self.india.sulphuric_acid_inr_kg + max(0.0, 2.0 - condition.pH) * 90.0
        heat_cost = self.india.heat_cost_inr(f.mass_kg, max(0.0, condition.temperature_K - 298.15))
        waste = impurity / max(mn + fe + na, 1e-9) + 0.015 * condition.acid_M + 0.002 * max(0.0, temp_C - 70.0)
        recovery_score = 0.5 * mn_rec + 0.3 * fe_rec + 0.2 * na_rec
        return RecoveredStream(
            mn_kg=float(mn),
            fe_kg=float(fe),
            na_kg=float(na),
            impurity_kg=float(impurity),
            purity=float(purity),
            recovery_score=float(recovery_score),
            reagent_cost_inr=float(reagent_cost),
            heat_cost_inr=float(heat_cost),
            reagent_cost_usd=float(self.india.rupees_to_usd(reagent_cost)),
            heat_cost_usd=float(self.india.rupees_to_usd(heat_cost)),
            waste_index=float(waste),
        )

    def candidate_conditions(self, n: int = 160) -> List[LeachCondition]:
        base = [
            LeachCondition(343.15, 1.2, 1.8, 95.0, 50.0),
            LeachCondition(353.15, 1.0, 2.0, 110.0, 35.0),
            LeachCondition(333.15, 1.6, 1.4, 125.0, 60.0),
        ]
        for _ in range(max(0, n - len(base))):
            base.append(
                LeachCondition(
                    temperature_K=float(self.rng.uniform(323.15, 363.15)),
                    pH=float(self.rng.uniform(0.6, 2.8)),
                    acid_M=float(self.rng.uniform(0.8, 3.2)),
                    time_min=float(self.rng.uniform(45.0, 180.0)),
                    particle_um=float(self.rng.uniform(15.0, 110.0)),
                )
            )
        return base

    def condition_score(self, stream: RecoveredStream) -> float:
        return (
            0.42 * stream.recovery_score
            + 0.24 * stream.purity
            - 0.16 * stream.waste_index
            - 0.10 * stream.reagent_cost_inr / max(self.feedstock.mass_kg * 100.0, 1.0)
            - 0.08 * stream.heat_cost_inr / max(self.feedstock.mass_kg * 100.0, 1.0)
        )

    def choose_condition(self) -> tuple:
        best = None
        for condition in self.candidate_conditions():
            stream = self.estimate_stream(condition)
            score = self.condition_score(stream)
            item = (score, condition, stream)
            if best is None or score > best[0]:
                best = item
        return best

    def makeable_mass(self, comp: Dict[str, Any], stream: RecoveredStream) -> tuple:
        mn_req = max(float(comp.get("Mn", 0.0)), 1e-9)
        fe_req = max(float(comp.get("Fe", 0.0)), 1e-9)
        na_req = max(float(comp.get("Na", 1.0)) * 0.42, 1e-9)
        limits = {
            "Mn": stream.mn_kg / mn_req,
            "Fe": stream.fe_kg / fe_req,
            "Na": stream.na_kg / na_req,
        }
        limiting = min(limits, key=limits.get)
        return float(limits[limiting]), limiting

    def rank_cathode_loop(self, top_n: int = 6) -> List[ClosedLoopPlan]:
        _, condition, stream = self.choose_condition()
        cathodes = screen_compositions(n=100, T=318)
        plans: List[ClosedLoopPlan] = []
        for item in cathodes[:30]:
            comp = item["comp"]
            makeable, limiting = self.makeable_mass(comp, stream)
            material_value = makeable * max(item.get("energy_density", 0.0), 1.0) / 1000.0
            process_cost_inr = stream.reagent_cost_inr + stream.heat_cost_inr + self.feedstock.black_mass_price_inr_kg * self.feedstock.mass_kg
            margin_proxy_inr = material_value * 2200.0 - process_cost_inr
            loop_score = item["score"] + 0.10 * stream.purity + 0.04 * np.log1p(max(makeable, 0.0)) - 0.05 * stream.waste_index
            economics = {
                "loop_score": float(loop_score),
                "process_cost_inr": float(process_cost_inr),
                "margin_proxy_inr": float(margin_proxy_inr),
                "material_value_proxy_kwh": float(material_value),
                "purity": float(stream.purity),
                "usd_to_inr_assumption": float(self.india.usd_to_inr),
                "recovery_prior_Mn_mean": float(self.recovery_priors["Mn"].mean),
                "recovery_prior_Fe_mean": float(self.recovery_priors["Fe"].mean),
                "recovery_prior_Na_mean": float(self.recovery_priors["Na"].mean),
            }
            plans.append(
                ClosedLoopPlan(
                    target_composition=comp,
                    condition=condition,
                    stream=stream,
                    makeable_mass_kg=makeable,
                    limiting_element=limiting,
                    economics=economics,
                    validation_gates=self.validation_gates(comp, stream),
                    operational_notes=self.operational_notes(comp, condition, stream),
                )
            )
        plans.sort(key=lambda p: p.economics["loop_score"], reverse=True)
        return plans[:top_n]

    def validation_gates(self, comp: Dict[str, Any], stream: RecoveredStream) -> List[str]:
        return [
            "ICP-OES recovered liquor assay must confirm Mn and Fe recovery within 7 percent absolute of model output.",
            "Impurity-to-transition-metal mass ratio must stay below 0.08 before cathode precursor synthesis.",
            "Recovered salt batch must produce the same XRD dominant phase as virgin salts.",
            "First-cycle capacity from recovered feedstock must stay above 90 percent of virgin-feedstock control.",
        ]

    def operational_notes(self, comp: Dict[str, Any], condition: LeachCondition, stream: RecoveredStream) -> List[str]:
        notes = [
            f"Run leach at {condition.temperature_K - 273.15:.1f} C, pH {condition.pH:.2f}, acid {condition.acid_M:.2f} M for {condition.time_min:.0f} min.",
            f"Current purity estimate is {100.0 * stream.purity:.1f} percent; solvent extraction or precipitation cleanup is needed before high-confidence cathode synthesis.",
            "Use the recovered Mn:Fe ratio as a feed constraint, not as a guaranteed product stoichiometry.",
        ]
        if comp.get("dopant"):
            notes.append(f"Add {comp['dopant']} from virgin precursor until recycling stream dopant recovery is experimentally measured.")
        return notes


def plan_to_dict(plan: ClosedLoopPlan) -> Dict[str, Any]:
    return {
        "target_composition": plan.target_composition,
        "condition": asdict(plan.condition),
        "stream": asdict(plan.stream),
        "makeable_mass_kg": plan.makeable_mass_kg,
        "limiting_element": plan.limiting_element,
        "economics": plan.economics,
        "validation_gates": plan.validation_gates,
        "operational_notes": plan.operational_notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/closed_loop_recycling_v2.json")
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()
    optimizer = ClosedLoopOptimizer()
    plans = [plan_to_dict(p) for p in optimizer.rank_cathode_loop(top_n=args.top_n)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    print(json.dumps({"plans": len(plans), "out": str(out), "top_limiting": plans[0]["limiting_element"] if plans else ""}, indent=2))


if __name__ == "__main__":
    main()
