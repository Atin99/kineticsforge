import numpy as np
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from core.india_context import IndiaOperatingContext
from modules.cathode.defect_chemistry import DefectChemistryModel
from modules.cathode.composition_sampler import get_100_compositions, initial_capacity_prior, cycle_life_prior

R_GAS = 8.314
F_CONST = 96485.0
K_B = 8.617e-5

ELEMENT_COSTS_USD_KG = {
    "Na": 3.1, "Mn": 2.4, "Fe": 0.45, "O": 0.0,
    "Al": 2.7, "Ti": 11.0, "Mg": 2.8, "Ni": 18.5,
    "Co": 35.0, "Li": 75.0, "V": 29.0, "Cr": 9.5,
}

DOPANT_EFFECTS = {
    "Al":  (0.82, 1.18, 0.97, 0.85, 1.05),
    "Ti":  (0.90, 1.10, 0.99, 0.90, 1.08),
    "Mg":  (0.94, 1.05, 1.02, 0.92, 1.02),
    "V":   (0.88, 1.12, 0.98, 0.88, 1.06),
    "Cr":  (0.91, 1.08, 0.97, 0.87, 1.03),
    None:  (1.00, 1.00, 1.00, 1.00, 1.00),
}


def _arrhenius_fade_rate(comp: Dict, T: float) -> float:
    Ea = 0.55 + 0.1 * comp["Mn"] - 0.03 * comp["Fe"]
    k0 = 1e-4 * (1.0 + 0.2 * comp["Fe"])
    return k0 * np.exp(-Ea * F_CONST / (R_GAS * T))


def _jahn_teller_factor(mn_frac: float) -> float:
    return 1.0 + 0.3 * max(0.0, mn_frac - 0.5)


def _structural_stability(mn_frac: float) -> float:
    return 1.0 / (1.0 + np.exp(-8.0 * (0.5 - mn_frac)))


def _na_mobility(na_frac: float) -> float:
    return 1.0 - 0.5 * abs(na_frac - 1.0)


class CostModel:
    def __init__(self, costs: Optional[Dict[str, float]] = None):
        self.costs = costs or ELEMENT_COSTS_USD_KG

    def cathode_cost_per_kg(self, comp: Dict) -> float:
        total = 0.0
        total += comp.get("Na", 0) * self.costs.get("Na", 3.1) * 0.23
        total += comp.get("Mn", 0) * self.costs.get("Mn", 2.4) * 0.55
        total += comp.get("Fe", 0) * self.costs.get("Fe", 0.45) * 0.56
        dopant = comp.get("dopant")
        dopant_frac = comp.get("dopant_frac", 0.0)
        if dopant and dopant in self.costs:
            mw = {"Al": 27.0, "Ti": 47.9, "Mg": 24.3, "V": 50.9, "Cr": 52.0}.get(dopant, 40.0)
            total += dopant_frac * self.costs[dopant] * (mw / 100.0)
        total += 2.5
        return total

    def cost_per_kwh(self, comp: Dict, energy_density_wh_kg: float) -> float:
        if energy_density_wh_kg < 1.0:
            return 9999.0
        return self.cathode_cost_per_kg(comp) / (energy_density_wh_kg / 1000.0)


class PhaseStabilityCalculator:
    def __init__(self):
        self.formation_energies = {
            "NaMnO2": -4.82, "NaFeO2": -4.55, "NaMn0.5Fe0.5O2": -4.68,
            "Na2MnO3": -5.10, "Na2FeO3": -4.90, "MnO2": -3.45,
            "Fe2O3": -3.12, "Na2O": -2.58, "MnO": -3.90, "FeO": -2.72,
        }

    def formation_energy_surrogate(self, comp: Dict) -> float:
        mn, fe, na = comp["Mn"], comp["Fe"], comp["Na"]
        e_base = -4.2 - 0.6 * mn - 0.35 * fe - 0.4 * na
        dopant = comp.get("dopant")
        dopant_bonus = {"Al": -0.12, "Ti": -0.18, "Mg": -0.08, "V": -0.15, "Cr": -0.10}
        if dopant:
            e_base += dopant_bonus.get(dopant, 0.0) * comp.get("dopant_frac", 0.0) * 10
        return e_base

    def decomposition_energy(self, comp: Dict) -> float:
        e_form = self.formation_energy_surrogate(comp)
        e_hull_products = -4.0 - 0.3 * comp["Mn"] - 0.2 * comp["Fe"]
        return e_form - e_hull_products

    def ehull(self, comp: Dict) -> float:
        return max(0.0, self.decomposition_energy(comp) + 0.05)

    def thermodynamic_stability_score(self, comp: Dict) -> float:
        eh = self.ehull(comp)
        return 1.0 / (1.0 + np.exp(20.0 * (eh - 0.05)))


class ThermalAbuseScreener:
    def __init__(self, onset_base: float = 250.0):
        self.onset_base = onset_base

    def onset_temperature(self, comp: Dict) -> float:
        mn_penalty = -30.0 * max(0.0, comp["Mn"] - 0.5)
        fe_bonus = 15.0 * comp["Fe"]
        dopant = comp.get("dopant")
        dopant_bonus = {"Al": 20.0, "Ti": 25.0, "Mg": 10.0, "V": 15.0, "Cr": 12.0}
        d_bonus = dopant_bonus.get(dopant, 0.0) * comp.get("dopant_frac", 0.0) * 10
        return self.onset_base + mn_penalty + fe_bonus + d_bonus

    def thermal_stability_score(self, comp: Dict) -> float:
        t_onset = self.onset_temperature(comp)
        return min(1.0, max(0.0, (t_onset - 180.0) / 120.0))


class SynthesizabilityScore:
    ROUTES = {"solid_state": 0, "sol_gel": 1, "coprecipitation": 2, "hydrothermal": 3}

    def score(self, comp: Dict) -> Dict[str, float]:
        mn, fe = comp["Mn"], comp["Fe"]
        dopant = comp.get("dopant")
        dopant_frac = comp.get("dopant_frac", 0.0)
        ss = 0.7 + 0.1 * fe - 0.15 * max(0.0, mn - 0.6)
        if dopant in ("Ti", "V"):
            ss -= 0.05
        sg = 0.8 - 0.1 * abs(mn - fe) - 0.05 * (dopant_frac * 10)
        cp = 0.85 - 0.08 * abs(mn - 0.5) - 0.03 * dopant_frac * 10
        ht = 0.6 + 0.15 * fe - 0.1 * mn
        return {
            "solid_state": float(np.clip(ss, 0.1, 1.0)),
            "sol_gel": float(np.clip(sg, 0.1, 1.0)),
            "coprecipitation": float(np.clip(cp, 0.1, 1.0)),
            "hydrothermal": float(np.clip(ht, 0.1, 1.0)),
            "best_route": max({"solid_state": ss, "sol_gel": sg,
                               "coprecipitation": cp, "hydrothermal": ht}.items(),
                              key=lambda x: x[1])[0],
        }


class ElectrolyteCompatibility:
    def score(self, comp: Dict) -> float:
        mn = comp["Mn"]
        fe = comp["Fe"]
        dopant = comp.get("dopant")
        base = 0.85
        mn_dissolution = -0.15 * max(0.0, mn - 0.5)
        fe_redox = 0.05 * fe
        dopant_effect = {"Al": 0.05, "Ti": 0.08, "Mg": 0.03, "V": 0.04, "Cr": 0.02}
        d_eff = dopant_effect.get(dopant, 0.0) * comp.get("dopant_frac", 0.0) * 10
        return float(np.clip(base + mn_dissolution + fe_redox + d_eff, 0.1, 1.0))


def _score_composition(comp: Dict, T: float, cost_model: CostModel,
                        phase_calc: PhaseStabilityCalculator,
                        thermal_screener: ThermalAbuseScreener,
                        synth_scorer: SynthesizabilityScore,
                        elyte_compat: ElectrolyteCompatibility,
                        n_mc: int = 50) -> Dict:
    q0 = initial_capacity_prior(comp)
    cl = cycle_life_prior(comp)
    k_fade = _arrhenius_fade_rate(comp, T)
    jt = _jahn_teller_factor(comp["Mn"])
    ss = _structural_stability(comp["Mn"])
    na_mob = _na_mobility(comp["Na"])
    fe_stab = 0.9 + 0.2 * comp["Fe"]

    fade_mult, life_mult, cap_mult, vol_mult, rate_mult = DOPANT_EFFECTS.get(comp["dopant"], (1, 1, 1, 1, 1))

    eff_fade = k_fade * jt * fade_mult / fe_stab
    fade_500 = float(np.clip(1.0 - np.exp(-eff_fade * 500 ** 1.15), 0.02, 0.48))
    q0_adj = q0 * cap_mult * na_mob
    q_500 = q0_adj * (1.0 - fade_500)
    cl_adj = int(cl * life_mult / jt)

    rate_cap = (0.85 + 0.1 * comp["Fe"] - 0.05 * comp["Mn"]) * rate_mult
    if comp["dopant"] == "Al":
        rate_cap += 0.03

    thermal_stab = ss * fe_stab
    ce = 0.995 - 0.005 * comp["Mn"] + 0.002 * comp["Fe"]
    if comp["dopant"]:
        ce += 0.001

    avg_voltage = 3.3 + 0.2 * comp["Fe"] - 0.1 * comp["Mn"]
    energy_density = q0_adj * avg_voltage
    vol_change = (2.0 + 3.0 * comp["Mn"] - 1.0 * comp["Fe"]) * vol_mult
    if comp["dopant"] == "Ti":
        vol_change *= 0.85

    q0_samples = np.array([initial_capacity_prior(comp) for _ in range(n_mc)])
    uncertainty = float(np.std(q0_samples))
    epistemic_unc = uncertainty * (1.0 + 0.5 * abs(comp["Mn"] - 0.5))

    phase_stab = phase_calc.thermodynamic_stability_score(comp)
    thermal_abuse = thermal_screener.thermal_stability_score(comp)
    defect = DefectChemistryModel().evaluate(comp)
    synth = synth_scorer.score(comp)
    elyte = elyte_compat.score(comp)
    cost_kg = cost_model.cathode_cost_per_kg(comp)
    cost_kwh = cost_model.cost_per_kwh(comp, energy_density)
    india = IndiaOperatingContext.from_env()
    cost_inr_kg = india.usd_to_rupees(cost_kg)
    cost_inr_kwh = india.usd_to_rupees(cost_kwh)

    synth_scores = [v for k, v in synth.items() if k != "best_route"]
    score = (0.15 * (q0_adj / 180.0) +
             0.18 * (1.0 - fade_500) +
             0.10 * thermal_stab +
             0.10 * (cl_adj / 600.0) +
             0.08 * rate_cap +
             0.07 * ce +
             0.04 * (1.0 - vol_change / 10.0) +
             0.08 * phase_stab +
             0.06 * thermal_abuse +
             0.06 * max(synth_scores) +
             0.04 * elyte +
             0.04 * max(0, 1.0 - cost_kwh / 200.0) +
             0.05 * defect.defect_tolerance_score)

    return {
        "comp": comp,
        "Q0": float(q0_adj),
        "Q_500": float(q_500),
        "fade_500": fade_500,
        "fade": fade_500,
        "cycle_life": cl_adj,
        "score": float(score),
        "uncertainty": float(epistemic_unc),
        "thermal_stability": float(thermal_stab),
        "rate_capability": float(rate_cap),
        "coulombic_efficiency": float(ce),
        "energy_density": float(energy_density),
        "volumetric_change": float(vol_change),
        "Ea_effective": float(0.55 + 0.1 * comp["Mn"]),
        "jahn_teller_factor": float(jt),
        "phase_stability": float(phase_stab),
        "thermal_abuse_onset_C": float(thermal_screener.onset_temperature(comp)),
        "thermal_abuse_score": float(thermal_abuse),
        "synthesizability": synth,
        "electrolyte_compatibility": float(elyte),
        "defect_tolerance_score": float(defect.defect_tolerance_score),
        "oxygen_redox_risk": float(defect.oxygen_redox_risk),
        "tm_mixing_risk": float(defect.transition_metal_mixing_risk),
        "moisture_sensitivity": float(defect.moisture_sensitivity),
        "charge_balance_error": float(defect.charge_balance_error),
        "defect_compensation": defect.suggested_compensation,
        "cost_usd_kg": float(cost_kg),
        "cost_usd_kwh": float(cost_kwh),
        "cost_inr_kg": float(cost_inr_kg),
        "cost_inr_kwh": float(cost_inr_kwh),
        "cost_index_india": float(india.normalized_cost_index(cost_inr_kwh)),
        "india_context": {
            "operating_temperature_C": float(T - 273.15),
            "ambient_reference_C": float(india.ambient_hot_C),
            "usd_to_inr_assumption": float(india.usd_to_inr),
        },
        "avg_voltage": float(avg_voltage),
    }


def screen_compositions(n=100, T=318) -> List[Dict]:
    comps = get_100_compositions()[:n]
    cost_model = CostModel()
    phase_calc = PhaseStabilityCalculator()
    thermal_screener = ThermalAbuseScreener()
    synth_scorer = SynthesizabilityScore()
    elyte_compat = ElectrolyteCompatibility()

    results = []
    for i, comp in enumerate(comps):
        result = _score_composition(comp, T, cost_model, phase_calc,
                                     thermal_screener, synth_scorer, elyte_compat)
        result["comp_id"] = i
        results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


class ParetoFrontScreener:
    def __init__(self, objectives: Optional[List[str]] = None, directions: Optional[List[str]] = None):
        self.objectives = objectives or ["Q0", "fade_500", "cycle_life", "cost_usd_kwh"]
        self.directions = directions or ["max", "min", "max", "min"]

    def _dominates(self, a: Dict, b: Dict) -> bool:
        dominated = False
        for obj, d in zip(self.objectives, self.directions):
            va, vb = a.get(obj, 0), b.get(obj, 0)
            if d == "min":
                va, vb = -va, -vb
            if va < vb:
                return False
            if va > vb:
                dominated = True
        return dominated

    def nondominated_sort(self, results: List[Dict]) -> List[List[int]]:
        n = len(results)
        domination_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(results[i], results[j]):
                    dominated_set[i].append(j)
                elif self._dominates(results[j], results[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def crowding_distance(self, results: List[Dict], front: List[int]) -> np.ndarray:
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        for obj in self.objectives:
            vals = np.array([results[front[i]].get(obj, 0.0) for i in range(n)])
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            obj_range = vals[order[-1]] - vals[order[0]]
            if obj_range < 1e-12:
                continue
            for k in range(1, n - 1):
                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / obj_range
        return dist

    def screen(self, results: List[Dict], top_n: int = 20) -> List[Dict]:
        fronts = self.nondominated_sort(results)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= top_n:
                for idx in front:
                    results[idx]["pareto_rank"] = len(selected) // max(len(front), 1)
                    selected.append(results[idx])
            else:
                cd = self.crowding_distance(results, front)
                remaining = top_n - len(selected)
                top_cd = np.argsort(cd)[-remaining:]
                for k in top_cd:
                    results[front[k]]["pareto_rank"] = len(fronts)
                    results[front[k]]["crowding_distance"] = float(cd[k])
                    selected.append(results[front[k]])
                break
        return selected


class BayesianCompositionOptimizer:
    def __init__(self, bounds: Optional[Dict] = None, n_initial: int = 20, seed: int = 42):
        self.bounds = bounds or {
            "Na": (0.8, 1.05), "Mn": (0.2, 0.8), "Fe": (0.2, 0.8),
            "dopant_frac": (0.0, 0.08),
        }
        self.n_initial = n_initial
        self.rng = np.random.RandomState(seed)
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.kernel_lengthscale = 0.3
        self.kernel_variance = 1.0
        self.noise_variance = 0.01

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
        return self.kernel_variance * np.exp(-0.5 * sq_dist / (self.kernel_lengthscale ** 2))

    def _gp_predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.X_observed) < 2:
            return np.zeros(len(X_new)), np.ones(len(X_new)) * self.kernel_variance
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        K = self._rbf_kernel(X_obs, X_obs) + self.noise_variance * np.eye(len(X_obs))
        K_inv = np.linalg.solve(K, np.eye(len(K)))
        K_star = self._rbf_kernel(X_new, X_obs)
        K_ss = self._rbf_kernel(X_new, X_new)
        mu = K_star @ K_inv @ y_obs
        var = np.diag(K_ss - K_star @ K_inv @ K_star.T)
        return mu, np.maximum(var, 1e-8)

    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray, best: float,
                               xi: float = 0.01) -> np.ndarray:
        sigma = np.maximum(sigma, 1e-8)
        z = (mu - best - xi) / sigma
        from scipy.stats import norm
        return sigma * (z * norm.cdf(z) + norm.pdf(z))

    def suggest_next(self, n_candidates: int = 500) -> Dict:
        candidates = []
        for _ in range(n_candidates):
            x = np.array([
                self.rng.uniform(*self.bounds["Na"]),
                self.rng.uniform(*self.bounds["Mn"]),
                self.rng.uniform(*self.bounds["Fe"]),
                self.rng.uniform(*self.bounds["dopant_frac"]),
            ])
            total_tm = x[1] + x[2] + x[3]
            if total_tm > 1.05 or total_tm < 0.5:
                continue
            candidates.append(x)
        candidates = np.array(candidates) if candidates else self.rng.rand(10, 4)
        mu, var = self._gp_predict(candidates)
        sigma = np.sqrt(var)
        best_y = max(self.y_observed) if self.y_observed else 0.0
        ei = self._expected_improvement(mu, sigma, best_y)
        best_idx = np.argmax(ei)
        best_x = candidates[best_idx]
        dopants = [None, "Al", "Ti", "Mg", "V", "Cr"]
        dopant = dopants[int(best_x[3] * 10) % len(dopants)] if best_x[3] > 0.01 else None
        return {
            "Na": float(best_x[0]),
            "Mn": float(best_x[1]),
            "Fe": float(best_x[2]),
            "dopant": dopant,
            "dopant_frac": float(best_x[3]),
            "expected_improvement": float(ei[best_idx]),
            "predicted_score": float(mu[best_idx]),
            "predicted_uncertainty": float(sigma[best_idx]),
        }

    def observe(self, comp_vec: np.ndarray, score: float):
        self.X_observed.append(comp_vec)
        self.y_observed.append(score)

    def optimize(self, eval_fn, n_iterations: int = 50) -> List[Dict]:
        history = []
        for it in range(n_iterations):
            suggestion = self.suggest_next()
            comp = {
                "Na": suggestion["Na"], "Mn": suggestion["Mn"], "Fe": suggestion["Fe"],
                "dopant": suggestion["dopant"], "dopant_frac": suggestion["dopant_frac"],
            }
            result = eval_fn(comp)
            score = result.get("score", 0.0)
            vec = np.array([comp["Na"], comp["Mn"], comp["Fe"], comp["dopant_frac"]])
            self.observe(vec, score)
            suggestion["iteration"] = it
            suggestion["observed_score"] = score
            history.append(suggestion)
        return history


class QNEHVICompositionOptimizer(BayesianCompositionOptimizer):
    def __init__(self, bounds: Optional[Dict] = None, n_initial: int = 20, seed: int = 42, reference_point: Optional[List[float]] = None):
        super().__init__(bounds=bounds, n_initial=n_initial, seed=seed)
        self.reference_point = np.array(reference_point or [80.0, -0.45, 80.0, -220.0], dtype=float)
        self.Y_observed: List[np.ndarray] = []

    @staticmethod
    def objective_vector(result: Dict) -> np.ndarray:
        return np.array([
            float(result.get("Q0", result.get("capacity", 0.0))),
            -float(result.get("fade_500", 1.0)),
            float(result.get("cycle_life", 0.0)),
            -float(result.get("cost_usd_kwh", 999.0)),
        ], dtype=float)

    def observe_multi(self, comp_vec: np.ndarray, result: Dict) -> None:
        score = float(result.get("score", 0.0))
        self.observe(comp_vec, score)
        self.Y_observed.append(self.objective_vector(result))

    def _pareto_mask(self, Y: np.ndarray) -> np.ndarray:
        mask = np.ones(len(Y), dtype=bool)
        for i in range(len(Y)):
            if not mask[i]:
                continue
            dominates_i = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
            if dominates_i.any():
                mask[i] = False
        return mask

    def _hvi_proxy(self, candidates: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        if not self.Y_observed:
            return mu + 0.25 * sigma
        Y = np.array(self.Y_observed)
        pareto = Y[self._pareto_mask(Y)]
        current = np.maximum(pareto - self.reference_point[None, :], 0.0).prod(axis=1).max(initial=0.0)
        improvements = []
        for i, x in enumerate(candidates):
            synthetic = np.array([mu[i] * 160.0, -0.08 / (1.0 + sigma[i]), mu[i] * 900.0, -160.0 * (1.0 + x[3])])
            hv = max(current, float(np.maximum(synthetic - self.reference_point, 0.0).prod()))
            improvements.append(hv - current + 0.05 * sigma[i])
        return np.array(improvements)

    def suggest_next(self, n_candidates: int = 700) -> Dict:
        candidates = []
        for _ in range(n_candidates):
            x = np.array([
                self.rng.uniform(*self.bounds["Na"]),
                self.rng.uniform(*self.bounds["Mn"]),
                self.rng.uniform(*self.bounds["Fe"]),
                self.rng.uniform(*self.bounds["dopant_frac"]),
            ])
            total_tm = x[1] + x[2] + x[3]
            if 0.5 <= total_tm <= 1.05:
                candidates.append(x)
        candidates = np.array(candidates) if candidates else self.rng.rand(10, 4)
        mu, var = self._gp_predict(candidates)
        sigma = np.sqrt(var)
        acquisition = self._hvi_proxy(candidates, mu, sigma)
        best_idx = int(np.argmax(acquisition))
        best_x = candidates[best_idx]
        dopants = [None, "Al", "Ti", "Mg", "V", "Cr"]
        dopant = dopants[int(best_x[3] * 10) % len(dopants)] if best_x[3] > 0.01 else None
        return {
            "Na": float(best_x[0]),
            "Mn": float(best_x[1]),
            "Fe": float(best_x[2]),
            "dopant": dopant,
            "dopant_frac": float(best_x[3]),
            "qnehvi_proxy": float(acquisition[best_idx]),
            "predicted_score": float(mu[best_idx]),
            "predicted_uncertainty": float(sigma[best_idx]),
            "acquisition": "qNEHVI-style noisy hypervolume improvement proxy; use BoTorch qNEHVI on Kaggle when available",
        }

    def optimize(self, eval_fn, n_iterations: int = 50) -> List[Dict]:
        history = []
        for it in range(n_iterations):
            suggestion = self.suggest_next()
            comp = {
                "Na": suggestion["Na"], "Mn": suggestion["Mn"], "Fe": suggestion["Fe"],
                "dopant": suggestion["dopant"], "dopant_frac": suggestion["dopant_frac"],
            }
            result = eval_fn(comp)
            vec = np.array([comp["Na"], comp["Mn"], comp["Fe"], comp["dopant_frac"]])
            self.observe_multi(vec, result)
            suggestion["iteration"] = it
            suggestion["observed_objectives"] = self.objective_vector(result).tolist()
            suggestion["observed_score"] = float(result.get("score", 0.0))
            history.append(suggestion)
        return history
