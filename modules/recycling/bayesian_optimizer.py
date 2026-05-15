import torch
import numpy as np
import optuna
import json
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

REAGENT_COSTS = {"H2SO4": 0.08, "HCl": 0.12, "HNO3": 0.25, "citric_acid": 0.95}
ENERGY_COST_KWH = 0.07


class LeachingODESolver:
    def __init__(self):
        self.D0_Mn = 3.5e-8
        self.D0_Fe = 1.8e-8
        self.D0_Na = 4.2e-8
        self.Ea_D = 0.24
        self.rho = 5000.0
        self.R = 8.314
        self.k_A0 = torch.tensor([0.75, 0.42, 0.90])
        self.n_avrami = torch.tensor([2.0, 1.8, 2.5])

    def D_eff(self, T, species_idx):
        D0s = [self.D0_Mn, self.D0_Fe, self.D0_Na]
        return D0s[species_idx] * np.exp(-self.Ea_D * 96485 / (self.R * T))

    def shrinking_core_rate(self, alpha, T, c_acid, r0, species_idx):
        D = self.D_eff(T, species_idx)
        if alpha >= 0.999:
            return 0.0
        acidity = 0.65 + 0.35 * np.log1p(c_acid) / np.log(4.0)
        return acidity * (3 * D * c_acid * 1000.0) / (r0**2 * self.rho * max((1 - alpha)**(1/3), 1e-6))

    def avrami_rate(self, t, T, pH, species_idx):
        k = self.k_A0[species_idx].item() * np.exp((T - 323.0) / 55.0) * (1 + 0.42 * (3.0 - pH))
        n = self.n_avrami[species_idx].item()
        if t <= 0:
            return 0.0
        tau = max(t / 180.0, 1e-6)
        return (k * n * tau**(n-1) * np.exp(-k * tau**n)) / 180.0

    def blend_gamma(self, T, pH, r0):
        x1 = (T - 323) / 40
        x2 = (pH - 0.5) / 2.5
        x3 = (r0 - 10) / 90
        return 1.0 / (1.0 + np.exp(-(2*x1 - x2 + 0.5*x3)))

    def extraction_ceiling(self, T, pH, c_acid, species_idx):
        base = np.array([0.93, 0.84, 0.96], dtype=float)
        acid_bonus = 0.025 * np.log1p(c_acid) / np.log(4.0)
        temp_bonus = 0.015 * np.clip((T - 323.0) / 40.0, 0.0, 1.0)
        ph_penalty = np.array([0.035, 0.055, 0.020], dtype=float) * np.clip((pH - 1.2) / 1.8, 0.0, 1.0)
        ceiling = base + acid_bonus + temp_bonus - ph_penalty
        return float(np.clip(ceiling[species_idx], 0.55, 0.985))

    def solve(self, T, pH, c_acid, r0=50e-6, duration=180, dt=1.0):
        n_steps = int(duration / dt)
        alpha = np.zeros((3, n_steps + 1))
        gamma = self.blend_gamma(T, pH, r0)
        for step in range(n_steps):
            t = step * dt
            for s in range(3):
                da_sc = self.shrinking_core_rate(alpha[s, step], T, c_acid, r0, s)
                da_av = self.avrami_rate(t, T, pH, s)
                da_dt = gamma * da_sc + (1 - gamma) * da_av
                alpha[s, step + 1] = min(alpha[s, step] + dt * da_dt,
                                          self.extraction_ceiling(T, pH, c_acid, s))
        return alpha[:, -1]


class CostObjective:
    def __init__(self, reagent: str = "H2SO4", batch_volume_L: float = 10.0):
        self.reagent = reagent
        self.batch_volume = batch_volume_L
        self.reagent_cost = REAGENT_COSTS.get(reagent, 0.10)

    def reagent_cost_total(self, c_acid: float, duration_min: float) -> float:
        mass_reagent_kg = c_acid * self.batch_volume * 0.098
        return mass_reagent_kg * self.reagent_cost

    def energy_cost(self, T: float, duration_min: float) -> float:
        delta_T = max(0.0, T - 298.0)
        power_kw = 4.186 * self.batch_volume * delta_T / (duration_min * 60 + 1e-6)
        return power_kw * (duration_min / 60.0) * ENERGY_COST_KWH

    def time_cost(self, duration_min: float, labor_rate: float = 0.50) -> float:
        return (duration_min / 60.0) * labor_rate

    def total(self, T: float, pH: float, c_acid: float, duration_min: float) -> float:
        return (self.reagent_cost_total(c_acid, duration_min) +
                self.energy_cost(T, duration_min) +
                self.time_cost(duration_min))


class SelectivityPenalizer:
    def __init__(self, impurity_fractions: Optional[Dict[str, float]] = None):
        self.impurities = impurity_fractions or {"Al": 0.02, "Cu": 0.005, "Co": 0.001}

    def co_extraction_penalty(self, T: float, pH: float, c_acid: float) -> float:
        penalty = 0.0
        for element, frac in self.impurities.items():
            base_rate = frac * (1.0 + 0.5 * c_acid) * np.exp((T - 323) / 80.0)
            ph_factor = 1.0 / (1.0 + np.exp(2.0 * (pH - 1.5)))
            penalty += base_rate * ph_factor
        return penalty

    def selectivity_mn_fe(self, alpha_mn: float, alpha_fe: float) -> float:
        if alpha_mn + alpha_fe < 0.01:
            return 0.0
        return alpha_mn / (alpha_mn + alpha_fe + 1e-8)


class EnvironmentalImpact:
    def waste_acidity_penalty(self, pH: float, c_acid: float) -> float:
        neutralization_cost = max(0.0, 3.0 - pH) * c_acid * 0.05
        return neutralization_cost

    def heavy_metal_discharge(self, alpha_mn: float, alpha_fe: float,
                               initial_solid_kg: float = 1.0) -> float:
        residual_mn = (1.0 - alpha_mn) * initial_solid_kg * 0.3
        residual_fe = (1.0 - alpha_fe) * initial_solid_kg * 0.25
        return 0.1 * (residual_mn + residual_fe)


class ConstraintHandler:
    def __init__(self):
        self.T_min, self.T_max = 313.0, 373.0
        self.pH_min, self.pH_max = 0.3, 4.0
        self.c_acid_min, self.c_acid_max = 0.1, 4.0
        self.t_min, self.t_max = 15.0, 240.0
        self.r0_min, self.r0_max = 5e-6, 150e-6
        self.T_safety_max = 363.0

    def feasible(self, T, pH, c_acid, t, r0) -> Tuple[bool, str]:
        if T > self.T_safety_max:
            return False, "temperature exceeds safety limit"
        if pH < self.pH_min:
            return False, "pH too low, corrosion risk"
        if c_acid > self.c_acid_max:
            return False, "acid concentration too high"
        return True, "ok"

    def penalty(self, T, pH, c_acid, t, r0) -> float:
        p = 0.0
        p += max(0.0, T - self.T_safety_max) * 0.01
        p += max(0.0, self.pH_min - pH) * 0.5
        p += max(0.0, c_acid - self.c_acid_max) * 0.1
        return p


class NSGAII:
    def __init__(self, objective_fns: List, n_pop: int = 100, n_gen: int = 80,
                 crossover_prob: float = 0.9, mutation_prob: float = 0.1, seed: int = 42):
        self.objectives = objective_fns
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.rng = np.random.RandomState(seed)
        self.bounds = np.array([
            [313.0, 373.0],
            [0.3, 4.0],
            [0.1, 4.0],
            [15.0, 240.0],
            [5e-6, 150e-6],
        ])

    def _init_population(self) -> np.ndarray:
        pop = np.zeros((self.n_pop, 5))
        for i in range(5):
            pop[:, i] = self.rng.uniform(self.bounds[i, 0], self.bounds[i, 1], self.n_pop)
        return pop

    def _evaluate(self, pop: np.ndarray) -> np.ndarray:
        fitnesses = np.zeros((len(pop), len(self.objectives)))
        for i, ind in enumerate(pop):
            for j, obj_fn in enumerate(self.objectives):
                fitnesses[i, j] = obj_fn(*ind)
        return fitnesses

    def _fast_nondominated_sort(self, fitnesses: np.ndarray) -> List[List[int]]:
        n = len(fitnesses)
        dom_count = np.zeros(n, dtype=int)
        dom_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(fitnesses[i] <= fitnesses[j]) and np.any(fitnesses[i] < fitnesses[j]):
                    dom_set[i].append(j)
                elif np.all(fitnesses[j] <= fitnesses[i]) and np.any(fitnesses[j] < fitnesses[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            nf = []
            for i in fronts[k]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nf.append(j)
            k += 1
            fronts.append(nf)
        return [f for f in fronts if f]

    def _crowding_distance(self, fitnesses: np.ndarray, front: List[int]) -> np.ndarray:
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(fitnesses.shape[1]):
            vals = fitnesses[front, m]
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            vrange = vals[order[-1]] - vals[order[0]]
            if vrange < 1e-12:
                continue
            for k in range(1, n - 1):
                dist[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / vrange
        return dist

    def _tournament(self, pop, fitnesses, fronts_map, crowd_map):
        i, j = self.rng.randint(0, len(pop), 2)
        if fronts_map[i] < fronts_map[j]:
            return pop[i]
        elif fronts_map[j] < fronts_map[i]:
            return pop[j]
        elif crowd_map[i] > crowd_map[j]:
            return pop[i]
        return pop[j]

    def _sbx_crossover(self, p1, p2, eta=20.0):
        child = np.copy(p1)
        if self.rng.random() > self.cx_prob:
            return child
        for i in range(len(p1)):
            if self.rng.random() < 0.5:
                continue
            u = self.rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            child[i] = np.clip(child[i], self.bounds[i, 0], self.bounds[i, 1])
        return child

    def _polynomial_mutation(self, ind, eta=20.0):
        child = np.copy(ind)
        for i in range(len(ind)):
            if self.rng.random() > self.mut_prob:
                continue
            u = self.rng.random()
            delta = self.bounds[i, 1] - self.bounds[i, 0]
            if u < 0.5:
                delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            child[i] += delta_q * delta
            child[i] = np.clip(child[i], self.bounds[i, 0], self.bounds[i, 1])
        return child

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        pop = self._init_population()
        fitnesses = self._evaluate(pop)

        for gen in range(self.n_gen):
            fronts = self._fast_nondominated_sort(fitnesses)
            fronts_map = np.zeros(len(pop), dtype=int)
            crowd_map = np.zeros(len(pop))
            for fi, front in enumerate(fronts):
                cd = self._crowding_distance(fitnesses, front)
                for k, idx in enumerate(front):
                    fronts_map[idx] = fi
                    crowd_map[idx] = cd[k]

            offspring = []
            for _ in range(self.n_pop):
                p1 = self._tournament(pop, fitnesses, fronts_map, crowd_map)
                p2 = self._tournament(pop, fitnesses, fronts_map, crowd_map)
                child = self._sbx_crossover(p1, p2)
                child = self._polynomial_mutation(child)
                offspring.append(child)
            offspring = np.array(offspring)
            off_fit = self._evaluate(offspring)

            combined = np.vstack([pop, offspring])
            combined_fit = np.vstack([fitnesses, off_fit])
            c_fronts = self._fast_nondominated_sort(combined_fit)

            new_pop = []
            new_fit = []
            for front in c_fronts:
                if len(new_pop) + len(front) <= self.n_pop:
                    for idx in front:
                        new_pop.append(combined[idx])
                        new_fit.append(combined_fit[idx])
                else:
                    remaining = self.n_pop - len(new_pop)
                    cd = self._crowding_distance(combined_fit, front)
                    top = np.argsort(cd)[-remaining:]
                    for k in top:
                        new_pop.append(combined[front[k]])
                        new_fit.append(combined_fit[front[k]])
                    break

            pop = np.array(new_pop)
            fitnesses = np.array(new_fit)

        return pop, fitnesses


class MultiObjectiveLeaching:
    def __init__(self):
        self.solver = LeachingODESolver()
        self.cost = CostObjective()
        self.selectivity = SelectivityPenalizer()
        self.env = EnvironmentalImpact()
        self.constraints = ConstraintHandler()

    def obj_recovery(self, T, pH, c_acid, t, r0) -> float:
        alpha = self.solver.solve(T, pH, c_acid, r0, t)
        return -(0.5 * alpha[0] + 0.3 * alpha[1] + 0.2 * alpha[2])

    def obj_cost(self, T, pH, c_acid, t, r0) -> float:
        return self.cost.total(T, pH, c_acid, t) + self.constraints.penalty(T, pH, c_acid, t, r0)

    def obj_impurity(self, T, pH, c_acid, t, r0) -> float:
        return self.selectivity.co_extraction_penalty(T, pH, c_acid)

    def obj_environmental(self, T, pH, c_acid, t, r0) -> float:
        alpha = self.solver.solve(T, pH, c_acid, r0, t)
        return (self.env.waste_acidity_penalty(pH, c_acid) +
                self.env.heavy_metal_discharge(alpha[0], alpha[1]))

    def optimize(self, n_pop=80, n_gen=60) -> Dict:
        nsga = NSGAII(
            [self.obj_recovery, self.obj_cost, self.obj_impurity],
            n_pop=n_pop, n_gen=n_gen
        )
        pop, fitnesses = nsga.run()
        pareto_front = []
        for i in range(len(pop)):
            alpha = self.solver.solve(pop[i][0], pop[i][1], pop[i][2], pop[i][4], pop[i][3])
            pareto_front.append({
                "T": float(pop[i][0]), "pH": float(pop[i][1]),
                "conc": float(pop[i][2]), "t": float(pop[i][3]), "r0": float(pop[i][4]),
                "alpha_Mn": float(alpha[0]), "alpha_Fe": float(alpha[1]), "alpha_Na": float(alpha[2]),
                "recovery": float(-fitnesses[i][0]),
                "cost": float(fitnesses[i][1]),
                "impurity": float(fitnesses[i][2]),
            })
        pareto_front.sort(key=lambda x: x["recovery"], reverse=True)
        return {"pareto_front": pareto_front, "n_solutions": len(pareto_front)}


def optimize_leaching(n_trials=200):
    solver = LeachingODESolver()
    def objective(trial):
        T = trial.suggest_float('T', 323, 363)
        pH = trial.suggest_float('pH', 0.5, 3.0)
        conc = trial.suggest_float('conc', 0.5, 3.0)
        t = trial.suggest_float('t', 30, 180)
        r0 = trial.suggest_float('r0', 10e-6, 100e-6)
        alpha_final = solver.solve(T, pH, conc, r0, t)
        alpha_Mn, alpha_Fe, alpha_Na = alpha_final
        recovery = 0.5 * alpha_Mn + 0.3 * alpha_Fe + 0.2 * alpha_Na
        cost_penalty = 0.01 * (T - 323) / 40 + 0.005 * conc / 3
        return recovery - cost_penalty
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    alpha_opt = solver.solve(best['T'], best['pH'], best['conc'], best.get('r0', 50e-6), best['t'])
    return {
        'T': best['T'], 'pH': best['pH'], 'conc': best['conc'], 't': best['t'],
        'alpha_Mn': alpha_opt[0], 'alpha_Fe': alpha_opt[1], 'alpha_Na': alpha_opt[2],
        'best_value': study.best_value
    }


def generate_contour_data(solver, grid_resolution=20):
    T_range = np.linspace(323, 363, grid_resolution)
    pH_range = np.linspace(0.5, 3.0, grid_resolution)
    recovery_grid = np.zeros((grid_resolution, grid_resolution))
    for i, T in enumerate(T_range):
        for j, pH in enumerate(pH_range):
            alpha = solver.solve(T, pH, 1.8, 50e-6, 95)
            recovery_grid[i, j] = 0.5 * alpha[0] + 0.3 * alpha[1] + 0.2 * alpha[2]
    return T_range, pH_range, recovery_grid


def validate_recycling():
    opt = optimize_leaching(n_trials=50)
    assert opt['alpha_Mn'] > 0.70, "Mn recovery must exceed 70%"
    assert opt['T'] < 363, "Temperature must stay below 90C"
    return True
