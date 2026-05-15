import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


@dataclass
class BaselineResult:
    name: str
    mae: float
    mape: float
    rmse: float
    max_error: float
    r2: float
    description: str


@dataclass
class BenchmarkReport:
    target_metric: str
    n_holdout: int
    baselines: List[BaselineResult]
    kineticsforge_result: BaselineResult
    ranking: List[str]
    summary: str


class PhysicsOnlyBaseline:
    """Empirical Arrhenius + sqrt(cycle) fade model. No ML."""
    def predict(self, cycles: np.ndarray, Q0: float, T_K: float = 318.15) -> np.ndarray:
        Ea_eV = 0.32
        kB_eV_K = 8.617e-5
        k_ref = 0.0012
        k = k_ref * math.exp(-Ea_eV / (kB_eV_K * T_K))
        return Q0 * (1.0 - k * np.sqrt(cycles))


class ExponentialFadeBaseline:
    """Simple exponential fade: Q(n) = Q0 * exp(-lambda * n)."""
    def fit_predict(self, cycles: np.ndarray, capacity: np.ndarray) -> np.ndarray:
        Q0 = float(capacity[0]) if len(capacity) else 150.0
        if len(capacity) > 10:
            ratio = np.clip(capacity[-1] / max(Q0, 1e-9), 0.01, 0.999)
            lam = -math.log(max(ratio, 1e-9)) / max(float(cycles[-1]), 1.0)
        else:
            lam = 0.0004
        return Q0 * np.exp(-lam * cycles)


class RandomForestBaseline:
    """Simplified random forest surrogate using feature hashing (no sklearn dependency)."""
    def __init__(self, n_trees: int = 25, seed: int = 42):
        self.n_trees = n_trees
        self.rng = np.random.RandomState(seed)
        self.trees: List[Dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(y)
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=n, replace=True)
            feat = self.rng.randint(0, X.shape[1])
            thresh = float(np.median(X[idx, feat]))
            left = y[idx][X[idx, feat] <= thresh]
            right = y[idx][X[idx, feat] > thresh]
            self.trees.append({"feat": int(feat), "thresh": thresh, "left_val": float(np.mean(left)) if len(left) else float(np.mean(y)), "right_val": float(np.mean(right)) if len(right) else float(np.mean(y))})

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(X))
        for tree in self.trees:
            mask = X[:, tree["feat"]] <= tree["thresh"]
            preds[mask] += tree["left_val"]
            preds[~mask] += tree["right_val"]
        return preds / max(len(self.trees), 1)


class SmallNeuralBaseline:
    """Two-layer MLP trained with numpy gradient descent (no torch/tf dependency)."""
    def __init__(self, hidden: int = 16, lr: float = 0.005, epochs: int = 200, seed: int = 42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.RandomState(seed)
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        d = X.shape[1]
        self.W1 = self.rng.randn(d, self.hidden) * 0.1
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.randn(self.hidden, 1) * 0.1
        self.b2 = np.zeros(1)
        y = y.reshape(-1, 1)
        for _ in range(self.epochs):
            h = self._relu(X @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            err = pred - y
            grad_W2 = h.T @ err / len(y)
            grad_b2 = err.mean(axis=0)
            grad_h = err @ self.W2.T
            grad_h[h <= 0] = 0
            grad_W1 = X.T @ grad_h / len(y)
            grad_b1 = grad_h.mean(axis=0)
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = self._relu(X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    mape = float(np.mean(np.abs(residual) / np.maximum(np.abs(y_true), 1e-9)) * 100.0)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_err = float(np.max(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return {"mae": mae, "mape": mape, "rmse": rmse, "max_error": max_err, "r2": r2}


def generate_synthetic_holdout(n_compositions: int = 30, n_cycles: int = 200, seed: int = 99) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    X_list, y_list, cycles_list = [], [], []
    for _ in range(n_compositions):
        Q0 = rng.uniform(100, 200)
        fade_rate = rng.uniform(0.0002, 0.0015)
        T_K = rng.uniform(298, 333)
        mn = rng.uniform(0.3, 0.7)
        fe = 1.0 - mn - rng.uniform(0, 0.1)
        cycles = np.arange(0, n_cycles + 1, dtype=float)
        noise = rng.normal(0, 1.5, size=len(cycles))
        capacity = Q0 * np.exp(-fade_rate * cycles) + noise
        capacity = np.clip(capacity, 10.0, 260.0)
        for i, c in enumerate(cycles):
            X_list.append([c, Q0, T_K, mn, fe, fade_rate])
            y_list.append(capacity[i])
            cycles_list.append(c)
    return {"X": np.array(X_list), "y": np.array(y_list), "cycles": np.array(cycles_list)}


def run_holdout_benchmark(holdout: Optional[Dict[str, np.ndarray]] = None) -> BenchmarkReport:
    if holdout is None:
        holdout = generate_synthetic_holdout()
    X, y = holdout["X"], holdout["y"]
    n = len(y)
    split = int(0.7 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results: List[BaselineResult] = []

    # Physics-only
    phys = PhysicsOnlyBaseline()
    phys_pred = np.array([phys.predict(np.array([x[0]]), x[1], x[2])[0] for x in X_test])
    m = _metrics(y_test, phys_pred)
    results.append(BaselineResult("physics_only_arrhenius", m["mae"], m["mape"], m["rmse"], m["max_error"], m["r2"], "Arrhenius + sqrt(n) empirical fade model"))

    # Exponential fade
    exp_b = ExponentialFadeBaseline()
    exp_pred = np.array([exp_b.fit_predict(np.array([x[0]]), np.array([x[1]]))[0] for x in X_test])
    m = _metrics(y_test, exp_pred)
    results.append(BaselineResult("exponential_fade", m["mae"], m["mape"], m["rmse"], m["max_error"], m["r2"], "Q0 * exp(-lambda*n) fitted per composition"))

    # Random forest
    rf = RandomForestBaseline(n_trees=50)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    m = _metrics(y_test, rf_pred)
    results.append(BaselineResult("random_forest_25tree", m["mae"], m["mape"], m["rmse"], m["max_error"], m["r2"], "Simplified 25-tree random forest on cycle+composition features"))

    # Small MLP
    nn = SmallNeuralBaseline(hidden=32, lr=0.003, epochs=300)
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-9
    nn.fit((X_train - x_mean) / x_std, (y_train - y_mean) / y_std)
    nn_pred = nn.predict((X_test - x_mean) / x_std) * y_std + y_mean
    m = _metrics(y_test, nn_pred)
    results.append(BaselineResult("small_mlp_32h", m["mae"], m["mape"], m["rmse"], m["max_error"], m["r2"], "2-layer MLP with 32 hidden units, numpy-only"))

    # KineticsForge surrogate (uses physics-constrained fade model from screener)
    kf_pred = np.array([_kineticsforge_predict(x) for x in X_test])
    m = _metrics(y_test, kf_pred)
    kf_result = BaselineResult("kineticsforge_v2", m["mae"], m["mape"], m["rmse"], m["max_error"], m["r2"], "KineticsForge physics-constrained surrogate model")

    all_results = results + [kf_result]
    ranking = sorted(all_results, key=lambda r: r.mae)
    ranking_names = [r.name for r in ranking]
    kf_rank = ranking_names.index("kineticsforge_v2") + 1
    summary = f"KineticsForge ranks #{kf_rank}/{len(ranking_names)} by MAE on {len(y_test)} holdout points. MAE={kf_result.mae:.2f}, R2={kf_result.r2:.3f}."
    return BenchmarkReport(target_metric="discharge_capacity_mAh_g", n_holdout=len(y_test), baselines=results, kineticsforge_result=kf_result, ranking=ranking_names, summary=summary)


def _kineticsforge_predict(x: np.ndarray) -> float:
    cycle, Q0, T_K, mn, fe, _ = x
    Ea = 0.28 + 0.08 * mn
    kB = 8.617e-5
    k = 0.0015 * math.exp(-Ea / (kB * T_K))
    phase_stability = 0.45 + 0.35 * fe + 0.15 * mn
    k_adj = k * (1.0 + 0.3 * (1.0 - phase_stability))
    capacity = Q0 * (1.0 - k_adj * cycle ** 0.72)
    return float(max(capacity, 10.0))


def report_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    return {"target_metric": report.target_metric, "n_holdout": report.n_holdout, "baselines": [asdict(b) for b in report.baselines], "kineticsforge_result": asdict(report.kineticsforge_result), "ranking": report.ranking, "summary": report.summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/holdout_benchmark_v2.json")
    args = parser.parse_args()
    report = run_holdout_benchmark()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    print(json.dumps({"summary": report.summary, "ranking": report.ranking}, indent=2))


if __name__ == "__main__":
    main()
