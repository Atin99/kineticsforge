from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np


ELEMENTS = ("Mn", "Fe", "Na", "Al", "Cu", "Ni", "Co")


@dataclass
class UncertainFeedstockStream:
    name: str
    mass_kg: float
    mean_wt: Dict[str, float]
    std_wt: Dict[str, float]
    unknown_fraction: float = 0.0


@dataclass
class BlendResult:
    weights: Dict[str, float]
    expected_composition: Dict[str, float]
    reliability: float
    violation_probability: float
    mass_kg: float
    objective: float


class StochasticFeedstockBlender:
    def __init__(self, streams: Sequence[UncertainFeedstockStream], seed: int = 20260511):
        self.streams = list(streams)
        self.rng = np.random.RandomState(seed)

    def sample_stream_compositions(self, n_samples: int = 2048) -> np.ndarray:
        samples = np.zeros((len(self.streams), n_samples, len(ELEMENTS)), dtype=float)
        for i, stream in enumerate(self.streams):
            means = np.array([stream.mean_wt.get(el, 0.0) for el in ELEMENTS], dtype=float)
            stds = np.array([stream.std_wt.get(el, 0.0) for el in ELEMENTS], dtype=float)
            draw = self.rng.normal(means[None, :], stds[None, :], size=(n_samples, len(ELEMENTS)))
            draw = np.clip(draw, 0.0, None)
            total = draw.sum(axis=1, keepdims=True)
            excess = np.clip(total - (1.0 - stream.unknown_fraction), 0.0, None)
            draw = np.clip(draw - excess * draw / np.maximum(total, 1e-12), 0.0, None)
            samples[i] = draw
        return samples

    def optimize(self, target_wt: Dict[str, float], n_candidates: int = 4096, n_samples: int = 2048, tolerance: float = 0.035) -> BlendResult:
        samples = self.sample_stream_compositions(n_samples=n_samples)
        target = np.array([target_wt.get(el, 0.0) for el in ELEMENTS], dtype=float)
        masses = np.array([max(s.mass_kg, 0.0) for s in self.streams], dtype=float)
        best = None
        for _ in range(n_candidates):
            raw = self.rng.gamma(1.2, 1.0, size=len(self.streams))
            weights = raw / max(raw.sum(), 1e-12)
            available = np.minimum(weights * masses.sum(), masses)
            if available.sum() <= 0:
                continue
            blend_weights = available / available.sum()
            comp_samples = np.einsum("s,sne->ne", blend_weights, samples)
            err = comp_samples - target[None, :]
            transition_err = np.sqrt(np.sum(err[:, :3] ** 2, axis=1))
            impurity = comp_samples[:, 3:].sum(axis=1)
            reliability = float(np.mean(transition_err <= tolerance))
            violation_probability = float(np.mean((transition_err > tolerance) | (impurity > 0.10)))
            expected = comp_samples.mean(axis=0)
            objective = reliability - 0.35 * violation_probability - 0.18 * float(np.mean(impurity))
            if best is None or objective > best.objective:
                best = BlendResult(
                    weights={s.name: float(blend_weights[i]) for i, s in enumerate(self.streams)},
                    expected_composition={el: float(expected[j]) for j, el in enumerate(ELEMENTS)},
                    reliability=reliability,
                    violation_probability=violation_probability,
                    mass_kg=float(available.sum()),
                    objective=float(objective),
                )
        if best is None:
            raise ValueError("No feasible feedstock blend could be sampled.")
        return best


def result_to_dict(result: BlendResult) -> Dict[str, object]:
    return asdict(result)
