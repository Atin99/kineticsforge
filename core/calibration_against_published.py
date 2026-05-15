"""Calibrate KineticsForge fade model against real published Na-ion cycling data.

This module takes published experimental values (specific capacity, retention
at N cycles, temperature) and fits the Arrhenius fade parameters so the model
reproduces reality, not synthetic data.

Every calibrated parameter is tied to a DOI. If the model cannot reproduce
a published result within tolerance, it reports the failure explicitly.
"""
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PublishedDataPoint:
    doi: str
    material: str
    paper_title: str
    year: int
    metric: str
    condition: str
    value: float
    unit: str
    confidence: float


@dataclass
class CalibrationResult:
    parameter: str
    fitted_value: float
    literature_support: List[str]
    residual_vs_published: List[Dict[str, Any]]
    mean_absolute_error: float
    worst_case_error: float
    passes_threshold: bool
    threshold_used: float


@dataclass
class CalibratedFadeModel:
    """Arrhenius-based capacity fade model calibrated against published data.

    Q(n, T) = Q0 * (1 - k(T) * n^beta)
    k(T) = k_ref * exp(-Ea / (kB * T))

    Parameters fitted from published Na(Mn,Fe)O2 cycling data.
    """
    k_ref: float  # Reference rate constant
    Ea_eV: float  # Activation energy
    beta: float   # Cycle exponent (sqrt-like = 0.5, linear = 1.0)
    calibration_dois: List[str]
    calibration_results: List[CalibrationResult]
    overall_mae_vs_published: float
    honest_assessment: str

    def predict_capacity(self, Q0: float, cycle: float, T_K: float = 298.15) -> float:
        kB = 8.617e-5  # eV/K
        k = self.k_ref * math.exp(-self.Ea_eV / (kB * T_K))
        fade = k * (cycle ** self.beta)
        return float(max(Q0 * (1.0 - fade), 0.0))

    def predict_retention(self, cycle: float, T_K: float = 298.15) -> float:
        kB = 8.617e-5
        k = self.k_ref * math.exp(-self.Ea_eV / (kB * T_K))
        return float(max(1.0 - k * (cycle ** self.beta), 0.0))


def load_published_data(csv_path: Path) -> List[PublishedDataPoint]:
    """Load real published values from the literature citations CSV."""
    points: List[PublishedDataPoint] = []
    if not csv_path.exists():
        return points
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(PublishedDataPoint(
                doi=row.get("doi", ""),
                material=row.get("material_system", ""),
                paper_title=row.get("paper_title", ""),
                year=int(row.get("year", 0)),
                metric=row.get("metric", ""),
                condition=row.get("condition", ""),
                value=float(row.get("value", 0)),
                unit=row.get("unit", ""),
                confidence=float(row.get("extraction_confidence", 0.5)),
            ))
    return points


def extract_retention_points(data: List[PublishedDataPoint]) -> List[Dict[str, Any]]:
    """Extract (material, Q0, retention_%, cycles, T_K) from published data."""
    # Group by material: find Q0 and retention entries
    by_material: Dict[str, List[PublishedDataPoint]] = {}
    for p in data:
        by_material.setdefault(p.material, []).append(p)

    calibration_points: List[Dict[str, Any]] = []
    for material, entries in by_material.items():
        q0_entry = None
        retention_entries = []
        for e in entries:
            if e.metric == "specific_capacity":
                q0_entry = e
            elif e.metric == "capacity_retention":
                retention_entries = [e]

        if q0_entry and retention_entries:
            for ret in retention_entries:
                # Parse cycles from condition string
                cycles = _parse_cycles(ret.condition)
                T_K = _parse_temperature(ret.condition)
                if cycles > 0:
                    calibration_points.append({
                        "material": material,
                        "doi": q0_entry.doi,
                        "Q0": q0_entry.value,
                        "retention_pct": ret.value,
                        "cycles": cycles,
                        "T_K": T_K,
                        "confidence": min(q0_entry.confidence, ret.confidence),
                    })
    return calibration_points


def _parse_cycles(condition: str) -> int:
    """Extract cycle count from condition strings like '1C, 500 cycles' or '0.1C, 50 cycles'."""
    parts = condition.lower().replace(",", " ").split()
    for i, p in enumerate(parts):
        if "cycle" in p and i > 0:
            try:
                return int(parts[i - 1])
            except ValueError:
                pass
    # Try direct number extraction
    for p in parts:
        try:
            n = int(p)
            if 10 <= n <= 10000:
                return n
        except ValueError:
            pass
    return 0


def _parse_temperature(condition: str) -> float:
    """Extract temperature from condition string. Default 298.15 K (RT)."""
    cond_lower = condition.lower()
    if "rt" in cond_lower or "room" in cond_lower:
        return 298.15
    # Look for explicit temperature
    parts = cond_lower.replace(",", " ").split()
    for i, p in enumerate(parts):
        if "c" in p and i >= 0:
            try:
                t = float(p.replace("c", "").replace("°", ""))
                if 20 <= t <= 80:
                    return t + 273.15
            except ValueError:
                pass
    return 298.15


def calibrate_fade_model(project_root: Path) -> CalibratedFadeModel:
    """Fit Arrhenius fade parameters against published Na-ion cycling data."""
    csv_path = project_root / "data" / "real" / "scraped" / "literature_citations.csv"
    raw_data = load_published_data(csv_path)
    cal_points = extract_retention_points(raw_data)

    if not cal_points:
        return _fallback_model("No published retention data found for calibration.")

    # Grid search over (k_ref, Ea, beta) to minimize error vs published retention
    # Note: k(T) = k_ref * exp(-Ea/(kB*T)). At T=298K, exp(-Ea/(kB*298)) ~ exp(-Ea/0.0257).
    # For Ea=0.3 eV, this is exp(-11.7) ~ 8e-6. So k_ref must be LARGE to produce
    # meaningful fade. Published Na-ion fade is 10-25% over 30-500 cycles.
    best_error = float("inf")
    best_params = (100.0, 0.30, 0.55)

    for log_k in np.linspace(-1, 6, 40):  # k_ref from 0.1 to 1e6
        k_ref = 10.0 ** log_k
        for Ea in np.linspace(0.05, 0.65, 25):
            for beta in np.linspace(0.3, 1.0, 20):
                error = 0.0
                valid = True
                for pt in cal_points:
                    kB = 8.617e-5
                    k = k_ref * math.exp(-Ea / (kB * pt["T_K"]))
                    predicted_retention = max(0.0, 1.0 - k * (pt["cycles"] ** beta)) * 100.0
                    if predicted_retention < 0:
                        valid = False
                        break
                    error += abs(predicted_retention - pt["retention_pct"]) * pt["confidence"]
                if valid and error < best_error:
                    best_error = error
                    best_params = (float(k_ref), float(Ea), float(beta))

    k_ref_fit, Ea_fit, beta_fit = best_params

    # Compute per-point residuals
    residuals: List[Dict[str, Any]] = []
    errors: List[float] = []
    for pt in cal_points:
        kB = 8.617e-5
        k = k_ref_fit * math.exp(-Ea_fit / (kB * pt["T_K"]))
        pred = max(0.0, 1.0 - k * (pt["cycles"] ** beta_fit)) * 100.0
        err = abs(pred - pt["retention_pct"])
        errors.append(err)
        residuals.append({
            "material": pt["material"],
            "doi": pt["doi"],
            "published_retention_pct": pt["retention_pct"],
            "predicted_retention_pct": round(pred, 1),
            "error_pct": round(err, 1),
            "cycles": pt["cycles"],
            "T_K": pt["T_K"],
        })

    mae = float(np.mean(errors)) if errors else float("inf")
    worst = float(max(errors)) if errors else float("inf")
    passes = mae < 15.0  # 15% MAE threshold for "acceptable" calibration

    dois = list(set(pt["doi"] for pt in cal_points))

    cal_results = [CalibrationResult(
        parameter="arrhenius_fade",
        fitted_value=k_ref_fit,
        literature_support=dois,
        residual_vs_published=residuals,
        mean_absolute_error=mae,
        worst_case_error=worst,
        passes_threshold=passes,
        threshold_used=15.0,
    )]

    assessment = _honest_assessment(mae, worst, len(cal_points), passes)

    return CalibratedFadeModel(
        k_ref=k_ref_fit,
        Ea_eV=Ea_fit,
        beta=beta_fit,
        calibration_dois=dois,
        calibration_results=cal_results,
        overall_mae_vs_published=mae,
        honest_assessment=assessment,
    )


def _honest_assessment(mae: float, worst: float, n_points: int, passes: bool) -> str:
    if n_points < 3:
        return (f"WEAK: Only {n_points} published data points used for calibration. "
                "This is not enough to claim validated performance. Treat all predictions "
                "as rough estimates until more experimental data is available.")
    if not passes:
        return (f"FAILING: MAE {mae:.1f}% against published data exceeds 15% threshold. "
                "The model does not reproduce published cycling results well enough. "
                "Do NOT use these predictions for experiment planning without independent validation.")
    if mae < 5.0:
        return (f"GOOD: MAE {mae:.1f}% against {n_points} published data points. "
                "Model reproduces published retention trends. Still not a substitute for "
                "lab measurement on YOUR specific composition and conditions.")
    return (f"ACCEPTABLE: MAE {mae:.1f}% against {n_points} published data points. "
            f"Worst case error {worst:.1f}%. Predictions are directionally useful for "
            "screening but should be validated experimentally for any go/no-go decision.")


def _fallback_model(reason: str) -> CalibratedFadeModel:
    return CalibratedFadeModel(
        k_ref=0.0008,
        Ea_eV=0.32,
        beta=0.65,
        calibration_dois=[],
        calibration_results=[],
        overall_mae_vs_published=float("inf"),
        honest_assessment=f"UNCALIBRATED: {reason} Using textbook defaults. Do not trust predictions.",
    )


def model_to_dict(model: CalibratedFadeModel) -> Dict[str, Any]:
    return {
        "k_ref": model.k_ref,
        "Ea_eV": model.Ea_eV,
        "beta": model.beta,
        "calibration_dois": model.calibration_dois,
        "calibration_results": [asdict(r) for r in model.calibration_results],
        "overall_mae_vs_published": model.overall_mae_vs_published,
        "honest_assessment": model.honest_assessment,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="data/cache/calibrated_fade_model_v2.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    model = calibrate_fade_model(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model_to_dict(model), indent=2), encoding="utf-8")
    print(json.dumps({
        "k_ref": model.k_ref,
        "Ea_eV": model.Ea_eV,
        "beta": model.beta,
        "mae_vs_published": model.overall_mae_vs_published,
        "assessment": model.honest_assessment,
        "n_dois": len(model.calibration_dois),
    }, indent=2))


if __name__ == "__main__":
    main()
