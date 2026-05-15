import numpy as np
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats, optimize
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = PROJECT_ROOT / "data" / "real"
NORMALIZED_DIR = REAL_DIR / "normalized"
SCRAPED_DIR = REAL_DIR / "scraped"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def _load_parquet_safe(path: Path) -> Optional[Any]:
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("Cannot load %s: %s", path, exc)
        return None


class CapacityFadeAnalyzer:
    def __init__(self, cycles: np.ndarray, capacities: np.ndarray):
        self.cycles = cycles.astype(np.float64)
        self.caps = capacities.astype(np.float64)
        self.q0 = float(self.caps[0]) if len(self.caps) > 0 else 150.0
        self.fits: Dict[str, Dict] = {}

    def _safe_fit(self, name, func, p0, bounds=(-np.inf, np.inf)):
        try:
            popt, pcov = optimize.curve_fit(func, self.cycles, self.caps / self.q0,
                                            p0=p0, bounds=bounds, maxfev=8000)
            residuals = self.caps / self.q0 - func(self.cycles, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((self.caps / self.q0 - np.mean(self.caps / self.q0)) ** 2) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            self.fits[name] = {"params": [float(p) for p in popt], "r2": float(r2), "rmse": rmse}
        except Exception as exc:
            self.fits[name] = {"params": [], "r2": -1.0, "rmse": 999.0, "error": str(exc)}

    def fit_power_law(self):
        self._safe_fit("power_law",
                        lambda n, a, b: 1.0 - a * np.power(n + 1, b),
                        p0=[1e-4, 0.5], bounds=([0, 0.01], [1.0, 2.0]))

    def fit_sqrt_time(self):
        self._safe_fit("sqrt_time",
                        lambda n, a: 1.0 - a * np.sqrt(n + 1),
                        p0=[1e-3], bounds=([0], [1.0]))

    def fit_exponential(self):
        self._safe_fit("exponential",
                        lambda n, a, b: np.exp(-a * np.power(n, b)),
                        p0=[1e-4, 1.0], bounds=([0, 0.1], [0.1, 2.0]))

    def fit_stretched_exp(self):
        self._safe_fit("stretched_exp",
                        lambda n, a, beta: np.exp(-np.power(a * n, beta)),
                        p0=[1e-4, 0.7], bounds=([0, 0.1], [0.1, 1.0]))

    def fit_linear(self):
        self._safe_fit("linear",
                        lambda n, a: 1.0 - a * n,
                        p0=[1e-4], bounds=([0], [0.01]))

    def fit_all(self):
        self.fit_power_law()
        self.fit_sqrt_time()
        self.fit_exponential()
        self.fit_stretched_exp()
        self.fit_linear()

    def best_model(self) -> Tuple[str, Dict]:
        if not self.fits:
            self.fit_all()
        valid = {k: v for k, v in self.fits.items() if v["r2"] > 0}
        if not valid:
            return "linear", {"params": [1e-4], "r2": 0.0, "rmse": 999.0}
        return max(valid.items(), key=lambda kv: kv[1]["r2"])

    def fade_rate_per_cycle(self) -> float:
        if len(self.caps) < 2:
            return 0.0
        total_fade = 1.0 - self.caps[-1] / (self.q0 + 1e-12)
        return float(total_fade / max(len(self.caps), 1))

    def knee_point_index(self) -> int:
        if len(self.caps) < 20:
            return len(self.caps) - 1
        smooth = savgol_filter(self.caps, min(21, len(self.caps) // 3 * 2 + 1), 3)
        d2 = np.gradient(np.gradient(smooth))
        return int(np.argmax(np.abs(d2[5:])) + 5)

    def as_dict(self) -> Dict:
        name, fit = self.best_model()
        return {
            "q0": self.q0,
            "best_model": name,
            "best_fit": fit,
            "all_fits": self.fits,
            "fade_rate_per_cycle": self.fade_rate_per_cycle(),
            "knee_point_cycle": self.knee_point_index(),
            "total_cycles": len(self.cycles),
        }


class WeibullLifetimeEstimator:
    def __init__(self, cycle_lives: np.ndarray):
        self.lives = cycle_lives[cycle_lives > 0].astype(np.float64)
        self.shape: float = 2.0
        self.scale: float = 500.0
        self.loc: float = 0.0

    def fit(self):
        if len(self.lives) < 5:
            return
        try:
            self.shape, self.loc, self.scale = stats.weibull_min.fit(self.lives, floc=0)
        except Exception:
            self.shape = 2.0
            self.scale = float(np.median(self.lives))

    def fit_lognormal(self) -> Dict:
        if len(self.lives) < 5:
            return {"mu": np.log(500), "sigma": 0.5}
        s, loc, scale = stats.lognorm.fit(self.lives, floc=0)
        return {"mu": float(np.log(scale)), "sigma": float(s)}

    def survival_probability(self, n_cycles: np.ndarray) -> np.ndarray:
        return stats.weibull_min.sf(n_cycles, self.shape, self.loc, self.scale)

    def percentile(self, p: float) -> float:
        return float(stats.weibull_min.ppf(p, self.shape, self.loc, self.scale))

    def as_dict(self) -> Dict:
        ln = self.fit_lognormal()
        return {
            "weibull_shape": float(self.shape),
            "weibull_scale": float(self.scale),
            "lognormal_mu": ln["mu"],
            "lognormal_sigma": ln["sigma"],
            "median_life": self.percentile(0.5),
            "p10_life": self.percentile(0.1),
            "p90_life": self.percentile(0.9),
            "n_samples": len(self.lives),
        }


class DegradationEnvelopeFitter:
    def __init__(self, all_capacity_traces: List[np.ndarray], max_cycles: int = 2000):
        self.traces = all_capacity_traces
        self.max_cycles = max_cycles

    def _align_traces(self) -> Tuple[np.ndarray, np.ndarray]:
        aligned = []
        for tr in self.traces:
            if len(tr) < 5:
                continue
            normed = tr / (tr[0] + 1e-12)
            if len(normed) > self.max_cycles:
                normed = normed[:self.max_cycles]
            aligned.append(normed)
        if not aligned:
            return np.linspace(0, 1, 100), np.ones((1, 100))
        max_len = max(len(a) for a in aligned)
        matrix = np.full((len(aligned), max_len), np.nan)
        for i, a in enumerate(aligned):
            matrix[i, :len(a)] = a
        return np.arange(max_len), matrix

    def fit_quantile_envelopes(self, quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)) -> Dict:
        cycles, matrix = self._align_traces()
        result = {}
        for q in quantiles:
            envelope = np.nanquantile(matrix, q, axis=0)
            valid = ~np.isnan(envelope)
            result[f"q{int(q*100):02d}"] = {
                "values": envelope[valid].tolist()[:200],
                "n_valid": int(valid.sum()),
            }
        return result

    def fit_mean_std(self) -> Dict:
        cycles, matrix = self._align_traces()
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        valid = ~np.isnan(mean)
        return {
            "mean": mean[valid].tolist()[:200],
            "std": std[valid].tolist()[:200],
            "n_traces": len(self.traces),
        }


class VoltageRegimeClassifier:
    def __init__(self, voltages: np.ndarray, currents: np.ndarray, temperatures: np.ndarray):
        self.V = voltages.astype(np.float64)
        self.I = currents.astype(np.float64)
        self.T = temperatures.astype(np.float64)

    def voltage_window_stats(self) -> Dict:
        return {
            "v_min_mean": float(np.mean(np.nanmin(self.V, axis=-1))) if self.V.ndim > 1 else float(np.nanmin(self.V)),
            "v_max_mean": float(np.mean(np.nanmax(self.V, axis=-1))) if self.V.ndim > 1 else float(np.nanmax(self.V)),
            "v_mean": float(np.nanmean(self.V)),
            "v_std": float(np.nanstd(self.V)),
        }

    def crate_distribution(self, nominal_capacity: float = 150.0) -> Dict:
        c_rates = np.abs(self.I) / (nominal_capacity + 1e-12)
        c_rates = c_rates[~np.isnan(c_rates)]
        if len(c_rates) < 2:
            return {"mean": 0.5, "std": 0.2, "p10": 0.1, "p90": 1.0}
        return {
            "mean": float(np.mean(c_rates)),
            "std": float(np.std(c_rates)),
            "p10": float(np.percentile(c_rates, 10)),
            "p90": float(np.percentile(c_rates, 90)),
        }

    def temperature_distribution(self) -> Dict:
        t_flat = self.T[~np.isnan(self.T)]
        if len(t_flat) < 2:
            return {"mean": 308.0, "std": 10.0, "min": 293.0, "max": 333.0}
        return {
            "mean": float(np.mean(t_flat)),
            "std": float(np.std(t_flat)),
            "min": float(np.min(t_flat)),
            "max": float(np.max(t_flat)),
        }


class CoulombicEfficiencyTracker:
    def __init__(self, charge_caps: np.ndarray, discharge_caps: np.ndarray):
        valid = (charge_caps > 0) & (discharge_caps > 0)
        self.ce = discharge_caps[valid] / charge_caps[valid]

    def stats(self) -> Dict:
        if len(self.ce) < 2:
            return {"mean": 0.995, "std": 0.003, "min": 0.98}
        return {
            "mean": float(np.nanmean(self.ce)),
            "std": float(np.nanstd(self.ce)),
            "min": float(np.nanmin(self.ce)),
            "median": float(np.nanmedian(self.ce)),
        }


class CalibrationPriorExporter:
    def __init__(self, output_path: Optional[Path] = None):
        self.output = output_path or SCRAPED_DIR / "calibration_priors.json"

    def export(self, priors: Dict) -> Path:
        priors["_checksum"] = hashlib.sha256(
            json.dumps(priors, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(priors, indent=2, default=str), encoding="utf-8")
        log.info("Calibration priors exported to %s (%d keys)", self.output, len(priors))
        return self.output


class CalibrationValidator:
    def __init__(self, priors: Dict):
        self.priors = priors
        self.checks: List[Dict] = []

    def _check(self, name: str, ok: bool, detail: str = ""):
        self.checks.append({"name": name, "pass": ok, "detail": detail})

    def validate(self) -> bool:
        cap = self.priors.get("capacity_fade", {})
        self._check("capacity_fade_exists", bool(cap), "missing capacity_fade")
        if cap:
            r2 = cap.get("best_fit", {}).get("r2", -1)
            self._check("capacity_fade_r2", r2 > 0.5, f"r2={r2}")

        life = self.priors.get("lifetime", {})
        self._check("lifetime_exists", bool(life), "missing lifetime")
        if life:
            self._check("weibull_shape_positive", life.get("weibull_shape", 0) > 0)

        env = self.priors.get("degradation_envelope", {})
        self._check("envelope_exists", bool(env))

        volt = self.priors.get("voltage_regime", {})
        self._check("voltage_regime_exists", bool(volt))

        temp = self.priors.get("temperature_regime", {})
        self._check("temperature_regime_exists", bool(temp))

        all_pass = all(c["pass"] for c in self.checks)
        return all_pass

    def report(self) -> Dict:
        return {
            "status": "pass" if all(c["pass"] for c in self.checks) else "fail",
            "checks": self.checks,
            "n_pass": sum(c["pass"] for c in self.checks),
            "n_total": len(self.checks),
        }


def run_calibration(cycle_summary_path: Optional[Path] = None,
                    timeseries_path: Optional[Path] = None,
                    output_path: Optional[Path] = None) -> Dict:
    cs_path = cycle_summary_path or NORMALIZED_DIR / "batterylife_naion_cycle_summary.parquet"
    ts_path = timeseries_path or NORMALIZED_DIR / "batterylife_naion_timeseries_sample.parquet"

    df_cycles = _load_parquet_safe(cs_path)
    df_ts = _load_parquet_safe(ts_path)

    priors: Dict[str, Any] = {"source": str(cs_path), "calibration_version": "2.0"}

    if df_cycles is not None and len(df_cycles) > 0:
        cap_col = None
        for c in ["discharge_capacity", "capacity", "Discharge_Capacity", "cap"]:
            if c in df_cycles.columns:
                cap_col = c
                break

        cycle_col = None
        for c in ["cycle", "cycle_number", "Cycle_Index", "cycle_idx"]:
            if c in df_cycles.columns:
                cycle_col = c
                break

        cell_col = None
        for c in ["cell_id", "cell", "Cell_ID", "barcode"]:
            if c in df_cycles.columns:
                cell_col = c
                break

        if cap_col and cycle_col:
            caps = df_cycles[cap_col].values.astype(np.float64)
            cycles = df_cycles[cycle_col].values.astype(np.float64)
            valid = ~(np.isnan(caps) | np.isnan(cycles))
            caps, cycles = caps[valid], cycles[valid]

            analyzer = CapacityFadeAnalyzer(cycles, caps)
            analyzer.fit_all()
            priors["capacity_fade"] = analyzer.as_dict()

            if cell_col:
                cell_ids = df_cycles[cell_col].values[valid]
                unique_cells = np.unique(cell_ids)
                traces = []
                cell_lives = []
                for cid in unique_cells:
                    mask = cell_ids == cid
                    cell_caps = caps[mask]
                    cell_cyc = cycles[mask]
                    order = np.argsort(cell_cyc)
                    cell_caps = cell_caps[order]
                    if len(cell_caps) > 5:
                        traces.append(cell_caps)
                        eol_80 = np.where(cell_caps < 0.8 * cell_caps[0])[0]
                        life = int(eol_80[0]) if len(eol_80) > 0 else len(cell_caps)
                        cell_lives.append(life)

                if traces:
                    envelope = DegradationEnvelopeFitter(traces)
                    priors["degradation_envelope"] = envelope.fit_mean_std()
                    priors["degradation_envelope"]["quantiles"] = envelope.fit_quantile_envelopes()

                if cell_lives:
                    lifetime_est = WeibullLifetimeEstimator(np.array(cell_lives))
                    lifetime_est.fit()
                    priors["lifetime"] = lifetime_est.as_dict()

        v_col = next((c for c in ["voltage", "Voltage", "V", "mean_voltage"] if c in df_cycles.columns), None)
        i_col = next((c for c in ["current", "Current", "I", "mean_current"] if c in df_cycles.columns), None)
        t_col = next((c for c in ["temperature", "Temperature", "T", "mean_temp"] if c in df_cycles.columns), None)

        voltages = df_cycles[v_col].values if v_col else np.array([3.5])
        currents = df_cycles[i_col].values if i_col else np.array([0.5])
        temperatures = df_cycles[t_col].values if t_col else np.array([308.0])

        regime = VoltageRegimeClassifier(voltages, currents, temperatures)
        priors["voltage_regime"] = regime.voltage_window_stats()
        priors["crate_regime"] = regime.crate_distribution()
        priors["temperature_regime"] = regime.temperature_distribution()

        ch_col = next((c for c in ["charge_capacity", "Charge_Capacity"] if c in df_cycles.columns), None)
        dc_col = cap_col
        if ch_col and dc_col:
            ch = df_cycles[ch_col].values.astype(np.float64)
            dc = df_cycles[dc_col].values.astype(np.float64)
            ce_tracker = CoulombicEfficiencyTracker(ch, dc)
            priors["coulombic_efficiency"] = ce_tracker.stats()

    existing_priors_path = SCRAPED_DIR / "calibration_priors.json"
    if existing_priors_path.exists():
        try:
            existing = json.loads(existing_priors_path.read_text(encoding="utf-8"))
            for k, v in existing.items():
                if k not in priors and not k.startswith("_"):
                    priors[k] = v
        except Exception:
            pass

    validator = CalibrationValidator(priors)
    valid = validator.validate()
    priors["_validation"] = validator.report()

    exporter = CalibrationPriorExporter(output_path)
    exporter.export(priors)

    log.info("Calibration %s: %d checks passed / %d total",
             "PASSED" if valid else "FAILED",
             validator.report()["n_pass"], validator.report()["n_total"])
    return priors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_calibration()
    print(json.dumps(result.get("_validation", {}), indent=2))
