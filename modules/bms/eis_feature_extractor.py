from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class EISFeatures:
    R_ohm: float
    R_ct: float
    R_sei: float
    sigma_warburg: float
    cpe_alpha: float
    quality_flag: str
    n_points: int
    frequency_min_Hz: float
    frequency_max_Hz: float


class RandlesCircuitFitter:
    def __init__(self, min_points: int = 8):
        self.min_points = min_points

    def fit_eis(self, frequency: Iterable[float], z_real: Iterable[float], z_imag: Iterable[float]) -> EISFeatures:
        f = np.asarray(list(frequency), dtype=float)
        zr = np.asarray(list(z_real), dtype=float)
        zi = np.asarray(list(z_imag), dtype=float)
        mask = np.isfinite(f) & np.isfinite(zr) & np.isfinite(zi) & (f > 0)
        f, zr, zi = f[mask], zr[mask], zi[mask]
        if len(f) < self.min_points:
            return EISFeatures(np.nan, np.nan, np.nan, np.nan, np.nan, "too_few_points", int(len(f)), np.nan, np.nan)
        order = np.argsort(f)[::-1]
        f, zr, zi = f[order], zr[order], zi[order]
        high_n = max(3, len(f) // 10)
        low_n = max(3, len(f) // 8)
        R_ohm = float(np.nanmedian(zr[:high_n]))
        low_real = float(np.nanmedian(zr[-low_n:]))
        arc_span = max(low_real - R_ohm, 0.0)
        mid = slice(len(f) // 3, max(len(f) // 3 + 3, 2 * len(f) // 3))
        R_sei = float(max(np.nanpercentile(zr[mid], 75) - R_ohm, 0.0))
        R_ct = float(max(arc_span - R_sei, 0.0))
        omega_inv_sqrt = 1.0 / np.sqrt(2.0 * np.pi * f[-low_n:])
        zr_low = zr[-low_n:]
        if np.nanstd(omega_inv_sqrt) > 0:
            sigma = float(max(np.polyfit(omega_inv_sqrt, zr_low, 1)[0], 0.0))
        else:
            sigma = 0.0
        cpe_alpha = self._estimate_cpe_alpha(f, zi)
        flag = self._quality_flag(R_ohm, R_ct, R_sei, sigma, zr, zi)
        return EISFeatures(
            R_ohm=R_ohm,
            R_ct=R_ct,
            R_sei=R_sei,
            sigma_warburg=sigma,
            cpe_alpha=cpe_alpha,
            quality_flag=flag,
            n_points=int(len(f)),
            frequency_min_Hz=float(np.nanmin(f)),
            frequency_max_Hz=float(np.nanmax(f)),
        )

    def transform_many(self, rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        out = []
        for row in rows:
            features = self.fit_eis(row.get("frequency_Hz", []), row.get("z_real_ohm", []), row.get("z_imag_ohm", []))
            item = dict(row)
            item.update(asdict(features))
            out.append(item)
        return out

    @staticmethod
    def _estimate_cpe_alpha(frequency: np.ndarray, z_imag: np.ndarray) -> float:
        mag = np.abs(z_imag)
        mask = np.isfinite(mag) & (mag > 0) & np.isfinite(frequency) & (frequency > 0)
        if mask.sum() < 5:
            return float("nan")
        x = np.log10(frequency[mask])
        y = np.log10(mag[mask])
        slope = np.polyfit(x, y, 1)[0]
        return float(np.clip(-slope, 0.0, 1.0))

    @staticmethod
    def _quality_flag(R_ohm: float, R_ct: float, R_sei: float, sigma: float, z_real: np.ndarray, z_imag: np.ndarray) -> str:
        if not np.isfinite([R_ohm, R_ct, R_sei, sigma]).all():
            return "invalid_fit"
        if R_ohm < 0 or R_ct < 0 or R_sei < 0:
            return "negative_resistance"
        if R_ohm > 10 or R_ct > 100 or R_sei > 100:
            return "extreme_resistance_outlier"
        if np.nanmax(np.abs(z_real)) > 1e4 or np.nanmax(np.abs(z_imag)) > 1e4:
            return "raw_impedance_outlier"
        return "usable"


def eis_features_to_node_tensor(features: EISFeatures, fallback: Optional[float] = 0.0) -> np.ndarray:
    vals = np.array([features.R_ohm, features.R_ct, features.R_sei, features.sigma_warburg, features.cpe_alpha], dtype=float)
    vals[~np.isfinite(vals)] = float(fallback or 0.0)
    return vals
