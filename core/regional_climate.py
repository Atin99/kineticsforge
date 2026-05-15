import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests


@dataclass(frozen=True)
class RegionSeed:
    name: str
    latitude: float
    longitude: float
    mean_temp_C: float
    seasonal_amp_C: float
    diurnal_amp_C: float
    mean_rh_percent: float
    monsoon_amp_percent: float


@dataclass
class RegionalClimateProfile:
    region: str
    latitude: float
    longitude: float
    source_kind: str
    hours: List[int]
    temperature_C: List[float]
    relative_humidity_percent: List[float]
    heat_stress_index: List[float]
    cold_plating_index: List[float]
    metadata: Dict[str, Any]


REGIONS: Dict[str, RegionSeed] = {
    "delhi_hot": RegionSeed("delhi_hot", 28.6139, 77.2090, 29.0, 12.0, 6.5, 46.0, 28.0),
    "chennai_coastal": RegionSeed("chennai_coastal", 13.0827, 80.2707, 29.0, 4.0, 3.2, 72.0, 16.0),
    "mumbai_monsoon": RegionSeed("mumbai_monsoon", 19.0760, 72.8777, 28.0, 5.0, 3.5, 76.0, 20.0),
    "jaipur_desert": RegionSeed("jaipur_desert", 26.9124, 75.7873, 28.0, 14.0, 8.0, 38.0, 24.0),
    "leh_cold": RegionSeed("leh_cold", 34.1526, 77.5770, 6.0, 17.0, 8.5, 35.0, 10.0),
    "guwahati_humid": RegionSeed("guwahati_humid", 26.1445, 91.7362, 25.0, 7.0, 3.5, 78.0, 18.0),
}


class RegionalClimateEngine:
    def __init__(self, timeout_s: int = 45):
        self.timeout_s = timeout_s

    def profile(
        self,
        region: str = "delhi_hot",
        days: int = 14,
        start_date: str = "",
        end_date: str = "",
        prefer_api: bool = False,
    ) -> RegionalClimateProfile:
        seed = REGIONS.get(region, REGIONS["delhi_hot"])
        if prefer_api and start_date and end_date:
            api = self.nasa_power(seed, start_date, end_date)
            if api is not None:
                return api
        return self.synthetic_climatology(seed, days)

    def nasa_power(self, seed: RegionSeed, start_date: str, end_date: str) -> Optional[RegionalClimateProfile]:
        params = {
            "parameters": "T2M,RH2M",
            "community": "RE",
            "longitude": seed.longitude,
            "latitude": seed.latitude,
            "start": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "format": "JSON",
        }
        try:
            response = requests.get("https://power.larc.nasa.gov/api/temporal/hourly/point", params=params, timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("properties", {}).get("parameter", {})
            temp = data.get("T2M") or {}
            rh = data.get("RH2M") or {}
            keys = sorted(set(temp) & set(rh))
            if not keys:
                return None
            temperatures = [float(temp[k]) for k in keys]
            humidity = [float(rh[k]) for k in keys]
            hours = list(range(len(keys)))
            return self._build(seed, "nasa_power_hourly", hours, temperatures, humidity, {"api_keys": keys[:3] + keys[-3:], "start": start_date, "end": end_date})
        except Exception:
            return None

    def synthetic_climatology(self, seed: RegionSeed, days: int) -> RegionalClimateProfile:
        n = max(24, int(days) * 24)
        hours = np.arange(n, dtype=float)
        day = hours / 24.0
        seasonal = seed.seasonal_amp_C * np.sin(2.0 * math.pi * (day / 365.0 + 0.18))
        diurnal = seed.diurnal_amp_C * np.sin(2.0 * math.pi * (hours % 24.0 - 14.0) / 24.0)
        temp = seed.mean_temp_C + seasonal + diurnal
        monsoon = seed.monsoon_amp_percent * np.sin(2.0 * math.pi * (day / 365.0 - 0.05))
        humidity = np.clip(seed.mean_rh_percent + monsoon - 0.45 * diurnal, 8.0, 98.0)
        return self._build(
            seed,
            "regional_climatology_fallback",
            hours.astype(int).tolist(),
            temp.astype(float).tolist(),
            humidity.astype(float).tolist(),
            {"days": int(days), "note": "Use NASA POWER or ERA5 for deployment claims; this fallback is for design sweeps only."},
        )

    def _build(
        self,
        seed: RegionSeed,
        source_kind: str,
        hours: List[int],
        temperature_C: List[float],
        humidity_percent: List[float],
        metadata: Dict[str, Any],
    ) -> RegionalClimateProfile:
        temp = np.asarray(temperature_C, dtype=float)
        rh = np.asarray(humidity_percent, dtype=float)
        heat = 1.0 / (1.0 + np.exp(-(temp - 42.0) / 3.5)) * (1.0 + 0.004 * np.maximum(rh - 60.0, 0.0))
        cold = 1.0 / (1.0 + np.exp((temp - 2.0) / 3.0))
        return RegionalClimateProfile(
            region=seed.name,
            latitude=seed.latitude,
            longitude=seed.longitude,
            source_kind=source_kind,
            hours=hours,
            temperature_C=temp.astype(float).tolist(),
            relative_humidity_percent=rh.astype(float).tolist(),
            heat_stress_index=np.clip(heat, 0.0, 1.5).astype(float).tolist(),
            cold_plating_index=np.clip(cold, 0.0, 1.0).astype(float).tolist(),
            metadata=metadata,
        )


def all_regions_365(engine: Optional["RegionalClimateEngine"] = None) -> Dict[str, "RegionalClimateProfile"]:
    """Generate full 365-day profiles for all 6 Indian regions.

    This is the India battery intelligence moat: no Western platform has
    region-specific degradation conditioning at this granularity.

    Returns dict keyed by region name -> RegionalClimateProfile.
    """
    eng = engine or RegionalClimateEngine()
    return {name: eng.profile(name, days=365) for name in REGIONS}


def temperature_conditioning_tensor(profile: "RegionalClimateProfile") -> np.ndarray:
    """Convert climate profile to a T(t) conditioning array for UDE integration.

    Returns shape (N_hours,) array of temperatures in Kelvin, ready to be
    passed as conditioning input to the UDE degradation model.
    """
    return np.array(profile.temperature_C, dtype=np.float64) + 273.15


def humidity_conditioning_tensor(profile: "RegionalClimateProfile") -> np.ndarray:
    """Convert climate profile to RH(t) conditioning array (fraction 0-1)."""
    return np.array(profile.relative_humidity_percent, dtype=np.float64) / 100.0


def compare_regions_summary(profiles: Optional[Dict[str, "RegionalClimateProfile"]] = None) -> Dict[str, Dict[str, float]]:
    """Quick comparison table: mean T, max T, mean RH, heat-stress hours, cold-plating hours per region."""
    if profiles is None:
        profiles = all_regions_365()
    summary = {}
    for name, p in profiles.items():
        t = np.array(p.temperature_C)
        rh = np.array(p.relative_humidity_percent)
        hs = np.array(p.heat_stress_index)
        cp = np.array(p.cold_plating_index)
        summary[name] = {
            "mean_T_C": float(np.mean(t)),
            "max_T_C": float(np.max(t)),
            "min_T_C": float(np.min(t)),
            "mean_RH_pct": float(np.mean(rh)),
            "heat_stress_hours": int(np.sum(hs > 0.1)),
            "cold_plating_hours": int(np.sum(cp > 0.1)),
            "total_hours": len(t),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regional India climate stress profiles for battery degradation sweeps.")
    parser.add_argument("--region", default="delhi_hot", choices=sorted(REGIONS))
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--prefer-api", action="store_true")
    parser.add_argument("--all-regions-365", action="store_true", help="Generate full 365-day profiles for all 6 regions.")
    parser.add_argument("--out", default="data/cache/regional_climate_profile.json")
    args = parser.parse_args()

    if args.all_regions_365:
        profiles = all_regions_365()
        summary = compare_regions_summary(profiles)
        out = Path(args.out).parent / "all_regions_365_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"regions": list(summary.keys()), "out": str(out)}, indent=2))
    else:
        profile = RegionalClimateEngine().profile(args.region, args.days, args.start_date, args.end_date, args.prefer_api)
        payload = asdict(profile)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"region": profile.region, "source_kind": profile.source_kind, "hours": len(profile.hours), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
