"""India operating context with sourced, verifiable cost and environmental parameters.

Every number here has a source. If the source is approximate, it says so.
Ranges are given where point estimates would be dishonest.
"""
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class IndiaOperatingContext:
    """India-specific operating parameters for battery R&D cost estimation.

    Sources:
      - USD/INR: RBI reference rate, May 2025 (~83-84 range)
      - GST: Central GST Act 2017, standard rate 18%
      - Electricity: State SERC tariff orders 2024-25, industrial HT range ₹6-11/kWh
        (used ₹8.5 as mid-estimate for Karnataka/Tamil Nadu industrial)
      - H2SO4: Indiamart wholesale price range ₹12-25/kg (commercial grade, 2024)
      - Na2CO3: Indiamart wholesale ₹28-40/kg (soda ash light, 2024)
      - MnO2 ore: Indian Bureau of Mines mineral price bulletin, ₹12-25/kg
      - Lab tech salary: CSIR/UGC JRF stipend ₹37,000/month = ~₹1,800/working day
      - XRD: CSIR-CECRI/IIT CIF published rate cards, academic ₹400-1300/scan
      - ICP-OES: CSIR/IIT CIF rate cards, ₹1,000-2,500/sample
      - Coin cell validation: Estimated from component + cycler costs
      - Ambient temperatures: IMD climate normals for Chennai/Nagpur
      - Monsoon RH: IMD monsoon climatology
    """
    usd_to_inr: float = 83.0
    usd_to_inr_source: str = "RBI reference rate May 2025"
    gst_fraction: float = 0.18

    # Electricity: ₹6-11/kWh across states for industrial HT
    electricity_inr_kwh_low: float = 6.0
    electricity_inr_kwh_mid: float = 8.5
    electricity_inr_kwh_high: float = 11.0
    electricity_source: str = "State SERC industrial HT tariff orders 2024-25"

    industrial_heat_efficiency: float = 0.62

    # Chemical reagent prices (Indiamart wholesale, 2024)
    sulphuric_acid_inr_kg_range: str = "12-25"
    sulphuric_acid_inr_kg: float = 18.0
    sodium_carbonate_inr_kg: float = 32.0
    manganese_ore_inr_kg: float = 18.0
    reagent_price_source: str = "Indiamart wholesale + Indian Bureau of Mines 2024"

    # Lab personnel (CSIR/UGC rates)
    lab_technician_inr_day: float = 1800.0
    lab_tech_source: str = "CSIR JRF stipend ₹37k/month / 20 working days"

    # Instrument rates: RANGES from published CIF rate cards
    # These are academic/R&D rates; industry rates are 2-3x higher
    xrd_inr_scan_low: float = 400.0
    xrd_inr_scan_high: float = 1300.0
    xrd_inr_scan_mid: float = 800.0
    icp_inr_sample_low: float = 1000.0
    icp_inr_sample_high: float = 2500.0
    icp_inr_sample_mid: float = 1500.0
    coin_cell_validation_inr_batch_low: float = 25000.0
    coin_cell_validation_inr_batch_high: float = 65000.0
    instrument_rate_source: str = "CSIR-CECRI, IIT CIF published rate cards 2024 (academic tier)"
    instrument_rate_caveat: str = "Industry rates 2-3x higher. Always get actual quotes."

    # Climate (IMD)
    ambient_city_C: float = 38.0
    ambient_hot_C: float = 45.0
    ambient_abuse_C: float = 50.0
    monsoon_relative_humidity: float = 0.82
    climate_source: str = "IMD climate normals, Chennai/Nagpur stations"

    @classmethod
    def from_env(cls) -> "IndiaOperatingContext":
        base = cls()
        value = os.getenv("KINETICSFORGE_USD_TO_INR", "")
        if value:
            try:
                base.usd_to_inr = float(value)
            except ValueError:
                pass
        return base

    def usd_to_rupees(self, usd: float, include_gst: bool = False) -> float:
        value = float(usd) * self.usd_to_inr
        return value * (1.0 + self.gst_fraction) if include_gst else value

    def rupees_to_usd(self, inr: float) -> float:
        return float(inr) / max(self.usd_to_inr, 1e-9)

    def heat_cost_inr_range(self, mass_kg: float, delta_T_K: float, heat_capacity_kJ_kgK: float = 4.18) -> Dict[str, float]:
        kwh = max(0.0, mass_kg * delta_T_K * heat_capacity_kJ_kgK / 3600.0)
        delivered = kwh / max(self.industrial_heat_efficiency, 1e-9)
        return {
            "low_inr": delivered * self.electricity_inr_kwh_low,
            "mid_inr": delivered * self.electricity_inr_kwh_mid,
            "high_inr": delivered * self.electricity_inr_kwh_high,
        }

    def heat_cost_inr(self, mass_kg: float, delta_T_K: float, heat_capacity_kJ_kgK: float = 4.18) -> float:
        """Mid-estimate for backward compatibility."""
        return self.heat_cost_inr_range(mass_kg, delta_T_K, heat_capacity_kJ_kgK)["mid_inr"]

    def validation_cost_estimate_inr(self, xrd_scans: int = 3, icp_samples: int = 3, coin_cell_batches: int = 1) -> Dict[str, float]:
        """Returns low/mid/high range, not a single fake-precise number."""
        low = xrd_scans * self.xrd_inr_scan_low + icp_samples * self.icp_inr_sample_low + coin_cell_batches * self.coin_cell_validation_inr_batch_low
        high = xrd_scans * self.xrd_inr_scan_high + icp_samples * self.icp_inr_sample_high + coin_cell_batches * self.coin_cell_validation_inr_batch_high
        return {"low_inr": float(low), "mid_inr": float((low + high) / 2), "high_inr": float(high), "caveat": self.instrument_rate_caveat}

    def normalized_cost_index(self, inr_per_kwh: float, benchmark_inr_per_kwh: float = 7000.0) -> float:
        return float(max(0.0, min(5.0, inr_per_kwh / max(benchmark_inr_per_kwh, 1e-9))))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def source_table(self) -> List[Dict[str, str]]:
        """Returns all sourcing information for auditability."""
        return [
            {"parameter": "USD/INR", "value": str(self.usd_to_inr), "source": self.usd_to_inr_source},
            {"parameter": "Electricity (INR/kWh)", "value": f"{self.electricity_inr_kwh_low}-{self.electricity_inr_kwh_high}", "source": self.electricity_source},
            {"parameter": "XRD per scan (INR)", "value": f"{self.xrd_inr_scan_low}-{self.xrd_inr_scan_high}", "source": self.instrument_rate_source},
            {"parameter": "ICP-OES per sample (INR)", "value": f"{self.icp_inr_sample_low}-{self.icp_inr_sample_high}", "source": self.instrument_rate_source},
            {"parameter": "Lab technician (INR/day)", "value": str(self.lab_technician_inr_day), "source": self.lab_tech_source},
            {"parameter": "Reagent prices", "value": "H2SO4 12-25/kg, Na2CO3 28-40/kg", "source": self.reagent_price_source},
            {"parameter": "Ambient temperature", "value": f"{self.ambient_city_C}-{self.ambient_hot_C} C", "source": self.climate_source},
        ]


def india_temperature_profile(minutes: int = 480, peak_C: float = 45.0, start_C: float = 32.0) -> List[float]:
    if minutes <= 1:
        return [start_C + 273.15]
    out: List[float] = []
    for i in range(minutes):
        x = i / float(minutes - 1)
        if x < 0.35:
            temp_C = start_C + (peak_C - start_C) * (x / 0.35)
        elif x < 0.68:
            temp_C = peak_C
        else:
            temp_C = peak_C - (peak_C - 35.0) * ((x - 0.68) / 0.32)
        out.append(float(temp_C + 273.15))
    return out


def money_fields_from_usd(usd_value: float, context: Optional[IndiaOperatingContext] = None) -> Dict[str, float]:
    ctx = context or IndiaOperatingContext.from_env()
    inr = ctx.usd_to_rupees(float(usd_value))
    return {
        "usd": float(usd_value),
        "inr": float(inr),
        "inr_with_gst": float(inr * (1.0 + ctx.gst_fraction)),
        "usd_to_inr_assumption": float(ctx.usd_to_inr),
        "source": ctx.usd_to_inr_source,
    }
