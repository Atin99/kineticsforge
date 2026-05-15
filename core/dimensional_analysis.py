import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    dimension: str


@dataclass
class UnitCheck:
    name: str
    passed: bool
    normalized_value: float
    normalized_unit: str
    message: str


class UnitNormalizer:
    ENERGY = {
        "J": 1.0,
        "kJ": 1000.0,
        "Wh": 3600.0,
        "kWh": 3_600_000.0,
        "eV": 1.602176634e-19,
    }
    TEMPERATURE = {"K": "K", "C": "C", "degC": "C"}
    LENGTH = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "micron": 1e-6, "nm": 1e-9}
    MASS = {"kg": 1.0, "g": 1e-3, "mg": 1e-6}
    TIME = {"s": 1.0, "min": 60.0, "h": 3600.0}
    CURRENT = {"A": 1.0, "mA": 1e-3}
    VOLTAGE = {"V": 1.0, "mV": 1e-3}
    MONEY_INR = {"INR": 1.0, "rupee": 1.0, "paise": 0.01}

    def to_si(self, value: float, unit: str) -> Quantity:
        unit = unit.strip()
        if unit in self.TEMPERATURE:
            return Quantity(float(value) + 273.15 if self.TEMPERATURE[unit] == "C" else float(value), "K", "temperature")
        for dimension, table, si in [
            ("energy", self.ENERGY, "J"),
            ("length", self.LENGTH, "m"),
            ("mass", self.MASS, "kg"),
            ("time", self.TIME, "s"),
            ("current", self.CURRENT, "A"),
            ("voltage", self.VOLTAGE, "V"),
            ("money", self.MONEY_INR, "INR"),
        ]:
            if unit in table:
                return Quantity(float(value) * table[unit], si, dimension)
        if unit in ("fraction", "score", "1"):
            return Quantity(float(value), "1", "dimensionless")
        raise ValueError(f"Unknown unit: {unit}")

    def check_range(self, name: str, value: float, unit: str, lo_si: float, hi_si: float, si_unit: str) -> UnitCheck:
        q = self.to_si(value, unit)
        passed = q.unit == si_unit and lo_si <= q.value <= hi_si and math.isfinite(q.value)
        if not math.isfinite(q.value):
            message = f"{name} is not finite after unit normalization."
        elif q.unit != si_unit:
            message = f"{name} normalized to {q.unit}, expected {si_unit}."
        elif q.value < lo_si or q.value > hi_si:
            message = f"{name}={q.value:.6g} {q.unit} outside [{lo_si:.6g}, {hi_si:.6g}] {si_unit}."
        else:
            message = f"{name}={q.value:.6g} {q.unit} passes."
        return UnitCheck(name=name, passed=passed, normalized_value=q.value, normalized_unit=q.unit, message=message)


class BatteryParameterSanity:
    def __init__(self) -> None:
        self.units = UnitNormalizer()

    def cathode(self, params: Dict[str, float]) -> List[UnitCheck]:
        return [
            self.units.check_range("capacity", params.get("capacity_mAh_g", 150.0), "1", 40.0, 260.0, "1"),
            self.units.check_range("temperature", params.get("temperature_C", 45.0), "C", 273.15, 363.15, "K"),
            self.units.check_range("fade_fraction", params.get("fade_fraction", 0.10), "fraction", 0.0, 0.70, "1"),
            self.units.check_range("cost_inr_kwh", params.get("cost_inr_kwh", 7000.0), "INR", 0.0, 60000.0, "INR"),
        ]

    def bms(self, params: Dict[str, float]) -> List[UnitCheck]:
        return [
            self.units.check_range("cell_voltage", params.get("voltage_V", 3.3), "V", 1.5, 5.2, "V"),
            self.units.check_range("cell_temperature", params.get("temperature_C", 45.0), "C", 250.0, 460.0, "K"),
            self.units.check_range("current", params.get("current_A", 5.0), "A", -500.0, 500.0, "A"),
            self.units.check_range("risk", params.get("risk", 0.5), "fraction", 0.0, 1.0, "1"),
        ]

    def recycling(self, params: Dict[str, float]) -> List[UnitCheck]:
        return [
            self.units.check_range("leach_temperature", params.get("temperature_C", 70.0), "C", 293.15, 383.15, "K"),
            self.units.check_range("leach_time", params.get("time_min", 120.0), "min", 60.0, 21600.0, "s"),
            self.units.check_range("particle_radius", params.get("particle_um", 50.0), "um", 1e-6, 300e-6, "m"),
            self.units.check_range("recovery", params.get("recovery", 0.85), "fraction", 0.0, 1.0, "1"),
        ]


def checks_to_dict(checks: Iterable[UnitCheck]) -> List[Dict[str, object]]:
    return [check.__dict__ for check in checks]
