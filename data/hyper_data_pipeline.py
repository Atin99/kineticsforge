import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from data.dataset_contracts import (
    DatasetManifest,
    arrhenius,
    bounded_sigmoid,
    clip01,
    describe_array,
    ensure_dir,
    finite_slope,
    monotonicity_fraction,
    register_file,
    source_catalog,
    stable_noise,
    utc_now,
    validate_range,
    write_json,
    validate_finite,
    write_source_catalog,
    write_validation_report,
    ValidationIssue,
)


SCHEMA_VERSION = "kineticsforge-data-v3"
DOPANT_MAP = {"None": 0.0, "Al": 1.0, "Ti": 2.0, "Mg": 3.0, "Ni": 4.0, "Cu": 5.0, "Zn": 6.0}
DOPANT_REVERSE = {v: k for k, v in DOPANT_MAP.items()}


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    cathode_compositions: int
    cathode_cycles: int
    cathode_temperatures: Tuple[float, ...]
    cathode_c_rates: Tuple[float, ...]
    bms_scenarios: int
    bms_seconds: int
    leaching_temperatures: Tuple[float, ...]
    leaching_ph: Tuple[float, ...]
    leaching_conc: Tuple[float, ...]
    leaching_particles_um: Tuple[float, ...]
    compatibility_files: int


PROFILES: Dict[str, PipelineProfile] = {
    "smoke": PipelineProfile(
        "smoke",
        cathode_compositions=48,
        cathode_cycles=251,
        cathode_temperatures=(298.15, 318.15),
        cathode_c_rates=(0.5, 1.0),
        bms_scenarios=8,
        bms_seconds=1200,
        leaching_temperatures=(323.15, 343.15, 363.15),
        leaching_ph=(0.5, 1.5, 2.5),
        leaching_conc=(0.5, 1.5, 3.0),
        leaching_particles_um=(10.0, 50.0, 100.0),
        compatibility_files=16,
    ),
    "foundation": PipelineProfile(
        "foundation",
        cathode_compositions=240,
        cathode_cycles=501,
        cathode_temperatures=(298.15, 308.15, 318.15, 328.15),
        cathode_c_rates=(0.3, 0.5, 1.0),
        bms_scenarios=40,
        bms_seconds=3600,
        leaching_temperatures=(323.15, 333.15, 343.15, 353.15, 363.15),
        leaching_ph=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        leaching_conc=(0.5, 1.0, 1.5, 2.0, 3.0),
        leaching_particles_um=(10.0, 25.0, 50.0, 75.0, 100.0),
        compatibility_files=96,
    ),
    "hyper": PipelineProfile(
        "hyper",
        cathode_compositions=1200,
        cathode_cycles=751,
        cathode_temperatures=(288.15, 298.15, 308.15, 318.15, 328.15, 338.15),
        cathode_c_rates=(0.2, 0.5, 1.0, 2.0),
        bms_scenarios=120,
        bms_seconds=28800,
        leaching_temperatures=(313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 373.15),
        leaching_ph=(0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0),
        leaching_conc=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
        leaching_particles_um=(5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0),
        compatibility_files=240,
    ),
}


class CompositionLibrary:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def dense_grid(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        comp_id = 0
        for na in np.linspace(0.86, 1.14, 15):
            for mn in np.linspace(0.12, 0.88, 33):
                for dopant in ("None", "Al", "Ti", "Mg", "Ni", "Cu", "Zn"):
                    fractions = (0.0,) if dopant == "None" else (0.015, 0.03, 0.05, 0.075)
                    for dopant_frac in fractions:
                        transition_total = 1.0 - dopant_frac
                        mn_eff = mn * transition_total
                        fe_eff = max(0.0, transition_total - mn_eff)
                        oxygen = 2.0 + 0.04 * (na - 1.0) - 0.02 * dopant_frac
                        rows.append(
                            {
                                "comp_id": comp_id,
                                "Na": float(na),
                                "Mn": float(mn_eff),
                                "Fe": float(fe_eff),
                                "dopant": dopant,
                                "dopant_frac": float(dopant_frac),
                                "O": float(oxygen),
                                "Mn_fraction_raw": float(mn),
                            }
                        )
                        comp_id += 1
        return pd.DataFrame(rows)

    def feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        dopant_code = df["dopant"].map(DOPANT_MAP).to_numpy(dtype=float) / max(DOPANT_MAP.values())
        x = np.column_stack(
            [
                df["Na"].to_numpy(dtype=float),
                df["Mn"].to_numpy(dtype=float),
                df["Fe"].to_numpy(dtype=float),
                df["dopant_frac"].to_numpy(dtype=float),
                df["O"].to_numpy(dtype=float),
                dopant_code,
            ]
        )
        scale = np.std(x, axis=0) + 1e-9
        return (x - np.mean(x, axis=0)) / scale

    def farthest_point_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n >= len(df):
            return df.reset_index(drop=True)
        x = self.feature_matrix(df)
        selected = [int(self.rng.integers(0, len(df)))]
        min_dist = np.linalg.norm(x - x[selected[0]], axis=1)
        for _ in range(1, n):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            dist = np.linalg.norm(x - x[idx], axis=1)
            min_dist = np.minimum(min_dist, dist)
        out = df.iloc[selected].copy().reset_index(drop=True)
        out["comp_id"] = np.arange(len(out), dtype=int)
        return out

    def build(self, n: int) -> pd.DataFrame:
        df = self.farthest_point_sample(self.dense_grid(), n)
        df["formula"] = df.apply(self.formula, axis=1)
        df["theoretical_capacity_mAh_g"] = df.apply(self.theoretical_capacity, axis=1)
        df["hull_energy_proxy_meV_atom"] = df.apply(self.hull_energy_proxy, axis=1)
        df["na_diffusivity_log10"] = df.apply(self.diffusivity_proxy, axis=1)
        df["redox_center_balance"] = df["Mn"] / (df["Mn"] + df["Fe"] + df["dopant_frac"] + 1e-9)
        return df

    @staticmethod
    def formula(row: pd.Series) -> str:
        dop = "" if row["dopant"] == "None" else f"{row['dopant']}{row['dopant_frac']:.3f}"
        return f"Na{row['Na']:.3f}Mn{row['Mn']:.3f}Fe{row['Fe']:.3f}{dop}O{row['O']:.3f}"

    @staticmethod
    def theoretical_capacity(row: pd.Series) -> float:
        dopant_penalty = {"None": 0.0, "Al": -4.0, "Ti": -2.0, "Mg": -6.0, "Ni": 6.0, "Cu": 2.0, "Zn": -5.0}[row["dopant"]]
        q = 122.0 + 58.0 * row["Mn"] + 18.0 * row["Fe"] + dopant_penalty - 22.0 * abs(row["Na"] - 1.0)
        q -= 35.0 * max(0.0, row["dopant_frac"] - 0.05)
        return float(np.clip(q, 70.0, 205.0))

    @staticmethod
    def hull_energy_proxy(row: pd.Series) -> float:
        balance = row["Mn"] / (row["Mn"] + row["Fe"] + 1e-9)
        dopant_bonus = {"None": 0.0, "Al": -9.0, "Ti": -6.0, "Mg": -3.0, "Ni": 8.0, "Cu": 12.0, "Zn": 4.0}[row["dopant"]]
        return float(np.clip(18.0 + 62.0 * abs(balance - 0.48) + 45.0 * abs(row["Na"] - 1.0) + dopant_bonus, 0.0, 120.0))

    @staticmethod
    def diffusivity_proxy(row: pd.Series) -> float:
        channel = 1.0 - 1.6 * abs(row["Na"] - 1.0)
        dopant = {"None": 0.0, "Al": -0.05, "Ti": 0.12, "Mg": 0.05, "Ni": -0.02, "Cu": -0.08, "Zn": 0.02}[row["dopant"]]
        return float(np.clip(-12.4 + 0.7 * channel + 0.25 * row["Fe"] + dopant, -14.5, -10.2))


class CathodeHyperGenerator:
    def __init__(self, rng: np.random.Generator, root: Path, profile: PipelineProfile, calibration_priors: Dict[str, Any] | None = None):
        self.rng = rng
        self.root = root
        self.profile = profile
        self.calibration_priors = calibration_priors or {}
        self.out_dir = ensure_dir(root / "synthetic" / "hyper")
        self.compat_dir = ensure_dir(root / "synthetic" / "cathode")

    def trajectory_parameters(self, comp: pd.Series, temp_k: float, c_rate: float) -> Dict[str, float]:
        mn_ratio = comp["Mn"] / (comp["Mn"] + comp["Fe"] + comp["dopant_frac"] + 1e-9)
        dopant = comp["dopant"]
        dopant_life = {"None": 1.0, "Al": 1.24, "Ti": 1.16, "Mg": 1.08, "Ni": 0.95, "Cu": 0.88, "Zn": 1.02}[dopant]
        dopant_capacity = {"None": 0.0, "Al": 9.0, "Ti": 5.0, "Mg": 2.0, "Ni": 12.0, "Cu": 4.0, "Zn": 1.5}[dopant]
        q0_center = float(self.calibration_priors.get("q0_center_mAh_g", 145.0))
        q0_spread = float(self.calibration_priors.get("q0_spread_mAh_g", 14.0))
        literature_anchor = q0_center + self.rng.normal(0.0, 0.35 * q0_spread)
        q0 = 0.74 * (comp["theoretical_capacity_mAh_g"] + dopant_capacity) + 0.26 * literature_anchor + self.rng.normal(0.0, 3.5)
        q0 = float(np.clip(q0, 80.0, 220.0))
        ea_ev = 0.48 + 0.18 * mn_ratio + 0.04 * c_rate + 0.03 * abs(comp["Na"] - 1.0)
        ea_ev += {"None": 0.0, "Al": 0.05, "Ti": 0.04, "Mg": 0.02, "Ni": -0.02, "Cu": -0.04, "Zn": 0.0}[dopant]
        thermal_factor = arrhenius(3.5e5, ea_ev, temp_k) * 1.0e3
        mn_dissolution = (0.010 + 0.028 * max(0.0, mn_ratio - 0.52) ** 2) * np.exp((temp_k - 298.15) / 65.0)
        structure_penalty = comp["hull_energy_proxy_meV_atom"] / 2200.0
        c_rate_penalty = 0.004 * c_rate ** 1.35
        retention_center = float(self.calibration_priors.get("retention_center_percent", 84.0))
        retention_pressure = np.clip((88.0 - retention_center) / 120.0, -0.04, 0.08)
        base_fade = float(np.clip(thermal_factor + mn_dissolution + structure_penalty + c_rate_penalty + retention_pressure, 0.004, 0.24))
        knee_cycle = float(np.clip((760.0 * dopant_life) / (1.0 + 0.75 * base_fade + 0.22 * c_rate), 140.0, 1400.0))
        knee_sharpness = float(np.clip(0.010 + 0.020 * mn_ratio + 0.006 * c_rate, 0.006, 0.055))
        resistance0 = float(0.011 + 0.006 * c_rate + 0.00008 * comp["hull_energy_proxy_meV_atom"] + self.rng.normal(0.0, 0.0009))
        return {
            "q0": q0,
            "ea_ev": float(ea_ev),
            "base_fade": base_fade,
            "knee_cycle": knee_cycle,
            "knee_sharpness": knee_sharpness,
            "resistance0": max(0.006, resistance0),
        }

    def simulate(self, compositions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        cycles = np.arange(self.profile.cathode_cycles, dtype=np.float32)
        row_frames: List[pd.DataFrame] = []
        summaries: List[Dict[str, Any]] = []
        compat_written = 0
        trajectory_id = 0
        for _, comp in compositions.iterrows():
            for temp_k in self.profile.cathode_temperatures:
                for c_rate in self.profile.cathode_c_rates:
                    params = self.trajectory_parameters(comp, temp_k, c_rate)
                    normalized_cycle = cycles / max(cycles[-1], 1.0)
                    linear_loss = params["base_fade"] * normalized_cycle ** (0.72 + 0.18 * c_rate)
                    knee = bounded_sigmoid((cycles - params["knee_cycle"]) * params["knee_sharpness"], 0.0, 1.0)
                    knee_loss = (0.035 + 0.18 * params["base_fade"]) * knee * normalized_cycle
                    sei_loss = 0.012 * np.sqrt(cycles / 500.0 + 1e-6) * np.exp((temp_k - 298.15) / 80.0)
                    reversible_loss = 0.004 * np.sin(2 * np.pi * cycles / 47.0 + comp["Mn"] * 4.0)
                    clean_capacity = params["q0"] * (1.0 - linear_loss - knee_loss - sei_loss + reversible_loss)
                    clean_capacity = np.maximum(clean_capacity, params["q0"] * 0.42)
                    measurement_noise = stable_noise(self.rng, len(cycles), sigma=0.006 * params["q0"], phi=0.72)
                    capacity = np.maximum(clean_capacity + measurement_noise, params["q0"] * 0.35)
                    resistance = params["resistance0"] + 0.000018 * cycles ** 1.18 + 0.004 * knee + 0.0006 * c_rate
                    resistance += stable_noise(self.rng, len(cycles), 0.00012, phi=0.88)
                    resistance = np.maximum(resistance, 0.004)
                    voltage_mean = 3.05 + 0.62 * (capacity / params["q0"]) + 0.16 * comp["Mn"] - 0.08 * comp["Fe"]
                    voltage_mean -= c_rate * resistance * 0.9
                    voltage_mean += stable_noise(self.rng, len(cycles), 0.0045, phi=0.65)
                    voltage_mean = np.clip(voltage_mean, 2.0, 4.25)
                    dqdv_peak_1 = 2.72 + 0.42 * comp["Mn"] - 0.10 * normalized_cycle + self.rng.normal(0, 0.006, len(cycles))
                    dqdv_peak_2 = 3.35 + 0.25 * comp["Fe"] - 0.06 * normalized_cycle + self.rng.normal(0, 0.006, len(cycles))
                    entropy_coeff = (0.00012 + 0.00004 * comp["Mn"] - 0.00003 * comp["Fe"]) * (1.0 + 0.8 * normalized_cycle)
                    coulombic_eff = 0.9992 - 0.0025 * normalized_cycle - 0.0008 * max(0.0, c_rate - 1.0)
                    coulombic_eff -= 0.0015 * bounded_sigmoid((cycles - params["knee_cycle"]) * params["knee_sharpness"])
                    coulombic_eff = np.clip(coulombic_eff + self.rng.normal(0, 0.00025, len(cycles)), 0.965, 1.002)
                    temperature_trace = temp_k + 2.0 * c_rate + 0.8 * np.sin(2 * np.pi * cycles / 37.0)
                    temperature_trace += stable_noise(self.rng, len(cycles), 0.55, phi=0.70)
                    soh = capacity / capacity[0]
                    eol = soh < 0.80
                    first_eol_cycle = int(cycles[np.argmax(eol)]) if np.any(eol) else -1
                    frame = pd.DataFrame(
                        {
                            "trajectory_id": trajectory_id,
                            "comp_id": int(comp["comp_id"]),
                            "cycle": cycles.astype(np.int32),
                            "temperature_K": np.float32(temp_k),
                            "c_rate": np.float32(c_rate),
                            "capacity_mAh_g": capacity.astype(np.float32),
                            "clean_capacity_mAh_g": clean_capacity.astype(np.float32),
                            "voltage_mean_V": voltage_mean.astype(np.float32),
                            "resistance_ohm": resistance.astype(np.float32),
                            "coulombic_efficiency": coulombic_eff.astype(np.float32),
                            "entropy_coeff_V_per_K": entropy_coeff.astype(np.float32),
                            "dqdv_peak_1_V": dqdv_peak_1.astype(np.float32),
                            "dqdv_peak_2_V": dqdv_peak_2.astype(np.float32),
                            "soh": soh.astype(np.float32),
                            "ambient_regime": "indian_hot" if temp_k >= 318.15 else "temperate",
                            "source_kind": "physics_synthetic",
                            "schema_version": SCHEMA_VERSION,
                        }
                    )
                    row_frames.append(frame)
                    summary = {
                        "trajectory_id": trajectory_id,
                        "comp_id": int(comp["comp_id"]),
                        "formula": comp["formula"],
                        "Na": float(comp["Na"]),
                        "Mn": float(comp["Mn"]),
                        "Fe": float(comp["Fe"]),
                        "O": float(comp["O"]),
                        "dopant": comp["dopant"],
                        "dopant_frac": float(comp["dopant_frac"]),
                        "temperature_K": float(temp_k),
                        "c_rate": float(c_rate),
                        "q0_mAh_g": float(capacity[0]),
                        "q_end_mAh_g": float(capacity[-1]),
                        "fade_fraction": float(1.0 - capacity[-1] / capacity[0]),
                        "first_eol_cycle": first_eol_cycle,
                        "knee_cycle_proxy": float(params["knee_cycle"]),
                        "ea_ev": float(params["ea_ev"]),
                        "hull_energy_proxy_meV_atom": float(comp["hull_energy_proxy_meV_atom"]),
                        "na_diffusivity_log10": float(comp["na_diffusivity_log10"]),
                        "capacity_slope_mAh_g_cycle": finite_slope(capacity[-min(120, len(capacity)) :]),
                    }
                    summaries.append(summary)
                    if compat_written < self.profile.compatibility_files:
                        self.write_cathode_compatibility(comp, frame, summary, compat_written)
                        compat_written += 1
                    trajectory_id += 1
        return pd.concat(row_frames, ignore_index=True), pd.DataFrame(summaries), summaries

    def write_cathode_compatibility(self, comp: pd.Series, frame: pd.DataFrame, summary: Dict[str, Any], idx: int) -> None:
        cycles = frame["cycle"].to_numpy(dtype=np.int32)
        capacity = frame["capacity_mAh_g"].to_numpy(dtype=np.float32)
        resistance = frame["resistance_ohm"].to_numpy(dtype=np.float32)
        temp = frame["temperature_K"].to_numpy(dtype=np.float32)
        x = np.linspace(0.05, 0.95, 100, dtype=np.float32)
        base_curve = 2.55 + 1.25 * x - 0.18 * x**2 + 0.08 * comp["Mn"] - 0.04 * comp["Fe"]
        fade_shift = (1.0 - capacity / capacity[0]).astype(np.float32)
        voltage_curves = base_curve[None, :] - 0.12 * fade_shift[:, None] - resistance[:, None] * frame["c_rate"].iloc[0]
        voltage_curves += self.rng.normal(0, 0.006, voltage_curves.shape).astype(np.float32)
        np.savez_compressed(
            self.compat_dir / f"cathode_hyper_{idx:04d}.npz",
            cycles=cycles,
            capacity=capacity,
            voltage_curves=voltage_curves.astype(np.float32),
            resistance=resistance,
            sei_thickness=(1e-9 + 2e-11 * np.sqrt(cycles + 1)).astype(np.float32),
            temperature=temp,
            Na=float(comp["Na"]),
            Mn=float(comp["Mn"]),
            Fe=float(comp["Fe"]),
            dopant=str(comp["dopant"]),
            dopant_frac=float(comp["dopant_frac"]),
            q0=float(summary["q0_mAh_g"]),
            fade_fraction=float(summary["fade_fraction"]),
            schema_version=SCHEMA_VERSION,
        )

    def write(self, compositions: pd.DataFrame, manifest: DatasetManifest) -> Dict[str, Any]:
        cycles, summary, _ = self.simulate(compositions)
        comp_path = self.out_dir / "cathode_compositions.parquet"
        cycles_path = self.out_dir / f"cathode_cycles_{self.profile.name}.parquet"
        summary_path = self.out_dir / f"cathode_summary_{self.profile.name}.parquet"
        compositions.to_parquet(comp_path, index=False)
        cycles.to_parquet(cycles_path, index=False)
        summary.to_parquet(summary_path, index=False)
        register_file(manifest, comp_path, "cathode", len(compositions), SCHEMA_VERSION)
        register_file(manifest, cycles_path, "cathode", len(cycles), SCHEMA_VERSION)
        register_file(manifest, summary_path, "cathode", len(summary), SCHEMA_VERSION)
        return {
            "cathode_cycle_rows": int(len(cycles)),
            "cathode_trajectories": int(len(summary)),
            "cathode_compositions": int(len(compositions)),
            "cathode_capacity": describe_array(cycles["capacity_mAh_g"].to_numpy()),
            "cathode_fade": describe_array(summary["fade_fraction"].to_numpy()),
        }


class BMSHyperGenerator:
    def __init__(self, rng: np.random.Generator, root: Path, profile: PipelineProfile):
        self.rng = rng
        self.root = root
        self.profile = profile
        self.out_dir = ensure_dir(root / "synthetic" / "hyper")
        self.compat_dir = ensure_dir(root / "synthetic" / "bms")
        self.edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [4, 5],
                [5, 6],
                [6, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            dtype=np.int32,
        )

    def drive_profile(self, n: int, scenario: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n, dtype=np.float32)
        day_frac = t / max(n - 1, 1)
        current = np.zeros(n, dtype=np.float32)
        ambient = 303.15 + 13.0 * np.sin(np.pi * np.minimum(day_frac / 0.65, 1.0))
        city = day_frac < 0.32
        highway = (day_frac >= 0.32) & (day_frac < 0.48)
        charge = (day_frac >= 0.48) & (day_frac < 0.72)
        evening = (day_frac >= 0.72) & (day_frac < 0.88)
        overnight = day_frac >= 0.88
        current[city] = 3.2 * np.sin(2 * np.pi * t[city] / 37.0) + 1.4 * np.sign(np.sin(2 * np.pi * t[city] / 83.0))
        current[highway] = 1.2 + 0.15 * np.sin(2 * np.pi * t[highway] / 240.0)
        current[charge] = -0.65 + 0.08 * np.sin(2 * np.pi * t[charge] / 600.0)
        current[evening] = 3.0 * np.sin(2 * np.pi * t[evening] / 41.0) + 1.2 * np.sign(np.sin(2 * np.pi * t[evening] / 91.0))
        overnight_phase = np.clip((day_frac[overnight] - 0.88) / 0.12, 0, 1)
        current[overnight] = -0.30 * np.exp(-3.0 * overnight_phase)
        current += self.rng.normal(0, 0.05, n).astype(np.float32)
        ambient += self.rng.normal(0, 0.45, n).astype(np.float32)
        current *= np.float32(0.9 + 0.25 * ((scenario % 5) / 4.0))
        return current, ambient.astype(np.float32)

    def scenario_failure(self, idx: int) -> Tuple[str, int, float]:
        types = ("none", "sei", "dendrite", "thermal_cascade", "cooling_loss", "sensor_bias")
        failure_type = types[idx % len(types)]
        if idx < max(4, self.profile.bms_scenarios // 3):
            failure_type = types[1 + (idx % (len(types) - 1))]
        fail_cell = int((idx * 3 + 1) % 8)
        onset = float(0.42 + 0.18 * ((idx % 7) / 6.0))
        return failure_type, fail_cell, onset

    def simulate_one(self, idx: int) -> Dict[str, Any]:
        n = self.profile.bms_seconds
        current, ambient = self.drive_profile(n, idx)
        v = np.zeros((n, 8), dtype=np.float32)
        temp = np.zeros((n, 8), dtype=np.float32)
        soc = np.zeros((n, 8), dtype=np.float32)
        resistance = np.zeros((n, 8), dtype=np.float32)
        sei = np.zeros((n, 8), dtype=np.float32)
        plating = np.zeros((n, 8), dtype=np.float32)
        risk = np.zeros((n, 8), dtype=np.float32)
        cell_capacity_ah = (4.6 + self.rng.normal(0, 0.08, 8)).astype(np.float32)
        soc_state = np.clip(0.78 + self.rng.normal(0, 0.018, 8), 0.55, 0.92).astype(np.float32)
        r_state = np.clip(0.013 + self.rng.normal(0, 0.0015, 8), 0.008, 0.025).astype(np.float32)
        sei_state = np.full(8, 1.0e-9, dtype=np.float32)
        plate_state = np.zeros(8, dtype=np.float32)
        temp_state = ambient[0] + self.rng.normal(0, 0.35, 8).astype(np.float32)
        failure_type, fail_cell, onset = self.scenario_failure(idx)
        onset_step = int(onset * n)
        adjacency_gain = np.zeros(8, dtype=np.float32)
        for a, b in self.edges:
            if a == fail_cell:
                adjacency_gain[b] += 1.0
            if b == fail_cell:
                adjacency_gain[a] += 1.0
        for step in range(n):
            i_cell = current[step] / 2.0
            dt_hr = 1.0 / 3600.0
            soc_state = np.clip(soc_state - i_cell * dt_hr / np.maximum(cell_capacity_ah, 0.1), 0.02, 0.98)
            heat = (i_cell**2) * r_state * 42.0
            cooling = 0.018 * (temp_state - ambient[step])
            thermal_coupling = np.zeros(8, dtype=np.float32)
            for a, b in self.edges:
                flux = 0.006 * (temp_state[b] - temp_state[a])
                thermal_coupling[a] += flux
                thermal_coupling[b] -= flux
            temp_state = temp_state + heat - cooling + thermal_coupling + self.rng.normal(0, 0.025, 8).astype(np.float32)
            temp_state = np.nan_to_num(temp_state, nan=620.0, posinf=620.0, neginf=260.0)
            temp_state = np.clip(temp_state, 260.0, 620.0).astype(np.float32)
            sei_exponent = np.clip((temp_state - 298.15) / 36.0, -8.0, 6.0)
            sei_rate = 2.5e-13 * np.exp(sei_exponent) * (1.0 + 0.35 * abs(i_cell))
            sei_state = sei_state + sei_rate / np.maximum(2 * sei_state, 1e-12)
            sei_state = np.clip(np.nan_to_num(sei_state, nan=5.0e-6, posinf=5.0e-6, neginf=1.0e-10), 1.0e-10, 5.0e-6)
            plating_drive = max(0.0, -i_cell - 0.55) * np.exp(-(temp_state - 285.0) / 45.0)
            plate_state = plate_state + (1.5e-5 * plating_drive).astype(np.float32)
            if step > onset_step and failure_type != "none":
                progress = (step - onset_step) / max(n - onset_step, 1)
                if failure_type == "sei":
                    sei_state[fail_cell] *= 1.0 + 0.00035 + 0.00075 * progress
                elif failure_type == "dendrite":
                    plate_state[fail_cell] += 0.00018 * np.exp(np.clip(4.0 * progress, 0.0, 4.0))
                elif failure_type == "thermal_cascade":
                    temp_state[fail_cell] += 0.018 + 0.25 * progress
                    temp_state += adjacency_gain * (0.004 + 0.070 * progress)
                elif failure_type == "cooling_loss":
                    temp_state += 0.010 + 0.035 * progress
                    temp_state[fail_cell] += 0.055 * progress
                elif failure_type == "sensor_bias":
                    r_state[fail_cell] += 0.000018 * (1.0 + 8.0 * progress)
            plate_state = np.clip(np.nan_to_num(plate_state, nan=0.0, posinf=3.0, neginf=0.0), 0.0, 3.0)
            r_state = 0.012 + 0.0025 * (1.0 - soc_state) + 115.0 * sei_state + 0.045 * plate_state
            r_state = np.clip(np.nan_to_num(r_state, nan=0.35, posinf=0.55, neginf=0.004), 0.004, 0.55).astype(np.float32)
            ocv = 2.85 + 1.32 * soc_state - 0.22 * soc_state**2
            voltage = ocv - i_cell * r_state + self.rng.normal(0, 0.0035, 8).astype(np.float32)
            delta_t = temp_state - ambient[step]
            delta_r = r_state - 0.012
            dtemp = temp_state - (temp[step - 1] if step > 0 else temp_state)
            logit = 19.0 * delta_r + 0.050 * delta_t + 9.0 * plate_state + 0.055 * np.maximum(dtemp, 0)
            logit -= 2.15
            risk_state = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
            v[step] = voltage
            temp[step] = temp_state
            soc[step] = soc_state
            resistance[step] = r_state
            sei[step] = sei_state
            plating[step] = plate_state
            risk[step] = risk_state.astype(np.float32)
        failure_step = int(np.argmax(np.max(risk, axis=1) > 0.92)) if np.any(np.max(risk, axis=1) > 0.92) else -1
        return {
            "scenario_id": idx,
            "time_s": np.arange(n, dtype=np.float32),
            "current_A": current,
            "ambient_K": ambient,
            "V": v,
            "T": temp,
            "SOC": soc,
            "R_int": resistance,
            "L_sei": sei,
            "P_plating": plating,
            "risk": risk,
            "failure_type": failure_type,
            "fail_cell": fail_cell,
            "onset_step": onset_step,
            "failure_step": failure_step,
        }

    def write_compatibility(self, data: Dict[str, Any], idx: int) -> None:
        np.savez_compressed(
            self.compat_dir / f"bms_hyper_{idx:04d}.npz",
            V=data["V"],
            T=data["T"],
            risk=data["risk"],
            I=data["current_A"],
            T_amb=data["ambient_K"],
            SOC=data["SOC"],
            R_int=data["R_int"],
            L_sei=data["L_sei"],
            P_plating=data["P_plating"],
            failure_type=data["failure_type"],
            fail_cell=int(data["fail_cell"]),
            onset_step=int(data["onset_step"]),
            failure_step=int(data["failure_step"]),
            schema_version=SCHEMA_VERSION,
        )

    def write(self, manifest: DatasetManifest) -> Dict[str, Any]:
        scenarios = [self.simulate_one(i) for i in range(self.profile.bms_scenarios)]
        for i, scenario in enumerate(scenarios[: self.profile.compatibility_files]):
            self.write_compatibility(scenario, i)
        v = np.stack([s["V"] for s in scenarios])
        temp = np.stack([s["T"] for s in scenarios])
        soc = np.stack([s["SOC"] for s in scenarios])
        r_int = np.stack([s["R_int"] for s in scenarios])
        risk = np.stack([s["risk"] for s in scenarios])
        current = np.stack([s["current_A"] for s in scenarios])
        ambient = np.stack([s["ambient_K"] for s in scenarios])
        path = self.out_dir / f"bms_pack_timeseries_{self.profile.name}.npz"
        np.savez_compressed(
            path,
            V=v.astype(np.float32),
            T=temp.astype(np.float32),
            SOC=soc.astype(np.float32),
            R_int=r_int.astype(np.float32),
            risk=risk.astype(np.float32),
            I=current.astype(np.float32),
            T_amb=ambient.astype(np.float32),
            edges=self.edges,
            schema_version=SCHEMA_VERSION,
        )
        metadata = pd.DataFrame(
            [
                {
                    "scenario_id": s["scenario_id"],
                    "failure_type": s["failure_type"],
                    "fail_cell": s["fail_cell"],
                    "onset_step": s["onset_step"],
                    "failure_step": s["failure_step"],
                    "max_risk": float(np.max(s["risk"])),
                    "max_temperature_K": float(np.max(s["T"])),
                    "min_voltage_V": float(np.min(s["V"])),
                    "max_resistance_ohm": float(np.max(s["R_int"])),
                }
                for s in scenarios
            ]
        )
        meta_path = self.out_dir / f"bms_scenario_metadata_{self.profile.name}.parquet"
        metadata.to_parquet(meta_path, index=False)
        rows = int(self.profile.bms_scenarios * self.profile.bms_seconds * 8)
        register_file(manifest, path, "bms", rows, SCHEMA_VERSION)
        register_file(manifest, meta_path, "bms", len(metadata), SCHEMA_VERSION)
        return {
            "bms_cell_timestep_rows": rows,
            "bms_scenarios": int(len(scenarios)),
            "bms_failures": int(np.sum(metadata["failure_type"] != "none")),
            "bms_risk": describe_array(risk),
            "bms_temperature_K": describe_array(temp),
        }


class LeachingHyperGenerator:
    def __init__(self, rng: np.random.Generator, root: Path, profile: PipelineProfile, calibration_priors: Dict[str, Any] | None = None):
        self.rng = rng
        self.root = root
        self.profile = profile
        self.calibration_priors = calibration_priors or {}
        self.out_dir = ensure_dir(root / "synthetic" / "hyper")
        self.compat_dir = ensure_dir(root / "synthetic" / "leaching")

    def condition_grid(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        idx = 0
        for temp_k in self.profile.leaching_temperatures:
            for ph in self.profile.leaching_ph:
                for conc in self.profile.leaching_conc:
                    for radius_um in self.profile.leaching_particles_um:
                        rows.append(
                            {
                                "condition_id": idx,
                                "T": float(temp_k),
                                "pH": float(ph),
                                "c_acid": float(conc),
                                "r0": float(radius_um * 1e-6),
                                "r0_um": float(radius_um),
                            }
                        )
                        idx += 1
        return pd.DataFrame(rows)

    def simulate_condition(self, cond: pd.Series) -> np.ndarray:
        minutes = 181
        t = np.arange(minutes, dtype=np.float32)
        alpha = np.zeros((3, minutes), dtype=np.float32)
        state = np.zeros(3, dtype=np.float64)
        d0 = np.array([6.5e-10, 4.7e-10, 9.0e-10], dtype=np.float64)
        ea = np.array([0.31, 0.35, 0.27], dtype=np.float64)
        avrami_base = np.array([0.012, 0.008, 0.018], dtype=np.float64)
        n_avrami = np.array([2.15, 1.78, 2.45], dtype=np.float64)
        recovery_center = float(self.calibration_priors.get("recovery_center_percent", 73.0))
        recovery_scale = float(np.clip(recovery_center / 73.0, 0.78, 1.22))
        pH_factor = recovery_scale * np.exp(-0.62 * (cond["pH"] - 0.5))
        acid_factor = np.log1p(cond["c_acid"]) / np.log(4.0)
        radius = cond["r0"]
        gamma = 1.0 / (1.0 + np.exp(-(2.0 * (cond["T"] - 323.15) / 40.0 - 0.9 * (cond["pH"] - 1.5) + 0.35 * np.log10(cond["r0_um"] / 50.0))))
        for step in range(minutes):
            time_min = max(float(step), 1e-3)
            for species in range(3):
                d_eff = d0[species] * np.exp(-(ea[species] * 96485.33212) / (8.314462618 * cond["T"]))
                sc = (3.0 * d_eff * cond["c_acid"] * 1000.0 * pH_factor) / (radius**2 * 5100.0 * max((1.0 - state[species]) ** (1.0 / 3.0), 1e-5))
                k_a = avrami_base[species] * acid_factor * np.exp((cond["T"] - 323.15) / 48.0) * pH_factor
                av = k_a * n_avrami[species] * (time_min / 180.0) ** (n_avrami[species] - 1.0)
                av *= np.exp(-k_a * (time_min / 180.0) ** n_avrami[species])
                da = 60.0 * (gamma * sc + (1.0 - gamma) * av / 180.0)
                state[species] = min(1.0, state[species] + da)
            noise = self.rng.normal(0.0, 0.008 + 0.004 * (1.0 - state), 3)
            alpha[:, step] = np.clip(state + noise, 0.0, 1.0)
        alpha = np.maximum.accumulate(alpha, axis=1)
        return alpha.astype(np.float32)

    def write(self, manifest: DatasetManifest) -> Dict[str, Any]:
        conditions = self.condition_grid()
        trajectories = np.stack([self.simulate_condition(row) for _, row in conditions.iterrows()])
        compat_conditions = [
            {"T": float(r.T), "pH": float(r.pH), "c_acid": float(r.c_acid), "r0": float(r.r0)}
            for r in conditions.itertuples()
        ]
        ensure_dir(self.compat_dir)
        compat_path = self.compat_dir / "leaching_grid.npz"
        np.savez_compressed(
            compat_path,
            conditions=np.array(compat_conditions, dtype=object),
            alpha_trajectories=trajectories.astype(np.float32),
            trajectories=trajectories.astype(np.float32),
            time_minutes=np.arange(trajectories.shape[2], dtype=np.float32),
            schema_version=SCHEMA_VERSION,
        )
        final = conditions.copy()
        final["alpha_Mn"] = trajectories[:, 0, -1]
        final["alpha_Fe"] = trajectories[:, 1, -1]
        final["alpha_Na"] = trajectories[:, 2, -1]
        final["weighted_recovery"] = 0.5 * final["alpha_Mn"] + 0.3 * final["alpha_Fe"] + 0.2 * final["alpha_Na"]
        cond_path = self.out_dir / f"leaching_conditions_{self.profile.name}.parquet"
        final_path = self.out_dir / f"leaching_final_{self.profile.name}.parquet"
        traj_path = self.out_dir / f"leaching_trajectories_{self.profile.name}.npz"
        final.to_parquet(final_path, index=False)
        conditions.to_parquet(cond_path, index=False)
        np.savez_compressed(
            traj_path,
            alpha_trajectories=trajectories.astype(np.float32),
            trajectories=trajectories.astype(np.float32),
            condition_id=conditions["condition_id"].to_numpy(dtype=np.int32),
            time_minutes=np.arange(trajectories.shape[2], dtype=np.float32),
            species=np.array(["Mn", "Fe", "Na"], dtype=object),
            schema_version=SCHEMA_VERSION,
        )
        rows = int(trajectories.shape[0] * trajectories.shape[1] * trajectories.shape[2])
        register_file(manifest, compat_path, "recycling", rows, SCHEMA_VERSION)
        register_file(manifest, cond_path, "recycling", len(conditions), SCHEMA_VERSION)
        register_file(manifest, final_path, "recycling", len(final), SCHEMA_VERSION)
        register_file(manifest, traj_path, "recycling", rows, SCHEMA_VERSION)
        return {
            "leaching_species_timestep_rows": rows,
            "leaching_conditions": int(len(conditions)),
            "leaching_weighted_recovery": describe_array(final["weighted_recovery"].to_numpy()),
            "leaching_alpha_Mn": describe_array(final["alpha_Mn"].to_numpy()),
        }


class HyperDatasetValidator:
    def __init__(self, root: Path, profile: PipelineProfile):
        self.root = root
        self.profile = profile
        self.issues: List[ValidationIssue] = []
        self.metrics: Dict[str, Any] = {}

    def validate_cathode(self) -> None:
        path = self.root / "synthetic" / "hyper" / f"cathode_cycles_{self.profile.name}.parquet"
        summary_path = self.root / "synthetic" / "hyper" / f"cathode_summary_{self.profile.name}.parquet"
        if not path.exists():
            self.issues.append(ValidationIssue("error", "cathode", "file", "missing cathode cycle parquet", str(path)))
            return
        df = pd.read_parquet(path)
        summary = pd.read_parquet(summary_path)
        validate_range(self.issues, "cathode", "capacity_mAh_g", df["capacity_mAh_g"].to_numpy(), 40.0, 240.0)
        validate_finite(self.issues, "cathode", "capacity_mAh_g", df["capacity_mAh_g"].to_numpy())
        validate_range(self.issues, "cathode", "voltage_mean_V", df["voltage_mean_V"].to_numpy(), 1.7, 4.4)
        validate_finite(self.issues, "cathode", "voltage_mean_V", df["voltage_mean_V"].to_numpy())
        validate_range(self.issues, "cathode", "resistance_ohm", df["resistance_ohm"].to_numpy(), 0.001, 0.35)
        validate_finite(self.issues, "cathode", "resistance_ohm", df["resistance_ohm"].to_numpy())
        fade = summary["fade_fraction"].to_numpy()
        validate_range(self.issues, "cathode", "fade_fraction", fade, -0.05, 0.70, "warning")
        sample_ids = summary["trajectory_id"].head(min(48, len(summary))).to_numpy()
        mono = []
        for tid in sample_ids:
            y = df.loc[df["trajectory_id"] == tid, "clean_capacity_mAh_g"].to_numpy()
            mono.append(monotonicity_fraction(y))
        self.metrics["cathode_rows"] = int(len(df))
        self.metrics["cathode_trajectories"] = int(len(summary))
        self.metrics["cathode_clean_monotonicity_sample_mean"] = float(np.mean(mono)) if mono else 0.0
        self.metrics["cathode_fade"] = describe_array(fade)
        if len(df) < 10000:
            self.issues.append(ValidationIssue("error", "cathode", "rows", "cathode rows below required tens-of-thousands floor", len(df), ">=10000"))

    def validate_bms(self) -> None:
        path = self.root / "synthetic" / "hyper" / f"bms_pack_timeseries_{self.profile.name}.npz"
        if not path.exists():
            self.issues.append(ValidationIssue("error", "bms", "file", "missing BMS npz", str(path)))
            return
        data = np.load(path, allow_pickle=True)
        v = data["V"]
        temp = data["T"]
        risk = data["risk"]
        r_int = data["R_int"]
        validate_finite(self.issues, "bms", "V", v)
        validate_finite(self.issues, "bms", "T", temp)
        validate_finite(self.issues, "bms", "risk", risk)
        validate_finite(self.issues, "bms", "R_int", r_int)
        validate_range(self.issues, "bms", "V", v, 1.8, 4.5)
        validate_range(self.issues, "bms", "T", temp, 260.0, 650.0)
        if float(np.percentile(temp, 95)) > 500.0:
            self.issues.append(
                ValidationIssue(
                    "warning",
                    "bms",
                    "T",
                    "95th percentile pack temperature is in severe runaway regime",
                    {"p95": float(np.percentile(temp, 95)), "max": float(np.max(temp))},
                    {"p95": "<=500 K for ordinary abuse runs; higher allowed for cascade failure shards"},
                )
            )
        validate_range(self.issues, "bms", "risk", risk, 0.0, 1.0)
        validate_range(self.issues, "bms", "R_int", r_int, 0.001, 0.7)
        rows = int(v.shape[0] * v.shape[1] * v.shape[2])
        self.metrics["bms_rows"] = rows
        self.metrics["bms_scenarios"] = int(v.shape[0])
        self.metrics["bms_max_risk"] = float(np.max(risk))
        self.metrics["bms_temperature"] = describe_array(temp)
        if rows < 10000:
            self.issues.append(ValidationIssue("error", "bms", "rows", "BMS rows below required tens-of-thousands floor", rows, ">=10000"))
        data.close()

    def validate_leaching(self) -> None:
        path = self.root / "synthetic" / "hyper" / f"leaching_trajectories_{self.profile.name}.npz"
        final_path = self.root / "synthetic" / "hyper" / f"leaching_final_{self.profile.name}.parquet"
        if not path.exists():
            self.issues.append(ValidationIssue("error", "recycling", "file", "missing leaching trajectory npz", str(path)))
            return
        data = np.load(path, allow_pickle=True)
        alpha = data["alpha_trajectories"]
        validate_finite(self.issues, "recycling", "alpha_trajectories", alpha)
        validate_range(self.issues, "recycling", "alpha_trajectories", alpha, 0.0, 1.0)
        final = pd.read_parquet(final_path)
        rows = int(np.prod(alpha.shape))
        self.metrics["leaching_rows"] = rows
        self.metrics["leaching_conditions"] = int(alpha.shape[0])
        self.metrics["leaching_weighted_recovery"] = describe_array(final["weighted_recovery"].to_numpy())
        if rows < 10000:
            self.issues.append(ValidationIssue("error", "recycling", "rows", "leaching rows below required tens-of-thousands floor", rows, ">=10000"))
        data.close()

    def run(self) -> Dict[str, Any]:
        self.validate_cathode()
        self.validate_bms()
        self.validate_leaching()
        self.metrics["total_rows"] = int(self.metrics.get("cathode_rows", 0) + self.metrics.get("bms_rows", 0) + self.metrics.get("leaching_rows", 0))
        return {"issues": self.issues, "metrics": self.metrics}


class HyperDataPipeline:
    def __init__(self, root: Path, profile: PipelineProfile, seed: int):
        self.root = root
        self.profile = profile
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        ensure_dir(root / "cache")
        ensure_dir(root / "synthetic" / "hyper")
        self.calibration_priors = self.load_calibration_priors()
        self.real_manifest = self.load_real_manifest()

    def load_calibration_priors(self) -> Dict[str, Any]:
        path = self.root / "real" / "scraped" / "calibration_priors.json"
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                priors = json.load(f)
            if isinstance(priors, dict):
                return priors
            return {}
        except Exception:
            return {}

    def load_real_manifest(self) -> Dict[str, Any]:
        path = self.root / "real" / "assembled" / "real_dataset_manifest.json"
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def run(self) -> DatasetManifest:
        manifest = DatasetManifest(
            project="KineticsForge",
            schema_version=SCHEMA_VERSION,
            created_at=utc_now(),
            profile=self.profile.name,
            seed=self.seed,
            root=str(self.root),
            sources=source_catalog(),
        )
        write_source_catalog(self.root / "real" / "source_catalog.json")
        compositions = CompositionLibrary(self.rng).build(self.profile.cathode_compositions)
        cathode_metrics = CathodeHyperGenerator(self.rng, self.root, self.profile, self.calibration_priors).write(compositions, manifest)
        bms_metrics = BMSHyperGenerator(self.rng, self.root, self.profile).write(manifest)
        leaching_metrics = LeachingHyperGenerator(self.rng, self.root, self.profile, self.calibration_priors).write(manifest)
        manifest.metrics.update(cathode_metrics)
        manifest.metrics.update(bms_metrics)
        manifest.metrics.update(leaching_metrics)
        manifest.metrics["literature_calibration"] = {
            "loaded": bool(self.calibration_priors),
            "accepted_rows": int(self.calibration_priors.get("accepted_rows", 0)) if self.calibration_priors else 0,
            "source": self.calibration_priors.get("source", "") if self.calibration_priors else "",
            "q0_center_mAh_g": self.calibration_priors.get("q0_center_mAh_g") if self.calibration_priors else None,
            "retention_center_percent": self.calibration_priors.get("retention_center_percent") if self.calibration_priors else None,
        }
        manifest.metrics["real_dataset"] = {
            "loaded": bool(self.real_manifest),
            "status": self.real_manifest.get("status", "") if self.real_manifest else "",
            "total_real_rows": self.real_manifest.get("metrics", {}).get("total_real_rows", 0) if self.real_manifest else 0,
            "cycle_rows": self.real_manifest.get("metrics", {}).get("cycle_rows", 0) if self.real_manifest else 0,
            "timeseries_rows": self.real_manifest.get("metrics", {}).get("timeseries_rows", 0) if self.real_manifest else 0,
            "literature_rows_accepted": self.real_manifest.get("metrics", {}).get("literature_rows_accepted", 0) if self.real_manifest else 0,
            "manifest": str(self.root / "real" / "assembled" / "real_dataset_manifest.json") if self.real_manifest else "",
        }
        validator = HyperDatasetValidator(self.root, self.profile)
        result = validator.run()
        manifest.metrics.update(result["metrics"])
        manifest_path = self.root / "cache" / f"hyper_manifest_{self.profile.name}.json"
        validation_path = self.root / "cache" / f"data_quality_report_{self.profile.name}.json"
        write_json(manifest_path, manifest.to_json())
        write_validation_report(validation_path, result["issues"], result["metrics"])
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-volume KineticsForge data with provenance and validation.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="foundation")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--seed", type=int, default=20260430)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    root = Path(args.root).resolve()
    manifest = HyperDataPipeline(root=root, profile=profile, seed=args.seed).run()
    print(json.dumps({"profile": profile.name, "total_rows": manifest.to_json()["total_rows"], "manifest": str(root / "cache" / f"hyper_manifest_{profile.name}.json")}, indent=2))


if __name__ == "__main__":
    main()
