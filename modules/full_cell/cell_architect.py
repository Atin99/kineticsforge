import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.india_context import IndiaOperatingContext
from modules.cathode.defect_chemistry import DefectChemistryModel, DefectChemistryResult
from modules.cathode.screener import screen_compositions
from modules.cathode.uncertainty_quantification import UncertaintyPropagation


@dataclass(frozen=True)
class AnodeOption:
    name: str
    chemistry: str
    specific_capacity_mAh_g: float
    average_potential_V: float
    first_cycle_efficiency: float
    volume_change_fraction: float
    relative_cost_index: float
    sodium_compatible: bool
    lithium_compatible: bool
    safety_score: float
    notes: str


@dataclass(frozen=True)
class AdditiveRule:
    name: str
    dose_wt_percent: float
    mechanism: str
    trigger: str
    penalty: str


@dataclass
class FullCellArchitecture:
    cathode: Dict[str, Any]
    anode: AnodeOption
    np_ratio: float
    cathode_active_mass_g_per_Ah: float
    anode_active_mass_g_per_Ah: float
    active_material_mass_g_per_Ah: float
    nominal_voltage_V: float
    active_specific_energy_Wh_kg: float
    first_cycle_loss_mAh_per_Ah: float
    electrolyte_additives: List[AdditiveRule]
    defect_summary: Dict[str, float]
    score: float
    tradeoffs: Dict[str, float]
    uncertainty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    validation_gates: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


ANODES: Sequence[AnodeOption] = (
    AnodeOption(
        name="hard_carbon",
        chemistry="Na-ion",
        specific_capacity_mAh_g=300.0,
        average_potential_V=0.12,
        first_cycle_efficiency=0.88,
        volume_change_fraction=0.09,
        relative_cost_index=0.52,
        sodium_compatible=True,
        lithium_compatible=True,
        safety_score=0.78,
        notes="Default practical Na-ion anode; needs ICE compensation or pre-sodiation for high-energy cells.",
    ),
    AnodeOption(
        name="soft_carbon",
        chemistry="Na-ion",
        specific_capacity_mAh_g=240.0,
        average_potential_V=0.18,
        first_cycle_efficiency=0.91,
        volume_change_fraction=0.06,
        relative_cost_index=0.48,
        sodium_compatible=True,
        lithium_compatible=True,
        safety_score=0.82,
        notes="Lower capacity than hard carbon but lower irreversible loss and gentler swelling.",
    ),
    AnodeOption(
        name="graphite",
        chemistry="Li-ion",
        specific_capacity_mAh_g=345.0,
        average_potential_V=0.09,
        first_cycle_efficiency=0.93,
        volume_change_fraction=0.10,
        relative_cost_index=0.60,
        sodium_compatible=False,
        lithium_compatible=True,
        safety_score=0.70,
        notes="Do not pair with ordinary Na-ion electrolyte unless co-intercalation chemistry is explicitly validated.",
    ),
    AnodeOption(
        name="lto",
        chemistry="Li-ion/safety",
        specific_capacity_mAh_g=165.0,
        average_potential_V=1.55,
        first_cycle_efficiency=0.96,
        volume_change_fraction=0.01,
        relative_cost_index=1.45,
        sodium_compatible=False,
        lithium_compatible=True,
        safety_score=0.97,
        notes="Very safe high-power anode, but voltage penalty is too large for most Na-ion cathode startup targets.",
    ),
)


class FullCellArchitect:
    def __init__(self, chemistry: str = "Na-ion"):
        self.chemistry = chemistry
        self.defects = DefectChemistryModel()
        self.india = IndiaOperatingContext.from_env()

    def design(self, cathode: Dict[str, Any], target_Ah: float = 1.0) -> FullCellArchitecture:
        defect = self.defects.evaluate(cathode.get("comp", cathode))
        options = [self.evaluate_pairing(cathode, defect, anode, target_Ah) for anode in ANODES]
        feasible = [o for o in options if not any("incompatible" in w.lower() for w in o.warnings)]
        candidates = feasible or options
        return max(candidates, key=lambda item: item.score)

    def evaluate_pairing(
        self,
        cathode: Dict[str, Any],
        defect: DefectChemistryResult,
        anode: AnodeOption,
        target_Ah: float,
    ) -> FullCellArchitecture:
        comp = cathode.get("comp", cathode)
        q_cathode = float(cathode.get("Q0", cathode.get("q0_mAh_g", 145.0)))
        cathode_voltage = float(cathode.get("avg_voltage", 3.25))
        cathode_voltage = float(np.clip(cathode_voltage, 2.2, 4.4))
        np_ratio = self.np_ratio(defect, anode)
        cathode_mass = 1000.0 * target_Ah / max(q_cathode, 1e-9)
        anode_required_mAh = 1000.0 * target_Ah * np_ratio
        anode_mass = anode_required_mAh / max(anode.specific_capacity_mAh_g * anode.first_cycle_efficiency, 1e-9)
        nominal_voltage = cathode_voltage - anode.average_potential_V
        active_mass = cathode_mass + anode_mass
        specific_energy = 1000.0 * target_Ah * nominal_voltage / max(active_mass, 1e-9)
        first_loss = 1000.0 * target_Ah * (1.0 / max(anode.first_cycle_efficiency, 1e-9) - 1.0)
        additives = self.additives(defect, comp, anode)
        warnings = self.warnings(comp, anode, nominal_voltage, defect)
        tradeoffs = self.tradeoffs(q_cathode, anode, np_ratio, specific_energy, defect, additives)
        uncertainty = self.propagate_uncertainty(cathode, q_cathode, np_ratio, specific_energy, nominal_voltage, defect, additives)
        score = self.score(tradeoffs, warnings) - min(0.08, 0.02 * uncertainty["active_specific_energy_Wh_kg"]["variance"] ** 0.5 / 20.0)
        gates = self.validation_gates(comp, anode, defect, additives)
        return FullCellArchitecture(
            cathode=comp,
            anode=anode,
            np_ratio=float(np_ratio),
            cathode_active_mass_g_per_Ah=float(cathode_mass),
            anode_active_mass_g_per_Ah=float(anode_mass),
            active_material_mass_g_per_Ah=float(active_mass),
            nominal_voltage_V=float(nominal_voltage),
            active_specific_energy_Wh_kg=float(specific_energy),
            first_cycle_loss_mAh_per_Ah=float(first_loss),
            electrolyte_additives=additives,
            defect_summary={
                "charge_balance_error": float(defect.charge_balance_error),
                "oxygen_redox_risk": float(defect.oxygen_redox_risk),
                "transition_metal_mixing_risk": float(defect.transition_metal_mixing_risk),
                "moisture_sensitivity": float(defect.moisture_sensitivity),
                "jahn_teller_risk": float(defect.jahn_teller_risk),
                "defect_tolerance_score": float(defect.defect_tolerance_score),
            },
            score=float(score),
            tradeoffs=tradeoffs,
            uncertainty=uncertainty,
            validation_gates=gates,
            warnings=warnings,
        )

    def np_ratio(self, defect: DefectChemistryResult, anode: AnodeOption) -> float:
        ratio = 1.06
        ratio += 0.08 * max(0.0, 0.90 - anode.first_cycle_efficiency)
        ratio += 0.04 * defect.moisture_sensitivity
        ratio += 0.03 * defect.transition_metal_mixing_risk
        ratio += 0.03 * defect.oxygen_redox_risk
        if anode.name == "hard_carbon":
            ratio += 0.02
        if anode.name == "lto":
            ratio -= 0.02
        return float(np.clip(ratio, 1.03, 1.24))

    def additives(self, defect: DefectChemistryResult, comp: Dict[str, Any], anode: AnodeOption) -> List[AdditiveRule]:
        out: List[AdditiveRule] = []
        if self.chemistry.lower().startswith("na"):
            if anode.name in {"hard_carbon", "soft_carbon"}:
                out.append(AdditiveRule("FEC", 3.0, "SEI-forming carbonate additive for carbon anode passivation.", "Na-ion carbon anode baseline", "Can increase gas if overdosed or moisture is high."))
            if defect.moisture_sensitivity > 0.55 or defect.transition_metal_mixing_risk > 0.45:
                out.append(AdditiveRule("NaDFOB", 1.5, "Boron/oxalate salt additive to strengthen CEI and bind trace moisture.", "Moisture or transition-metal crossover risk", "Must verify Al-current-collector compatibility."))
            if defect.oxygen_redox_risk > 0.52:
                out.append(AdditiveRule("TMSP", 0.8, "Phosphite scavenger for high-voltage oxygen-redox side reactions.", "High oxygen-redox risk", "May raise impedance after formation."))
        else:
            out.append(AdditiveRule("VC", 1.5, "Graphite SEI stabilization.", "Li-ion graphite or mixed Li-ion baseline", "Can polymerize and increase impedance if abused."))
            if defect.oxygen_redox_risk > 0.52:
                out.append(AdditiveRule("LiDFOB", 1.0, "High-voltage CEI support and metal dissolution suppression.", "High-voltage cathode risk", "Requires electrolyte compatibility screen."))
        seen = set()
        deduped = []
        for item in out:
            if item.name not in seen:
                seen.add(item.name)
                deduped.append(item)
        return deduped

    def warnings(self, comp: Dict[str, Any], anode: AnodeOption, nominal_voltage: float, defect: DefectChemistryResult) -> List[str]:
        warnings: List[str] = []
        if self.chemistry.lower().startswith("na") and not anode.sodium_compatible:
            warnings.append(f"{anode.name} is incompatible with ordinary Na-ion full-cell chemistry without a separately validated co-intercalation mechanism.")
        if nominal_voltage < 1.7:
            warnings.append("Nominal full-cell voltage is too low for the target product class.")
        if defect.moisture_sensitivity > 0.70:
            warnings.append("Moisture sensitivity is high; dry-room handling and Karl Fischer limits are mandatory.")
        if abs(defect.charge_balance_error) > 0.35:
            warnings.append("Charge-balance error is large; verify stoichiometry before full-cell claims.")
        if comp.get("dopant") in {"Ni", "Cu"} and self.chemistry.lower().startswith("na"):
            warnings.append("Dopant may increase cost or side-reaction risk; require ICP and post-cycle XPS.")
        return warnings

    def tradeoffs(
        self,
        q_cathode: float,
        anode: AnodeOption,
        np_ratio: float,
        specific_energy: float,
        defect: DefectChemistryResult,
        additives: Sequence[AdditiveRule],
    ) -> Dict[str, float]:
        energy_score = float(np.clip(specific_energy / 260.0, 0.0, 1.25))
        safety_score = float(anode.safety_score)
        balance_score = float(np.clip(1.0 - abs(np_ratio - 1.10) / 0.18, 0.0, 1.0))
        defect_score = float(defect.defect_tolerance_score)
        cost_score = float(np.clip(1.0 - 0.18 * anode.relative_cost_index - 0.018 * len(additives), 0.0, 1.0))
        manufacturability = float(np.clip(0.90 - 0.25 * anode.volume_change_fraction - 0.035 * len(additives), 0.0, 1.0))
        capacity_score = float(np.clip(q_cathode / 180.0, 0.0, 1.2))
        return {
            "energy_score": energy_score,
            "safety_score": safety_score,
            "balance_score": balance_score,
            "defect_score": defect_score,
            "cost_score": cost_score,
            "manufacturability_score": manufacturability,
            "capacity_score": capacity_score,
        }

    @staticmethod
    def score(tradeoffs: Dict[str, float], warnings: Sequence[str]) -> float:
        score = (
            0.24 * tradeoffs["energy_score"]
            + 0.16 * tradeoffs["safety_score"]
            + 0.14 * tradeoffs["balance_score"]
            + 0.18 * tradeoffs["defect_score"]
            + 0.12 * tradeoffs["cost_score"]
            + 0.10 * tradeoffs["manufacturability_score"]
            + 0.06 * tradeoffs["capacity_score"]
        )
        score -= 0.06 * len(warnings)
        return float(np.clip(score, 0.0, 1.2))

    def validation_gates(
        self,
        comp: Dict[str, Any],
        anode: AnodeOption,
        defect: DefectChemistryResult,
        additives: Sequence[AdditiveRule],
    ) -> List[str]:
        gates = [
            "Assemble cathode half-cell first; full-cell design is blocked until first-cycle cathode capacity is within 10 percent of prediction.",
            "Measure anode first-cycle efficiency and adjust N/P ratio before pouch-cell build.",
            "Run three formation protocols and require coulombic efficiency above 99.2 percent by cycle 5.",
            "Post-formation EIS must not exceed 1.35x the baseline electrolyte impedance.",
            "Cycle 25 full cells and compare capacity retention against cathode half-cell control.",
        ]
        if defect.moisture_sensitivity > 0.55:
            gates.append("Karl Fischer water content must be below 20 ppm for electrolyte and below 500 ppm for handled powder before cell build.")
        if defect.transition_metal_mixing_risk > 0.45:
            gates.append("ICP-OES electrolyte metal dissolution after formation must stay below the agreed Mn and Fe ppm limit.")
        if additives:
            gates.append("Run additive ablation cells so the compensation package is proven, not assumed.")
        if anode.name == "hard_carbon":
            gates.append("Quantify pre-sodiation or sacrificial sodium source needed to cover hard-carbon irreversible capacity.")
        return gates

    def propagate_uncertainty(
        self,
        cathode: Dict[str, Any],
        q_cathode: float,
        np_ratio: float,
        specific_energy: float,
        nominal_voltage: float,
        defect: DefectChemistryResult,
        additives: Sequence[AdditiveRule],
    ) -> Dict[str, Dict[str, float]]:
        raw = cathode.get("uncertainty", {}) if isinstance(cathode, dict) else {}
        q_var = float(raw.get("Q0_variance", raw.get("capacity_variance", (0.08 * q_cathode) ** 2)))
        voltage_var = float(raw.get("voltage_variance", 0.03 ** 2))
        defect_var = 0.0025 + 0.010 * max(defect.moisture_sensitivity, defect.transition_metal_mixing_risk, defect.oxygen_redox_risk)
        additive_var = 0.001 * len(additives)
        np_var = (0.015 ** 2) + 0.0015 * defect_var
        energy_relative_var = q_var / max(q_cathode ** 2, 1e-12) + voltage_var / max(nominal_voltage ** 2, 1e-12) + np_var / max(np_ratio ** 2, 1e-12)
        return {
            "cathode_capacity_mAh_g": UncertaintyPropagation.scalar(q_cathode, q_var, "cathode_uq_or_default"),
            "np_ratio": UncertaintyPropagation.scalar(np_ratio, np_var, "full_cell_delta_method"),
            "nominal_voltage_V": UncertaintyPropagation.scalar(nominal_voltage, voltage_var, "voltage_window_uncertainty"),
            "active_specific_energy_Wh_kg": UncertaintyPropagation.scalar(specific_energy, (specific_energy ** 2) * energy_relative_var, "propagated_capacity_voltage_balance"),
            "defect_compensation": UncertaintyPropagation.scalar(defect.defect_tolerance_score, defect_var + additive_var, "defect_to_additive_uncertainty"),
        }


def architecture_to_dict(architecture: FullCellArchitecture) -> Dict[str, Any]:
    payload = asdict(architecture)
    payload["anode"] = asdict(architecture.anode)
    payload["electrolyte_additives"] = [asdict(a) for a in architecture.electrolyte_additives]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Design a compensated full-cell architecture from a cathode candidate.")
    parser.add_argument("--composition-json", default="")
    parser.add_argument("--out", default="data/cache/full_cell_architecture_v3.json")
    parser.add_argument("--chemistry", default="Na-ion")
    args = parser.parse_args()
    if args.composition_json:
        cathode = {"comp": json.loads(args.composition_json)}
    else:
        cathode = screen_compositions(n=1, T=318)[0]
    architecture = FullCellArchitect(chemistry=args.chemistry).design(cathode)
    payload = architecture_to_dict(architecture)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"score": payload["score"], "anode": payload["anode"]["name"], "np_ratio": payload["np_ratio"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
