import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


MOLAR_MASS = {
    "Na2CO3": 105.988,
    "MnCO3": 114.947,
    "Fe2O3": 159.687,
    "Al2O3": 101.961,
    "TiO2": 79.866,
    "MgO": 40.304,
    "V2O5": 181.880,
    "Cr2O3": 151.990,
}


@dataclass
class ProtocolStep:
    step: int
    operation: str
    setpoints: Dict[str, Any]
    acceptance_gate: str
    failure_response: str


@dataclass
class SynthesisProtocol:
    formula: str
    route: str
    target_batch_g: float
    precursor_masses_g: Dict[str, float]
    process_steps: List[ProtocolStep]
    qc_plan: List[str] = field(default_factory=list)
    scaleup_notes: List[str] = field(default_factory=list)
    risk_register: List[str] = field(default_factory=list)


class SynthesisProtocolPlanner:
    def __init__(self, sodium_excess_fraction: float = 0.04):
        self.sodium_excess_fraction = sodium_excess_fraction

    def formula(self, comp: Dict[str, Any]) -> str:
        na = float(comp.get("Na", 1.0))
        mn = float(comp.get("Mn", 0.5))
        fe = float(comp.get("Fe", 0.5))
        dopant = comp.get("dopant")
        dopant_frac = float(comp.get("dopant_frac", 0.0))
        parts = [f"Na{na:.2f}", f"Mn{mn:.2f}", f"Fe{fe:.2f}"]
        if dopant and dopant_frac > 0:
            parts.append(f"{dopant}{dopant_frac:.2f}")
        parts.append("O2")
        return "".join(parts)

    def precursor_masses(self, comp: Dict[str, Any], target_batch_g: float) -> Dict[str, float]:
        na = float(comp.get("Na", 1.0)) * (1.0 + self.sodium_excess_fraction)
        mn = float(comp.get("Mn", 0.5))
        fe = float(comp.get("Fe", 0.5))
        dopant = comp.get("dopant")
        dopant_frac = float(comp.get("dopant_frac", 0.0))
        oxide_formula_mass = 22.989 * na + 54.938 * mn + 55.845 * fe + 2.0 * 15.999
        if dopant and dopant_frac:
            oxide_formula_mass += self._dopant_atomic_mass(dopant) * dopant_frac
        moles_product = target_batch_g / max(oxide_formula_mass, 1e-9)
        masses = {
            "Na2CO3": 0.5 * na * moles_product * MOLAR_MASS["Na2CO3"],
            "MnCO3": mn * moles_product * MOLAR_MASS["MnCO3"],
            "Fe2O3": 0.5 * fe * moles_product * MOLAR_MASS["Fe2O3"],
        }
        if dopant and dopant_frac > 0:
            precursor, stoich = self._dopant_precursor(dopant)
            masses[precursor] = stoich * dopant_frac * moles_product * MOLAR_MASS[precursor]
        return {k: round(float(v), 4) for k, v in masses.items() if v > 1e-6}

    def _dopant_atomic_mass(self, dopant: str) -> float:
        return {"Al": 26.982, "Ti": 47.867, "Mg": 24.305, "V": 50.942, "Cr": 51.996}.get(dopant, 40.0)

    def _dopant_precursor(self, dopant: str) -> tuple:
        return {
            "Al": ("Al2O3", 0.5),
            "Ti": ("TiO2", 1.0),
            "Mg": ("MgO", 1.0),
            "V": ("V2O5", 0.5),
            "Cr": ("Cr2O3", 0.5),
        }.get(dopant, ("Al2O3", 0.5))

    def build(
        self,
        comp: Dict[str, Any],
        route: str = "coprecipitation",
        target_batch_g: float = 5.0,
        lab_constraints: Optional[Dict[str, Any]] = None,
    ) -> SynthesisProtocol:
        route = route.lower().replace("-", "_")
        masses = self.precursor_masses(comp, target_batch_g)
        if route == "coprecipitation":
            steps = self._coprecipitation_steps(comp, lab_constraints or {})
        elif route == "sol_gel":
            steps = self._sol_gel_steps(comp, lab_constraints or {})
        elif route == "hydrothermal":
            steps = self._hydrothermal_steps(comp, lab_constraints or {})
        else:
            steps = self._solid_state_steps(comp, lab_constraints or {})
            route = "solid_state"
        return SynthesisProtocol(
            formula=self.formula(comp),
            route=route,
            target_batch_g=float(target_batch_g),
            precursor_masses_g=masses,
            process_steps=steps,
            qc_plan=self._qc_plan(comp),
            scaleup_notes=self._scaleup_notes(route),
            risk_register=self._risk_register(comp, route),
        )

    def _solid_state_steps(self, comp: Dict[str, Any], constraints: Dict[str, Any]) -> List[ProtocolStep]:
        return [
            ProtocolStep(1, "Dry carbonate and oxide precursors", {"temperature_C": 120, "time_h": 6}, "Mass loss below 0.2 percent after repeat weighing", "Extend drying by 2 h and reweigh."),
            ProtocolStep(2, "Planetary mill or agate mortar homogenization", {"time_min": 45, "solvent": "isopropanol optional"}, "No visible Fe-rich or Mn-rich agglomerates", "Repeat grinding with 5 wt percent extra solvent."),
            ProtocolStep(3, "Pre-calcination under air", {"temperature_C": 500, "ramp_C_min": 3, "hold_h": 5}, "Powder color is homogeneous and mass loss matches carbonate removal", "Lower ramp and repeat if foaming or crusting occurs."),
            ProtocolStep(4, "Final calcination", {"temperature_C": 850, "ramp_C_min": 2, "hold_h": 12, "cooling": "furnace cool"}, "XRD target phase above 85 percent relative intensity", "Regrind and repeat at 875 C if carbonate peaks remain."),
            ProtocolStep(5, "Moisture-controlled storage", {"container": "sealed vial", "desiccant": True}, "No clumping after 24 h", "Dry at 100 C before electrode slurry preparation."),
        ]

    def _sol_gel_steps(self, comp: Dict[str, Any], constraints: Dict[str, Any]) -> List[ProtocolStep]:
        return [
            ProtocolStep(1, "Dissolve nitrate or acetate salts in DI water", {"metal_concentration_M": 0.5, "citric_acid_metal_ratio": 1.5}, "Clear solution with pH 6 to 7", "Filter and remake if insoluble residue persists."),
            ProtocolStep(2, "Chelation and gel formation", {"temperature_C": 80, "stirring_rpm": 450, "time_h": 4}, "Viscosity rises without precipitation", "Add citric acid in 0.1 equivalents and continue stirring."),
            ProtocolStep(3, "Dry gel", {"temperature_C": 140, "time_h": 10}, "Foam-like dry gel with no free liquid", "Extend drying at 120 C."),
            ProtocolStep(4, "Organic burnout", {"temperature_C": 450, "ramp_C_min": 1, "hold_h": 4}, "No visible carbon residue after grinding", "Repeat burnout at 475 C."),
            ProtocolStep(5, "Crystallization calcination", {"temperature_C": 780, "ramp_C_min": 2, "hold_h": 10}, "XRD layered oxide phase passes gate", "Raise to 820 C only if phase impurity remains high."),
        ]

    def _coprecipitation_steps(self, comp: Dict[str, Any], constraints: Dict[str, Any]) -> List[ProtocolStep]:
        return [
            ProtocolStep(1, "Prepare transition-metal sulfate feed", {"total_metal_M": 1.0, "Mn_Fe_ratio": round(comp.get("Mn", 0.5) / max(comp.get("Fe", 0.5), 1e-9), 3)}, "ICP or mass balance feed ratio within 2 percent", "Correct feed solution before precipitation."),
            ProtocolStep(2, "Controlled hydroxide precipitation", {"pH": 10.8, "temperature_C": 55, "stirring_rpm": 700, "residence_time_h": 4}, "Particle slurry pH stable within 0.1 for 30 min", "Reduce base addition rate and age longer."),
            ProtocolStep(3, "Wash and dry precursor", {"wash_until_conductivity_uS_cm": 150, "dry_temperature_C": 110, "time_h": 12}, "Filtrate conductivity below gate", "Continue washing to prevent sodium sulfate contamination."),
            ProtocolStep(4, "Mix with sodium carbonate excess", {"sodium_excess_percent": round(100 * self.sodium_excess_fraction, 1)}, "Dry blend is visually uniform", "Regrind with small ethanol addition."),
            ProtocolStep(5, "Calcination under air", {"temperature_C": 820, "ramp_C_min": 2, "hold_h": 10}, "XRD target phase above 85 percent", "Regrind and rerun calcination at plus 20 C."),
        ]

    def _hydrothermal_steps(self, comp: Dict[str, Any], constraints: Dict[str, Any]) -> List[ProtocolStep]:
        return [
            ProtocolStep(1, "Prepare alkaline precursor suspension", {"NaOH_M": 4.0, "fill_fraction": 0.70}, "No large settled agglomerates before sealing", "Increase sonication time to 20 min."),
            ProtocolStep(2, "Hydrothermal reaction", {"temperature_C": 180, "time_h": 18}, "Autoclave pressure trace remains stable", "Reject run if pressure spike occurs."),
            ProtocolStep(3, "Wash to neutral pH", {"target_pH": 7.5}, "Filtrate pH below 8", "Repeat washing in centrifuge cycles."),
            ProtocolStep(4, "Low-temperature anneal", {"temperature_C": 650, "time_h": 6}, "XRD shows crystalline sodium transition metal oxide", "Escalate to solid-state route if crystallinity remains poor."),
        ]

    def _qc_plan(self, comp: Dict[str, Any]) -> List[str]:
        return [
            "XRD immediately after synthesis with Rietveld or reference-intensity phase fraction estimate.",
            "SEM particle size distribution before electrode coating.",
            "ICP-OES or XRF composition check against target stoichiometry.",
            "Coin-cell formation at C/20, then rate capability at C/10, C/5, C/2 and 1C.",
            "EIS after formation, after cycle 25, and after cycle 100.",
            "45 C accelerated cycling for the same electrode loading used at room temperature.",
        ]

    def _scaleup_notes(self, route: str) -> List[str]:
        base = [
            "Do not claim industrial readiness until a second independent batch reproduces XRD and first-cycle capacity.",
            "Track batch humidity, powder residence time, and sodium precursor lot because sodium layered oxides are moisture-sensitive.",
        ]
        if route == "coprecipitation":
            base.append("Scale mixing by constant impeller tip speed and residence time, not by fixed rpm.")
        if route == "solid_state":
            base.append("Scale by thermal mass and powder bed height; calcination failure often comes from oxygen and heat-transfer gradients.")
        return base

    def _risk_register(self, comp: Dict[str, Any], route: str) -> List[str]:
        risks = [
            "Sodium volatility can shift stoichiometry during calcination.",
            "Mn-rich candidates can suffer Jahn-Teller distortion and Mn dissolution.",
            "Fe-rich candidates can lose capacity if redox participation is weaker than expected.",
        ]
        dopant = comp.get("dopant")
        if dopant in ("Ti", "Al"):
            risks.append(f"{dopant} improves stability but can reduce practical capacity if overdosed.")
        if route == "coprecipitation":
            risks.append("Transition metal segregation during precipitation will invalidate the model assumption of homogeneous mixing.")
        return risks


def protocol_to_dict(protocol: SynthesisProtocol) -> Dict[str, Any]:
    payload = asdict(protocol)
    payload["process_steps"] = [asdict(s) for s in protocol.process_steps]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition-json", default="")
    parser.add_argument("--route", default="coprecipitation")
    parser.add_argument("--batch-g", type=float, default=5.0)
    parser.add_argument("--out", default="data/cache/synthesis_protocol_v2.json")
    args = parser.parse_args()
    if args.composition_json:
        comp = json.loads(args.composition_json)
    else:
        comp = {"Na": 1.02, "Mn": 0.48, "Fe": 0.47, "dopant": "Al", "dopant_frac": 0.05}
    planner = SynthesisProtocolPlanner()
    protocol = planner.build(comp, route=args.route, target_batch_g=args.batch_g)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol_to_dict(protocol), indent=2), encoding="utf-8")
    print(json.dumps({"formula": protocol.formula, "route": protocol.route, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
