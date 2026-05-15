import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DefectChemistryResult:
    charge_balance_error: float
    sodium_vacancy_fraction: float
    oxygen_redox_risk: float
    transition_metal_mixing_risk: float
    moisture_sensitivity: float
    jahn_teller_risk: float
    defect_tolerance_score: float
    suggested_compensation: List[str]


class DefectChemistryModel:
    def __init__(self) -> None:
        self.valence = {"Na": 1.0, "Mn": 3.4, "Fe": 3.0, "Al": 3.0, "Ti": 4.0, "Mg": 2.0, "V": 4.0, "Cr": 3.0, "O": -2.0}

    def evaluate(self, comp: Dict[str, Any], humidity: float = 0.82, calcination_C: float = 820.0) -> DefectChemistryResult:
        na = float(comp.get("Na", 1.0))
        mn = float(comp.get("Mn", 0.5))
        fe = float(comp.get("Fe", 0.5))
        dopant = comp.get("dopant")
        dop = float(comp.get("dopant_frac", 0.0)) if dopant else 0.0
        tm_sum = max(mn + fe + dop, 1e-9)
        mn_n = mn / tm_sum
        fe_n = fe / tm_sum
        dop_n = dop / tm_sum
        oxygen_charge = -4.0
        cation_charge = na * self.valence["Na"] + mn_n * self.valence["Mn"] + fe_n * self.valence["Fe"]
        if dopant:
            cation_charge += dop_n * self.valence.get(dopant, 3.0)
        charge_balance_error = cation_charge + oxygen_charge
        sodium_vacancy = float(np.clip(1.0 - na, 0.0, 0.22))
        oxygen_redox = self._oxygen_redox_risk(na, mn_n, fe_n, dopant, dop_n, charge_balance_error)
        mixing = self._tm_mixing_risk(na, mn_n, fe_n, dopant, calcination_C)
        moisture = self._moisture_sensitivity(na, sodium_vacancy, humidity)
        jt = float(np.clip((mn_n - 0.48) * 1.8, 0.0, 1.0))
        defect_score = float(np.clip(1.0 - (0.24 * oxygen_redox + 0.22 * mixing + 0.20 * moisture + 0.24 * jt + 0.10 * abs(charge_balance_error)), 0.0, 1.0))
        return DefectChemistryResult(
            charge_balance_error=float(charge_balance_error),
            sodium_vacancy_fraction=sodium_vacancy,
            oxygen_redox_risk=float(oxygen_redox),
            transition_metal_mixing_risk=float(mixing),
            moisture_sensitivity=float(moisture),
            jahn_teller_risk=float(jt),
            defect_tolerance_score=defect_score,
            suggested_compensation=self._compensation(comp, charge_balance_error, oxygen_redox, mixing, moisture, jt),
        )

    def _oxygen_redox_risk(self, na: float, mn: float, fe: float, dopant: Optional[str], dop: float, charge_error: float) -> float:
        high_voltage_drive = max(0.0, mn - 0.55) + max(0.0, 1.0 - na) * 0.8
        charge_drive = max(0.0, -charge_error) * 0.35
        dopant_relief = {"Al": 0.08, "Ti": 0.12, "Mg": 0.02, "V": 0.06, "Cr": 0.04}.get(dopant, 0.0) * (dop / 0.05 if dop else 0.0)
        return float(np.clip(0.22 + high_voltage_drive + charge_drive - dopant_relief, 0.0, 1.0))

    def _tm_mixing_risk(self, na: float, mn: float, fe: float, dopant: Optional[str], calcination_C: float) -> float:
        radius_mismatch = abs(mn - fe) * 0.35
        sodium_deficit = max(0.0, 0.98 - na) * 1.2
        thermal_drive = max(0.0, calcination_C - 840.0) / 180.0
        dopant_penalty = {"Ti": 0.04, "Al": -0.03, "Mg": 0.02, "V": 0.03, "Cr": 0.02}.get(dopant, 0.0)
        return float(np.clip(0.18 + radius_mismatch + sodium_deficit + thermal_drive + dopant_penalty, 0.0, 1.0))

    def _moisture_sensitivity(self, na: float, sodium_vacancy: float, humidity: float) -> float:
        sodium_surface = max(0.0, na - 0.98) * 0.9
        vacancy_pathways = sodium_vacancy * 2.2
        humidity_drive = max(0.0, humidity - 0.55) * 0.9
        return float(np.clip(0.20 + sodium_surface + vacancy_pathways + humidity_drive, 0.0, 1.0))

    def _compensation(
        self,
        comp: Dict[str, Any],
        charge_error: float,
        oxygen_redox: float,
        mixing: float,
        moisture: float,
        jt: float,
    ) -> List[str]:
        actions: List[str] = []
        if charge_error < -0.08:
            actions.append("Increase Na excess or use a higher-valence dopant to reduce oxygen-redox compensation burden.")
        if charge_error > 0.12:
            actions.append("Reduce sodium excess or lower high-valence dopant fraction to avoid cation overcharge.")
        if oxygen_redox > 0.55:
            actions.append("Add Al or Ti at 2-5 mol percent and lower upper cutoff voltage during first validation.")
        if mixing > 0.50:
            actions.append("Reduce calcination peak or add sodium excess during final firing to limit transition-metal migration.")
        if moisture > 0.60:
            actions.append("Store powder under desiccant and transfer electrodes quickly during monsoon humidity.")
        if jt > 0.55:
            actions.append("Reduce Mn fraction or compensate with Fe/Ti to lower Jahn-Teller distortion risk.")
        return actions or ["No first-order defect compensation required; proceed to XRD and first-cycle validation."]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition-json", default="")
    parser.add_argument("--out", default="data/cache/defect_chemistry_v2.json")
    args = parser.parse_args()
    comp = json.loads(args.composition_json) if args.composition_json else {"Na": 1.02, "Mn": 0.48, "Fe": 0.47, "dopant": "Al", "dopant_frac": 0.05}
    result = DefectChemistryModel().evaluate(comp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
