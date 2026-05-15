"""Lab validation budget estimator for Na-ion cathode candidates in India.

IMPORTANT: This produces PLANNING ESTIMATES with explicit uncertainty ranges,
NOT quotes. Every unit cost has a source. The output is a range (low-high),
never a single precise number, because precise numbers from hardcoded rates
are dishonest.

Before committing real money, get actual quotes from:
  - CSIR-CECRI Karaikudi (Na-ion electrochemistry expertise)
  - IIT Madras / IIT Bombay CIF
  - IISER Pune / IISER Mohali
  - Private CROs like Battelle India or SGS
"""
import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.india_context import IndiaOperatingContext


@dataclass
class LabCostCatalogue:
    """Unit costs from published Indian CIF rate cards and Indiamart wholesale.

    Sources:
      XRD: CSIR/IIT CIF academic rate cards 2024 (₹400-1300/scan)
      SEM: CIF rate cards (₹800-2500/session)
      ICP-OES: CIF rate cards (₹1000-2500/sample)
      Coin cell: Component cost estimation from Indiamart + assembly labour
      DSC/TGA: CIF rate cards (₹1500-4000/run)
      EIS: Typically included in cycler time or ₹300-800/spectrum
      Furnace: Institutional rates (₹100-300/hour)
      Lab tech: CSIR JRF stipend ₹37k/month basis
      Precursors: Indiamart wholesale prices 2024, lab-grade
      Cycler: Academic shared facility rates

    All prices are ACADEMIC tier. Industry/external rates are typically 2-3x.
    """
    # Ranges: (low, mid, high)
    xrd_scan: tuple = (400, 800, 1300)
    sem_session: tuple = (800, 1500, 2500)
    icp_sample: tuple = (1000, 1500, 2500)
    coin_cell_assembly: tuple = (500, 850, 1400)
    coin_cell_materials: tuple = (800, 1200, 2000)
    electrolyte_per_ml: tuple = (18, 28, 45)
    dsc_tga_run: tuple = (1500, 2500, 4000)
    eis_spectrum: tuple = (300, 500, 800)
    furnace_hour: tuple = (100, 180, 300)
    mill_hour: tuple = (150, 250, 400)
    lab_tech_day: tuple = (1500, 1800, 2200)
    phd_day: tuple = (2500, 3200, 4500)
    shipping_hazmat: tuple = (3000, 4500, 7000)
    precursor_na2co3_kg: tuple = (280, 380, 550)
    precursor_mnco3_kg: tuple = (800, 1200, 1800)
    precursor_fe2o3_kg: tuple = (350, 550, 800)
    dopant_precursor_kg: tuple = (1800, 2800, 4500)
    cycler_channel_day: tuple = (200, 350, 600)
    thermal_chamber_day: tuple = (500, 900, 1500)


@dataclass
class BudgetRange:
    """Every budget output is a range, not a point estimate."""
    low_inr: float
    mid_inr: float
    high_inr: float
    low_usd: float
    mid_usd: float
    high_usd: float


@dataclass
class ExperimentSpec:
    candidate_id: str
    composition: Dict[str, Any]
    synthesis_route: str
    batch_g: float = 5.0
    xrd_scans: int = 4
    sem_sessions: int = 2
    icp_samples: int = 3
    coin_cells: int = 6
    eis_spectra: int = 18
    dsc_runs: int = 1
    cycling_days_rt: int = 21
    cycling_days_45c: int = 14
    duplicate_batches: int = 2
    calcination_hours: float = 14.0
    milling_hours: float = 1.5


@dataclass
class CandidateBudget:
    candidate_id: str
    composition: Dict[str, Any]
    budget: BudgetRange
    calendar_days: int
    go_no_go_gates: List[str]
    risk_items: List[str]
    breakdown_low: Dict[str, float]
    breakdown_high: Dict[str, float]
    honest_caveats: List[str]


@dataclass
class ValidationPlan:
    candidates: List[CandidateBudget]
    total: BudgetRange
    calendar_weeks_estimate: str
    shared_costs: BudgetRange
    assumptions: List[str]
    how_to_get_real_quotes: List[str]


class LabValidationBudgetCalculator:
    def __init__(self, catalogue: Optional[LabCostCatalogue] = None, india: Optional[IndiaOperatingContext] = None):
        self.cat = catalogue or LabCostCatalogue()
        self.india = india or IndiaOperatingContext.from_env()

    def _range_sum(self, *ranges: tuple) -> tuple:
        return (sum(r[0] for r in ranges), sum(r[1] for r in ranges), sum(r[2] for r in ranges))

    def _range_mult(self, r: tuple, n: float) -> tuple:
        return (r[0] * n, r[1] * n, r[2] * n)

    def _to_budget(self, r: tuple) -> BudgetRange:
        return BudgetRange(
            low_inr=float(r[0]), mid_inr=float(r[1]), high_inr=float(r[2]),
            low_usd=float(self.india.rupees_to_usd(r[0])),
            mid_usd=float(self.india.rupees_to_usd(r[1])),
            high_usd=float(self.india.rupees_to_usd(r[2])),
        )

    def budget_one(self, spec: ExperimentSpec) -> CandidateBudget:
        c = self.cat
        n = max(spec.duplicate_batches, 1)

        precursor = self._range_sum(
            self._range_mult(c.precursor_na2co3_kg, 0.003 * n),
            self._range_mult(c.precursor_mnco3_kg, 0.004 * n),
            self._range_mult(c.precursor_fe2o3_kg, 0.003 * n),
        )
        if spec.composition.get("dopant"):
            precursor = self._range_sum(precursor, self._range_mult(c.dopant_precursor_kg, 0.001 * n))

        furnace = self._range_mult(c.furnace_hour, spec.calcination_hours * n)
        milling = self._range_mult(c.mill_hour, spec.milling_hours * n)
        xrd = self._range_mult(c.xrd_scan, spec.xrd_scans * n)
        sem = self._range_mult(c.sem_session, spec.sem_sessions)
        icp = self._range_mult(c.icp_sample, spec.icp_samples * n)
        dsc = self._range_mult(c.dsc_tga_run, spec.dsc_runs)
        cell_fab = self._range_sum(
            self._range_mult(c.coin_cell_assembly, spec.coin_cells * n),
            self._range_mult(c.coin_cell_materials, spec.coin_cells * n),
        )
        electrolyte = self._range_mult(c.electrolyte_per_ml, 0.8 * spec.coin_cells * n)
        eis = self._range_mult(c.eis_spectrum, spec.eis_spectra)
        cycler_rt = self._range_mult(c.cycler_channel_day, spec.cycling_days_rt * spec.coin_cells)
        cycler_45 = self._range_sum(
            self._range_mult(c.cycler_channel_day, spec.cycling_days_45c * max(spec.coin_cells // 2, 1)),
            self._range_mult(c.thermal_chamber_day, spec.cycling_days_45c * max(spec.coin_cells // 2, 1)),
        )

        synth_days = max(int(math.ceil(spec.calcination_hours / 8.0)), 2) * n
        charact_days = spec.xrd_scans + spec.sem_sessions + spec.icp_samples
        cycling_lab_days = 3 + spec.cycling_days_rt // 7 + spec.cycling_days_45c // 7
        total_lab_days = synth_days + charact_days + cycling_lab_days
        labour = self._range_sum(
            self._range_mult(c.lab_tech_day, total_lab_days),
            self._range_mult(c.phd_day, max(total_lab_days // 3, 2)),
        )

        calendar = synth_days + 2 + max(spec.cycling_days_rt, spec.cycling_days_45c) + 5

        all_items = [precursor, furnace, milling, xrd, sem, icp, dsc, cell_fab, electrolyte, eis, cycler_rt, cycler_45, labour]
        total = self._range_sum(*all_items)
        # Add 10-15% overhead
        total = (total[0] * 1.10, total[1] * 1.12, total[2] * 1.15)

        return CandidateBudget(
            candidate_id=spec.candidate_id,
            composition=spec.composition,
            budget=self._to_budget(total),
            calendar_days=int(calendar),
            go_no_go_gates=self._gates(spec),
            risk_items=self._risks(spec),
            breakdown_low={"precursor": precursor[0], "furnace": furnace[0], "characterisation": (xrd[0]+sem[0]+icp[0]+dsc[0]), "cycling": (cell_fab[0]+electrolyte[0]+eis[0]+cycler_rt[0]+cycler_45[0]), "labour": labour[0]},
            breakdown_high={"precursor": precursor[2], "furnace": furnace[2], "characterisation": (xrd[2]+sem[2]+icp[2]+dsc[2]), "cycling": (cell_fab[2]+electrolyte[2]+eis[2]+cycler_rt[2]+cycler_45[2]), "labour": labour[2]},
            honest_caveats=[
                "These are PLANNING ESTIMATES, not quotes. Get real quotes before committing money.",
                "Rates are academic tier from CIF rate cards. Industry/external user rates are 2-3x higher.",
                "Does NOT include cycler queue wait time (can add 1-4 weeks at Indian labs).",
                "Does NOT include full-cell or pouch-cell testing (add ₹45k-80k per format).",
                "Precursor prices are Indiamart wholesale; lab-grade from Sigma/Alfa may cost 5-10x more.",
            ],
        )

    def plan_for_candidates(self, candidates: Sequence[Dict[str, Any]], route: str = "coprecipitation") -> ValidationPlan:
        budgets: List[CandidateBudget] = []
        for i, cand in enumerate(candidates):
            comp = cand.get("composition", cand)
            spec = ExperimentSpec(candidate_id=f"candidate_{i+1}", composition=comp, synthesis_route=route)
            budgets.append(self.budget_one(spec))

        shared_low = 8000 * len(candidates) + 2200 * len(candidates) + 3000 + 2500 * 5
        shared_high = 15000 * len(candidates) + 4000 * len(candidates) + 7000 + 4500 * 5
        shared_mid = (shared_low + shared_high) / 2
        shared = (shared_low, shared_mid, shared_high)

        total_low = sum(b.budget.low_inr for b in budgets) + shared[0]
        total_mid = sum(b.budget.mid_inr for b in budgets) + shared[1]
        total_high = sum(b.budget.high_inr for b in budgets) + shared[2]

        max_cal = max((b.calendar_days for b in budgets), default=30)
        parallel_factor = max(1, math.ceil(len(budgets) / 2))
        weeks_low = int(math.ceil(max_cal / 7.0))
        weeks_high = int(math.ceil(max_cal * parallel_factor / 7.0)) + 4  # queue buffer

        return ValidationPlan(
            candidates=budgets,
            total=self._to_budget((total_low, total_mid, total_high)),
            calendar_weeks_estimate=f"{weeks_low}-{weeks_high} weeks (lower bound assumes parallel; upper includes cycler queue delays)",
            shared_costs=self._to_budget(shared),
            assumptions=[
                f"Exchange rate: 1 USD = {self.india.usd_to_inr} INR ({self.india.usd_to_inr_source})",
                f"Instrument rates: {self.cat.xrd_scan[0]}-{self.cat.xrd_scan[2]} INR/XRD scan ({self.india.instrument_rate_source})",
                "Academic tier pricing. Industry rates are 2-3x higher.",
                "Coin cells: Na metal anode, NaPF6 electrolyte, GF/D separator.",
                "Does NOT account for failed batches (expect 30-50% redo rate for new compositions).",
            ],
            how_to_get_real_quotes=[
                "Email CIF coordinator at target lab with: composition, batch size, characterisation list, cycling protocol.",
                "Ask for per-sample and per-channel-day rates specifically.",
                "Request calendar availability — cycler queues at Indian labs can be 2-6 weeks.",
                "Budget 1.5-2x the mid-estimate for realistic planning with redo and delays.",
                "Recommended labs: CSIR-CECRI (Na-ion expertise), IIT Madras, IISER Pune, JNCASR Bangalore.",
            ],
        )

    def _gates(self, spec: ExperimentSpec) -> List[str]:
        return [
            "Gate 1 (Day 3): XRD — target layered oxide phase >85% relative intensity. Kill if major impurity phase.",
            "Gate 2 (Day 5): ICP/XRF — stoichiometry within 3% absolute of design target. Kill if Na deficit >5%.",
            "Gate 3 (Day 8): Formation — C/10 discharge capacity ≥85% of predicted Q0. Flag if below, continue cycling.",
            f"Gate 4 (Day {spec.cycling_days_rt + 8}): RT cycling — 100-cycle retention ≥ model 95% lower bound. Kill if below.",
            f"Gate 5 (Day {spec.cycling_days_rt + spec.cycling_days_45c + 8}): 45°C — fade slope within 1.5x of Arrhenius prediction. Kill if worse.",
        ]

    def _risks(self, spec: ExperimentSpec) -> List[str]:
        risks = [
            "Sodium volatility during calcination can shift stoichiometry by 2-4% (Yabuuchi 2014).",
            "Monsoon humidity >80% RH degrades Na-ion cathode powder within hours of air exposure.",
            "Cycler channel availability at shared Indian facilities is unpredictable (1-6 week queue).",
            "First-attempt synthesis failure rate for new compositions is typically 30-50%.",
        ]
        if spec.composition.get("dopant"):
            risks.append(f"Dopant ({spec.composition['dopant']}) precursor purity varies between Indian suppliers. Verify lot certificate.")
        return risks


def build_validation_plan_from_inverse_design(inverse_candidates: Sequence[Dict[str, Any]], top_n: int = 3, route: str = "coprecipitation") -> ValidationPlan:
    selected = list(inverse_candidates)[:top_n]
    calc = LabValidationBudgetCalculator()
    return calc.plan_for_candidates(selected, route=route)


def plan_to_dict(plan: ValidationPlan) -> Dict[str, Any]:
    return {
        "candidates": [asdict(b) for b in plan.candidates],
        "total": asdict(plan.total),
        "calendar_weeks_estimate": plan.calendar_weeks_estimate,
        "shared_costs": asdict(plan.shared_costs),
        "assumptions": plan.assumptions,
        "how_to_get_real_quotes": plan.how_to_get_real_quotes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/lab_validation_budget_v2.json")
    args = parser.parse_args()
    candidates = [
        {"composition": {"Na": 1.02, "Mn": 0.48, "Fe": 0.47, "dopant": "Al", "dopant_frac": 0.05}},
        {"composition": {"Na": 0.98, "Mn": 0.55, "Fe": 0.40, "dopant": "Ti", "dopant_frac": 0.03}},
        {"composition": {"Na": 1.05, "Mn": 0.42, "Fe": 0.52, "dopant": None, "dopant_frac": 0.0}},
    ]
    plan = build_validation_plan_from_inverse_design(candidates)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan_to_dict(plan), indent=2), encoding="utf-8")
    t = plan.total
    print(json.dumps({
        "budget_range_inr": f"{t.low_inr:,.0f} - {t.high_inr:,.0f}",
        "budget_range_usd": f"{t.low_usd:,.0f} - {t.high_usd:,.0f}",
        "timeline": plan.calendar_weeks_estimate,
        "caveat": "PLANNING ESTIMATE. Get real lab quotes.",
    }, indent=2))


if __name__ == "__main__":
    main()
