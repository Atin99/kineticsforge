import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PilotCustomerProfile:
    company_name: str
    segment: str
    chemistry: str
    pain_point: str
    available_data: str
    lab_capability: str
    decision_maker: str
    expected_value_inr: float
    urgency: float = 0.5
    validation_access: float = 0.5
    strategic_fit: float = 0.5


@dataclass
class PilotMilestone:
    week: int
    deliverable: str
    acceptance_metric: str
    payment_percent: float
    evidence_required: str


@dataclass
class PilotOffer:
    customer: PilotCustomerProfile
    offer_name: str
    price_inr: float
    duration_weeks: int
    milestones: List[PilotMilestone]
    data_required: List[str]
    exclusions: List[str]
    ip_terms: List[str]
    success_metrics: List[str]
    red_flags: List[str]
    roi_case: Dict[str, float] = field(default_factory=dict)


def build_pilot_offer(customer: PilotCustomerProfile, base_price_inr: Optional[float] = None) -> PilotOffer:
    complexity = _complexity_multiplier(customer)
    price = base_price_inr if base_price_inr is not None else round(450000.0 * complexity, -4)
    duration = 4 if customer.segment in ("recycling", "bms") else 6
    milestones = [
        PilotMilestone(1, "Data ingestion and physics baseline", "All supplied files fingerprinted; baseline model reproduces known trend direction.", 20.0, "Data manifest, units table, baseline report"),
        PilotMilestone(2, "Surrogate calibration and uncertainty envelope", "Holdout error or literature replay error below agreed threshold.", 25.0, "Run ledger, validation split, model card"),
        PilotMilestone(duration - 1, "Actionable recommendation set", "Top 3 candidate conditions or compositions include constraints, risks, and lab protocol.", 30.0, "Recommendation pack and audit report"),
        PilotMilestone(duration, "Validation handoff and executive review", "Client can decide go/no-go on at least one lab experiment.", 25.0, "Final report, evidence trace, next experiment protocol"),
    ]
    return PilotOffer(
        customer=customer,
        offer_name=_offer_name(customer),
        price_inr=float(price),
        duration_weeks=duration,
        milestones=milestones,
        data_required=_data_required(customer),
        exclusions=[
            "No guaranteed electrochemical performance without independent lab validation.",
            "No ownership transfer of KineticsForge source code or model backbone.",
            "No regulatory certification claim unless a certified lab executes the relevant standard test.",
        ],
        ip_terms=[
            "Client owns raw confidential input data.",
            "KineticsForge owns pre-existing models, tooling, and generic improvements.",
            "Joint inventions from client-specific composition or process outputs require a separate patent and licensing addendum.",
            "Anonymized error statistics may be retained to improve model calibration unless the client opts out in writing.",
        ],
        success_metrics=_success_metrics(customer),
        red_flags=_red_flags(customer),
        roi_case=roi_case(customer, float(price)),
    )


def _complexity_multiplier(customer: PilotCustomerProfile) -> float:
    base = 1.0
    if "none" in customer.available_data.lower() or "limited" in customer.available_data.lower():
        base += 0.35
    if customer.segment == "cathode":
        base += 0.35
    if customer.segment == "bms":
        base += 0.20
    if customer.validation_access < 0.4:
        base += 0.25
    return max(0.8, min(2.2, base))


def _offer_name(customer: PilotCustomerProfile) -> str:
    if customer.segment == "cathode":
        return "4-6 week sodium cathode inverse-design pilot"
    if customer.segment == "bms":
        return "4 week battery telemetry risk twin pilot"
    if customer.segment == "recycling":
        return "4 week black-mass leaching optimization pilot"
    return "KineticsForge physics-constrained battery materials pilot"


def _data_required(customer: PilotCustomerProfile) -> List[str]:
    common = [
        "Material or cell identifier table with units and batch dates",
        "Any failed experiments, not only successful ones",
        "Operating temperature, current, and time basis for all measurements",
        "Permission to retain de-identified aggregate error statistics",
    ]
    if customer.segment == "cathode":
        common.extend([
            "Composition targets and actual ICP/XRF values where available",
            "Cycling CSV with cycle number, discharge capacity, charge capacity, voltage window, C-rate",
            "Synthesis route, calcination temperature profile, particle size if measured",
        ])
    elif customer.segment == "bms":
        common.extend([
            "Timestamped V/I/T telemetry at cell or module level",
            "Known fault windows, service events, and pack topology",
            "BMS sampling interval and sensor resolution",
        ])
    elif customer.segment == "recycling":
        common.extend([
            "Black mass assay for Mn, Fe, Na, Al, Cu, moisture",
            "Leaching condition history: temperature, acid molarity, pH, time, solid-liquid ratio",
            "Recovery and impurity assay after each run",
        ])
    return common


def _success_metrics(customer: PilotCustomerProfile) -> List[str]:
    if customer.segment == "cathode":
        return [
            "Shortlist 3 cathode compositions with predicted 500-cycle fade and confidence interval.",
            "Generate a synthesis protocol that fits the client's lab capability.",
            "Replay at least one known composition trend from literature or client data.",
        ]
    if customer.segment == "bms":
        return [
            "Detect injected or historical fault windows before thermal threshold crossing.",
            "Produce per-cell risk trajectory and explain the top contributing sensor residuals.",
            "Estimate false-alert budget at selected threshold.",
        ]
    if customer.segment == "recycling":
        return [
            "Recommend leaching conditions with recovery, impurity, cost, and waste tradeoff.",
            "Show sensitivity to pH, temperature, and particle size.",
            "Convert recovered stream into cathode feedstock feasibility estimate.",
        ]
    return ["Deliver audited physics-constrained recommendation pack."]


def _red_flags(customer: PilotCustomerProfile) -> List[str]:
    flags: List[str] = []
    if customer.validation_access < 0.35:
        flags.append("No lab validation access; frame as computational screening only.")
    if customer.urgency < 0.35:
        flags.append("Pain may be curiosity, not budget-backed urgency.")
    if customer.expected_value_inr < 5.0 * 450000.0:
        flags.append("Estimated value may not justify even a small paid pilot.")
    if "none" in customer.available_data.lower():
        flags.append("No data means the first engagement must be a paid feasibility audit, not a performance pilot.")
    return flags or ["No immediate red flags if data access and validation owner are confirmed."]


def roi_case(customer: PilotCustomerProfile, price_inr: float) -> Dict[str, float]:
    value = max(customer.expected_value_inr, 1.0)
    downside = 0.35 * value
    conservative_savings = 0.08 * value
    expected_roi = conservative_savings / max(price_inr, 1.0)
    return {
        "client_expected_value_inr": float(value),
        "client_downside_if_unsolved_inr": float(downside),
        "conservative_savings_inr": float(conservative_savings),
        "pilot_price_inr": float(price_inr),
        "expected_roi_multiple": float(expected_roi),
    }


def build_target_account_playbook() -> List[Dict[str, Any]]:
    accounts = [
        PilotCustomerProfile("Na-ion cell startup", "cathode", "Na-Mn-Fe-O", "Needs low-cost cathode adapted to Indian high-temperature cycling", "limited cycling data", "coin-cell and furnace access", "CTO or founder", 12000000.0, urgency=0.75, validation_access=0.65, strategic_fit=0.95),
        PilotCustomerProfile("Battery recycler", "recycling", "spent LFP or sodium cathode black mass", "Recovery is lower than target and impurity penalty is high", "batch leaching logs", "wet lab with ICP access", "Plant head or process R&D lead", 9000000.0, urgency=0.80, validation_access=0.75, strategic_fit=0.90),
        PilotCustomerProfile("Two-wheeler pack OEM", "bms", "Li-ion now, Na-ion later", "Thermal incidents and warranty returns need earlier warning", "telemetry plus warranty logs", "pack test bench", "Battery systems head", 25000000.0, urgency=0.70, validation_access=0.55, strategic_fit=0.78),
        PilotCustomerProfile("Academic validation lab", "cathode", "Na-ion layered oxides", "Needs computational shortlist for publishable experiments", "literature and small lab data", "XRD, coin cells, furnace", "Professor or PhD lead", 2000000.0, urgency=0.45, validation_access=0.90, strategic_fit=0.82),
    ]
    return [offer_to_dict(build_pilot_offer(a)) for a in accounts]


def generate_90_day_plan() -> List[Dict[str, Any]]:
    return [
        {"day": 1, "workstream": "evidence", "action": "Freeze v2 demo claims and mark each as proven, plausible, or validation-needed.", "output": "claim ledger"},
        {"day": 3, "workstream": "product", "action": "Run readiness report on cathode, BMS, recycling and fix all critical audit findings.", "output": "audit JSON plus screenshots"},
        {"day": 7, "workstream": "validation", "action": "Send one-page validation ask to two labs with exact experiments and materials list.", "output": "lab email plus protocol"},
        {"day": 10, "workstream": "sales", "action": "Approach 10 target accounts with a paid feasibility audit, not a generic demo.", "output": "CRM sheet"},
        {"day": 21, "workstream": "pilot", "action": "Close one unpaid academic validation and one paid industrial discovery call.", "output": "signed data NDA or email acceptance"},
        {"day": 35, "workstream": "model", "action": "Train Kaggle standard profile and publish run ledger, model card, and failure analysis.", "output": "reproducible training bundle"},
        {"day": 60, "workstream": "business", "action": "Convert one discovery call to a 4-6 week pilot with milestone payment.", "output": "pilot SOW"},
        {"day": 90, "workstream": "moat", "action": "File provisional disclosure only after a lab-backed result or unique telemetry result exists.", "output": "provisional draft"},
    ]


def offer_to_dict(offer: PilotOffer) -> Dict[str, Any]:
    return {
        "customer": asdict(offer.customer),
        "offer_name": offer.offer_name,
        "price_inr": offer.price_inr,
        "duration_weeks": offer.duration_weeks,
        "milestones": [asdict(m) for m in offer.milestones],
        "data_required": offer.data_required,
        "exclusions": offer.exclusions,
        "ip_terms": offer.ip_terms,
        "success_metrics": offer.success_metrics,
        "red_flags": offer.red_flags,
        "roi_case": offer.roi_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/pilot_contract_pack_v2.json")
    args = parser.parse_args()
    payload = {
        "target_accounts": build_target_account_playbook(),
        "ninety_day_plan": generate_90_day_plan(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"accounts": len(payload["target_accounts"]), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
