import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.india_context import IndiaOperatingContext


@dataclass
class Prospect:
    company_name: str
    segment: str
    city: str
    contact_role: str
    contact_name: str
    contact_email: str
    pain_point: str
    data_readiness: str
    urgency: float
    strategic_fit: float
    status: str = "identified"
    last_contacted: str = ""
    next_action: str = ""
    notes: str = ""


@dataclass
class OutreachScript:
    segment: str
    subject_line: str
    opening: str
    value_prop: str
    ask: str
    social_proof: str
    closing: str


CRM_FIELDS = ["company_name", "segment", "city", "contact_role", "contact_name", "contact_email", "pain_point", "data_readiness", "urgency", "strategic_fit", "status", "last_contacted", "next_action", "notes"]


def seed_crm_prospects() -> List[Prospect]:
    return [
        Prospect("Sodion Energy", "cathode", "Bangalore", "CTO", "", "", "Need low-cost Na-ion cathode optimized for Indian 45C ambient cycling", "limited cycling CSV", 0.80, 0.95),
        Prospect("Faradion (Reliance)", "cathode", "Sheffield/Mumbai", "Materials R&D Lead", "", "", "Layered oxide fade prediction at scale", "proprietary cycling data", 0.65, 0.85),
        Prospect("Log9 Materials", "bms", "Bangalore", "Battery Systems Lead", "", "", "Thermal incidents in 2W packs during Indian summer", "BMS telemetry", 0.85, 0.78),
        Prospect("Attero Recycling", "recycling", "Noida", "Process R&D Head", "", "", "Black-mass recovery below target, high impurity penalty", "batch leaching logs", 0.80, 0.92),
        Prospect("Amara Raja (ARENERGY)", "cathode", "Tirupati", "New Chemistry Head", "", "", "Evaluating Na-ion for stationary storage", "internal cycling data", 0.55, 0.88),
        Prospect("ISRO VSSC", "bms", "Thiruvananthapuram", "Battery Pack Section", "", "", "Space-grade BMS risk prediction for Li/Na cells", "classified telemetry", 0.40, 0.72),
        Prospect("IIT Madras (Prof X lab)", "cathode", "Chennai", "PI / PhD Student", "", "", "Computational shortlist for publishable Na-ion cathode experiments", "literature + small lab data", 0.50, 0.82),
        Prospect("Tata Chemicals", "recycling", "Pune", "VP Innovation", "", "", "Urban mine feedstock qualification for Na-ion re-synthesis", "assay data", 0.60, 0.80),
        Prospect("Ola Electric", "bms", "Bangalore", "Battery Analytics Lead", "", "", "Field failure prediction for 2W fleet", "fleet telemetry", 0.90, 0.88),
        Prospect("CSIR-CECRI", "cathode", "Karaikudi", "Senior Scientist", "", "", "Collaborative validation of computational cathode screening", "shared lab infrastructure", 0.45, 0.90),
    ]


def build_outreach_scripts() -> List[OutreachScript]:
    return [
        OutreachScript(
            segment="cathode",
            subject_line="Shorten your Na-ion cathode screening cycle from 6 months to 6 weeks",
            opening="We noticed your team is working on sodium-ion layered oxide cathodes for the Indian market.",
            value_prop="KineticsForge is a physics-constrained AI platform that screens 10,000+ Na(Mn,Fe)O2 compositions, predicts 500-cycle fade with uncertainty bounds, and generates lab-ready synthesis protocols - all calibrated to Indian operating conditions (45C ambient, monsoon humidity, INR-first costing).",
            ask="Would a 30-minute technical demo showing inverse-design results for your target chemistry be useful? We can run it on your composition space if you share a target stoichiometry range.",
            social_proof="Our readiness suite passes physics audit, evidence registry, and defect chemistry checks across cathode, BMS, and recycling domains. All predictions carry explicit uncertainty and kill criteria.",
            closing="Happy to send a one-page technical summary first if you prefer reading to calls. Best, [Name]",
        ),
        OutreachScript(
            segment="bms",
            subject_line="Detect battery pack risk 18 minutes before thermal threshold crossing",
            opening="Indian 2W and 3W fleets face unique thermal stress during summer operation. We built a digital twin that assimilates real BMS telemetry and flags risk before standard threshold alarms fire.",
            value_prop="KineticsForge's EKF digital twin tracks SOC, SOH, SEI growth, dendrite index, and core temperature per cell. It produces a risk trajectory with monsoon/hot-zone correction and can flag early precursors to thermal incidents.",
            ask="Could we run a blind test on one month of your pack telemetry and show you the risk timeline?",
            social_proof="The platform's BMS audit checks temperature, voltage, and risk ranges, and validates that risk alerts fire before thermal stress appears in the data.",
            closing="No data leaves your systems during evaluation - we can run on-premise if needed. Best, [Name]",
        ),
        OutreachScript(
            segment="recycling",
            subject_line="Turn your black-mass recovery data into cathode feedstock decisions",
            opening="Recovery yield and impurity control are the two bottlenecks in battery recycling economics. We close the loop from leaching optimization to cathode re-synthesis feasibility.",
            value_prop="KineticsForge's closed-loop optimizer takes your black-mass assay, screens 160 leaching conditions, and ranks recovered streams by which cathode compositions they can actually feed - with purity, cost, and waste tradeoff.",
            ask="If you share a typical black-mass assay and one set of leaching results, we can show a ranked recovery-to-cathode plan within 48 hours.",
            social_proof="The recycling module carries validation gates: ICP-OES recovery confirmation, impurity ratio limits, and first-cycle capacity comparison against virgin feedstock.",
            closing="This is a paid feasibility audit, not a free demo. We believe the results are worth more than the price. Best, [Name]",
        ),
    ]


def write_crm_csv(path: Path, prospects: Optional[List[Prospect]] = None) -> None:
    prospects = prospects or seed_crm_prospects()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CRM_FIELDS)
        writer.writeheader()
        for p in prospects:
            writer.writerow(asdict(p))


def write_outreach_pack(path: Path, scripts: Optional[List[OutreachScript]] = None) -> None:
    scripts = scripts or build_outreach_scripts()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in scripts]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crm-csv", default="data/cache/crm_prospects_india.csv")
    parser.add_argument("--outreach-json", default="data/cache/outreach_scripts_v2.json")
    args = parser.parse_args()
    write_crm_csv(Path(args.crm_csv))
    write_outreach_pack(Path(args.outreach_json))
    print(json.dumps({"crm": args.crm_csv, "outreach": args.outreach_json, "prospects": len(seed_crm_prospects()), "scripts": len(build_outreach_scripts())}, indent=2))


if __name__ == "__main__":
    main()
