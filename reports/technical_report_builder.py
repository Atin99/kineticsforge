import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def build_markdown_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# KineticsForge V2 Technical Readiness Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append("")
    readiness = payload.get("readiness", {})
    lines.append("## System Readiness")
    lines.append("")
    lines.append(f"- Passed: {readiness.get('passed')}")
    lines.append(f"- Score: {readiness.get('score')}")
    lines.append(f"- Findings: {len(readiness.get('findings', []))}")
    lines.append("")
    if readiness.get("findings"):
        lines.append("## Findings")
        lines.append("")
        for item in readiness["findings"][:20]:
            lines.append(f"- {item.get('severity', 'unknown')}: {item.get('check_id')} - {item.get('message')}")
        lines.append("")
    lines.append("## Evidence")
    lines.append("")
    evidence = payload.get("evidence", {})
    lines.append(f"- Sources: {evidence.get('sources', 0)}")
    lines.append(f"- Records: {evidence.get('records', 0)}")
    for claim in evidence.get("claims", []):
        lines.append(f"- Claim verdict: {claim.get('verdict')} - {claim.get('claim')}")
    lines.append("")
    lines.append("## Top Inverse-Design Candidates")
    lines.append("")
    for item in payload.get("inverse_design_top5", [])[:5]:
        comp = item.get("composition", {})
        metrics = item.get("predicted_metrics", {})
        lines.append(
            f"- Rank {item.get('rank')}: Na={comp.get('Na'):.3f}, Mn={comp.get('Mn'):.3f}, "
            f"Fe={comp.get('Fe'):.3f}, dopant={comp.get('dopant')} {comp.get('dopant_frac'):.3f}; "
            f"Q0={metrics.get('Q0'):.1f} mAh/g, fade500={metrics.get('fade_500'):.3f}, "
            f"cost={metrics.get('cost_inr_kwh', 0.0):.0f} INR/kWh"
        )
    lines.append("")
    lines.append("## Synthesis Protocol Snapshot")
    lines.append("")
    protocol = payload.get("synthesis_protocol", {})
    lines.append(f"- Formula: {protocol.get('formula')}")
    lines.append(f"- Route: {protocol.get('route')}")
    lines.append(f"- Target batch: {protocol.get('target_batch_g')} g")
    for step in protocol.get("process_steps", [])[:6]:
        lines.append(f"- Step {step.get('step')}: {step.get('operation')} | gate: {step.get('acceptance_gate')}")
    lines.append("")
    lines.append("## Defect Chemistry Snapshot")
    lines.append("")
    defect = payload.get("defect_chemistry_top_candidate", {})
    if defect:
        lines.append(f"- Defect tolerance score: {_fmt(defect.get('defect_tolerance_score'))}")
        lines.append(f"- Oxygen redox risk: {_fmt(defect.get('oxygen_redox_risk'))}")
        lines.append(f"- Transition-metal mixing risk: {_fmt(defect.get('transition_metal_mixing_risk'))}")
        lines.append(f"- Moisture sensitivity: {_fmt(defect.get('moisture_sensitivity'))}")
        for action in defect.get("suggested_compensation", [])[:4]:
            lines.append(f"- Compensation: {action}")
    lines.append("")
    lines.append("## Unit Sanity")
    lines.append("")
    for domain, checks in payload.get("unit_sanity", {}).items():
        failed = [c for c in checks if not c.get("passed")]
        lines.append(f"- {domain}: {len(checks) - len(failed)}/{len(checks)} checks passed")
        for item in failed[:3]:
            lines.append(f"  - {item.get('message')}")
    lines.append("")
    lines.append("## Lab Validation Budget")
    lines.append("")
    budget = payload.get("lab_validation_budget", {})
    if budget:
        lines.append(f"- Total: INR {budget.get('total_inr', 0):,.0f} (~USD {budget.get('total_usd', 0):,.0f})")
        lines.append(f"- Timeline: {budget.get('calendar_weeks', 0)} weeks")
        lines.append(f"- Summary: {budget.get('summary', '')}")
        for cand in budget.get("candidates", [])[:3]:
            lines.append(f"  - {cand.get('candidate_id')}: INR {cand.get('total_inr', 0):,.0f} ({cand.get('calendar_days', 0)} days)")
        for a in budget.get("assumptions", [])[:4]:
            lines.append(f"  - Assumption: {a}")
    lines.append("")
    lines.append("## Closed-Loop Recycling")
    lines.append("")
    for plan in payload.get("closed_loop_top4", [])[:4]:
        econ = plan.get("economics", {})
        comp = plan.get("target_composition", {})
        lines.append(
            f"- Target Mn={comp.get('Mn'):.3f}, Fe={comp.get('Fe'):.3f}; "
            f"makeable={plan.get('makeable_mass_kg'):.2f} kg; limiting={plan.get('limiting_element')}; "
            f"process cost={econ.get('process_cost_inr', 0.0):.0f} INR"
        )
    lines.append("")
    lines.append("## Startup Pilot")
    lines.append("")
    startup = payload.get("startup", {})
    for account in startup.get("target_accounts", [])[:3]:
        customer = account.get("customer", {})
        lines.append(f"- {customer.get('company_name')}: {account.get('offer_name')} at {account.get('price_inr'):.0f} INR")
    crm_count = startup.get("crm_prospects_count", 0)
    if crm_count:
        lines.append(f"- CRM prospect pipeline: {crm_count} companies seeded")
    lines.append("")
    lines.append("## Non-Negotiable Caveat")
    lines.append("")
    lines.append("Synthetic and simulation-backed outputs are not experimental validation. Use this report to choose experiments, not to claim final battery performance.")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cache/v2_readiness_report.json")
    parser.add_argument("--out", default="data/cache/v2_technical_report.md")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    text = build_markdown_report(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print({"out": str(out), "lines": len(text.splitlines())})


if __name__ == "__main__":
    main()
