import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from business.pilot_contract import build_target_account_playbook, generate_90_day_plan
from business.lab_validation_budget import LabValidationBudgetCalculator, plan_to_dict as budget_plan_to_dict
from business.customer_discovery import seed_crm_prospects
from core.dimensional_analysis import BatteryParameterSanity, checks_to_dict
from core.evidence_registry import build_registry_from_project, default_claims
from core.physics_audit import PhysicsAuditSuite
from modules.bms.digital_twin_assimilation import BatteryDigitalTwin, simulate_telemetry
from modules.cathode.defect_chemistry import DefectChemistryModel
from modules.cathode.inverse_design import InverseCathodeDesigner, candidate_to_dict
from modules.cathode.screener import screen_compositions
from modules.cathode.synthesis_protocol import SynthesisProtocolPlanner, protocol_to_dict
from modules.recycling.closed_loop_optimizer import ClosedLoopOptimizer, plan_to_dict


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_industrial_report(project_root: Path) -> Dict[str, Any]:
    runs = project_root / "training" / "colab_kaggle" / "runs"
    if not runs.exists():
        return {}
    reports = sorted(runs.glob("industrial_*/*industrial_training_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return load_json(reports[0]) if reports else {}


def contains_stale_attempt_path(payload: Any) -> bool:
    stale = "attempt 1(antigravity gemini)"
    if isinstance(payload, dict):
        return any(contains_stale_attempt_path(v) for v in payload.values())
    if isinstance(payload, list):
        return any(contains_stale_attempt_path(v) for v in payload)
    return stale in str(payload)


def issue_count(report: Dict[str, Any], severity: str) -> int:
    total = 0
    for contract in report.get("contracts", {}).values():
        for issue in contract.get("issues", []):
            if issue.get("severity") == severity:
                total += 1
    return total


def build_claim_gates(project_root: Path, simulation_passed: bool) -> Dict[str, Any]:
    real_manifest = load_json(project_root / "data" / "real" / "assembled" / "real_dataset_manifest.json")
    hyper_manifest = load_json(project_root / "data" / "cache" / "hyper_manifest_foundation.json")
    quality = load_json(project_root / "data" / "cache" / "data_quality_report_foundation.json")
    curation = load_json(project_root / "data" / "real" / "scraped" / "curation_report.json")
    real_holdout = load_json(project_root / "data" / "cache" / "real_holdout_benchmark.json")
    industrial = latest_industrial_report(project_root)

    real_metrics = real_manifest.get("metrics", {})
    accepted_literature = int(curation.get("accepted_rows", real_metrics.get("literature_rows_accepted", 0)) or 0)
    critical_contract_issues = issue_count(industrial, "critical")
    bms_baseline = industrial.get("baselines", {}).get("bms", {})
    recycling_baseline = industrial.get("baselines", {}).get("recycling", {})
    stale_paths = contains_stale_attempt_path(real_manifest) or contains_stale_attempt_path(hyper_manifest)
    gates = {
        "simulation_smoke_ready": {
            "passed": bool(simulation_passed),
            "detail": "Synthetic integration smoke checks passed." if simulation_passed else "Synthetic integration smoke checks failed.",
        },
        "real_dataset_ready": {
            "passed": real_manifest.get("status") == "pass" and int(real_metrics.get("total_real_rows", 0) or 0) >= 100000,
            "total_real_rows": int(real_metrics.get("total_real_rows", 0) or 0),
            "minimum_total_real_rows": 100000,
        },
        "literature_scale_ready": {
            "passed": accepted_literature >= 500,
            "accepted_rows": accepted_literature,
            "minimum_for_training_claims": 500,
            "detail": "Current literature rows can calibrate priors but are not enough for independent model training.",
        },
        "provenance_relocated": {
            "passed": not stale_paths,
            "detail": "No cached manifest should point to an old attempt directory.",
        },
        "foundation_data_quality_ready": {
            "passed": quality.get("status") == "pass",
            "status": quality.get("status", "missing"),
            "errors": int(quality.get("errors", 0) or 0),
            "warnings": int(quality.get("warnings", 0) or 0),
        },
        "training_contract_ready": {
            "passed": bool(industrial) and critical_contract_issues == 0,
            "critical_issues": critical_contract_issues,
        },
        "recycling_baseline_ready": {
            "passed": bool(recycling_baseline) and not bool(recycling_baseline.get("skipped")),
            "baseline": recycling_baseline,
        },
        "bms_alert_ready": {
            "passed": bool(bms_baseline) and float(bms_baseline.get("mean_lead_steps", -1.0)) > 0.0 and float(bms_baseline.get("missed_failures", 1.0)) == 0.0,
            "baseline": bms_baseline,
            "detail": "A safety claim needs positive lead time and zero missed failures on the smoke set.",
        },
        "real_holdout_benchmark_ready": {
            "passed": real_holdout.get("status") == "pass",
            "quality": real_holdout.get("quality", "missing"),
            "holdout_metrics": real_holdout.get("holdout_metrics", {}),
        },
        "real_holdout_model_quality_ready": {
            "passed": real_holdout.get("quality") == "real_holdout_baseline_pass",
            "quality": real_holdout.get("quality", "missing"),
            "detail": "This gate is intentionally stricter than data loading.",
        },
    }
    required_for_claims = [
        "simulation_smoke_ready",
        "real_dataset_ready",
        "provenance_relocated",
        "foundation_data_quality_ready",
        "training_contract_ready",
        "recycling_baseline_ready",
        "real_holdout_benchmark_ready",
    ]
    required_for_training_claims = required_for_claims + ["literature_scale_ready", "bms_alert_ready", "real_holdout_model_quality_ready"]
    gates["claim_status"] = {
        "simulation_backed_claims_allowed": all(gates[k]["passed"] for k in required_for_claims),
        "training_quality_claims_allowed": all(gates[k]["passed"] for k in required_for_training_claims),
        "required_for_training_claims": required_for_training_claims,
    }
    return gates


def build_v2_readiness_report(project_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    audit = PhysicsAuditSuite()
    reports = []
    cathode_results = screen_compositions(n=100, T=318)
    reports.append(audit.audit_cathode_screening(cathode_results))
    top_curve_cycles = np.arange(0, 501)
    top = cathode_results[0]
    defect = DefectChemistryModel().evaluate(top["comp"])
    unit_sanity = BatteryParameterSanity()
    top_curve = top["Q0"] * (1.0 - top["fade_500"] * (top_curve_cycles / 500.0) ** 1.45)
    reports.append(audit.audit_capacity_curve(top_curve_cycles, top_curve, subject="top_candidate_curve"))
    designer = InverseCathodeDesigner()
    inverse_candidates = designer.search(n_return=5, n_jitter=220)
    planner = SynthesisProtocolPlanner()
    protocol = planner.build(inverse_candidates[0].composition, route="coprecipitation", target_batch_g=5.0)
    twin = BatteryDigitalTwin()
    twin_result = twin.ingest(simulate_telemetry(n=180, inject_fault=True))
    history = {
        "risk": np.array([[row["risk"]] for row in twin_result["trajectory"]], dtype=float),
        "T_cells": np.array([[row["core_temperature_K"]] for row in twin_result["trajectory"]], dtype=float),
        "V_cells": np.array([[3.2] for _ in twin_result["trajectory"]], dtype=float),
    }
    reports.append(audit.audit_bms_history(history, twin_result["alerts"]))
    loop_optimizer = ClosedLoopOptimizer()
    closed_loop = loop_optimizer.rank_cathode_loop(top_n=4)
    reports.append(audit.audit_recycling_solution(asdict_condition_solution(closed_loop[0]), None))
    combined = audit.merge_reports(reports, subject="kineticsforge_v2_readiness")
    registry = build_registry_from_project(project_root)
    claim_assessments = default_claims(registry)

    # Lab validation budget for top 3 inverse-design candidates
    budget_calc = LabValidationBudgetCalculator()
    budget_input = [{"composition": c.composition} for c in inverse_candidates[:3]]
    lab_budget = budget_calc.plan_for_candidates(budget_input)

    # CRM prospect count
    crm_count = len(seed_crm_prospects())
    claim_gates = build_claim_gates(project_root, combined.passed)

    return {
        "readiness": combined.to_dict(),
        "claim_gates": claim_gates,
        "evidence": {
            "sources": len(registry.sources),
            "records": len(registry.records),
            "claims": [c.__dict__ for c in claim_assessments],
        },
        "inverse_design_top5": [candidate_to_dict(c) for c in inverse_candidates],
        "synthesis_protocol": protocol_to_dict(protocol),
        "defect_chemistry_top_candidate": defect.__dict__,
        "unit_sanity": {
            "cathode": checks_to_dict(unit_sanity.cathode({
                "capacity_mAh_g": top["Q0"],
                "temperature_C": 45.0,
                "fade_fraction": top["fade_500"],
                "cost_inr_kwh": top.get("cost_inr_kwh", 7000.0),
            })),
            "bms": checks_to_dict(unit_sanity.bms({
                "voltage_V": 3.2,
                "temperature_C": 45.0,
                "current_A": 5.0,
                "risk": max(row["risk"] for row in twin_result["trajectory"]),
            })),
            "recycling": checks_to_dict(unit_sanity.recycling({
                "temperature_C": closed_loop[0].condition.temperature_K - 273.15,
                "time_min": closed_loop[0].condition.time_min,
                "particle_um": closed_loop[0].condition.particle_um,
                "recovery": closed_loop[0].stream.recovery_score,
            })),
        },
        "digital_twin": {
            "frames": len(twin_result["trajectory"]),
            "alerts": twin_result["alerts"],
            "final_state": twin_result["final_state"],
        },
        "closed_loop_top4": [plan_to_dict(p) for p in closed_loop],
        "lab_validation_budget": budget_plan_to_dict(lab_budget),
        "startup": {
            "target_accounts": build_target_account_playbook(),
            "ninety_day_plan": generate_90_day_plan(),
            "crm_prospects_count": crm_count,
        },
    }


def asdict_condition_solution(plan: Any) -> Dict[str, float]:
    recovery = float(plan.stream.recovery_score)
    return {
        "T": float(plan.condition.temperature_K),
        "pH": float(plan.condition.pH),
        "conc": float(plan.condition.acid_M),
        "t": float(plan.condition.time_min),
        "alpha_Mn": min(1.0, recovery + 0.05),
        "alpha_Fe": min(1.0, recovery),
        "alpha_Na": min(1.0, recovery + 0.08),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="data/cache/v2_readiness_report.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    payload = build_v2_readiness_report(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": payload["readiness"]["passed"],
        "score": payload["readiness"]["score"],
        "simulation_backed_claims_allowed": payload["claim_gates"]["claim_status"]["simulation_backed_claims_allowed"],
        "training_quality_claims_allowed": payload["claim_gates"]["claim_status"]["training_quality_claims_allowed"],
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
