"""FastAPI backend for KineticsForge.

Decouples the engineering API from Streamlit state so it can serve
the web frontend, CLI tools, or third-party integrations.

Run with: uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn", file=sys.stderr)
    raise

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.india_context import IndiaOperatingContext, money_fields_from_usd
from core.evidence_registry import build_registry_from_project, default_claims, EvidenceRegistry
from core.dimensional_analysis import BatteryParameterSanity, checks_to_dict
from modules.cathode.screener import screen_compositions
from modules.cathode.inverse_design import InverseCathodeDesigner, candidate_to_dict
from modules.cathode.synthesis_protocol import SynthesisProtocolPlanner, protocol_to_dict
from modules.cathode.defect_chemistry import DefectChemistryModel
from modules.full_cell.cell_architect import FullCellArchitect, architecture_to_dict
from modules.bms.digital_twin_assimilation import BatteryDigitalTwin, simulate_telemetry
from modules.recycling.closed_loop_optimizer import ClosedLoopOptimizer, plan_to_dict as recycling_plan_to_dict
from business.pilot_contract import build_target_account_playbook, generate_90_day_plan
from business.lab_validation_budget import LabValidationBudgetCalculator, plan_to_dict as budget_plan_to_dict
from validation.holdout_benchmarks import run_holdout_benchmark, report_to_dict
from api.auth import configured_api_token
from api.rate_limiter import limiter


app = FastAPI(
    title="KineticsForge API",
    description="Battery physics workbench for cathode screening, pack risk, and recycling optimization",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def product_guards(request: Request, call_next):
    if request.url.path != "/health":
        try:
            limiter.check(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        expected = configured_api_token()
        if expected:
            authorization = request.headers.get("authorization", "")
            if not authorization.lower().startswith("bearer "):
                return JSONResponse(status_code=401, content={"detail": "Bearer token required"})
            if authorization.split(" ", 1)[1].strip() != expected:
                return JSONResponse(status_code=403, content={"detail": "Invalid API token"})
    return await call_next(request)


# ---- Request/Response models ----

class CompositionInput(BaseModel):
    Na: float = 1.0
    Mn: float = 0.5
    Fe: float = 0.5
    dopant: Optional[str] = None
    dopant_frac: float = 0.0


class ScreenRequest(BaseModel):
    n: int = 100
    temperature_K: float = 318.0


class InverseDesignRequest(BaseModel):
    n_return: int = 8
    n_jitter: int = 320


class TwinRequest(BaseModel):
    n_frames: int = 240
    inject_fault: bool = True


class RecyclingRequest(BaseModel):
    top_n: int = 6
    feedstock_mass_kg: float = 100.0


class LifetimeRequest(BaseModel):
    composition: CompositionInput = Field(default_factory=CompositionInput)
    temperature_K: float = 318.0
    cycles: int = 500


class BMSAlertRequest(BaseModel):
    seed: int = 42
    inject_failure: bool = True
    n_cells: int = 8
    duration_seconds: int = 3600


class UIScreenRequest(BaseModel):
    na: float = 1.0
    mn: float = 0.5
    fe: float = 0.5
    al_doped: bool = False
    ti_doped: bool = False
    temperature_K: float = 318.15
    upper_voltage: float = 4.10
    ehull_slope: float = 20.0
    w_capacity: float = 0.32
    w_stability: float = 0.32
    w_fade: float = 0.22
    w_cost: float = 0.14
    charge_penalty: float = 0.10
    defect_penalty: float = 0.06


# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status": "ok", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


@app.get("/india-context")
def india_context():
    ctx = IndiaOperatingContext.from_env()
    return ctx.to_dict()


@app.post("/cathode/screen")
def cathode_screen(req: ScreenRequest):
    results = screen_compositions(n=min(req.n, 500), T=req.temperature_K)
    return {"count": len(results), "results": results[:50]}


@app.post("/cathode/inverse-design")
def cathode_inverse_design(req: InverseDesignRequest):
    designer = InverseCathodeDesigner()
    candidates = designer.search(n_return=min(req.n_return, 20), n_jitter=min(req.n_jitter, 600))
    return {"count": len(candidates), "candidates": [candidate_to_dict(c) for c in candidates]}


@app.post("/cathode/defect-chemistry")
def cathode_defect_chemistry(comp: CompositionInput):
    model = DefectChemistryModel()
    result = model.evaluate(comp.dict())
    from dataclasses import asdict
    return asdict(result)


@app.post("/cathode/synthesis-protocol")
def cathode_synthesis_protocol(comp: CompositionInput, route: str = "coprecipitation", batch_g: float = 5.0):
    planner = SynthesisProtocolPlanner()
    protocol = planner.build(comp.dict(), route=route, target_batch_g=batch_g)
    return protocol_to_dict(protocol)


@app.post("/full-cell/architect")
def full_cell_architect(comp: CompositionInput, chemistry: str = "Na-ion"):
    architect = FullCellArchitect(chemistry=chemistry)
    return architecture_to_dict(architect.design({"comp": comp.dict()}))


@app.post("/bms/digital-twin")
def bms_digital_twin(req: TwinRequest):
    twin = BatteryDigitalTwin()
    frames = simulate_telemetry(n=min(req.n_frames, 1000), inject_fault=req.inject_fault)
    result = twin.ingest(frames)
    return {
        "frames": len(result["trajectory"]),
        "alerts": result["alerts"],
        "final_state": result["final_state"],
        "trajectory_summary": {
            "risk_max": max(r["risk"] for r in result["trajectory"]),
            "temp_max_K": max(r["core_temperature_K"] for r in result["trajectory"]),
        },
    }


@app.post("/recycling/closed-loop")
def recycling_closed_loop(req: RecyclingRequest):
    from modules.recycling.closed_loop_optimizer import FeedstockAssay
    feedstock = FeedstockAssay(mass_kg=req.feedstock_mass_kg)
    optimizer = ClosedLoopOptimizer(feedstock=feedstock)
    plans = optimizer.rank_cathode_loop(top_n=min(req.top_n, 15))
    return {"count": len(plans), "plans": [recycling_plan_to_dict(p) for p in plans]}


@app.post("/predict/lifetime")
def predict_lifetime(req: LifetimeRequest):
    import torch
    from core.composition_embedder import parse_composition
    from modules.cathode.degradation_ode import CathodeDegradationSimulator, DegradationVectorField

    comp_dict = req.composition.dict()
    comp_vec = parse_composition(comp_dict).unsqueeze(0)
    model = DegradationVectorField()
    sim = CathodeDegradationSimulator(model, solver="rk4", dt=0.01, max_cycles=min(req.cycles, 1000))
    with torch.no_grad():
        result = sim.simulate(comp_vec, T=req.temperature_K, n_cycles=min(req.cycles, 1000))
    fade = float(result["fade_pct"])
    uncertainty = {
        "lower_fade_pct": max(0.0, fade - 0.035),
        "upper_fade_pct": min(1.0, fade + 0.060),
        "basis": "untrained UDE simulation interval; not a training-quality claim",
    }
    return {
        "result": {
            "cycles": int(min(req.cycles, 1000)),
            "capacity_start": float(result["capacity"][0]),
            "capacity_end": float(result["capacity"][-1]),
            "fade_pct": fade,
        },
        "uncertainty": uncertainty,
        "provenance": {
            "model": "Na-ion UDE physics plus neural residual scaffold",
            "local_training": False,
            "claim": "simulation-backed",
        },
    }


@app.post("/alert/bms")
def alert_bms(req: BMSAlertRequest):
    from modules.bms.precursor_detector import run_drive_cycle

    duration = min(max(int(req.duration_seconds), 60), 7200)
    result = run_drive_cycle(seed=req.seed, inject_failure=req.inject_failure, n_cells=min(req.n_cells, 32), duration_seconds=duration)
    lead_steps = [a.get("lead_steps", 0) for a in result["alerts"]]
    return {
        "result": {
            "alert_fired": bool(result["alert_fired"]),
            "n_alerts": int(result["n_alerts"]),
            "alert_cells": result["alert_cells"],
            "lead_steps_min": int(min(lead_steps)) if lead_steps else None,
            "alerts": result["alerts"][:20],
        },
        "uncertainty": {
            "basis": "smoke simulator output; BMS safety claim remains blocked until positive real benchmark lead time and zero missed failures",
        },
        "provenance": {
            "model": "multi-scale precursor detector with TGN-ready graph code",
            "local_training": False,
            "claim": "simulation-backed",
        },
    }


@app.post("/optimize/recycling")
def optimize_recycling(req: RecyclingRequest):
    response = recycling_closed_loop(req)
    return {
        "result": response,
        "uncertainty": {
            "basis": "Bayesian recovery priors can update from observed outcomes; current response uses prior means plus physics heuristics",
        },
        "provenance": {
            "model": "Bayesian closed-loop recycling optimizer",
            "local_training": False,
            "claim": "simulation-backed",
        },
    }


@app.get("/evidence/registry")
def evidence_registry():
    reg = build_registry_from_project(PROJECT_ROOT)
    claims = default_claims(reg)
    return {
        "sources": len(reg.sources),
        "records": len(reg.records),
        "claims": [c.__dict__ for c in claims],
    }


@app.get("/validation/benchmarks")
def validation_benchmarks():
    report = run_holdout_benchmark()
    return report_to_dict(report)


@app.get("/business/pilot-playbook")
def pilot_playbook():
    return {
        "target_accounts": build_target_account_playbook(),
        "ninety_day_plan": generate_90_day_plan(),
    }


@app.get("/business/lab-budget")
def lab_budget():
    candidates = [
        {"composition": {"Na": 1.02, "Mn": 0.48, "Fe": 0.47, "dopant": "Al", "dopant_frac": 0.05}},
        {"composition": {"Na": 0.98, "Mn": 0.55, "Fe": 0.40, "dopant": "Ti", "dopant_frac": 0.03}},
        {"composition": {"Na": 1.05, "Mn": 0.42, "Fe": 0.52, "dopant": None, "dopant_frac": 0.0}},
    ]
    from business.lab_validation_budget import build_validation_plan_from_inverse_design
    plan = build_validation_plan_from_inverse_design(candidates)
    return budget_plan_to_dict(plan)


@app.post("/unit-check/cathode")
def unit_check_cathode(params: Dict[str, float]):
    sanity = BatteryParameterSanity()
    return {"checks": checks_to_dict(sanity.cathode(params))}


@app.post("/unit-check/bms")
def unit_check_bms(params: Dict[str, float]):
    sanity = BatteryParameterSanity()
    return {"checks": checks_to_dict(sanity.bms(params))}


@app.post("/unit-check/recycling")
def unit_check_recycling(params: Dict[str, float]):
    sanity = BatteryParameterSanity()
    return {"checks": checks_to_dict(sanity.recycling(params))}


@app.get("/money/convert")
def money_convert(usd: float = 100.0, include_gst: bool = False):
    ctx = IndiaOperatingContext.from_env()
    return money_fields_from_usd(usd, ctx)


@app.get("/models/registry")
def model_registry_list():
    from core.model_registry import ModelRegistry
    registry = ModelRegistry(PROJECT_ROOT)
    return {"models": registry.summary(), "total": len(registry.summary())}


@app.get("/api/models")
def api_model_registry_list():
    return model_registry_list()


@app.post("/api/predict/degradation")
def api_predict_degradation(req: LifetimeRequest):
    return predict_lifetime(req)


@app.post("/api/simulate/bms")
def api_simulate_bms(req: BMSAlertRequest):
    return alert_bms(req)


@app.post("/api/optimize/recycling")
def api_optimize_recycling(req: RecyclingRequest):
    return optimize_recycling(req)


@app.post("/api/screen/cathode")
def api_screen_cathode(req: UIScreenRequest):
    from core.materials_physics import score_composition as score_material_composition, screen_batch

    knobs = {
        "upper_voltage": req.upper_voltage,
        "ehull_slope": req.ehull_slope,
        "w_capacity": req.w_capacity,
        "w_stability": req.w_stability,
        "w_fade": req.w_fade,
        "w_cost": req.w_cost,
        "charge_penalty": req.charge_penalty,
        "defect_penalty": req.defect_penalty,
    }
    comp = {
        "Na": req.na,
        "Mn": req.mn,
        "Fe": req.fe,
        "al": req.al_doped,
        "ti": req.ti_doped,
    }
    predicted = score_material_composition(comp, temp_K=req.temperature_K, knobs=knobs)
    candidates = screen_batch(n=120, temp_K=req.temperature_K, knobs=knobs)[:80]
    return {"predicted": predicted, "candidates": candidates, "claim": "simulation-backed"}


@app.get("/features/catalog")
def feature_store_catalog():
    from core.feature_store import FeatureStore
    store = FeatureStore(PROJECT_ROOT)
    return {"feature_sets": store.summary(), "total": len(store.summary())}


@app.get("/climate/compare")
def climate_compare():
    from core.regional_climate import compare_regions_summary
    return {"regions": compare_regions_summary(), "source": "synthetic_climatology_fallback", "note": "Use NASA POWER or ERA5 for deployment claims."}
