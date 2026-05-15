import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "data" / "cache" / "v2_readiness_report.json"


st.set_page_config(page_title="KineticsForge V2 Readiness", layout="wide")
st.title("V2 Readiness")

if not REPORT_PATH.exists():
    st.warning("Run `python -m validation.v2_readiness_report --project-root .` to generate the readiness report.")
    st.stop()

payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
readiness = payload.get("readiness", {})
left, mid, right = st.columns(3)
left.metric("Physics Audit Score", f"{float(readiness.get('score', 0.0)):.2f}")
mid.metric("Evidence Records", int(payload.get("evidence", {}).get("records", 0)))
right.metric("Findings", len(readiness.get("findings", [])))

st.subheader("Top Cathode Candidates")
rows = []
for item in payload.get("inverse_design_top5", []):
    comp = item.get("composition", {})
    metrics = item.get("predicted_metrics", {})
    rows.append({
        "rank": item.get("rank"),
        "Na": comp.get("Na"),
        "Mn": comp.get("Mn"),
        "Fe": comp.get("Fe"),
        "dopant": comp.get("dopant") or "None",
        "Q0_mAh_g": metrics.get("Q0"),
        "fade_500": metrics.get("fade_500"),
        "cost_inr_kwh": metrics.get("cost_inr_kwh"),
        "india_cost_index": metrics.get("cost_index_india"),
        "utility": item.get("utility"),
    })
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Synthesis Protocol")
protocol = payload.get("synthesis_protocol", {})
st.write({"formula": protocol.get("formula"), "route": protocol.get("route"), "target_batch_g": protocol.get("target_batch_g")})
step_rows = []
for step in protocol.get("process_steps", []):
    step_rows.append({
        "step": step.get("step"),
        "operation": step.get("operation"),
        "setpoints": json.dumps(step.get("setpoints", {})),
        "gate": step.get("acceptance_gate"),
        "failure_response": step.get("failure_response"),
    })
if step_rows:
    st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

st.subheader("Defect Chemistry")
defect = payload.get("defect_chemistry_top_candidate", {})
if defect:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Defect Tolerance", f"{float(defect.get('defect_tolerance_score', 0.0)):.2f}")
    d2.metric("Oxygen Redox Risk", f"{float(defect.get('oxygen_redox_risk', 0.0)):.2f}")
    d3.metric("TM Mixing Risk", f"{float(defect.get('transition_metal_mixing_risk', 0.0)):.2f}")
    d4.metric("Moisture Sensitivity", f"{float(defect.get('moisture_sensitivity', 0.0)):.2f}")
    st.write(defect.get("suggested_compensation", []))

st.subheader("Unit Sanity")
unit_rows = []
for domain, checks in payload.get("unit_sanity", {}).items():
    for check in checks:
        row = dict(check)
        row["domain"] = domain
        unit_rows.append(row)
if unit_rows:
    st.dataframe(pd.DataFrame(unit_rows), use_container_width=True, hide_index=True)

st.subheader("Closed-Loop Recycling Plans")
loop_rows = []
for plan in payload.get("closed_loop_top4", []):
    econ = plan.get("economics", {})
    comp = plan.get("target_composition", {})
    condition = plan.get("condition", {})
    loop_rows.append({
        "Mn": comp.get("Mn"),
        "Fe": comp.get("Fe"),
        "dopant": comp.get("dopant") or "None",
        "T_C": condition.get("temperature_K", 0) - 273.15,
        "pH": condition.get("pH"),
        "acid_M": condition.get("acid_M"),
        "makeable_kg": plan.get("makeable_mass_kg"),
        "limiting": plan.get("limiting_element"),
        "process_cost_inr": econ.get("process_cost_inr"),
    })
if loop_rows:
    st.dataframe(pd.DataFrame(loop_rows), use_container_width=True, hide_index=True)

st.subheader("Evidence Claims")
claim_rows = payload.get("evidence", {}).get("claims", [])
if claim_rows:
    st.dataframe(pd.DataFrame(claim_rows), use_container_width=True, hide_index=True)

st.subheader("Critical Caveat")
st.write("Simulation-backed results choose experiments. They do not replace wet-lab validation, AIS certification, or supplier quotations.")
