import streamlit as st
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Inverse Design | KineticsForge", page_icon="🔬", layout="wide")

st.markdown("# 🔬 Inverse Cathode Design")
st.markdown("Run acquisition-function-guided search over Na(Mn,Fe)O₂ composition space.")

try:
    from modules.cathode.inverse_design import InverseCathodeDesigner, candidate_to_dict, DesignTarget
    from modules.cathode.synthesis_protocol import SynthesisProtocolPlanner, protocol_to_dict
    from modules.cathode.defect_chemistry import DefectChemistryModel
    from business.lab_validation_budget import LabValidationBudgetCalculator

    with st.sidebar:
        st.markdown("### Design Targets")
        min_q0 = st.slider("Min Q₀ (mAh/g)", 100, 200, 145)
        max_fade = st.slider("Max fade @ 500 cycles", 0.05, 0.30, 0.10, step=0.01)
        max_cost = st.slider("Max cost (INR/kWh)", 3000, 12000, 7000, step=500)
        n_return = st.slider("Candidates to return", 3, 15, 8)
        route = st.selectbox("Synthesis route", ["coprecipitation", "solid_state", "sol_gel", "hydrothermal"])

    if st.button("🚀 Run Inverse Design Search"):
        with st.spinner("Searching composition space..."):
            target = DesignTarget(min_q0_mAh_g=float(min_q0), max_fade_500=float(max_fade), max_cost_inr_kwh=float(max_cost))
            designer = InverseCathodeDesigner(target=target)
            candidates = designer.search(n_return=n_return, n_jitter=320)

        st.success(f"Found {len(candidates)} candidates")

        for c in candidates:
            comp = c.composition
            met = c.predicted_metrics
            with st.expander(f"**Rank {c.rank}** — Na{comp.get('Na',1):.2f}Mn{comp.get('Mn',0.5):.2f}Fe{comp.get('Fe',0.5):.2f} — utility={c.utility:.3f}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Q₀ (mAh/g)", f"{met.get('Q0', 0):.1f}")
                col2.metric("Fade@500", f"{met.get('fade_500', 0):.3f}")
                col3.metric("Cost (INR/kWh)", f"{met.get('cost_inr_kwh', 0):.0f}")
                col4.metric("Phase Stability", f"{met.get('phase_stability', 0):.3f}")

                st.markdown("**Constraints:**")
                for cn in c.constraints:
                    icon = "✅" if cn.passed else "❌"
                    st.markdown(f"- {icon} {cn.name}: {cn.value:.3f} (threshold {cn.threshold:.3f}, margin {cn.margin:+.3f})")

                st.markdown("**Rationale:**")
                for r in c.rationale:
                    st.markdown(f"- {r}")

                st.markdown("**Next measurements:**")
                for m in c.next_measurements:
                    st.markdown(f"- {m}")

                st.markdown("**Kill criteria:**")
                for k in c.kill_criteria:
                    st.markdown(f"- {k}")

                # Defect chemistry
                defect = DefectChemistryModel().evaluate(comp)
                st.markdown(f"**Defect tolerance:** {defect.defect_tolerance_score:.3f} | O₂ redox risk: {defect.oxygen_redox_risk:.3f} | JT risk: {defect.jahn_teller_risk:.3f}")

                # Synthesis protocol
                planner = SynthesisProtocolPlanner()
                protocol = planner.build(comp, route=route, target_batch_g=5.0)
                st.markdown(f"**Synthesis:** {protocol.formula} via {protocol.route}")
                st.json(protocol.precursor_masses_g)

        # Lab budget
        st.markdown("---")
        st.markdown("## Lab Validation Budget (Top 3)")
        calc = LabValidationBudgetCalculator()
        top3 = [{"composition": c.composition} for c in candidates[:3]]
        plan = calc.plan_for_candidates(top3, route=route)
        st.metric("Total Budget (INR)", f"₹{plan.total_inr:,.0f}")
        st.metric("Timeline", f"{plan.calendar_weeks} weeks")
        st.markdown(plan.summary)
        for b in plan.candidates:
            st.markdown(f"- **{b.candidate_id}**: ₹{b.total_inr:,.0f} ({b.calendar_days} days)")

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
