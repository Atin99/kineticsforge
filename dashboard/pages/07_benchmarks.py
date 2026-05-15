import streamlit as st
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Benchmarks | KineticsForge", page_icon="📊", layout="wide")

st.markdown("# 📊 Holdout Benchmarks")
st.markdown("Compare KineticsForge against physics-only, exponential fade, random forest, and small MLP baselines.")

try:
    from validation.holdout_benchmarks import run_holdout_benchmark

    if st.button("🔄 Run Benchmark Suite"):
        with st.spinner("Running holdout benchmarks (pure numpy, no GPU)..."):
            report = run_holdout_benchmark()

        st.success(report.summary)

        st.markdown("## Ranking (by MAE)")
        for i, name in enumerate(report.ranking, 1):
            icon = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            highlight = " ← **KineticsForge**" if name == "kineticsforge_v2" else ""
            st.markdown(f"{icon} {name}{highlight}")

        st.markdown("---")
        st.markdown("## Detailed Results")
        all_results = report.baselines + [report.kineticsforge_result]
        cols = st.columns(len(all_results))
        for i, r in enumerate(all_results):
            with cols[i]:
                is_kf = r.name == "kineticsforge_v2"
                st.markdown(f"### {'🔥 ' if is_kf else ''}{r.name}")
                st.metric("MAE", f"{r.mae:.2f}")
                st.metric("MAPE", f"{r.mape:.1f}%")
                st.metric("RMSE", f"{r.rmse:.2f}")
                st.metric("R²", f"{r.r2:.3f}")
                st.caption(r.description)

        st.markdown("---")
        st.markdown("## Holdout Details")
        st.markdown(f"- **Target metric:** {report.target_metric}")
        st.markdown(f"- **Holdout points:** {report.n_holdout}")
        st.markdown("- **Data:** Synthetic compositions with Arrhenius fade + noise (no data leakage)")

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
