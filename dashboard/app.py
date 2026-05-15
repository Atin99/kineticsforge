import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from pathlib import Path

st.set_page_config(page_title="KineticsForge", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'
REAL_DIR = PROJECT_ROOT / 'data' / 'real' / 'scraped'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;500;700&display=swap');
:root {
    --bg: #0A0A0F;
    --accent: #FF6B2B;
    --secondary: #00D4FF;
    --text: #E0E0E0;
    --card-bg: #12121A;
    --border: #1E1E2E;
}
.stApp { background-color: var(--bg); }
h1, h2, h3 { color: var(--accent) !important; font-family: 'Space Grotesk', sans-serif !important; }
p, span, label, div { color: var(--text) !important; font-family: 'Space Grotesk', sans-serif !important; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }
.stButton>button {
    background: linear-gradient(135deg, #FF6B2B, #FF8F5E) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.6rem 1.5rem !important; transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(255,107,43,0.3) !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #00D4FF, #00A8CC) !important;
    box-shadow: 0 4px 15px rgba(0,212,255,0.3) !important;
    transform: translateY(-2px) !important;
}
.stSlider>div>div>div>div { background-color: var(--accent) !important; }
.stRadio>div { gap: 0.5rem !important; }
div[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stProgress > div > div > div > div { background-color: var(--accent) !important; }
div[data-testid="stSidebar"] { background-color: var(--card-bg) !important; border-right: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color: var(--text) !important; font-family: 'Space Grotesk' !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    plot_bgcolor='#0A0A0F',
    paper_bgcolor='#0A0A0F',
    font=dict(color='#E0E0E0', family='Space Grotesk'),
    xaxis=dict(gridcolor='#1E1E2E', zerolinecolor='#1E1E2E'),
    yaxis=dict(gridcolor='#1E1E2E', zerolinecolor='#1E1E2E'),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
)

@st.cache_data(ttl=3600)
def load_system_status():
    status = {"calibration": False, "cathode_cache": False, "bms_cache": False, "recycling_cache": False, "metrics": {}}
    
    priors_path = REAL_DIR / "calibration_priors.json"
    if priors_path.exists():
        try:
            priors = json.loads(priors_path.read_text())
            status["calibration"] = True
            status["calibration_ver"] = priors.get("calibration_version", "unknown")
            status["real_cycles_analyzed"] = priors.get("capacity_fade", {}).get("total_cycles", 0)
        except: pass

    if (CACHE_DIR / "cathode_screening.npz").exists(): status["cathode_cache"] = True
    if (CACHE_DIR / "bms_simulation.npz").exists(): status["bms_cache"] = True
    if (CACHE_DIR / "recycling_optimization.npz").exists(): status["recycling_cache"] = True

    metrics_path = CHECKPOINT_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            if metrics:
                status["metrics"] = metrics[-1]
        except: pass

    return status

st.markdown("# ⚡ KineticsForge")
st.markdown("### Physics-Constrained UDE Platform for Battery Intelligence")
st.warning("🔧 **INTERNAL DEBUG TOOL** — This Streamlit dashboard is for development and demo use only. "
           "The production surface is the FastAPI API at `/predict/lifetime`, `/alert/bms`, `/optimize/recycling`. "
           "See `api/server.py` for the B2B product endpoints.")

status = load_system_status()

with st.sidebar:
    st.markdown("### System Status")
    st.success("✅ Real-Data Calibration loaded") if status["calibration"] else st.warning("⚠️ No real-data calibration found")
    st.success("✅ Cathode Cache loaded") if status["cathode_cache"] else st.warning("⚠️ No cathode cache")
    st.success("✅ BMS Cache loaded") if status["bms_cache"] else st.warning("⚠️ No BMS cache")
    st.success("✅ Recycling Cache loaded") if status["recycling_cache"] else st.warning("⚠️ No Recycling cache")
    
    if status["calibration"]:
        st.markdown("---")
        st.markdown("### Real Data Priors")
        st.markdown(f"**Version:** {status.get('calibration_ver')}")
        st.markdown(f"**Cycles Analyzed:** {status.get('real_cycles_analyzed'):,}")
        
    if status["metrics"]:
        st.markdown("---")
        st.markdown("### Latest Training Metrics")
        m = status["metrics"]
        st.markdown(f"**Task:** {m.get('task')}")
        st.markdown(f"**Epoch:** {m.get('epoch')}")
        st.markdown(f"**MAE:** {m.get('mae', 0):.4f}")
        st.markdown(f"**MAPE:** {m.get('mape', 0):.2f}%")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Compositions Evaluated", "10,000+", "qNEHVI Bayesian Pareto")
with col2:
    st.metric("TGN BMS Alert Lead", "scaffold", "needs Kaggle training")
with col3:
    st.metric("Bayesian Recycling", "closed-loop", "priors update from outcomes")

st.markdown("---")
st.markdown("### Navigate to modules using the sidebar →")
st.markdown("""
- **Cathode Discovery Lab** — qNEHVI Bayesian Pareto screening, UDE degradation physics, Uncertainty Quantification
- **BMS Edge Intelligence** — Temporal Graph Network pack monitoring, Multi-scale precursor detection, EIS features
- **Recycling Optimizer** — Bayesian closed-loop optimization, Stochastic black-mass blending, Shrinking-core leaching ODEs
""")
