import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import os

st.set_page_config(page_title="Recycling Optimizer", layout="wide")
st.title("Recycling Optimizer")

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'cache'
cache_file = CACHE_DIR / 'recycling_optimization.npz'

PLOT_LAYOUT = dict(
    plot_bgcolor='#0A0A0F', paper_bgcolor='#0A0A0F',
    font=dict(color='#E0E0E0', family='Space Grotesk'),
    xaxis=dict(gridcolor='#1E1E2E', zerolinecolor='#1E1E2E'), 
    yaxis=dict(gridcolor='#1E1E2E', zerolinecolor='#1E1E2E'),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(font=dict(color='#E0E0E0'), bgcolor='rgba(0,0,0,0)')
)

@st.cache_data
def load_recycling_data():
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file, allow_pickle=True)
        pareto_raw = data['pareto_front']
        if hasattr(pareto_raw, 'item'): pareto_raw = pareto_raw.item()
        
        # Ensure it's a list of dicts for the dataframe
        if isinstance(pareto_raw, dict) and 'pareto_front' in pareto_raw:
            pareto_list = pareto_raw['pareto_front']
        elif isinstance(pareto_raw, list):
            pareto_list = pareto_raw
        else:
            pareto_list = [pareto_raw] # Fallback
            
        df = pd.DataFrame(pareto_list)
        return df
    except Exception as e:
        st.error(f"Error loading cache: {e}")
        return None

df = load_recycling_data()

if df is None or df.empty:
    st.warning("Cache not found or empty. Please run the precompute pipeline first.")
    st.stop()

st.markdown("### NSGA-II Multi-Objective Leaching Optimization")
st.markdown("The bayesian optimizer balances 3 competing objectives: Maximizing **Recovery**, Minimizing **Cost** (Reagent/Energy/Time), and Minimizing **Impurity Extraction** (Al/Cu/Co).")

# Parallel Coordinates Plot
st.markdown("#### High-Dimensional Pareto Front (Parallel Coordinates)")

# Normalize variables for better parallel coords display
color_col = 'recovery' if 'recovery' in df.columns else df.columns[0]

fig_par = go.Figure(data=
    go.Parcoords(
        line = dict(color = df[color_col],
                   colorscale = 'Inferno',
                   showscale = True,
                   cmin = df[color_col].min(),
                   cmax = df[color_col].max()),
        dimensions = list([
            dict(range = [df['T'].min(), df['T'].max()],
                 label = 'Temperature (K)', values = df['T']),
            dict(range = [df['pH'].min(), df['pH'].max()],
                 label = 'pH', values = df['pH']),
            dict(range = [df['conc'].min(), df['conc'].max()],
                 label = 'Acid Conc (M)', values = df['conc']),
            dict(range = [df['t'].min(), df['t'].max()],
                 label = 'Duration (min)', values = df['t']),
            dict(range = [df['recovery'].min(), df['recovery'].max()],
                 label = 'Recovery %', values = df['recovery']),
            dict(range = [df['cost'].min(), df['cost'].max()],
                 label = 'Cost Penalty', values = df['cost']),
            dict(range = [df['impurity'].min(), df['impurity'].max()],
                 label = 'Impurity Penalty', values = df['impurity'])
        ])
    )
)
fig_par.update_layout(paper_bgcolor='#0A0A0F', font=dict(color='#E0E0E0', family='Space Grotesk'), margin=dict(l=50, r=50, b=40, t=50))
st.plotly_chart(fig_par, use_container_width=True)

col_table, col_detail = st.columns([0.6, 0.4])

with col_table:
    st.markdown("#### Non-Dominated Solutions")
    display_cols = ['recovery', 'cost', 'impurity', 'alpha_Mn', 'alpha_Fe', 'alpha_Na', 'T', 'pH', 'conc', 't']
    st.dataframe(df[display_cols].style.background_gradient(cmap='viridis', subset=['recovery']).background_gradient(cmap='Reds', subset=['cost', 'impurity']), use_container_width=True)

with col_detail:
    st.markdown("#### Detailed Mechanism Simulation")
    selected_idx = st.slider("Select Solution Rank", 1, len(df), 1) - 1
    row = df.iloc[selected_idx]
    
    st.markdown(f"**Selected Conditions:** {row['T']:.1f} K, pH {row['pH']:.2f}, {row['conc']:.2f}M Acid, {row['t']:.0f} min")
    
    # Simulate the ODE mechanisms for the selected point
    t_arr = np.linspace(0, row['t'], 100)
    
    # Simple surrogate for visualization matching the backend ODE
    gamma = 1.0 / (1.0 + np.exp(-(2*(row['T']-323)/40 - (row['pH']-0.5)/2.5)))
    
    k_sc_mn = 0.05 * (row['conc']/1.8) * np.exp((row['T']-323)/30)
    k_av_mn = 0.08 * (1 + 0.4*(3-row['pH'])) * np.exp((row['T']-323)/55)
    
    alpha_sc = 1.0 - (1.0 - np.clip(k_sc_mn * t_arr / 60, 0, 1))**3
    alpha_av = 1.0 - np.exp(-k_av_mn * (t_arr / 180)**2)
    
    alpha_total = gamma * alpha_sc + (1 - gamma) * alpha_av
    # Scale to match the final calculated recovery
    alpha_total = alpha_total * (row['alpha_Mn'] / max(alpha_total[-1], 1e-6))
    
    fig_mech = go.Figure()
    fig_mech.add_trace(go.Scatter(x=t_arr, y=alpha_sc, mode='lines', line=dict(dash='dash', color='#AAAAAA'), name="Shrinking Core (Diffusion)"))
    fig_mech.add_trace(go.Scatter(x=t_arr, y=alpha_av, mode='lines', line=dict(dash='dot', color='#00D4FF'), name="Avrami (Nucleation)"))
    fig_mech.add_trace(go.Scatter(x=t_arr, y=alpha_total, mode='lines', line=dict(color='#FF6B2B', width=3), name="Total Mn Extraction"))
    
    fig_mech.update_layout(title="Multi-Mechanism Leaching ODE", xaxis_title="Time (min)", yaxis_title="Extraction Fraction (α)", **PLOT_LAYOUT)
    fig_mech.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig_mech, use_container_width=True)
    
    st.markdown("#### Environmental Impact Scorecard")
    neutralization_cost = max(0.0, 3.0 - row['pH']) * row['conc'] * 0.05
    heavy_metal = 0.1 * ((1.0 - row['alpha_Mn'])*0.3 + (1.0 - row['alpha_Fe'])*0.25)
    
    st.metric("Waste Acidity Penalty", f"${neutralization_cost:.3f} / kg", delta="Neutralization Base", delta_color="inverse")
    st.metric("Heavy Metal Discharge Risk", f"{heavy_metal*1000:.1f} g / kg", delta="Slag Contamination", delta_color="inverse")
