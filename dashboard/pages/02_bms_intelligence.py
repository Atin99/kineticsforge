import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pathlib import Path
import os

st.set_page_config(page_title="BMS Edge Intelligence", layout="wide")
st.title("BMS Edge Intelligence")

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'cache'
cache_file = CACHE_DIR / 'bms_simulation.npz'

PLOT_LAYOUT = dict(
    plot_bgcolor='#0A0A0F', paper_bgcolor='#0A0A0F',
    font=dict(color='#E0E0E0', family='Space Grotesk'),
    xaxis=dict(gridcolor='#1E1E2E', zerolinecolor='#1E1E2E'), 
    yaxis=dict(gridcolor='#1E1E2E', zerolinecolor='#1E1E2E'),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(font=dict(color='#E0E0E0'), bgcolor='rgba(0,0,0,0)')
)

@st.cache_data
def load_bms_data():
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file, allow_pickle=True)
        return {
            'scenarios': data['scenarios'],
            'history': data['history']
        }
    except Exception as e:
        st.error(f"Error loading cache: {e}")
        return None

data = load_bms_data()

if data is None:
    st.warning("Cache not found. Please run the precompute pipeline first.")
    st.stop()

scenarios = data['scenarios']
history = data['history']

# UI Selection
st.sidebar.markdown("### Simulation Controls")
selected_scenario_idx = st.sidebar.selectbox(
    "Select Drive Cycle Scenario", 
    range(len(scenarios)), 
    format_func=lambda i: f"Scenario {i+1}: {scenarios[i].get('failure_type', 'Nominal')} ({scenarios[i].get('lead_time_min', 0):.1f}m lead)"
)

scenario = scenarios[selected_scenario_idx]
hist = history[selected_scenario_idx]

time_arr = hist['time']
n_steps = len(time_arr)

step_idx = st.sidebar.slider("Timeline (seconds)", 0, int(time_arr[-1]), 0, step=60)
closest_step = np.argmin(np.abs(time_arr - step_idx))

tab1, tab2, tab3 = st.tabs(["Graph Topology (Live)", "Thermal & Risk Dynamics", "Graph-NODE Diagnostics"])

with tab1:
    st.markdown("### Spatiotemporal Pack Topology")
    st.markdown("Visualizing the internal battery pack graph. Edges represent electrical and thermal coupling coefficients learned by the Graph-NODE.")
    
    col_graph, col_stats = st.columns([0.7, 0.3])
    
    with col_graph:
        n_cells = hist['T_cells'].shape[1]
        
        # Build NetworkX graph for topology visualization
        G = nx.Graph()
        for i in range(n_cells):
            G.add_node(i)
            
        # Simplified 2x4 layout
        pos = {}
        for i in range(n_cells):
            row = i // 4
            col = i % 4
            pos[i] = (col, -row)
            
            # Add edges
            if col < 3: G.add_edge(i, i+1)
            if row < 1 and i+4 < n_cells: G.add_edge(i, i+4)
        
        # Get live data for coloring
        live_T = hist['T_cells'][closest_step]
        live_risk = hist['risk'][closest_step]
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, line=dict(width=2, color='#1E1E2E'),
            hoverinfo='none', mode='lines'
        )

        node_x = [pos[i][0] for i in range(n_cells)]
        node_y = [pos[i][1] for i in range(n_cells)]

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            hoverinfo='text',
            text=[f"C{i}" for i in range(n_cells)],
            textposition="middle center",
            textfont=dict(color='white', size=10),
            marker=dict(
                showscale=True, colorscale='Inferno', 
                color=live_T, size=40,
                colorbar=dict(thickness=15, title='Temp (K)', outlinewidth=0, titlefont=dict(color='#E0E0E0'), tickfont=dict(color='#E0E0E0')),
                line_width=3, line_color=['#FF4444' if r > 0.75 else '#00D4FF' for r in live_risk]
            )
        )
        
        hover_text = []
        for i in range(n_cells):
            hover_text.append(f"Cell {i}<br>Temp: {live_T[i]:.1f} K<br>Risk: {live_risk[i]:.2f}")
        node_trace.hovertext = hover_text

        fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            plot_bgcolor='#0A0A0F', paper_bgcolor='#0A0A0F',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ))
        st.plotly_chart(fig_net, use_container_width=True)

    with col_stats:
        st.markdown("#### Live State")
        st.metric("Time", f"{time_arr[closest_step]:.0f} s")
        st.metric("Pack Current", f"{hist['I_pack'][closest_step]:.2f} A")
        
        max_t_idx = np.argmax(live_T)
        st.metric("Peak Temp", f"{live_T[max_t_idx]:.1f} K", f"Cell {max_t_idx}")
        
        max_risk_idx = np.argmax(live_risk)
        st.metric("Peak Risk", f"{live_risk[max_risk_idx]:.2f}", f"Cell {max_risk_idx}")
        
        if len(scenario.get('alert_cells', [])) > 0:
            st.error(f"🚨 ALERT ON CELLS: {scenario['alert_cells']}")
        else:
            st.success("✅ Nominal Operation")

with tab2:
    st.markdown("### Thermal Cascade & Risk Trajectories")
    
    fig_tr = go.Figure()
    fig_risk = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i in range(hist['T_cells'].shape[1]):
        c = colors[i % len(colors)]
        width = 4 if i in scenario.get('alert_cells', []) else 1
        
        fig_tr.add_trace(go.Scatter(x=time_arr, y=hist['T_cells'][:, i], mode='lines', name=f"Cell {i}", line=dict(color=c, width=width)))
        fig_risk.add_trace(go.Scatter(x=time_arr, y=hist['risk'][:, i], mode='lines', name=f"Cell {i}", line=dict(color=c, width=width)))
        
    # Add vertical line for current step
    fig_tr.add_vline(x=time_arr[closest_step], line_width=2, line_dash="dash", line_color="white")
    fig_risk.add_vline(x=time_arr[closest_step], line_width=2, line_dash="dash", line_color="white")
    
    # Add alert threshold
    fig_risk.add_hline(y=0.75, line_width=2, line_dash="dot", line_color="#FF4444", annotation_text="Critical Alert Threshold")
        
    fig_tr.update_layout(title="Thermal Evolution (K)", xaxis_title="Time (s)", yaxis_title="Temperature (K)", **PLOT_LAYOUT)
    fig_risk.update_layout(title="Graph-NODE Risk Probability", xaxis_title="Time (s)", yaxis_title="Risk Score", **PLOT_LAYOUT)
    
    st.plotly_chart(fig_tr, use_container_width=True)
    st.plotly_chart(fig_risk, use_container_width=True)

with tab3:
    st.markdown("### Alert Lead-Time Performance")
    st.markdown("The Graph-NODE uses a custom asymmetric loss function to heavily penalize late alerts while tolerating early warnings, providing maximizing evacuation/shutdown time.")
    
    col_lead, col_loss = st.columns(2)
    
    with col_lead:
        # Distribution of lead times across all scenarios
        lead_times = [s.get('lead_time_min', 0) for s in scenarios if s.get('alert_fired', False)]
        if lead_times:
            fig_hist = px.histogram(lead_times, nbins=10, title="Early Warning Lead Time Distribution")
            fig_hist.update_layout(xaxis_title="Lead Time (minutes)", yaxis_title="Frequency", **PLOT_LAYOUT)
            fig_hist.add_vline(x=np.mean(lead_times), line_dash="dash", line_color="#00D4FF", annotation_text=f"Mean: {np.mean(lead_times):.1f}m")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No failure scenarios available to plot lead times.")
            
    with col_loss:
        # Theoretical Loss Surface
        x = np.linspace(-100, 300, 100) # True failure step - alert step
        y = np.where(x < 0, 5.0 * (x/10)**2, 0.5 * (x/10)**2) # Asymmetric penalty
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', line=dict(color='#FF6B2B')))
        fig_loss.update_layout(
            title="Asymmetric False Positive/Negative Loss Surface",
            xaxis_title="Time delta (True Failure - Alert Time) [sec]",
            yaxis_title="Loss Penalty",
            annotations=[
                dict(x=-50, y=100, text="Late Alert<br>(High Penalty)", showarrow=False, font=dict(color="#FF4444")),
                dict(x=150, y=100, text="Early Alert<br>(Low Penalty)", showarrow=False, font=dict(color="#00D4FF"))
            ],
            **PLOT_LAYOUT
        )
        st.plotly_chart(fig_loss, use_container_width=True)
