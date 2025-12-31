import streamlit as st
from supabase import create_client
import time
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ================= 1. CONFIGURAZIONE PAGINA =================
st.set_page_config(
    page_title="TITAN Oracle",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed" # Nascondiamo la sidebar per il look "Terminal"
)

# ================= 2. STILE CSS (REPLICA ESATTA REACT/TAILWIND) =================
st.markdown("""
<style>
    /* RESET E FONDO */
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* RIMUOVERE PADDING STREAMLIT */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* CARD STYLE (Replica di .bg-gray-900 .border-gray-800) */
    .titan-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 0.75rem; /* rounded-xl */
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        height: 100%;
    }
    
    /* HEADERS */
    .section-title {
        font-size: 1.125rem; /* text-lg */
        font-weight: 700;
        color: #d1d5db; /* text-gray-300 */
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
    }
    
    /* MAIN TITLE GRADIENT */
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* SIGNAL COLORS */
    .sig-text-buy { color: #10b981; font-size: 3rem; font-weight: 900; }
    .sig-text-sell { color: #ef4444; font-size: 3rem; font-weight: 900; }
    .sig-text-wait { color: #6b7280; font-size: 3rem; font-weight: 900; }
    
    /* BADGES */
    .badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-bull { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .badge-bear { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    
    /* METRICS BOXES (Entry, Stop, TP) */
    .metric-box {
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .box-cyan { background: rgba(8, 145, 178, 0.2); border: 1px solid #0e7490; }
    .box-red { background: rgba(127, 29, 29, 0.2); border: 1px solid #991b1b; }
    .box-emerald { background: rgba(6, 78, 59, 0.2); border: 1px solid #065f46; }
    
    /* STATUS INDICATORS */
    .status-dot { height: 0.5rem; width: 0.5rem; border-radius: 50%; display: inline-block; margin-right: 0.5rem; }
    .dot-green { background-color: #10b981; box-shadow: 0 0 10px #10b981; }
    .dot-yellow { background-color: #eab308; }
    
    /* ASSET BUTTONS (Simulati) */
    .asset-btn {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #1f2937;
        color: #9ca3af;
        border-radius: 0.5rem;
        margin-right: 0.5rem;
        font-family: monospace;
        font-size: 0.875rem;
        border: 1px solid #374151;
    }
    .asset-btn-active {
        background-color: #059669;
        color: white;
        border: 1px solid #10b981;
    }

</style>
""", unsafe_allow_html=True)

# ================= 3. CONNESSIONE DB =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# ================= 4. FUNZIONI GRAFICHE (PLOTLY REPLICA RECHARTS) =================
def plot_radar(d):
    """Replica il RadarChart di Recharts"""
    conf = d.get('prob_buy', 50) if "BUY" in d['recommendation'] else d.get('prob_sell', 50)
    z = abs(d.get('confidence_score', 0))
    
    categories = ['Momentum', 'Mean Reversion', 'Volatility', 'Volume', 'Sentiment', 'Macro']
    values = [
        min(conf + 10, 95),      # Momentum
        min(90 - z*10, 80),      # Mean Rev
        min(z * 30, 85),         # Volatility
        82,                      # Volume (Simulato)
        58,                      # Sentiment (Simulato)
        60 if "BUY" in d['recommendation'] else 40 # Macro
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', 
        line_color='#8b5cf6', fillcolor='rgba(139, 92, 246, 0.4)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#6b7280', showticklabels=False),
            bgcolor='rgba(0,0,0,0)',
            gridshape='circular'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=20, b=20),
        height=220,
        font=dict(color='#9ca3af', size=10),
        showlegend=False
    )
    return fig

def plot_pnl_area():
    """Replica l'AreaChart verde"""
    x = np.arange(30)
    y = np.cumsum(np.random.randn(30) + 0.5) * 100 + 1000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, fill='tozeroy', mode='lines',
        line=dict(color='#10b981', width=2),
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
        xaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False),
        showlegend=False
    )
    return fig

def plot_order_flow():
    """Replica il BarChart Order Flow"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Buy Vol'], y=[65], marker_color='#10b981', name='Buy'))
    fig.add_trace(go.Bar(x=['Sell Vol'], y=[35], marker_color='#ef4444', name='Sell'))
    fig.add_trace(go.Bar(x=['Inst.'], y=[20], marker_color='#8b5cf6', name='Inst'))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=120,
        xaxis=dict(showgrid=False, tickfont=dict(color='#9ca3af')),
        yaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False),
        showlegend=False,
        barmode='group'
    )
    return fig

# ================= 5. INTERFACCIA UTENTE =================

# --- TOP BAR: Asset Selector & Header ---
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown('<div class="hero-title">TITAN ORACLE</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#9ca3af; font-size:0.875rem;">Institutional-Grade Trading Intelligence</div>', unsafe_allow_html=True)
with c2:
    # Asset Selector (Simulato visivamente, funzionale tramite selectbox nascosta)
    asset = st.selectbox("Active Asset Feed", ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"], label_visibility="collapsed")
    
    # Status Badge
    st.markdown(f"""
    <div style="text-align:right; margin-top:10px;">
        <span style="background:rgba(6,78,59,0.3); border:1px solid #10b981; padding:5px 10px; border-radius:20px; color:#10b981; font-family:monospace; font-size:12px;">
            <span class="status-dot dot-green"></span>LIVE FEED
        </span>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# --- MAIN LOOP ---
placeholder = st.empty()

while True:
    try:
        # Fetch Dati Reali
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # Logic for colors/values
                conf_score = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                z_score = d.get('confidence_score', 0)
                regime = d.get('market_regime', 'ANALYZING')
                
                sig_color_class = "sig-text-buy" if "BUY" in rec else "sig-text-sell" if "SELL" in rec else "sig-text-wait"
                progress_color = "#10b981" if "BUY" in rec else "#ef4444" if "SELL" in rec else "#6b7280"

                # ================= GRID LAYOUT (12 Colonne come in React) =================
                # Streamlit usa ratio, qui facciamo 3 colonne principali come nel design React
                # Col 1: Signal & Execution (33%)
                # Col 2: Charts & Performance (40%)
                # Col 3: Side Metrics (27%)
                col1, col2, col3 = st.columns([1, 1.2, 0.8])
                
                # --- COLONNA 1: AI SIGNAL & EXECUTION ---
                with col1:
                    # SIGNAL CARD
                    st.markdown(f"""
                    <div class="titan-card">
                        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                            <div class="section-title">AI SIGNAL</div>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10 10 10 0 0 0-10-10zm0 16a6 6 0 1 1 6-6 6 6 0 0 1-6 6z"></path></svg>
                        </div>
                        
                        <div style="text-align:center; padding:10px;">
                             <div class="{sig_color_class}">{rec}</div>
                             <div style="color:#6b7280; font-size:0.875rem;">{asset} • H1</div>
                        </div>
                        
                        <div style="margin-top:15px; margin-bottom:5px; display:flex; justify-content:space-between; font-size:12px; color:#6b7280;">
                            <span>ML Confidence</span><span>{conf_score:.1f}%</span>
                        </div>
                        <div style="background:#1f2937; height:8px; border-radius:99px; overflow:hidden;">
                            <div style="width:{conf_score}%; background:{progress_color}; height:100%;"></div>
                        </div>
                        
                        <div style="margin-top:20px;">
                            <div class="metric-box box-cyan">
                                <div style="font-size:10px; color:#22d3ee; margin-bottom:2px;">ENTRY</div>
                                <div style="font-size:16px; font-weight:bold; font-family:monospace; color:white;">${d['entry_price']}</div>
                            </div>
                            <div style="display:flex; gap:10px;">
                                <div class="metric-box box-red" style="width:50%;">
                                    <div style="font-size:10px; color:#f87171; margin-bottom:2px;">STOP</div>
                                    <div style="font-size:14px; font-weight:bold; font-family:monospace; color:white;">${d['stop_loss']}</div>
                                </div>
                                <div class="metric-box box-emerald" style="width:50%;">
                                    <div style="font-size:10px; color:#34d399; margin-bottom:2px;">TARGET</div>
                                    <div style="font-size:14px; font-weight:bold; font-family:monospace; color:white;">${d['take_profit']}</div>
                                </div>
                            </div>
                        </div>
                        
                        <div style="margin-top:10px; display:flex; justify-content:space-between; align-items:center; background:#1f2937; padding:8px; border-radius:8px;">
                            <span style="font-size:12px; color:#9ca3af;">Risk/Reward</span>
                            <span style="color:#34d399; font-weight:bold;">1:{d['risk_reward']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # PATTERNS DETECTED (Simulato basato su logica AI)
                    pat_name = "Bull Flag" if "BUY" in rec else "Bear Flag" if "SELL" in rec else "Consolidation"
                    pat_type = "badge-bull" if "BUY" in rec else "badge-bear" if "SELL" in rec else ""
                    st.markdown(f"""
                    <div class="titan-card">
                        <div class="section-title">PATTERNS DETECTED</div>
                        <div style="background:#1f2937; padding:10px; border-radius:8px; margin-bottom:8px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span style="font-weight:600; font-size:14px;">{pat_name}</span>
                                <span class="badge {pat_type}">{rec.split(' ')[-1]}</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:10px;">
                                <div style="flex-grow:1; background:#374151; height:6px; border-radius:99px;">
                                    <div style="width:75%; background:{progress_color}; height:100%; border-radius:99px;"></div>
                                </div>
                                <span style="font-size:12px; color:#9ca3af;">75%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- COLONNA 2: CHARTS & PNL ---
                with col2:
                    # PNL CHART
                    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">CUMULATIVE P&L (PROJ)</div>', unsafe_allow_html=True)
                    st.plotly_chart(plot_pnl_area(), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # RADAR CHART
                    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">MULTI-FACTOR ANALYSIS</div>', unsafe_allow_html=True)
                    st.plotly_chart(plot_radar(d), use_container_width=True, config={'displayModeBar': False})
                    
                    # Factor Grid HTML
                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:5px; text-align:center; margin-top:10px;">
                        <div><div style="font-size:10px; color:#6b7280;">Trend</div><div style="color:#a78bfa; font-weight:bold;">{conf_score:.0f}</div></div>
                        <div><div style="font-size:10px; color:#6b7280;">Vol</div><div style="color:#a78bfa; font-weight:bold;">{z_score*10:.0f}</div></div>
                        <div><div style="font-size:10px; color:#6b7280;">Macro</div><div style="color:#a78bfa; font-weight:bold;">65</div></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- COLONNA 3: SIDE METRICS & SYSTEM ---
                with col3:
                    # MARKET REGIME
                    regime_color = "border-blue-500 bg-blue-900/30" if regime == "TRENDING" else "border-gray-700 bg-gray-800"
                    st.markdown(f"""
                    <div class="titan-card">
                        <div class="section-title">MARKET REGIME</div>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                             <div style="padding:10px; border-radius:8px; border:1px solid #3b82f6; background:rgba(59,130,246,0.1); display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:14px; font-weight:600;">{regime}</span>
                                <div style="width:8px; height:8px; background:#60a5fa; border-radius:50%; box-shadow:0 0 5px #60a5fa;"></div>
                             </div>
                             <div style="padding:10px; border-radius:8px; border:1px solid #374151; background:#1f2937; color:#6b7280;">RANGING</div>
                             <div style="padding:10px; border-radius:8px; border:1px solid #374151; background:#1f2937; color:#6b7280;">VOLATILE</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # RISK METRICS
                    st.markdown(f"""
                    <div class="titan-card">
                        <div class="section-title">RISK METRICS</div>
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                            <div style="background:#1f2937; padding:10px; border-radius:8px;">
                                <div style="font-size:10px; color:#9ca3af;">Z-Score</div>
                                <div style="font-size:16px; font-weight:bold; color:#fbbf24;">{z_score} σ</div>
                            </div>
                            <div class="metric-box" style="background:#1f2937; padding:10px; border-radius:8px;">
                                <div style="font-size:10px; color:#9ca3af;">Win Rate</div>
                                <div style="font-size:16px; font-weight:bold; color:#34d399;">68%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # SYSTEM STATUS
                    st.markdown("""
                    <div class="titan-card">
                        <div class="section-title">SYSTEM STATUS</div>
                        <div style="font-size:12px; display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span style="color:#9ca3af;">ML Engine</span>
                            <span style="color:#34d399; font-family:monospace;">ACTIVE</span>
                        </div>
                         <div style="font-size:12px; display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span style="color:#9ca3af;">Risk Manager</span>
                            <span style="color:#34d399; font-family:monospace;">ACTIVE</span>
                        </div>
                         <div style="font-size:12px; display:flex; justify-content:space-between;">
                            <span style="color:#9ca3af;">Order Router</span>
                            <span style="color:#eab308; font-family:monospace;">STANDBY</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.warning(f"Initializing connection to TITAN Core for {asset}...")
                
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
