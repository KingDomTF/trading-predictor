import streamlit as st
from supabase import create_client
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# ================= CONFIGURAZIONE PAGINA =================
st.set_page_config(page_title="TITAN Oracle", layout="wide", page_icon="🛡️")

# ================= CSS TITAN (STILE REACT REPLICATO) =================
st.markdown("""
<style>
    /* RESET E BACKGROUND */
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* TITAN CARD CONTAINER */
    .titan-card {
        background-color: #111827; /* Gray 900 */
        border: 1px solid #1f2937; /* Gray 800 */
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* HEADER E TESTI */
    .titan-title {
        font-size: 2.25rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .titan-subtitle { color: #9ca3af; font-size: 0.875rem; }
    
    /* LIVE STATUS BADGE */
    .live-badge {
        background-color: rgba(6, 78, 59, 0.5);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-family: monospace;
        font-size: 0.875rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* SIGNAL DISPLAY */
    .signal-box { text-align: center; padding: 1rem; }
    .signal-text { font-size: 3rem; font-weight: 900; letter-spacing: -0.05em; }
    .sig-buy { color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
    .sig-sell { color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
    .sig-wait { color: #6b7280; }
    
    /* METRIC GRID */
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 1rem; }
    .metric-item {
        background-color: rgba(17, 24, 39, 0.5);
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 0.75rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; margin-bottom: 0.25rem; }
    .metric-value { font-size: 1rem; font-weight: 700; font-family: monospace; }
    
    /* PROGRESS BAR */
    .prog-container { width: 100%; background-color: #374151; height: 0.75rem; border-radius: 9999px; overflow: hidden; margin-top: 0.5rem; }
    .prog-fill { height: 100%; transition: width 0.5s ease; }
    
    /* UTILS */
    .text-cyan { color: #22d3ee; }
    .text-emerald { color: #34d399; }
    .text-red { color: #f87171; }
    .text-purple { color: #a78bfa; }
    .text-orange { color: #fbbf24; }
    
</style>
""", unsafe_allow_html=True)

# ================= CONNESSIONE DB =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### 🛡️ CONTROL PANEL")
    asset = st.selectbox("SELECT ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"])
    st.markdown("---")
    st.info("System: TITAN Oracle v4.0\nMode: Real-Time Sync")

# ================= FUNZIONI GRAFICHE =================
def create_radar_chart(data_dict):
    """Crea il grafico Radar per l'analisi multifattoriale"""
    categories = list(data_dict.keys())
    values = list(data_dict.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Titan Score',
        line_color='#8b5cf6',
        fillcolor='rgba(139, 92, 246, 0.5)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#6b7280'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        font=dict(color='#e5e7eb', size=10)
    )
    return fig

def create_pnl_chart():
    """Simula il grafico PnL cumulativo (estetico)"""
    dates = pd.date_range(end=datetime.now(), periods=30)
    values = np.cumsum(np.random.randn(30) * 100 + 50)
    
    fig = px.area(x=dates, y=values)
    fig.update_traces(line_color='#10b981', fill_color='rgba(16, 185, 129, 0.1)')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#6b7280'),
        yaxis=dict(showgrid=True, gridcolor='#374151', color='#6b7280'),
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        showlegend=False
    )
    return fig

# ================= LOOP PRINCIPALE =================
placeholder = st.empty()

while True:
    try:
        # Preleva dati reali
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # Calcoli derivati per la grafica
                conf_score = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                z_score = d.get('confidence_score', 0)
                
                # --- HEADER ---
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""
                    <div>
                        <div class="titan-title">TITAN ORACLE</div>
                        <div class="titan-subtitle">Institutional-Grade Trading Intelligence • {asset}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown("""
                    <div style="text-align:right;">
                        <span class="live-badge">● LIVE SYSTEM</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                # --- GRID PRINCIPALE ---
                col_left, col_right = st.columns([1, 1.5])
                
                # === COLONNA SINISTRA: SEGNALE & PREZZI ===
                with col_left:
                    # 1. SIGNAL CARD
                    sig_class = "sig-buy" if "BUY" in rec else "sig-sell" if "SELL" in rec else "sig-wait"
                    st.markdown(f"""
                    <div class="titan-card">
                        <div style="display:flex; justify-content:space-between; color:#9ca3af; font-size:0.8rem; margin-bottom:10px;">
                            <span>AI SIGNAL PROCESSOR</span>
                            <span>CONFIDENCE: {conf_score:.1f}%</span>
                        </div>
                        <div class="signal-box">
                            <div class="signal-text {sig_class}">{rec}</div>
                            <div style="color:#6b7280; font-size:0.9rem; margin-top:5px;">{d['details']}</div>
                        </div>
                        
                        <div class="prog-container">
                            <div class="prog-fill" style="width: {conf_score}%; background-color: {'#10b981' if 'BUY' in rec else '#ef4444' if 'SELL' in rec else '#6b7280'};"></div>
                        </div>
                        
                        <div class="metric-grid">
                            <div class="metric-item" style="border-color: rgba(34, 211, 238, 0.3);">
                                <div class="metric-label text-cyan">ENTRY</div>
                                <div class="metric-value text-cyan">{d['entry_price']}</div>
                            </div>
                            <div class="metric-item" style="border-color: rgba(248, 113, 113, 0.3);">
                                <div class="metric-label text-red">STOP</div>
                                <div class="metric-value text-red">{d['stop_loss']}</div>
                            </div>
                            <div class="metric-item" style="border-color: rgba(52, 211, 153, 0.3);">
                                <div class="metric-label text-emerald">TARGET</div>
                                <div class="metric-value text-emerald">{d['take_profit']}</div>
                            </div>
                        </div>
                        
                        <div style="margin-top:1rem; display:flex; justify-content:space-between; font-size:0.8rem;">
                            <span style="color:#9ca3af;">Risk/Reward</span>
                            <span class="text-emerald" style="font-weight:bold;">1:{d['risk_reward']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 2. RISK METRICS CARD
                    st.markdown(f"""
                    <div class="titan-card">
                        <div style="display:flex; justify-content:space-between; margin-bottom:1rem;">
                            <span style="color:#e5e7eb; font-weight:bold;">RISK METRICS</span>
                            <span style="color:#a78bfa;">🛡️ ACTIVE</span>
                        </div>
                        <div class="metric-grid" style="grid-template-columns: repeat(2, 1fr);">
                            <div class="metric-item">
                                <div class="metric-label">Z-SCORE (VOLATILITY)</div>
                                <div class="metric-value text-orange">{z_score} σ</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">MARKET REGIME</div>
                                <div class="metric-value text-cyan">{d.get('market_regime', 'ANALYZING')}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # === COLONNA DESTRA: GRAFICI ===
                with col_right:
                    # 3. MULTI-FACTOR RADAR
                    st.markdown("""
                    <div class="titan-card" style="min-height:300px;">
                        <span style="color:#e5e7eb; font-weight:bold;">MULTI-FACTOR ANALYSIS</span>
                    """, unsafe_allow_html=True)
                    
                    # Generiamo dati radar dinamici basati sul segnale
                    radar_data = {
                        "Momentum": min(conf_score + 10, 95),
                        "Trend": min(conf_score, 90),
                        "Volatility": min(abs(z_score)*30, 90),
                        "Volume": 75,
                        "Macro": 60 if "BUY" in rec else 40
                    }
                    st.plotly_chart(create_radar_chart(radar_data), use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 4. SYSTEM LOGS / PNL (Simulato per estetica)
                    st.markdown("""
                    <div class="titan-card">
                        <span style="color:#e5e7eb; font-weight:bold;">PROJECTED PERFORMANCE</span>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(create_pnl_chart(), use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.warning(f"Connecting to TITAN Neural Net for {asset}...")
                
        time.sleep(1)
        
    except Exception as e:
        # st.error(f"Error: {e}") # Debug only
        time.sleep(1)
