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
    initial_sidebar_state="collapsed"
)

# ================= 2. STILE CSS "PIXEL PERFECT" (TAILWIND REPLICA) =================
st.markdown("""
<style>
    /* IMPORT FONT INTER & JETBRAINS MONO */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

    /* BASE RESET */
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1600px; }
    header { visibility: hidden; }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #111827; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }

    /* CARD SYSTEM (Identico a React bg-gray-900) */
    .titan-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px; /* rounded-xl */
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* TYPOGRAPHY */
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
    }
    .section-title {
        font-size: 14px;
        font-weight: 700;
        color: #d1d5db; /* gray-300 */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* HERO TITLE GRADIENT */
    .hero-text {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(to right, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    /* ASSET BUTTONS */
    .stButton button {
        background-color: #1f2937 !important;
        color: #9ca3af !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover {
        border-color: #10b981 !important;
        color: white !important;
    }
    .stButton button:focus, .stButton button:active {
        background-color: #059669 !important;
        border-color: #10b981 !important;
        color: white !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4) !important;
    }

    /* SIGNAL STYLES */
    .signal-huge { font-size: 42px; font-weight: 900; letter-spacing: -0.02em; margin: 10px 0; }
    .sig-buy { color: #10b981; text-shadow: 0 0 30px rgba(16, 185, 129, 0.3); }
    .sig-sell { color: #ef4444; text-shadow: 0 0 30px rgba(239, 68, 68, 0.3); }
    .sig-wait { color: #6b7280; }

    /* METRICS BOXES */
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px; }
    .metric-box {
        background: rgba(31, 41, 55, 0.5);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .m-label { font-size: 10px; color: #9ca3af; text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }
    .m-value { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: white; }
    
    /* CUSTOM PROGRESS BAR */
    .prog-container { height: 8px; background: #1f2937; border-radius: 99px; overflow: hidden; margin-top: 8px; }
    .prog-fill { height: 100%; border-radius: 99px; transition: width 0.5s ease; }

    /* LIST ITEMS (System Status) */
    .list-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 12px;
    }
    
    /* BADGES */
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    .badge-yellow { background: rgba(234, 179, 8, 0.2); color: #facc15; }

</style>
""", unsafe_allow_html=True)

# ================= 3. CONNESSIONE DB =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# Stato Selezione Asset
if 'selected_asset' not in st.session_state: st.session_state.selected_asset = 'XAUUSD'

# ================= 4. MOTORE GRAFICO (PLOTLY CONFIGURATO) =================

def create_radar(d):
    """Radar Chart pulito senza griglie di fondo (Style React)"""
    conf = d.get('prob_buy', 50) if "BUY" in d['recommendation'] else d.get('prob_sell', 50)
    z = abs(d.get('confidence_score', 0))
    
    # Dati simulati realistici basati sui segnali
    categories = ['Momentum', 'Mean Rev', 'Volatility', 'Volume', 'Sentiment', 'Macro']
    values = [
        min(conf + 10, 95),      
        min(90 - z*10, 80),      
        min(z * 30, 85),         
        82, 58,                      
        60 if "BUY" in d['recommendation'] else 40 
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', 
        line_color='#8b5cf6', # Purple accent
        fillcolor='rgba(139, 92, 246, 0.3)',
        marker=dict(size=0)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#4b5563', showticklabels=False, gridcolor='#374151'),
            angularaxis=dict(color='#9ca3af', gridcolor='#374151'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=10, b=10),
        height=220,
        showlegend=False
    )
    return fig

def create_area_chart():
    """Area Chart (PnL) stile Recharts"""
    x = np.arange(30)
    y = np.cumsum(np.random.randn(30)) * 10 + 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, fill='tozeroy', mode='lines',
        line=dict(color='#10b981', width=2, shape='spline'),
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
        xaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False, zeroline=False),
        showlegend=False
    )
    return fig

def create_bar_chart():
    """Bar Chart (Order Flow)"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Buy', 'Sell', 'Inst.'], y=[65, 35, 20],
        marker_color=['#10b981', '#ef4444', '#8b5cf6'],
        textposition='auto'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=120,
        xaxis=dict(showgrid=False, tickfont=dict(color='#9ca3af', size=10)),
        yaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False),
        showlegend=False
    )
    return fig

# ================= 5. INTERFACCIA (LAYOUT REACT-LIKE) =================

# --- HEADER & ASSETS ---
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        <div>
            <div class="hero-text">TITAN ORACLE</div>
            <div style="color:#9ca3af; font-size:12px; letter-spacing:1px;">INSTITUTIONAL INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown('<div style="display:flex; justify-content:flex-end; align-items:center; height:100%;"><div style="background:rgba(6,78,59,0.3); border:1px solid #10b981; padding:6px 12px; border-radius:99px; color:#10b981; font-family:monospace; font-size:12px; font-weight:bold; display:flex; align-items:center; gap:6px;"><div style="width:8px; height:8px; background:#10b981; border-radius:50%; box-shadow:0 0 8px #10b981;"></div>LIVE FEED ACTIVE</div></div>', unsafe_allow_html=True)

st.write("") # Spacer

# Asset Buttons (Usiamo st.columns per metterli in riga)
assets = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
cols = st.columns(len(assets) + 2) # +2 per spaziatura
for i, asset in enumerate(assets):
    if cols[i].button(asset, key=asset, use_container_width=True):
        st.session_state.selected_asset = asset

st.markdown("---")

# --- MAIN DASHBOARD LOOP ---
placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", st.session_state.selected_asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # Logic
                conf = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                z = d.get('confidence_score', 0)
                regime = d.get('market_regime', 'ANALYZING')
                
                # Style Logic
                sig_cls = "sig-buy" if "BUY" in rec else "sig-sell" if "SELL" in rec else "sig-wait"
                bar_col = "#10b981" if "BUY" in rec else "#ef4444" if "SELL" in rec else "#6b7280"
                icon = "TrendingUp" if "BUY" in rec else "TrendingDown" if "SELL" in rec else "Activity"

                # === LOGICA "ZERO HIDER" (LA CORREZIONE) ===
                # Se il segnale è WAIT, nascondiamo i numeri 0 e mettiamo dei trattini
                if "WAIT" in rec:
                    display_entry = "---"
                    display_sl = "---"
                    display_tp = "---"
                    display_rr = "WAITING FOR SETUP..."
                    rr_color = "#6b7280" # Grigio
                else:
                    display_entry = f"${d['entry_price']}"
                    display_sl = f"${d['stop_loss']}"
                    display_tp = f"${d['take_profit']}"
                    display_rr = f"1:{d['risk_reward']}"
                    rr_color = "white"

                # ================= GRID LAYOUT =================
                col_left, col_mid, col_right = st.columns([1, 1.2, 0.8])
                
                # === COLONNA 1: AI SIGNAL ===
                with col_left:
                    st.markdown(f"""
                    <div class="titan-card">
                        <div class="section-header">
                            <span class="section-title"><span style="color:#22d3ee">🧠</span> AI SIGNAL</span>
                            <span style="font-family:'JetBrains Mono'; font-size:10px; color:#6b7280;">H1 • {st.session_state.selected_asset}</span>
                        </div>
                        
                        <div style="text-align:center; padding: 20px 0;">
                            <div class="{sig_cls} signal-huge">{rec}</div>
                            <div style="color:#9ca3af; font-size:13px; margin-top:5px;">{d['details']}</div>
                        </div>
                        
                        <div style="margin-top:10px;">
                            <div style="display:flex; justify-content:space-between; font-size:11px; color:#9ca3af; margin-bottom:4px;">
                                <span>CONFIDENCE SCORE</span>
                                <span style="color:white; font-weight:bold;">{conf:.1f}%</span>
                            </div>
                            <div class="prog-container">
                                <div class="prog-fill" style="width:{conf}%; background:{bar_col};"></div>
                            </div>
                        </div>
                        
                        <div style="margin-top:20px; border-top:1px solid #374151; padding-top:15px;">
                            <div style="font-size:11px; color:#6b7280; margin-bottom:5px;">CURRENT PRICE</div>
                            <div style="font-size:32px; font-weight:800; color:#ffffff; letter-spacing:-1px;">${d['current_price']}</div>
                        </div>
                        
                        <div class="metric-grid">
                            <div class="metric-box" style="border-color:#0e7490; background:rgba(8,145,178,0.1);">
                                <div class="m-label" style="color:#22d3ee">ENTRY</div>
                                <div class="m-value" style="color:#22d3ee">{display_entry}</div>
                            </div>
                            <div class="metric-box" style="border-color:#991b1b; background:rgba(127,29,29,0.1);">
                                <div class="m-label" style="color:#f87171">STOP</div>
                                <div class="m-value" style="color:#f87171">{display_sl}</div>
                            </div>
                            <div class="metric-box" style="border-color:#065f46; background:rgba(6,78,59,0.1);">
                                <div class="m-label" style="color:#34d399">TARGET</div>
                                <div class="m-value" style="color:#34d399">{display_tp}</div>
                            </div>
                        </div>
                        
                        <div style="margin-top:15px; text-align:center;">
                            <span style="font-size:11px; color:#6b7280; background:#1f2937; padding:4px 10px; border-radius:20px;">
                                Risk/Reward: <strong style="color:{rr_color}">{display_rr}</strong>
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # === COLONNA 2: CHARTS ===
                with col_mid:
                    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">MULTI-FACTOR ANALYSIS</div>', unsafe_allow_html=True)
                    st.plotly_chart(create_radar(d), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">CUMULATIVE P&L (PROJ)</div>', unsafe_allow_html=True)
                    st.plotly_chart(create_area_chart(), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                # === COLONNA 3: STATUS & METRICS ===
                with col_right:
                    regime_active = "border-blue-500 bg-blue-900/20" if regime == "TRENDING" else "border-gray-700"
                    st.markdown(f"""
                    <div class="titan-card">
                        <div class="section-title"><span style="color:#60a5fa">⚡</span> MARKET REGIME</div>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            <div class="list-item" style="border:1px solid #3b82f6; background:rgba(59,130,246,0.1);">
                                <span style="font-weight:700; color:white;">{regime}</span>
                                <div style="width:6px; height:6px; background:#60a5fa; border-radius:50%; box-shadow:0 0 5px #60a5fa;"></div>
                            </div>
                            <div class="list-item"><span style="color:#6b7280;">RANGING</span></div>
                            <div class="list-item"><span style="color:#6b7280;">VOLATILE</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">ORDER FLOW</div>', unsafe_allow_html=True)
                    st.plotly_chart(create_bar_chart(), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="titan-card">
                        <div class="section-title">SYSTEM STATUS</div>
                        <div class="list-item">
                            <span style="color:#9ca3af;">ML Engine</span>
                            <span class="badge badge-green">ACTIVE</span>
                        </div>
                        <div class="list-item">
                            <span style="color:#9ca3af;">Risk Manager</span>
                            <span class="badge badge-green">ACTIVE</span>
                        </div>
                        <div class="list-item">
                            <span style="color:#9ca3af;">Execution</span>
                            <span class="badge badge-yellow">STANDBY</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info(f"Establishing encrypted link to TITAN Core for {st.session_state.selected_asset}...")
                
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
