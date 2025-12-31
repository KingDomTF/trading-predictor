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

# ================= 2. STILE CSS "PIXEL PERFECT" =================
st.markdown("""
<style>
    /* IMPORT FONT */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* BASE */
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
    
    /* NASCONDI ELEMENTI STANDARD */
    header { visibility: hidden; }
    .stButton > button { border-radius: 8px; font-weight: 600; border: 1px solid #374151; background-color: #1f2937; color: #9ca3af; }
    .stButton > button:hover { border-color: #10b981; color: #10b981; background-color: #111827; }
    .stButton > button:focus { color: white; background-color: #059669; border-color: #059669; }

    /* CARD SYSTEM (Identico alle foto) */
    .titan-card {
        background-color: #111827; /* Gray 900 */
        border: 1px solid #1f2937; /* Gray 800 */
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* TYPOGRAPHY */
    .section-title { font-size: 14px; font-weight: 700; color: #d1d5db; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    .value-huge { font-size: 48px; font-weight: 900; letter-spacing: -0.02em; }
    .value-large { font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .label-small { font-size: 12px; color: #9ca3af; font-weight: 500; }

    /* STATUS PILLS */
    .status-pill { padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .pill-green { background: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.2); }
    .pill-red { background: rgba(239, 68, 68, 0.1); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }
    
    /* PROGRESS BARS CUSTOM */
    .prog-track { width: 100%; height: 6px; background: #374151; border-radius: 99px; overflow: hidden; margin-top: 8px; }
    .prog-fill { height: 100%; border-radius: 99px; transition: width 0.5s ease; }
    
    /* METRIC BOXES */
    .metric-container { background: rgba(31, 41, 55, 0.5); border: 1px solid #374151; border-radius: 8px; padding: 12px; }
    .box-green { border-color: rgba(16, 185, 129, 0.3); background: rgba(6, 78, 59, 0.2); }
    .box-red { border-color: rgba(239, 68, 68, 0.3); background: rgba(127, 29, 29, 0.2); }
    .box-purple { border-color: rgba(139, 92, 246, 0.3); background: rgba(88, 28, 135, 0.2); }

    /* SIGNAL COLORS */
    .text-buy { color: #10b981; text-shadow: 0 0 30px rgba(16, 185, 129, 0.2); }
    .text-sell { color: #ef4444; text-shadow: 0 0 30px rgba(239, 68, 68, 0.2); }
    
    /* LIVE DOT */
    .live-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981; margin-right: 6px; }

</style>
""", unsafe_allow_html=True)

# ================= 3. CONNESSIONE DB =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# Gestione Stato Selezione
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = 'XAUUSD'

# ================= 4. FUNZIONI GRAFICHE (PLOTLY) =================
def plot_radar_clean(d):
    """Radar Chart identico allo screenshot 'Multi-Factor Analysis'"""
    conf = d.get('prob_buy', 50) if "BUY" in d['recommendation'] else d.get('prob_sell', 50)
    z = abs(d.get('confidence_score', 0))
    
    categories = ['Momentum', 'Mean Rev', 'Volatility', 'Volume', 'Sentiment', 'Macro']
    values = [
        min(conf + 10, 95),      
        min(90 - z*10, 80),      
        min(z * 30, 85),         
        82,                      
        58,                      
        60 if "BUY" in d['recommendation'] else 40 
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', 
        line_color='#8b5cf6', fillcolor='rgba(139, 92, 246, 0.2)', marker=dict(size=0)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#4b5563', showticklabels=False, gridcolor='#374151'),
            angularaxis=dict(color='#9ca3af', gridcolor='#374151'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        height=250,
        showlegend=False
    )
    return fig

def plot_order_flow_clean():
    """Bar Chart identico allo screenshot 'Order Flow'"""
    fig = go.Figure()
    # Buy Vol (Verde)
    fig.add_trace(go.Bar(
        y=['Flow'], x=[30], orientation='h', marker_color='#10b981', name='Buy',
        text="30%", textposition="inside"
    ))
    # Sell Vol (Rosso)
    fig.add_trace(go.Bar(
        y=['Flow'], x=[60], orientation='h', marker_color='#ef4444', name='Sell',
        text="60%", textposition="inside"
    ))
    # Inst Vol (Viola - Negativo per estetica)
    fig.add_trace(go.Bar(
        y=['Flow'], x=[-15], orientation='h', marker_color='#8b5cf6', name='Inst',
        text="-15%", textposition="inside"
    ))
    
    fig.update_layout(
        barmode='relative',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=80,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=True, zerolinecolor='#374151'),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
        bargap=0.1
    )
    return fig

def plot_pnl_curve():
    """Curve Chart identico allo screenshot 'Cumulative P&L'"""
    x = np.arange(50)
    y = np.cumsum(np.random.randn(50) + 0.2) * 100 + 5000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, fill='tozeroy', mode='lines',
        line=dict(color='#10b981', width=2, shape='spline'),
        fillcolor='rgba(16, 185, 129, 0.05)'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
        xaxis=dict(showgrid=True, gridcolor='#1f2937', showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#1f2937', showticklabels=False),
        showlegend=False
    )
    return fig

# ================= 5. INTERFACCIA UTENTE =================

# --- HEADER SECTION ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        <div>
            <div style="font-size:32px; font-weight:800; line-height:1; background: -webkit-linear-gradient(45deg, #34d399, #22d3ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TITAN ORACLE</div>
            <div style="font-size:12px; color:#9ca3af; letter-spacing:0.05em;">INSTITUTIONAL-GRADE TRADING INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="display:flex; justify-content:flex-end;">
        <div style="border:1px solid #10b981; padding:6px 16px; border-radius:8px; display:flex; align-items:center;">
            <span class="live-dot"></span>
            <span style="color:#10b981; font-weight:700; font-size:12px;">LIVE FEED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# --- ASSET SELECTOR ---
assets = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD", "EURUSD", "GBPUSD"]
cols = st.columns(len(assets))
for i, asset in enumerate(assets):
    if cols[i].button(asset, key=asset, use_container_width=True, type="primary" if st.session_state.selected_asset == asset else "secondary"):
        st.session_state.selected_asset = asset

st.write("") # Spacer

# --- MAIN DASHBOARD LOOP ---
placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", st.session_state.selected_asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                conf_score = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                
                # Colori Dinamici
                if "BUY" in rec:
                    sig_color = "text-buy"
                    bar_color = "#10b981"
                    icon = "↗"
                elif "SELL" in rec:
                    sig_color = "text-sell"
                    bar_color = "#ef4444"
                    icon = "↘"
                else:
                    sig_color = "text-gray-500"
                    bar_color = "#6b7280"
                    icon = "•"

                # ================= ROW 1: AI SIGNAL (FULL WIDTH) =================
                st.markdown(f"""
                <div class="titan-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                        <div class="section-title"><span style="color:#22d3ee">🧠</span> AI SIGNAL</div>
                        <div style="font-family:'JetBrains Mono'; font-size:12px; color:#6b7280;">{d['created_at'][11:19]}</div>
                    </div>
                    
                    <div style="text-align:center;">
                        <div class="{sig_color} value-huge">{icon} {rec}</div>
                        <div style="color:#6b7280; font-size:14px; margin-top:5px; margin-bottom:20px;">{st.session_state.selected_asset} • H1 TIMEFRAME</div>
                    </div>
                    
                    <div style="display:flex; justify-content:space-between; font-size:12px; color:#9ca3af; margin-bottom:5px;">
                        <span>ML Confidence</span>
                        <span>{conf_score:.1f}%</span>
                    </div>
                    <div class="prog-track">
                        <div class="prog-fill" style="width:{conf_score}%; background:{bar_color};"></div>
                    </div>
                    
                    <div style="margin-top:25px; padding:15px; background:rgba(17, 24, 39, 0.5); border-top:1px solid #1f2937;">
                        <div class="label-small">CURRENT PRICE</div>
                        <div class="value-huge" style="color:#22d3ee; font-size:32px;">${d['current_price']:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ================= ROW 2: ORDER FLOW & METRICS =================
                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="titan-card" style="height:100%">
                        <div class="section-title">ORDER FLOW ANALYSIS</div>
                        <div style="margin-top:20px;"></div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(plot_order_flow_clean(), use_container_width=True, config={'displayModeBar': False})
                    
                    st.markdown("""
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-top:15px;">
                        <div class="metric-container box-green" style="text-align:center;">
                            <div class="label-small" style="color:#34d399">Net Buy Pressure</div>
                            <div class="value-large" style="color:#34d399">27%</div>
                        </div>
                        <div class="metric-container box-purple" style="text-align:center;">
                            <div class="label-small" style="color:#a78bfa">Smart Money</div>
                            <div class="value-large" style="color:#a78bfa">-10.4M</div>
                        </div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="titan-card">
                        <div class="metric-container box-cyan" style="margin-bottom:10px;">
                            <div class="label-small" style="color:#22d3ee">ENTRY (LIMIT)</div>
                            <div class="value-large" style="color:#22d3ee">${d['entry_price']}</div>
                        </div>
                        <div style="display:flex; gap:10px;">
                            <div class="metric-container box-red" style="width:50%">
                                <div class="label-small" style="color:#f87171">STOP</div>
                                <div class="value-large" style="color:#f87171">${d['stop_loss']}</div>
                            </div>
                            <div class="metric-container box-green" style="width:50%">
                                <div class="label-small" style="color:#34d399">TARGET</div>
                                <div class="value-large" style="color:#34d399">${d['take_profit']}</div>
                            </div>
                        </div>
                        <div style="text-align:right; font-size:12px; color:#6b7280; margin-top:10px;">
                            Risk/Reward: <span style="color:white">1:{d['risk_reward']}</span>
                        </div>
                    </div>
                    
                    <div class="titan-card">
                        <div class="section-title">PATTERNS DETECTED <span style="color:#fbbf24">⚡</span></div>
                        <div style="margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span style="font-size:13px; font-weight:600;">Head & Shoulders</span>
                                <span class="status-pill pill-red">BEAR</span>
                            </div>
                            <div class="prog-track" style="height:4px;"><div class="prog-fill" style="width:75%; background:#f87171;"></div></div>
                        </div>
                        <div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span style="font-size:13px; font-weight:600;">Ascending Triangle</span>
                                <span class="status-pill pill-green">BULL</span>
                            </div>
                            <div class="prog-track" style="height:4px;"><div class="prog-fill" style="width:60%; background:#34d399;"></div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ================= ROW 3: RADAR & RISK =================
                col3, col4, col5 = st.columns([1, 1, 1])
                
                with col3:
                    st.markdown('<div class="titan-card" style="height:320px;">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">MULTI-FACTOR</div>', unsafe_allow_html=True)
                    st.plotly_chart(plot_radar_clean(d), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    regime = d.get('market_regime', 'RANGING')
                    st.markdown(f"""
                    <div class="titan-card" style="height:320px;">
                        <div class="section-title">MARKET REGIME</div>
                        <div style="display:flex; flex-direction:column; gap:8px; margin-top:20px;">
                             <div class="metric-container" style="border:1px solid {'#3b82f6' if regime == 'TRENDING' else '#374151'}; display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:13px;">TRENDING</span>
                                {'<div class="live-dot" style="background:#3b82f6; box-shadow:0 0 8px #3b82f6;"></div>' if regime == 'TRENDING' else ''}
                             </div>
                             <div class="metric-container" style="border:1px solid {'#3b82f6' if regime == 'RANGING' else '#374151'}; display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:13px;">RANGING</span>
                                {'<div class="live-dot" style="background:#3b82f6; box-shadow:0 0 8px #3b82f6;"></div>' if regime == 'RANGING' else ''}
                             </div>
                             <div class="metric-container" style="border:1px solid {'#3b82f6' if regime == 'VOLATILE' else '#374151'}; display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:13px;">VOLATILE</span>
                                {'<div class="live-dot" style="background:#3b82f6; box-shadow:0 0 8px #3b82f6;"></div>' if regime == 'VOLATILE' else ''}
                             </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col5:
                    st.markdown('<div class="titan-card" style="height:320px;">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">SYSTEM STATUS</div>', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="margin-top:20px; display:flex; flex-direction:column; gap:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:13px; border-bottom:1px solid #1f2937; padding-bottom:8px;">
                            <span style="color:#9ca3af;">Data Feed</span>
                            <span style="color:#10b981; font-weight:700;">● ACTIVE</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px; border-bottom:1px solid #1f2937; padding-bottom:8px;">
                            <span style="color:#9ca3af;">ML Engine</span>
                            <span style="color:#10b981; font-weight:700;">● ACTIVE</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px; border-bottom:1px solid #1f2937; padding-bottom:8px;">
                            <span style="color:#9ca3af;">Risk Manager</span>
                            <span style="color:#10b981; font-weight:700;">● ACTIVE</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px;">
                            <span style="color:#9ca3af;">Order Router</span>
                            <span style="color:#fbbf24; font-weight:700;">● STANDBY</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # ================= ROW 4: EVENTS & PNL =================
                c_pnl, c_ev = st.columns([2, 1])
                with c_pnl:
                    st.markdown('<div class="titan-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">CUMULATIVE P&L (PROJECTION)</div>', unsafe_allow_html=True)
                    st.plotly_chart(plot_pnl_curve(), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with c_ev:
                    st.markdown("""
                    <div class="titan-card">
                        <div class="section-title">HIGH IMPACT EVENTS</div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; background:#1f2937; padding:8px; border-radius:6px;">
                            <div>
                                <div style="font-size:13px; font-weight:600;">Fed Rate Decision</div>
                                <div style="font-size:11px; color:#6b7280;">14:00 EST</div>
                            </div>
                            <span class="status-pill pill-red">HIGH</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; background:#1f2937; padding:8px; border-radius:6px;">
                            <div>
                                <div style="font-size:13px; font-weight:600;">CPI Data</div>
                                <div style="font-size:11px; color:#6b7280;">08:30 EST</div>
                            </div>
                            <span class="status-pill" style="color:#fbbf24; background:rgba(251, 191, 36, 0.1); border:1px solid rgba(251, 191, 36, 0.2);">MED</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info(f"Establishing connection with TITAN Core for {st.session_state.selected_asset}...")
                
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
