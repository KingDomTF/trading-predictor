import streamlit as st
from supabase import create_client
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# ================= 1. CONFIGURAZIONE PAGINA =================
st.set_page_config(page_title="TITAN Oracle", layout="wide", page_icon="🛡️")

# ================= 2. STILE CSS (TITAN THEME) =================
# Questo blocco definisce l'aspetto grafico (Colori scuri, Neon, Card)
st.markdown("""
<style>
    /* Sfondo Generale */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* TITAN Card Style */
    .titan-card {
        background-color: #111827; 
        border: 1px solid #1f2937; 
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    
    /* Testi */
    .titan-header {
        font-size: 28px; 
        font-weight: 800; 
        background: -webkit-linear-gradient(45deg, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .titan-sub { color: #9ca3af; font-size: 14px; }
    
    /* Segnali */
    .sig-box { text-align: center; padding: 15px; }
    .sig-text { font-size: 42px; font-weight: 900; letter-spacing: -1px; margin: 10px 0; }
    
    .buy { color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
    .sell { color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
    .wait { color: #6b7280; }
    
    /* Metriche */
    .metric-row { display: flex; justify-content: space-between; gap: 10px; margin-top: 15px; }
    .metric-box {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        width: 100%;
    }
    .m-label { font-size: 10px; color: #9ca3af; text-transform: uppercase; }
    .m-val { font-size: 16px; font-weight: bold; font-family: monospace; }
    
    /* Barra Progresso */
    .bar-bg { width: 100%; background: #374151; height: 8px; border-radius: 4px; margin-top: 10px; overflow: hidden; }
    .bar-fill { height: 100%; transition: width 0.5s; }
    
    /* Colori */
    .c-cyan { color: #22d3ee; }
    .c-red { color: #f87171; }
    .c-green { color: #34d399; }
    .c-orange { color: #fbbf24; }
    
</style>
""", unsafe_allow_html=True)

# ================= 3. CONNESSIONE DATABASE =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.markdown("### 🛡️ CONTROL PANEL")
    asset = st.selectbox("SELECT ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"])
    st.markdown("---")
    st.success("System: ONLINE")
    st.caption("Mode: TITAN v4.1 (Stable)")

# ================= 5. FUNZIONI GRAFICHE (PLOTLY) =================
def create_radar(d):
    """Crea il grafico a ragnatela"""
    conf = d.get('prob_buy', 50) if "BUY" in d['recommendation'] else d.get('prob_sell', 50)
    z = abs(d.get('confidence_score', 0))
    
    # Dati simulati basati sui valori reali per creare la forma
    categories = ['Trend', 'Momentum', 'Macro', 'Volume', 'Volatility']
    values = [
        min(conf + 10, 95),      # Trend
        min(conf, 90),           # Momentum
        60 if "BUY" in d['recommendation'] else 40, # Macro
        75,                      # Volume
        min(z * 30, 90)          # Volatility
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', 
        line_color='#8b5cf6', fillcolor='rgba(139, 92, 246, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#4b5563'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=250,
        font=dict(color='white')
    )
    return fig

# ================= 6. LOOP PRINCIPALE =================
placeholder = st.empty()

while True:
    try:
        # Preleva dati
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # Variabili per l'HTML
                conf_score = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                z_score = d.get('confidence_score', 0)
                regime = d.get('market_regime', 'ANALYZING')
                
                # Colori dinamici
                sig_class = "buy" if "BUY" in rec else "sell" if "SELL" in rec else "wait"
                bar_color = "#10b981" if "BUY" in rec else "#ef4444" if "SELL" in rec else "#6b7280"
                
                # --- HEADER ---
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""
                    <div>
                        <div class="titan-header">TITAN ORACLE</div>
                        <div class="titan-sub">Institutional Intelligence • {asset}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div style="text-align:right; color:#10b981; border:1px solid #10b981; padding:5px; border-radius:15px; font-size:12px;">● LIVE CONNECTION</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                # --- COLONNE CONTENUTO ---
                col_left, col_right = st.columns([1, 1.3])
                
                # === SINISTRA: SEGNALE ===
                with col_left:
                    # HTML Card Costruita come stringa f-string
                    html_signal = f"""
                    <div class="titan-card">
                        <div style="display:flex; justify-content:space-between; color:#9ca3af; font-size:12px;">
                            <span>AI CONFIDENCE</span>
                            <span>{conf_score:.1f}%</span>
                        </div>
                        
                        <div class="sig-box">
                            <div class="sig-text {sig_class}">{rec}</div>
                            <div style="color:#9ca3af; font-size:14px;">{d['details']}</div>
                        </div>
                        
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: {conf_score}%; background-color: {bar_color};"></div>
                        </div>
                        
                        <div class="metric-row">
                            <div class="metric-box" style="border-color: rgba(34, 211, 238, 0.3);">
                                <div class="m-label c-cyan">ENTRY</div>
                                <div class="m-val c-cyan">{d['entry_price']}</div>
                            </div>
                            <div class="metric-box" style="border-color: rgba(248, 113, 113, 0.3);">
                                <div class="m-label c-red">STOP</div>
                                <div class="m-val c-red">{d['stop_loss']}</div>
                            </div>
                            <div class="metric-box" style="border-color: rgba(52, 211, 153, 0.3);">
                                <div class="m-label c-green">TARGET</div>
                                <div class="m-val c-green">{d['take_profit']}</div>
                            </div>
                        </div>
                        
                        <div style="margin-top:15px; text-align:center; font-size:12px; color:#6b7280;">
                            Risk/Reward Ratio: <span style="color:white">1:{d['risk_reward']}</span>
                        </div>
                    </div>
                    """
                    st.markdown(html_signal, unsafe_allow_html=True)
                    
                    # Risk Metrics Card
                    st.markdown(f"""
                    <div class="titan-card">
                        <div style="color:white; font-weight:bold; margin-bottom:10px;">RISK METRICS</div>
                        <div class="metric-row">
                            <div class="metric-box">
                                <div class="m-label">Z-SCORE</div>
                                <div class="m-val c-orange">{z_score} σ</div>
                            </div>
                            <div class="metric-box">
                                <div class="m-label">REGIME</div>
                                <div class="m-val c-cyan">{regime}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # === DESTRA: GRAFICI ===
                with col_right:
                    st.markdown('<div class="titan-card" style="height: 100%;">', unsafe_allow_html=True)
                    st.caption("MULTI-FACTOR ANALYSIS")
                    st.plotly_chart(create_radar(d), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.warning(f"Waiting for data on {asset}...")
                
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
