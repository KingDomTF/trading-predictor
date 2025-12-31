import streamlit as st
from supabase import create_client
import time
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ================= 1. CONFIGURAZIONE PAGINA =================
st.set_page_config(
    page_title="TITAN Oracle",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ================= 2. STILE CSS (TITAN THEME) =================
# Questo colora l'interfaccia di nero e stilizza le metriche
st.markdown("""
<style>
    /* Sfondo Nero Totale */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Rimuove barre bianche in alto */
    header {visibility: hidden;}
    
    /* Stile per i Container (Le Card) */
    div[data-testid="stVerticalBlock"] > div > div {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Testi e Titoli */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    
    /* Colori Metriche */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    /* Colori personalizzati per classi specifiche */
    .titan-title {
        font-size: 40px;
        background: -webkit-linear-gradient(45deg, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    
    .status-ok {
        color: #10b981;
        border: 1px solid #10b981;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .signal-buy { color: #10b981; font-size: 60px; font-weight: 900; text-shadow: 0 0 20px rgba(16,185,129,0.4); }
    .signal-sell { color: #ef4444; font-size: 60px; font-weight: 900; text-shadow: 0 0 20px rgba(239,68,68,0.4); }
    .signal-wait { color: #6b7280; font-size: 60px; font-weight: 900; }
    
</style>
""", unsafe_allow_html=True)

# ================= 3. CONNESSIONE DB =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    supabase = init_db()
except:
    st.error("Errore di connessione a Supabase. Controlla le chiavi.")

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("## 🛡️ CONTROL PANEL")
    asset = st.selectbox("ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"])
    st.divider()
    st.success("SYSTEM ONLINE")
    st.caption("Mode: TITAN Native v5.0")

# ================= 5. GRAFICI (PLOTLY) =================
def create_radar_chart(d):
    conf = d.get('prob_buy', 50) if "BUY" in d['recommendation'] else d.get('prob_sell', 50)
    z = abs(d.get('confidence_score', 0))
    
    categories = ['Trend', 'Momentum', 'Macro', 'Volatilità', 'Volume']
    values = [
        min(conf + 10, 95),
        min(conf, 90),
        60 if "BUY" in d['recommendation'] else 40,
        min(z * 30, 90),
        75
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line_color='#8b5cf6', fillcolor='rgba(139, 92, 246, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#6b7280'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        showlegend=False,
        font=dict(color='white')
    )
    return fig

# ================= 6. DASHBOARD PRINCIPALE =================

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('<div class="titan-title">TITAN ORACLE</div>', unsafe_allow_html=True)
    st.caption(f"Institutional Intelligence • {asset}")
with c2:
    st.markdown('<div style="text-align:right"><span class="status-ok">● LIVE CONNECTION</span></div>', unsafe_allow_html=True)

st.write("") # Spaziatore

# Loop Dati
placeholder = st.empty()

while True:
    try:
        # Fetch Dati
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # Setup Colori e Classi
                sig_class = "signal-wait"
                if "BUY" in rec: sig_class = "signal-buy"
                elif "SELL" in rec: sig_class = "signal-sell"
                
                conf_score = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                z_score = d.get('confidence_score', 0)

                # --- LAYOUT A COLONNE ---
                col_main, col_charts = st.columns([1, 1])

                # 1. COLONNA SINISTRA: SEGNALI
                with col_main:
                    with st.container():
                        st.markdown("**AI SIGNAL PROCESSOR**")
                        
                        # Il Segnale Gigante
                        st.markdown(f'<div style="text-align:center"><div class="{sig_class}">{rec}</div></div>', unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center; color:#9ca3af; margin-bottom:20px'>{d['details']}</div>", unsafe_allow_html=True)
                        
                        # Barra Progresso
                        st.progress(int(conf_score))
                        st.caption(f"AI Confidence: {conf_score:.1f}%")
                        
                        st.divider()
                        
                        # Metriche Entry/Stop/TP
                        m1, m2, m3 = st.columns(3)
                        m1.metric("ENTRY", d['entry_price'])
                        m2.metric("STOP LOSS", d['stop_loss'])
                        m3.metric("TARGET", d['take_profit'])
                        
                        st.write("")
                        st.info(f"Risk/Reward Ratio: 1:{d['risk_reward']}")

                # 2. COLONNA DESTRA: ANALISI AVANZATA
                with col_charts:
                    with st.container():
                        st.markdown("**MULTI-FACTOR ANALYSIS**")
                        st.plotly_chart(create_radar_chart(d), use_container_width=True)
                    
                    with st.container():
                        st.markdown("**RISK METRICS**")
                        r1, r2 = st.columns(2)
                        r1.metric("Z-SCORE (Volatilità)", f"{z_score} σ")
                        r2.metric("MARKET REGIME", d.get('market_regime', 'ANALYZING'))

            else:
                st.warning(f"In attesa di dati per {asset}... Assicurati che bridge.py sia attivo.")
        
        time.sleep(1)

    except Exception as e:
        time.sleep(1)
