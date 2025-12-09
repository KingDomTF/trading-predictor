import streamlit as st
from supabase import create_client
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# ================= CONFIGURAZIONE =================
st.set_page_config(
    page_title="AI Trading Oracle",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

# CREDENZIALI (Hardcoded per tua comodità immediata)
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

# ================= STILE CSS "INSTITUTIONAL" =================
st.markdown("""
<style>
    /* Sfondo Scuro Profondo */
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    
    /* Box KPI */
    .kpi-box {
        background: #1E2329;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .kpi-title { font-size: 14px; color: #888; letter-spacing: 1.5px; text-transform: uppercase; }
    .kpi-val { font-size: 32px; font-weight: 700; color: #FFF; margin: 10px 0; }
    
    /* Box Segnale AI */
    .signal-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        animation: pulse 2s infinite;
    }
    
    /* Colori Segnali */
    .sig-buy { background: linear-gradient(135deg, #004d26 0%, #00C805 100%); box-shadow: 0 0 25px rgba(0, 200, 5, 0.3); }
    .sig-sell { background: linear-gradient(135deg, #4d0000 0%, #FF3B30 100%); box-shadow: 0 0 25px rgba(255, 59, 48, 0.3); }
    .sig-wait { background: linear-gradient(135deg, #4d3a00 0%, #FFD700 100%); box-shadow: 0 0 25px rgba(255, 215, 0, 0.3); color: black !important; }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    
    /* Barre Probabilità */
    .prob-container { background: #333; border-radius: 10px; height: 24px; width: 100%; position: relative; overflow: hidden; margin-top: 5px;}
    .prob-bar { height: 100%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; text-shadow: 0 1px 2px black; transition: width 0.5s; }
    
</style>
""", unsafe_allow_html=True)

# ================= CONNESSIONE DB =================
@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    supabase = init_connection()
except:
    st.error("Connessione Database fallita.")
    st.stop()

# ================= LOOP PRINCIPALE =================
placeholder = st.empty()

while True:
    try:
        # 1. Recupera ultimi dati (Prezzi + Predizioni)
        feed = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(60).execute()
        oracle = supabase.table("ai_oracle").select("*").order("id", desc=True).limit(1).execute()
        
        if feed.data:
            df = pd.DataFrame(feed.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at')
            live = df.iloc[-1]
            
            with placeholder.container():
                
                # HEADER
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"## 🦅 {live['symbol']} LIVE ANALYSIS")
                with c2:
                    st.markdown(f"*{datetime.now().strftime('%H:%M:%S UTC')}*")

                # --- SEZIONE 1: DATI LIVE (KPI) ---
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown(f"""<div class="kpi-box"><div class="kpi-title">MARKET PRICE</div>
                    <div class="kpi-val" style="color: #00BFFF;">{live['price']:.5f}</div></div>""", unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""<div class="kpi-box"><div class="kpi-title">EQUITY</div>
                    <div class="kpi-val">€ {live['equity']:,.2f}</div></div>""", unsafe_allow_html=True)
                with k3:
                    # Recupera stato AI
                    rec = "INITIALIZING..."
                    color = "#555"
                    if oracle.data:
                        rec = oracle.data[0]['recommendation']
                        if "BUY" in rec: color = "#00C805"
                        elif "SELL" in rec: color = "#FF3B30"
                        elif "WAIT" in rec: color = "#FFD700"
                    
                    st.markdown(f"""<div class="kpi-box" style="border-color: {color};">
                    <div class="kpi-title">AI VERDICT</div>
                    <div class="kpi-val" style="color: {color}; font-size: 24px;">{rec}</div></div>""", unsafe_allow_html=True)

                st.markdown("---")

                # --- SEZIONE 2: CERVELLO AI (LE 3 SCELTE) ---
                col_graph, col_ai = st.columns([2, 1])

                with col_graph:
                    # Grafico Prezzo
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['created_at'], y=df['price'], mode='lines', 
                                           line=dict(color='#00BFFF', width=2), fill='tozeroy', fillcolor='rgba(0,191,255,0.1)'))
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        height=400, margin=dict(l=0,r=0,t=10,b=0), xaxis_showgrid=False
                    )
                    # Key univoca per evitare errore ID
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")

                with col_ai:
                    if oracle.data:
                        pred = oracle.data[0]
                        p_buy = float(pred['prob_buy'])
                        p_sell = float(pred['prob_sell'])
                        p_hold = float(pred['prob_hold'])
                        
                        # BOX SEGNALE GRANDE
                        css = "sig-wait"
                        if "BUY" in pred['recommendation']: css = "sig-buy"
                        elif "SELL" in pred['recommendation']: css = "sig-sell"
                        
                        st.markdown(f"""
                        <div class="signal-card {css}">
                            <h2 style="margin:0; font-size: 36px;">{pred['recommendation']}</h2>
                            <p style="margin:5px 0 0 0; opacity: 0.9;">{pred['analysis_depth']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 🤖 Previsioni Modello")
                        
                        # BARRA BUY
                        st.markdown(f"**LONG (Buy)** - {p_buy}%")
                        st.markdown(f"""<div class="prob-container"><div class="prob-bar" style="width: {p_buy}%; background: #00C805;"></div></div>""", unsafe_allow_html=True)
                        
                        # BARRA HOLD
                        st.markdown(f"**NEUTRAL (Wait)** - {p_hold}%")
                        st.markdown(f"""<div class="prob-container"><div class="prob-bar" style="width: {p_hold}%; background: #FFD700; color: black;"></div></div>""", unsafe_allow_html=True)
                        
                        # BARRA SELL
                        st.markdown(f"**SHORT (Sell)** - {p_sell}%")
                        st.markdown(f"""<div class="prob-container"><div class="prob-bar" style="width: {p_sell}%; background: #FF3B30;"></div></div>""", unsafe_allow_html=True)
                        
                        st.caption(f"Confidence Model: Random Forest | Data Source: Yahoo Finance + MT4 Live")
                        
                    else:
                        st.info("🔄 L'AI sta analizzando lo storico... (Attendere il Bridge Python)")
                
                # DISCLAIMER (Come da tuo file)
                st.markdown("---")
                st.markdown("""
                <div style='text-align: center; color: #666; font-size: 12px; padding: 10px;'>
                    ⚠️ <strong>DISCLAIMER ISTITUZIONALE:</strong> Questo strumento utilizza algoritmi di Intelligenza Artificiale per l'analisi statistica. 
                    Non costituisce sollecitazione all'investimento. Il trading comporta rischi di perdita del capitale.
                    L'utente è l'unico responsabile delle decisioni operative.
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("In attesa di connessione dal PC locale...")
            
        time.sleep(1)

    except Exception as e:
        time.sleep(1)
