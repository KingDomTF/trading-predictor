import streamlit as st
from supabase import create_client
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# ==============================================================================
# 1. CONFIGURAZIONE & CONNESSIONE
# ==============================================================================
st.set_page_config(
    page_title="AI QUANT TERMINAL",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

# CREDENZIALI (In produzione usa st.secrets, qui hardcoded per velocità)
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    supabase = init_connection()
except:
    st.error("Errore critico di connessione al Database.")
    st.stop()

# ==============================================================================
# 2. STILE "PALANTIR" (CSS AVANZATO)
# ==============================================================================
st.markdown("""
<style>
    /* Sfondo Generale Scuro */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Card KPI Personalizzate */
    .kpi-card {
        background-color: #1E2329;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 24px; font-weight: bold; color: #FFF; margin-top: 5px; }
    .kpi-sub   { font-size: 10px; color: #00C805; }
    
    /* Box Segnale AI */
    .signal-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .signal-buy { background: linear-gradient(135deg, #054d24 0%, #00C805 100%); box-shadow: 0 0 15px rgba(0,200,5,0.4); }
    .signal-sell { background: linear-gradient(135deg, #4d0505 0%, #FF3B30 100%); box-shadow: 0 0 15px rgba(255,59,48,0.4); }
    .signal-wait { background: linear-gradient(135deg, #4d3a05 0%, #FFCC00 100%); box-shadow: 0 0 15px rgba(255,204,0,0.4); }
    
    .signal-title { font-size: 30px; font-weight: 900; color: #FFF; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    .signal-desc { font-size: 14px; color: rgba(255,255,255,0.9); margin-top: 5px; }

    /* Barre Probabilità */
    .prob-bar-container { background-color: #262626; border-radius: 5px; margin-top: 5px; height: 8px; width: 100%; }
    .prob-fill { height: 100%; border-radius: 5px; }
    
    /* Rimuovi padding inutili */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. INTERFACCIA UTENTE
# ==============================================================================

# Header
col_logo, col_title, col_time = st.columns([1, 4, 1])
with col_title:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>QUANTUM AI MONITOR</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #666;'>Live Connection: {SUPABASE_URL.split('//')[1][:10]}...</p>", unsafe_allow_html=True)

# Loop principale
placeholder = st.empty()

while True:
    try:
        # --- 1. FETCH DATI (LIVE FEED) ---
        feed_res = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(60).execute()
        
        # --- 2. FETCH DATI (AI ORACLE) ---
        oracle_res = supabase.table("ai_oracle").select("*").order("id", desc=True).limit(1).execute()

        if feed_res.data:
            df = pd.DataFrame(feed_res.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at')
            last_tick = df.iloc[-1]

            with placeholder.container():
                
                # --- RIGA 1: KPI CARDS ---
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">ASSET</div>
                        <div class="kpi-value" style="color: #FFD700;">{last_tick['symbol']}</div>
                    </div>""", unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">LIVE PRICE</div>
                        <div class="kpi-value">{last_tick['price']:.5f}</div>
                    </div>""", unsafe_allow_html=True)
                
                with c3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">EQUITY</div>
                        <div class="kpi-value">€ {last_tick['equity']:,.2f}</div>
                    </div>""", unsafe_allow_html=True)
                
                with c4:
                    # Recupera l'ultimo stato AI
                    ai_status = "ANALYZING..."
                    ai_color = "#888"
                    if oracle_res.data:
                         ai_status = oracle_res.data[0]['recommendation']
                         if "BUY" in ai_status: ai_color = "#00C805"
                         elif "SELL" in ai_status: ai_color = "#FF3B30"
                         elif "WAIT" in ai_status: ai_color = "#FFCC00"

                    st.markdown(f"""
                    <div class="kpi-card" style="border-color: {ai_color};">
                        <div class="kpi-label">AI STATUS</div>
                        <div class="kpi-value" style="color: {ai_color}; font-size: 20px;">{ai_status}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # --- RIGA 2: DASHBOARD ORACOLO & GRAFICO ---
                col_chart, col_oracle = st.columns([2, 1])

                # SINISTRA: GRAFICO
                with col_chart:
                    st.subheader("📊 Market Structure")
                    fig = go.Figure(data=[go.Scatter(
                        x=df['created_at'], 
                        y=df['price'], 
                        mode='lines', 
                        line=dict(color='#00BFFF', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 191, 255, 0.1)'
                    )])
                    
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=400,
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='#333')
                    )
                    
                    # Chiave univoca per evitare errori di ID
                    unique_id = f"chart_{datetime.now().timestamp()}"
                    st.plotly_chart(fig, use_container_width=True, key=unique_id)

                # DESTRA: ANALISI AI
                with col_oracle:
                    st.subheader("🧠 Neural Network Output")
                    
                    if oracle_res.data:
                        pred = oracle_res.data[0]
                        
                        # 1. IL SEGNALE PRINCIPALE
                        rec = pred['recommendation']
                        css_class = "signal-wait"
                        if "BUY" in rec: css_class = "signal-buy"
                        elif "SELL" in rec: css_class = "signal-sell"
                        
                        st.markdown(f"""
                        <div class="signal-box {css_class}">
                            <div class="signal-title">{rec}</div>
                            <div class="signal-desc">{pred['analysis_depth']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 2. PROBABILITA' (LE 3 BARRE)
                        st.markdown("**Probabilità Scenari:**")
                        
                        # BUY
                        st.caption(f"BULLISH (Buy): {pred['prob_buy']}%")
                        st.progress(int(pred['prob_buy']))
                        
                        # HOLD
                        st.caption(f"NEUTRAL (Hold): {pred['prob_hold']}%")
                        st.progress(int(pred['prob_hold']))
                        
                        # SELL
                        st.caption(f"BEARISH (Sell): {pred['prob_sell']}%")
                        st.progress(int(pred['prob_sell']))
                        
                        st.caption(f"Last update: {pd.to_datetime(pred['created_at']).strftime('%H:%M:%S')}")

                    else:
                        st.info("Attesa dati dal Neural Engine (Python Bridge)...")
                        st.progress(0)

                # --- RIGA 3: DATI GREZZI ---
                with st.expander("💾 Raw Data Feed (Live)", expanded=False):
                    st.dataframe(df.sort_values('created_at', ascending=False), use_container_width=True)

        else:
            st.warning("📡 Waiting for data stream from MT4 Bridge...")
        
        time.sleep(1)

    except Exception as e:
        # Silenzia errori minori durante il refresh
        time.sleep(1)
