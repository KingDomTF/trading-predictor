import streamlit as st
from supabase import create_client
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# CONFIGURAZIONE
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

st.set_page_config(page_title="Oracle AI Trader", layout="wide", page_icon="🔮")

# CSS PERSONALIZZATO (Stile "Palantir")
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .rec-box { padding: 20px; border-radius: 10px; text-align: center; color: white; }
    .buy { background-color: #00C805; }
    .sell { background-color: #FF3B30; }
    .wait { background-color: #FFCC00; color: black; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_db():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_db()

# HEADER
st.title("🔮 Oracle AI: Strategic Decision System")

# LAYOUT
col_kpi, col_oracle = st.columns([1, 2])
chart_spot = st.empty()

while True:
    try:
        # 1. PRENDI DATI LIVE (Prezzo)
        feed_res = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(1).execute()
        
        # 2. PRENDI DATI ORACOLO (Predizioni)
        oracle_res = supabase.table("ai_oracle").select("*").order("id", desc=True).limit(1).execute()
        
        if feed_res.data:
            live = feed_res.data[0]
            
            # --- COLONNA SINISTRA: DATI DI MERCATO ---
            with col_kpi:
                st.subheader(f"Asset: {live['symbol']}")
                st.metric("Live Price", f"{live['price']}")
                st.metric("Account Equity", f"€ {live['equity']}")
                
                # Grafico rapido ultimi 50 tick
                hist_res = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(50).execute()
                df_hist = pd.DataFrame(hist_res.data).sort_values('created_at')
                st.line_chart(df_hist, x='created_at', y='price', height=200)

            # --- COLONNA DESTRA: LE 3 SCELTE (ORACLE) ---
            with col_oracle:
                st.subheader("🤖 AI Strategic Analysis")
                
                if oracle_res.data:
                    pred = oracle_res.data[0]
                    
                    # BOX RACCOMANDAZIONE PRINCIPALE
                    rec = pred['recommendation']
                    color_class = "wait"
                    if "BUY" in rec: color_class = "buy"
                    elif "SELL" in rec: color_class = "sell"
                    
                    st.markdown(f"""
                    <div class="rec-box {color_class}">
                        <h1>{rec}</h1>
                        <p>{pred['analysis_depth']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # LE TRE PROBABILITÀ (Visualizzazione a Barre)
                    c1, c2, c3 = st.columns(3)
                    
                    # 1. SCENARIO LONG
                    c1.markdown("### 🟢 LONG")
                    c1.progress(int(pred['prob_buy']))
                    c1.caption(f"Probabilità: {pred['prob_buy']}%")
                    
                    # 2. SCENARIO HOLD
                    c2.markdown("### 🟡 HOLD")
                    c2.progress(int(pred['prob_hold']))
                    c2.caption(f"Probabilità: {pred['prob_hold']}%")
                    
                    # 3. SCENARIO SHORT
                    c3.markdown("### 🔴 SHORT")
                    c3.progress(int(pred['prob_sell']))
                    c3.caption(f"Probabilità: {pred['prob_sell']}%")
                    
                    st.info(f"Ultima analisi AI: {pd.to_datetime(pred['created_at']).strftime('%H:%M:%S')}")
                else:
                    st.warning("L'AI sta analizzando lo storico... attendere.")

        time.sleep(1)

    except Exception as e:
        time.sleep(1)
