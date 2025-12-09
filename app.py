import streamlit as st
from supabase import create_client
import pandas as pd
import time
import json
import plotly.express as px
from datetime import datetime

# ================= CONFIGURAZIONE =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

st.set_page_config(page_title="AI Trading Oracle", layout="wide", page_icon="🧿")

# CSS Personalizzato per le Card
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .buy-signal { color: #00ff00; font-weight: bold; }
    .sell-signal { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# ================= LAYOUT PRINCIPALE =================
st.title("🧿 ORACOL: Live AI Scalping System")

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
chart_place = st.empty()
st.markdown("---")
st.subheader("⚡ TACTICAL SCALPING OPTIONS")
options_place = st.empty()

while True:
    try:
        # Scarica l'ultimo dato
        response = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(30).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            last = df.sort_values('created_at').iloc[-1]
            
            # 1. KPI
            col_kpi1.metric("Asset", last['symbol'])
            col_kpi2.metric("Live Price", f"{last['price']:.2f}")
            col_kpi3.metric("Account Equity", f"€ {last['equity']:.2f}")

            # 2. GRAFICO
            fig = px.line(df, x='created_at', y='price', title=f"{last['symbol']} Live Trend")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            chart_place.plotly_chart(fig, use_container_width=True, key=f"ch_{time.time()}")

            # 3. LE 3 SCELTE DI TRADING (Decodifica JSON)
            try:
                # Il campo comment contiene il JSON con le strategie
                ai_data = json.loads(last['comment'])
                
                direction = ai_data.get("direction", "N/A")
                confidence = ai_data.get("confidence", 0)
                options = ai_data.get("options", [])
                
                # Colore segnale
                color_class = "buy-signal" if "LONG" in direction else "sell-signal"
                
                with options_place.container():
                    st.markdown(f"### 🔮 PREDICTION: <span class='{color_class}'>{direction}</span> (Conf: {confidence:.0%})", unsafe_allow_html=True)
                    
                    if options:
                        c1, c2, c3 = st.columns(3)
                        # Mostriamo le 3 opzioni come card
                        for i, col in enumerate([c1, c2, c3]):
                            opt = options[i]
                            with col:
                                st.error(f"{opt['type']}") if "SHORT" in direction else st.success(f"{opt['type']}")
                                st.markdown(f"**ENTRY:** {opt['entry']}")
                                st.markdown(f"**🎯 TP:** {opt['tp']}")
                                st.markdown(f"**🛡️ SL:** {opt['sl']}")
                                st.caption(f"*{opt['desc']}*")
            except:
                with options_place.container():
                    st.info("🧠 L'AI sta analizzando i dati storici... attendere.")

        time.sleep(1)

    except Exception as e:
        time.sleep(1)
