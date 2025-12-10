import streamlit as st
from supabase import create_client
import pandas as pd
import plotly.graph_objects as go
import time

# CONFIGURAZIONE PAGINA
st.set_page_config(page_title="AI Oracle", layout="wide", page_icon="🦅")

# CSS PER LE 3 BARRE E KPI
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .metric-card { background: #1c1f26; padding: 15px; border-radius: 10px; border: 1px solid #333; text-align: center; }
    .rec-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; }
    .buy { background: #00C805; color: white; box-shadow: 0 0 15px rgba(0,200,5,0.4); }
    .sell { background: #FF3B30; color: white; box-shadow: 0 0 15px rgba(255,59,48,0.4); }
    .wait { background: #FFD700; color: black; box-shadow: 0 0 15px rgba(255,215,0,0.4); }
</style>
""", unsafe_allow_html=True)

# CONNESSIONE
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

st.title("🦅 ORACLE AI: Live Trading System")
placeholder = st.empty()

while True:
    try:
        # Prendi ultimi dati
        oracle = supabase.table("ai_oracle").select("*").order("id", desc=True).limit(1).execute()
        feed = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(50).execute()
        
        if oracle.data and feed.data:
            ai = oracle.data[0]
            live = feed.data[0]
            df = pd.DataFrame(feed.data).sort_values('created_at')
            
            with placeholder.container():
                # --- KPI IN ALTO ---
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-card'><h3>ASSET</h3><h1>{live['symbol']}</h1></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><h3>PREZZO LIVE</h3><h1 style='color:#00BFFF'>{live['price']}</h1></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><h3>EQUITY</h3><h1>€ {live['equity']:,.2f}</h1></div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # --- CUORE DEL SISTEMA: GRAFICO + AI ---
                col_chart, col_ai = st.columns([2, 1])
                
                with col_chart:
                    st.subheader("Analisi Mercato")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['created_at'], y=df['price'], mode='lines', line=dict(color='#00BFFF', width=2), fill='tozeroy'))
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
                
                with col_ai:
                    st.subheader("🤖 Previsione AI")
                    
                    # 1. IL VERDETTO
                    rec = ai['recommendation']
                    cls = "wait"
                    if "BUY" in rec: cls = "buy"
                    elif "SELL" in rec: cls = "sell"
                    
                    st.markdown(f"<div class='rec-box {cls}'>{rec}</div>", unsafe_allow_html=True)
                    st.caption(f"Dettagli: {ai['details']}")
                    
                    # 2. LE 3 PROBABILITÀ (Visualizzazione)
                    st.write("Probabilità Scenari:")
                    
                    # Buy
                    st.progress(int(ai['prob_buy']))
                    st.caption(f"🟢 Probabilità RIALZO: {ai['prob_buy']}%")
                    
                    # Hold
                    st.progress(int(ai['prob_hold']))
                    st.caption(f"🟡 Probabilità LATERALE: {ai['prob_hold']}%")
                    
                    # Sell
                    st.progress(int(ai['prob_sell']))
                    st.caption(f"🔴 Probabilità RIBASSO: {ai['prob_sell']}%")
                    
        else:
            st.info("📡 In attesa del Bridge Python... assicurati che stia girando sul PC.")
            
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
