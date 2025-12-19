import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="AEGIS Terminal", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    .signal-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 30px; text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .buy-sig { border-color: #2ea043; background: linear-gradient(180deg, #161b22 0%, #0f2d1e 100%); }
    .sell-sig { border-color: #da3633; background: linear-gradient(180deg, #161b22 0%, #3a1010 100%); }
    
    .macro-pill {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        background: #21262d; border: 1px solid #30363d; font-size: 12px; color: #8b949e; margin-top: 10px;
    }
    
    .kpi-val { font-size: 28px; font-weight: bold; color: white; }
    .kpi-lbl { font-size: 12px; color: #8b949e; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

with st.sidebar:
    st.title("🛡️ AEGIS")
    asset = st.radio("ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD"])
    st.info("System: Ensemble AI + Intermarket Correlations")

placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # STILE
                style = ""
                color = "#8b949e"
                if "BUY" in rec: 
                    style = "buy-sig"
                    color = "#2ea043"
                elif "SELL" in rec: 
                    style = "sell-sig"
                    color = "#da3633"
                
                # BOX PRINCIPALE
                st.markdown(f"""
                <div class="signal-card {style}">
                    <h4 style="margin:0; opacity:0.7;">{asset} STRATEGY</h4>
                    <h1 style="font-size: 72px; margin: 10px 0; color: {color};">{rec}</h1>
                    <div style="font-size: 18px; margin-bottom: 10px;">{d['details']}</div>
                    <div class="macro-pill">{d.get('macro_filter', 'ANALYZING...')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # LIVELLI
                if "WAIT" not in rec:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<div style='text-align:center'><div class='kpi-lbl'>ENTRY</div><div class='kpi-val' style='color:#58a6ff'>{d['entry_price']}</div></div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div style='text-align:center'><div class='kpi-lbl'>STOP LOSS</div><div class='kpi-val' style='color:#da3633'>{d['stop_loss']}</div></div>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div style='text-align:center'><div class='kpi-lbl'>TAKE PROFIT</div><div class='kpi-val' style='color:#2ea043'>{d['take_profit']}</div></div>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.success(f"⚡ Risk/Reward Ratio: 1:{d['risk_reward']}")
                else:
                    st.info("Il sistema attende un allineamento tra AI (Tecnica) e Macro (Fondamentali).")

            else:
                st.warning(f"Attesa dati per {asset}...")
        
        time.sleep(1)
    except: time.sleep(1)
