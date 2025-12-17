import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="Vantage Point", layout="wide", page_icon="🔭")

st.markdown("""
<style>
    .stApp { background-color: #09090b; color: #e4e4e7; }
    
    /* ZONA DI ATTIVAZIONE */
    .zone-box {
        background: rgba(39, 39, 42, 0.5); border: 2px dashed #52525b;
        border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;
    }
    .zone-label { color: #a1a1aa; font-size: 14px; text-transform: uppercase; }
    .zone-val { font-size: 24px; font-weight: bold; color: #fff; }
    
    /* SEGNALE */
    .rec-box {
        padding: 30px; border-radius: 16px; text-align: center; margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* COLORI */
    .sniper-buy { background: linear-gradient(135deg, #064e3b, #10b981); border: 2px solid #34d399; }
    .sniper-sell { background: linear-gradient(135deg, #7f1d1d, #ef4444); border: 2px solid #f87171; }
    .wait-mode { background: #18181b; border: 2px solid #f59e0b; color: #fbbf24; }
    
    /* CARTE */
    .card { background: #18181b; border: 1px solid #27272a; padding: 15px; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

with st.sidebar:
    st.title("🔭 VANTAGE")
    asset = st.radio("ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD"])
    st.caption("Analisi SMC: Fair Value Gaps + Liquidity Sweeps")

placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # DETERMINA STILE
                css_class = "wait-mode"
                if "SNIPER BUY" in rec or "TREND BUY" in rec: css_class = "sniper-buy"
                elif "SNIPER SELL" in rec or "TREND SELL" in rec: css_class = "sniper-sell"
                
                # --- HEADER ---
                st.markdown(f"""
                <div class="{css_class} rec-box">
                    <h4 style="margin:0; opacity:0.8;">{asset} SIGNAL</h4>
                    <h1 style="font-size: 56px; margin: 10px 0; color:white;">{rec}</h1>
                    <p style="font-size: 18px; color:white;">{d['details']}</p>
                </div>
                """, unsafe_allow_html=True)

                # --- ZONA ISTITUZIONALE (SE PRESENTE) ---
                if d['gap_type'] != "NONE" and "WAIT" in rec:
                    st.markdown(f"""
                    <div class="zone-box">
                        <div class="zone-label">🎯 ZONA DI ATTIVAZIONE (ATTENDI IL PREZZO QUI)</div>
                        <div class="zone-val">{d.get('institutional_zone_bottom', 0)} - {d.get('institutional_zone_top', 0)}</div>
                        <div style="color: #71717a; font-size: 12px; margin-top:5px;">Tipo: {d['gap_type']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- LIVELLI OPERATIVI ---
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"<div class='card'><div style='color:#a1a1aa'>ENTRY (LIMIT)</div><div style='font-size:24px; font-weight:bold; color:#60a5fa'>{d['entry_price']}</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='card'><div style='color:#a1a1aa'>STOP LOSS</div><div style='font-size:24px; font-weight:bold; color:#f87171'>{d['stop_loss']}</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='card'><div style='color:#a1a1aa'>TAKE PROFIT</div><div style='font-size:24px; font-weight:bold; color:#34d399'>{d['take_profit']}</div></div>", unsafe_allow_html=True)
                
                st.success(f"💰 Potenziale Risk/Reward: 1:{d['risk_reward']}")

            else:
                st.info(f"Attesa dati SMC per {asset}...")
        
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
