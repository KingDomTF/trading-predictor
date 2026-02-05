import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    PAGE_TITLE = "TITAN Oracle Prime"
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    # REFRESH ACCELERATO: 1 secondo
    AUTO_REFRESH_RATE = 1 

st.set_page_config(page_title=AppConfig.PAGE_TITLE, page_icon=AppConfig.PAGE_ICON, layout=AppConfig.LAYOUT, initial_sidebar_state="collapsed")

def load_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@400;600&display=swap');
* { box-sizing: border-box; }
.main { background-color: #0E1012; background-image: linear-gradient(145deg, #16181C 0%, #0B0D0F 100%); color: #E4E8F0; font-family: 'Inter', sans-serif; }
#MainMenu, footer {visibility: hidden;}
.titan-header { background: #1B1E23; border: 1px solid #2D333B; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center; border-top: 3px solid #69F0AE; }
.titan-title { font-family: 'Rajdhani'; font-size: 2.5rem; font-weight: 700; color: #FFFFFF; margin: 0; }
.stTabs [data-baseweb="tab-list"] { background-color: #16181C; padding: 5px; border-radius: 10px; border: 1px solid #2D333B; justify-content: center; }
.stTabs [data-baseweb="tab"] { color: #888; font-family: 'Rajdhani'; font-weight: 600; font-size: 1rem; }
.stTabs [aria-selected="true"] { background-color: #252930; color: #69F0AE !important; border-radius: 6px; }
.signal-card { background: #1B1E23; border: 1px solid #2D333B; border-radius: 16px; padding: 2rem; margin: 0.5rem auto; max-width: 600px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
.signal-card-buy { border-top: 4px solid #69F0AE; }
.signal-card-sell { border-top: 4px solid #FF5252; }
.signal-symbol { font-size: 1.1rem; color: #69F0AE; font-weight: 600; text-align: center; font-family: 'Rajdhani'; letter-spacing: 2px; }
.price-display { font-size: 3.5rem; font-weight: 700; color: #FFF; text-align: center; font-family: 'Rajdhani'; margin: 10px 0; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 20px; }
.stat-box { background: #23272E; border: 1px solid #2D333B; border-radius: 8px; padding: 12px; text-align: center; }
.stat-label { font-size: 0.65rem; color: #888; text-transform: uppercase; }
.stat-value { font-size: 1.1rem; font-weight: 700; color: #FFF; font-family: 'Rajdhani'; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_supabase():
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
    return create_client(url, key)

supabase = init_supabase()

def get_latest_data(symbol):
    try:
        # Prendi l'ultimo prezzo dalla cronologia (price_history) per zero latenza
        p_res = supabase.table("price_history").select("price, created_at").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        # Prendi l'ultimo segnale (trading_signals)
        s_res = supabase.table("trading_signals").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        return (s_res.data[0] if s_res.data else None), (p_res.data[0] if p_res.data else None)
    except: return None, None

def render_signal_panel(symbol, signal_data, live_price):
    is_market_closed = True
    current_p = 0
    
    if live_price:
        t = datetime.fromisoformat(live_price['created_at'].replace('Z', '+00:00'))
        is_market_closed = (datetime.now(t.tzinfo) - t).total_seconds() > 600
        current_p = live_price['price']

    if is_market_closed:
        st.markdown(f'<div class="signal-card"><div class="signal-symbol">{symbol}</div><div class="price-display" style="color:#555;">MARKET CLOSED</div></div>', unsafe_allow_html=True)
        return

    rec = signal_data['recommendation'] if signal_data else "SCANNING"
    col = "#69F0AE" if rec == "BUY" else "#FF5252" if rec == "SELL" else "#40C4FF"
    cls = "signal-card-buy" if rec == "BUY" else "signal-card-sell" if rec == "SELL" else ""

    st.markdown(f"""
<div class="signal-card {cls}">
<div class="signal-symbol">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:2rem; font-weight:800; color:{col}; text-align:center;">{rec}</div>
<div class="price-display">${current_p:,.2f}</div>
<div class="stats-grid">
<div class="stat-box"><div class="stat-label">ENTRY</div><div class="stat-value">${signal_data['entry_price'] if signal_data else 0:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">CONFIDENCE</div><div class="stat-value">{signal_data['confidence_score'] if signal_data else 0}%</div></div>
<div class="stat-box"><div class="stat-label">TARGET</div><div class="stat-value" style="color:#69F0AE;">${signal_data['take_profit'] if signal_data else 0:,.2f}</div></div>
</div>
</div>
""", unsafe_allow_html=True)

def main():
    load_custom_css()
    st.markdown('<div class="titan-header"><div class="titan-title">TITAN ORACLE PRIME</div></div>', unsafe_allow_html=True)
    tabs = st.tabs(AppConfig.ASSETS)
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            sig, price = get_latest_data(symbol)
            render_signal_panel(symbol, sig, price)
    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__": main()
