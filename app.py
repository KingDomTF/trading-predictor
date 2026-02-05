import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.error("❌ Librerie mancanti.")
    st.stop()

class AppConfig:
    PAGE_TITLE = "TITAN Oracle Prime"
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    # AGGIORNAMENTO RAPIDO: 1 secondo
    AUTO_REFRESH_RATE = 1 

st.set_page_config(page_title=AppConfig.PAGE_TITLE, page_icon=AppConfig.PAGE_ICON, layout=AppConfig.LAYOUT, initial_sidebar_state="collapsed")

def load_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@400;600&display=swap');
* { box-sizing: border-box; }
.main { background-color: #0E1012; background-image: linear-gradient(145deg, #16181C 0%, #0B0D0F 100%); color: #E4E8F0; font-family: 'Inter', sans-serif; }
#MainMenu, footer {visibility: hidden;}
.titan-header { background: #1B1E23; border: 1px solid #2D333B; border-radius: 16px; padding: 2rem; margin-bottom: 2rem; text-align: center; border-top: 3px solid #69F0AE; }
.titan-title { font-family: 'Rajdhani'; font-size: 3rem; font-weight: 700; color: #FFFFFF; margin: 0; }
.titan-subtitle { font-family: 'Rajdhani'; font-size: 1rem; color: #69F0AE; text-transform: uppercase; letter-spacing: 2px; }
.stTabs [data-baseweb="tab-list"] { background-color: #16181C; padding: 8px; border-radius: 12px; border: 1px solid #2D333B; justify-content: center; }
.stTabs [data-baseweb="tab"] { color: #888; font-family: 'Rajdhani'; font-weight: 600; font-size: 1.1rem; }
.stTabs [aria-selected="true"] { background-color: #252930; color: #69F0AE !important; border-radius: 8px; }
.signal-card { background: #1B1E23; border: 1px solid #2D333B; border-radius: 16px; padding: 2.5rem; margin: 1rem auto; max-width: 700px; box-shadow: 0 15px 35px rgba(0,0,0,0.3); }
.signal-card-buy { border-top: 4px solid #69F0AE; }
.signal-card-sell { border-top: 4px solid #FF5252; }
.signal-symbol { font-size: 1.2rem; color: #69F0AE; font-weight: 600; text-align: center; font-family: 'Rajdhani'; }
.price-display { font-size: 4rem; font-weight: 700; color: #FFF; text-align: center; font-family: 'Rajdhani'; margin: 15px 0; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 25px; }
.stat-box { background: #23272E; border: 1px solid #2D333B; border-radius: 10px; padding: 15px; text-align: center; }
.stat-label { font-size: 0.7rem; color: #888; text-transform: uppercase; }
.stat-value { font-size: 1.2rem; font-weight: 700; color: #FFF; font-family: 'Rajdhani'; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_supabase():
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key: return None
    return create_client(url, key)

supabase = init_supabase()

def get_latest_data(symbol):
    if not supabase: return None, None
    try:
        # Prendi l'ultimo segnale
        sig_res = supabase.table("trading_signals").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        # Prendi il prezzo più recente in assoluto (Tabella price_history)
        price_res = supabase.table("price_history").select("price, created_at").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        
        signal = sig_res.data[0] if sig_res.data else None
        live_price = price_res.data[0] if price_res.data else None
        return signal, live_price
    except: return None, None

def render_signal_panel(symbol, signal_data, live_price_data):
    # Controllo freschezza dato (se il prezzo è più vecchio di 10 min, mercato chiuso)
    is_stale = True
    current_p = 0
    if live_price_data:
        t = datetime.fromisoformat(live_price_data['created_at'].replace('Z', '+00:00'))
        diff = (datetime.now(t.tzinfo) - t).total_seconds()
        is_stale = diff > 600
        current_p = live_price_data['price']

    if is_stale:
        st.markdown(f"""
<div class="signal-card" style="border-top: 4px solid #444;">
<div class="signal-symbol">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:2.5rem; font-weight:800; color:#555; text-align:center;">MARKET CLOSED</div>
<div class="price-display" style="color:#444;">${current_p:,.2f}</div>
<div style="text-align:center; color:#444; font-size:0.8rem;">Check Bridge Status</div>
</div>
""", unsafe_allow_html=True)
        return

    # Visualizza segnale attivo con prezzo in tempo reale
    rec = signal_data.get('recommendation', 'SCANNING') if signal_data else 'SCANNING'
    sl = signal_data.get('stop_loss', 0) if signal_data else 0
    tp = signal_data.get('take_profit', 0) if signal_data else 0
    conf = signal_data.get('confidence_score', 0) if signal_data else 0
    
    col = "#69F0AE" if rec == 'BUY' else "#FF5252" if rec == 'SELL' else "#40C4FF"
    cls = "signal-card-buy" if rec == 'BUY' else "signal-card-sell" if rec == 'SELL' else ""

    st.markdown(f"""
<div class="signal-card {cls}">
<div class="signal-symbol">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:3rem; font-weight:800; color:{col}; text-align:center;">{rec}</div>
<div class="price-display">${current_p:,.2f}</div>
<div style="background:#23272E; border-radius:8px; padding:10px; margin: 20px 0;">
<div style="display:flex; justify-content:space-between; margin-bottom:5px;">
<span style="color:#888; font-size:0.8rem;">AI CONFIDENCE</span>
<span style="color:{col}; font-weight:700;">{conf}%</span>
</div>
<div style="background:#333; height:6px; border-radius:3px;">
<div style="background:{col}; width:{conf}%; height:100%; border-radius:3px;"></div>
</div>
</div>
<div class="stats-grid">
<div class="stat-box"><div class="stat-label">ENTRY</div><div class="stat-value" style="color:#40C4FF;">${signal_data.get('entry_price',0) if signal_data else 0:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">STOP LOSS</div><div class="stat-value" style="color:#FF5252;">${sl:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">TARGET</div><div class="stat-value" style="color:#69F0AE;">${tp:,.2f}</div></div>
</div>
</div>
""", unsafe_allow_html=True)

def main():
    load_custom_css()
    st.markdown('<div class="titan-header"><div class="titan-title">TITAN ORACLE</div><div class="titan-subtitle">Ultra-Fast Institutional Feed</div></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(AppConfig.ASSETS)
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            signal, live_price = get_latest_data(symbol)
            render_signal_panel(symbol, signal, live_price)

    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__": main()
