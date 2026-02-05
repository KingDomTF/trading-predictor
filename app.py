import os, time, pandas as pd, streamlit as st
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    PAGE_TITLE = "TITAN Oracle Prime"
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    AUTO_REFRESH_RATE = 1 

st.set_page_config(page_title=AppConfig.PAGE_TITLE, layout="wide", initial_sidebar_state="collapsed")

def load_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap');
.main { background-color: #0E1012; color: #E4E8F0; font-family: 'Inter', sans-serif; }
.titan-header { background: #1B1E23; border-radius: 16px; padding: 1.5rem; text-align: center; border-top: 3px solid #69F0AE; margin-bottom: 2rem; }
.signal-card { background: #1B1E23; border-radius: 16px; padding: 2rem; margin: 0 auto; max-width: 600px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
.price-display { font-size: 3.5rem; font-weight: 700; color: #FFF; text-align: center; font-family: 'Rajdhani'; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 20px; }
.stat-box { background: #23272E; border-radius: 8px; padding: 12px; text-align: center; border: 1px solid #2D333B; }
.stat-label { font-size: 0.65rem; color: #888; text-transform: uppercase; }
.stat-value { font-size: 1.1rem; font-weight: 700; color: #FFF; font-family: 'Rajdhani'; }
</style>
""", unsafe_allow_html=True)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def get_data(symbol):
    try:
        p = supabase.table("price_history").select("price, created_at").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        s = supabase.table("trading_signals").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        return (s.data[0] if s.data else None), (p.data[0] if p.data else None)
    except: return None, None

def render_ui(symbol, sig, price_data):
    current_p = price_data['price'] if price_data else 0
    is_open = price_data and (datetime.now(datetime.fromisoformat(price_data['created_at'].replace('Z', '+00:00')).tzinfo) - datetime.fromisoformat(price_data['created_at'].replace('Z', '+00:00'))).total_seconds() < 600

    if not is_open:
        st.markdown(f'<div class="signal-card"><div style="text-align:center; color:#555;">MARKET CLOSED / OFFLINE</div></div>', unsafe_allow_html=True)
        return

    # Se il segnale è vecchio (> 15 min), mostriamo solo SCANNING
    sig_valid = sig and (datetime.now(datetime.fromisoformat(sig['created_at'].replace('Z', '+00:00')).tzinfo) - datetime.fromisoformat(sig['created_at'].replace('Z', '+00:00'))).total_seconds() < 900
    
    rec = sig['recommendation'] if sig_valid else "SCANNING"
    col = "#69F0AE" if rec == "BUY" else "#FF5252" if rec == "SELL" else "#40C4FF"
    
    st.markdown(f"""
<div class="signal-card" style="border-top: 4px solid {col};">
<div style="color:#69F0AE; text-align:center; font-weight:700; letter-spacing:2px;">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:2.5rem; font-weight:800; color:{col}; text-align:center;">{rec}</div>
<div class="price-display">${current_p:,.2f}</div>
<div class="stats-grid">
<div class="stat-box"><div class="stat-label">ENTRY</div><div class="stat-value">{f"${sig['entry_price']:,.2f}" if sig_valid else "---"}</div></div>
<div class="stat-box"><div class="stat-label">STOP LOSS</div><div class="stat-value" style="color:#FF5252;">{f"${sig['stop_loss']:,.2f}" if sig_valid else "---"}</div></div>
<div class="stat-box"><div class="stat-label">TARGET</div><div class="stat-value" style="color:#69F0AE;">{f"${sig['take_profit']:,.2f}" if sig_valid else "---"}</div></div>
</div>
</div>
""", unsafe_allow_html=True)

def main():
    load_css()
    st.markdown('<div class="titan-header"><div style="font-family:Rajdhani; font-size:2rem; font-weight:700; color:#FFF;">TITAN ORACLE PRIME</div></div>', unsafe_allow_html=True)
    tabs = st.tabs(AppConfig.ASSETS)
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            sig, price = get_data(symbol)
            render_ui(symbol, sig, price)
    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__": main()
