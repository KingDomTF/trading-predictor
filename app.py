"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 DASHBOARD - PREMIUM FRONTEND INTERFACE
═══════════════════════════════════════════════════════════════════════════════
Professional Real-Time Trading Terminal powered by TITAN V90 Backend
Visualizes market data, AI signals, and performance metrics via Streamlit
Theme: Titanium Antracite & Zen Green
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Dependency check & Import
try:
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.error("❌ Missing libraries. Run: pip install supabase python-dotenv plotly pandas")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class AppConfig:
    PAGE_TITLE = "TITAN Oracle Prime"
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    AUTO_REFRESH_RATE = 2 

st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL STYLING
# ═══════════════════════════════════════════════════════════════════════════════

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
        .signal-card-scan { border-top: 4px solid #40C4FF; }
        
        .signal-symbol { font-size: 1.2rem; color: #69F0AE; font-weight: 600; text-align: center; font-family: 'Rajdhani'; }
        .price-display { font-size: 4rem; font-weight: 700; color: #FFF; text-align: center; font-family: 'Rajdhani'; margin: 15px 0; }
        
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 25px; }
        .stat-box { background: #23272E; border: 1px solid #2D333B; border-radius: 10px; padding: 15px; text-align: center; }
        .stat-label { font-size: 0.7rem; color: #888; text-transform: uppercase; }
        .stat-value { font-size: 1.2rem; font-weight: 700; color: #FFF; font-family: 'Rajdhani'; }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE & DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key: return None
    return create_client(url, key)

supabase = init_supabase()

def get_latest_signal(symbol):
    if not supabase: return None
    try:
        # Legge l'ultimo segnale di trading
        res = supabase.table("trading_signals").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        return res.data[0] if res.data else None
    except: return None

def get_latest_price(symbol):
    if not supabase: return None
    try:
        # Legge il prezzo più recente dalla cronologia (che il bridge sta inviando ora)
        res = supabase.table("price_history").select("price").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]['price']
    except: pass
    return None

def get_24h_stats():
    if not supabase: return None
    try:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        res = supabase.table("trading_signals").select("*").gte("created_at", cutoff).in_("recommendation", ["BUY", "SELL"]).execute()
        if not res.data: return None
        total = len(res.data)
        buy = sum(1 for s in res.data if s['recommendation'] == 'BUY')
        sell = sum(1 for s in res.data if s['recommendation'] == 'SELL')
        avg_conf = sum(s.get('confidence_score', 0) for s in res.data) / total if total > 0 else 0
        return {'total': total, 'buy': buy, 'sell': sell, 'confidence': avg_conf}
    except: return None

# ═══════════════════════════════════════════════════════════════════════════════
# UI RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def render_signal_panel(symbol, signal_data, current_price):
    # Se non c'è segnale ma abbiamo un prezzo live -> MOSTRA "SCANNING"
    if not signal_data:
        if current_price:
            st.markdown(f"""
<div class="signal-card signal-card-scan">
<div class="signal-symbol">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:2.5rem; font-weight:800; color:#40C4FF; text-align:center;">SCANNING</div>
<div class="price-display">${current_price:,.2f}</div>
<div style="text-align:center; color:#888; margin-top:20px;">Analyzing market conditions...</div>
</div>
""", unsafe_allow_html=True)
        else:
            # Se non c'è nemmeno il prezzo -> AWAITING DATA
            st.markdown(f"""
<div class="signal-card" style="border-top: 4px solid #555;">
<div class="signal-symbol">{symbol}</div>
<div class="price-display" style="font-size:2rem; color:#666;">AWAITING DATA...</div>
<div style="text-align:center; color:#555; margin-top:10px;">Check bridge connection</div>
</div>
""", unsafe_allow_html=True)
        return

    # Se c'è un segnale -> MOSTRA SEGNALE
    rec = signal_data.get('recommendation', 'WAIT')
    # Usa il prezzo live se disponibile, altrimenti quello del segnale
    display_price = current_price if current_price else signal_data.get('current_price', 0)
    
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    
    col = "#69F0AE" if rec == 'BUY' else "#FF5252" if rec == 'SELL' else "#888"
    cls = "signal-card-buy" if rec == 'BUY' else "signal-card-sell" if rec == 'SELL' else ""

    st.markdown(f"""
<div class="signal-card {cls}">
<div class="signal-symbol">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:3.5rem; font-weight:800; color:{col}; text-align:center; letter-spacing:2px;">{rec}</div>
<div class="price-display">${display_price:,.2f}</div>
<div style="background:#23272E; border-radius:8px; padding:10px; margin: 20px 0;">
<div style="display:flex; justify-content:space-between; margin-bottom:5px;">
<span style="color:#888; font-size:0.8rem;">CONFIDENCE</span>
<span style="color:{col}; font-weight:700;">{conf}%</span>
</div>
<div style="background:#333; height:6px; border-radius:3px;">
<div style="background:{col}; width:{conf}%; height:100%; border-radius:3px;"></div>
</div>
</div>
<div class="stats-grid">
<div class="stat-box"><div class="stat-label">ENTRY</div><div class="stat-value">${entry:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">STOP LOSS</div><div class="stat-value" style="color:#FF5252;">${sl:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">TARGET</div><div class="stat-value" style="color:#69F0AE;">${tp:,.2f}</div></div>
</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    load_custom_css()
    if not supabase:
        st.error("Secrets mancanti su Streamlit Cloud.")
        st.stop()

    st.markdown('<div class="titan-header"><div class="titan-title">TITAN ORACLE</div><div class="titan-subtitle">Enterprise Trading Intelligence</div></div>', unsafe_allow_html=True)

    stats = get_24h_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Signals (24h)", stats['total'] if stats else 0)
    with c2: st.metric("Buy", stats['buy'] if stats else 0)
    with c3: st.metric("Sell", stats['sell'] if stats else 0)
    with c4: st.metric("Avg Conf", f"{stats['confidence'] if stats else 0:.0f}%")

    tabs = st.tabs(AppConfig.ASSETS)
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            # Recupera sia il segnale che il prezzo live
            signal_data = get_latest_signal(symbol)
            current_price = get_latest_price(symbol)
            
            # Passa entrambi al renderer
            render_signal_panel(symbol, signal_data, current_price)

    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
