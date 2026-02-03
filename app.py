"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 DASHBOARD - PREMIUM FRONTEND INTERFACE
═══════════════════════════════════════════════════════════════════════════════
Professional Real-Time Trading Terminal powered by TITAN V90 Backend
Visualizes market data, AI signals, and performance metrics via Streamlit
Theme: Neo-Financial Brutalism (No Chart Version)
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
    """Frontend Configuration"""
    PAGE_TITLE = "TITAN Oracle Prime"
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    
    # --- LISTA ASSET ---
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    
    # Refresh rates
    AUTO_REFRESH_RATE = 5  # Seconds

# Initialize Page
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL STYLING (CSS ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Injects the Style (Note: HTML strings are unindented to fix rendering)"""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@300;400;600;800&family=Syne:wght@400;600;700;800&display=swap');

/* === GLOBAL THEME === */
* { box-sizing: border-box; margin: 0; padding: 0; }

.main { 
    background: linear-gradient(135deg, #0A0E12 0%, #0D1117 50%, #0A0E12 100%);
    background-attachment: fixed;
    color: #E8ECF1;
    font-family: 'JetBrains Mono', monospace;
}

h1, h2, h3, h4, h5, h6 { 
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    letter-spacing: -0.02em;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

/* === BUTTON STYLING === */
.stButton>button {
    background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
    color: #E8ECF1;
    border: 2px solid #2A3340;
    border-radius: 12px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    transition: all 0.3s;
}
.stButton>button:hover {
    border-color: #00FFF0;
    color: #00FFF0;
    box-shadow: 0 0 15px rgba(0, 255, 240, 0.2);
}

/* === HEADER === */
.titan-header {
    background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
    border: 2px solid #2A3340;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}

.titan-title {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #FFFFFF 0%, #00FFF0 50%, #FF006E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(0, 255, 240, 0.3);
}

.titan-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    color: #8892A0;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.5rem;
}

/* === STATUS BADGE === */
.status-badge {
    background: rgba(0, 255, 240, 0.1);
    border: 1px solid rgba(0, 255, 240, 0.3);
    color: #00FFF0;
    padding: 0.5rem 1rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Space Mono', monospace;
}
.status-dot {
    width: 8px; height: 8px; background: #00FFF0;
    border-radius: 50%; box-shadow: 0 0 10px #00FFF0;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background: #13171D;
    padding: 8px;
    border-radius: 12px;
    border: 2px solid #2A3340;
    justify-content: center; /* Center Tabs */
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5A6678;
    border: none;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 255, 240, 0.1);
    color: #00FFF0 !important;
    border: 1px solid rgba(0, 255, 240, 0.3);
}

/* === SIGNAL CARDS === */
.signal-card {
    background: #13171D;
    border: 2px solid #2A3340;
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    transition: transform 0.3s;
}
.signal-card:hover { transform: translateY(-4px); }

.signal-card-buy { border-left: 4px solid #00FFF0; }
.signal-card-sell { border-left: 4px solid #FF006E; }
.signal-card-wait { border-left: 4px solid #5A6678; opacity: 0.7; }
.signal-card-closed { border-left: 4px solid #3A4350; background: #0F1216; opacity: 0.8; }

.signal-type {
    font-size: 2.5rem; font-weight: 800; font-family: 'Syne', sans-serif;
}
.signal-symbol {
    font-size: 0.9rem; color: #8892A0; font-family: 'Space Mono', monospace; letter-spacing: 0.1em;
}
.price-display {
    font-size: 3.5rem; font-weight: 800; color: #FFF; font-family: 'Syne', sans-serif;
    text-shadow: 0 0 20px rgba(255,255,255,0.1);
}
.price-label { font-size: 0.7rem; color: #5A6678; text-transform: uppercase; font-family: 'Space Mono', monospace; }

/* === STATS GRID === */
.stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 1.5rem; }
.stat-box {
    background: rgba(42, 51, 64, 0.2);
    border: 1px solid #2A3340;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-label { font-size: 0.65rem; color: #5A6678; text-transform: uppercase; margin-bottom: 5px; font-family: 'Space Mono', monospace; }
.stat-value { font-size: 1.3rem; font-weight: 800; color: #E8ECF1; font-family: 'Syne', sans-serif; }

.val-buy { color: #00FFF0; }
.val-sell { color: #FF006E; }
.val-blue { color: #5B9FFF; }

/* === TOP METRICS === */
.metric-card-top {
    background: #13171D; border: 2px solid #2A3340;
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.metric-label-top { font-size: 0.7rem; color: #5A6678; margin-bottom: 0.5rem; font-family: 'Space Mono', monospace; text-transform: uppercase; }
.metric-val-top { font-size: 2.2rem; font-weight: 800; color: #E8ECF1; font-family: 'Syne', sans-serif; }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    try:
        # Check both Secrets (Cloud) and .env (Local)
        url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

        if not url or not key: return None
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

supabase = init_supabase()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_latest_signal(symbol):
    if not supabase: return None
    try:
        response = supabase.table("ai_oracle").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        return response.data[0] if response.data else None
    except: return None

def get_24h_stats():
    if not supabase: return None
    try:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        response = supabase.table("ai_oracle").select("*").gte("created_at", cutoff).in_("recommendation", ["BUY", "SELL"]).execute()
        if not response.data: return None
        total = len(response.data)
        buy = sum(1 for s in response.data if s['recommendation'] == 'BUY')
        sell = sum(1 for s in response.data if s['recommendation'] == 'SELL')
        avg_conf = sum(s.get('confidence_score', 0) for s in response.data) / total if total > 0 else 0
        return {'total': total, 'buy': buy, 'sell': sell, 'confidence': avg_conf}
    except: return None

# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS (RENDERERS)
# ═══════════════════════════════════════════════════════════════════════════════

def render_signal_panel(symbol, signal_data):
    # --- MARKET CLOSED CHECK (10 Min) ---
    created_at = signal_data.get('created_at', '') if signal_data else ''
    is_stale = False
    time_str = "Waiting..."
    
    if created_at:
        try:
            signal_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            time_diff = (datetime.now(signal_time.tzinfo) - signal_time).total_seconds()
            if time_diff > 600: is_stale = True
            
            if time_diff < 60: time_str = "Now"
            elif time_diff < 3600: time_str = f"{int(time_diff/60)}m ago"
            else: time_str = f"{int(time_diff/3600)}h ago"
        except: pass

    # --- CARD: MARKET CLOSED ---
    if not signal_data or is_stale:
        last_price = signal_data.get('current_price', 0) if signal_data else 0
        st.markdown(f"""
<div class="signal-card signal-card-closed">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<div class="signal-symbol">{symbol}</div>
<div style="font-size: 2rem; color: #5A6678; font-weight:800; font-family: 'Syne', sans-serif;">MARKET CLOSED</div>
</div>
<div style="font-size: 3rem; opacity: 0.3;">💤</div>
</div>
<div style="margin-top:1.5rem; border-top:1px solid #2A3340; padding-top:1.5rem;">
<div class="price-label">LAST KNOWN PRICE</div>
<div style="font-family:'Syne', sans-serif; font-size:2.5rem; font-weight: 800; color:#3A4350;">${last_price:,.2f}</div>
<div style="color:#5A6678; font-size:0.75rem; margin-top:0.75rem; font-family: 'Space Mono', monospace;">Last update: {time_str}</div>
</div>
</div>
""", unsafe_allow_html=True)
        return

    # --- CARD: ACTIVE ---
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0)
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', 'Analysis')

    if rec == 'BUY': card_cls, icon, col = "signal-card-buy", "▲", "#00FFF0"
    elif rec == 'SELL': card_cls, icon, col = "signal-card-sell", "▼", "#FF006E"
    else: card_cls, icon, col = "signal-card-wait", "●", "#5A6678"

    st.markdown(f"""
<div class="signal-card {card_cls}">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div class="signal-symbol">{symbol}</div>
<div class="signal-type" style="color:{col}">{rec}</div>
</div>
<div style="font-size:3rem; opacity: 0.8; line-height: 1;">{icon}</div>
</div>
<div style="margin: 1.5rem 0;">
<div class="price-label">CURRENT PRICE</div>
<div class="price-display">${price:,.2f}</div>
</div>
<div style="background:rgba(42, 51, 64, 0.3); border-radius:10px; padding:1rem; margin-bottom:1.5rem; border:1px solid #2A3340;">
<div style="display:flex; justify-content:space-between; margin-bottom:0.75rem;">
<span style="color:#5A6678; font-size:0.75rem; font-family:'Space Mono'; font-weight:700;">CONFIDENCE</span>
<span style="color:{col}; font-weight:800; font-family:'Syne';">{conf}%</span>
</div>
<div style="background:#1A1F28; height:8px; border-radius:4px; overflow:hidden;">
<div style="background:{col}; width:{conf}%; height:100%; border-radius:4px; box-shadow: 0 0 10px {col};"></div>
</div>
</div>
<div class="stats-grid">
<div class="stat-box"><div class="stat-label">ENTRY</div><div class="stat-value val-blue">${entry:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">STOP LOSS</div><div class="stat-value val-sell">${sl:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">TARGET</div><div class="stat-value val-buy">${tp:,.2f}</div></div>
</div>
<div style="margin-top:1.5rem; padding-top:1.5rem; border-top:1px solid #2A3340; color:#5A6678; font-size:0.8rem; text-align:center; font-family:'JetBrains Mono';">
{details}
</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    load_custom_css()
    
    if not supabase:
        st.error("❌ Database connection error. Check Secrets.")
        st.stop()
    
    # HEADER
    st.markdown("""
<div class="titan-header">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:20px;">
<div class="titan-branding">
<div class="titan-title">TITAN ORACLE</div>
<div class="titan-subtitle">Neo-Financial Intelligence</div>
</div>
<div class="status-badge">
<div class="status-dot"></div> SYSTEM ONLINE
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # TOP METRICS
    stats = get_24h_stats()
    c1, c2, c3, c4 = st.columns(4)
    
    vals = {
        'total': stats['total'] if stats else 0,
        'buy': stats['buy'] if stats else 0,
        'sell': stats['sell'] if stats else 0,
        'conf': stats['confidence'] if stats else 0
    }
    
    with c1: st.markdown(f"""
<div class="metric-card-top">
<div class="metric-label-top">Total Signals</div>
<div class="metric-val-top val-blue">{vals["total"]}</div>
</div>
""", unsafe_allow_html=True)
    
    with c2: st.markdown(f"""
<div class="metric-card-top">
<div class="metric-label-top">Buy Signals</div>
<div class="metric-val-top val-buy">{vals["buy"]}</div>
</div>
""", unsafe_allow_html=True)
    
    with c3: st.markdown(f"""
<div class="metric-card-top">
<div class="metric-label-top">Sell Signals</div>
<div class="metric-val-top val-sell">{vals["sell"]}</div>
</div>
""", unsafe_allow_html=True)
    
    with c4: st.markdown(f"""
<div class="metric-card-top">
<div class="metric-label-top">Avg Confidence</div>
<div class="metric-val-top" style="color:#E8ECF1;">{vals["conf"]:.0f}%</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TABS & CONTENT
    tabs = st.tabs(AppConfig.ASSETS)
    
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            # LAYOUT CENTRATO (3 Colonne: Spazio - Card - Spazio)
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c2:
                signal_data = get_latest_signal(symbol)
                render_signal_panel(symbol, signal_data)

    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
