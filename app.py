"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 DASHBOARD - PREMIUM FRONTEND INTERFACE
═══════════════════════════════════════════════════════════════════════════════
Professional Real-Time Trading Terminal powered by TITAN V90 Backend
Visualizes market data, AI signals, and performance metrics via Streamlit
Theme: Titanium Antracite & Zen Green (No Chart Version)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Dependency check & Import
try:
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.error("❌ Missing libraries. Run: pip install supabase python-dotenv pandas")
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
# VISUAL STYLING (CSS ENGINE) - ANTHRACITE & ZEN GREEN THEME
# ═══════════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Injects the Titanium Anthracite & Zen Green Style"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Inter:wght@300;400;600&display=swap');
        
        /* === GLOBAL THEME === */
        * { box-sizing: border-box; }
        
        .main { 
            background-color: #0E1012;
            background-image: linear-gradient(145deg, #16181C 0%, #0B0D0F 100%);
            color: #E4E8F0;
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 { font-family: 'Rajdhani', sans-serif !important; }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
        
        /* === HEADER === */
        .titan-header {
            background: linear-gradient(180deg, #1B1E23 0%, #16181C 100%);
            border: 1px solid #2D333B;
            border-radius: 16px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            position: relative;
            overflow: hidden;
            text-align: center;
        }
        
        .titan-header::after {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, #69F0AE, transparent);
            opacity: 0.5;
        }

        .titan-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, #FFFFFF 0%, #B0B0B0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }
        
        .titan-subtitle {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1rem;
            color: #69F0AE; /* Verde Riposante */
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-top: 5px;
            font-weight: 600;
        }
        
        /* === STATUS BADGE === */
        .status-badge {
            background: rgba(105, 240, 174, 0.08);
            border: 1px solid rgba(105, 240, 174, 0.2);
            color: #69F0AE;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 15px;
        }
        .status-dot {
            width: 8px; height: 8px; background: #69F0AE;
            border-radius: 50%; box-shadow: 0 0 8px #69F0AE;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

        /* === TABS STYLING === */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #16181C;
            padding: 8px;
            border-radius: 12px;
            border: 1px solid #2D333B;
            gap: 5px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #888;
            border-radius: 8px;
            border: none;
            padding: 10px 30px;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
            font-size: 1.2rem;
            flex: 1; 
        }
        .stTabs [aria-selected="true"] {
            background-color: #252930;
            color: #69F0AE !important;
            border: 1px solid #2D333B;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        /* === SIGNAL CARDS (Centrale) === */
        .signal-card {
            background: #1B1E23;
            border: 1px solid #2D333B;
            border-radius: 16px;
            padding: 3rem;
            margin: 1rem auto;
            max-width: 800px; /* Limita larghezza per eleganza */
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        
        .signal-card:hover { transform: translateY(-3px); border-color: #3E4650; }

        .signal-card-buy { border-top: 4px solid #69F0AE; }
        .signal-card-sell { border-top: 4px solid #FF5252; }
        .signal-card-wait { border-top: 4px solid #555; opacity: 0.9; }
        .signal-card-closed { border-top: 4px solid #555; background: #141619; }

        .signal-type {
            font-size: 3rem; font-weight: 800; letter-spacing: 2px;
            font-family: 'Rajdhani', sans-serif;
            text-align: center; margin-bottom: 10px;
        }
        .signal-symbol {
            font-size: 1.2rem; color: #69F0AE;
            font-weight: 600; letter-spacing: 2px; 
            text-align: center; margin-bottom: 2rem;
        }
        .price-display {
            font-size: 4.5rem; font-weight: 700; color: #FFF;
            font-family: 'Rajdhani', sans-serif;
            text-align: center;
            text-shadow: 0 4px 20px rgba(0,0,0,0.4);
            margin: 20px 0;
        }
        .price-label { font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 2px; text-align: center; }

        /* === STATS GRID === */
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 30px; }
        .stat-box {
            background: #23272E;
            border: 1px solid #2D333B;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .stat-label { font-size: 0.8rem; color: #888; text-transform: uppercase; margin-bottom: 8px; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: #FFF; font-family: 'Rajdhani', sans-serif; }
        
        .val-buy { color: #69F0AE; }
        .val-sell { color: #FF5252; }
        .val-blue { color: #40C4FF; }

        /* === TOP METRICS === */
        .metric-card-top {
            background: #1B1E23; border: 1px solid #2D333B;
            border-radius: 12px; padding: 15px; text-align: center;
        }
        .metric-label-top { font-size: 0.8rem; color: #888; margin-bottom: 5px; }
        .metric-val-top { font-size: 1.8rem; font-weight: 700; color: #FFF; font-family: 'Rajdhani', sans-serif; }

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
            if time_diff > 600: is_stale = True # 10 minuti
            
            if time_diff < 60: time_str = "Now"
            elif time_diff < 3600: time_str = f"{int(time_diff/60)}m ago"
            else: time_str = f"{int(time_diff/3600)}h ago"
        except: pass

    # --- CARD: MARKET CLOSED ---
    if not signal_data or is_stale:
        last_price = signal_data.get('current_price', 0) if signal_data else 0
        st.markdown(f"""
<div class="signal-card signal-card-closed">
<div style="text-align:center;">
<div class="signal-symbol">{symbol}</div>
<div style="font-size: 2.5rem; color: #888; font-weight:700; margin: 10px 0;">MARKET CLOSED</div>
<div style="font-size: 4rem; margin: 20px 0;">💤</div>
<div class="price-label">LAST KNOWN PRICE</div>
<div class="price-display" style="color: #666; font-size: 3rem;">${last_price:,.2f}</div>
<div style="color:#555; font-size:0.9rem; margin-top:20px;">Last update: {time_str}</div>
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

    if rec == 'BUY': card_cls, icon, col = "signal-card-buy", "▲", "#69F0AE"
    elif rec == 'SELL': card_cls, icon, col = "signal-card-sell", "▼", "#FF5252"
    else: card_cls, icon, col = "signal-card-wait", "●", "#888"

    st.markdown(f"""
<div class="signal-card {card_cls}">
<div class="signal-symbol">{symbol}</div>
<div class="signal-type" style="color:{col}">{rec} {icon}</div>

<div class="price-label">CURRENT PRICE</div>
<div class="price-display">${price:,.2f}</div>

<div style="background:#23272E; border-radius:8px; padding:15px; margin: 25px 0;">
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#888; font-size:0.9rem; font-weight:600;">AI CONFIDENCE</span>
<span style="color:{col}; font-weight:700; font-size:1rem;">{conf}%</span>
</div>
<div style="background:#333; height:8px; border-radius:4px;">
<div style="background:{col}; width:{conf}%; height:100%; border-radius:4px; box-shadow: 0 0 10px {col};"></div>
</div>
</div>

<div class="stats-grid">
<div class="stat-box"><div class="stat-label">ENTRY</div><div class="stat-value val-blue">${entry:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">STOP LOSS</div><div class="stat-value val-sell">${sl:,.2f}</div></div>
<div class="stat-box"><div class="stat-label">TAKE PROFIT</div><div class="stat-value val-buy">${tp:,.2f}</div></div>
</div>

<div style="margin-top:25px; padding-top:15px; border-top:1px solid #2D333B; color:#666; font-size:0.9rem; text-align:center;">
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
<div class="titan-title">TITAN ORACLE</div>
<div class="titan-subtitle">Enterprise Trading Intelligence</div>
<div class="status-badge">
<div class="status-dot"></div> SYSTEM ONLINE
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
    
    with c1: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">TOTAL SIGNALS</div><div class="metric-val-top val-blue">{vals["total"]}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">BUY</div><div class="metric-val-top val-buy">{vals["buy"]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">SELL</div><div class="metric-val-top val-sell">{vals["sell"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">AVG CONFIDENCE</div><div class="metric-val-top" style="color:#E0E0E0;">{vals["conf"]:.0f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TABS & CONTENT
    tabs = st.tabs(AppConfig.ASSETS)
    
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            # Centered layout using a single main call
            signal_data = get_latest_signal(symbol)
            render_signal_panel(symbol, signal_data)

    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
