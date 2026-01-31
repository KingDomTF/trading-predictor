"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 DASHBOARD - PREMIUM FRONTEND INTERFACE
═══════════════════════════════════════════════════════════════════════════════
Professional Real-Time Trading Terminal powered by TITAN V90 Backend
Visualizes market data, AI signals, and performance metrics via Streamlit
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
    PAGE_ICON = "🏛️"
    LAYOUT = "wide"
    
    # Assets to display
    ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30", "ETHUSD"]
    
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
    """Injects the Premium 'Cyberpunk/Bloomberg' Style"""
    st.markdown("""
    <style>
        /* === GLOBAL THEME === */
        .main { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%); }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* === HEADER === */
        .titan-header {
            background: linear-gradient(135deg, rgba(0,217,255,0.1) 0%, rgba(0,153,255,0.1) 100%);
            border: 2px solid rgba(0,217,255,0.3);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,217,255,0.2);
        }
        .titan-title {
            font-size: 56px; font-weight: 900;
            background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin: 0; text-shadow: 0 0 30px rgba(0,217,255,0.5);
        }
        .titan-subtitle { font-size: 18px; color: #aaaaaa; margin-top: 10px; }
        
        /* === STATUS BADGE === */
        .status-badge {
            display: inline-block; background: rgba(0,255,136,0.2);
            border: 1px solid #00ff88; padding: 8px 20px;
            border-radius: 20px; font-size: 14px; color: #00ff88; margin-top: 15px;
        }
        .status-dot {
            display: inline-block; width: 10px; height: 10px;
            background: #00ff88; border-radius: 50%; margin-right: 8px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 10px #00ff88; }
            50% { opacity: 0.5; box-shadow: 0 0 20px #00ff88; }
        }

        /* === TABS STYLING === */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px; background: rgba(26,31,58,0.6);
            padding: 15px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, rgba(22,33,62,0.8) 0%, rgba(15,52,96,0.8) 100%);
            border: 2px solid rgba(0,217,255,0.3); border-radius: 12px;
            padding: 12px 24px; color: #ffffff; font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
            border-color: #00d9ff; color: #000000; box-shadow: 0 8px 25px rgba(0,217,255,0.4);
        }

        /* === SIGNAL CARDS === */
        .signal-card-buy {
            background: linear-gradient(135deg, rgba(30,60,40,0.95) 0%, rgba(45,95,62,0.95) 100%);
            border: 2px solid #00ff88; border-radius: 20px; padding: 25px;
            box-shadow: 0 10px 40px rgba(0,255,136,0.3); margin: 20px 0;
        }
        .signal-card-sell {
            background: linear-gradient(135deg, rgba(60,30,30,0.95) 0%, rgba(95,45,45,0.95) 100%);
            border: 2px solid #ff0044; border-radius: 20px; padding: 25px;
            box-shadow: 0 10px 40px rgba(255,0,68,0.3); margin: 20px 0;
        }
        .signal-card-wait {
            background: linear-gradient(135deg, rgba(42,42,60,0.8) 0%, rgba(58,58,78,0.8) 100%);
            border: 2px solid rgba(136,136,136,0.5); border-radius: 20px;
            padding: 25px; margin: 20px 0; opacity: 0.7;
        }

        /* === TYPOGRAPHY & METRICS === */
        .signal-icon { font-size: 48px; margin-bottom: 15px; }
        .signal-type { font-size: 32px; font-weight: bold; margin-bottom: 10px; }
        .price-display { font-size: 56px; font-weight: 900; margin: 20px 0; text-shadow: 0 0 20px currentColor; }
        
        .confidence-bar { background: rgba(255,255,255,0.1); height: 8px; border-radius: 10px; margin: 15px 0; overflow: hidden; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #00ff88 0%, #00d9ff 100%); border-radius: 10px; }

        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 25px 0; }
        .stat-box { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; text-align: center; }
        .stat-label { font-size: 12px; color: #888888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
        .stat-value { font-size: 28px; font-weight: bold; color: #ffffff; }
        
        .stat-value-green { color: #00ff88; }
        .stat-value-red { color: #ff0044; }
        .stat-value-blue { color: #00d9ff; }

        .metric-card {
            background: linear-gradient(135deg, rgba(22,33,62,0.6) 0%, rgba(15,52,96,0.6) 100%);
            border: 1px solid rgba(0,217,255,0.3); border-radius: 15px; padding: 20px; margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    """Initialize Supabase client (Hybrid: Cloud Secrets + Local .env)"""
    try:
        # Try Streamlit Secrets first (Production)
        if 'SUPABASE_URL' in st.secrets:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        # Fallback to Environment Variables (Local)
        else:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            return None
            
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

supabase = init_supabase()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_latest_signal(symbol):
    """Fetch latest AI signal from database"""
    if not supabase: return None
    try:
        response = supabase.table("ai_oracle")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None
    except: return None

def get_price_history(symbol, hours=4):
    """Fetch price candles for charting"""
    if not supabase: return pd.DataFrame()
    try:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        response = supabase.table("mt4_feed")\
            .select("*")\
            .eq("symbol", symbol)\
            .gte("created_at", cutoff)\
            .order("created_at", desc=False)\
            .execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

def get_24h_stats():
    """Calculate 24h performance metrics"""
    if not supabase: return None
    try:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        response = supabase.table("ai_oracle")\
            .select("*")\
            .gte("created_at", cutoff)\
            .in_("recommendation", ["BUY", "SELL"])\
            .execute()
        
        if not response.data: return None
        
        total = len(response.data)
        buy = sum(1 for s in response.data if s['recommendation'] == 'BUY')
        sell = sum(1 for s in response.data if s['recommendation'] == 'SELL')
        avg_conf = sum(s.get('confidence_score', 0) for s in response.data) / total if total > 0 else 0
        
        return {'total': total, 'buy': buy, 'sell': sell, 'confidence': avg_conf}
    except: return None

# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df, signal_data):
    """Render interactive Plotly chart"""
    if df.empty: return None
    
    fig = go.Figure()
    
    # Main Price Line
    fig.add_trace(go.Scatter(
        x=df['created_at'], y=df['price'], mode='lines', name='Price',
        line=dict(color='#00d9ff', width=3),
        fill='tozeroy', fillcolor='rgba(0,217,255,0.1)'
    ))
    
    # Trading Levels (Entry/SL/TP)
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        if entry > 0: fig.add_hline(y=entry, line_dash="dash", line_color="#ffffff", annotation_text="ENTRY")
        if sl > 0: fig.add_hline(y=sl, line_dash="dot", line_color="#ff0044", annotation_text="SL")
        if tp > 0: fig.add_hline(y=tp, line_dash="dot", line_color="#00ff88", annotation_text="TP")
    
    # Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,39,0.5)',
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Price"),
        hovermode='x unified', font=dict(color='#ffffff')
    )
    return fig

def render_signal_panel(symbol, signal_data):
    """Render the Premium Card for a specific symbol"""
    if not signal_data:
        st.markdown(f"""
        <div class="signal-card-wait">
            <div class="signal-icon">⚪</div>
            <div class="signal-type">{symbol}</div>
            <p style="color: #888;">Scanning Market...</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Extract Data
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0)
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', 'No details')

    # Styling Logic
    if rec == 'BUY':
        card_class, icon, p_color = "signal-card-buy", "🟢", "#00ff88"
    elif rec == 'SELL':
        card_class, icon, p_color = "signal-card-sell", "🔴", "#ff0044"
    else:
        card_class, icon, p_color = "signal-card-wait", "⚪", "#888888"

    # HTML Render
    st.markdown(f"""
    <div class="{card_class}">
        <div class="signal-icon">{icon}</div>
        <div class="signal-type">{symbol} - {rec}</div>
        <div class="price-display" style="color: {p_color};">${price:,.5f}</div>
        
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {conf}%;"></div>
        </div>
        <div style="text-align: center; color: #aaa; font-size: 14px;">Confidence: {conf}%</div>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Entry</div>
                <div class="stat-value stat-value-blue">${entry:,.5f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Stop Loss</div>
                <div class="stat-value stat-value-red">${sl:,.5f}</div>
            </div>
            <div class="stat
