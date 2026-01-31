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
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    
    # --- LISTA ASSET AGGIORNATA ---
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
    """Injects the Premium Professional & Futuristic Style"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;500;600;700&display=swap');
        
        /* === GLOBAL THEME === */
        * { box-sizing: border-box; }
        
        .main { 
            background: #0B0C10;
            background-image: 
                radial-gradient(ellipse at top, rgba(30, 39, 73, 0.3) 0%, transparent 50%),
                radial-gradient(ellipse at bottom, rgba(17, 24, 39, 0.5) 0%, transparent 50%);
            color: #E4E8F0;
            font-family: 'Rajdhani', sans-serif;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
        
        /* === HEADER === */
        .titan-header {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.9) 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 24px;
            padding: 3rem 2rem;
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .titan-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                linear-gradient(135deg, transparent 0%, rgba(56, 189, 248, 0.03) 50%, transparent 100%);
            pointer-events: none;
        }
        
        .titan-header-content {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 2rem;
        }
        
        .titan-branding {
            flex: 1;
            min-width: 300px;
        }
        
        .titan-title {
            font-family: 'Orbitron', monospace;
            font-size: 3.5rem;
            font-weight: 900;
            letter-spacing: 2px;
            margin: 0;
            background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 0 40px rgba(56, 189, 248, 0.3));
            line-height: 1.1;
        }
        
        .titan-subtitle {
            font-family: 'Space Mono', monospace;
            font-size: 0.95rem;
            color: #94A3B8;
            margin-top: 0.75rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            font-weight: 400;
        }
        
        .titan-meta {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        
        /* === STATUS BADGE === */
        .status-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #10B981;
            letter-spacing: 1px;
            font-family: 'Space Mono', monospace;
            backdrop-filter: blur(10px);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: #10B981;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 2s ease-in-out infinite;
            box-shadow: 0 0 12px #10B981;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.1); }
        }
        
        .live-time {
            font-family: 'Space Mono', monospace;
            font-size: 0.85rem;
            color: #64748B;
            padding: 0.6rem 1.2rem;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }

        /* === TABS STYLING === */
        .stTabs {
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.6);
            padding: 0.75rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            backdrop-filter: blur(20px);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            color: #94A3B8;
            font-weight: 600;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1rem;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(56, 189, 248, 0.1);
            border-color: rgba(56, 189, 248, 0.3);
            color: #E4E8F0;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(129, 140, 248, 0.15) 100%);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #F0F9FF;
            box-shadow: 
                0 4px 20px rgba(56, 189, 248, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        /* === SIGNAL CARDS === */
        .signal-card {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.9) 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        .signal-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, transparent, currentColor, transparent);
            opacity: 0.8;
        }
        
        .signal-card-buy {
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10B981;
        }
        
        .signal-card-buy::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at top right, rgba(16, 185, 129, 0.08) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .signal-card-sell {
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #EF4444;
        }
        
        .signal-card-sell::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at top right, rgba(239, 68, 68, 0.08) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .signal-card-wait {
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: #64748B;
            opacity: 0.75;
        }
        
        .signal-card-closed {
            border: 1px solid rgba(100, 116, 139, 0.4);
            color: #94A3B8;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(20, 25, 35, 0.98) 100%);
        }
        
        .signal-card:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        /* === SIGNAL CONTENT === */
        .signal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 1;
        }
        
        .signal-icon {
            font-size: 2.5rem;
            line-height: 1;
            filter: drop-shadow(0 0 20px currentColor);
        }
        
        .signal-type {
            font-family: 'Orbitron', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 1px;
        }
        
        .signal-symbol {
            font-family: 'Space Mono', monospace;
            font-size: 0.9rem;
            color: #94A3B8;
            letter-spacing: 2px;
            margin-top: 0.25rem;
        }
        
        .price-display {
            font-family: 'Orbitron', monospace;
            font-size: 3.5rem;
            font-weight: 900;
            margin: 1.5rem 0;
            line-height: 1;
            position: relative;
            z-index: 1;
            text-shadow: 0 0 40px currentColor;
        }
        
        .price-label {
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
        }

        /* === CONFIDENCE BAR === */
        .confidence-container {
            margin: 2rem 0;
            position: relative;
            z-index: 1;
        }
        
        .confidence-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
            color: #94A3B8;
            font-family: 'Space Mono', monospace;
        }
        
        .confidence-bar {
            background: rgba(30, 41, 59, 0.6);
            height: 10px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, currentColor 0%, rgba(255, 255, 255, 0.8) 100%);
            position: relative;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 20px currentColor;
        }
        
        .confidence-fill::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.3) 50%, transparent 100%);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        /* === STATS GRID === */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 2rem 0;
            position: relative;
            z-index: 1;
        }
        
        .stat-box {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
        }
        
        .stat-box:hover {
            background: rgba(30, 41, 59, 0.6);
            border-color: rgba(148, 163, 184, 0.2);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.75rem;
            font-weight: 600;
            font-family: 'Rajdhani', sans-serif;
        }
        
        .stat-value {
            font-family: 'Orbitron', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            color: #E4E8F0;
            line-height: 1;
        }
        
        .stat-value-green { color: #10B981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
        .stat-value-red { color: #EF4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        .stat-value-blue { color: #38BDF8; text-shadow: 0 0 20px rgba(56, 189, 248, 0.5); }
        .stat-value-purple { color: #818CF8; text-shadow: 0 0 20px rgba(129, 140, 248, 0.5); }

        /* === STRATEGY DETAILS === */
        .strategy-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-left: 3px solid currentColor;
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 1.5rem;
            position: relative;
            z-index: 1;
            backdrop-filter: blur(10px);
        }
        
        .strategy-label {
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.5rem;
            font-weight: 600;
            font-family: 'Rajdhani', sans-serif;
        }
        
        .strategy-text {
            color: #94A3B8;
            font-size: 0.9rem;
            line-height: 1.6;
            font-family: 'Space Mono', monospace;
        }

        /* === METRICS CARDS === */
        .metric-card {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.8) 0%, 
                rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(20px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 4px 20px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            border-color: rgba(148, 163, 184, 0.3);
        }
        
        /* === CHART CONTAINER === */
        .chart-container {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.9) 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        .chart-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        .chart-title {
            font-family: 'Orbitron', monospace;
            font-size: 1.25rem;
            font-weight: 700;
            color: #E4E8F0;
            letter-spacing: 1px;
        }
        
        .chart-timeframe {
            font-family: 'Space Mono', monospace;
            font-size: 0.8rem;
            color: #64748B;
            background: rgba(30, 41, 59, 0.6);
            padding: 0.4rem 0.9rem;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        /* === SCROLLBAR === */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
        ::-webkit-scrollbar-thumb { 
            background: rgba(56, 189, 248, 0.3); 
            border-radius: 5px; 
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(56, 189, 248, 0.5); }
        
        /* === RESPONSIVE === */
        @media (max-width: 768px) {
            .titan-title { font-size: 2rem; }
            .price-display { font-size: 2.5rem; }
            .stats-grid { grid-template-columns: 1fr; }
            .titan-header-content { flex-direction: column; text-align: center; }
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
        if 'SUPABASE_URL' in st.secrets:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
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
    """Render interactive Plotly chart with enhanced styling"""
    if df.empty: return None
    
    fig = go.Figure()
    
    # Main Price Line with gradient fill
    fig.add_trace(go.Scatter(
        x=df['created_at'], 
        y=df['price'], 
        mode='lines',
        name='Price',
        line=dict(color='#38BDF8', width=3, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.1)',
        hovertemplate='<b>%{y:,.2f}</b><br>%{x}<extra></extra>'
    ))
    
    # Trading Levels with enhanced styling
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        if entry > 0:
            fig.add_hline(
                y=entry, 
                line_dash="dash", 
                line_color="#F0F9FF", 
                line_width=2,
                annotation_text="ENTRY",
                annotation_position="right",
                annotation_font=dict(color="#F0F9FF", size=11, family="Space Mono")
            )
        if sl > 0:
            fig.add_hline(
                y=sl, 
                line_dash="dot", 
                line_color="#EF4444", 
                line_width=2,
                annotation_text="STOP LOSS",
                annotation_position="right",
                annotation_font=dict(color="#EF4444", size=11, family="Space Mono")
            )
        if tp > 0:
            fig.add_hline(
                y=tp, 
                line_dash="dot", 
                line_color="#10B981", 
                line_width=2,
                annotation_text="TAKE PROFIT",
                annotation_position="right",
                annotation_font=dict(color="#10B981", size=11, family="Space Mono")
            )
    
    # Enhanced Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=500,
        margin=dict(l=20, r=60, t=20, b=40),
        xaxis=dict(
            showgrid=True, 
            gridcolor='rgba(148, 163, 184, 0.1)',
            gridwidth=1,
            title="",
            color='#94A3B8',
            showline=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(148, 163, 184, 0.1)',
            gridwidth=1,
            title="",
            color='#94A3B8',
            showline=False,
            zeroline=False
        ),
        hovermode='x unified',
        font=dict(color='#E4E8F0', family='Rajdhani'),
        hoverlabel=dict(
            bgcolor='rgba(15, 23, 42, 0.95)',
            bordercolor='rgba(56, 189, 248, 0.3)',
            font=dict(family='Space Mono', size=12, color='#E4E8F0')
        ),
        showlegend=False
    )
    
    return fig

def render_signal_panel(symbol, signal_data):
    """Render the Premium Card for a specific symbol - WITH MARKET CLOSED LOGIC"""
    
    # 1. CHECK NO DATA
    if not signal_data:
        st.markdown(f"""
<div class="signal-card signal-card-wait">
<div class="signal-header">
<div>
<div class="signal-type">SCANNING</div>
<div class="signal-symbol">{symbol}</div>
</div>
<div class="signal-icon">⚪</div>
</div>
<div style="text-align: center; padding: 2rem 0; color: #64748B; font-family: 'Space Mono', monospace; font-size: 0.9rem;">
Analyzing market conditions...
</div>
</div>
""", unsafe_allow_html=True)
        return

    # 2. TIME CHECK (MARKET CLOSED LOGIC)
    created_at = signal_data.get('created_at', '')
    is_stale = False
    time_str = "Unknown"
    
    try:
        # Parse timestamp from Supabase
        signal_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        # Calculate difference from NOW
        time_diff = (datetime.now(signal_time.tzinfo) - signal_time).total_seconds()
        
        # If older than 10 minutes (600 seconds) -> MARKET CLOSED
        if time_diff > 600:
            is_stale = True
            
        # Format time string
        if time_diff < 60: time_str = "Just now"
        elif time_diff < 3600: time_str = f"{int(time_diff/60)}m ago"
        elif time_diff < 86400: time_str = f"{int(time_diff/3600)}h ago"
        else: time_str = f"{int(time_diff/86400)}d ago"
            
    except Exception as e:
        time_str = "Unknown"

    # 3. RENDER "MARKET CLOSED" CARD
    if is_stale:
        st.markdown(f"""
<div class="signal-card signal-card-closed">
<div class="signal-header">
<div>
<div class="signal-type" style="color: #94A3B8;">MARKET CLOSED</div>
<div class="signal-symbol">{symbol}</div>
</div>
<div class="signal-icon">💤</div>
</div>
<div style="text-align: center; padding: 1.5rem 0;">
<div style="font-family: 'Space Mono'; color: #64748B; margin-bottom: 1.5rem; font-size: 0.9rem;">
Last update received {time_str}.<br>System in standby mode.
</div>
<div style="background: rgba(30, 41, 59, 0.4); padding: 1rem; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.1); display: inline-block; min-width: 200px;">
<div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; margin-bottom: 5px; font-family: 'Rajdhani', sans-serif; font-weight: 600;">Last Known Price</div>
<div style="font-family: 'Orbitron', monospace; font-size: 1.5rem; color: #94A3B8;">${signal_data.get('current_price', 0):,.2f}</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
        return

    # 4. RENDER ACTIVE SIGNAL CARD (If market open)
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0)
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', 'No strategy details available')

    # Styling Logic
    if rec == 'BUY':
        card_class, icon, color = "signal-card-buy", "🟢", "#10B981"
    elif rec == 'SELL':
        card_class, icon, color = "signal-card-sell", "🔴", "#EF4444"
    else:
        card_class, icon, color = "signal-card-wait", "⚪", "#64748B"

    # HTML Render
    st.markdown(f"""
<div class="signal-card {card_class}">
<div class="signal-header">
<div>
<div class="signal-type">{rec}</div>
<div class="signal-symbol">{symbol}</div>
</div>
<div class="signal-icon">{icon}</div>
</div>
<div>
<div class="price-label">Current Price</div>
<div class="price-display" style="color: {color};">${price:,.2f}</div>
</div>
<div class="confidence-container">
<div class="confidence-label">
<span>AI CONFIDENCE</span>
<span style="color: {color}; font-weight: 700;">{conf}%</span>
</div>
<div class="confidence-bar">
<div class="confidence-fill" style="width: {conf}%; color: {color};"></div>
</div>
</div>
<div class="stats-grid">
<div class="stat-box">
<div class="stat-label">Entry Point</div>
<div class="stat-value stat-value-blue">${entry:,.2f}</div>
</div>
<div class="stat-box">
<div class="stat-label">Stop Loss</div>
<div class="stat-value stat-value-red">${sl:,.2f}</div>
</div>
<div class="stat-box">
<div class="stat-label">Take Profit</div>
<div class="stat-value stat-value-green">${tp:,.2f}</div>
</div>
</div>
<div class="strategy-box" style="color: {color};">
<div class="strategy-label">Strategy Analysis</div>
<div class="strategy-text">{details}</div>
</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Apply Styles
    load_custom_css()
    
    # 2. Check Connection
    if not supabase:
        st.error("❌ CRITICAL: Database connection failed. Check .env or Secrets.")
        st.stop()
    
    # 3. Render Enhanced Header
    current_time = datetime.now().strftime("%H:%M:%S UTC")
    st.markdown(f"""
    <div class="titan-header">
        <div class="titan-header-content">
            <div class="titan-branding">
                <div class="titan-title">⚡ TITAN ORACLE</div>
                <div class="titan-subtitle">Enterprise Trading Intelligence • V90 Core</div>
            </div>
            <div class="titan-meta">
                <div class="status-badge">
                    <span class="status-dot"></span>
                    SYSTEM ONLINE
                </div>
                <div class="live-time">⏱ {current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Top Statistics Dashboard
    stats = get_24h_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        (col1, "Total Signals", stats["total"] if stats else 0, "stat-value-blue"),
        (col2, "Buy Signals", stats["buy"] if stats else 0, "stat-value-green"),
        (col3, "Sell Signals", stats["sell"] if stats else 0, "stat-value-red"),
        (col4, "Avg Confidence", f'{stats["confidence"] if stats else 0:.0f}%', "stat-value-purple")
    ]
    
    for col, label, value, color_class in metrics:
        with col:
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value {color_class}">{value}</div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Asset Tabs with Enhanced UI
    tabs = st.tabs(AppConfig.ASSETS)
    
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            # Fetch Data
            signal_data = get_latest_signal(symbol)
            
            # Layout: Panel Left, Chart Right
            col_left, col_right = st.columns([1, 1.5])
            
            with col_left:
                render_signal_panel(symbol, signal_data)
                
            with col_right:
                st.markdown(f"""
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">📈 {symbol} Price Action</div>
                        <div class="chart-timeframe">4H Timeframe</div>
                    </div>
                """, unsafe_allow_html=True)
                
                df = get_price_history(symbol, hours=4)
                if not df.empty:
                    chart = create_price_chart(df, signal_data)
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 3rem 0; color: #64748B; font-family: 'Space Mono', monospace;">
                        ⏳ Waiting for price feed data...
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    # 6. Auto-Refresh
    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
