"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 DASHBOARD - NEXT-GEN TRADING INTERFACE
═══════════════════════════════════════════════════════════════════════════════
Professional Real-Time Trading Terminal powered by TITAN V90 Backend
Visualizes market data, AI signals, and performance metrics via Streamlit
Design Philosophy: Neo-Financial Brutalism with Dynamic Market Reactions
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
# VISUAL STYLING (CSS ENGINE) - NEO-FINANCIAL BRUTALISM
# ═══════════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Injects the Neo-Financial Brutalism Style with Dynamic Market Reactions"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@300;400;600;800&family=Syne:wght@400;600;700;800&display=swap');
        
        /* === COLOR SYSTEM === 
           Background: Deep Space (#0A0E12)
           Surface: Carbon (#13171D)
           Accent Bull: Electric Cyan (#00FFF0)
           Accent Bear: Neon Magenta (#FF006E)
           Neutral: Steel (#8892A0)
           Text: Platinum (#E8ECF1)
        */

        /* === GLOBAL RESET & BASE === */
        * { 
            box-sizing: border-box; 
            margin: 0;
            padding: 0;
        }
        
        .main { 
            background: linear-gradient(135deg, #0A0E12 0%, #0D1117 50%, #0A0E12 100%);
            background-attachment: fixed;
            color: #E8ECF1;
            font-family: 'JetBrains Mono', monospace;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Animated background grid */
        .main::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 255, 240, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 240, 0.02) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
            animation: gridMove 20s linear infinite;
        }
        
        @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }
        
        h1, h2, h3, h4, h5, h6 { 
            font-family: 'Syne', sans-serif !important;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { 
            padding-top: 1rem !important; 
            padding-bottom: 2rem !important;
            position: relative;
            z-index: 1;
        }
        
        /* === STREAMLIT BUTTON STYLING === */
        .stButton>button {
            background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
            color: #E8ECF1;
            border: 2px solid #2A3340;
            border-radius: 12px;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.05em;
            padding: 0.75rem 1.5rem;
            text-transform: uppercase;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 240, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .stButton>button:hover::before {
            left: 100%;
        }
        
        .stButton>button:hover {
            border-color: #00FFF0;
            color: #00FFF0;
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 255, 240, 0.2);
        }

        /* === HEADER SECTION === */
        .titan-header {
            background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
            border: 2px solid #2A3340;
            border-radius: 20px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }
        
        /* Animated accent line */
        .titan-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, 
                transparent, 
                #00FFF0, 
                #FF006E, 
                transparent
            );
            animation: shimmer 3s linear infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .titan-branding { 
            display: flex; 
            flex-direction: column;
            position: relative;
            z-index: 2;
        }
        
        .titan-title {
            font-size: 4rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #FFFFFF 0%, #00FFF0 50%, #FF006E 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.03em;
            text-shadow: 0 0 40px rgba(0, 255, 240, 0.3);
            animation: titleGlow 4s ease-in-out infinite;
        }
        
        @keyframes titleGlow {
            0%, 100% { filter: brightness(1); }
            50% { filter: brightness(1.2); }
        }
        
        .titan-subtitle {
            font-family: 'Space Mono', monospace;
            font-size: 0.95rem;
            color: #8892A0;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-top: 0.5rem;
            font-weight: 400;
            opacity: 0.9;
        }
        
        /* === STATUS BADGE === */
        .status-badge {
            background: linear-gradient(135deg, rgba(0, 255, 240, 0.1) 0%, rgba(0, 255, 240, 0.05) 100%);
            border: 2px solid rgba(0, 255, 240, 0.3);
            color: #00FFF0;
            padding: 0.6rem 1.2rem;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'Space Mono', monospace;
            display: flex;
            align-items: center;
            gap: 10px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            box-shadow: 0 0 20px rgba(0, 255, 240, 0.2);
        }
        
        .status-dot {
            width: 10px; 
            height: 10px; 
            background: #00FFF0;
            border-radius: 50%; 
            box-shadow: 0 0 12px #00FFF0;
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        @keyframes pulse { 
            0%, 100% { 
                opacity: 1; 
                transform: scale(1);
            } 
            50% { 
                opacity: 0.5; 
                transform: scale(1.1);
            } 
        }

        /* === TABS STYLING === */
        .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
            padding: 10px;
            border-radius: 16px;
            border: 2px solid #2A3340;
            gap: 8px;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #5A6678;
            border-radius: 10px;
            border: none;
            padding: 12px 24px;
            font-family: 'Syne', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            letter-spacing: 0.02em;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #8892A0;
            background: rgba(136, 146, 160, 0.05);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 255, 240, 0.15) 0%, rgba(0, 255, 240, 0.08) 100%);
            color: #00FFF0 !important;
            border: 2px solid rgba(0, 255, 240, 0.3);
            box-shadow: 
                0 4px 16px rgba(0, 255, 240, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        /* === SIGNAL CARDS - The Heart of the Design === */
        .signal-card {
            background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
            border: 2px solid #2A3340;
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .signal-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.03) 0%, transparent 70%);
            pointer-events: none;
        }
        
        .signal-card:hover { 
            transform: translateY(-4px); 
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        /* Signal type variants */
        .signal-card-buy { 
            border-left: 4px solid #00FFF0;
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.4),
                -4px 0 20px rgba(0, 255, 240, 0.2);
        }
        
        .signal-card-sell { 
            border-left: 4px solid #FF006E;
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.4),
                -4px 0 20px rgba(255, 0, 110, 0.2);
        }
        
        .signal-card-wait { 
            border-left: 4px solid #5A6678;
            opacity: 0.7;
        }
        
        .signal-card-closed { 
            border-left: 4px solid #3A4350;
            background: linear-gradient(135deg, #0F1216 0%, #13171D 100%);
            opacity: 0.6;
        }

        /* Typography inside cards */
        .signal-type {
            font-size: 2.5rem; 
            font-weight: 800; 
            letter-spacing: -0.02em;
            font-family: 'Syne', sans-serif;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .signal-symbol {
            font-size: 0.85rem; 
            color: #8892A0;
            font-weight: 700; 
            letter-spacing: 0.15em; 
            margin-bottom: 0.5rem;
            font-family: 'Space Mono', monospace;
            text-transform: uppercase;
        }
        
        .price-display {
            font-size: 3.5rem; 
            font-weight: 800; 
            color: #FFFFFF;
            font-family: 'Syne', sans-serif;
            letter-spacing: -0.02em;
            text-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
            line-height: 1;
        }
        
        .price-label { 
            font-size: 0.7rem; 
            color: #5A6678;
            text-transform: uppercase; 
            letter-spacing: 0.12em;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        /* === STATS GRID === */
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 12px; 
            margin-top: 1.5rem;
        }
        
        .stat-box {
            background: linear-gradient(135deg, rgba(42, 51, 64, 0.3) 0%, rgba(42, 51, 64, 0.1) 100%);
            border: 1px solid #2A3340;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stat-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, transparent, rgba(255, 255, 255, 0.02));
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .stat-box:hover::before {
            opacity: 1;
        }
        
        .stat-box:hover {
            border-color: #3A4350;
            transform: translateY(-2px);
        }
        
        .stat-label { 
            font-size: 0.65rem; 
            color: #5A6678;
            text-transform: uppercase; 
            margin-bottom: 0.5rem;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            letter-spacing: 0.1em;
        }
        
        .stat-value { 
            font-size: 1.3rem; 
            font-weight: 800; 
            color: #E8ECF1;
            font-family: 'Syne', sans-serif;
            letter-spacing: -0.01em;
        }
        
        .val-buy { 
            color: #00FFF0;
            text-shadow: 0 0 10px rgba(0, 255, 240, 0.3);
        }
        
        .val-sell { 
            color: #FF006E;
            text-shadow: 0 0 10px rgba(255, 0, 110, 0.3);
        }
        
        .val-blue { 
            color: #5B9FFF;
            text-shadow: 0 0 10px rgba(91, 159, 255, 0.3);
        }

        /* === CHART CONTAINER === */
        .chart-container {
            background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
            border: 2px solid #2A3340;
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        .chart-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #2A3340;
        }
        
        .chart-title { 
            font-size: 1.2rem; 
            font-family: 'Syne', sans-serif; 
            font-weight: 800; 
            color: #E8ECF1;
            letter-spacing: -0.01em;
        }
        
        .chart-badge { 
            background: rgba(42, 51, 64, 0.5);
            padding: 6px 14px; 
            border-radius: 8px; 
            font-size: 0.75rem;
            color: #8892A0;
            border: 1px solid #2A3340;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        /* === TOP METRICS === */
        .metric-card-top {
            background: linear-gradient(135deg, #13171D 0%, #1A1F28 100%);
            border: 2px solid #2A3340;
            border-radius: 16px; 
            padding: 1.5rem; 
            text-align: center;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .metric-card-top::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, transparent, rgba(255, 255, 255, 0.02));
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .metric-card-top:hover {
            transform: translateY(-4px);
            border-color: #3A4350;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card-top:hover::before {
            opacity: 1;
        }
        
        .metric-label-top { 
            font-size: 0.7rem; 
            color: #5A6678;
            margin-bottom: 0.75rem;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        
        .metric-val-top { 
            font-size: 2.2rem; 
            font-weight: 800; 
            color: #E8ECF1;
            font-family: 'Syne', sans-serif;
            letter-spacing: -0.02em;
            line-height: 1;
        }

        /* === CONFIDENCE PROGRESS BAR === */
        .confidence-container {
            background: rgba(42, 51, 64, 0.3);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            border: 1px solid #2A3340;
        }
        
        .confidence-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }
        
        .confidence-label {
            color: #5A6678;
            font-size: 0.75rem;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        
        .confidence-value {
            font-weight: 800;
            font-size: 0.9rem;
            font-family: 'Syne', sans-serif;
        }
        
        .progress-bar-bg {
            background: #1A1F28;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .progress-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 12px currentColor;
        }

        /* === SCROLLBAR STYLING === */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #13171D;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #2A3340;
            border-radius: 5px;
            border: 2px solid #13171D;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #3A4350;
        }

        /* === RESPONSIVE ADJUSTMENTS === */
        @media (max-width: 768px) {
            .titan-title { font-size: 2.5rem; }
            .price-display { font-size: 2.5rem; }
            .signal-type { font-size: 2rem; }
            .stats-grid { grid-template-columns: 1fr; }
        }

    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    try:
        if 'SUPABASE_URL' in st.secrets:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        else:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")

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

def get_price_history(symbol, hours=4):
    if not supabase: return pd.DataFrame()
    try:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        response = supabase.table("mt4_feed").select("*").eq("symbol", symbol).gte("created_at", cutoff).order("created_at", desc=False).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

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

def create_price_chart(df, signal_data):
    if df.empty: return None
    
    fig = go.Figure()
    
    # Main price line with gradient fill
    fig.add_trace(go.Scatter(
        x=df['created_at'], 
        y=df['price'], 
        mode='lines', 
        name='Price',
        line=dict(color='#00FFF0', width=3, shape='spline'),
        fill='tozeroy', 
        fillcolor='rgba(0, 255, 240, 0.08)',
        hovertemplate='<b>%{y:,.2f}</b><br>%{x}<extra></extra>'
    ))
    
    # Add signal levels if active
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        if entry > 0: 
            fig.add_hline(
                y=entry, 
                line_dash="dash", 
                line_color="#5B9FFF", 
                line_width=2,
                annotation_text="ENTRY",
                annotation_position="right",
                annotation=dict(font_size=10, font_color="#5B9FFF")
            )
        if sl > 0: 
            fig.add_hline(
                y=sl, 
                line_dash="dot", 
                line_color="#FF006E", 
                line_width=2,
                annotation_text="STOP LOSS",
                annotation_position="right",
                annotation=dict(font_size=10, font_color="#FF006E")
            )
        if tp > 0: 
            fig.add_hline(
                y=tp, 
                line_dash="dot", 
                line_color="#00FFF0", 
                line_width=2,
                annotation_text="TARGET",
                annotation_position="right",
                annotation=dict(font_size=10, font_color="#00FFF0")
            )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            showgrid=True, 
            gridcolor='rgba(42, 51, 64, 0.3)',
            gridwidth=1,
            title="",
            showline=False,
            color='#5A6678'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(42, 51, 64, 0.3)',
            gridwidth=1,
            title="",
            showline=False,
            color='#5A6678'
        ),
        hovermode='x unified',
        font=dict(
            color='#E8ECF1', 
            family='JetBrains Mono, monospace',
            size=11
        ),
        hoverlabel=dict(
            bgcolor='#13171D',
            font_size=12,
            font_family='JetBrains Mono, monospace'
        )
    )
    
    return fig

def render_signal_panel(symbol, signal_data):
    # --- MARKET CLOSED LOGIC (10 Minutes) ---
    created_at = signal_data.get('created_at', '') if signal_data else ''
    is_stale = False
    time_str = "Waiting..."
    
    if created_at:
        try:
            signal_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            time_diff = (datetime.now(signal_time.tzinfo) - signal_time).total_seconds()
            if time_diff > 600: is_stale = True # 10 minutes
            
            if time_diff < 60: time_str = "Now"
            elif time_diff < 3600: time_str = f"{int(time_diff/60)}m ago"
            else: time_str = f"{int(time_diff/3600)}h ago"
        except: pass

    # --- RENDER CARD: MARKET CLOSED ---
    if not signal_data or is_stale:
        last_price = signal_data.get('current_price', 0) if signal_data else 0
        # CORREZIONE: HTML allineato a sinistra per evitare che markdown lo interpreti come codice
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

    # --- RENDER CARD: ACTIVE SIGNAL ---
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0)
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', 'AI Analysis')

    # Dynamic colors based on signal type
    if rec == 'BUY': 
        card_cls, icon, col = "signal-card-buy", "▲", "#00FFF0"
    elif rec == 'SELL': 
        card_cls, icon, col = "signal-card-sell", "▼", "#FF006E"
    else: 
        card_cls, icon, col = "signal-card-wait", "●", "#5A6678"

    # CORREZIONE: HTML allineato a sinistra per evitare che markdown lo interpreti come codice
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

    <div class="confidence-container">
        <div class="confidence-header">
            <span class="confidence-label">AI Confidence</span>
            <span class="confidence-value" style="color:{col};">{conf}%</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{conf}%; background:{col};"></div>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-label">Entry</div>
            <div class="stat-value val-blue">${entry:,.2f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Stop Loss</div>
            <div class="stat-value val-sell">${sl:,.2f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Target</div>
            <div class="stat-value val-buy">${tp:,.2f}</div>
        </div>
    </div>
    
    <div style="margin-top:1.5rem; padding-top:1.5rem; border-top:1px solid #2A3340; color:#5A6678; font-size:0.8rem; text-align:center; font-family: 'JetBrains Mono', monospace;">
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
        st.error("❌ Database connection error.")
        st.stop()
    
    # HEADER
    st.markdown("""
<div class="titan-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:20px;">
        <div class="titan-branding">
            <div class="titan-title">TITAN ORACLE</div>
            <div class="titan-subtitle">Next-Gen Trading Intelligence</div>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> SYSTEM ACTIVE
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
    
    with c1: 
        st.markdown(f'''<div class="metric-card-top">
            <div class="metric-label-top">Total Signals</div>
            <div class="metric-val-top val-blue">{vals["total"]}</div>
        </div>''', unsafe_allow_html=True)
    
    with c2: 
        st.markdown(f'''<div class="metric-card-top">
            <div class="metric-label-top">Buy Signals</div>
            <div class="metric-val-top val-buy">{vals["buy"]}</div>
        </div>''', unsafe_allow_html=True)
    
    with c3: 
        st.markdown(f'''<div class="metric-card-top">
            <div class="metric-label-top">Sell Signals</div>
            <div class="metric-val-top val-sell">{vals["sell"]}</div>
        </div>''', unsafe_allow_html=True)
    
    with c4: 
        st.markdown(f'''<div class="metric-card-top">
            <div class="metric-label-top">Avg Confidence</div>
            <div class="metric-val-top" style="color:#E8ECF1;">{vals["conf"]:.0f}%</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TABS & CONTENT
    tabs = st.tabs(AppConfig.ASSETS)
    
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            col_left, col_right = st.columns([1, 1.6])
            signal_data = get_latest_signal(symbol)
            
            with col_left:
                render_signal_panel(symbol, signal_data)
                
            with col_right:
                st.markdown(f"""
<div class="chart-container">
    <div class="chart-header">
        <div class="chart-title">Price Action</div>
        <div class="chart-badge">{symbol}</div>
    </div>
""", unsafe_allow_html=True)
                
                df = get_price_history(symbol, hours=4)
                if not df.empty:
                    chart = create_price_chart(df, signal_data)
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("📡 Awaiting market data feed...")
                
                st.markdown("</div>", unsafe_allow_html=True)

    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
