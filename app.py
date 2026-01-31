"""
═══════════════════════════════════════════════════════════════════════════════
TITAN ORACLE DASHBOARD - STREAMLIT FRONTEND
═══════════════════════════════════════════════════════════════════════════════
Real-time trading signals visualization powered by TITAN V90 Oracle Prime
Author: TITAN Trading Systems
Version: 1.0
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from supabase import create_client
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & SETUP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TITAN Oracle Prime",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caricamento Variabili d'Ambiente (Ibrido: Locale + Cloud)
load_dotenv()

# Tenta di prendere le credenziali da Streamlit Secrets (Cloud) o .env (Locale)
try:
    if 'SUPABASE_URL' in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    else:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
except:
    # Fallback a .env (locale)
    from dotenv import load_dotenv
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Assets tracking
ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"]

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS (CYBERPUNK THEME)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #00d9ff;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.1);
    }
    
    /* Signal cards */
    .signal-buy {
        background: linear-gradient(135deg, #0f2e1d 0%, #1a4a30 100%);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #2e0f0f 0%, #4a1a1a 100%);
        border: 2px solid #ff0044;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 15px rgba(255, 0, 68, 0.2);
    }
    
    .signal-wait {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #444;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        opacity: 0.8;
    }
    
    /* Typography */
    .price-big {
        font-size: 36px;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .signal-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-label {
        font-size: 12px;
        color: #8b949e;
        margin-bottom: 2px;
        text-transform: uppercase;
    }
    
    .stat-value {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }
    
    /* Animations */
    .status-active {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #00ff88;
        box-shadow: 0 0 10px #00ff88;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d9ff 0%, #0099ff 100%);
        color: white;
        border: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    """Initialize Supabase client safely"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return client
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

supabase = init_supabase()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_latest_signals():
    """Fetch latest trading signals for all assets"""
    if not supabase: return {}
    
    try:
        signals = {}
        for symbol in ASSETS:
            response = supabase.table("ai_oracle")\
                .select("*")\
                .eq("symbol", symbol)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if response.data:
                signals[symbol] = response.data[0]
        return signals
    except Exception:
        return {}

def get_price_history(symbol, hours=4):
    """Fetch price history for charting"""
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
    except Exception:
        return pd.DataFrame()

def get_performance_stats():
    """Calculate overall performance statistics"""
    if not supabase: return None
    
    try:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        response = supabase.table("ai_oracle")\
            .select("*")\
            .gte("created_at", cutoff)\
            .in_("recommendation", ["BUY", "SELL"])\
            .execute()
        
        if not response.data: return None
        
        total_signals = len(response.data)
        buy_signals = sum(1 for s in response.data if s['recommendation'] == 'BUY')
        sell_signals = sum(1 for s in response.data if s['recommendation'] == 'SELL')
        avg_confidence = sum(s.get('confidence_score', 0) for s in response.data) / total_signals if total_signals > 0 else 0
        
        return {
            'total': total_signals,
            'buy': buy_signals,
            'sell': sell_signals,
            'confidence': avg_confidence
        }
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df, signal_data):
    """Create interactive price chart with Plotly"""
    if df.empty: return None
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['created_at'], y=df['price'],
        mode='lines', name='Price',
        line=dict(color='#00d9ff', width=2),
        fill='tozeroy', fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    # Add entry/SL/TP lines if signal exists
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        if entry: fig.add_hline(y=entry, line_dash="dash", line_color="white", annotation_text="ENTRY")
        if sl: fig.add_hline(y=sl, line_dash="dot", line_color="#ff0044", annotation_text="SL")
        if tp: fig.add_hline(y=tp, line_dash="dot", line_color="#00ff88", annotation_text="TP")
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Price"),
        hovermode='x unified',
        showlegend=False
    )
    return fig

def render_signal_card(symbol, signal_data):
    """Render a trading signal card"""
    if not signal_data:
        st.markdown(f"""
        <div class="signal-wait">
            <div class="signal-title">⚪ {symbol}</div>
            <div style="color: #aaa; font-size: 14px;">Scanning Market...</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0)
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', '')
    
    # Card Logic
    card_class = "signal-wait"
    icon = "⚪"
    color = "#888"
    
    if rec == 'BUY':
        card_class = "signal-buy"
        icon = "🟢"
        color = "#00ff88"
    elif rec == 'SELL':
        card_class = "signal-sell"
        icon = "🔴"
        color = "#ff0044"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="signal-title">{icon} {symbol} &nbsp; {rec}</div>
            <div style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px; font-weight: bold; color: {color};">
                {conf}% CONFIDENCE
            </div>
        </div>
        
        <div class="price-big">${price:,.2f}</div>
        
        <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
            <div>
                <div class="stat-label">ENTRY</div>
                <div class="stat-value" style="color: #fff;">${entry:,.2f}</div>
            </div>
            <div>
                <div class="stat-label">STOP LOSS</div>
                <div class="stat-value" style="color: #ff0044;">${sl:,.2f}</div>
            </div>
            <div>
                <div class="stat-label">TAKE PROFIT</div>
                <div class="stat-value" style="color: #00ff88;">${tp:,.2f}</div>
            </div>
        </div>
        
        <div style="margin-top: 15px; font-size: 12px; color: #888; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
            DETAILS: {details}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding-bottom: 20px;">
        <h1 style="color: #00d9ff; font-size: 42px; margin: 0; text-shadow: 0 0 20px rgba(0,217,255,0.5);">
            🏛️ TITAN ORACLE PRIME
        </h1>
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 10px;">
            <div class="status-active"></div>
            <span style="color: #00ff88; font-weight: bold; font-size: 14px;">SYSTEM ONLINE</span>
            <span style="color: #666;">•</span>
            <span style="color: #aaa; font-size: 14px;">V90 ENTERPRISE CORE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ CONTROL PANEL")
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 ASSETS")
        selected_assets = st.multiselect("Active Feeds", ASSETS, default=ASSETS)
        
        st.markdown("---")
        st.markdown("### 📈 24H STATISTICS")
        stats = get_performance_stats()
        
        if stats:
            c1, c2 = st.columns(2)
            c1.metric("Signals", stats['total'])
            c1.metric("Buy", stats['buy'])
            c2.metric("Confidence", f"{stats['confidence']:.0f}%")
            c2.metric("Sell", stats['sell'])
        else:
            st.info("No data in 24h")
            
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.rerun()

    # Main Grid
    if not supabase:
        st.error("🚨 CRITICAL: Database connection failed. Check Credentials.")
        st.stop()
        
    signals = get_latest_signals()
    
    if not selected_assets:
        st.warning("Select assets from sidebar")
        return

    # Dynamic Grid
    cols = st.columns(2)
    for idx, symbol in enumerate(selected_assets):
        with cols[idx % 2]:
            signal_data = signals.get(symbol)
            render_signal_card(symbol, signal_data)
            
            # Chart Toggle
            with st.expander(f"📈 {symbol} Analysis Chart", expanded=False):
                df = get_price_history(symbol, hours=4)
                if not df.empty:
                    chart = create_price_chart(df, signal_data)
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.caption("Waiting for price feed...")

    # Auto Refresh Loop
    if auto_refresh:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
