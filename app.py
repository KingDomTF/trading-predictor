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

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TITAN Oracle Prime",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Assets tracking
ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"]

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #00d9ff;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2);
    }
    
    /* Signal cards */
    .signal-buy {
        background: linear-gradient(135deg, #1e3c28 0%, #2d5f3e 100%);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 24px rgba(0, 255, 136, 0.3);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #3c1e1e 0%, #5f2d2d 100%);
        border: 2px solid #ff0044;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 24px rgba(255, 0, 68, 0.3);
    }
    
    .signal-wait {
        background: linear-gradient(135deg, #2a2a3c 0%, #3a3a4e 100%);
        border: 2px solid #888888;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        opacity: 0.6;
    }
    
    /* Text styles */
    .price-big {
        font-size: 48px;
        font-weight: bold;
        color: #00d9ff;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
    }
    
    .signal-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 14px;
        color: #aaaaaa;
        margin-bottom: 5px;
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: bold;
        color: #ffffff;
    }
    
    /* Status indicators */
    .status-active {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #00ff88;
        box-shadow: 0 0 12px rgba(0, 255, 136, 0.8);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #1a1a2e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
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
    if not supabase:
        return {}
    
    try:
        # Get latest signal for each symbol
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
    except Exception as e:
        st.error(f"Error fetching signals: {e}")
        return {}

def get_price_history(symbol, hours=4):
    """Fetch price history for charting"""
    if not supabase:
        return pd.DataFrame()
    
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
    except Exception as e:
        st.error(f"Error fetching price history: {e}")
        return pd.DataFrame()

def get_performance_stats():
    """Calculate overall performance statistics"""
    if not supabase:
        return None
    
    try:
        # Get all signals from last 24 hours
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        
        response = supabase.table("ai_oracle")\
            .select("*")\
            .gte("created_at", cutoff)\
            .in_("recommendation", ["BUY", "SELL"])\
            .execute()
        
        if not response.data:
            return None
        
        total_signals = len(response.data)
        buy_signals = sum(1 for s in response.data if s['recommendation'] == 'BUY')
        sell_signals = sum(1 for s in response.data if s['recommendation'] == 'SELL')
        
        # Calculate average confidence
        avg_confidence = sum(s.get('confidence_score', 0) for s in response.data) / total_signals if total_signals > 0 else 0
        
        return {
            'total': total_signals,
            'buy': buy_signals,
            'sell': sell_signals,
            'confidence': avg_confidence
        }
    except Exception as e:
        st.error(f"Error calculating stats: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df, signal_data):
    """Create interactive price chart with Plotly"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['created_at'],
        y=df['price'],
        mode='lines',
        name='Price',
        line=dict(color='#00d9ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    # Add entry/SL/TP lines if signal exists
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        # Entry line
        fig.add_hline(y=entry, line_dash="dash", line_color="#ffffff", 
                     annotation_text="Entry", annotation_position="right")
        
        # Stop Loss
        fig.add_hline(y=sl, line_dash="dot", line_color="#ff0044",
                     annotation_text="Stop Loss", annotation_position="right")
        
        # Take Profit
        fig.add_hline(y=tp, line_dash="dot", line_color="#00ff88",
                     annotation_text="Take Profit", annotation_position="right")
    
    # Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title=""
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title="Price"
        ),
        hovermode='x unified'
    )
    
    return fig

def render_signal_card(symbol, signal_data):
    """Render a trading signal card"""
    
    if not signal_data:
        st.markdown(f"""
        <div class="signal-wait">
            <div class="signal-title">⚪ {symbol} - NO DATA</div>
            <p>Waiting for signal...</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    recommendation = signal_data.get('recommendation', 'WAIT')
    current_price = signal_data.get('current_price', 0)
    entry_price = signal_data.get('entry_price', 0)
    stop_loss = signal_data.get('stop_loss', 0)
    take_profit = signal_data.get('take_profit', 0)
    confidence = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', 'No details')
    created_at = signal_data.get('created_at', '')
    
    # Calculate PnL if active trade
    pnl = 0
    pnl_pct = 0
    if recommendation == 'BUY' and entry_price > 0:
        pnl = current_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
    elif recommendation == 'SELL' and entry_price > 0:
        pnl = entry_price - current_price
        pnl_pct = (pnl / entry_price) * 100
    
    # Determine card style
    if recommendation == 'BUY':
        card_class = "signal-buy"
        icon = "🟢"
        color = "#00ff88"
    elif recommendation == 'SELL':
        card_class = "signal-sell"
        icon = "🔴"
        color = "#ff0044"
    else:
        card_class = "signal-wait"
        icon = "⚪"
        color = "#888888"
    
    # Time ago
    try:
        signal_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        time_ago = (datetime.now(signal_time.tzinfo) - signal_time).total_seconds()
        if time_ago < 60:
            time_str = f"{int(time_ago)}s ago"
        elif time_ago < 3600:
            time_str = f"{int(time_ago/60)}m ago"
        else:
            time_str = f"{int(time_ago/3600)}h ago"
    except:
        time_str = "Unknown"
    
    # Render card
    st.markdown(f"""
    <div class="{card_class}">
        <div class="signal-title">{icon} {symbol} - {recommendation}</div>
        <div class="price-big">${current_price:,.2f}</div>
        <div style="margin-top: 20px;">
            <div class="stat-label">CONFIDENCE</div>
            <div class="stat-value" style="color: {color};">{confidence}%</div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.2); margin: 15px 0;">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
            <div>
                <div class="stat-label">ENTRY</div>
                <div class="stat-value" style="font-size: 20px;">${entry_price:,.2f}</div>
            </div>
            <div>
                <div class="stat-label">STOP LOSS</div>
                <div class="stat-value" style="font-size: 20px; color: #ff0044;">${stop_loss:,.2f}</div>
            </div>
            <div>
                <div class="stat-label">TAKE PROFIT</div>
                <div class="stat-value" style="font-size: 20px; color: #00ff88;">${take_profit:,.2f}</div>
            </div>
        </div>
        {f'''<div style="margin-top: 15px;">
            <div class="stat-label">CURRENT P&L</div>
            <div class="stat-value" style="font-size: 24px; color: {'#00ff88' if pnl >= 0 else '#ff0044'};">
                ${pnl:,.2f} ({pnl_pct:+.2f}%)
            </div>
        </div>''' if recommendation in ['BUY', 'SELL'] else ''}
        <div style="margin-top: 15px; font-size: 14px; color: #aaaaaa;">
            {details} • {time_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #00d9ff; font-size: 48px; margin: 0;">
            🏛️ TITAN ORACLE PRIME
        </h1>
        <p style="color: #aaaaaa; font-size: 18px;">
            Enterprise Trading Intelligence • Real-Time Signals
        </p>
        <div class="status-active" style="margin: 10px auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ CONTROL PANEL")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        st.markdown("---")
        
        # Asset filter
        st.markdown("### 📊 ASSETS")
        selected_assets = st.multiselect(
            "Select assets to display",
            ASSETS,
            default=ASSETS
        )
        
        st.markdown("---")
        
        # Performance stats
        st.markdown("### 📈 24H STATISTICS")
        stats = get_performance_stats()
        
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Signals", stats['total'])
                st.metric("Buy Signals", stats['buy'], delta_color="normal")
            with col2:
                st.metric("Sell Signals", stats['sell'], delta_color="inverse")
                st.metric("Avg Confidence", f"{stats['confidence']:.0f}%")
        else:
            st.info("No signals in last 24h")
        
        st.markdown("---")
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 12px;">
            <p>TITAN Trading Systems</p>
            <p>v1.0 Oracle Prime</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    signals = get_latest_signals()
    
    # Display signals
    if not selected_assets:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        return
    
    # Grid layout for signal cards
    cols = st.columns(2)
    
    for idx, symbol in enumerate(selected_assets):
        with cols[idx % 2]:
            signal_data = signals.get(symbol)
            render_signal_card(symbol, signal_data)
            
            # Price chart
            with st.expander(f"📈 {symbol} Price Chart (4H)", expanded=False):
                df = get_price_history(symbol, hours=4)
                if not df.empty:
                    chart = create_price_chart(df, signal_data)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("No price history available")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("❌ Environment variables not set. Create a .env file with SUPABASE_URL and SUPABASE_KEY")
        st.stop()
    
    main()
