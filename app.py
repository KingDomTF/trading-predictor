"""
═══════════════════════════════════════════════════════════════════════════════
TITAN ORACLE DASHBOARD - PREMIUM UI WITH LATEST FEATURES
═══════════════════════════════════════════════════════════════════════════════
Beautiful real-time trading dashboard with asset tabs and modern design
Compatible with TITAN V90 backend
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os

# Supabase connection
try:
    from supabase import create_client
    # Try Streamlit Cloud secrets first, fallback to .env
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    except:
        from dotenv import load_dotenv
        load_dotenv()
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
except ImportError:
    st.error("Missing dependencies. Run: pip install supabase python-dotenv")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TITAN Oracle Prime",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Default assets
DEFAULT_ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30", "ETHUSD"]

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* === GLOBAL THEME === */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    }
    
    /* === HIDE STREAMLIT DEFAULTS === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
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
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-shadow: 0 0 30px rgba(0,217,255,0.5);
    }
    
    .titan-subtitle {
        font-size: 18px;
        color: #aaaaaa;
        margin-top: 10px;
    }
    
    .status-badge {
        display: inline-block;
        background: rgba(0,255,136,0.2);
        border: 1px solid #00ff88;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 14px;
        color: #00ff88;
        margin-top: 15px;
    }
    
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px #00ff88; }
        50% { opacity: 0.5; box-shadow: 0 0 20px #00ff88; }
    }
    
    /* === ASSET TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(26,31,58,0.6);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(22,33,62,0.8) 0%, rgba(15,52,96,0.8) 100%);
        border: 2px solid rgba(0,217,255,0.3);
        border-radius: 12px;
        padding: 12px 24px;
        color: #ffffff;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(0,217,255,0.2) 0%, rgba(0,153,255,0.2) 100%);
        border-color: #00d9ff;
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,217,255,0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
        border-color: #00d9ff;
        color: #000000;
        box-shadow: 0 8px 25px rgba(0,217,255,0.4);
    }
    
    /* === SIGNAL CARDS === */
    .signal-card-buy {
        background: linear-gradient(135deg, rgba(30,60,40,0.95) 0%, rgba(45,95,62,0.95) 100%);
        border: 2px solid #00ff88;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0,255,136,0.3);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .signal-card-buy:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,255,136,0.5);
    }
    
    .signal-card-sell {
        background: linear-gradient(135deg, rgba(60,30,30,0.95) 0%, rgba(95,45,45,0.95) 100%);
        border: 2px solid #ff0044;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(255,0,68,0.3);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .signal-card-sell:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(255,0,68,0.5);
    }
    
    .signal-card-wait {
        background: linear-gradient(135deg, rgba(42,42,60,0.8) 0%, rgba(58,58,78,0.8) 100%);
        border: 2px solid rgba(136,136,136,0.5);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        opacity: 0.7;
    }
    
    .signal-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    
    .signal-type {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .price-display {
        font-size: 56px;
        font-weight: 900;
        margin: 20px 0;
        text-shadow: 0 0 20px currentColor;
    }
    
    .confidence-bar {
        background: rgba(255,255,255,0.1);
        height: 8px;
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff88 0%, #00d9ff 100%);
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    /* === STATS GRID === */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin: 25px 0;
    }
    
    .stat-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    
    .stat-label {
        font-size: 12px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: bold;
        color: #ffffff;
    }
    
    .stat-value-green {
        color: #00ff88;
    }
    
    .stat-value-red {
        color: #ff0044;
    }
    
    .stat-value-blue {
        color: #00d9ff;
    }
    
    /* === PNL DISPLAY === */
    .pnl-section {
        background: rgba(0,0,0,0.3);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .pnl-positive {
        color: #00ff88;
        font-size: 32px;
        font-weight: bold;
    }
    
    .pnl-negative {
        color: #ff0044;
        font-size: 32px;
        font-weight: bold;
    }
    
    /* === DETAILS SECTION === */
    .details-box {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #00d9ff;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
        font-size: 14px;
        color: #aaaaaa;
    }
    
    /* === METRIC CARDS === */
    .metric-card {
        background: linear-gradient(135deg, rgba(22,33,62,0.6) 0%, rgba(15,52,96,0.6) 100%);
        border: 1px solid rgba(0,217,255,0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* === BUTTONS === */
    .stButton>button {
        background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,217,255,0.4);
    }
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
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_latest_signal(symbol):
    """Get latest signal for specific symbol"""
    if not supabase:
        return None
    
    try:
        response = supabase.table("ai_oracle")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        st.error(f"Error fetching signal for {symbol}: {e}")
        return None

def get_price_history(symbol, hours=4):
    """Get price history for charting"""
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
    except:
        return pd.DataFrame()

def get_24h_stats():
    """Get 24-hour performance statistics"""
    if not supabase:
        return None
    
    try:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        
        response = supabase.table("ai_oracle")\
            .select("*")\
            .gte("created_at", cutoff)\
            .in_("recommendation", ["BUY", "SELL"])\
            .execute()
        
        if not response.data:
            return None
        
        total = len(response.data)
        buy = sum(1 for s in response.data if s['recommendation'] == 'BUY')
        sell = sum(1 for s in response.data if s['recommendation'] == 'SELL')
        avg_conf = sum(s.get('confidence_score', 0) for s in response.data) / total if total > 0 else 0
        
        return {
            'total': total,
            'buy': buy,
            'sell': sell,
            'confidence': avg_conf
        }
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df, signal_data):
    """Create beautiful price chart"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=df['created_at'],
        y=df['price'],
        mode='lines',
        name='Price',
        line=dict(color='#00d9ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,217,255,0.1)'
    ))
    
    # Add signal levels if available
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        # Entry line
        if entry > 0:
            fig.add_hline(y=entry, line_dash="dash", line_color="#ffffff", 
                         line_width=2, annotation_text="Entry", 
                         annotation_position="right")
        
        # Stop Loss
        if sl > 0 and sl != entry:
            fig.add_hline(y=sl, line_dash="dot", line_color="#ff0044",
                         line_width=2, annotation_text="SL", 
                         annotation_position="right")
        
        # Take Profit
        if tp > 0 and tp != entry:
            fig.add_hline(y=tp, line_dash="dot", line_color="#00ff88",
                         line_width=2, annotation_text="TP", 
                         annotation_position="right")
    
    # Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,39,0.5)',
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            title=""
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            title="Price"
        ),
        hovermode='x unified',
        font=dict(color='#ffffff')
    )
    
    return fig

def render_signal_panel(symbol, signal_data):
    """Render premium signal display panel"""
    
    if not signal_data:
        st.markdown(f"""
        <div class="signal-card-wait">
            <div class="signal-icon">⚪</div>
            <div class="signal-type">{symbol}</div>
            <p style="color: #888;">Waiting for signal...</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Extract data
    recommendation = signal_data.get('recommendation', 'WAIT')
    current_price = signal_data.get('current_price', 0)
    entry_price = signal_data.get('entry_price', 0)
    stop_loss = signal_data.get('stop_loss', 0)
    take_profit = signal_data.get('take_profit', 0)
    confidence = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', 'No details')
    created_at = signal_data.get('created_at', '')
    
    # Calculate PnL if active
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
        card_class = "signal-card-buy"
        icon = "🟢"
        price_color = "#00ff88"
    elif recommendation == 'SELL':
        card_class = "signal-card-sell"
        icon = "🔴"
        price_color = "#ff0044"
    else:
        card_class = "signal-card-wait"
        icon = "⚪"
        price_color = "#888888"
    
    # Time ago calculation
    try:
        signal_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        time_diff = (datetime.now(signal_time.tzinfo) - signal_time).total_seconds()
        if time_diff < 60:
            time_str = f"{int(time_diff)}s ago"
        elif time_diff < 3600:
            time_str = f"{int(time_diff/60)}m ago"
        else:
            time_str = f"{int(time_diff/3600)}h ago"
    except:
        time_str = "Just now"
    
    # Render card
    st.markdown(f"""
    <div class="{card_class}">
        <div class="signal-icon">{icon}</div>
        <div class="signal-type">{symbol} - {recommendation}</div>
        <div class="price-display" style="color: {price_color};">
            ${current_price:,.5f}
        </div>
        
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence}%;"></div>
        </div>
        <div style="text-align: center; color: #aaa; font-size: 14px; margin-top: 5px;">
            Confidence: {confidence}%
        </div>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Entry</div>
                <div class="stat-value stat-value-blue">${entry_price:,.5f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Stop Loss</div>
                <div class="stat-value stat-value-red">${stop_loss:,.5f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Take Profit</div>
                <div class="stat-value stat-value-green">${take_profit:,.5f}</div>
            </div>
        </div>
        
        {f'''
        <div class="pnl-section">
            <div style="text-align: center; color: #aaa; font-size: 14px; margin-bottom: 10px;">
                Current P&L
            </div>
            <div style="text-align: center;" class="{'pnl-positive' if pnl >= 0 else 'pnl-negative'}">
                ${pnl:,.5f} ({pnl_pct:+.2f}%)
            </div>
        </div>
        ''' if recommendation in ['BUY', 'SELL'] else ''}
        
        <div class="details-box">
            <strong>Strategy:</strong> {details}<br>
            <strong>Signal Time:</strong> {time_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="titan-header">
        <div class="titan-title">🏛️ TITAN ORACLE PRIME</div>
        <div class="titan-subtitle">
            Enterprise Trading Intelligence • Real-Time Signal Generation
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get 24h stats for sidebar
    stats = get_24h_stats()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">Total Signals (24h)</div>
            <div class="stat-value stat-value-blue">{}</div>
        </div>
        """.format(stats['total'] if stats else 0), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">Buy Signals</div>
            <div class="stat-value stat-value-green">{}</div>
        </div>
        """.format(stats['buy'] if stats else 0), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">Sell Signals</div>
            <div class="stat-value stat-value-red">{}</div>
        </div>
        """.format(stats['sell'] if stats else 0), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">Avg Confidence</div>
            <div class="stat-value">{:.0f}%</div>
        </div>
        """.format(stats['confidence'] if stats else 0), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Asset tabs
    tabs = st.tabs(DEFAULT_ASSETS)
    
    for idx, symbol in enumerate(DEFAULT_ASSETS):
        with tabs[idx]:
            # Get signal data
            signal_data = get_latest_signal(symbol)
            
            # Two columns: signal panel + chart
            col_left, col_right = st.columns([1, 1.5])
            
            with col_left:
                render_signal_panel(symbol, signal_data)
            
            with col_right:
                st.markdown(f"### 📈 {symbol} Price Chart (4H)")
                df = get_price_history(symbol, hours=4)
                
                if not df.empty:
                    chart = create_price_chart(df, signal_data)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("No price history available. Make sure backend is running.")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #555; font-size: 12px; padding: 20px;">
        <p>TITAN Trading Systems • Oracle Prime v1.0</p>
        <p>Powered by AI • Real-time market analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(5)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("❌ Missing environment variables. Check .env or Streamlit secrets.")
        st.stop()
    
    main()
