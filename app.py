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
        
        /* === PALETTE COLORI === 
           Sfondo: #121417 (Antracite Scuro)
           Card: #1B1E23 (Antracite Medio)
           Accent: #69F0AE (Verde Riposante/Zen)
           Text: #E0E0E0 (Bianco/Grigio)
        */

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
        
        /* === PULSANTI STREAMLIT (Antracite Style) === */
        .stButton>button {
            background: #1B1E23;
            color: #E0E0E0;
            border: 1px solid #2D333B;
            border-radius: 8px;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            border-color: #69F0AE;
            color: #69F0AE;
            background: #252930;
            box-shadow: 0 0 10px rgba(105, 240, 174, 0.1);
        }

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
        }
        
        .titan-header::after {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, #69F0AE, transparent);
            opacity: 0.5;
        }

        .titan-branding { display: flex; flex-direction: column; }
        
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
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px; height: 8px; background: #69F0AE;
            border-radius: 50%; box-shadow: 0 0 8px #69F0AE;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

        /* === TABS STYLING (Moderno & Antracite) === */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #16181C;
            padding: 8px;
            border-radius: 12px;
            border: 1px solid #2D333B;
            gap: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #888;
            border-radius: 8px;
            border: none;
            padding: 8px 20px;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: #252930;
            color: #69F0AE !important; /* Verde selezionato */
            border: 1px solid #2D333B;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        /* === SIGNAL CARDS (Il cuore del design) === */
        .signal-card {
            background: #1B1E23; /* Antracite */
            border: 1px solid #2D333B;
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }
        
        .signal-card:hover { transform: translateY(-2px); border-color: #3E4650; }

        /* Colori Segnali */
        .signal-card-buy { border-left: 4px solid #69F0AE; }
        .signal-card-sell { border-left: 4px solid #FF5252; }
        .signal-card-wait { border-left: 4px solid #555; opacity: 0.8; }
        .signal-card-closed { border-left: 4px solid #555; background: #141619; }

        /* Typography interna */
        .signal-type {
            font-size: 2rem; font-weight: 800; letter-spacing: 1px;
            font-family: 'Rajdhani', sans-serif;
        }
        .signal-symbol {
            font-size: 1rem; color: #69F0AE; /* Nome strumento Verde */
            font-weight: 600; letter-spacing: 1px; margin-bottom: 1rem;
        }
        .price-display {
            font-size: 3.2rem; font-weight: 700; color: #FFF;
            font-family: 'Rajdhani', sans-serif;
            text-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .price-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }

        /* === STATS GRID === */
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 20px; }
        .stat-box {
            background: #23272E;
            border: 1px solid #2D333B;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; margin-bottom: 5px; }
        .stat-value { font-size: 1.2rem; font-weight: 700; color: #FFF; font-family: 'Rajdhani', sans-serif; }
        
        .val-buy { color: #69F0AE; }
        .val-sell { color: #FF5252; }
        .val-blue { color: #40C4FF; }

        /* === CHART CONTAINER === */
        .chart-container {
            background: #1B1E23;
            border: 1px solid #2D333B;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .chart-title { font-size: 1.3rem; font-family: 'Rajdhani', sans-serif; font-weight: 700; color: #69F0AE; }
        .chart-badge { background: #23272E; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; color: #888; border: 1px solid #2D333B; }

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
    
    # Linea Prezzo (Sottile ed elegante)
    fig.add_trace(go.Scatter(
        x=df['created_at'], y=df['price'], mode='lines', name='Price',
        line=dict(color='#69F0AE', width=2), # Verde Riposante
        fill='tozeroy', fillcolor='rgba(105, 240, 174, 0.05)'
    ))
    
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        if entry > 0: fig.add_hline(y=entry, line_dash="dash", line_color="#E0E0E0", annotation_text="ENTRY")
        if sl > 0: fig.add_hline(y=sl, line_dash="dot", line_color="#FF5252", annotation_text="SL")
        if tp > 0: fig.add_hline(y=tp, line_dash="dot", line_color="#69F0AE", annotation_text="TP")
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=450, margin=dict(l=10, r=10, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor='#2D333B', title="", showline=False),
        yaxis=dict(showgrid=True, gridcolor='#2D333B', title="", showline=False),
        hovermode='x unified', font=dict(color='#E0E0E0', family='Rajdhani')
    )
    return fig

def render_signal_panel(symbol, signal_data):
    # --- LOGICA MERCATO CHIUSO (10 Minuti) ---
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

    # --- RENDER CARD: MARKET CLOSED ---
    if not signal_data or is_stale:
        last_price = signal_data.get('current_price', 0) if signal_data else 0
        st.markdown(f"""
        <div class="signal-card signal-card-closed">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="signal-symbol">{symbol}</div>
                    <div style="font-size: 1.5rem; color: #888; font-weight:700;">MARKET CLOSED</div>
                </div>
                <div style="font-size: 2.5rem;">💤</div>
            </div>
            <div style="margin-top:20px; border-top:1px solid #333; padding-top:15px;">
                <div class="price-label">LAST KNOWN PRICE</div>
                <div style="font-family:'Rajdhani'; font-size:2rem; color:#666;">${last_price:,.2f}</div>
                <div style="color:#555; font-size:0.8rem; margin-top:5px;">Last update: {time_str}</div>
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
    details = signal_data.get('details', 'Analysis')

    # Colori dinamici
    if rec == 'BUY': card_cls, icon, col = "signal-card-buy", "🟢", "#69F0AE"
    elif rec == 'SELL': card_cls, icon, col = "signal-card-sell", "🔴", "#FF5252"
    else: card_cls, icon, col = "signal-card-wait", "⚪", "#888"

    st.markdown(f"""
    <div class="signal-card {card_cls}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="signal-symbol">{symbol}</div>
                <div class="signal-type" style="color:{col}">{rec}</div>
            </div>
            <div style="font-size:2.5rem;">{icon}</div>
        </div>
        
        <div style="margin: 1.5rem 0;">
            <div class="price-label">CURRENT PRICE</div>
            <div class="price-display">${price:,.2f}</div>
        </div>

        <div style="background:#23272E; border-radius:8px; padding:10px; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="color:#888; font-size:0.8rem;">CONFIDENCE</span>
                <span style="color:{col}; font-weight:700;">{conf}%</span>
            </div>
            <div style="background:#333; height:6px; border-radius:3px;">
                <div style="background:{col}; width:{conf}%; height:100%; border-radius:3px;"></div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-box"><div class="stat-label">ENTRY</div><div class="stat-value val-blue">${entry:,.2f}</div></div>
            <div class="stat-box"><div class="stat-label">STOP LOSS</div><div class="stat-value val-sell">${sl:,.2f}</div></div>
            <div class="stat-box"><div class="stat-label">TAKE PROFIT</div><div class="stat-value val-buy">${tp:,.2f}</div></div>
        </div>
        
        <div style="margin-top:15px; color:#666; font-size:0.8rem; text-align:center;">
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
                <div class="titan-subtitle">Enterprise Trading Intelligence</div>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span> SYSTEM ONLINE
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
    
    with c1: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">TOTAL SIGNALS</div><div class="metric-val-top val-blue">{vals["total"]}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">BUY</div><div class="metric-val-top val-buy">{vals["buy"]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">SELL</div><div class="metric-val-top val-sell">{vals["sell"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card-top"><div class="metric-label-top">AVG CONFIDENCE</div><div class="metric-val-top" style="color:#E0E0E0;">{vals["conf"]:.0f}%</div></div>', unsafe_allow_html=True)

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
                        <div class="chart-title">PRICE ACTION</div>
                        <div class="chart-badge">{symbol}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                df = get_price_history(symbol, hours=4)
                if not df.empty:
                    chart = create_price_chart(df, signal_data)
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Waiting for data feed...")
                
                st.markdown("</div>", unsafe_allow_html=True)

    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
