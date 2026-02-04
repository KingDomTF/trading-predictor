"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 DASHBOARD - PREMIUM FRONTEND INTERFACE (CORRECTED)
═══════════════════════════════════════════════════════════════════════════════
Professional Real-Time Trading Terminal powered by TITAN V90 Backend
Visualizes market data, AI signals, and performance metrics via Streamlit
Theme: Neo-Financial Brutalism

CORREZIONI APPLICATE:
- Fix gestione errori nel database
- Aggiunta logica per rilevare accumulo/distribuzione istituzionale
- Migliorato calcolo volumi anomali
- Fix timezone handling
- Aggiunto indicatore Order Flow istituzionale
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# Dependency check & Import
try:
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.error("❌ Missing libraries. Run: pip install supabase python-dotenv plotly pandas numpy")
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
    
    # Institutional Detection Thresholds
    VOLUME_MULTIPLIER = 2.0  # Volume deve essere 2x la media
    PRICE_DEVIATION_THRESHOLD = 0.003  # 0.3% movimento minimo
    ACCUMULATION_BARS = 10  # Numero di candele per rilevare accumulo

# Initialize Page
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    """Initialize Supabase client with error handling"""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            st.error("❌ SUPABASE_URL e SUPABASE_KEY devono essere configurati in .env")
            return None
            
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Errore connessione database: {str(e)}")
        return None

supabase = init_supabase()

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL STYLING (CSS ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Injects the Style"""
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

/* === INSTITUTIONAL FLOW INDICATOR === */
.inst-flow-card {
    background: rgba(19, 23, 29, 0.8);
    border: 2px solid #2A3340;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}
.inst-badge {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.inst-accumulation { background: rgba(0, 255, 240, 0.15); color: #00FFF0; border: 1px solid rgba(0, 255, 240, 0.3); }
.inst-distribution { background: rgba(255, 0, 110, 0.15); color: #FF006E; border: 1px solid rgba(255, 0, 110, 0.3); }
.inst-neutral { background: rgba(90, 102, 120, 0.15); color: #5A6678; border: 1px solid rgba(90, 102, 120, 0.3); }

/* === CHART CONTAINER === */
.chart-container {
    background: #13171D;
    border: 2px solid #2A3340;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}
.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.chart-title { 
    font-size: 1.2rem; 
    font-weight: 800; 
    color: #E8ECF1; 
    font-family: 'Syne', sans-serif; 
}
.chart-badge { 
    background: rgba(42, 51, 64, 0.5); 
    padding: 0.4rem 1rem; 
    border-radius: 20px; 
    font-size: 0.7rem; 
    color: #8892A0; 
    font-family: 'Space Mono', monospace; 
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS (CORRECTED)
# ═══════════════════════════════════════════════════════════════════════════════

def get_latest_signal(symbol: str) -> Optional[Dict]:
    """Fetch latest signal for a symbol with proper error handling"""
    if not supabase:
        return None
    
    try:
        response = supabase.table('trading_signals') \
            .select('*') \
            .eq('symbol', symbol) \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        st.error(f"⚠️ Errore nel recupero segnale per {symbol}: {str(e)}")
        return None

def get_price_history(symbol: str, hours: int = 4) -> pd.DataFrame:
    """
    Fetch price history with volume data for institutional detection
    NOTA: Questa funzione assume che tu abbia una tabella 'price_history' 
    con colonne: symbol, created_at, price, volume
    """
    if not supabase:
        return pd.DataFrame()
    
    try:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        response = supabase.table('price_history') \
            .select('created_at, price, volume') \
            .eq('symbol', symbol) \
            .gte('created_at', cutoff) \
            .order('created_at', desc=False) \
            .execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df
        return pd.DataFrame()
    except Exception as e:
        # Se la tabella non esiste, genera dati mock per testing
        st.warning(f"⚠️ Tabella price_history non trovata. Usando dati mock. Errore: {str(e)}")
        return generate_mock_price_data(symbol, hours)

def generate_mock_price_data(symbol: str, hours: int) -> pd.DataFrame:
    """Generate mock price data for testing when DB is not available"""
    base_prices = {
        'XAUUSD': 2650.0,
        'BTCUSD': 95000.0,
        'US500': 5800.0,
        'ETHUSD': 3400.0,
        'XAGUSD': 31.5
    }
    
    base_price = base_prices.get(symbol, 100.0)
    n_points = hours * 12  # 5 min intervals
    
    dates = pd.date_range(end=datetime.utcnow(), periods=n_points, freq='5min')
    prices = base_price + np.cumsum(np.random.randn(n_points) * base_price * 0.001)
    volumes = np.random.randint(1000, 10000, n_points)
    
    return pd.DataFrame({
        'created_at': dates,
        'price': prices,
        'volume': volumes
    })

def get_24h_stats() -> Optional[Dict]:
    """Get 24h statistics with proper error handling"""
    if not supabase:
        return None
    
    try:
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        response = supabase.table('trading_signals') \
            .select('recommendation, confidence_score') \
            .gte('created_at', cutoff) \
            .execute()
        
        if not response.data:
            return {'total': 0, 'buy': 0, 'sell': 0, 'confidence': 0}
        
        df = pd.DataFrame(response.data)
        total = len(df)
        buy = len(df[df['recommendation'] == 'BUY'])
        sell = len(df[df['recommendation'] == 'SELL'])
        avg_conf = df['confidence_score'].mean() if 'confidence_score' in df else 0
        
        return {'total': total, 'buy': buy, 'sell': sell, 'confidence': avg_conf}
    except Exception as e:
        st.warning(f"⚠️ Errore nel calcolo statistiche 24h: {str(e)}")
        return {'total': 0, 'buy': 0, 'sell': 0, 'confidence': 0}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL FLOW DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_institutional_flow(df: pd.DataFrame) -> Dict:
    """
    Rileva accumulo/distribuzione istituzionale analizzando:
    1. Volume anomalo (smart money footprint)
    2. Price action (support/resistance tests)
    3. Order flow imbalance
    """
    if df.empty or len(df) < AppConfig.ACCUMULATION_BARS:
        return {
            'status': 'NEUTRAL',
            'confidence': 0,
            'description': 'Dati insufficienti'
        }
    
    try:
        # Calcola medie mobili
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['price_ma'] = df['price'].rolling(window=20).mean()
        
        # Analizza gli ultimi N bar
        recent = df.tail(AppConfig.ACCUMULATION_BARS)
        
        # 1. VOLUME ANALYSIS - Cerca volumi anomali (smart money)
        avg_volume = df['volume'].mean()
        recent_avg_volume = recent['volume'].mean()
        volume_ratio = recent_avg_volume / avg_volume if avg_volume > 0 else 1
        
        # 2. PRICE ACTION - Analizza range compression/expansion
        price_volatility = recent['price'].std()
        overall_volatility = df['price'].std()
        volatility_ratio = price_volatility / overall_volatility if overall_volatility > 0 else 1
        
        # 3. ORDER FLOW - Up bars vs Down bars con volume
        up_bars = recent[recent['price'].diff() > 0]
        down_bars = recent[recent['price'].diff() < 0]
        
        up_volume = up_bars['volume'].sum() if len(up_bars) > 0 else 0
        down_volume = down_bars['volume'].sum() if len(down_bars) > 0 else 0
        total_volume = up_volume + down_volume
        
        # Delta volume (order flow imbalance)
        delta_ratio = (up_volume - down_volume) / total_volume if total_volume > 0 else 0
        
        # 4. DECISIONE FINALE
        confidence = 0
        status = 'NEUTRAL'
        description = ''
        
        # ACCUMULO: Alto volume + prezzi stabili/up + delta positivo
        if (volume_ratio > AppConfig.VOLUME_MULTIPLIER and 
            delta_ratio > 0.2 and 
            volatility_ratio < 1.2):
            status = 'ACCUMULATION'
            confidence = min(85, int(50 + (volume_ratio * 10) + (delta_ratio * 50)))
            description = f'Istituzionali in accumulo: Volume {volume_ratio:.1f}x, Delta +{delta_ratio*100:.1f}%'
        
        # DISTRIBUZIONE: Alto volume + prezzi stabili/down + delta negativo
        elif (volume_ratio > AppConfig.VOLUME_MULTIPLIER and 
              delta_ratio < -0.2 and 
              volatility_ratio < 1.2):
            status = 'DISTRIBUTION'
            confidence = min(85, int(50 + (volume_ratio * 10) + (abs(delta_ratio) * 50)))
            description = f'Istituzionali in distribuzione: Volume {volume_ratio:.1f}x, Delta {delta_ratio*100:.1f}%'
        
        # NEUTRAL
        else:
            status = 'NEUTRAL'
            confidence = 30
            description = 'Nessun pattern istituzionale chiaro rilevato'
        
        return {
            'status': status,
            'confidence': confidence,
            'description': description,
            'volume_ratio': volume_ratio,
            'delta_ratio': delta_ratio
        }
    
    except Exception as e:
        return {
            'status': 'ERROR',
            'confidence': 0,
            'description': f'Errore analisi: {str(e)}'
        }

# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df: pd.DataFrame, signal_data: Optional[Dict], inst_flow: Dict):
    """Create enhanced price chart with institutional flow indicators"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Price Line con gradient fill
    fig.add_trace(go.Scatter(
        x=df['created_at'], 
        y=df['price'], 
        mode='lines', 
        name='Price',
        line=dict(color='#00FFF0', width=2), 
        fill='tozeroy', 
        fillcolor='rgba(0, 255, 240, 0.05)',
        hovertemplate='%{y:,.2f}<extra></extra>'
    ))
    
    # Volume bars (se disponibili)
    if 'volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df['created_at'],
            y=df['volume'],
            name='Volume',
            yaxis='y2',
            marker=dict(
                color=df['volume'],
                colorscale=[[0, '#2A3340'], [1, '#00FFF0']],
                showscale=False
            ),
            opacity=0.3,
            hovertemplate='Vol: %{y:,.0f}<extra></extra>'
        ))
    
    # Signal levels
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        if entry > 0:
            fig.add_hline(
                y=entry, 
                line_dash="dash", 
                line_color="#5B9FFF", 
                annotation_text="ENTRY",
                annotation_position="right"
            )
        if sl > 0:
            fig.add_hline(
                y=sl, 
                line_dash="dot", 
                line_color="#FF006E", 
                annotation_text="SL",
                annotation_position="right"
            )
        if tp > 0:
            fig.add_hline(
                y=tp, 
                line_dash="dot", 
                line_color="#00FFF0", 
                annotation_text="TP",
                annotation_position="right"
            )
    
    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=10, r=10, t=30, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='#2A3340',
            title="",
            showline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#2A3340',
            title="",
            showline=False,
            side='right'
        ),
        yaxis2=dict(
            showgrid=False,
            overlaying='y',
            side='left',
            showticklabels=False
        ),
        hovermode='x unified',
        font=dict(color='#E8ECF1', family='JetBrains Mono'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def render_institutional_flow_badge(inst_flow: Dict):
    """Render institutional flow indicator badge"""
    status = inst_flow.get('status', 'NEUTRAL')
    confidence = inst_flow.get('confidence', 0)
    description = inst_flow.get('description', '')
    
    if status == 'ACCUMULATION':
        badge_class = 'inst-accumulation'
        icon = '📈'
        title = 'ACCUMULO ISTITUZIONALE'
    elif status == 'DISTRIBUTION':
        badge_class = 'inst-distribution'
        icon = '📉'
        title = 'DISTRIBUZIONE ISTITUZIONALE'
    else:
        badge_class = 'inst-neutral'
        icon = '➖'
        title = 'FLUSSO NEUTRALE'
    
    st.markdown(f"""
<div class="inst-flow-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <div>
                <div class="inst-badge {badge_class}">{title}</div>
                <div style="font-size: 0.75rem; color: #5A6678; margin-top: 0.3rem; font-family: 'Space Mono', monospace;">
                    Confidence: {confidence}%
                </div>
            </div>
        </div>
    </div>
    <div style="font-size: 0.8rem; color: #8892A0; line-height: 1.5; font-family: 'JetBrains Mono', monospace;">
        {description}
    </div>
</div>
""", unsafe_allow_html=True)

def render_signal_panel(symbol: str, signal_data: Optional[Dict]):
    """Render signal panel with improved time handling"""
    # Market closed check (10 min)
    created_at = signal_data.get('created_at', '') if signal_data else ''
    is_stale = False
    time_str = "Waiting..."
    
    if created_at:
        try:
            # Fix timezone handling
            if isinstance(created_at, str):
                created_at = created_at.replace('Z', '+00:00')
                signal_time = datetime.fromisoformat(created_at)
            else:
                signal_time = created_at
            
            # Calcola differenza tempo con timezone awareness
            now = datetime.now(signal_time.tzinfo) if signal_time.tzinfo else datetime.utcnow()
            time_diff = (now - signal_time).total_seconds()
            
            if time_diff > 600:  # 10 minuti
                is_stale = True
            
            if time_diff < 60:
                time_str = "Now"
            elif time_diff < 3600:
                time_str = f"{int(time_diff/60)}m ago"
            else:
                time_str = f"{int(time_diff/3600)}h ago"
        except Exception as e:
            st.warning(f"⚠️ Errore parsing tempo: {str(e)}")
            time_str = "Unknown"
    
    # CARD: MARKET CLOSED
    if not signal_data or is_stale:
        last_price = signal_data.get('current_price', 0) if signal_data else 0
        st.markdown(f"""
<div class="signal-card signal-card-closed">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="signal-symbol">{symbol}</div>
            <div style="font-size: 2rem; color: #5A6678; font-weight:800; font-family: 'Syne', sans-serif;">
                MARKET CLOSED
            </div>
        </div>
        <div style="font-size: 3rem; opacity: 0.3;">💤</div>
    </div>
    <div style="margin-top:1.5rem; border-top:1px solid #2A3340; padding-top:1.5rem;">
        <div class="price-label">LAST KNOWN PRICE</div>
        <div style="font-family:'Syne', sans-serif; font-size:2.5rem; font-weight: 800; color:#3A4350;">
            ${last_price:,.2f}
        </div>
        <div style="color:#5A6678; font-size:0.75rem; margin-top:0.75rem; font-family: 'Space Mono', monospace;">
            Last update: {time_str}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        return
    
    # CARD: ACTIVE
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0)
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    details = signal_data.get('details', 'Analysis')
    
    if rec == 'BUY':
        card_cls, icon, col = "signal-card-buy", "▲", "#00FFF0"
    elif rec == 'SELL':
        card_cls, icon, col = "signal-card-sell", "▼", "#FF006E"
    else:
        card_cls, icon, col = "signal-card-wait", "●", "#5A6678"
    
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
        <div class="stat-box">
            <div class="stat-label">ENTRY</div>
            <div class="stat-value val-blue">${entry:,.2f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">STOP LOSS</div>
            <div class="stat-value val-sell">${sl:,.2f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">TARGET</div>
            <div class="stat-value val-buy">${tp:,.2f}</div>
        </div>
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
        st.error("❌ Database connection error. Check .env file for SUPABASE_URL and SUPABASE_KEY")
        st.stop()
    
    # HEADER
    st.markdown("""
<div class="titan-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:20px;">
        <div class="titan-branding">
            <div class="titan-title">TITAN ORACLE</div>
            <div class="titan-subtitle">Neo-Financial Intelligence + Institutional Flow</div>
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
    
    with c1:
        st.markdown(f"""
<div class="metric-card-top">
    <div class="metric-label-top">Total Signals</div>
    <div class="metric-val-top val-blue">{vals["total"]}</div>
</div>
""", unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
<div class="metric-card-top">
    <div class="metric-label-top">Buy Signals</div>
    <div class="metric-val-top val-buy">{vals["buy"]}</div>
</div>
""", unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
<div class="metric-card-top">
    <div class="metric-label-top">Sell Signals</div>
    <div class="metric-val-top val-sell">{vals["sell"]}</div>
</div>
""", unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
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
            col_left, col_right = st.columns([1, 1.6])
            
            # Fetch data
            signal_data = get_latest_signal(symbol)
            df = get_price_history(symbol, hours=4)
            
            # Detect institutional flow
            inst_flow = detect_institutional_flow(df)
            
            with col_left:
                render_signal_panel(symbol, signal_data)
                render_institutional_flow_badge(inst_flow)
            
            with col_right:
                st.markdown(f"""
<div class="chart-container">
    <div class="chart-header">
        <div class="chart-title">Price Action & Volume</div>
        <div class="chart-badge">{symbol}</div>
    </div>
""", unsafe_allow_html=True)
                
                if not df.empty:
                    chart = create_price_chart(df, signal_data, inst_flow)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("📡 Awaiting market data...")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
