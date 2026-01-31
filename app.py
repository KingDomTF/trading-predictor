import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TITAN Oracle Prime",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caricamento Variabili d'Ambiente (Ibrido: Locale + Cloud)
load_dotenv()

try:
    if 'SUPABASE_URL' in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    else:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
except:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CSS STYLE (Esattamente quello richiesto: Cyberpunk Neon)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main theme - Background Gradiente Viola/Blu scuro */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%) !important;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #00d9ff33;
    }
    
    /* Metric cards (Blu/Viola con bordo Ciano) */
    .metric-card {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #00d9ff;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.15);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* Signal cards - BUY (Verde) */
    .signal-buy {
        background: linear-gradient(135deg, #1e3c28 0%, #2d5f3e 100%);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        text-align: center;
    }
    
    /* Signal cards - SELL (Rosso) */
    .signal-sell {
        background: linear-gradient(135deg, #3c1e1e 0%, #5f2d2d 100%);
        border: 2px solid #ff0044;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(255, 0, 68, 0.3);
        text-align: center;
    }
    
    /* Signal cards - WAIT (Grigio) */
    .signal-wait {
        background: linear-gradient(135deg, #2a2a3c 0%, #3a3a4e 100%);
        border: 2px solid #888888;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        opacity: 0.8;
        text-align: center;
    }
    
    /* Typography */
    .price-big {
        font-size: 42px;
        font-weight: bold;
        color: #00d9ff;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.6);
        line-height: 1;
    }
    
    .signal-title {
        font-size: 28px;
        font-weight: 900;
        margin-bottom: 5px;
        letter-spacing: 1px;
        color: #fff;
    }
    
    .stat-label {
        font-size: 13px;
        color: #aaaaaa;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    
    /* Status indicators */
    .status-active {
        display: inline-block;
        width: 10px;
        height: 10px;
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
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4);
    }
    
    /* Pulizia */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #00d9ff !important; text-shadow: 0 0 15px rgba(0, 217, 255, 0.3); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOGICA DATABASE (La mia robusta)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_db():
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    try: return create_client(SUPABASE_URL, SUPABASE_KEY)
    except: return None

supabase = init_db()

def get_latest_data(symbol):
    """Recupera Segnale e Prezzo Recente"""
    if not supabase: return None, pd.DataFrame()
    
    # 1. Segnale
    signal = None
    try:
        res = supabase.table("ai_oracle").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        if res.data: signal = res.data[0]
    except: pass

    # 2. Storico Prezzi (ultime 4 ore)
    history = pd.DataFrame()
    try:
        cutoff = (datetime.now() - timedelta(hours=4)).isoformat()
        res = supabase.table("mt4_feed").select("created_at, price").eq("symbol", symbol).gte("created_at", cutoff).order("created_at", desc=False).execute()
        if res.data:
            history = pd.DataFrame(res.data)
            history['created_at'] = pd.to_datetime(history['created_at'])
    except: pass
    
    return signal, history

# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTERFACCIA UTENTE (Layout Mio + Stile Tuo)
# ═══════════════════════════════════════════════════════════════════════════════

# --- HEADER STILE TITAN ---
st.markdown("""
<div style="text-align: center; padding-bottom: 20px;">
    <h1 style="color: #00d9ff; font-size: 48px; margin: 0; display: inline-block;">
        🏛️ TITAN ORACLE PRIME
    </h1>
    <div style="margin-top: 10px;">
        <span class="status-active"></span> 
        <span style="color: #00ff88; font-weight: bold; margin-left: 8px;">SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ CONTROL PANEL")
    symbol = st.radio("ASSET SELECTION:", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"], index=3)
    
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
    if st.button("🚀 Force Refresh", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("TITAN Trading Systems v1.0")

# Fetch Dati
signal_data, history_df = get_latest_data(symbol)

# Determinare Prezzo Attuale
current_price = 0.0
if not history_df.empty: current_price = history_df['price'].iloc[-1]
elif signal_data: current_price = signal_data.get('current_price', 0.0)

if not supabase:
    st.error("❌ Database Connection Failed. Check .env")
    st.stop()

# Estrazione Dati
rec = signal_data.get('recommendation', 'WAIT') if signal_data else 'WAIT'
conf = signal_data.get('confidence_score', 0) if signal_data else 0
details = signal_data.get('details', 'Scanning...') if signal_data else 'Initializing...'

# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS ROW (Stile: Neon Cards)
# ═══════════════════════════════════════════════════════════════════════════════

col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    # Selezione Classe CSS in base al segnale
    card_class = "signal-wait"
    icon = "⚪"
    if rec == "BUY": 
        card_class = "signal-buy"
        icon = "🟢"
    elif rec == "SELL": 
        card_class = "signal-sell"
        icon = "🔴"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div class="stat-label">AI SIGNAL</div>
        <div class="signal-title">{icon} {rec}</div>
        <div style="color: #ccc; font-size: 14px;">{details.split('|')[0] if '|' in details else details}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Card Confidenza (Stile Blu/Viola)
    bar_color = "#888"
    if rec == "BUY": bar_color = "#00ff88"
    elif rec == "SELL": bar_color = "#ff0044"

    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-label">CONFIDENCE</div>
        <div class="price-big" style="font-size: 36px; color: {bar_color}; text-shadow: 0 0 15px {bar_color};">{conf}%</div>
        <div style="width: 80%; background: #0f1525; height: 6px; border-radius: 3px; margin-top: 10px;">
            <div style="width: {conf}%; background: {bar_color}; height: 100%; border-radius: 3px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Card Prezzo (Stile Blu/Viola)
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-label">LIVE PRICE</div>
        <div class="price-big">${current_price:,.2f}</div>
        <div style="color: #00d9ff; font-size: 12px; margin-top: 5px;">● REAL-TIME FEED</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LEVELS ROW (Stile: Custom Cards)
# ═══════════════════════════════════════════════════════════════════════════════

if rec in ["BUY", "SELL"] and signal_data:
    st.markdown("<br>", unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    
    # Helper per creare le card dei livelli con lo stile Cyberpunk
    def level_card(label, value, color):
        return f"""
        <div class="metric-card" style="border-color: {color}; box-shadow: 0 0 10px {color}44;">
            <div class="stat-label">{label}</div>
            <div class="stat-value" style="color: {color}; font-size: 28px;">${value:,.2f}</div>
        </div>
        """
        
    with l1:
        st.markdown(level_card("ENTRY PRICE", entry, "#ffffff"), unsafe_allow_html=True)
    with l2:
        st.markdown(level_card("STOP LOSS", sl, "#ff0044"), unsafe_allow_html=True)
    with l3:
        st.markdown(level_card("TAKE PROFIT", tp, "#00ff88"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CHART (Stile Neon Cyan)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📈 Market Analysis")

if not history_df.empty:
    fig = go.Figure()
    
    # Linea Prezzo (Ciano Neon)
    fig.add_trace(go.Scatter(
        x=history_df['created_at'], y=history_df['price'],
        mode='lines', name='Price',
        line=dict(color='#00d9ff', width=2),
        fill='tozeroy', fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    # Livelli
    if rec in ['BUY', 'SELL']:
        # Entry
        fig.add_hline(y=entry, line_dash="dash", line_color="white", annotation_text="ENTRY")
        # SL
        fig.add_hline(y=sl, line_dash="dot", line_color="#ff0044", annotation_text="SL")
        # TP
        fig.add_hline(y=tp, line_dash="dot", line_color="#00ff88", annotation_text="TP")

    # Layout Plotly Dark/Transparent
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Price"),
        hovermode="x unified",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Waiting for Price Feed...")

# Auto-Refresh Loop
if auto_refresh:
    time.sleep(2)
    st.rerun()
