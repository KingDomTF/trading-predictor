import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE & SETUP (TITAN V38 PRO FRONTEND)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TITAN V38 PRO Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caricamento Credenziali
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
# 2. CSS "BLACK & GREEN" PROFESSIONAL THEME (Il tuo stile preferito)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* --- SFONDO E TESTI GLOBALI --- */
    .stApp {
        background-color: #000000 !important; /* Nero Assoluto */
        color: #e0e0e0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #1a1a1a;
    }
    
    /* --- TITOLI --- */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* --- CARD CONTAINER --- */
    .metric-container {
        background-color: #0e0e0e;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        transition: transform 0.2s, border-color 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-container:hover {
        border-color: #333;
        transform: translateY(-2px);
    }

    /* --- TYPOGRAPHY INTERNA --- */
    .metric-label {
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #fff;
        line-height: 1.1;
    }
    
    .metric-sub {
        font-size: 13px;
        color: #555;
        margin-top: 8px;
        font-family: 'Consolas', monospace;
    }

    /* --- COLORI SEGNALI V38 PRO --- */
    .buy-signal { 
        color: #00FF9D !important; 
        text-shadow: 0 0 15px rgba(0, 255, 157, 0.2);
    }
    
    .sell-signal { 
        color: #FF3B30 !important; 
        text-shadow: 0 0 15px rgba(255, 59, 48, 0.2);
    }
    
    .neutral-signal { 
        color: #888 !important; 
    }

    /* --- BORDATURE LIVELLI --- */
    .entry-box { border-left: 3px solid #2979FF !important; }
    .stop-box { border-left: 3px solid #FF3B30 !important; }
    .target-box { border-left: 3px solid #00FF9D !important; }
    .size-box { border-left: 3px solid #FFD700 !important; } /* Oro per Size */

    .level-value { font-family: 'Consolas', monospace; font-size: 22px; font-weight: bold; }
    .level-entry { color: #2979FF; }
    .level-sl { color: #FF3B30; }
    .level-tp { color: #00FF9D; }
    .level-size { color: #FFD700; }

    /* --- BADGES --- */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
        margin-right: 5px;
    }
    .badge-regime { background: #222; color: #aaa; border: 1px solid #444; }
    .badge-zscore { background: #111; color: #2979FF; border: 1px solid #2979FF; }

    /* --- PULSANTI --- */
    .stButton>button {
        background-color: #1a1a1a;
        color: #fff;
        border: 1px solid #333;
    }
    .stButton>button:hover {
        border-color: #00FF9D;
        color: #00FF9D;
    }
    
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOGICA RECUPERO DATI (Adattata a V38 PRO)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_db():
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    try: return create_client(SUPABASE_URL, SUPABASE_KEY)
    except: return None

supabase = init_db()

def get_v38_data(symbol):
    """Recupera i dati specifici del motore V38 Pro"""
    if not supabase: return None, pd.DataFrame()
    
    # 1. Oracle Data (Segnali & Statistiche)
    signal = None
    try:
        res = supabase.table("ai_oracle").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        if res.data: signal = res.data[0]
    except: pass

    # 2. Feed Data (Grafico)
    history = pd.DataFrame()
    try:
        cutoff = (datetime.now() - timedelta(hours=6)).isoformat()
        res = supabase.table("mt4_feed").select("created_at, price").eq("symbol", symbol).gte("created_at", cutoff).order("created_at", desc=False).execute()
        if res.data:
            history = pd.DataFrame(res.data)
            history['created_at'] = pd.to_datetime(history['created_at'])
    except: pass
    
    return signal, history

# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTERFACCIA V38 PRO (UI)
# ═══════════════════════════════════════════════════════════════════════════════

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚡ TITAN V38")
    st.caption("LITE EDITION • NUMPY CORE")
    
    st.markdown("### 📡 Asset Feed")
    symbol = st.radio("MARKET", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"], index=3, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ System")
    auto_refresh = st.checkbox("Live Refresh (1s)", value=True)
    if st.button("Force Sync", use_container_width=True): st.rerun()

# --- HEADER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(f"# {symbol} <span style='font-size:18px; color:#555;'>/ V38 PRO</span>", unsafe_allow_html=True)

# Fetch Dati
signal_data, history_df = get_v38_data(symbol)

# Prezzo Corrente
current_price = 0.0
if not history_df.empty: current_price = history_df['price'].iloc[-1]
elif signal_data: current_price = signal_data.get('current_price', 0.0)

if not supabase:
    st.error("🚨 Database Connection Failed. Check .env")
    st.stop()

# --- DATI V38 PRO ---
rec = signal_data.get('recommendation', 'NEUTRAL') if signal_data else 'NEUTRAL'
conf = signal_data.get('confidence_score', 0) if signal_data else 0
regime = signal_data.get('market_regime', 'analyzing') if signal_data else '...'
z_score = signal_data.get('z_score', 0.0) if signal_data else 0.0
volatility = signal_data.get('volatility', 0.0) if signal_data else 0.0
pos_size = signal_data.get('position_size', 0.0) if signal_data else 0.0

# 1. RIGA KPI (Adattata a V38)
col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    # Colore dinamico
    sig_class = "neutral-signal"
    if rec == "BUY": sig_class = "buy-signal"
    elif rec == "SELL": sig_class = "sell-signal"
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">V38 SIGNAL</div>
        <div class="metric-value {sig_class}">{rec}</div>
        <div class="metric-sub">
            <span class="status-badge badge-regime">{regime.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Colore Z-Score (Rosso se estremo negativo, Verde se estremo positivo, Grigio se neutro)
    z_color = "#888"
    if abs(z_score) > 2.0: z_color = "#fff"
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">STATISTICS</div>
        <div class="metric-value" style="color: {z_color};">{conf}%</div>
        <div class="metric-sub">
            Z-SCORE: <span style="color:#2979FF;">{z_score:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">MARKET DATA</div>
        <div class="metric-value">${current_price:,.2f}</div>
        <div class="metric-sub">
            VOL: {volatility:.5f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. RIGA LIVELLI OPERATIVI (Se c'è segnale attivo)
if rec in ["BUY", "SELL"] and signal_data:
    st.markdown("<br>", unsafe_allow_html=True)
    
    l1, l2, l3, l4 = st.columns(4) # 4 Colonne per includere Position Size
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    
    with l1:
        st.markdown(f"""<div class="metric-container entry-box"><div class="metric-label">ENTRY</div><div class="level-value level-entry">{entry:.4f}</div></div>""", unsafe_allow_html=True)
    with l2:
        st.markdown(f"""<div class="metric-container stop-box"><div class="metric-label">STOP LOSS</div><div class="level-value level-sl">{sl:.4f}</div></div>""", unsafe_allow_html=True)
    with l3:
        st.markdown(f"""<div class="metric-container target-box"><div class="metric-label">TAKE PROFIT</div><div class="level-value level-tp">{tp:.4f}</div></div>""", unsafe_allow_html=True)
    with l4:
        st.markdown(f"""<div class="metric-container size-box"><div class="metric-label">POS SIZE</div><div class="level-value level-size">{pos_size:.2f}</div></div>""", unsafe_allow_html=True)

# 3. GRAFICO BLACK CHART
st.markdown("---")
st.markdown("### 📉 V38 Analysis")

if not history_df.empty:
    fig = go.Figure()
    
    # Linea Prezzo
    fig.add_trace(go.Scatter(
        x=history_df['created_at'], y=history_df['price'],
        mode='lines', name='Price',
        line=dict(color='#2979FF', width=2),
        fill='tozeroy', fillcolor='rgba(41, 121, 255, 0.05)'
    ))
    
    # Livelli
    if rec in ['BUY', 'SELL']:
        fig.add_hline(y=entry, line_dash="dash", line_color="white", line_width=1, annotation_text="ENTRY")
        fig.add_hline(y=sl, line_dash="dot", line_color="#FF3B30", line_width=2, annotation_text="SL")
        fig.add_hline(y=tp, line_dash="dot", line_color="#00FF9D", line_width=2, annotation_text="TP")

    # Stile Terminale
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,10,1)',
        height=450,
        margin=dict(l=0, r=40, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='#222', title=None),
        yaxis=dict(showgrid=True, gridcolor='#222', title=None, side='right'),
        showlegend=False,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Waiting for V38 Engine Data...")

if auto_refresh:
    time.sleep(1)
    st.rerun()
