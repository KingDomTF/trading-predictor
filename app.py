import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE PAGINA & SETUP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TITAN V90 Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caricamento Credenziali (Ibrido: Locale + Cloud)
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
# 2. CSS "BLACK & GREEN" PROFESSIONAL THEME
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
        background-color: #0a0a0a !important; /* Nero leggermente meno profondo */
        border-right: 1px solid #1a1a1a;
    }
    
    /* --- TITOLI --- */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* --- CARD CONTAINER (Il cuore del design) --- */
    .metric-container {
        background-color: #0e0e0e;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .metric-container:hover {
        border-color: #333;
        transform: translateY(-2px);
    }

    /* --- TYPOGRAPHY INTERNA --- */
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #fff;
        line-height: 1.1;
    }
    
    .metric-sub {
        font-size: 13px;
        color: #444;
        margin-top: 5px;
    }

    /* --- SEGNALI TRADING (Colori Professionali) --- */
    /* GREEN NEON per BUY */
    .buy-signal { 
        color: #00FF9D !important; 
        text-shadow: 0 0 15px rgba(0, 255, 157, 0.2);
    }
    
    /* RED NEON per SELL */
    .sell-signal { 
        color: #FF3B30 !important; 
        text-shadow: 0 0 15px rgba(255, 59, 48, 0.2);
    }
    
    /* GRAY per WAIT */
    .wait-signal { 
        color: #555 !important; 
    }

    /* --- BORDATURE LIVELLI (Entry, SL, TP) --- */
    .entry-box { border-left: 3px solid #2979FF !important; } /* Blu Elettrico */
    .stop-box { border-left: 3px solid #FF3B30 !important; }  /* Rosso */
    .target-box { border-left: 3px solid #00FF9D !important; } /* Verde */

    .level-value { font-family: 'Consolas', 'Courier New', monospace; font-size: 24px; font-weight: bold; }
    .level-entry { color: #2979FF; }
    .level-sl { color: #FF3B30; }
    .level-tp { color: #00FF9D; }

    /* --- STATUS BADGE --- */
    .status-badge-online {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background: rgba(0, 255, 157, 0.1);
        border: 1px solid rgba(0, 255, 157, 0.3);
        color: #00FF9D;
        font-size: 12px;
        font-weight: bold;
    }

    /* --- PULSANTI --- */
    .stButton>button {
        background-color: #1a1a1a;
        color: #fff;
        border: 1px solid #333;
        border-radius: 4px;
        font-weight: 600;
    }
    .stButton>button:hover {
        border-color: #00FF9D;
        color: #00FF9D;
    }

    /* Rimuovere padding extra */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOGICA DATABASE & DATI
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
# 4. INTERFACCIA UTENTE (UI)
# ═══════════════════════════════════════════════════════════════════════════════

# --- SIDEBAR PROFESSIONALE ---
with st.sidebar:
    st.markdown("## ⚡ TITAN V90")
    st.markdown("<div style='margin-bottom: 20px;'><span class='status-badge-online'>● SYSTEM ONLINE</span></div>", unsafe_allow_html=True)
    
    st.markdown("### 📡 Asset Feed")
    symbol = st.radio("SELECT MARKET", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"], index=3, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ Operations")
    auto_refresh = st.checkbox("Live Refresh (1s)", value=True)
    if st.button("Force Sync", use_container_width=True): st.rerun()

# --- HEADER PRINCIPALE ---
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.markdown(f"# {symbol} <span style='font-size:20px; color:#555;'>/ USD</span>", unsafe_allow_html=True)
with c_head2:
    st.markdown("<div style='text-align:right; color:#444; font-family:monospace;'>ORACLE PRIME CORE</div>", unsafe_allow_html=True)

# Fetch Dati
signal_data, history_df = get_latest_data(symbol)

# Determinare Prezzo Attuale
current_price = 0.0
if not history_df.empty: current_price = history_df['price'].iloc[-1]
elif signal_data: current_price = signal_data.get('current_price', 0.0)

# --- DASHBOARD CARD (LAYOUT GRIGLIA) ---

if not supabase:
    st.error("🚨 Database Connection Failed. Check .env")
    st.stop()

# Estrazione Dati Segnale
rec = signal_data.get('recommendation', 'WAIT') if signal_data else 'WAIT'
conf = signal_data.get('confidence_score', 0) if signal_data else 0
details = signal_data.get('details', 'Scanning...') if signal_data else 'Initializing...'

# 1. RIGA KPI PRINCIPALI
col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    # Colore dinamico per il segnale
    sig_class = "wait-signal"
    if rec == "BUY": sig_class = "buy-signal"
    elif rec == "SELL": sig_class = "sell-signal"
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">AI DECISION</div>
        <div class="metric-value {sig_class}">{rec}</div>
        <div class="metric-sub">{details.split('|')[0] if '|' in details else details}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Colore barra confidenza
    bar_c = "#333"
    if rec == "BUY": bar_c = "#00FF9D"
    elif rec == "SELL": bar_c = "#FF3B30"
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">CONFIDENCE</div>
        <div class="metric-value" style="color: {bar_c};">{conf}%</div>
        <div style="width: 100%; background: #222; height: 4px; margin-top: 15px;">
            <div style="width: {conf}%; background: {bar_c}; height: 100%; box-shadow: 0 0 10px {bar_c};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">LIVE QUOTE</div>
        <div class="metric-value">${current_price:,.2f}</div>
        <div class="metric-sub" style="color: #00FF9D;">● Real-Time</div>
    </div>
    """, unsafe_allow_html=True)

# 2. RIGA LIVELLI OPERATIVI (Solo se attivo)
if rec in ["BUY", "SELL"] and signal_data:
    st.markdown("<br>", unsafe_allow_html=True)
    
    l1, l2, l3 = st.columns(3)
    entry = signal_data.get('entry_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    
    with l1:
        st.markdown(f"""<div class="metric-container entry-box"><div class="metric-label">ENTRY PRICE</div><div class="level-value level-entry">${entry:,.2f}</div></div>""", unsafe_allow_html=True)
    with l2:
        st.markdown(f"""<div class="metric-container stop-box"><div class="metric-label">STOP LOSS</div><div class="level-value level-sl">${sl:,.2f}</div></div>""", unsafe_allow_html=True)
    with l3:
        st.markdown(f"""<div class="metric-container target-box"><div class="metric-label">TAKE PROFIT</div><div class="level-value level-tp">${tp:,.2f}</div></div>""", unsafe_allow_html=True)

# 3. GRAFICO PROFESSIONALE (BLACK CHART)
st.markdown("---")
st.markdown("### 📉 Market Depth")

if not history_df.empty:
    fig = go.Figure()
    
    # Linea Prezzo (Blu Elettrico Sottile)
    fig.add_trace(go.Scatter(
        x=history_df['created_at'], y=history_df['price'],
        mode='lines', name='Price',
        line=dict(color='#2979FF', width=2),
        fill='tozeroy', fillcolor='rgba(41, 121, 255, 0.1)'
    ))
    
    # Aggiunta Livelli se trade attivo
    if rec in ['BUY', 'SELL']:
        # Entry (Bianco tratteggiato)
        fig.add_hline(y=entry, line_dash="dash", line_color="white", line_width=1, annotation_text="ENTRY", annotation_font_color="white")
        # SL (Rosso)
        fig.add_hline(y=sl, line_dash="dot", line_color="#FF3B30", line_width=2, annotation_text="SL", annotation_font_color="#FF3B30")
        # TP (Verde)
        fig.add_hline(y=tp, line_dash="dot", line_color="#00FF9D", line_width=2, annotation_text="TP", annotation_font_color="#00FF9D")

    # Stile Bloomberg/Terminal
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,10,1)', # Sfondo grafico quasi nero
        height=450,
        margin=dict(l=0, r=40, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='#222', title=None),
        yaxis=dict(showgrid=True, gridcolor='#222', title=None, side='right'), # Prezzo a destra come nei terminali pro
        showlegend=False,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Waiting for Price Feed...")

# Auto-Refresh Loop
if auto_refresh:
    time.sleep(1)
    st.rerun()
