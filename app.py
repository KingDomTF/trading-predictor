import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import time
import os
from dotenv import load_dotenv

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="TITAN V90 Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS DARK MODE ESTREMO & STILE CARD (Il ritorno del look professionale) ---
st.markdown("""
<style>
    /* Sfondo Generale Profondo */
    .stApp {
        background-color: #0E1117 !important; /* Nero profondo */
        color: #FAFAFA;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30333D;
    }
    
    /* Stile delle "Card" (i box dei dati) */
    .metric-container {
        background-color: #1E2127;
        border: 1px solid #30333D;
        border-radius: 12px; /* Arrotondamento maggiore */
        padding: 25px;
        text-align: center;
        height: 100%;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3); /* Ombra più profonda */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-2px); /* Leggero effetto hover */
    }

    /* Etichette e Valori */
    .metric-label {
        font-size: 14px;
        color: #8b949e;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }
    .metric-value {
        font-size: 42px; /* Testo più grande */
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 10px;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 13px;
        color: #8b949e;
        margin-top: auto;
        font-style: italic;
    }

    /* Segnali Luminosi (Glow Effect) */
    .buy-signal { 
        color: #27C469 !important; 
        font-size: 64px !important; 
        text-shadow: 0 0 20px rgba(39, 196, 105, 0.4); 
    }
    .sell-signal { 
        color: #E74C3C !important; 
        font-size: 64px !important; 
        text-shadow: 0 0 20px rgba(231, 76, 60, 0.4); 
    }
    .wait-signal { 
        color: #7F8C8D !important; 
        font-size: 64px !important;
    }

    /* Bordi colorati specifici per i box inferiori */
    .entry-box { border-bottom: 4px solid #5865F2 !important; }
    .stop-box { border-bottom: 4px solid #E74C3C !important; }
    .target-box { border-bottom: 4px solid #27C469 !important; }
    
    .entry-text { color: #5865F2 !important; }
    .stop-text { color: #E74C3C !important; }
    .target-text { color: #27C469 !important; }

    /* Pulizia layout */
    .block-container { padding-top: 3rem; padding-bottom: 3rem; }
    h1 { font-weight: 900 !important; letter-spacing: -1px; }
    
    /* Messaggi di stato */
    .waiting-box {
        background-color: #1E2127;
        border: 1px solid #30333D;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: #8b949e;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNESSIONE DATABASE (Robustezza V90) ---
try:
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Override con Secrets se siamo su Cloud
    if 'SUPABASE_URL' in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if 'SUPABASE_KEY' in st.secrets:
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("🚨 DATABASE ERROR: Credenziali mancanti (.env o Secrets).")
        st.stop()

    @st.cache_resource
    def init_db():
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    supabase = init_db()

except Exception as e:
    st.error(f"Errore Connessione DB: {e}")
    st.stop()

# --- 4. FUNZIONI DI RECUPERO DATI ---
def get_last_signal(symbol):
    try:
        response = supabase.table("ai_oracle")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        if response.data: return response.data[0]
    except: pass
    return None

def get_price_history(symbol):
    try:
        response = supabase.table("mt4_feed")\
            .select("created_at, price")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(200)\
            .execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df.sort_values('created_at')
    except: pass
    return pd.DataFrame()

# --- 5. INTERFACCIA UTENTE (Layout Professionale) ---

# Sidebar
with st.sidebar:
    st.markdown("# ⚡ TITAN V90")
    st.markdown("### 🎛️ Control Panel")
    symbol = st.radio("ASSET SELECTION:", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"], index=3)
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    auto_refresh = st.toggle("Auto-Refresh (1s)", value=True)
    if st.button("🔄 Force Update Now", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.caption(f"📡 System Status: ONLINE\n🎯 Mode: HUNTER V90 Enterprise")

# Header
st.markdown(f"# 📊 {symbol} Market Analysis")
st.markdown("---")

# Caricamento Dati
signal_data = get_last_signal(symbol)
history_df = get_price_history(symbol)

current_price = 0.0
if not history_df.empty:
    current_price = history_df['price'].iloc[-1]
elif signal_data:
    current_price = signal_data.get('current_price', 0.0)

# --- SEZIONE 1: KPI CARDS (Il look che volevi) ---
if signal_data and signal_data.get('market_regime') != 'SCANNING':
    rec = signal_data.get('recommendation', 'WAIT')
    conf = signal_data.get('confidence_score', 0)
    regime = signal_data.get('market_regime', 'WAIT')
    details = signal_data.get('details', '')

    # Riga Superiore: Segnale, Confidenza, Prezzo
    c1, c2, c3 = st.columns([1.5, 1.2, 1])

    with c1:
        # Colore dinamico con GLOW
        sig_class = "wait-signal"
        if rec == "BUY": sig_class = "buy-signal"
        elif rec == "SELL": sig_class = "sell-signal"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">AI SIGNAL</div>
            <div class="metric-value {sig_class}">{rec}</div>
            <div class="metric-sub">{regime.upper()} | {details.split('|')[0] if '|' in details else details}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Barra Confidenza Professionale
        bar_c = "#7F8C8D"
        if rec == "BUY": bar_c = "#27C469"
        elif rec == "SELL": bar_c = "#E74C3C"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CONFIDENCE SCORE</div>
            <div class="metric-value" style="color: {bar_c};">{conf}%</div>
            <div style="width: 100%; background: #161B22; height: 10px; border-radius: 5px; margin-top: 15px; border: 1px solid #30333D;">
                <div style="width: {conf}%; background: {bar_c}; height: 100%; border-radius: 4px; box-shadow: 0 0 10px {bar_c};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        # Prezzo Live
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">LIVE PRICE</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-sub">Real-Time Tick Feed</div>
        </div>
        """, unsafe_allow_html=True)

    # Riga Inferiore: Livelli Operativi (Solo se Active)
    if rec in ["BUY", "SELL"]:
        st.markdown("<br>", unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        # Card specifiche con bordi colorati
        with l1:
            st.markdown(f"""<div class="metric-container entry-box"><div class="metric-label">ENTRY PRICE</div><div class="metric-value entry-text">${entry:.2f}</div></div>""", unsafe_allow_html=True)
        with l2:
            st.markdown(f"""<div class="metric-container stop-box"><div class="metric-label">STOP LOSS</div><div class="metric-value stop-text">${sl:.2f}</div></div>""", unsafe_allow_html=True)
        with l3:
            st.markdown(f"""<div class="metric-container target-box"><div class="metric-label">TAKE PROFIT</div><div class="metric-value target-text">${tp:.2f}</div></div>""", unsafe_allow_html=True)

else:
    # Stato di attesa stiloso
    st.markdown("""
    <div class="waiting-box">
        🔄 <b>TITAN V90 sta analizzando il mercato...</b><br>
        <span style="font-size: 14px;">In attesa di dati dal Bridge Python. Assicurati che sia attivo.</span>
    </div>
    """, unsafe_allow_html=True)

# --- 6. GRAFICO LIVE DARK ---
st.markdown("---")
st.markdown("### 📈 Live Chart")
if not history_df.empty:
    fig = go.Figure()

    # Linea Prezzo con Area
    fig.add_trace(go.Scatter(
        x=history_df['created_at'], 
        y=history_df['price'],
        mode='lines',
        name='Price',
        line=dict(color='#5865F2', width=3),
        fill='tozeroy',
        fillcolor='rgba(88, 101, 242, 0.15)' # Area più visibile
    ))

    # Linee Livelli (Entry/SL/TP) se presenti
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price')
        sl = signal_data.get('stop_loss')
        tp = signal_data.get('take_profit')
        
        if entry: fig.add_hline(y=entry, line_dash="dot", line_color="white", line_width=1, annotation_text="ENTRY", annotation_position="top right")
        if sl: fig.add_hline(y=sl, line_dash="dash", line_color="#E74C3C", line_width=2, annotation_text="SL", annotation_position="bottom right")
        if tp: fig.add_hline(y=tp, line_dash="dash", line_color="#27C469", line_width=2, annotation_text="TP", annotation_position="top right")

    # Styling Dark Plotly Avanzato
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(14, 17, 23, 0.5)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
        xaxis=dict(showgrid=False, title=None, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#30333D', title=None, zeroline=False),
        showlegend=False,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

# --- 7. AUTO REFRESH ---
if auto_refresh:
    time.sleep(1)
    st.rerun()
