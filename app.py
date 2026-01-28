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

# --- 2. CSS DARK MODE FORZATO & STILE CARD ---
st.markdown("""
<style>
    /* Sfondo Generale e Testo */
    .stApp {
        background-color: #0E1117 !important;
        color: #FAFAFA;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30333D;
    }
    
    /* Container Box */
    .metric-container {
        background-color: #1E2127;
        border: 1px solid #30333D;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: auto;
    }

    /* Colori Segnali */
    .buy-signal { color: #27C469 !important; text-shadow: 0 0 15px rgba(39, 196, 105, 0.2); }
    .sell-signal { color: #E74C3C !important; text-shadow: 0 0 15px rgba(231, 76, 60, 0.2); }
    .wait-signal { color: #7F8C8D !important; }

    /* Bordi Colorati per SL/TP */
    .stop-box { border-bottom: 3px solid #E74C3C !important; }
    .target-box { border-bottom: 3px solid #27C469 !important; }
    .entry-box { border-bottom: 3px solid #5865F2 !important; }
    
    /* Rimozione spazi bianchi eccessivi */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { padding-top: 0px; margin-top: 0px; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNESSIONE DATABASE ---
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
            .limit(100)\
            .execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df.sort_values('created_at')
    except: pass
    return pd.DataFrame()

# --- 5. INTERFACCIA UTENTE ---

# Sidebar
with st.sidebar:
    st.title("⚡ TITAN V90")
    st.markdown("### Control Panel")
    symbol = st.radio("ASSET:", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"], index=3)
    st.markdown("---")
    auto_refresh = st.toggle("Auto-Refresh (1s)", value=True)
    if st.button("🔄 Force Update", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.caption(f"System Status: ONLINE\nMode: HUNTER V90")

# Main Content
st.markdown(f"# 📊 {symbol} Market Analysis")

signal_data = get_last_signal(symbol)
history_df = get_price_history(symbol)

current_price = 0.0
if not history_df.empty:
    current_price = history_df['price'].iloc[-1]
elif signal_data:
    current_price = signal_data.get('current_price', 0.0)

# --- VISUALIZZAZIONE KPI (LE CARD) ---
if signal_data:
    rec = signal_data.get('recommendation', 'WAIT')
    conf = signal_data.get('confidence_score', 0)
    regime = signal_data.get('market_regime', 'SCANNING')
    details = signal_data.get('details', '')

    # Riga Superiore: Segnale, Confidenza, Prezzo
    c1, c2, c3 = st.columns([1.5, 1, 1])

    with c1:
        # Colore dinamico
        sig_color = "wait-signal"
        if rec == "BUY": sig_color = "buy-signal"
        elif rec == "SELL": sig_color = "sell-signal"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">AI RECOMMENDATION</div>
            <div class="metric-value {sig_color}">{rec}</div>
            <div class="metric-sub">{regime} | {details.split('|')[0] if '|' in details else details}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Barra Confidenza
        bar_c = "#7F8C8D"
        if rec == "BUY": bar_c = "#27C469"
        elif rec == "SELL": bar_c = "#E74C3C"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CONFIDENCE SCORE</div>
            <div class="metric-value" style="color: {bar_c};">{conf}%</div>
            <div style="width: 100%; background: #30333D; height: 6px; border-radius: 3px; margin-top: 10px;">
                <div style="width: {conf}%; background: {bar_c}; height: 100%; border-radius: 3px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        # Prezzo
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">LIVE PRICE</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-sub">Real-Time Feed</div>
        </div>
        """, unsafe_allow_html=True)

    # Riga Inferiore: Livelli Operativi (Solo se Active)
    if rec in ["BUY", "SELL"]:
        st.markdown("<br>", unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        with l1:
            st.markdown(f"""<div class="metric-container entry-box"><div class="metric-label">ENTRY PRICE</div><div class="metric-value" style="font-size: 28px; color: #5865F2;">${entry:.2f}</div></div>""", unsafe_allow_html=True)
        with l2:
            st.markdown(f"""<div class="metric-container stop-box"><div class="metric-label">STOP LOSS</div><div class="metric-value" style="font-size: 28px; color: #E74C3C;">${sl:.2f}</div></div>""", unsafe_allow_html=True)
        with l3:
            st.markdown(f"""<div class="metric-container target-box"><div class="metric-label">TAKE PROFIT</div><div class="metric-value" style="font-size: 28px; color: #27C469;">${tp:.2f}</div></div>""", unsafe_allow_html=True)

else:
    st.info("🔄 Waiting for TITAN V90 Bridge connection...")

# --- 6. GRAFICO LIVE ---
st.markdown("---")
if not history_df.empty:
    fig = go.Figure()

    # Linea Prezzo
    fig.add_trace(go.Scatter(
        x=history_df['created_at'], 
        y=history_df['price'],
        mode='lines',
        name='Price',
        line=dict(color='#5865F2', width=2),
        fill='tozeroy',
        fillcolor='rgba(88, 101, 242, 0.1)'
    ))

    # Linee Livelli (Entry/SL/TP)
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price')
        sl = signal_data.get('stop_loss')
        tp = signal_data.get('take_profit')
        
        if entry: fig.add_hline(y=entry, line_dash="dot", line_color="white", annotation_text="ENTRY")
        if sl: fig.add_hline(y=sl, line_dash="dash", line_color="#E74C3C", annotation_text="SL")
        if tp: fig.add_hline(y=tp, line_dash="dash", line_color="#27C469", annotation_text="TP")

    # Styling Dark Plotly
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor='#30333D', title=None),
        showlegend=False
    )
    
    # Render Grafico
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 7. AUTO REFRESH ---
if auto_refresh:
    time.sleep(1)
    st.rerun()
