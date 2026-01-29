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

# --- 2. CSS DARK MODE & STILE "CARD" (Look Professionale) ---
st.markdown("""
<style>
    /* Sfondo Generale */
    .stApp {
        background-color: #0E1117 !important;
        color: #FAFAFA;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30333D;
    }
    
    /* Stile delle Card (Box Dati) */
    .metric-container {
        background-color: #1E2127;
        border: 1px solid #30333D;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    /* Testi */
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
    }
    .metric-value {
        font-size: 38px;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 5px;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: auto;
        font-style: italic;
    }

    /* Colori Segnali */
    .buy-signal { 
        color: #27C469 !important; 
        text-shadow: 0 0 20px rgba(39, 196, 105, 0.3); 
    }
    .sell-signal { 
        color: #E74C3C !important; 
        text-shadow: 0 0 20px rgba(231, 76, 60, 0.3); 
    }
    .wait-signal { 
        color: #7F8C8D !important; 
    }

    /* Bordi Colorati per SL/TP */
    .entry-box { border-bottom: 4px solid #5865F2 !important; }
    .stop-box { border-bottom: 4px solid #E74C3C !important; }
    .target-box { border-bottom: 4px solid #27C469 !important; }
    
    .entry-text { color: #5865F2 !important; }
    .stop-text { color: #E74C3C !important; }
    .target-text { color: #27C469 !important; }

    /* Pulizia spazi */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNESSIONE DATABASE ---
try:
    load_dotenv()
    # Tenta prima dal file .env locale
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Se siamo su Streamlit Cloud, usa i Secrets
    if 'SUPABASE_URL' in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if 'SUPABASE_KEY' in st.secrets:
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("🚨 ERRORE: Credenziali Database mancanti. Controlla il file .env")
        st.stop()

    @st.cache_resource
    def init_db():
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    supabase = init_db()

except Exception as e:
    st.error(f"Errore Connessione DB: {e}")
    st.stop()

# --- 4. FUNZIONI RECUPERO DATI ---
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

# --- 5. INTERFACCIA UTENTE ---

# Sidebar
with st.sidebar:
    st.title("⚡ TITAN V90")
    st.markdown("### 🎛️ Control Panel")
    symbol = st.radio("ASSET:", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"], index=3)
    st.markdown("---")
    auto_refresh = st.toggle("Auto-Refresh (1s)", value=True)
    if st.button("🔄 Force Update"):
        st.rerun()
    st.markdown("---")
    st.caption("Status: ONLINE | Mode: HUNTER V90")

# Header Principale
st.markdown(f"# 📊 {symbol} Market Analysis")

# Fetch Dati
signal_data = get_last_signal(symbol)
history_df = get_price_history(symbol)

current_price = 0.0
if not history_df.empty:
    current_price = history_df['price'].iloc[-1]
elif signal_data:
    current_price = signal_data.get('current_price', 0.0)

# --- DASHBOARD KPI ---
if signal_data and signal_data.get('market_regime') != 'SCANNING':
    rec = signal_data.get('recommendation', 'WAIT')
    conf = signal_data.get('confidence_score', 0)
    regime = signal_data.get('market_regime', 'WAIT')
    details = signal_data.get('details', '')

    # RIGA 1: Segnale, Confidenza, Prezzo
    c1, c2, c3 = st.columns([1.5, 1.2, 1])

    with c1:
        # Colore dinamico
        sig_class = "wait-signal"
        if rec == "BUY": sig_class = "buy-signal"
        elif rec == "SELL": sig_class = "sell-signal"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">AI SIGNAL</div>
            <div class="metric-value {sig_class}">{rec}</div>
            <div class="metric-sub">{regime} | {details.split('|')[0] if '|' in details else details}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Barra Confidenza
        bar_color = "#7F8C8D"
        if rec == "BUY": bar_color = "#27C469"
        elif rec == "SELL": bar_color = "#E74C3C"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CONFIDENCE SCORE</div>
            <div class="metric-value" style="color: {bar_color};">{conf}%</div>
            <div style="width: 100%; background: #161B22; height: 8px; border-radius: 4px; margin-top: 10px;">
                <div style="width: {conf}%; background: {bar_color}; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">LIVE PRICE</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-sub">Real-Time Feed</div>
        </div>
        """, unsafe_allow_html=True)

    # RIGA 2: Livelli Operativi (Solo se Active)
    if rec in ["BUY", "SELL"]:
        st.markdown("<br>", unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        with l1:
            st.markdown(f"""<div class="metric-container entry-box"><div class="metric-label">ENTRY</div><div class="metric-value entry-text">${entry:.2f}</div></div>""", unsafe_allow_html=True)
        with l2:
            st.markdown(f"""<div class="metric-container stop-box"><div class="metric-label">STOP LOSS</div><div class="metric-value stop-text">${sl:.2f}</div></div>""", unsafe_allow_html=True)
        with l3:
            st.markdown(f"""<div class="metric-container target-box"><div class="metric-label">TARGET</div><div class="metric-value target-text">${tp:.2f}</div></div>""", unsafe_allow_html=True)

else:
    # Box di Attesa
    st.info("🔄 In attesa di dati dal Bridge V90 (Assicurati che bridge.py stia girando)...")

# --- 6. GRAFICO LIVE ---
st.markdown("---")
st.markdown("### 📈 Live Chart")

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
        fillcolor='rgba(88, 101, 242, 0.15)'
    ))

    # Linee Livelli (Se presenti)
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data.get('entry_price')
        sl = signal_data.get('stop_loss')
        tp = signal_data.get('take_profit')
        
        if entry: fig.add_hline(y=entry, line_dash="dot", line_color="white", annotation_text="ENTRY")
        if sl: fig.add_hline(y=sl, line_dash="dash", line_color="#E74C3C", annotation_text="SL")
        if tp: fig.add_hline(y=tp, line_dash="dash", line_color="#27C469", annotation_text="TP")

    # Stile Grafico Dark
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(14, 17, 23, 0.5)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor='#30333D', title=None),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 7. AUTO REFRESH ---
if auto_refresh:
    time.sleep(1)
    st.rerun()
