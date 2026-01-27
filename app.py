import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import time
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="TITAN V90 Oracle",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PER STILE DARK ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30333d;
        text-align: center;
    }
    .signal-buy { color: #00ff00; font-weight: bold; font-size: 24px; }
    .signal-sell { color: #ff0000; font-weight: bold; font-size: 24px; }
    .signal-wait { color: #888888; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# --- CONNESSIONE DATABASE ---
# Tenta di prendere le credenziali da Streamlit Secrets o .env locale
try:
    if 'SUPABASE_URL' in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    else:
        from dotenv import load_dotenv
        load_dotenv()
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL:
        st.error("❌ Credenziali Database mancanti! Imposta i Secrets su Streamlit Cloud.")
        st.stop()

    @st.cache_resource
    def init_db():
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    supabase = init_db()

except Exception as e:
    st.error(f"Errore Connessione: {e}")
    st.stop()

# --- FUNZIONI DI CARICAMENTO DATI ---
def get_last_signal(symbol):
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
    except: return None

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
        return pd.DataFrame()
    except: return pd.DataFrame()

# --- INTERFACCIA UTENTE ---

# 1. SIDEBAR
st.sidebar.title("⚡ TITAN V90")
st.sidebar.markdown("---")
symbol = st.sidebar.radio("Seleziona Asset:", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"])
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-Refresh (2s)", value=True)

if st.sidebar.button("🔄 Aggiorna Manuale"):
    st.rerun()

# 2. CARICAMENTO DATI
signal_data = get_last_signal(symbol)
history_df = get_price_history(symbol)

# 3. HEADER
st.title(f"Analisi Mercato: {symbol}")

if signal_data:
    # --- DASHBOARD SEGNALI ---
    col1, col2, col3 = st.columns(3)
    
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0.0)
    
    # Colore e Icona dinamici
    color_class = "signal-wait"
    if rec == "BUY": color_class = "signal-buy"
    elif rec == "SELL": color_class = "signal-sell"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #aaa;">SIGNAL</div>
            <div class="{color_class}">{rec}</div>
            <div style="font-size: 12px; margin-top: 5px;">{signal_data.get('market_regime', 'SCANNING')}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #aaa;">LEVELS</div>
            <div style="font-size: 18px; font-weight: bold;">Entry: {entry}</div>
            <div style="font-size: 14px; color: #ff4b4b;">SL: {sl}</div>
            <div style="font-size: 14px; color: #00ff00;">TP: {tp}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        conf = signal_data.get('confidence_score', 0)
        details = signal_data.get('details', 'No data')
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #aaa;">CONFIDENCE</div>
            <div style="font-size: 24px; font-weight: bold;">{conf}%</div>
            <div style="font-size: 12px; color: #aaa;">{details}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning(f"Nessun dato segnale per {symbol}. Il bridge è attivo?")

# 4. GRAFICO
st.markdown("### 📈 Live Chart")

if not history_df.empty:
    current_price = history_df['price'].iloc[-1]
    last_time = history_df['created_at'].iloc[-1].strftime("%H:%M:%S")
    
    # Delta prezzo rispetto a 10 tick fa
    delta = 0
    if len(history_df) > 10:
        delta = current_price - history_df['price'].iloc[-10]
    
    st.metric(label=f"Prezzo Attuale ({last_time})", value=f"{current_price}", delta=f"{delta:.5f}")
    
    fig = go.Figure()
    
    # Linea Prezzo
    fig.add_trace(go.Scatter(
        x=history_df['created_at'], 
        y=history_df['price'],
        mode='lines',
        name='Prezzo',
        line=dict(color='#00ccff', width=2)
    ))
    
    # Se c'è un segnale attivo, disegna Entry, SL e TP
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        entry = signal_data['entry_price']
        sl = signal_data['stop_loss']
        tp = signal_data['take_profit']
        
        # Linea Entry
        fig.add_hline(y=entry, line_dash="dot", annotation_text="ENTRY", annotation_position="top right", line_color="white")
        # Linea SL
        fig.add_hline(y=sl, line_dash="dash", annotation_text="STOP", annotation_position="bottom right", line_color="red")
        # Linea TP
        fig.add_hline(y=tp, line_dash="dash", annotation_text="TARGET", annotation_position="top right", line_color="green")

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        xaxis_title="Time",
        yaxis_title="Price"
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("In attesa di dati dal feed... (Assicurati che bridge.py stia girando)")

# 5. LOGICA AUTO-REFRESH
if auto_refresh:
    time.sleep(2)
    st.rerun()
