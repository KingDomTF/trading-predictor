import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import time
import os
from dotenv import load_dotenv

# --- 1. CONFIGURAZIONE PAGINA & TEMA DARK ---
st.set_page_config(
    page_title="TITAN V90 Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS PERSONALIZZATO (ESTETICA DARK) ---
# Questo blocco forza il look "finanziario scuro" dei tuoi screenshot precedenti
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
    
    /* Stile delle "Card" (i box dei dati) */
    .metric-container {
        background-color: #1E2127;
        border: 1px solid #30333D;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 14px;
        color: #8b949e;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .metric-sub {
        font-size: 14px;
        color: #8b949e;
    }

    /* Colori specifici per i Segnali */
    .buy-signal { color: #27C469 !important; font-size: 56px !important; text-shadow: 0 0 15px rgba(39, 196, 105, 0.3); }
    .sell-signal { color: #E74C3C !important; font-size: 56px !important; text-shadow: 0 0 15px rgba(231, 76, 60, 0.3); }
    .wait-signal { color: #7F8C8D !important; font-size: 56px !important; }

    /* Colori per SL e TP nei box */
    .stop-loss-box { border-color: #E74C3C !important; }
    .stop-value { color: #E74C3C !important; }
    .target-box { border-color: #27C469 !important; }
    .target-value { color: #27C469 !important; }
    
    /* Rimuovere padding extra in alto */
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNESSIONE DATABASE (Robustezza) ---
try:
    # Prova a caricare da .env locale per sviluppo
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Se siamo sul cloud, sovrascrivi con i Secrets
    if 'SUPABASE_URL' in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if 'SUPABASE_KEY' in st.secrets:
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("🚨 ERRORE CRITICO: Credenziali Supabase mancanti (controlla .env o Secrets).")
        st.stop()

    @st.cache_resource
    def init_db():
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    supabase = init_db()

except Exception as e:
    st.error(f"Errore di connessione al database: {e}")
    st.stop()

# --- 4. FUNZIONI DATI ---
def get_last_signal(symbol):
    try:
        response = supabase.table("ai_oracle")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        if response.data: return response.data[0]
    except Exception as e: print(f"DB Error (Signal): {e}")
    return None

def get_price_history(symbol):
    try:
        response = supabase.table("mt4_feed")\
            .select("created_at, price")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(150)\
            .execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df.sort_values('created_at')
    except Exception as e: print(f"DB Error (History): {e}")
    return pd.DataFrame()

# --- 5. INTERFACCIA UTENTE ---

# Sidebar
with st.sidebar:
    st.title("⚡ TITAN V90")
    st.markdown("### Terminale Operativo")
    symbol = st.radio("ASSET SELECTION:", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"], index=3) # Default XAUUSD
    st.markdown("---")
    st.markdown("### Impostazioni")
    auto_refresh = st.toggle("Auto-Refresh (1s)", value=True)
    if st.button("🔄 Aggiorna Dati Ora", use_container_width=True):
        st.rerun()
    st.markdown("---")
    status_col1, status_col2 = st.columns(2)
    status_col1.metric("Status", "ONLINE", delta_color="normal")
    status_col2.metric("Mode", "HUNTER", delta_color="off")

# Main Content
st.markdown(f"# 📊 Analisi Mercato: {symbol}")

# Caricamento Dati
signal_data = get_last_signal(symbol)
history_df = get_price_history(symbol)

current_price = 0.0
if not history_df.empty:
    current_price = history_df['price'].iloc[-1]

# --- VISUALIZZAZIONE SEGNALE (Le Card Belle) ---
if signal_data:
    rec = signal_data.get('recommendation', 'WAIT')
    conf = signal_data.get('confidence_score', 0)
    regime = signal_data.get('market_regime', 'SCANNING')
    details = signal_data.get('details', '')

    # Layout a 3 colonne per i box principali
    col_sig, col_conf, col_price = st.columns([1.5, 1, 1])

    with col_sig:
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
        
    with col_conf:
        # Barra di confidenza colorata
        bar_color = "#7F8C8D"
        if rec == "BUY": bar_color = "#27C469"
        elif rec == "SELL": bar_color = "#E74C3C"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CONFIDENCE</div>
            <div class="metric-value" style="color: {bar_color};">{conf}%</div>
            <div style="background-color: #30333D; height: 8px; border-radius: 4px; margin-top: 15px;">
                <div style="background-color: {bar_color}; width: {conf}%; height: 100%; border-radius: 4px; transition: width 0.5s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_price:
        # Prezzo attuale grande
        price_display = current_price if current_price > 0 else signal_data.get('current_price', 0)
        st.markdown(f"""
        <div class="metric-container" style="border-color: #5865F2;">
            <div class="metric-label">CURRENT PRICE</div>
            <div class="metric-value">${price_display:.2f}</div>
            <div class="metric-sub">Live Feed</div>
        </div>
        """, unsafe_allow_html=True)

    # --- LIVELLI OPERATIVI (Se Attivi) ---
    if rec in ["BUY", "SELL"]:
        st.markdown("### 🎯 Livelli Operativi")
        col_entry, col_sl, col_tp = st.columns(3)
        
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)

        with col_entry:
             st.markdown(f"""<div class="metric-container"><div class="metric-label">ENTRY</div><div class="metric-value" style="color: #5865F2;">${entry:.2f}</div></div>""", unsafe_allow_html=True)
        with col_sl:
             st.markdown(f"""<div class="metric-container stop-loss-box"><div class="metric-label">STOP LOSS</div><div class="metric-value stop-value">${sl:.2f}</div></div>""", unsafe_allow_html=True)
        with col_tp:
             st.markdown(f"""<div class="metric-container target-box"><div class="metric-label">TAKE PROFIT</div><div class="metric-value target-value">${tp:.2f}</div></div>""", unsafe_allow_html=True)
    else:
        st.info("System is Scanning for Setup...")

else:
    # Stato iniziale o errore ponte
    st.warning("In attesa del Bridge V90... (Assicurati che il terminale nero stia trasmettendo)")

# --- 6. GRAFICO LIVE ---
st.markdown("---")
if not history_df.empty:
    fig = go.Figure()

    # Linea Prezzo
    fig.add_trace(go.Scatter(
        x=history_df['created_at'], y=history_df['price'],
        mode='lines', name='Price',
        line=dict(color='#5865F2', width=2.5),
        fill='tozeroy', fillcolor='rgba(88, 101, 242, 0.1)' # Effetto area sotto la linea
    ))

    # Aggiunta Livelli se c'è un trade attivo
    if signal_data and signal_data.get('recommendation') in ['BUY', 'SELL']:
        sl = signal_data['stop_loss']
        tp = signal_data['take_profit']
        entry = signal_data['entry_price']
        
        fig.add_hline(y=entry, line_dash="dot", line_color="white", annotation_text="ENTRY", annotation_font_color="white")
        fig.add_hline(y=sl, line_dash="dash", line_color="#E74C3C", annotation_text="STOP", annotation_font_color="#E74C3C")
        fig.add_hline(y=tp, line_dash="dash", line_color="#27C469", annotation_text="TARGET", annotation_font_color="#27C469")

    # Layout Grafico Dark
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', # Sfondo trasparente per fondersi
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        height=450,
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='#30333D', title="Price"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container
