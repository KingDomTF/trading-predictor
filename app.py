import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import warnings
import os
import time
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE BRIDGE ====================
# [IMPORTANTE] Modifica questo percorso con il tuo
DEFAULT_MT4_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\COMMON\Files"
DATA_FILENAME = "mt4_live_prices.csv"

# ==================== GESTIONE DATI MT4 (ROBUSTA) ====================
def get_mt4_data():
    """
    Legge il file CSV generato da MT4.
    Gestisce conflitti di lettura e restituisce un DataFrame pulito.
    """
    path = st.session_state.get('mt4_path', DEFAULT_MT4_PATH)
    file_path = os.path.join(path, DATA_FILENAME)
    
    # 1. Verifica esistenza file
    if not os.path.exists(file_path):
        return None, "File non trovato. Verifica il percorso e che l'EA sia attivo."
    
    # 2. Verifica aggiornamento file (se il file è vecchio di > 10 secondi, l'EA è fermo)
    try:
        mtime = os.path.getmtime(file_path)
        if time.time() - mtime > 15:
            return None, "Dati obsoleti. L'EA su MT4 sembra fermo o il mercato è chiuso."
    except:
        pass

    # 3. Tentativo di lettura con retry (per evitare conflitti I/O)
    max_retries = 5
    for _ in range(max_retries):
        try:
            # Leggiamo il CSV
            # Formato atteso: Symbol, Bid, Ask, Time
            data = []
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        data.append({
                            'Symbol': row[0],
                            'Bid': float(row[1]),
                            'Ask': float(row[2]),
                            'Time': row[3] if len(row) > 3 else "N/A"
                        })
            
            if not data:
                return None, "Il file è vuoto. L'EA sta inizializzando..."
                
            return pd.DataFrame(data), "OK"
            
        except PermissionError:
            # Se MT4 sta scrivendo, aspettiamo un attimo
            time.sleep(0.05)
        except Exception as e:
            return None, f"Errore lettura: {str(e)}"
    
    return None, "Impossibile accedere al file dopo vari tentativi."

# ==================== FUNZIONI AI & CORE (Standard) ====================
# (Mantengo le funzioni di calcolo tecnico invariate per l'analisi)

def calculate_technical_indicators(df):
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
    df = df.dropna()
    return df

@st.cache_data
def load_historical_data_yf(symbol, interval='1h'):
    """Scarica storico da YFinance per l'analisi tecnica (le candele)."""
    # Mapping nomi MT4 -> Yahoo Finance
    yf_map = {
        "XAUUSD": "GC=F",
        "EURUSD": "EURUSD=X",
        "BTCUSD": "BTC-USD",
        "XAGUSD": "SI=F",
        "US500": "^GSPC"
    }
    
    yf_ticker = yf_map.get(symbol, symbol)
    try:
        data = yf.download(yf_ticker, period='6mo', interval=interval, progress=False)
        if len(data) < 50: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return None

# ==================== INTERFACCIA UTENTE ====================
st.set_page_config(page_title="MT4 Live Monitor", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .stMetric { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px; }
    .status-ok { color: #198754; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
    .live-card { border-left: 5px solid #ffc107; background: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Configurazione ---
with st.sidebar:
    st.header("⚙️ Configurazione Bridge")
    path_input = st.text_input("Percorso Cartella 'Files' MT4", value=DEFAULT_MT4_PATH)
    if path_input: st.session_state['mt4_path'] = path_input
    st.caption("Assicurati che il percorso sia corretto e l'EA sia attivo.")
    if st.button("Pulisci Cache"):
        st.cache_data.clear()

st.title("📡 MT4 Live Monitor AI")
st.markdown("Monitoraggio prezzi reali da MetaTrader 4 e Analisi AI sui dati storici.")

# --- SEZIONE 1: PREZZI LIVE DA MT4 ---
st.markdown("### 🔴 Quotazioni Real-Time (MT4)")

# Pulsante di refresh manuale (Streamlit non si aggiorna da solo senza loop complessi)
if st.button("🔄 Aggiorna Quotazioni Ora", use_container_width=True):
    pass 

df_mt4, status_msg = get_mt4_data()

if df_mt4 is not None:
    st.markdown(f"<span class='status-ok'>✅ Connesso a MT4 - Ultimo aggiornamento EA: {df_mt4['Time'].iloc[0]}</span>", unsafe_allow_html=True)
    
    # Creazione colonne dinamiche in base a quanti simboli trova
    cols = st.columns(len(df_mt4))
    for index, row in df_mt4.iterrows():
        with cols[index]:
            symbol = row['Symbol']
            bid = row['Bid']
            ask = row['Ask']
            spread = (ask - bid)
            
            # Formattazione spread
            if "JPY" in symbol or "XAU" in symbol or "XAG" in symbol or "500" in symbol:
                spread_fmt = f"{spread:.2f}"
            else:
                spread_fmt = f"{spread:.5f}"
                
            st.metric(
                label=f"**{symbol}**",
                value=f"{bid}",
                delta=f"Ask: {ask} | Spread: {spread_fmt}",
                delta_color="off"
            )
else:
    st.markdown(f"<span class='status-err'>⚠️ Stato Connessione: {status_msg}</span>", unsafe_allow_html=True)
    st.info("Consiglio: Controlla che il percorso file nella sidebar sia corretto e che l'EA 'PyBridge_Feeder' stia girando.")

st.markdown("---")

# --- SEZIONE 2: ANALISI AI IBRIDA ---
# Logica: Usa i prezzi LIVE per il livello attuale, ma lo storico Yahoo per calcolare gli indicatori

st.markdown("### 🧠 Analisi Tecnica AI (Hybrid Mode)")

# Selettore basato sui simboli trovati in MT4 (o default se non connesso)
available_symbols = df_mt4['Symbol'].tolist() if df_mt4 is not None else ["XAUUSD", "EURUSD", "BTCUSD"]
selected_symbol = st.selectbox("Seleziona Strumento da Analizzare", available_symbols)

analyze_btn = st.button(f"Analizza {selected_symbol}")

if analyze_btn:
    # 1. Recupera Prezzo Live (se disponibile)
    live_price = None
    if df_mt4 is not None:
        row = df_mt4[df_mt4['Symbol'] == selected_symbol]
        if not row.empty:
            live_price = row['Bid'].values[0]
    
    if live_price is None:
        st.warning(f"⚠️ Prezzo live per {selected_symbol} non disponibile. Uso ultimo prezzo di chiusura storico (potrebbe essere in ritardo).")
    
    # 2. Carica Storico e Calcola Indicatori
    with st.spinner("Elaborazione dati storici..."):
        df_hist = load_historical_data_yf(selected_symbol)
        
        if df_hist is not None:
            df_ind = calculate_technical_indicators(df_hist)
            last_hist = df_ind.iloc[-1]
            
            # Se abbiamo il prezzo live, usiamo quello come "Close" attuale per ricalcolare livelli critici
            current_price = live_price if live_price else last_hist['Close']
            
            # Calcolo livelli dinamici basati su ATR
            atr = last_hist['ATR']
            trend = "RIALSISTA 🟢" if last_hist['Trend'] == 1 else "RIBASSISTA 🔴"
            rsi = last_hist['RSI']
            
            # Visualizzazione
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### 📊 Dati Tecnici ({selected_symbol})")
                st.write(f"Prezzo Analisi: **{current_price}**")
                st.write(f"Trend (SMA20): **{trend}**")
                st.write(f"RSI (14): **{rsi:.2f}**")
                st.write(f"ATR (Volatilità): **{atr:.4f}**")
            
            with col2:
                st.markdown("#### 🎯 Livelli Chiave Stimati")
                st.markdown(f"""
                <div class='live-card'>
                    <strong>Resistenza Dinamica / TP Short:</strong> {current_price + (atr*1.5):.4f}<br>
                    <strong>Supporto Dinamico / TP Long:</strong> {current_price - (atr*1.5):.4f}<br>
                    <hr style='margin:5px 0'>
                    <em>Questi livelli sono calcolati partendo dal prezzo LIVE MT4 + volatilità storica.</em>
                </div>
                """, unsafe_allow_html=True)
                
            # Avviso finale
            st.info("Nota: Questa analisi usa la volatilità storica (Yahoo) applicata al prezzo reale (MT4) per darti i livelli più precisi possibili senza eseguire ordini.")
            
        else:
            st.error("Errore nel recupero dei dati storici. Verifica la connessione internet o il simbolo.")

st.markdown("<br><br><div style='text-align:center; color:gray; font-size:0.8em'>System V3.0 - Read Only Mode</div>", unsafe_allow_html=True)
