import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import os
import time
import csv

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE GLOBALE ====================
# Percorso standard MT4 (Modificare se necessario nella sidebar)
DEFAULT_MT4_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\COMMON\Files"
DATA_FILENAME = "mt4_live_feed.csv"

# Mapping nomi visuali
proper_names = {
    'GC=F': 'XAU/USD (Gold)',
    'XAUUSD': 'XAU/USD (Gold)',
    'EURUSD': 'EUR/USD',
    'BTCUSD': 'BTC/USD',
}

# ==================== FUNZIONI LETTURA DATI MT4 ====================
def get_mt4_live_data_safe():
    """
    Legge il file CSV generato dall'EA.
    Gestisce i lock del file e i tentativi falliti.
    """
    path = st.session_state.get('mt4_path', DEFAULT_MT4_PATH)
    file_path = os.path.join(path, DATA_FILENAME)
    
    # Se il file non esiste ancora
    if not os.path.exists(file_path):
        return None
    
    try:
        # Tentativo di lettura
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows: return None
            
            # Formato atteso: [SYMBOL, BID, ASK, TIME]
            last_row = rows[-1] # Prende l'ultima riga scritta
            if len(last_row) >= 3:
                return {
                    'symbol': last_row[0],
                    'bid': float(last_row[1]),
                    'ask': float(last_row[2]),
                    'time': last_row[3] if len(last_row) > 3 else "N/A"
                }
    except Exception:
        # Se il file è bloccato da MT4 in quel millisecondo, ritorna None
        # Il loop principale gestirà questo mantenendo il vecchio valore
        return None
    return None

# ==================== FUNZIONI CORE AI (Invariate nella logica) ====================
def calculate_technical_indicators(df):
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    # Bollinger
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])
    
    df = df.dropna()
    return df

def generate_features(df_ind, entry, sl, tp, direction):
    latest = df_ind.iloc[-1]
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    
    features = {
        'sl_distance_pct': sl_distance,
        'tp_distance_pct': abs(tp - entry) / entry * 100,
        'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0,
        'main_tf': 60,
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_signal'],
        'atr': latest['ATR'],
        'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']), 
        'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        'trend': latest['Trend']
    }
    return np.array(list(features.values()), dtype=np.float32)

def simulate_historical_trades(df_ind, n_trades=500):
    X_list = []
    y_list = []
    for _ in range(n_trades):
        idx = np.random.randint(50, len(df_ind) - 50)
        row = df_ind.iloc[idx]
        direction = np.random.choice(['long', 'short'])
        entry = row['Close']
        atr = row['ATR']
        if np.isnan(atr) or atr == 0: continue
        
        sl_mult = np.random.uniform(1.0, 2.0)
        tp_mult = np.random.uniform(1.5, 3.0)
        
        if direction == 'long':
            sl = entry - (atr * sl_mult)
            tp = entry + (atr * tp_mult)
        else:
            sl = entry + (atr * sl_mult)
            tp = entry - (atr * tp_mult)
            
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction)
        future_prices = df_ind.iloc[idx+1:idx+40]['Close'].values
        
        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp)
                hit_sl = np.any(future_prices <= sl)
            else:
                hit_tp = np.any(future_prices <= tp)
                hit_sl = np.any(future_prices >= sl)
            success = 1 if hit_tp and not hit_sl else 0
            X_list.append(features)
            y_list.append(success)
    return np.array(X_list), np.array(y_list)

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler

@st.cache_data
def load_historical_training_data(symbol, interval='1h'):
    """Scarica dati storici da Yahoo per il training dell'AI (non per il live)."""
    try:
        yf_sym = "GC=F" if "XAU" in symbol or "Gold" in symbol else symbol
        data = yf.download(yf_sym, period='1y', interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        if len(data) < 50: return None
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return None

# ==================== INTERFACCIA UTENTE ====================
st.set_page_config(page_title="Gold AI Monitor", page_icon="🥇", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1600px; }
    .stMetric { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 10px; }
    .live-box { background: #d4edda; border: 2px solid #28a745; border-radius: 10px; padding: 15px; text-align: center; }
    .offline-box { background: #f8d7da; border: 2px solid #dc3545; border-radius: 10px; padding: 15px; text-align: center; }
    h1 { color: #b8860b; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Configurazione e Controllo ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Input Percorso
    mt4_path_input = st.text_input("Percorso 'Files' MT4", value=DEFAULT_MT4_PATH)
    if mt4_path_input: st.session_state['mt4_path'] = mt4_path_input
    
    st.markdown("---")
    
    # CONTROLLO STREAMING
    st.subheader("📡 Controllo Dati")
    streaming_active = st.toggle("🔴 ATTIVA STREAMING LIVE", value=False)
    
    st.info("""
    **Istruzioni:**
    1. Trascina l'EA 'PyBridge_DataFeed' su un grafico MT4 (es. XAUUSD).
    2. Copia il percorso della cartella Files qui sopra.
    3. Attiva lo streaming per vedere i prezzi cambiare in tempo reale.
    """)

# --- MAIN PAGE ---
st.title("🥇 Gold Trading Predictor - Live Monitor")

# Placeholder per i dati live (così possiamo aggiornarli nel loop)
live_container = st.empty()
analysis_container = st.container()

# LOGICA DI STREAMING
if streaming_active:
    while True:
        # 1. Recupera dati live
        data = get_mt4_live_data_safe()
        
        with live_container.container():
            if data:
                # Interfaccia LIVE
                col_live_1, col_live_2, col_live_3 = st.columns([1, 2, 1])
                with col_live_2:
                    st.markdown(f"""
                    <div class='live-box'>
                        <h2 style='margin:0; color: #155724;'>LIVE: {data['symbol']}</h2>
                        <h1 style='margin:0; font-size: 3.5rem; color: #155724;'>{data['bid']:.2f}</h1>
                        <p>Ask: {data['ask']:.2f} | Time: {data['time']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Interfaccia OFFLINE
                st.markdown("""
                <div class='offline-box'>
                    <h3>⚠️ Segnale MT4 Assente</h3>
                    <p>Controlla che l'EA sia attivo sul grafico e il percorso cartella sia corretto.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Pausa per non fondere la CPU (0.5s è buono per l'occhio umano)
        time.sleep(0.5)
else:
    # Se lo streaming è spento, mostra l'ultimo stato o messaggio statico
    data = get_mt4_live_data_safe()
    with live_container.container():
        if data:
            st.warning(f"⏸️ Streaming in Pausa. Ultimo prezzo rilevato: **{data['bid']}** ({data['symbol']})")
        else:
            st.info("ℹ️ Attiva lo switch nella sidebar per connetterti a MT4.")

st.markdown("---")

# --- SEZIONE ANALISI AI (Statica, richiede ricalcolo manuale o trigger) ---
# L'analisi AI non deve ricalcolarsi ogni 0.5 secondi (troppo pesante), 
# la facciamo solo su richiesta o se abbiamo dati validi
if data and not streaming_active: # Mostriamo analisi solo se in pausa o statico
    with analysis_container:
        st.header("🧠 Analisi Intelligenza Artificiale")
        
        if st.button("Genera Analisi su Prezzo Attuale"):
            with st.spinner("Calcolo probabilità in corso..."):
                # 1. Recupero Storico per Training
                hist_data = load_historical_training_data("GC=F") # Usa Yahoo per lo storico profondo
                
                if hist_data is not None:
                    # 2. Training
                    df_ind = calculate_technical_indicators(hist_data)
                    X, y = simulate_historical_trades(df_ind)
                    model, scaler = train_model(X, y)
                    
                    # 3. Predizione basata sul PREZZO LIVE MT4
                    current_price = data['bid']
                    latest_atr = df_ind['ATR'].iloc[-1]
                    
                    # Scenario LONG
                    sl_long = current_price - (latest_atr * 1.5)
                    tp_long = current_price + (latest_atr * 2.5)
                    feat_long = generate_features(df_ind, current_price, sl_long, tp_long, 'long')
                    prob_long = model.predict_proba(scaler.transform([feat_long]))[0][1] * 100
                    
                    # Scenario SHORT
                    sl_short = current_price + (latest_atr * 1.5)
                    tp_short = current_price - (latest_atr * 2.5)
                    feat_short = generate_features(df_ind, current_price, sl_short, tp_short, 'short')
                    prob_short = model.predict_proba(scaler.transform([feat_short]))[0][1] * 100
                    
                    # Visualizzazione
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🐂 Scenario LONG")
                        st.metric("Probabilità AI", f"{prob_long:.1f}%")
                        st.write(f"**Entry:** {current_price}")
                        st.write(f"**SL:** {sl_long:.2f} | **TP:** {tp_long:.2f}")
                        if prob_long > 65: st.success("✅ Setup Forte")
                        elif prob_long > 50: st.warning("⚠️ Setup Incerto")
                        else: st.error("❌ Setup Debole")

                    with col2:
                        st.markdown("### 🐻 Scenario SHORT")
                        st.metric("Probabilità AI", f"{prob_short:.1f}%")
                        st.write(f"**Entry:** {current_price}")
                        st.write(f"**SL:** {sl_short:.2f} | **TP:** {tp_short:.2f}")
                        if prob_short > 65: st.success("✅ Setup Forte")
                        elif prob_short > 50: st.warning("⚠️ Setup Incerto")
                        else: st.error("❌ Setup Debole")
                        
                else:
                    st.error("Impossibile recuperare storico dati per l'addestramento.")

elif streaming_active:
    with analysis_container:
        st.info("💡 Per eseguire l'analisi AI dettagliata, metti in pausa lo streaming.")
