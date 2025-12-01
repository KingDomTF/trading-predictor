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

# ==================== CONFIGURAZIONE BRIDGE MT4 ====================
# Percorso standard (da aggiornare con il tuo percorso reale se diverso)
DEFAULT_MT4_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\COMMON\Files" 
DATA_FILENAME = "mt4_market_data.csv"
SIGNAL_FILENAME = "py_signal.csv"

# ==================== FUNZIONI DI COMUNICAZIONE ====================
def get_mt4_live_price(symbol_mt4_name):
    """Legge il prezzo reale scritto dall'EA di MT4."""
    path = st.session_state.get('mt4_path', DEFAULT_MT4_PATH)
    file_path = os.path.join(path, DATA_FILENAME)
    
    if not os.path.exists(file_path):
        return None, 0.0
    
    try:
        # Legge il file CSV: Symbol, Bid, Ask, Time
        # Esempio contenuto: XAUUSD,2650.50,2650.80,123456789
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows: return None, 0.0
            
            # Cerca l'ultima riga corrispondente al simbolo
            # MT4 potrebbe scrivere più simboli nello stesso file o sovrascrivere
            # In questo EA semplice, sovrascriviamo, quindi leggiamo l'unica riga
            for row in rows:
                if len(row) >= 3:
                    if row[0] == symbol_mt4_name:
                        return float(row[1]), float(row[2]) # Bid, Ask
            return None, 0.0
    except Exception:
        return None, 0.0

def send_signal_to_mt4(symbol, direction, entry, sl, tp, volume=0.1):
    """Invia ordine a MT4."""
    try:
        # Pulizia simbolo
        mt4_symbol = symbol
        if "GC" in symbol or "XAU" in symbol: mt4_symbol = "XAUUSD" # Adatta al tuo broker
        elif "EUR" in symbol: mt4_symbol = "EURUSD"
        
        op_type = 0 if direction.lower() in ['long', 'buy'] else 1
        line = f"{mt4_symbol},{op_type},{volume},{sl},{tp}"
        
        file_path = os.path.join(st.session_state.get('mt4_path', DEFAULT_MT4_PATH), SIGNAL_FILENAME)
        
        with open(file_path, "w") as f:
            f.write(line)
        return True, f"Segnale inviato per {mt4_symbol}"
    except Exception as e:
        return False, str(e)

# ==================== FUNZIONI CORE AI & ANALISI ====================
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
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
    
    # Bollinger Bands (Aggiunto per completezza feature)
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    df = df.dropna()
    return df

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    latest = df_ind.iloc[-1]
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    features = {
        'sl_distance_pct': sl_distance, 'tp_distance_pct': abs(tp - entry) / entry * 100,
        'rr_ratio': rr_ratio, 'direction': 1 if direction == 'long' else 0, 'main_tf': 60,
        'rsi': latest['RSI'], 'macd': latest['MACD'], 'macd_signal': latest['MACD_signal'],
        'atr': latest['ATR'], 'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'bb_position': latest['BB_position'], 
        'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        'price_change': latest['Price_Change'] * 100, 'trend': latest['Trend']
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
            
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60)
        future_prices = df_ind.iloc[idx+1:idx+40]['Close'].values # 40 candele future
        
        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp)
                hit_sl = np.any(future_prices <= sl)
            else:
                hit_tp = np.any(future_prices <= tp)
                hit_sl = np.any(future_prices >= sl)
            
            # Se prende TP prima di SL o prende TP senza SL
            success = 1 if hit_tp and not hit_sl else 0
            # Raffinamento: se prende entrambi, controlla chi viene prima (qui semplificato)
            
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
def load_historical_data(symbol, interval='1h'):
    """Usa yfinance SOLO per lo storico (training), non per il prezzo live."""
    try:
        # Mapping per yfinance se necessario
        yf_symbol = symbol
        if symbol == "XAUUSD": yf_symbol = "GC=F"
        
        data = yf.download(yf_symbol, period='1y', interval=interval, progress=False)
        if len(data) < 100: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return None

# ==================== INTERFACCIA STREAMLIT ====================
st.set_page_config(page_title="MT4 Live Bridge AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 10px; border: 1px solid #d1d5db; }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .live-badge { background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .delayed-badge { background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurazione")
    mt4_path_input = st.text_input("Percorso Cartella 'Files' MT4", value=DEFAULT_MT4_PATH)
    if mt4_path_input: st.session_state['mt4_path'] = mt4_path_input
    
    st.info("""
    **Istruzioni:**
    1. Incolla il percorso 'Files' di MT4 qui sopra.
    2. Assicurati che l'EA 'PyBridge' sia attivo sul grafico MT4.
    3. L'EA scriverà i prezzi reali nel file e leggerà i segnali.
    """)

st.title("⚡ MT4 Live Bridge AI - 100% Real Time")

# --- SEZIONE DATI LIVE DA MT4 ---
st.markdown("### 🔴 Prezzi Real-Time (Dal Broker)")

col1, col2, col3, col4 = st.columns(4)
mt4_assets = ["XAUUSD", "EURUSD", "BTCUSD", "US500"] # Simboli come in MT4

# Contenitore per aggiornamento automatico (simulato col bottone per efficienza Streamlit)
if st.button("🔄 Aggiorna Prezzi Live da MT4", type="primary", use_container_width=True):
    pass # Il rerun ricaricherà i dati

live_prices = {}

for i, asset in enumerate(mt4_assets):
    bid, ask = get_mt4_live_price(asset)
    live_prices[asset] = bid
    
    col = [col1, col2, col3, col4][i]
    with col:
        if bid is not None and bid > 0:
            st.markdown(f"**{asset}** <span class='live-badge'>LIVE MT4</span>", unsafe_allow_html=True)
            st.metric("Bid Price", f"{bid}")
            st.caption(f"Spread: {((ask-bid)*100):.2f} pts")
        else:
            st.markdown(f"**{asset}** <span class='delayed-badge'>OFFLINE</span>", unsafe_allow_html=True)
            st.warning("EA non attivo o simbolo errato")

st.markdown("---")

# --- ANALISI STRUMENTO ---
selected_asset = st.selectbox("Seleziona Asset per Analisi AI", mt4_assets)
current_live_price = live_prices.get(selected_asset, 0)

if st.button(f"🧠 Analizza {selected_asset}"):
    if current_live_price == 0:
        st.error("❌ Impossibile analizzare: Dati MT4 non rilevati. Attiva l'EA su MT4!")
    else:
        with st.spinner("Caricamento storico e calcolo AI..."):
            # 1. Carica storico (usiamo yfinance solo per i dati passati per addestrare il modello)
            hist_data = load_historical_data(selected_asset)
            
            if hist_data is not None:
                # 2. Addestra AI
                df_ind = calculate_technical_indicators(hist_data)
                X, y = simulate_historical_trades(df_ind)
                model, scaler = train_model(X, y)
                
                latest = df_ind.iloc[-1]
                atr = latest['ATR']
                
                # 3. Genera Setup usando il PREZZO LIVE DI MT4
                st.success(f"✅ Analisi completata su prezzo LIVE: {current_live_price}")
                
                c1, c2 = st.columns(2)
                
                # Setup LONG
                sl_long = current_live_price - (atr * 1.5)
                tp_long = current_live_price + (atr * 2.5)
                feat_long = generate_features(df_ind, current_live_price, sl_long, tp_long, 'long', 60)
                prob_long = model.predict_proba(scaler.transform([feat_long]))[0][1] * 100
                
                # Setup SHORT
                sl_short = current_live_price + (atr * 1.5)
                tp_short = current_live_price - (atr * 2.5)
                feat_short = generate_features(df_ind, current_live_price, sl_short, tp_short, 'short', 60)
                prob_short = model.predict_proba(scaler.transform([feat_short]))[0][1] * 100
                
                with c1:
                    st.markdown(f"### 🐂 LONG SETUP ({prob_long:.1f}%)")
                    st.write(f"Entry: **{current_live_price}**")
                    st.write(f"SL: {sl_long:.2f} | TP: {tp_long:.2f}")
                    if prob_long > 60:
                        if st.button("🚀 APRI BUY SU MT4", key="buy_btn"):
                            ok, msg = send_signal_to_mt4(selected_asset, "BUY", current_live_price, sl_long, tp_long)
                            if ok: st.toast(msg, icon="✅")
                            else: st.error(msg)
                            
                with c2:
                    st.markdown(f"### 🐻 SHORT SETUP ({prob_short:.1f}%)")
                    st.write(f"Entry: **{current_live_price}**")
                    st.write(f"SL: {sl_short:.2f} | TP: {tp_short:.2f}")
                    if prob_short > 60:
                        if st.button("🚀 APRI SELL SU MT4", key="sell_btn"):
                            ok, msg = send_signal_to_mt4(selected_asset, "SELL", current_live_price, sl_short, tp_short)
                            if ok: st.toast(msg, icon="✅")
                            else: st.error(msg)
                            
            else:
                st.error("Dati storici insufficienti per addestramento modello.")
