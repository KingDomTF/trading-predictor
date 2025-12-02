import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import json
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE PATH ====================
# IMPORTANTE: Inserisci qui il percorso della cartella MQL4/Files del tuo MT4
# Esempio: "C:/Users/NomeUtente/AppData/Roaming/MetaQuotes/Terminal/ID_TERMINALE/MQL4/Files"
# Se usi C:/MT4_Bridge, devi abilitare le DLL in MT4 o usare i Link Simbolici.
BRIDGE_FOLDER = "C:/MT4_Bridge" 

# ==================== MT4 BRIDGE SYSTEM ====================
class MT4Bridge:
    """Gestisce comunicazione bidirezionale con MT4"""
    
    def __init__(self, bridge_folder):
        self.bridge_folder = Path(bridge_folder)
        self.signals_file = self.bridge_folder / "signals.json"
        self.status_file = self.bridge_folder / "status.json"
        self.trades_file = self.bridge_folder / "trades.json"
        self.heartbeat_file = self.bridge_folder / "heartbeat.json"
        
        # Crea cartella se non esiste
        try:
            self.bridge_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f"❌ Errore creazione cartella bridge: {e}")

    def send_signal(self, signal_data):
        """Invia segnale a MT4"""
        try:
            signal = {
                "timestamp": datetime.datetime.now().isoformat(),
                "symbol": signal_data.get("symbol", "XAUUSD"),
                "direction": signal_data.get("direction", "BUY"),
                "entry": float(signal_data.get("entry", 0)),
                "stop_loss": float(signal_data.get("sl", 0)),
                "take_profit": float(signal_data.get("tp", 0)),
                "lot_size": float(signal_data.get("lot_size", 0.01)),
                "probability": float(signal_data.get("probability", 0)),
                "ai_confidence": float(signal_data.get("ai_confidence", 0)),
                "risk_reward": float(signal_data.get("rr_ratio", 0)),
                "comment": signal_data.get("comment", "AI_Signal"),
                "magic_number": int(signal_data.get("magic", 12345)),
                "status": "PENDING"
            }
            
            with open(self.signals_file, 'w') as f:
                json.dump(signal, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Errore invio segnale: {e}")
            return False

    def get_status(self):
        """Legge status MT4"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None

    def get_open_trades(self):
        """Recupera trade aperti"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []

    def is_mt4_connected(self):
        """Verifica connessione MT4"""
        try:
            if self.heartbeat_file.exists():
                with open(self.heartbeat_file, 'r') as f:
                    data = json.load(f)
                    # Gestione parsing data semplice
                    last_beat_str = data.get('timestamp', '2000-01-01T00:00:00')
                    last_beat = datetime.datetime.fromisoformat(last_beat_str)
                    age = (datetime.datetime.now() - last_beat).total_seconds()
                    return age < 15  # Connesso se heartbeat < 15 secondi
        except Exception:
            pass
        return False

    def clear_signal(self):
        """Pulisce file segnale"""
        try:
            if self.signals_file.exists():
                self.signals_file.unlink()
            return True
        except:
            return False

# ==================== HELPER FUNCTIONS ====================
def convert_yfinance_to_mt4_symbol(yf_symbol):
    mapping = {
        'GC=F': 'XAUUSD', 'SI=F': 'XAGUSD', 'EURUSD=X': 'EURUSD',
        'GBPUSD=X': 'GBPUSD', 'USDJPY=X': 'USDJPY', 'BTC-USD': 'BTCUSD',
        '^GSPC': 'US500', '^DJI': 'US30',
    }
    return mapping.get(yf_symbol, yf_symbol.replace('=F', '').replace('=X', ''))

def calculate_lot_size(balance, risk_pct, entry, sl, pip_value=10):
    if entry == 0 or sl == 0: return 0.01
    risk_amount = balance * (risk_pct / 100)
    sl_distance = abs(entry - sl)
    # Approssimazione generica: assumiamo entry price standardizzato
    if sl_distance == 0: return 0.01
    
    # Formula semplificata (Risk / StopLossDistance) * Scaling
    # Nota: Questo richiede aggiustamenti per asset specifici
    lot_size = risk_amount / (sl_distance * 100) # Scaling factor generico
    
    return max(0.01, min(round(lot_size, 2), 10.0))

# ==================== TECHNICAL INDICATORS ====================
def calculate_technical_indicators(df):
    df = df.copy()
    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    # BB
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    # Volume & Trend
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
    
    return df.dropna()

# ==================== ML MODEL ====================
def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    latest = df_ind.iloc[-1]
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    
    features = {
        'sl_distance_pct': sl_distance,
        'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0,
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'atr': latest['ATR'],
        'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'trend': latest['Trend']
    }
    return np.array(list(features.values()), dtype=np.float32)

def simulate_historical_trades(df_ind, n_trades=500):
    X_list, y_list = [], []
    for _ in range(n_trades):
        idx = np.random.randint(50, len(df_ind) - 50)
        row = df_ind.iloc[idx]
        direction = np.random.choice(['long', 'short'])
        entry = row['Close']
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)
        
        if direction == 'long':
            sl = entry * (1 - sl_pct / 100)
            tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100)
            tp = entry * (1 - tp_pct / 100)
            
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60)
        
        # Simple Outcome simulation
        future_prices = df_ind.iloc[idx+1:idx+51]['Close'].values
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
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler

def predict_success(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

# ==================== DATA & SIGNALS ====================
def get_web_signals(symbol, df_ind):
    try:
        latest = df_ind.iloc[-1]
        current_price = latest['Close']
        atr = latest['ATR']
        trend = latest['Trend']
        rsi = latest['RSI']
        suggestions = []
        
        # Logica base segnali
        if rsi < 40 and trend == 1:
            suggestions.append({
                'Direction': 'Long', 'Entry': current_price,
                'SL': current_price - atr * 1.5, 'TP': current_price + atr * 3.0,
                'Probability': 75
            })
        if rsi > 60 and trend == 0:
            suggestions.append({
                'Direction': 'Short', 'Entry': current_price,
                'SL': current_price + atr * 1.5, 'TP': current_price - atr * 3.0,
                'Probability': 75
            })
        return suggestions
    except Exception:
        return []

@st.cache_data(ttl=300)
def load_data(symbol, interval):
    data = yf.download(symbol, period='1y', interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    if len(data) < 50: return None
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="AI Trade Bridge", layout="wide")

# CSS
st.markdown("""<style>
    .stMetric {background: #f0f2f6; padding: 10px; border-radius: 10px;}
    .trade-card {border-left: 5px solid gold; background: #2b2b2b; padding: 10px; margin-bottom: 10px;}
</style>""", unsafe_allow_html=True)

st.title("🤖 AI Trading Bridge + MT4")

if 'mt4_bridge' not in st.session_state:
    st.session_state.mt4_bridge = MT4Bridge(BRIDGE_FOLDER)
bridge = st.session_state.mt4_bridge

# Sidebar inputs
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Symbol", "GC=F")
with col2:
    interval = st.selectbox("Interval", ["1h", "4h", "1d"])
with col3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

# Main Logic
data = load_data(symbol, interval)

if data is not None:
    df_ind = calculate_technical_indicators(data)
    
    # Train Model (Simplified for speed in UI)
    if 'model' not in st.session_state:
        X, y = simulate_historical_trades(df_ind)
        model, scaler = train_model(X, y)
        st.session_state.model = model
        st.session_state.scaler = scaler
    
    model = st.session_state.model
    scaler = st.session_state.scaler

    # UI Layout
    col_main, col_mt4 = st.columns([2, 1])
    
    with col_main:
        st.subheader("📊 Market Analysis")
        curr_price = df_ind['Close'].iloc[-1]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Price", f"{curr_price:.2f}")
        col_b.metric("RSI", f"{df_ind['RSI'].iloc[-1]:.1f}")
        col_c.metric("Trend", "Bullish" if df_ind['Trend'].iloc[-1] == 1 else "Bearish")
        
        st.subheader("💡 AI Suggestions")
        signals = get_web_signals(symbol, df_ind)
        
        if signals:
            for i, sig in enumerate(signals):
                with st.container():
                    st.markdown(f"""
                    <div class='trade-card'>
                        <b>{sig['Direction'].upper()}</b> @ {sig['Entry']:.2f} | 
                        Prob: {sig['Probability']}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Analyze & Prepare {sig['Direction']}", key=f"btn_{i}"):
                        st.session_state.selected_signal = sig

    with col_mt4:
        st.subheader("🔗 MT4 Connection")
        status = bridge.get_status()
        is_conn = bridge.is_mt4_connected()
        
        if is_conn:
            st.success("Connected")
            if status:
                st.metric("Balance", f"${status.get('balance', 0):.2f}")
                st.metric("Equity", f"${status.get('equity', 0):.2f}")
        else:
            st.error("Disconnected (Check EA)")

        st.markdown("---")
        
        # Order Execution Panel
        if 'selected_signal' in st.session_state:
            sig = st.session_state.selected_signal
            st.write(f"**Prepare {sig['Direction']} on {symbol}**")
            
            with st.form("mt4_send"):
                mt4_sym = convert_yfinance_to_mt4_symbol(symbol)
                entry_inp = st.number_input("Entry", value=float(sig['Entry']))
                sl_inp = st.number_input("SL", value=float(sig['SL']))
                tp_inp = st.number_input("TP", value=float(sig['TP']))
                
                # Calcola AI Confidence live
                dir_code = 1 if sig['Direction'] == 'Long' else 0
                feats = generate_features(df_ind, entry_inp, sl_inp, tp_inp, sig['Direction'].lower(), 60)
                ai_conf = predict_success(model, scaler, feats)
                st.info(f"🤖 AI Confidence: {ai_conf:.1f}%")
                
                risk_pct = st.slider("Risk %", 0.5, 5.0, 1.0)
                balance = status.get('balance', 10000) if status else 10000
                calc_lot = calculate_lot_size(balance, risk_pct, entry_inp, sl_inp)
                lot_inp = st.number_input("Lot Size", value=calc_lot)
                
                submitted = st.form_submit_button("🚀 SEND SIGNAL TO MT4")
                
                if submitted:
                    payload = {
                        "symbol": mt4_sym,
                        "direction": "BUY" if sig['Direction'] == 'Long' else "SELL",
                        "entry": entry_inp, "sl": sl_inp, "tp": tp_inp,
                        "lot_size": lot_inp, "probability": sig['Probability'],
                        "ai_confidence": ai_conf,
                        "magic": 999
                    }
                    if bridge.send_signal(payload):
                        st.success("Signal Sent! Check MT4.")
                        time.sleep(1)
                        st.rerun()

else:
    st.warning("No data found. Check symbol.")
