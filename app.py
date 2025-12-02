"""
Trading Predictor AI - MT4 Integration (Python Side)
Versione Finale: 2.0
"""
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
import os

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE PERCORSI ====================
# IMPORTANTE: Cambia questo percorso con la cartella 'Files' del tuo terminale MT4
# Esempio: C:/Users/NomeUtente/AppData/Roaming/MetaQuotes/Terminal/ID_TERMINALE/MQL4/Files
# Per testare facilmente, useremo una cartella C:/Temp/MT4_Bridge e useremo un symlink o copieremo i file
DEFAULT_BRIDGE_PATH = r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files"

# ==================== MT4 BRIDGE SYSTEM ====================

class MT4Bridge:
    def __init__(self, bridge_folder=DEFAULT_BRIDGE_PATH):
        self.bridge_folder = Path(bridge_folder)
        self.signals_file = self.bridge_folder / "signals.json"
        self.status_file = self.bridge_folder / "status.json"
        self.trades_file = self.bridge_folder / "trades.json"
        
        try:
            self.bridge_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f"Errore creazione cartella: {e}")
    
    def send_signal(self, signal_data):
        try:
            # Pulisci i tipi numpy per la serializzazione JSON
            for key, value in signal_data.items():
                if isinstance(value, (np.integer, np.int64)):
                    signal_data[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    signal_data[key] = float(value)

            signal = {
                "timestamp": datetime.datetime.now().isoformat(),
                "symbol": str(signal_data.get("symbol", "XAUUSD")),
                "direction": str(signal_data.get("direction", "BUY")),
                "entry": float(signal_data.get("entry", 0)),
                "stop_loss": float(signal_data.get("sl", 0)),
                "take_profit": float(signal_data.get("tp", 0)),
                "lot_size": float(signal_data.get("lot_size", 0.01)),
                "magic_number": int(signal_data.get("magic", 12345)),
                "command": "OPEN" # Comando esplicito per l'EA
            }
            
            with open(self.signals_file, 'w', encoding='utf-8') as f:
                json.dump(signal, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"❌ Errore invio: {str(e)}")
            return False
    
    def get_status(self):
        try:
            if self.status_file.exists():
                # Ritardo minimo per evitare conflitti di lettura/scrittura
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            return None
        return None
    
    def get_open_trades(self):
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            return []
        return []

    def clear_signal(self):
        try:
            if self.signals_file.exists():
                os.remove(self.signals_file)
            return True
        except:
            return False

# ==================== INDICATORS & AI ====================

def calculate_technical_indicators(df):
    df = df.copy()
    # Indicatori base
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
    
    # Bollinger Bands
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
    
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
    
    return df.dropna()

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    latest = df_ind.iloc[-1]
    features = [
        abs(entry - sl) / entry * 100, # SL Dist %
        abs(tp - entry) / entry * 100, # TP Dist %
        abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0, # RR
        1 if direction == 'long' else 0,
        latest['RSI'],
        latest['MACD'],
        latest['ATR'],
        (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        latest['Trend']
    ]
    return np.array(features, dtype=np.float32)

def simulate_training_data(df_ind, n_samples=300):
    # Generazione sintetica dati training basata su storico
    X, y = [], []
    for _ in range(n_samples):
        idx = np.random.randint(50, len(df_ind)-50)
        row = df_ind.iloc[idx]
        future = df_ind.iloc[idx+1:idx+20] # Guarda 20 candele avanti
        
        if len(future) < 5: continue
        
        # Logica semplificata: se il prezzo sale dell'ATR è un successo Long
        atr = row['ATR']
        if future['High'].max() > row['Close'] + atr:
            y.append(1) # Success Long
            X.append(generate_features(df_ind.iloc[:idx+1], row['Close'], row['Close']-atr, row['Close']+atr*2, 'long', 60))
        elif future['Low'].min() < row['Close'] - atr:
            y.append(1) # Success Short (etichettiamo 1 se la direzione è corretta)
            X.append(generate_features(df_ind.iloc[:idx+1], row['Close'], row['Close']+atr, row['Close']-atr*2, 'short', 60))
        else:
            y.append(0) # Choppy market
            X.append(generate_features(df_ind.iloc[:idx+1], row['Close'], row['Close']-atr, row['Close']+atr, 'long', 60))
            
    return np.array(X), np.array(y)

# ==================== DATA LOADING ====================

@st.cache_resource
def get_model_and_data(symbol, interval):
    data = yf.download(symbol, period='1y', interval=interval, progress=False)
    
    # Fix per yfinance MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    if len(data) < 100: return None, None, None
    
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_training_data(df_ind)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, df_ind

# ==================== MAIN UI ====================

st.set_page_config(page_title="AI Trader Bridge", layout="wide")
st.title("🤖 AI Trading Bridge - Python to MT4")

# Setup
if 'bridge' not in st.session_state:
    st.session_state.bridge = MT4Bridge()
    
# Sidebar Config
st.sidebar.header("Configurazione")
symbol_yf = st.sidebar.selectbox("Simbolo Yahoo", ["GC=F", "EURUSD=X", "GBPUSD=X", "BTC-USD"])
symbol_mt4 = st.sidebar.text_input("Simbolo MT4 (Esatto)", "XAUUSD" if "GC" in symbol_yf else "EURUSD")
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h"])

# Caricamento Modello
model, scaler, df = get_model_and_data(symbol_yf, interval)

if df is not None:
    current_price = df['Close'].iloc[-1]
    
    # Dashboard Prezzi
    c1, c2, c3 = st.columns(3)
    c1.metric("Prezzo Attuale", f"{current_price:.2f}")
    c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    c3.metric("ATR", f"{df['ATR'].iloc[-1]:.2f}")
    
    st.markdown("---")
    
    # Generatore Segnali
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("💡 Genera Segnale")
        direction = st.radio("Direzione", ["BUY", "SELL"], horizontal=True)
        
        atr = df['ATR'].iloc[-1]
        entry = st.number_input("Entry", value=float(current_price))
        
        if direction == "BUY":
            sl_def = entry - (atr * 1.5)
            tp_def = entry + (atr * 2.0)
        else:
            sl_def = entry + (atr * 1.5)
            tp_def = entry - (atr * 2.0)
            
        sl = st.number_input("Stop Loss", value=float(sl_def))
        tp = st.number_input("Take Profit", value=float(tp_def))
        lots = st.number_input("Lotti", value=0.01, step=0.01)
        
        # AI Check
        feat = generate_features(df, entry, sl, tp, 'long' if direction == "BUY" else 'short', 60)
        prob = model.predict_proba(scaler.transform([feat]))[0][1] * 100
        
        st.info(f"🤖 AI Confidence: **{prob:.1f}%**")
        
        if st.button("🚀 INVIA A MT4", type="primary"):
            signal = {
                "symbol": symbol_mt4,
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "lot_size": lots,
                "probability": prob
            }
            if st.session_state.bridge.send_signal(signal):
                st.success("Segnale salvato in JSON! In attesa che MT4 lo legga...")

    with col_r:
        st.subheader("📡 Stato MT4")
        if st.button("Aggiorna Stato"):
            st.rerun()
            
        status = st.session_state.bridge.get_status()
        if status:
            st.json(status)
            st.caption(f"Ultimo aggiornamento: {status.get('updated')}")
        else:
            st.warning("Nessun dato da MT4. Assicurati che l'EA sia attivo.")
            
        st.subheader("📋 Ordini Aperti")
        trades = st.session_state.bridge.get_open_trades()
        if trades:
            st.dataframe(trades)
        else:
            st.info("Nessun trade aperto.")

else:
    st.error("Impossibile scaricare i dati. Controlla la connessione o il simbolo.")
