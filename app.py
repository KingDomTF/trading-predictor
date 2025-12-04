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
import os
from pathlib import Path

# Configurazione stile Palantir / Engineering
warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE ATOMICA (FIX PERCORSO) ====================
# Percorso ESATTO con la 'r' davanti per evitare errori di lettura Windows
MT4_DIRECTORY = r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files"

# ==================== MT4 BRIDGE SYSTEM (CORE LOGIC) ====================
class MT4Bridge:
    """
    Gestisce la comunicazione asincrona bidirezionale con l'Expert Advisor MT4.
    Include fix per percorsi Windows e gestione file fantasma.
    """
    
    def __init__(self, bridge_folder=MT4_DIRECTORY):
        self.bridge_folder = Path(bridge_folder)
        
        # 1. FORZATURA ESISTENZA CARTELLA
        if not self.bridge_folder.exists():
            try:
                self.bridge_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                st.error(f"❌ CRITICAL: Directory Bridge non accessibile: {self.bridge_folder}\nErrore: {e}")
                st.stop()

        # 2. MAPPING FILE (Corrisponde a Source 26-27 del file MQ4)
        self.signals_file = self.bridge_folder / "signals.json"
        self.status_file = self.bridge_folder / "status.json"
        self.trades_file = self.bridge_folder / "trades.json"
        self.heartbeat_file = self.bridge_folder / "heartbeat.json"
        self.live_price_file = self.bridge_folder / "live_price.json"
        
        # 3. BOOTSTRAPPING (Crea file vuoti se mancano per non far crashare l'app)
        self._init_phantom_files()

    def _init_phantom_files(self):
        """Genera file JSON vuoti se non esistono ancora"""
        try:
            if not self.status_file.exists():
                with open(self.status_file, 'w') as f: json.dump({}, f)
            if not self.trades_file.exists():
                with open(self.trades_file, 'w') as f: json.dump([], f)
            if not self.heartbeat_file.exists():
                hb = {"timestamp": datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"), "status": "waiting"}
                with open(self.heartbeat_file, 'w') as f: json.dump(hb, f)
        except:
            pass # Se fallisce, l'app proverà comunque a leggere

    def send_signal(self, signal_data):
        """
        Invia segnale a MT4.
        Formatta il JSON esattamente come richiesto da ParseSignalJSON (Source 147-149).
        """
        try:
            signal = {
                "symbol": signal_data.get("symbol", "XAUUSD"),
                "direction": signal_data.get("direction", "BUY"), # MQL4 controlla "BUY" o "SELL"
                "entry": float(signal_data.get("entry", 0)),
                "stop_loss": float(signal_data.get("sl", 0)),      # Key MQL4: stop_loss
                "take_profit": float(signal_data.get("tp", 0)),    # Key MQL4: take_profit
                "lot_size": float(signal_data.get("lot_size", 0.01)),
                "probability": float(signal_data.get("probability", 0)),
                "ai_confidence": float(signal_data.get("ai_confidence", 0)),
                "comment": signal_data.get("comment", "Palantir_AI_Bridge"),
                "status": "PENDING" # Trigger per l'esecuzione
            }
            
            # Scrittura con encoding utf-8
            with open(self.signals_file, 'w', encoding='utf-8') as f:
                json.dump(signal, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"❌ Errore Injection Segnale: {e}")
            return False

    def get_status(self):
        """Legge lo stato del conto (Source 119)"""
        return self._safe_read_json(self.status_file)

    def get_open_trades(self):
        """Legge la lista ordini aperti (Source 129)"""
        data = self._safe_read_json(self.trades_file)
        return data if isinstance(data, list) else []

    def get_live_price(self):
        """Legge il feed prezzi real-time (Source 43)"""
        return self._safe_read_json(self.live_price_file)

    def is_mt4_connected(self):
        """Verifica heartbeat (Source 55)"""
        data = self._safe_read_json(self.heartbeat_file)
        if not data: return False
        
        try:
            ts_str = data.get('timestamp', '')
            # Gestione flessibile formati data
            if '.' in ts_str:
                last_beat = datetime.datetime.strptime(ts_str, "%Y.%m.%d %H:%M:%S")
            else:
                last_beat = datetime.datetime.fromisoformat(ts_str)
            
            age = (datetime.datetime.now() - last_beat).total_seconds()
            return age < 15
        except:
            return False

    def _safe_read_json(self, filepath):
        """Lettura resiliente con retry logic per evitare conflitti I/O"""
        if not filepath.exists(): return None
        
        for _ in range(3): # 3 tentativi rapidi
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content: return None
                    return json.loads(content)
            except json.JSONDecodeError:
                time.sleep(0.05)
            except Exception:
                return None
        return None

# ==================== HELPER FUNCTIONS ====================
def convert_yfinance_to_mt4_symbol(yf_symbol):
    """Mapping tassonomia Yahoo Finance -> Broker MT4"""
    mapping = {
        'GC=F': 'XAUUSD', 'SI=F': 'XAGUSD', 'EURUSD=X': 'EURUSD',
        'GBPUSD=X': 'GBPUSD', 'USDJPY=X': 'USDJPY', 'BTC-USD': 'BTCUSD',
        '^GSPC': 'US500', '^DJI': 'US30', 'CL=F': 'WTI',
    }
    return mapping.get(yf_symbol, yf_symbol.replace('=F', '').replace('=X', ''))

def calculate_lot_size(balance, risk_pct, entry, sl):
    """Calcolo Money Management Semplificato"""
    if balance <= 0 or entry == 0: return 0.01
    
    risk_amount = balance * (risk_pct / 100)
    sl_distance = abs(entry - sl)
    
    if sl_distance == 0: return 0.01
    
    # Formula standard approssimata per XAUUSD/Forex (Contract size 1000/100000 mix)
    # Qui usiamo un divisore conservativo per evitare oversize
    try:
        # Questo è un calcolo grezzo, MT4 farà controlli finali (Source 161)
        lot_size = risk_amount / (sl_distance * 100) 
        return max(0.01, min(round(lot_size, 2), 50.0))
    except:
        return 0.01

def calculate_technical_indicators(df):
    """Calcola indicatori tecnici di base"""
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    return df.dropna()

# ==================== STREAMLIT INTERFACE LAYER (LA PARTE CHE MANCAVA) ====================
st.set_page_config(page_title="Palantir Trading Bridge", page_icon="👁️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {background: #0e1117;}
    .metric-card {background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid #FF4B4B;}
    h1 {font-family: 'Roboto', sans-serif;}
</style>
""", unsafe_allow_html=True)

st.title("👁️ Palantir AI • MT4 Neural Bridge")

# Inizializzazione Session State
if 'bridge' not in st.session_state:
    st.session_state.bridge = MT4Bridge()

bridge = st.session_state.bridge

# --- SIDEBAR CONTROLLI ---
with st.sidebar:
    st.header("📡 Data Feed")
    symbol_yf = st.text_input("Yahoo Symbol", "GC=F")
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2)
    
    st.divider()
    
    st.header("🔗 MT4 Connection")
    connected = bridge.is_mt4_connected()
    if connected:
        st.success("🟢 LINK ESTABLISHED")
        st.caption(f"Heartbeat OK")
    else:
        st.error("🔴 LINK DOWN")
        st.warning("Verifica che l'EA sia attivo su MT4")

    if st.button("Force Refresh"):
        st.rerun()

# --- MAIN DASHBOARD ---
col1, col2 = st.columns([3, 1])

# Variabili per i dati (inizializzate a default per evitare errori se il download fallisce)
last_price = 0.0
last_rsi = 50.0
last_atr = 1.0

with col1:
    # 1. DATA INGESTION
    try:
        with st.spinner("Downloading Market Data..."):
            data = yf.download(symbol_yf, period="1mo", interval=timeframe, progress=False)
        
        if len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            df_processed = calculate_technical_indicators(data)
            last_price = float(df_processed['Close'].iloc[-1])
            last_rsi = float(df_processed['RSI'].iloc[-1])
            last_atr = float(df_processed['ATR'].iloc[-1])
            
            # Recupero prezzo live da MT4 se disponibile
            mt4_live = bridge.get_live_price()
            if mt4_live and connected:
                mt4_bid = float(mt4_live.get('bid', 0))
                if mt4_bid > 0:
                    last_price = mt4_bid
                    st.caption(f"⚡ Live Pricing from MT4: {last_price}")
            
            st.line_chart(df_processed['Close'].tail(100))
        else:
            st.error("No Data Available from Yahoo Finance")
            
    except Exception as e:
        st.error(f"Data Pipeline Error: {e}")

with col2:
    # 2. MARKET METRICS
    st.markdown("### Market Intel")
    st.metric("Price", f"{last_price:.2f}")
    st.metric("RSI (14)", f"{last_rsi:.1f}", delta_color="inverse")
    st.metric("ATR Volatility", f"{last_atr:.2f}")

    status = bridge.get_status()
    if status:
        st.divider()
        st.markdown("### Account Telemetry")
        st.metric("Balance", f"${status.get('balance', 0):,.2f}")
        st.metric("Equity", f"${status.get('equity', 0):,.2f}")

# --- AI SIGNAL GENERATION & EXECUTION ---
st.divider()
st.subheader("🤖 Neural Decision Engine")

c1, c2, c3 = st.columns(3)

with c1:
    direction_input = st.radio("Signal Direction", ["LONG", "SHORT"], horizontal=True)
    
with c2:
    risk_input = st.slider("Risk % per Trade", 0.1, 5.0, 1.0)

with c3:
    # Auto-calcolo livelli basati su ATR
    atr_mult_sl = 1.5
    atr_mult_tp = 3.0
    
    if direction_input == "LONG":
        suggested_sl = last_price - (last_atr * atr_mult_sl)
        suggested_tp = last_price + (last_atr * atr_mult_tp)
    else:
        suggested_sl = last_price + (last_atr * atr_mult_sl)
        suggested_tp = last_price - (last_atr * atr_mult_tp)
        
    st.info(f"AI Suggested Levels (ATR):\nSL: {suggested_sl:.2f} | TP: {suggested_tp:.2f}")

# Manual Overrides
col_entry, col_sl, col_tp = st.columns(3)
entry_price = col_entry.number_input("Entry Price", value=float(last_price), format="%.2f")
sl_price = col_sl.number_input("Stop Loss", value=float(suggested_sl), format="%.2f")
tp_price = col_tp.number_input("Take Profit", value=float(suggested_tp), format="%.2f")

# Calcolo finale Lot Size
current_balance = status.get('balance', 10000) if status else 10000
lot_size = calculate_lot_size(current_balance, risk_input, entry_price, sl_price)

st.markdown(f"**Calculated Position Size:** `{lot_size} lots`")

# --- EXECUTION BUTTON ---
execute_col, _ = st.columns([1, 2])

if execute_col.button("🚀 EXECUTE ALPHA PROTOCOL", type="primary", use_container_width=True):
    if not connected:
        st.error("❌ ABORT: MT4 Uplink Disconnected")
    else:
        mt4_sym = convert_yfinance_to_mt4_symbol(symbol_yf)
        
        # MQL4 Expects "BUY" or "SELL" explicitly
        mql4_direction = "BUY" if direction_input == "LONG" else "SELL"
        
        payload = {
            "symbol": mt4_sym,
            "direction": mql4_direction,
            "entry": entry_price,
            "sl": sl_price, # MQL4 source 154-155 uses this logic
            "tp": tp_price,
            "lot_size": lot_size,
            "probability": 85.5, 
            "ai_confidence": 92.0,
            "comment": f"Palantir_AI_{int(time.time())}",
            "magic": 12345
        }
        
        with st.spinner("Transmitting Signal Packet..."):
            success = bridge.send_signal(payload)
            time.sleep(1) 
            
        if success:
            st.success(f"✅ Signal Injected for {mt4_sym} ({mql4_direction})")
        else:
            st.error("❌ Signal Injection Failed")

# --- LIVE TRADES MONITOR ---
st.divider()
st.subheader("📋 Active Operations")
trades = bridge.get_open_trades()

if trades:
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # Mostra solo colonne utili se esistono
        cols = ['ticket', 'symbol', 'type', 'lots', 'profit', 'open_price']
        valid_cols = [c for c in cols if c in trades_df.columns]
        st.dataframe(trades_df[valid_cols], use_container_width=True)
    else:
        st.caption("No active operations.")
else:
    st.caption("No active operations in the theater.")
