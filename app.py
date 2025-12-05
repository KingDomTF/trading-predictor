import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import json
import time
from pathlib import Path
from datetime import datetime

# ==================== MT4 BRIDGE ====================
class MT4Bridge:
    def __init__(self):
        # Percorsi standard
        self.base_path = Path.home() / "AppData/Roaming/MetaQuotes/Terminal"
        
        # Tentativo di rilevamento automatico
        if self.base_path.exists():
            terminals = [d for d in self.base_path.iterdir() if d.is_dir()]
            if terminals:
                self.files_path = terminals[0] / "MQL4" / "Files"
            else:
                # Fallback path specifico se necessario
                self.files_path = Path(r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files")
        else:
            self.files_path = Path("C:/MT4_Files")
        
        self.status_file = self.files_path / "ai_status.json"
        self.price_file = self.files_path / "ai_price.json"
        self.signal_file = self.files_path / "ai_signal.json"
    
    def is_connected(self):
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    ts = data.get('timestamp', '')
                    last = datetime.strptime(ts, '%Y.%m.%d %H:%M:%S')
                    age = (datetime.now() - last).total_seconds()
                    return age < 5
        except:
            pass
        return False
    
    def get_status(self):
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def get_live_price(self):
        try:
            if self.price_file.exists():
                with open(self.price_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def send_signal(self, signal_data):
        try:
            with open(self.signal_file, 'w') as f:
                json.dump(signal_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Errore: {e}")
            return False

# ==================== ML FUNCTIONS ====================
def calculate_indicators(df):
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                   (-df['Close'].diff().clip(upper=0).rolling(14).mean())))
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    return df.dropna()

def train_simple_model(df):
    # Assicura di avere abbastanza dati
    if len(df) < 100:
        return None, None
        
    X = df[['EMA_20', 'RSI', 'ATR']].values[-100:]
    y = (df['Close'].shift(-1) > df['Close'])[-100:].astype(int).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict(model, scaler, features):
    if model is None:
        return 50.0
    return model.predict_proba(scaler.transform([features]))[0][1] * 100

# ==================== STREAMLIT APP ====================
st.set_page_config(page_title="AI Trading + MT4", layout="wide")

st.title("🎯 AI Trading Predictor + MT4")

# Initialize
if 'bridge' not in st.session_state:
    st.session_state.bridge = MT4Bridge()
    st.session_state.live_mode = False

bridge = st.session_state.bridge

# ==================== SIDEBAR CONFIG ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.info(f"📁 MT4 Files Path:\n{bridge.files_path}")
    
    if st.button("🔍 Test Connection"):
        if bridge.is_connected():
            st.success("✅ MT4 Connected!")
        else:
            st.error("❌ Not Connected")
    
    st.divider()
    
    symbol = st.text_input("Symbol", "GC=F")
    # UPDATED: Added 5m and 15m
    interval = st.selectbox("Timeframe", ['5m', '15m', '1h', '1d'], index=2)
    
    st.session_state.live_mode = st.checkbox("⚡ Live Price from MT4", 
                                              value=st.session_state.live_mode)

# ==================== MAIN AREA ====================
col1, col2, col3 = st.columns(3)

status = bridge.get_status()
mt4_price = bridge.get_live_price()

with col1:
    if bridge.is_connected():
        st.success("🟢 MT4 CONNECTED")
        if status:
            st.metric("Balance", f"${status.get('balance', 0):.2f}")
    else:
        st.error("🔴 MT4 DISCONNECTED")

with col2:
    if mt4_price and st.session_state.live_mode:
        st.metric("Live Price (MT4)", 
                  f"${mt4_price.get('bid', 0):.2f}",
                  f"Spread: {mt4_price.get('spread', 0):.1f}")
    else:
        st.info("📊 Use yfinance data")

with col3:
    if status:
        st.metric("Open Trades", status.get('open_trades', 0))

st.divider()

# ==================== DATA & ANALYSIS ====================
if st.button("🔄 Load & Analyze"):
    with st.spinner("Loading data..."):
        try:
            # UPDATED: Dynamic period selection (yfinance max 60d for intraday)
            if interval in ['5m', '15m']:
                period_lookup = '59d' 
            elif interval == '1h':
                period_lookup = '720d' # 2 years approx
            else:
                period_lookup = '5y'
            
            data = yf.download(symbol, period=period_lookup, interval=interval, progress=False)
            
            if data.empty:
                st.error("No data available from yfinance.")
            else:
                df = calculate_indicators(data)
                model, scaler = train_simple_model(df)
                
                if model is None:
                    st.error("Not enough data to train model (Need > 100 periods with indicators).")
                else:
                    if mt4_price and st.session_state.live_mode:
                        current_price = float(mt4_price.get('bid', df['Close'].iloc[-1]))
                        source = "MT4"
                    else:
                        current_price = float(df['Close'].iloc[-1])
                        source = "yfinance"
                    
                    st.success(f"✅ Data loaded! Current price: ${current_price:.2f} ({source})")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    latest = df.iloc[-1]
                    
                    with col1: st.metric("Price", f"${current_price:.2f}")
                    with col2: st.metric("RSI", f"{latest['RSI']:.1f}")
                    with col3: st.metric("ATR", f"{latest['ATR']:.2f}")
                    with col4: 
                        trend = "📈 UP" if latest['EMA_20'] > df['EMA_20'].iloc[-5] else "📉 DOWN"
                        st.metric("Trend", trend)
                    
                    st.subheader("🎯 AI Signals")
                    
                    atr = latest['ATR']
                    features = [latest['EMA_20'], latest['RSI'], latest['ATR']]
                    ai_prob = predict(model, scaler, features)
                    
                    # Logic for Buttons
                    col_sig, col_btn = st.columns([3, 1])
                    
                    # Long logic
                    if latest['RSI'] < 50:
                        with col_sig:
                            entry = current_price
                            sl = current_price - atr * 1.5
                            tp = current_price + atr * 3.0
                            st.success(f"**🟢 LONG SIGNAL** | AI Conf: {ai_prob:.1f}% | SL: {sl:.2f} | TP: {tp:.2f}")
                        with col_btn:
                            if st.button("📤 Send BUY", key="long"):
                                signal = {
                                    "symbol": mt4_price.get('symbol', 'XAUUSD') if mt4_price else 'XAUUSD',
                                    "direction": "BUY", "entry": entry, "stop_loss": sl, 
                                    "take_profit": tp, "lot_size": 0.01, "comment": "AI_LONG"
                                }
                                bridge.send_signal(signal)
                                st.toast("Signal Sent!")

                    # Short logic
                    elif latest['RSI'] > 50:
                        with col_sig:
                            entry = current_price
                            sl = current_price + atr * 1.5
                            tp = current_price - atr * 3.0
                            st.error(f"**🔴 SHORT SIGNAL** | AI Conf: {(100-ai_prob):.1f}% | SL: {sl:.2f} | TP: {tp:.2f}")
                        with col_btn:
                            if st.button("📤 Send SELL", key="short"):
                                signal = {
                                    "symbol": mt4_price.get('symbol', 'XAUUSD') if mt4_price else 'XAUUSD',
                                    "direction": "SELL", "entry": entry, "stop_loss": sl, 
                                    "take_profit": tp, "lot_size": 0.01, "comment": "AI_SHORT"
                                }
                                bridge.send_signal(signal)
                                st.toast("Signal Sent!")
                    
                    st.line_chart(df['Close'].tail(50))
                
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== LIVE PRICE UPDATER ====================
if st.session_state.live_mode and mt4_price:
    time.sleep(1)
    st.rerun()
