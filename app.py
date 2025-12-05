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
        # Auto-detect MT4 path
        self.base_path = Path.home() / "AppData/Roaming/MetaQuotes/Terminal"
        
        if self.base_path.exists():
            terminals = [d for d in self.base_path.iterdir() if d.is_dir()]
            if terminals:
                self.files_path = terminals[0] / "MQL4" / "Files"
            else:
                self.files_path = Path("C:/MT4_Files")
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
            st.error(f"Error: {e}")
            return False

# ==================== ML FUNCTIONS ====================
def calculate_indicators(df):
    df = df.copy()
    
    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    
    # Trend
    df['Trend'] = (df['Close'] > df['EMA_50']).astype(int)
    
    return df.dropna()

def generate_features(df):
    latest = df.iloc[-1]
    
    features = np.array([
        latest['RSI'],
        latest['MACD'],
        latest['MACD_signal'],
        latest['ATR'],
        (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
        latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        latest['Trend']
    ])
    
    return features

def train_model(df, n_samples=300):
    X_list = []
    y_list = []
    
    for i in range(50, min(len(df) - 10, n_samples)):
        row = df.iloc[i]
        
        features = np.array([
            row['RSI'],
            row['MACD'],
            row['MACD_signal'],
            row['ATR'],
            (row['EMA_20'] - row['EMA_50']) / row['Close'] * 100,
            (row['Close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower']),
            row['Volume'] / row['Volume_MA'] if row['Volume_MA'] > 0 else 1.0,
            row['Trend']
        ])
        
        # Future price movement
        future_price = df.iloc[i+5:i+10]['Close'].mean()
        success = 1 if future_price > row['Close'] else 0
        
        X_list.append(features)
        y_list.append(success)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_probability(model, scaler, features):
    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

def calculate_lot_size(balance, risk_pct, entry, sl):
    risk_amount = balance * (risk_pct / 100)
    sl_distance = abs(entry - sl)
    sl_pips = sl_distance / 0.01
    
    if sl_pips <= 0:
        return 0.01
    
    lot_size = risk_amount / (sl_pips * 10)
    return max(0.01, min(round(lot_size, 2), 10.0))

# ==================== STREAMLIT APP ====================
st.set_page_config(page_title="AI Trading System", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stMetric {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white;}
    .stButton>button {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px; padding: 0.6rem 1.5rem;}
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Trading System + MT4")

# Initialize
if 'bridge' not in st.session_state:
    st.session_state.bridge = MT4Bridge()
    st.session_state.live_mode = False

bridge = st.session_state.bridge

# ==================== HEADER ====================
col1, col2, col3, col4 = st.columns(4)

status = bridge.get_status()
mt4_price = bridge.get_live_price()
connected = bridge.is_connected()

with col1:
    if connected:
        st.success("🟢 MT4 Connected")
    else:
        st.error("🔴 MT4 Disconnected")

with col2:
    if status:
        st.metric("💰 Balance", f"${status.get('balance', 0):.2f}")
    else:
        st.metric("💰 Balance", "N/A")

with col3:
    if mt4_price:
        st.metric("📊 Live Price", f"${mt4_price.get('bid', 0):.2f}")
    else:
        st.metric("📊 Live Price", "N/A")

with col4:
    if status:
        st.metric("📈 Open Trades", status.get('open_trades', 0))
    else:
        st.metric("📈 Open Trades", "N/A")

st.divider()

# ==================== CONFIGURATION ====================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    symbol = st.text_input("🎯 Symbol", "GC=F", help="GC=F (Gold), SI=F (Silver), BTC-USD")

with col2:
    timeframe = st.selectbox("⏱️ Timeframe", ['5m', '15m', '1h', '1d'], index=0)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

st.session_state.live_mode = st.checkbox("⚡ Auto-refresh with Live MT4 prices", value=st.session_state.live_mode)

# ==================== MAIN ANALYSIS ====================
if analyze_btn or st.session_state.live_mode:
    
    with st.spinner("🤖 Loading and analyzing data..."):
        try:
            # Map timeframe
            tf_map = {'5m': '5m', '15m': '15m', '1h': '1h', '1d': '1d'}
            period_map = {'5m': '5d', '15m': '5d', '1h': '60d', '1d': '1y'}
            
            interval = tf_map[timeframe]
            period = period_map[timeframe]
            
            # Load data
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if data.empty:
                st.error("❌ No data available")
            else:
                # Calculate indicators
                df = calculate_indicators(data)
                
                # Train model
                model, scaler = train_model(df)
                
                # Get current price
                if mt4_price and st.session_state.live_mode:
                    current_price = float(mt4_price.get('bid', df['Close'].iloc[-1]))
                    price_source = "MT4 Live"
                else:
                    current_price = float(df['Close'].iloc[-1])
                    price_source = "yfinance"
                
                st.success(f"✅ Analysis complete! Price: ${current_price:.2f} ({price_source})")
                
                # Display metrics
                st.subheader("📊 Technical Indicators")
                
                latest = df.iloc[-1]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    rsi = latest['RSI']
                    rsi_color = "🟢" if 30 <= rsi <= 70 else "🔴"
                    st.metric(f"{rsi_color} RSI", f"{rsi:.1f}")
                
                with col2:
                    st.metric("📏 ATR", f"{latest['ATR']:.2f}")
                
                with col3:
                    macd_signal = "📈 Bullish" if latest['MACD'] > latest['MACD_signal'] else "📉 Bearish"
                    st.metric("MACD", macd_signal)
                
                with col4:
                    trend = "🟢 Up" if latest['Trend'] == 1 else "🔴 Down"
                    st.metric("Trend", trend)
                
                with col5:
                    bb_pos = (current_price - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
                    bb_text = "High" if bb_pos > 0.8 else "Low" if bb_pos < 0.2 else "Mid"
                    st.metric("BB Position", bb_text)
                
                st.divider()
                
                # Generate AI signals
                st.subheader("🤖 AI Trade Signals")
                
                features = generate_features(df)
                ai_probability = predict_probability(model, scaler, features)
                
                atr = latest['ATR']
                
                col_long, col_short = st.columns(2)
                
                # LONG SIGNAL
                with col_long:
                    st.markdown("### 🟢 LONG Setup")
                    
                    entry_long = current_price
                    sl_long = current_price - atr * 1.5
                    tp_long = current_price + atr * 3.0
                    rr_long = abs(tp_long - entry_long) / abs(entry_long - sl_long)
                    
                    st.metric("AI Confidence", f"{ai_probability:.1f}%")
                    st.metric("Entry", f"${entry_long:.2f}")
                    st.metric("Stop Loss", f"${sl_long:.2f}")
                    st.metric("Take Profit", f"${tp_long:.2f}")
                    st.metric("Risk/Reward", f"{rr_long:.2f}x")
                    
                    col_cfg1, col_cfg2 = st.columns(2)
                    with col_cfg1:
                        balance_long = st.number_input("Balance ($)", 1000.0, 1000000.0, 10000.0, key="bal_long")
                    with col_cfg2:
                        risk_long = st.number_input("Risk (%)", 0.1, 10.0, 2.0, key="risk_long")
                    
                    lot_long = calculate_lot_size(balance_long, risk_long, entry_long, sl_long)
                    st.info(f"📦 Calculated Lot Size: **{lot_long:.2f}**")
                    
                    if st.button("📤 SEND LONG to MT4", use_container_width=True, key="send_long"):
                        signal = {
                            "symbol": mt4_price.get('symbol', 'XAUUSD') if mt4_price else 'XAUUSD',
                            "direction": "BUY",
                            "entry": entry_long,
                            "stop_loss": sl_long,
                            "take_profit": tp_long,
                            "lot_size": lot_long,
                            "comment": f"AI_LONG_{timeframe}"
                        }
                        
                        if bridge.send_signal(signal):
                            st.success("✅ LONG signal sent to MT4!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to send signal")
                
                # SHORT SIGNAL
                with col_short:
                    st.markdown("### 🔴 SHORT Setup")
                    
                    entry_short = current_price
                    sl_short = current_price + atr * 1.5
                    tp_short = current_price - atr * 3.0
                    rr_short = abs(tp_short - entry_short) / abs(entry_short - sl_short)
                    
                    st.metric("AI Confidence", f"{100 - ai_probability:.1f}%")
                    st.metric("Entry", f"${entry_short:.2f}")
                    st.metric("Stop Loss", f"${sl_short:.2f}")
                    st.metric("Take Profit", f"${tp_short:.2f}")
                    st.metric("Risk/Reward", f"{rr_short:.2f}x")
                    
                    col_cfg1, col_cfg2 = st.columns(2)
                    with col_cfg1:
                        balance_short = st.number_input("Balance ($)", 1000.0, 1000000.0, 10000.0, key="bal_short")
                    with col_cfg2:
                        risk_short = st.number_input("Risk (%)", 0.1, 10.0, 2.0, key="risk_short")
                    
                    lot_short = calculate_lot_size(balance_short, risk_short, entry_short, sl_short)
                    st.info(f"📦 Calculated Lot Size: **{lot_short:.2f}**")
                    
                    if st.button("📤 SEND SHORT to MT4", use_container_width=True, key="send_short"):
                        signal = {
                            "symbol": mt4_price.get('symbol', 'XAUUSD') if mt4_price else 'XAUUSD',
                            "direction": "SELL",
                            "entry": entry_short,
                            "stop_loss": sl_short,
                            "take_profit": tp_short,
                            "lot_size": lot_short,
                            "comment": f"AI_SHORT_{timeframe}"
                        }
                        
                        if bridge.send_signal(signal):
                            st.success("✅ SHORT signal sent to MT4!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to send signal")
                
                # Chart
                st.divider()
                st.subheader("📈 Price Chart")
                chart_data = df[['Close', 'EMA_20', 'EMA_50']].tail(100)
                st.line_chart(chart_data)
                
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Auto-refresh for live mode
if st.session_state.live_mode:
    time.sleep(2)
    st.rerun()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ System Info")
    
    st.info(f"**MT4 Files Path:**\n`{bridge.files_path}`")
    
    if st.button("🔄 Test Connection"):
        if bridge.is_connected():
            st.success("✅ MT4 is Connected!")
        else:
            st.error("❌ MT4 not connected")
            st.caption("Ensure EA is running on MT4")
    
    st.divider()
    
    st.caption("**Debug Info:**")
    st.caption(f"Status file: {'✅' if bridge.status_file.exists() else '❌'}")
    st.caption(f"Price file: {'✅' if bridge.price_file.exists() else '❌'}")

# Footer
st.divider()
st.caption("⚠️ **Disclaimer:** This is for educational purposes only. Trading involves substantial risk.")
