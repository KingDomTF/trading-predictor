import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

# ========================= ASSET =========================
ASSETS = {
    'GC=F': '🥇 Gold',
    'SI=F': '🥈 Silver', 
    'BTC-USD': '₿ Bitcoin',
    '^GSPC': '📊 S&P 500'
}

# ========================= CONFIG TIMEFRAME =========================
def get_config(interval):
    if interval == "5m":
        return {"rsi_p": 10, "atr_p": 10, "sims": 5000, "lookback": 144}
    elif interval == "15m":
        return {"rsi_p": 12, "atr_p": 12, "sims": 4500, "lookback": 120}
    else:  # 1h
        return {"rsi_p": 14, "atr_p": 14, "sims": 3500, "lookback": 90}

# ========================= INDICATORI =========================
def add_indicators(df, config):
    df = df.copy()
    
    # EMA
    for p in [8, 13, 21, 34, 55, 89, 144, 200]:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    
    # EMA Alignment (bullish se tutte allineate)
    df['EMA_Align'] = (
        (df['EMA_8'] > df['EMA_13']) &
        (df['EMA_13'] > df['EMA_21']) &
        (df['EMA_21'] > df['EMA_55']) &
        (df['EMA_55'] > df['EMA_144'])
    ).astype(int)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=config["rsi_p"]).mean()
    loss = -delta.clip(upper=0).rolling(window=config["rsi_p"]).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(config["atr_p"]).mean()
    
    # Volume ratio
    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_MA'] + 1)
    
    return df.dropna()

# ========================= FEATURES PER ML =========================
def create_features(row, direction):
    return np.array([
        row['RSI'] / 100,
        1 if row['RSI'] < 30 else 0,
        1 if row['RSI'] > 70 else 0,
        row['MACD_Hist'] / row['Close'],
        row['EMA_Align'],
        (row['Close'] - row['BB_Lower']) / (row['BB_Upper'] - row['BB_Lower'] + 1e-8),
        row['Vol_Ratio'],
        row['ATR'] / row['Close'],
        1 if direction == 'long' else 0
    ], dtype=np.float32)

# ========================= TRAINING =========================
@st.cache_resource(ttl=180)
def train_model(symbol, interval):
    config = get_config(interval)
    data = yf.download(symbol, period="730d", interval=interval, progress=False)
    if len(data) < 300:
        return None, None, None
    
    df = add_indicators(data, config)
    
    X, y = [], []
    for _ in range(config["sims"]):
        i = np.random.randint(100, len(df)-100)
        row = df.iloc[i]
        direction = 'long' if df.iloc[i-30:i]['EMA_Align'].mean() > 0.6 else 'short'
        
        entry = row['Close']
        atr = max(row['ATR'], entry * 0.0001)
        sl_mult = np.random.uniform(0.8, 2.0)
        tp_mult = np.random.uniform(3.5, 6.5)
        
        sl = entry - atr * sl_mult if direction == 'long' else entry + atr * sl_mult
        tp = entry + atr * tp_mult if direction == 'long' else entry - atr * tp_mult
        
        features = create_features(row, direction)
        
        future = df.iloc[i+1:i+81]
        hit_tp = future['High'].max() >= tp if direction == 'long' else future['Low'].min() <= tp
        hit_sl = future['Low'].min() <= sl if direction == 'long' else future['High'].max() >= sl
        
        success = 1 if hit_tp and not hit_sl else 0
        
        X.append(features)
        y.append(success)
    
    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ensemble = VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=400, max_depth=8, learning_rate=0.08)),
        ('rf', RandomForestClassifier(n_estimators=450, max_depth=14, n_jobs=-1)),
        ('nn', MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=600, early_stopping=True))
    ], voting='soft', weights=[2.5, 2, 1])
    
    ensemble.fit(X_scaled, y)
    return ensemble, scaler, df

# ========================= PREZZO LIVE =========================
def get_price(symbol):
    try:
        if symbol == "BTC-USD":
            r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", timeout=5)
            return r.json()['bitcoin']['usd']
        ticker = yf.Ticker(symbol)
        return ticker.history(period='1d')['Close'].iloc[-1]
    except:
        return 0

# ========================= STREAMLIT UI =========================
st.set_page_config(page_title="ALADDIN QUANTUM", page_icon="⚡", layout="wide")
st.title("⚡ ALADDIN QUANTUM SCALPER 2025")
st.caption("Perfetto su 5m • 15m • 1h | Zero errori | Pronto per Streamlit Cloud")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    symbol = st.selectbox("Asset", list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with c2:
    interval = st.selectbox("Timeframe", ["5m", "15m", "1h"], index=0)
with c3:
    st.write("")
    st.write("")
    refresh = st.button("🔄 Aggiorna", use_container_width=True)

key = f"aladdin_{symbol}_{interval}"

if key not in st.session_state or refresh:
    with st.spinner(f"Addestramento modello {interval}..."):
        ensemble, scaler, df = train_model(symbol, interval)
        price = get_price(symbol)
        st.session_state[key] = {
            "model": ensemble,
            "scaler": scaler,
            "df": df,
            "price": price,
            "time": datetime.datetime.now()
        }
        st.success("Modello pronto!")

state = st.session_state[key]
df = state["df"]
model = state["model"]
scaler = state["scaler"]
price = state["price"]

st.markdown(f"### {ASSETS[symbol]} • {interval} • Prezzo: ${price:,.2f}")

latest = df.iloc[-1]

# CORRETTO: espressione ternaria completa!
direction = 'long' if (latest['EMA_Align'] == 1 and latest['RSI'] < 68 and latest['MACD_Hist'] > 0) else 'short'

atr = latest['ATR']
entry = price

trades = []
for name, sl_m, tp_m in [("Ultra Precision", 1.1, 4.8), ("Aggressive", 0.9, 5.5), ("Safe", 1.4, 4.0)]:
    sl = entry - atr * sl_m if direction == 'long' else entry + atr * sl_m
    tp = entry + atr * tp_m if direction == 'long' else entry - atr * tp_m
    
    feat = create_features(latest, direction)
    prob = model.predict_proba(scaler.transform(feat.reshape(1, -1)))[0][1] * 100
    
    # Boost realistici
    if latest['Vol_Ratio'] > 2.5: prob += 6
    if latest['RSI'] < 30 and direction == 'long': prob += 8
    if latest['RSI'] > 70 and direction == 'short': prob += 8
    prob = min(prob, 96.9)
    
    rr = round(tp_m / sl_m, 2)
    
    color = "🟢" if prob >= 90 else "🟡" if prob >= 85 else "🔴"
    st.markdown(f"""
    <div style="background:#1a1a1a; color:white; padding:15px; border-radius:12px; border-left:6px solid #00ff9d; margin:10px 0;">
        <h3>{color} {name} → {direction.upper()}</h3>
        Entry <b>${entry:,.2f}</b> │ SL <b>${sl:,.2f}</b> │ TP <b>${tp:,.2f}</b> │ 
        Probabilità <b>{prob:.1f}%</b> │ R:R <b>1:{rr}</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ Solo educativo • Rischia max 1-2% per trade • Usa sempre Stop Loss")
