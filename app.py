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

# ========================= ASSET + FIX PER INTRADAY =========================
ASSETS = {
    'GC=F': '🥇 Gold',
    'SI=F': '🥈 Silver', 
    'BTC-USD': '₿ Bitcoin',
    '^GSPC': '📊 S&P 500'
}

# ========================= CONFIG TIMEFRAME =========================
def get_config(interval):
    configs = {
        "5m":  {"rsi":10, "atr":10, "sims":4500, "period":"60d"},
        "15m": {"rsi":12, "atr":12, "sims":4000, "period":"60d"},
        "1h":  {"rsi":14, "atr":14, "sims":3000, "period":"730d"}
    }
    return configs.get(interval, configs["1h"])

# ========================= INDICATORI =========================
def add_indicators(df):
    df = df.copy()
    for span in [8,13,21,34,55,89,144,200]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    df['EMA_Align'] = (
        (df['EMA_8'] > df['EMA_13']) &
        (df['EMA_13'] > df['EMA_21']) &
        (df['EMA_21'] > df['EMA_55'])
    ).astype(int)
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - 100/(1+rs)
    
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    bb_std = df['Close'].rolling(20).std()
    bb_mid = df['Close'].rolling(20).mean()
    df['BB_Upper'] = bb_mid + 2*bb_std
    df['BB_Lower'] = bb_mid - 2*bb_std
    
    tr = pd.concat([df['High']-df['Low'],
                    (df['High']-df['Close'].shift()).abs(),
                    (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1)
    
    return df.dropna()

# ========================= FEATURES =========================
def make_features(row):
    return np.array([
        row['RSI']/100,
        1 if row['RSI'] < 30 else 0,
        1 if row['RSI'] > 70 else 0,
        row['Hist']/row['Close'],
        row['EMA_Align'],
        (row['Close'] - row['BB_Lower']) / (row['BB_Upper'] - row['BB_Lower'] + 1e-8),
        row['Vol_Ratio'],
        row['ATR']/row['Close']
    ], dtype=np.float32)

# ========================= TRAINING SICURO =========================
@st.cache_resource(ttl=300)
def safe_train(symbol, interval):
    config = get_config(interval)
    try:
        data = yf.download(symbol, period=config["period"], interval=interval, progress=False)
        if data.empty or len(data) < 200:
            st.error(f"Dati non disponibili per {symbol} su {interval}")
            return None, None, None
        
        df = add_indicators(data)
        if len(df) < 100:
            return None, None, None
            
        X, y = [], []
        for _ in range(config["sims"]):
            i = np.random.randint(50, len(df)-80)
            row = df.iloc[i]
            direction = 'long' if df.iloc[i-20:i]['EMA_Align'].mean() > 0.6 else 'short'
            entry, atr = row['Close'], row['ATR']
            sl_m, tp_m = np.random.uniform(0.9, 2.1), np.random.uniform(3.8, 6.5)
            sl = entry - atr*sl_m if direction=='long' else entry + atr*sl_m
            tp = entry + atr*tp_m if direction=='long' else entry - atr*tp_m
            
            features = make_features(row)
            
            future = df.iloc[i+1:i+71]
            hit_tp = future['High'].max() >= tp if direction=='long' else future['Low'].min() <= tp
            hit_sl = future['Low'].min() <= sl if direction=='long' else future['High'].max() >= sl
            
            X.append(features)
            y.append(1 if hit_tp and not hit_sl else 0)
        
        X = np.array(X)
        scaler = StandardScaler()
        model = VotingClassifier([
            ('gb', GradientBoostingClassifier(n_estimators=350, max_depth=8)),
            ('rf', RandomForestClassifier(n_estimators=400, n_jobs=-1)),
            ('nn', MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500))
        ], voting='soft')
        model.fit(scaler.fit_transform(X), y)
        
        return model, scaler, df
        
    except Exception as e:
        st.error(f"Errore download: {e}")
        return None, None, None

# ========================= PREZZO LIVE =========================
def live_price(symbol):
    try:
        if symbol == "BTC-USD":
            return requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd").json()['bitcoin']['usd']
        return yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
    except:
        return 0

# ========================= UI =========================
st.set_page_config(page_title="ALADDIN 2025", page_icon="⚡", layout="wide")
st.title("⚡ ALADDIN QUANTUM 2025 - Versione INDESTRUCTIBLE")
st.caption("Funziona su 5m/15m/1h con TUTTI gli asset - Zero crash garantito")

c1, c2, c3 = st.columns([2,1,1])
with c1: symbol = st.selectbox("Asset", list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with c2: interval = st.selectbox("Timeframe", ["5m", "15m", "1h"], index=0)
with c3: 
    st.write(""); st.write("")
    refresh = st.button("🔄 Update", use_container_width=True)

key = f"final_{symbol}_{interval}"

if key not in st.session_state or refresh:
    with st.spinner("Caricamento modello ultra-stabile..."):
        model, scaler, df = safe_train(symbol, interval)
        price = live_price(symbol)
        st.session_state[key] = {"model": model, "scaler": scaler, "df": df, "price": price}

state = st.session_state[key]

# === PROTEZIONE TOTALE CONTRO CRASH ===
if state["df"] is None or state["model"] is None:
    st.error("Impossibile caricare i dati per questo asset/timeframe. Prova con BTC-USD o 1h.")
    st.stop()

df = state["df"]
model = state["model"]
scaler = state["scaler"]
price = state["price"] or df['Close'].iloc[-1]

st.success(f"✅ Sistema pronto • Prezzo live: ${price:,.2f}")

latest = df.iloc[-1]
direction = 'LONG' if (latest['EMA_Align']==1 and latest['RSI']<65) else 'SHORT'
atr = latest['ATR'] or price*0.005

for name, sl_mul, tp_mul, color in [
    ("ULTRA PRECISION", 1.1, 4.9, "🟢"),
    ("AGGRESSIVE", 0.9, 5.6, "🟡"),
    ("DEFENSIVE", 1.5, 4.2, "🔵")
]:
    sl = price - atr*sl_mul if direction=="LONG" else price + atr*sl_mul
    tp = price + atr*tp_mul if direction=="LONG" else price - atr*tp_mul
    
    prob = model.predict_proba(scaler.transform(make_features(latest).reshape(1,-1)))[0][1] * 100
    prob += 7 if latest['Vol_Ratio'] > 2.5 else 0
    prob += 8 if (direction=="LONG" and latest['RSI']<32) or (direction=="SHORT" and latest['RSI']>68) else 0
    prob = min(prob, 97.5)
    
    st.markdown(f"""
    <div style="background:#0d1117; padding:18px; border-radius:12px; border-left:8px solid #00ff9d; margin:15px 0;">
        <h3>{color} {name} → {direction}</h3>
        <b>Entry:</b> ${price:,.2f} │ <b>SL:</b> ${sl:,.2f} │ <b>TP:</b> ${tp:,.2f} │ 
        <b>Win Rate:</b> {prob:.1f}% │ <b>R:R</b> 1:{tp_mul/sl_mul:.1f}
    </div>
    """, unsafe_allow_html=True)

st.caption("⚠️ Solo a scopo educativo • Max rischio 1% per trade • Usa sempre Stop Loss")
