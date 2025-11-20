import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import requests
import warnings
warnings.filterwarnings('ignore')

ASSETS = {
    'GC=F': 'Gold',
    'SI=F': 'Silver', 
    'BTC-USD': 'Bitcoin',
    '^GSPC': 'S&P 500'
}

# ========================
# 1. LIVE PRICE
# ========================
def get_live_data(symbol):
    try:
        if symbol == 'BTC-USD':
            data = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd").json()
            price = data['bitcoin']['usd']
            return {'price': float(price), 'source': 'CoinGecko'}
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not price:
            price = ticker.history(period='1d')['Close'].iloc[-1]
        return {'price': float(price), 'source': 'Yahoo Finance'}
    except:
        return {'price': 0.0, 'source': 'Error'}

# ========================
# 2. INDICATORI (ORA SICURI AL 100%)
# ========================
def calculate_advanced_indicators(df):
    df = df.copy()
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    # EMA
    for p in [9, 20, 50, 100, 200]:
        df[f'EMA_{p}'] = c.ewm(span=p, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands (assegnazione sicura)
    for p in [20, 50]:
        mid = c.rolling(p).mean()
        std = c.rolling(p).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        width = (upper - lower) / (mid + 1e-10)

        df = df.assign(**{
            f'BB_upper_{p}': upper,
            f'BB_lower_{p}': lower,
            f'BB_width_{p}': width
        })

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # Volume
    df['Volume_MA20'] = v.rolling(20).mean()
    df['Volume_ratio'] = v / (df['Volume_MA20'] + 1)

    # ADX semplificato ma efficace
    plus_di = (h.diff().clip(lower=0).rolling(14).mean() / df['ATR']).rolling(14).mean() * 100
    minus_di = (l.diff().clip(upper=0).abs().rolling(14).mean() / df['ATR']).rolling(14).mean() * 100
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['ADX'] = dx.rolling(14).mean()

    # EMA Alignment Score
    df['EMA_Align_Score'] = (
        (df['EMA_9'] > df['EMA_20']).astype(int) +
        (df['EMA_20'] > df['EMA_50']).astype(int) +
        (df['EMA_50'] > df['EMA_100']).astype(int) +
        (df['EMA_100'] > df['EMA_200']).astype(int)
    ) / 4.0

    # Regime trending
    df['Trending'] = (df['ADX'] > 25).astype(int)

    return df.dropna().reset_index(drop=True)

# ========================
# 3. CARICAMENTO DATI MULTI-TF
# ========================
@st.cache_data(ttl=60, show_spinner=False)
def load_multi_tf(symbol):
    periods = {'5m': '30d', '15m': '60d', '1h': '730d'}
    data = {}
    for tf, period in periods.items():
        try:
            df = yf.download(symbol, period=period, interval=tf, progress=False)
            if len(df) >= 300:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df = calculate_advanced_indicators(df)
                if len(df) > 100:
                    data[tf] = df
        except:
            pass
    return data

# ========================
# 4. MODELLO VELOCE E STABILE
# ========================
def train_model(df):
    X, y = [], []
    for _ in range(3000):
        i = np.random.randint(50, len(df)-80)
        row = df.iloc[i]

        direction = 'long' if row['EMA_Align_Score'] > 0.6 else 'short'
        entry = row['Close']
        atr = row['ATR']

        sl = entry - atr*1.2 if direction == 'long' else entry + atr*1.2
        tp = entry + atr*4.5 if direction == 'long' else entry - atr*4.5

        features = [
            row['RSI']/100,
            row['MACD_Hist']/entry,
            row['Volume_ratio'],
            row['EMA_Align_Score'],
            row['ADX']/100,
            row['BB_width_20'],
            row['Trending']
        ]

        future = df['Close'].iloc[i+1:i+81]
        hit_tp = (future >= tp).any() if direction == 'long' else (future <= tp).any()
        hit_sl = (future <= sl).any() if direction == 'long' else (future >= sl).any()
        success = 1 if hit_tp and not hit_sl else 0

        X.append(features)
        y.append(success)

    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
    ], voting='soft')

    model.fit(Xs, y)
    return model, scaler

# ========================
# 5. ANALISI FINALE
# ========================
def analyze(symbol):
    data = load_multi_tf(symbol)
    if len(data) < 3:
        return None

    preds = {}
    for tf, df in data.items():
        model, scaler = train_model(df)
        latest = df.iloc[-1]
        entry = latest['Close']
        feat = np.array([[
            latest['RSI']/100,
            latest['MACD_Hist']/entry,
            latest['Volume_ratio'],
            latest['EMA_Align_Score'],
            latest['ADX']/100,
            latest['BB_width_20'],
            latest['Trending']
        ]])
        prob_long = model.predict_proba(scaler.transform(feat))[0][1]
        preds[tf] = {'long': prob_long*100, 'short': (1-prob_long)*100}

    # Confluenza pesata
    score_long = preds['5m']['long']*0.5 + preds['15m']['long']*0.3 + preds['1h']['long']*0.2
    direction = 'LONG' if score_long > 50 else 'SHORT'
    confidence = round(abs(score_long - 50)*2 + 50, 1)

    live = get_live_data(symbol)
    atr = data['5m'].iloc[-1]['ATR'] if '5m' in data else 0

    return {
        'direction': direction,
        'confidence': min(99.9, confidence),
        'live_price': live['price'],
        'preds': preds,
        'atr': atr * 1.6
    }

# ========================
# 6. STREAMLIT APP
# ========================
st.set_page_config(page_title="ALADDIN BLACK 3.0", page_icon="⚡", layout="wide")
st.markdown("<h1 style='text-align:center; background:-webkit-linear-gradient(left, #ffd700, #ff6b6b); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:70px;'>⚡ ALADDIN BLACK 3.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.6rem; color:gold;'>Confluenza 5m • 15m • 1h → Segnali Ultra-Precisi</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col1:
    symbol = st.selectbox("Seleziona Asset", list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    go = st.button("ANALIZZA", use_container_width=True)

if go:
    with st.spinner("ALADDIN sta analizzando tutti i timeframe..."):
        result = analyze(symbol)
        if not result:
            st.error("Dati non sufficienti")
            st.stop()
        st.session_state.result = result

if 'result' in st.session_state:
    r = st.session_state.result
    color = "#00ff00" if r['direction'] == 'LONG' else "#ff0066"

    st.markdown(f"<h2 style='text-align:center; color:{color};'>{r['direction']} • {r['confidence']}% Confidence</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center;'>Prezzo Live: ${r['live_price']:.4f}</h3>", unsafe_allow_html=True)

    entry = r['live_price']
    sl = entry - r['atr'] if r['direction'] == 'LONG' else entry + r['atr']
    tp = entry + r['atr']*4.8 if r['direction'] == 'LONG' else entry - r['atr']*4.8
    rr = round(abs(tp-entry)/abs(entry-sl), 2)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e); padding:30px; border-radius:20px; color:white; text-align:center;">
        <h2 style="color:gold;">{r['direction']} ORA</h2>
        <h3>Entry: ${entry:.4f} • SL: ${sl:.4f} • TP: ${tp:.4f}</h3>
        <h3>R/R: {rr}:1</h3>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("5m", f"{r['preds']['5m']['long']:.0f}% LONG")
    with c2: st.metric("15m", f"{r['preds']['15m']['long']:.0f}% LONG")
    with c3: st.metric("1h", f"{r['preds']['1h']['long']:.0f}% LONG")

st.caption("ALADDIN BLACK 3.0 — Non è consiglio finanziario. Usa sempre lo stop loss.")
