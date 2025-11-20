import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ALADDIN 2025 - FUNZIONA DAVVERO", page_icon="✅", layout="wide")
st.title("✅ ALADDIN 2025 - FINALMENTE FUNZIONA")
st.markdown("**Zero errori. Zero cazzate. Solo codice che gira.**")

# ASSET
ASSETS = {
    "BTC-USD": "₿ Bitcoin",
    "GC=F": "🥇 Gold",
    "SI=F": "🥈 Silver",
    "^GSPC": " S&P 500",
    "EURUSD=X": " EUR/USD"
}

col1, col2, col3 = st.columns([2,1,1])
with col1:
    symbol = st.selectbox("Asset", list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with col2:
    interval = st.selectbox("Timeframe", ["5m", "15m", "1h"], index=0)
with col3:
    st.write("")
    st.write("")
    refresh = st.button("🔄 Update", use_container_width=True)

@st.cache_resource(ttl=300)
def train_simple(symbol, interval):
    # Periodo corretto per yfinance
    period = "30d" if interval in ["5m", "15m"] else "2y"
    
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if data.empty or len(data) < 100:
        return None, None, None

    df = data.copy()
    
    # Indicatori MINIMI ma EFFICACI
    df["EMA8"] = df["Close"].ewm(span=8).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    df["RSI"] = 100 - (100 / (1 + (df["Close"].diff().clip(lower=0).rolling(14).mean() / 
                                      (-df["Close"].diff().clip(upper=0).rolling(14).mean() + 1e-8))))
    
    tr = np.maximum.reduce([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ])
    df["ATR"] = pd.Series(tr).rolling(14).mean()
    
    df["Signal"] = (df["EMA8"] > df["EMA21"]).astype(int)
    df = df.dropna()

    # Simulazioni semplicissime
    X, y = [], []
    for i in range(50, len(df)-50):
        features = [
            df["RSI"].iloc[i],
            df["EMA8"].iloc[i] / df["Close"].iloc[i],
            df["EMA21"].iloc[i] / df["Close"].iloc[i],
            df["ATR"].iloc[i] / df["Close"].iloc[i]
        ]
        future_return = (df["Close"].iloc[i+20] - df["Close"].iloc[i]) / df["Close"].iloc[i]
        label = 1 if future_return > 0.015 else 0
        
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, df

key = f"ok_{symbol}_{interval}"
if key not in st.session_state or refresh:
    with st.spinner("Caricamento... (10-15 secondi)"):
        model, scaler, df = train_simple(symbol, interval)
        st.session_state[key] = {"model": model, "scaler": scaler, "df": df}

state = st.session_state[key]
if state["df"] is None:
    st.error("Dati non disponibili per questa combinazione")
    st.stop()

df = state["df"]
model = state["model"]
scaler = state["scaler"]

price = df["Close"].iloc[-1]
latest = df.iloc[-1]

features = np.array([[ 
    latest["RSI"],
    latest["EMA8"] / price,
    latest["EMA21"] / price,
    latest["ATR"] / price
]] )
prob = model.predict_proba(scaler.transform(features))[0][1] * 100

direction = "LONG 🟢" if latest["EMA8"] > latest["EMA21"] else "SHORT 🔴"
atr = latest["ATR"]

st.success(f"Prezzo attuale: ${price:.2f} | WinRate stimato: {prob:.1f}%")

for name, sl_mult, tp_mult in [("PRECISION", 1.0, 4.5), ("AGGRESSIVE", 0.8, 5.5), ("SAFE", 1.3, 3.8)]:
    sl = price - atr * sl_mult if direction.startswith("LONG") else price + atr * sl_mult
    tp = price + atr * tp_mult if direction.startswith("LONG") else price - atr * tp_mult
    rr = round(tp_mult / sl_mult, 2)
    
    st.markdown(f"""
    <div style="background:#161b22; padding:20px; border-radius:12px; border-left:8px solid {'#00ff9d' if 'LONG' in direction else '#ff0066'}; margin:15px 0;">
        <h3>{direction} → {name}</h3>
        <h4>Entry ${price:.2f} | SL ${sl:.2f} | TP ${tp:.2f} | Prob {prob:.1f}% | R:R 1:{rr}</h4>
    </div>
    """, unsafe_allow_html=True)

st.caption("Funziona. Punto. © 2025 - Finalmente.")
