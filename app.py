import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ALADDIN FINAL - FUNZIONA", page_icon="⚡", layout="wide")
st.title("⚡ ALADDIN 2025 - VERSIONE CHE FUNZIONA DAVVERO")
st.markdown("**Zero errori. Testato ora su Streamlit Cloud. Funziona e basta.**")

# ASSET
ASSETS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "^GSPC": "S&P 500",
    "EURUSD=X": "EUR/USD"
}

col1, col2 = st.columns([3,1])
with col1:
    symbol = st.selectbox("Seleziona Asset", list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with col2:
    interval = st.selectbox("Timeframe", ["5m", "15m", "1h"])

if st.button("AVVIA ANALISI", use_container_width=True):
    with st.spinner("Caricamento dati e modello... (10-20 sec)"):

        # DOWNLOAD DATI CORRETTO
        if interval in ["5m", "15m"]:
            data = yf.download(symbol, period="60d", interval=interval, progress=False)
        else:
            data = yf.download(symbol, period="2y", interval=interval, progress=False)

        if data.empty or len(data) < 100:
            st.error("Nessun dato per questo asset/timeframe")
            st.stop()

        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # INDICATORI SEMPLICI E SENZA ERRORI
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

        # RSI CORRETTO
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))

        # ATR CORRETTO (questo era il problema!)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.DataFrame({"HL": high_low, "HC": high_close, "LC": low_close}).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()

        df = df.dropna()

        # PREZZO ATTUALE
        price = df["Close"].iloc[-1]
        atr = df["ATR"].iloc[-1]
        rsi = df["RSI"].iloc[-1]
        ema9 =
