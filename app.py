# app.py
# -*- coding: utf-8 -*-

import time
import warnings
warnings.filterwarnings("ignore")

# ---- Streamlit DEVE essere importato e configurato subito ----
import streamlit as st

st.set_page_config(
    page_title="Scalping AI System - Ultra High Frequency",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Altre librerie ----
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ==================== CONFIGURAZIONE MT4/MT5 ====================
def check_mt_connection():
    """Verifica disponibilità connessione MT4/MT5."""
    try:
        import MetaTrader5 as mt5  # noqa: F401
        return True, "MT5"
    except Exception:
        pass

    try:
        import MT4  # noqa: F401
        return True, "MT4"
    except Exception:
        pass

    return False, None


MT_AVAILABLE, MT_TYPE = check_mt_connection()


def get_mt_realtime_price(symbol: str):
    """Ottiene prezzo real-time da MT4/MT5 se disponibile."""
    if not MT_AVAILABLE:
        return None

    try:
        if MT_TYPE == "MT5":
            import MetaTrader5 as mt5

            if not mt5.initialize():
                return None

            # Mappa simboli Yahoo a MT5
            mt5_symbol_map = {
                "GC=F": "XAUUSD",
                "SI=F": "XAGUSD",
                "EURUSD=X": "EURUSD",
                "^GSPC": "US500",
                "BTC-USD": "BTCUSD",
            }
            mt5_symbol = mt5_symbol_map.get(symbol, symbol)

            # Assicurati che il simbolo sia selezionato
            info = mt5.symbol_info(mt5_symbol)
            if info is None or not info.visible:
                mt5.symbol_select(mt5_symbol, True)

            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is not None:
                return {
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last": tick.last if hasattr(tick, "last") else (tick.bid + tick.ask) / 2.0,
                    "spread": (tick.ask - tick.bid) if (tick.ask and tick.bid) else None,
                    "time": datetime.datetime.fromtimestamp(tick.time),
                }

        elif MT_TYPE == "MT4":
            # Placeholder per eventuale integrazione MT4
            pass

    except Exception as e:
        st.warning(f"Errore connessione MT: {e}")

    return None


def get_realtime_price(symbol: str):
    """Ottiene prezzo più aggiornato possibile (MT4/MT5 o Yahoo Finance)."""
    # 1) Prova MT4/MT5
    mt_price = get_mt_realtime_price(symbol)
    if mt_price:
        return (
            mt_price["last"],
            mt_price["bid"],
            mt_price["ask"],
            mt_price["spread"],
            "MT5/MT4",
            mt_price["time"],
        )

    # 2) Fallback Yahoo Finance (1m poi 5m)
    try:
        ticker = yf.Ticker(symbol)

        for period, interval, source in [("1d", "1m", "Yahoo-1m"), ("5d", "5m", "Yahoo-5m")]:
            data = ticker.history(period=period, interval=interval)
            if not data.empty:
                last_price = float(data["Close"].iloc[-1])
                timestamp = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else data.index[-1]

                # Spread stimato: forex più stretto
                spread_pct = 0.0001 if "=X" in symbol else 0.0005  # ~0.01% FX, ~0.05% altri
                spread = last_price * spread_pct
                bid = last_price - spread / 2.0
                ask = last_price + spread / 2.0

                return last_price, bid, ask, spread, source, timestamp

    except Exception as e:
        st.error(f"Errore recupero prezzo real-time: {e}")

    return None, None, None, None, None, None


# ==================== STRATEGIA SCALPING ====================
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")


def calculate_scalping_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola indicatori specifici per scalping (1m-5m)."""
    df = df.copy()

    # EMAs
    df["EMA_5"] = df["Close"].ewm(span=5).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()

    # RSI
    df["RSI_7"] = _rsi(df["Close"], 7)
    df["RSI"] = _rsi(df["Close"], 14)

    # MACD (scalping 5/13/5)
    exp1 = df["Close"].ewm(span=5).mean()
    exp2 = df["Close"].ewm(span=13).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=5).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # Bollinger (10)
    df["BB_middle"] = df["Close"].rolling(window=10).mean()
    bb_std = df["Close"].rolling(window=10).std()
    df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
    df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

    # ATR (7)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR"] = true_range.rolling(7).mean()
    df["ATR_pct"] = (df["ATR"] / df["Close"]) * 100

    # Volume
    df["Volume_MA"] = df["Volume"].rolling(window=10).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_MA"]

    # Momentum / velocità
    df["Price_Change"] = df["Close"].pct_change()
    df["Price_Velocity"] = df["Price_Change"].rolling(3).mean()
    df["Price_Acceleration"] = df["Price_Velocity"].diff()

    # Trend micro (5 periodi)
    df["Trend_micro"] = df["Close"].rolling(window=5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)

    # Pivot/Support/Resistance
    df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["R1"] = 2 * df["Pivot"] - df["Low"]
    df["S1"] = 2 * df["Pivot"] - df["High"]

    # EMA squeeze
    df["EMA_squeeze"] = (df["Close"] - df["EMA_20"]).abs() / df["EMA_20"] * 100

    # Candlestick features
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df = df.dropna()
    return df


def generate_scalping_features(df_ind: pd.DataFrame, entry: float, spread: float) -> np.ndarray:
    """Genera features ottimizzate per scalping."""
    if len(df_ind) < 10:
        return np.zeros(30, dtype=np.float32)  # fallback sicuro

    latest = df_ind.iloc[-1]
    window = min(5, max(1, len(df_ind) - 1))
    prev_win = df_ind.iloc[-(window + 1) : -1]

    bb_den = (latest["BB_upper"] - latest["BB_lower"])
    bb_position = (latest["Close"] - latest["BB_lower"]) / bb_den if bb_den > 0 else 0.5
    bb_squeeze = 1 if latest["BB_width"] < prev_win["BB_width"].mean() else 0

    features = {
        # Price action
        "close": latest["Close"],
        "ema_5": latest["EMA_5"],
        "ema_10": latest["EMA_10"],
        "ema_20": latest["EMA_20"],
        "ema_cross_5_10": 1 if latest["EMA_5"] > latest["EMA_10"] else 0,
        "ema_cross_10_20": 1 if latest["EMA_10"] > latest["EMA_20"] else 0,
        # Momentum
        "rsi_7": latest["RSI_7"],
        "rsi_14": latest["RSI"],
        "rsi_oversold": 1 if latest["RSI_7"] < 30 else 0,
        "rsi_overbought": 1 if latest["RSI_7"] > 70 else 0,
        # MACD
        "macd": latest["MACD"],
        "macd_signal": latest["MACD_signal"],
        "macd_histogram": latest["MACD_histogram"],
        "macd_bullish": 1 if latest["MACD_histogram"] > 0 else 0,
        # Bollinger
        "bb_position": bb_position,
        "bb_width": latest["BB_width"],
        "bb_squeeze": bb_squeeze,
        # Volatilità e spread
        "atr": latest["ATR"],
        "atr_pct": latest["ATR_pct"],
        "spread_to_atr": (spread / latest["ATR"]) if latest["ATR"] and latest["ATR"] > 0 else 0.0,
        # Volume
        "volume_ratio": latest["Volume_ratio"],
        "volume_surge": 1 if latest["Volume_ratio"] > 1.5 else 0,
        # Velocity / acceleration
        "price_velocity": latest["Price_Velocity"],
        "price_acceleration": latest["Price_Acceleration"],
        # Trend
        "trend_micro": latest["Trend_micro"],
        # Support/Resistance
        "distance_to_pivot": (latest["Close"] - latest["Pivot"]) / latest["Close"] * 100,
        "distance_to_r1": (latest["R1"] - latest["Close"]) / latest["Close"] * 100,
        "distance_to_s1": (latest["Close"] - latest["S1"]) / latest["Close"] * 100,
        # EMA squeeze
        "ema_squeeze": latest["EMA_squeeze"],
        # Candlestick
        "body_size": latest["body"] / latest["Close"] * 100 if latest["Close"] else 0,
        "upper_shadow_ratio": (latest["upper_shadow"] / latest["body"]) if latest["body"] > 0 else 0,
        "lower_shadow_ratio": (latest["lower_shadow_]()
