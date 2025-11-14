import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings("ignore")

# =========================================================
#               FUNZIONI DI ANALISI TECNICA
# =========================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola indicatori tecnici classici su un DataFrame OHLCV."""
    df = df.copy()

    # EMA
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bande di Bollinger
    rolling = df["Close"].rolling(window=20)
    bb_mid = rolling.mean()
    bb_std = rolling.std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = np.vstack([high_low, high_close, low_close])
    true_range = np.max(ranges, axis=0)
    df["ATR"] = pd.Series(true_range, index=df.index).rolling(window=14).mean()

    # Volume medio
    if "Volume" in df.columns:
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_ratio"] = df["Volume"] / df["Volume_MA"]
    else:
        df["Volume"] = np.nan
        df["Volume_MA"] = np.nan
        df["Volume_ratio"] = 1.0

    # Trend semplificato
    df["Price_change_pct"] = df["Close"].pct_change() * 100
    df["Trend"] = df["Close"].rolling(window=20).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
    )

    # Pulizia NaN
    df = df.dropna().copy()
    return df


# =========================================================
#                    FEATURE ENGINEERING
# =========================================================

def generate_features(
    df_ind: pd.DataFrame,
    entry: float,
    sl: float,
    tp: float,
    direction: str,
    main_tf: int,
) -> np.ndarray:
    """Genera il vettore di feature per il modello ML."""
    latest = df_ind.iloc[-1]

    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance_pct = abs(entry - sl) / entry * 100 if entry != 0 else 0.0
    tp_distance_pct = abs(tp - entry) / entry * 100 if entry != 0 else 0.0
    ema_diff_pct = (entry - latest["EMA_20"]) / latest["EMA_20"] * 100 if latest["EMA_20"] != 0 else 0.0

    # Posizione rispetto alle bande (clippata per evitare estremi)
    bb_pos = latest.get("BB_position", 0.5)
    if np.isnan(bb_pos):
        bb_pos = 0.5
    bb_pos = float(np.clip(bb_pos, 0.0, 1.0))

    volume_ratio = latest.get("Volume_ratio", 1.0)
    if np.isnan(volume_ratio) or np.isinf(volume_ratio):
        volume_ratio = 1.0

    features = np.array(
        [
            sl_distance_pct,
            tp_distance_pct,
            rr_ratio,
            1 if direction.lower() == "long" else 0,
            float(main_tf),
            latest["RSI"],
            latest["MACD"],
            latest["MACD_signal"],
            latest["ATR"],
            ema_diff_pct,
            bb_pos,
            volume_ratio,
            latest["Price_change_pct"],
            latest["Trend"],
        ],
        dtype=np.float32,
    )

    return features


# =========================================================
#            SIMULAZIONE TRADE STORICI (ORACOL DB)
# =========================================================

def simulate_historical_trades(df_ind: pd.DataFrame, n_trades: int = 500):
    """
    Simula trade storici per training e costruis
