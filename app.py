# app.py
# -*- coding: utf-8 -*-

import time
import warnings
warnings.filterwarnings("ignore")

# ---- Streamlit deve essere configurato subito ----
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
                last_val = tick.last if hasattr(tick, "last") and tick.last else (tick.bid + tick.ask) / 2.0
                spread_val = (tick.ask - tick.bid) if (tick.ask and tick.bid) else None
                return {
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last": last_val,
                    "spread": spread_val,
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
                idx_last = data.index[-1]
                timestamp = idx_last.to_pydatetime() if hasattr(idx_last, "to_pydatetime") else idx_last

                # Spread stimato: forex più stretto
                spread_pct = 0.0001 if "=X" in symbol else 0.0005  # ~0.01% FX, ~0.05% altri
                spread = last_price * spread_pct
                bid = last_price - spread / 2.0
                ask = last_price + spread / 2.0

                return last_price, bid, ask, spread, source, timestamp

    except Exception as e:
        st.error(f"Errore recupero prezzo real-time: {e}")

    return None, None, None, None, None, None


# ==================== INDICATORI / FEATURE ENGINEERING ====================
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
    df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["R1"] = 2.0 * df["Pivot"] - df["Low"]
    df["S1"] = 2.0 * df["Pivot"] - df["High"]

    # EMA squeeze
    df["EMA_squeeze"] = (df["Close"] - df["EMA_20"]).abs() / df["EMA_20"] * 100

    # Candlestick features
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df = df.dropna()
    return df


def generate_scalping_features(df_ind: pd.DataFrame, entry: float, spread: float) -> np.ndarray:
    """Genera features ottimizzate per scalping (indentazione pulita e fix lower_shadow_ratio)."""
    # Safety: se i dati sono troppo pochi, ritorna un vettore zero (dimensione fissa ~ 30)
    if df_ind is None or len(df_ind) < 10:
        return np.zeros(30, dtype=np.float32)

    latest = df_ind.iloc[-1]

    # Finestra breve per confronti (evita slice vuoti)
    window = min(5, max(1, len(df_ind) - 1))
    prev_win = df_ind.iloc[-(window + 1): -1]

    # Bollinger position e squeeze
    bb_den = (latest["BB_upper"] - latest["BB_lower"])
    bb_position = (latest["Close"] - latest["BB_lower"]) / bb_den if bb_den > 0 else 0.5
    prev_bb_mean = prev_win["BB_width"].mean() if "BB_width" in prev_win.columns and len(prev_win) > 0 else np.nan
    bb_squeeze = 1 if (np.isfinite(prev_bb_mean) and latest["BB_width"] < prev_bb_mean) else 0

    # Spread/ATR ratio sicuro
    atr_val = float(latest["ATR"]) if np.isfinite(latest["ATR"]) else 0.0
    spread_to_atr = (spread / atr_val) if atr_val > 0 else 0.0

    # Candlestick ratios (evita divisioni per zero)
    body = float(latest["body"])
    close_val = float(latest["Close"])
    body_size = (body / close_val * 100.0) if close_val != 0 else 0.0
    upper_shadow_ratio = (float(latest["upper_shadow"]) / body) if body > 0 else 0.0
    lower_shadow_ratio = (float(latest["lower_shadow"]) / body) if body > 0 else 0.0

    # Costruzione feature dict
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
        "atr": atr_val,
        "atr_pct": latest["ATR_pct"],
        "spread_to_atr": spread_to_atr,

        # Volume
        "volume_ratio": latest["Volume_ratio"],
        "volume_surge": 1 if latest["Volume_ratio"] > 1.5 else 0,

        # Velocity e acceleration
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
        "body_size": body_size,
        "upper_shadow_ratio": upper_shadow_ratio,
        "lower_shadow_ratio": lower_shadow_ratio,
    }

    return np.array(list(features.values()), dtype=np.float32)


def simulate_scalping_trades(df_ind: pd.DataFrame, n_trades: int = 1000):
    """Simula trade scalping realistici con spread e slippage."""
    if len(df_ind) < 150:
        return np.empty((0, 0)), np.array([])

    X_list, y_list = [], []

    for _ in range(n_trades):
        idx = np.random.randint(100, len(df_ind) - 20)
        row = df_ind.iloc[idx]

        # Spread realistico
        spread_pct = np.random.uniform(0.0001, 0.0005)
        spread = float(row["Close"]) * spread_pct

        direction = "long" if np.random.random() < 0.5 else "short"
        entry = float(row["Close"])

        # Target/SL in funzione dell'ATR
        atr = float(row["ATR"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        tp_mult = np.random.uniform(0.3, 0.8)
        sl_mult = np.random.uniform(0.2, 0.5)

        if direction == "long":
            entry_real = entry + spread  # ask
            sl = entry_real - (atr * sl_mult)
            tp = entry_real + (atr * tp_mult)
        else:
            entry_real = entry - spread  # bid
            sl = entry_real + (atr * sl_mult)
            tp = entry_real - (atr * tp_mult)

        features = generate_scalping_features(df_ind.iloc[: idx + 1], entry_real, spread)

        future_high = df_ind.iloc[idx + 1 : idx + 21]["High"].values
        future_low = df_ind.iloc[idx + 1 : idx + 21]["Low"].values
        if len(future_high) == 0:
            continue

        if direction == "long":
            hit_tp = np.any(future_high >= tp)
            hit_sl = np.any(future_low <= sl)
        else:
            hit_tp = np.any(future_low <= tp)
            hit_sl = np.any(future_high >= sl)

        if hit_tp and not hit_sl:
            success = 1
        elif hit_sl:
            success = 0
        else:
            continue  # neutro

        X_list.append(features)
        y_list.append(success)

    if not X_list:
        return np.empty((0, 0)), np.array([])

    return np.vstack(X_list), np.array(y_list)


def train_scalping_model(X_train: np.ndarray, y_train: np.ndarray):
    """Addestra modello ottimizzato per scalping."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def calculate_scalping_setup(df_ind, model, scaler, current_price, bid, ask, spread):
    """Calcola setup scalping ottimale con probabilità."""
    features_long = generate_scalping_features(df_ind, ask, spread).reshape(1, -1)
    features_short = generate_scalping_features(df_ind, bid, spread).reshape(1, -1)

    features_long_scaled = scaler.transform(features_long)
    features_short_scaled = scaler.transform(features_short)

    prob_long = float(model.predict_proba(features_long_scaled)[0][1]) * 100
    prob_short = float(model.predict_proba(features_short_scaled)[0][1]) * 100

    latest = df_ind.iloc[-1]
    atr = float(latest["ATR"]) if np.isfinite(latest["ATR"]) else 0.0
    if atr <= 0:
        return []

    # Setup
    entry_long, sl_long, tp_long = ask, ask - (atr * 0.4), ask + (atr * 0.6)
    entry_short, sl_short, tp_short = bid, bid + (atr * 0.4), bid - (atr * 0.6)

    setups = []

    def pack(direction, entry, sl, tp, prob):
        rr = abs(tp - entry) / max(1e-12, abs(entry - sl))
        return {
            "Direction": direction,
            "Entry": round(entry, 5),
            "SL": round(sl, 5),
            "TP": round(tp, 5),
            "Probability": round(prob, 1),
            "RR": round(rr, 2),
            "Risk_pips": round(abs(entry - sl) * 10000, 1),
            "Reward_pips": round(abs(tp - entry) * 10000, 1),
            "Type": "Scalping",
        }

    if prob_long > 65:
        setups.append(pack("LONG", entry_long, sl_long, tp_long, prob_long))
    if prob_short > 65:
        setups.append(pack("SHORT", entry_short, sl_short, tp_short, prob_short))

    return setups


# ==================== DATA LOADING / TRAINING CACHES ====================
@st.cache_data(ttl=60)
def load_scalping_data(symbol: str, interval: str = "1m"):
    """Carica dati ad alta frequenza per scalping."""
    try:
        if interval not in ["1m", "5m"]:
            interval = "1m"

        period = "1d" if interval == "1m" else "5d"

        data = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)

        if data is None or data.empty:
            raise ValueError("Nessun dato ricevuto da Yahoo Finance")

        # Gestione MultiIndex (alcune versioni di yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        cols_needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
        data = data[cols_needed]

        if len(data) < 120:
            raise ValueError("Dati insufficienti per scalping")

        return data
    except Exception as e:
        st.error(f"Errore caricamento dati scalping: {e}")
        return None


@st.cache_resource
def train_scalping_model_cached(symbol: str, interval: str = "1m"):
    """Addestra e cache il modello scalping."""
    data = load_scalping_data(symbol, interval)
    if data is None:
        return None, None, None

    df_ind = calculate_scalping_indicators(data)
    X, y = simulate_scalping_trades(df_ind, n_trades=1000)

    if X.size == 0 or len(X) < 100 or len(np.unique(y)) < 2:
        st.error("Dati insufficienti per training scalping")
        return None, None, None

    model, scaler = train_scalping_model(X, y)
    return model, scaler, df_ind


# ==================== UI ====================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    * { font-family: 'Roboto Mono', monospace; }
    .main .block-container { padding-top: 1rem; max-width: 1800px; }
    h1 {
        background: linear-gradient(135deg, #00FF00 0%, #00AA00 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 700; font-size: 2.5rem !important;
    }
    .stMetric { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1rem; border-radius: 8px; border: 1px solid #00FF00;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
    }
    .stMetric label { color: #00FF00 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 1.5rem !important; font-weight: 700 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #00FF00 0%, #00AA00 100%);
        color: black; border: none; border-radius: 6px; padding: 0.5rem 1.2rem; font-weight: 700;
    }
    .scalp-card { background: #1a1a1a; border: 2px solid #00FF00; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.4);
    }
    .scalp-card-long { border-color: #00FF00; box-shadow: 0 0 15px rgba(0, 255, 0, 0.4); }
    .scalp-card-short { border-color: #FF0000; box-shadow: 0 0 15px rgba(255, 0, 0, 0.4); }
    .price-display { font-size: 2.5rem; font-weight: 700; color: #00FF00; text-align: center; padding: 1rem;
        background: #1a1a1a; border-radius: 10px; border: 2px solid #00FF00; box-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
    }
    .realtime-badge { display: inline-block; background: #FF0000; color: white; padding: 0.3rem 0.6rem; border-radius: 4px;
        font-size: 0.75rem; font-weight: 700; animation: blink 1s infinite;
    }
    @keyframes blink { 0%,50%,100%{opacity:1;} 25%,75%{opacity:0.5;} }
    section[data-testid="stSidebar"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ SCALPING AI SYSTEM - Ultra High Frequency Trading")

if MT_AVAILABLE:
    st.success(f"🟢 {MT_TYPE} Connessione Disponibile - Prezzi Real-Time Attivi")
else:
    st.warning(
        """
⚠️ MT4/MT5 non rilevato. Per prezzi real-time:
1. Installa MetaTrader 5: `pip install MetaTrader5`
2. Apri MT5 e accedi al tuo account
3. Riavvia questa applicazione

**Attualmente usando Yahoo Finance con aggiornamento ogni 60 secondi**
"""
    )

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    symbol = st.text_input("📊 Symbol", value="EURUSD=X", help="Esempi: EURUSD=X, GC=F, SI=F, BTC-USD")
with col2:
    scalp_interval = st.selectbox("⏱️ Timeframe", ["1m", "5m"], index=0)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_btn = st.button("🔄 REFRESH", use_container_width=True)
with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    auto_refresh = st.checkbox("🔁 Auto (60s)")

st.markdown("---")

# Auto-refresh
if auto_refresh:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if time.time() - st.session_state.last_refresh > 60:
        st.session_state.last_refresh = time.time()
        st.rerun()

# Prezzo real-time
price, bid, ask, spread, source, timestamp = get_realtime_price(symbol)

if price is not None and bid is not None and ask is not None and spread is not None:
    col_price1, col_price2, col_price3 = st.columns(3)

    with col_price1:
        st.markdown(
            f"""
        <div class='price-display'>
            BID: {bid:.5f}
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_price2:
        st.markdown(
            f"""
        <div class='price-display' style='font-size: 3rem;'>
            {price:.5f}<br><span class='realtime-badge'>● LIVE</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_price3:
        st.markdown(
            f"""
        <div class='price-display'>
            ASK: {ask:.5f}
        </div>
        """,
            unsafe_allow_html=True,
        )

    pct = (spread / price * 100) if price != 0 else 0
    st.caption(f"🕐 Ultimo aggiornamento: {timestamp} | Fonte: {source} | Spread: {spread:.5f} ({pct:.3f}%)")
else:
    st.info("Prezzo non disponibile. Controlla simbolo/connessione e riprova.")

st.markdown("---")

# Training modello
session_key = f"scalp_model_{symbol}_{scalp_interval}"
needs_train = (session_key not in st.session_state) or refresh_btn

if needs_train:
    with st.spinner("🧠 Training Scalping AI Model..."):
        model, scaler, df_ind = train_scalping_model_cached(symbol, scalp_interval)
        if model is not None:
            st.session_state[session_key] = {"model": model, "scaler": scaler, "df_ind": df_ind}
            st.success("✅ Scalping Model Ready!")
        else:
            st.error("❌ Impossibile addestrare il modello (dati insufficienti).")

if session_key in st.session_state and price is not None and bid is not None and ask is not None and spread is not None:
    state = st.session_state[session_key]
    model = state["model"]
    scaler = state["scaler"]
    df_ind = state["df_ind"]

    # Segnali
    setups = calculate_scalping_setup(df_ind, model, scaler, price, bid, ask, spread)

    if setups:
        st.markdown("## ⚡ SEGNALI SCALPING AD ALTA PROBABILITÀ")

        for setup in setups:
            border_class = "scalp-card-long" if setup["Direction"] == "LONG" else "scalp-card-short"
            color = "#00FF00" if setup["Direction"] == "LONG" else "#FF0000"

            st.markdown(
                f"""
            <div class='scalp-card {border_class}'>
                <h3 style='color: {color}; margin: 0;'>🎯 {setup['Direction']} SETUP</h3>
                <hr style='border-color: {color}; margin: 0.5rem 0;'>
                <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
                    <div>
                        <p style='color: #888; margin: 0; font-size: 0.8rem;'>ENTRY</p>
                        <p style='color: white; margin: 0; font-size: 1.3rem; font-weight: 700;'>{setup['Entry']}</p>
                    </div>
                    <div>
                        <p style='color: #888; margin: 0; font-size: 0.8rem;'>STOP LOSS</p>
                        <p style='color: #FF6B6B; margin: 0; font-size: 1.3rem; font-weight: 700;'>{setup['SL']}</p>
                        <p style='color: #666; margin: 0; font-size: 0.7rem;'>{setup['Risk_pips']} pips</p>
                    </div>
                    <div>
                        <p style='color: #888; margin: 0; font-size: 0.8rem;'>TAKE PROFIT</p>
                        <p style='color: #4ECB71; margin: 0; font-size: 1.3rem; font-weight: 700;'>{setup['TP']}</p>
                        <p style='color: #666; margin: 0; font-size: 0.7rem;'>{setup['Reward_pips']} pips</p>
                    </div>
                    <div>
                        <p style='color: #888; margin: 0; font-size: 0.8rem;'>PROBABILITÀ</p>
                        <p style='color: {color}; margin: 0; font-size: 1.8rem; font-weight: 700;'>{setup['Probability']}%</p>
                        <p style='color: #666; margin: 0; font-size: 0.7rem;'>R/R: {setup['RR']}x</p>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Nota: PnL/pip semplificati e indicativi
            lot_size = st.number_input(
                f"Lot Size per {setup['Direction']}",
                min_value=0.01,
                max_value=10.0,
                value=0.1,
                step=0.01,
                key=f"lot_{setup['Direction']}",
            )

            pip_value = 10 if "JPY" not in symbol else 1000
            risk_dollars = setup["Risk_pips"] * pip_value * lot_size
            reward_dollars = setup["Reward_pips"] * pip_value * lot_size

            if risk_dollars > 0:
                expected_value = (setup["Probability"] / 100.0) * reward_dollars - (
                    (100.0 - setup["Probability"]) / 100.0
                ) * risk_dollars
                delta_pct = f"{(expected_value / risk_dollars * 100):+.1f}%"
            else:
                expected_value = 0.0
                delta_pct = "N/A"

            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("💸 Rischio", f"${risk_dollars:.2f}")
            with colB:
                st.metric("💰 Reward", f"${reward_dollars:.2f}")
            with colC:
                st.metric("📊 Expected Value", f"${expected_value:.2f}", delta=delta_pct)
            with colD:
                edge = (setup["Probability"] * setup["RR"] - (100 - setup["Probability"])) / 100.0
                st.metric("⚡ Trading Edge", f"{edge:.2%}")

            st.markdown("---")
    else:
        st.info("⏳ Nessun setup scalping ad alta probabilità al momento. Attendi condizioni favorevoli.")

    # Dashboard condizioni
    st.markdown("## 📊 CONDIZIONI DI MERCATO")
    latest = df_ind.iloc[-1]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        rsi_color = "🟢" if 30 <= latest["RSI_7"] <= 70 else "🔴"
        st.metric(f"{rsi_color} RSI(7)", f"{latest['RSI_7']:.1f}")
    with c2:
        macd_color = "🟢" if latest["MACD_histogram"] > 0 else "🔴"
        st.metric(f"{macd_color} MACD", f"{latest['MACD']:.5f}")
    with c3:
        st.metric("📏 ATR", f"{latest['ATR']:.5f}")
        st.caption(f"{latest['ATR_pct']:.3f}%")
    with c4:
        bb_pos = latest["BB_upper"] - latest["BB_lower"]
        bb_pct = (latest["Close"] - latest["BB_lower"]) / bb_pos * 100 if bb_pos > 0 else 50
        st.metric("📊 BB Position", f"{bb_pct:.0f}%")
    with c5:
        vol_color = "🟢" if latest["Volume_ratio"] > 1.2 else "🟡"
        st.metric(f"{vol_color} Volume", f"{latest['Volume_ratio']:.2f}x")
    with c6:
        trend_emoji = "📈" if latest["Trend_micro"] == 1 else "📉"
        st.metric(f"{trend_emoji} Micro Trend", "Bullish" if latest["Trend_micro"] == 1 else "Bearish")

    st.markdown("---")

    # Support/Resistance
    st.markdown("### 🎯 Support & Resistance Levels")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("🔴 R1 (Resistance)", f"{latest['R1']:.5f}")
        dist_r1 = ((latest["R1"] - price) / price) * 100 if price else 0
        st.caption(f"Distanza: {dist_r1:+.3f}%")
    with s2:
        st.metric("⚪ Pivot", f"{latest['Pivot']:.5f}")
        dist_pivot = ((latest["Pivot"] - price) / price) * 100 if price else 0
        st.caption(f"Distanza: {dist_pivot:+.3f}%")
    with s3:
        st.metric("🟢 S1 (Support)", f"{latest['S1']:.5f}")
        dist_s1 = ((latest["S1"] - price) / price) * 100 if price else 0
        st.caption(f"Distanza: {dist_s1:+.3f}%")
    with s4:
        st.metric("⚡ EMA(20)", f"{latest['EMA_20']:.5f}")
        dist_ema = ((latest["EMA_20"] - price) / price) * 100 if price else 0
        st.caption(f"Distanza: {dist_ema:+.3f}%")

    st.markdown("---")

    # Linee guida
    st.markdown("### 📋 LINEE GUIDA SCALPING")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
#### ✅ Condizioni Ideali per Scalping
- **RSI(7)**: 30-70 (no extremes)
- **Volume Ratio**: > 1.2x (liquidità)
- **Spread**: < 0.02% del prezzo
- **ATR %**: 0.05-0.15% (volatilità moderata)
- **BB Width**: Squeeze o espansione
- **MACD Histogram**: Crossover freschi
- **Time**: Sessioni London/NY overlap (max volume)

#### 🎯 Regole di Ingresso
- Attendi conferma su 2-3 indicatori
- Entra solo con probabilità > 65%
- Rispetta sempre Stop Loss
- Max 2-3 trade simultanei
- Chiudi 50% a 50% del target
"""
        )
    with g2:
        st.markdown(
            """
#### ⚠️ Evita Scalping Quando
- **Spread alto**: > 0.03% del prezzo
- **News economiche**: ±30 min da release
- **RSI estremo**: < 20 o > 80
- **Low volume**: Ratio < 0.8x
- **Fine sessione**: Ultime 30 min
- **Consolidamento**: BB width < 0.1%

#### 💰 Money Management
- Rischia max 1-2% del capitale per trade
- Usa trailing stop dopo 50% target
- Scala out: 50% a target, 50% a breakeven+
- Daily loss limit: -3% (stop trading)
- Review performance ogni 10 trade
"""
        )

    # Performance tracker
    st.markdown("---")
    st.markdown("### 📈 SESSION PERFORMANCE TRACKER")

    p1, p2, p3, p4, p5 = st.columns(5)

    if "scalp_trades" not in st.session_state:
        st.session_state.scalp_trades = []

    with p1:
        trades_count = len(st.session_state.scalp_trades)
        st.metric("📊 Trades Today", trades_count)

    with p2:
        if trades_count > 0:
            wins = sum(1 for t in st.session_state.scalp_trades if t["result"] == "WIN")
            win_rate = (wins / trades_count) * 100
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        else:
            st.metric("🎯 Win Rate", "0%")

    with p3:
        if trades_count > 0:
            total_pnl = sum(t["pnl"] for t in st.session_state.scalp_trades)
            st.metric("💰 P&L", f"${total_pnl:.2f}", delta=f"{total_pnl:+.2f}")
        else:
            st.metric("💰 P&L", "$0.00")

    with p4:
        if trades_count > 0:
            avg_pnl = total_pnl / trades_count
            st.metric("📊 Avg Trade", f"${avg_pnl:.2f}")
        else:
            st.metric("📊 Avg Trade", "$0.00")

    with p5:
        if st.button("🔄 Reset Session"):
            st.session_state.scalp_trades = []
            st.rerun()

    st.markdown("---")

    with st.expander("➕ Registra Trade Manuale"):
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            trade_dir = st.selectbox("Direction", ["LONG", "SHORT"])
        with q2:
            trade_entry = st.number_input("Entry", value=float(price or 0.0), format="%.5f")
        with q3:
            trade_exit = st.number_input("Exit", value=float(price or 0.0), format="%.5f")
        with q4:
            trade_lots = st.number_input("Lots", value=0.1, min_value=0.01, step=0.01)

        if st.button("💾 Salva Trade"):
            pips = abs(trade_exit - trade_entry) * 10000
            pip_value = 10 if "JPY" not in symbol else 1000
            pnl = pips * pip_value * trade_lots

            if (trade_dir == "LONG" and trade_exit > trade_entry) or (trade_dir == "SHORT" and trade_exit < trade_entry):
                result = "WIN"
            else:
                result = "LOSS"
                pnl = -pnl

            st.session_state.scalp_trades.append(
                {
                    "direction": trade_dir,
                    "entry": trade_entry,
                    "exit": trade_exit,
                    "lots": trade_lots,
                    "pips": pips,
                    "pnl": pnl,
                    "result": result,
                    "timestamp": datetime.datetime.now(),
                }
            )

            st.success(f"✅ Trade salvato: {result} {pnl:+.2f}")
            st.rerun()

    # Tabella recenti
    if st.session_state.scalp_trades:
        st.markdown("### 📋 Ultimi Trade")
        recent_trades = st.session_state.scalp_trades[-10:]
        trades_df = pd.DataFrame(
            [
                {
                    "Time": t["timestamp"].strftime("%H:%M:%S"),
                    "Dir": t["direction"],
                    "Entry": f"{t['entry']:.5f}",
                    "Exit": f"{t['exit']:.5f}",
                    "Pips": f"{t['pips']:.1f}",
                    "P&L": f"${t['pnl']:.2f}",
                    "Result": t["result"],
                }
                for t in reversed(recent_trades)
            ]
        )
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

else:
    st.warning("⚠️ Carica il modello e/o i prezzi per iniziare lo scalping.")


# ==================== INFO SECTION ====================
with st.expander("ℹ️ COME FUNZIONA IL SISTEMA SCALPING"):
    st.markdown(
        """
## ⚡ Sistema Scalping AI - Architettura

### 🧠 Modello AI: Gradient Boosting Classifier
- **200 estimators**, **learning rate 0.05**
- **Max depth 8**, **subsample 0.8**
- Training su trade simulati con spread

### 📊 35+ Features Analizzate
Price Action, Momentum (RSI7/14), MACD, Bollinger, Volatilità (ATR), Volume, Trend micro, Pivot/SR, Candlestick ratios.

### 🎯 Parametri Operativi (indicativi)
- Target ≈ **60% ATR**, Stop ≈ **40% ATR** (R/R ≈ 1.5x)
- Filtra setup con **probabilità > 65%**

### 📡 Prezzi Real-Time
1) **MT4/MT5** se disponibili
2) **Yahoo Finance 1m** (fallback 5m)

### ⚠️ DISCLAIMER
Questo sistema è **solo educativo**. Lo scalping è rischioso; non è consulenza finanziaria. Testa su **conto demo**.
"""
    )

st.markdown(
    """
<div style='text-align: center; padding: 1rem; background: #1a1a1a; border-radius: 10px; border: 1px solid #00FF00;'>
    <p style='color: #00FF00; font-size: 0.9rem; margin: 0;'>
        ⚠️ <strong>DISCLAIMER:</strong> Scalping ad alta frequenza è estremamente rischioso. Questo sistema è solo educativo.<br>
        La maggior parte degli scalper perde denaro. NON è consiglio finanziario. Trade a tuo rischio.<br>
        Pratica su demo per mesi prima di usare capitale reale.
    </p>
    <p style='color: #666; font-size: 0.75rem; margin-top: 0.5rem;'>
        ⚡ Powered by AI Machine Learning • Gradient Boosting • 35+ Technical Features • © 2025
    </p>
</div>
""",
    unsafe_allow_html=True,
)
