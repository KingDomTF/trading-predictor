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

def _ensure_series(df: pd.DataFrame, col: str):
    """Garantisce che df[col] sia una Series (non DataFrame con colonne duplicate)."""
    if col not in df.columns:
        return None
    s = df[col]
    if isinstance(s, pd.DataFrame):
        # se ci sono colonne duplicate, prendo la prima
        return s.iloc[:, 0]
    return s


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola indicatori tecnici classici su un DataFrame OHLCV (robusto a colonne duplicate)."""
    df = df.copy()

    # Normalizzo le colonne base in Series
    close = _ensure_series(df, "Close")
    high = _ensure_series(df, "High")
    low = _ensure_series(df, "Low")
    volume = _ensure_series(df, "Volume")

    if close is None or high is None or low is None:
        raise ValueError("Dati OHLC insufficienti per calcolare gli indicatori tecnici.")

    df["Close"] = close
    df["High"] = high
    df["Low"] = low
    if volume is not None:
        df["Volume"] = volume

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

    # Bande di Bollinger (20) - uso serie locali per evitare DataFrame
    rolling = df["Close"].rolling(window=20)
    bb_mid = rolling.mean()
    bb_std = rolling.std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_upper"] = bb_upper
    df["BB_lower"] = bb_lower

    diff = (bb_upper - bb_lower).replace(0, np.nan)
    bb_pos = (df["Close"] - bb_lower) / diff
    df["BB_position"] = bb_pos

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = np.vstack([high_low, high_close, low_close])
    true_range = np.max(ranges, axis=0)
    df["ATR"] = pd.Series(true_range, index=df.index).rolling(window=14).mean()

    # Volume medio
    if "Volume" in df.columns and df["Volume"].notna().any():
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_ratio"] = df["Volume"] / df["Volume_MA"]
    else:
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
    Simula trade storici per training e costruisce un database di comportamenti.
    Ritorna:
      - X: matrice feature
      - y: label (1=successo, 0=fallimento)
      - trades_meta: DataFrame con info per analisi analogie
    """
    X_list = []
    y_list = []
    meta = []

    if len(df_ind) <= 100:
        return np.empty((0, 14), dtype=np.float32), np.empty((0,), dtype=np.int32), pd.DataFrame()

    for _ in range(n_trades):
        idx = np.random.randint(50, len(df_ind) - 50)
        row = df_ind.iloc[idx]

        direction = np.random.choice(["long", "short"])
        entry = float(row["Close"])
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)

        if direction == "long":
            sl = entry * (1 - sl_pct / 100)
            tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100)
            tp = entry * (1 - tp_pct / 100)

        features = generate_features(df_ind.iloc[: idx + 1], entry, sl, tp, direction, 60)

        # Simula outcome nei prossimi 50 periodi
        future_prices = df_ind.iloc[idx + 1 : idx + 51]["Close"].values
        if future_prices.size == 0:
            continue

        if direction == "long":
            hit_tp = np.any(future_prices >= tp)
            hit_sl = np.any(future_prices <= sl)
        else:
            hit_tp = np.any(future_prices <= tp)
            hit_sl = np.any(future_prices >= sl)

        success = 1 if (hit_tp and not hit_sl) else 0

        risk_pct = abs(entry - sl) / entry * 100 if entry != 0 else 0.0
        reward_pct = abs(tp - entry) / entry * 100 if entry != 0 else 0.0
        rr = reward_pct / risk_pct if risk_pct > 0 else np.nan

        X_list.append(features)
        y_list.append(success)
        meta.append(
            {
                "timestamp": df_ind.index[idx],
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "risk_pct": risk_pct,
                "reward_pct": reward_pct,
                "rr": rr,
                "success": success,
            }
        )

    if not X_list:
        return np.empty((0, 14), dtype=np.float32), np.empty((0,), dtype=np.int32), pd.DataFrame()

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    trades_meta = pd.DataFrame(meta)

    return X, y, trades_meta


# =========================================================
#                 MODELLO ML E TRAINING
# =========================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """Addestra il modello Random Forest e restituisce anche le feature scalate."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train)

    return model, scaler, X_scaled


def predict_success(model, scaler, features: np.ndarray) -> float:
    """Predice la probabilità di successo (0-100%)."""
    feat_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(feat_scaled)[0, 1]
    return float(prob * 100)


def get_dominant_factors(model, features: np.ndarray):
    """Restituisce i 5 fattori più importanti e il loro valore nel setup attuale."""
    feature_names = [
        "SL Distance %",
        "TP Distance %",
        "R/R Ratio",
        "Direction",
        "TimeFrame",
        "RSI",
        "MACD",
        "MACD Signal",
        "ATR",
        "EMA Diff %",
        "BB Position",
        "Volume Ratio",
        "Price Change %",
        "Trend",
    ]

    importances = model.feature_importances_
    idx_top = np.argsort(importances)[-5:][::-1]

    out = []
    for i in idx_top:
        if i < len(feature_names):
            out.append(
                {
                    "Feature": feature_names[i],
                    "Importance": importances[i],
                    "CurrentValue": float(features[i]),
                }
            )
    return out


def find_similar_trades(
    features: np.ndarray,
    scaler,
    X_scaled_hist: np.ndarray,
    trades_meta: pd.DataFrame,
    top_k: int = 100,
):
    """
    Trova i trade storici più simili al setup corrente.
    Restituisce:
      - stats: dizionario con percentuali e medie
      - similar_meta: DataFrame con i trade storici selezionati
    """
    if X_scaled_hist is None or len(X_scaled_hist) == 0:
        return None, None
    if trades_meta is None or trades_meta.empty:
        return None, None

    feat_scaled = scaler.transform(features.reshape(1, -1))[0]
    dists = np.linalg.norm(X_scaled_hist - feat_scaled, axis=1)
    idx_sorted = np.argsort(dists)[: min(top_k, len(dists))]
    similar_meta = trades_meta.iloc[idx_sorted].copy()

    if similar_meta.empty:
        return None, None

    stats = {
        "n_trades": int(len(similar_meta)),
        "success_rate": float(similar_meta["success"].mean() * 100),
        "avg_rr": float(similar_meta["rr"].mean()),
        "avg_risk_pct": float(similar_meta["risk_pct"].mean()),
        "avg_reward_pct": float(similar_meta["reward_pct"].mean()),
        "long_share": float((similar_meta["direction"] == "long").mean() * 100),
        "short_share": float((similar_meta["direction"] == "short").mean() * 100),
    }
    return stats, similar_meta


# =========================================================
#        SENTIMENT NEWS (STILE ALADDIN / ORACOL)
# =========================================================

POSITIVE_WORDS = [
    "beat", "beats", "strong", "soars", "rally", "record", "growth",
    "upgraded", "bullish", "surge", "profit", "profits", "gains"
]
NEGATIVE_WORDS = [
    "miss", "cuts", "downgraded", "weak", "falls", "plunge", "crash",
    "loss", "losses", "bearish", "lawsuit", "fraud", "risk"
]


def get_sentiment(text: str):
    """
    Sentiment super leggero basato su parole chiave.
    Ritorna (label, score) dove score >0 = positivo, <0 = negativo.
    """
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0

    t = text.lower()
    score = 0
    for w in POSITIVE_WORDS:
        if w in t:
            score += 1
    for w in NEGATIVE_WORDS:
        if w in t:
            score -= 1

    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return label, float(score)


def fetch_latest_news(symbol: str, max_items: int = 10):
    """
    Recupera e normalizza le ultime news da Yahoo Finance per un dato simbolo.
    Ogni news ha: titolo, link, publisher, data, sentiment, sentiment_score.
    """
    try:
        ticker = yf.Ticker(symbol)
        news_raw = getattr(ticker, "news", None)
    except Exception:
        news_raw = None

    if not news_raw or not isinstance(news_raw, list):
        return []

    items = []
    for item in news_raw[:max_items]:
        if not isinstance(item, dict):
            continue

        title = item.get("title", "")
        if not title:
            continue

        link = item.get("link", "")
        publisher = item.get("publisher", "")
        ts = item.get("providerPublishTime", None)

        published = None
        if ts is not None:
            try:
                published = datetime.datetime.fromtimestamp(ts)
            except Exception:
                published = None

        sentiment_label, sentiment_score = get_sentiment(title)

        items.append(
            {
                "title": title,
                "link": link,
                "publisher": publisher,
                "published": published,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
            }
        )

    return items


def predict_price(df_ind: pd.DataFrame, steps: int = 5):
    """Previsione prezzo molto semplice basata su EMA."""
    try:
        last_price = df_ind["Close"].iloc[-1]
        ema = df_ind["Close"].ewm(span=steps).mean().iloc[-1]
        forecast_values = [
            last_price + (ema - last_price) * (i / steps) for i in range(1, steps + 1)
        ]
        forecast = np.array(forecast_values, dtype=float)
        return float(forecast.mean()), forecast
    except Exception:
        return None, None


def get_web_signals(symbol: str, df_ind: pd.DataFrame):
    """
    Motore stile Aladdin:
    - prezzo corrente
    - news + sentiment aggregato
    - stagionalità
    - nota di previsione semplice
    - suggerimenti Long/Short / Buy/Sell
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            return []

        current_price = float(hist["Close"].iloc[-1])

        # News recenti strutturate
        news_items = fetch_latest_news(symbol, max_items=8)
        if news_items:
            news_summary = " | ".join(n["title"] for n in news_items)
            total_score = sum(n["sentiment_score"] for n in news_items)
            if total_score > 0:
                sentiment_label = "Positive"
            elif total_score < 0:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            sentiment_score = float(total_score)
        else:
            news_summary = "Nessuna news recente disponibile."
            sentiment_label = "Neutral"
            sentiment_score = 0.0

        # Stagionalità (su dati mensili 10 anni)
        hist_monthly = yf.download(symbol, period="10y", interval="1mo", progress=False)
        if len(hist_monthly) < 12:
            seasonality_note = "Dati storici insufficienti per calcolare la stagionalità."
        else:
            hist_monthly["Return"] = hist_monthly["Close"].pct_change()
            hist_monthly["Month"] = hist_monthly.index.month
            monthly_returns = hist_monthly.groupby("Month")["Return"].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            seasonality_note = (
                f"Il mese corrente ha un ritorno medio storico di {avg_current:.2f}%."
            )

        # Previsione prezzo
        _, forecast_series = predict_price(df_ind, steps=5)
        forecast_note = (
            f"Previsione media per i prossimi 5 periodi: {forecast_series.mean():.2f}"
            if forecast_series is not None
            else "Previsione non disponibile."
        )

        latest = df_ind.iloc[-1]
        atr = float(latest["ATR"])
        trend = int(latest["Trend"])

        suggestions = []
        directions = ["Long", "Short"] if "=X" not in symbol else ["Buy", "Sell"]

        for dir_ in directions:
            is_positive_dir = (
                (dir_ in ["Long", "Buy"] and (sentiment_score > 0 or trend == 1))
                or (dir_ in ["Short", "Sell"] and (sentiment_score < 0 or trend == 0))
            )
            prob = 70 if is_positive_dir else 60
            entry = round(current_price, 4 if "=" in symbol else 2)
            sl_mult = 1.0 if is_positive_dir else 1.5
            tp_mult = 2.5 if is_positive_dir else 2.0

            if dir_ in ["Long", "Buy"]:
                sl = round(entry - atr * sl_mult, 4 if "=" in symbol else 2)
                tp = round(entry + atr * tp_mult, 4 if "=" in symbol else 2)
            else:
                sl = round(entry + atr * sl_mult, 4 if "=" in symbol else 2)
                tp = round(entry - atr * tp_mult, 4 if "=" in symbol else 2)

            suggestions.append(
                {
                    "Direction": dir_,
                    "Entry": entry,
                    "SL": sl,
                    "TP": tp,
                    "Probability": prob,
                    "Seasonality_Note": seasonality_note,
                    "News_Summary": news_summary,
                    "News_List": news_items,
                    "Sentiment": sentiment_label,
                    "Forecast_Note": forecast_note,
                }
            )

        # Aggiungi un terzo suggerimento se sentiment neutrale
        if sentiment_score == 0:
            dir_ = directions[0] if trend == 1 else directions[1]
            entry = round(current_price, 4 if "=" in symbol else 2)
            sl_mult = 1.2
            tp_mult = 2.2
            if dir_ in ["Long", "Buy"]:
                sl = round(entry - atr * sl_mult, 4 if "=" in symbol else 2)
                tp = round(entry + atr * tp_mult, 4 if "=" in symbol else 2)
            else:
                sl = round(entry + atr * sl_mult, 4 if "=" in symbol else 2)
                tp = round(entry - atr * tp_mult, 4 if "=" in symbol else 2)

            suggestions.append(
                {
                    "Direction": dir_,
                    "Entry": entry,
                    "SL": sl,
                    "TP": tp,
                    "Probability": 65,
                    "Seasonality_Note": seasonality_note,
                    "News_Summary": news_summary,
                    "News_List": news_items,
                    "Sentiment": sentiment_label,
                    "Forecast_Note": forecast_note,
                }
            )

        return suggestions

    except Exception as e:
        st.error(f"Errore nel recupero dati web: {e}")
        return []


# =========================================================
#              DATI LIVE & UTILITIES DI BASE
# =========================================================

def fetch_live_price(symbol: str):
    """Ritorna l'ultimo prezzo disponibile per il simbolo."""
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if data.empty:
            data = yf.download(symbol, period="5d", interval="1d", progress=False)
        if data.empty:
            return None
        close = _ensure_series(data, "Close")
        return float(close.iloc[-1])
    except Exception:
        return None


def load_sample_data(symbol: str, interval: str = "1h"):
    """Scarica dati storici per il simbolo scelto."""
    period_map = {
        "15m": "5d",
        "30m": "14d",
        "1h": "60d",
        "4h": "1y",
        "1d": "5y",
    }
    period = period_map.get(interval, "60d")

    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        data = data.dropna()
        if data.empty:
            return None
        return data
    except Exception:
        return None


@st.cache_resource
def train_or_load_model(symbol: str, interval: str = "1h"):
    """Addestra (o ricarica) il modello e costruisce il database storico."""
    data = load_sample_data(symbol, interval)
    if data is None or data.empty:
        return None, None, None, None, None, None, None

    df_ind = calculate_technical_indicators(data)
    X, y, trades_meta = simulate_historical_trades(df_ind, n_trades=500)
    if X.size == 0 or y.size == 0:
        return None, None, None, None, None, None, None

    model, scaler, X_scaled = train_model(X, y)
    return model, scaler, df_ind, X, y, X_scaled, trades_meta


# =========================================================
#               PSICOLOGIA / SPIEGAZIONI TESTUALI
# =========================================================

def get_investor_psychology(prob_success: float, rr_ratio: float) -> str:
    """
    Restituisce una breve nota di psicologia del trade
    in base a probabilità stimata e rapporto R/R.
    """
    notes = []

    if prob_success >= 70:
        notes.append(
            "Il modello vede una **probabilità elevata** di successo. "
            "Resta comunque fondamentale rispettare il piano di uscita."
        )
    elif prob_success <= 45:
        notes.append(
            "La probabilità stimata è **moderata/bassa**: è facile cadere nel bias di overconfidence. "
            "Valuta dimensione posizione ridotta."
        )
    else:
        notes.append(
            "Scenario **intermedio**: il risultato dipende molto dalla disciplina nell'esecuzione "
            "e dalla gestione delle emozioni."
        )

    if rr_ratio >= 2.0:
        notes.append(
            "Il R/R è **molto favorevole** (>2:1): pochi trade vincenti possono compensare diversi stop loss."
        )
    elif rr_ratio < 1.0:
        notes.append(
            "Il R/R è **sfavorevole** (<1:1): stai rischiando più di quanto potresti guadagnare, "
            "un pattern classico di loss aversion."
        )
    else:
        notes.append(
            "R/R **bilanciato**: lavora sulla coerenza del metodo e sulla riduzione dei trade impulsivi."
        )

    return "\n\n".join(notes)


# =========================================================
#                        UI STREAMLIT
# =========================================================

def main():
    st.set_page_config(
        page_title="Aladdin-Oracol AI Trading Lab",
        page_icon="💹",
        layout="wide",
    )

    st.markdown(
        """
        <h1 style='text-align: center;'>💹 Aladdin • Oracol AI Trading Lab</h1>
        <p style='text-align: center; color: #718096;'>
        Motore di analisi quantitativa + news + analogie storiche di comportamento trader.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ========================
    #      SELEZIONE INPUT
    # ========================
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🎯 Selezione Strumento")

        default_list = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "META",
            "TSLA",
            "NVDA",
            "EURUSD=X",
            "GBPUSD=X",
            "BTC-USD",
            "GC=F",
            "^GSPC",
        ]

        symbol = st.selectbox("Simbolo / Ticker", options=default_list, index=0)
        tf_label = st.selectbox(
            "Timeframe principale",
            options=["15m", "30m", "1h", "4h", "1d"],
            index=2,
        )

        if tf_label.endswith("m"):
            main_tf = int(tf_label.replace("m", ""))
        elif tf_label.endswith("h"):
            main_tf = int(tf_label.replace("h", "")) * 60
        else:
            main_tf = 1440

        st.caption(
            "Suggerimento: timeframe più alti (4H, Daily) generano pattern più stabili ma meno trade."
        )

    with col_right:
        st.subheader("📡 Info Rapide")

        live_price = fetch_live_price(symbol)
        if live_price is not None:
            st.metric("Prezzo live stimato", f"{live_price:.4f}")
        else:
            st.metric("Prezzo live stimato", "N/D")

        st.markdown("##### 🔔 News flash (ultime headline)")
        news_brief = fetch_latest_news(symbol, max_items=3)
        if news_brief:
            for n in news_brief:
                date_str = n["published"].strftime("%d/%m %H:%M") if n["published"] else ""
                st.markdown(
                    f"- **{n['title']}**  \n"
                    f"  _{n['publisher']} • {date_str} • Sentiment: {n['sentiment']}_"
                )
        else:
            st.caption("Nessuna news recente trovata su Yahoo Finance per questo ticker.")

    st.markdown("---")

    # ======================================================
    #                 CARICAMENTO / TRAINING
    # ======================================================
    with st.spinner("Addestro il modello e costruisco il database di analogie..."):
        model, scaler, df_ind, X_hist, y_hist, X_scaled_hist, trades_meta = train_or_load_model(
            symbol, tf_label
        )

    if model is None or df_ind is None:
        st.error(
            "Impossibile addestrare il modello per questo simbolo/timeframe. "
            "Prova a cambiare ticker o timeframe."
        )
        return

    # ============================
    #        DASHBOARD TOP
    # ============================
    st.subheader(f"📈 Panorama Tecnico su {symbol} ({tf_label})")

    latest = df_ind.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Ultimo Close", f"{latest['Close']:.4f}")
    with c2:
        rsi_emoji = "🟢" if 30 <= latest["RSI"] <= 70 else "🔴"
        st.metric(f"{rsi_emoji} RSI", f"{latest['RSI']:.1f}")
    with c3:
        st.metric("ATR", f"{latest['ATR']:.4f}")
    with c4:
        trend_emoji = "📈" if latest["Trend"] == 1 else "📉"
        trend_text = "Bullish" if latest["Trend"] == 1 else "Bearish"
        st.metric(f"{trend_emoji} Trend", trend_text)
    with c5:
        st.metric("Variazione % Ult. Barra", f"{latest['Price_change_pct']:.2f}%")

    # ============================
    #     MOTORE NEWS & SENTIMENT
    # ============================

    st.markdown("### 🧠 Motore News, Sentiment & Suggerimenti AI")

    suggestions = get_web_signals(symbol, df_ind)
    if not suggestions:
        st.warning("Non sono riuscito a generare suggerimenti web-based per questo simbolo.")
    else:
        suggestions_df = pd.DataFrame(suggestions)
        st.dataframe(
            suggestions_df[["Direction", "Entry", "SL", "TP", "Probability", "Sentiment"]],
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("📊 Dettagli Supplementari (Stagionalità, News, Previsioni)"):
            st.markdown("#### 📅 Analisi Stagionalità")
            st.info(suggestions_df.iloc[0]["Seasonality_Note"])

            st.markdown("#### 📰 News Recenti")
            first_row = suggestions_df.iloc[0]
            news_list = first_row.get("News_List", [])
            if news_list:
                for n in news_list:
                    published_str = (
                        n["published"].strftime("%d/%m/%Y %H:%M") if n["published"] else ""
                    )
                    st.markdown(
                        f"**{n['title']}**  \n"
                        f"_{n['publisher']} • {published_str} • Sentiment: {n['sentiment']}_  \n"
                        f"[Apri articolo]({n['link']})"
                    )
            else:
                st.write(first_row["News_Summary"])

            st.markdown("#### 😊 Sentiment Aggregato")
            sentiment = suggestions_df.iloc[0]["Sentiment"]
            if sentiment == "Positive":
                st.success(f"🟢 {sentiment} - Il mercato mostra segnali positivi")
            elif sentiment == "Negative":
                st.error(f"🔴 {sentiment} - Il mercato mostra segnali negativi")
            else:
                st.warning(f"🟡 {sentiment} - Il mercato è neutrale")

            st.markdown("#### 🔮 Previsione Prezzo (semplice)")
            st.info(suggestions_df.iloc[0]["Forecast_Note"])

    st.markdown("---")

    # ======================================================
    #        MOTORE ANALOGIE STORICHE (ORACOL)
    # ======================================================
    st.markdown("## 🧬 Analogie Storiche & Comportamento Trader")

    st.caption(
        "Qui usiamo il database di trade simulati per capire come si sono "
        "comportati i trader in condizioni **simili** a quelle attuali."
    )

    # Genera un setup base a partire dal primo suggerimento disponibile
    if suggestions:
        base = suggestions[0]
        direction = "long" if base["Direction"] in ["Long", "Buy"] else "short"
        entry = float(base["Entry"])
        sl = float(base["SL"])
        tp = float(base["TP"])
    else:
        # fallback: costruzione manuale attorno al prezzo attuale
        entry = float(latest["Close"])
        atr = float(latest["ATR"])
        direction = "long" if latest["Trend"] == 1 else "short"
        if direction == "long":
            sl = entry - atr * 1.5
            tp = entry + atr * 3.0
        else:
            sl = entry + atr * 1.5
            tp = entry - atr * 3.0

    features_current = generate_features(df_ind, entry, sl, tp, direction, main_tf)
    rr_ratio_current = float(abs(tp - entry) / abs(entry - sl)) if abs(entry - sl) > 0 else 1.0

    similar_stats, similar_trades_df = find_similar_trades(
        features=features_current,
        scaler=scaler,
        X_scaled_hist=X_scaled_hist,
        trades_meta=trades_meta,
        top_k=150,
    )

    if similar_stats is None:
        st.info("Non ho abbastanza trade storici simulati per costruire un'analogia affidabile.")
    else:
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            st.metric(
                "✅ Successo storico setup simili",
                f"{similar_stats['success_rate']:.1f}%",
                help=f"Basato su {similar_stats['n_trades']} trade simulati",
            )
        with cs2:
            st.metric("📉 Rischio medio", f"{similar_stats['avg_risk_pct']:.2f}%")
        with cs3:
            st.metric("📈 Reward medio", f"{similar_stats['avg_reward_pct']:.2f}%")

        st.caption(
            f"In questo cluster storico circa {similar_stats['long_share']:.1f}% dei trade erano LONG "
            f"e {similar_stats['short_share']:.1f}% SHORT, con R/R medio {similar_stats['avg_rr']:.2f}x."
        )

        with st.expander("📚 Esempi di trade storici analoghi"):
            show_cols = [
                "timestamp",
                "direction",
                "success",
                "entry",
                "sl",
                "tp",
                "rr",
                "risk_pct",
                "reward_pct",
            ]
            show_cols = [c for c in show_cols if c in similar_trades_df.columns]
            st.dataframe(
                similar_trades_df[show_cols].head(20),
                hide_index=True,
                use_container_width=True,
            )

    # ======================================================
    #        ANALISI DEL TRADE ATTUALE CON ML
    # ======================================================

    st.markdown("## 🤖 Analisi AI del Setup Corrente")

    prob_success = predict_success(model, scaler, features_current)

    col_ml1, col_ml2, col_ml3 = st.columns(3)
    with col_ml1:
        st.metric("Prob. successo stimata", f"{prob_success:.1f}%")
    with col_ml2:
        st.metric("R/R del setup mostrato", f"{rr_ratio_curren
