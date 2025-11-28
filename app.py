import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
from pathlib import Path
import os
import glob

warnings.filterwarnings("ignore")

# ==================== MAPPATURA SIMBOLI & MT4 ====================

# Ticker Yahoo -> simbolo MT4
YAHOO_TO_MT4 = {
    "GC=F": "XAUUSD",   # Oro
    "SI=F": "XAGUSD",   # Argento
    "BTC-USD": "BTCUSD",
    "^GSPC": "US500",
}

# Inverso: simbolo MT4 -> ticker Yahoo
MT4_TO_YAHOO = {v: k for k, v in YAHOO_TO_MT4.items()}


def resolve_symbols(user_symbol: str):
    """
    Ritorna (yahoo_symbol, mt4_symbol) a partire da quello che scrivi nella casella.
    Puoi scrivere: GC=F, SI=F, BTC-USD, ^GSPC oppure XAUUSD, XAGUSD, BTCUSD, US500.
    """
    s = user_symbol.strip()
    s_upper = s.upper()

    # Match su ticker Yahoo
    for y, m in YAHOO_TO_MT4.items():
        if s_upper == y.upper():
            return y, m

    # Match su simboli MT4
    for m, y in MT4_TO_YAHOO.items():
        if s_upper == m.upper():
            return y, m

    # Default: uso lo stesso
    return s, s


# ==================== RICERCA AUTOMATICA FILE MT4 ====================

def find_mt4_prices_file():
    """
    Trova automaticamente il file mt4_prices.csv nell'installazione MT4 standard:
    %APPDATA%\\MetaQuotes\\Terminal\\*\\MQL4\\Files\\mt4_prices.csv
    """
    appdata = os.getenv("APPDATA")
    if not appdata:
        return None

    pattern = os.path.join(
        appdata, "MetaQuotes", "Terminal", "*", "MQL4", "Files", "mt4_prices.csv"
    )
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    # Se ci sono più terminal, prendo il file modificato più di recente
    latest = max(candidates, key=os.path.getmtime)
    return Path(latest)


MT4_PRICES_FILE = find_mt4_prices_file()


def fetch_live_price(mt4_symbol: str):
    """
    Legge il prezzo live SOLO dal file mt4_prices.csv scritto dall'EA in MT4.
    Ritorna (last_price, prev_close) oppure (None, None) se qualcosa non va.
    """
    if "last_price_source" not in st.session_state:
        st.session_state["last_price_source"] = "N/D"

    if "mt4_file_path" not in st.session_state:
        st.session_state["mt4_file_path"] = str(MT4_PRICES_FILE) if MT4_PRICES_FILE else "non trovato"

    if MT4_PRICES_FILE is None:
        st.session_state["last_price_source"] = "File mt4_prices.csv NON trovato (MT4 non rilevato)"
        return None, None

    try:
        df = pd.read_csv(
            MT4_PRICES_FILE,
            sep=";",
            header=None,
            names=["symbol", "time", "last", "prev_close"],
        )
    except Exception as e:
        st.session_state["last_price_source"] = f"ERRORE lettura file MT4: {e}"
        return None, None

    if df.empty or "symbol" not in df.columns:
        st.session_state["last_price_source"] = "File MT4 vuoto o corrotto"
        return None, None

    df_sym = df[df["symbol"].str.upper() == mt4_symbol.upper()]
    if df_sym.empty:
        st.session_state["last_price_source"] = f"MT4: simbolo {mt4_symbol} non trovato nel file"
        return None, None

    row = df_sym.iloc[-1]
    last = float(row["last"])
    prev = float(row["prev_close"])
    st.session_state["last_price_source"] = f"MT4 ({mt4_symbol})"
    return last, prev


# ==================== FUNZIONI TECNICHE E ML ====================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    df["BB_middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + 2 * bb_std
    df["BB_lower"] = df["BB_middle"] - 2 * bb_std

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Price_Change"] = df["Close"].pct_change()
    df["Trend"] = df["Close"].rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)

    df = df.dropna()
    return df


def generate_features(df_ind: pd.DataFrame, entry, sl, tp, direction: str, main_tf: int):
    latest = df_ind.iloc[-1]
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100

    features = {
        "sl_distance_pct": sl_distance,
        "tp_distance_pct": tp_distance,
        "rr_ratio": rr_ratio,
        "direction": 1 if direction == "long" else 0,
        "main_tf": main_tf,
        "rsi": latest["RSI"],
        "macd": latest["MACD"],
        "macd_signal": latest["MACD_signal"],
        "atr": latest["ATR"],
        "ema_diff": (latest["EMA_20"] - latest["EMA_50"]) / latest["Close"] * 100,
        "bb_position": (latest["Close"] - latest["BB_lower"]) / (latest["BB_upper"] - latest["BB_lower"]),
        "volume_ratio": latest["Volume"] / latest["Volume_MA"] if latest["Volume_MA"] > 0 else 1.0,
        "price_change": latest["Price_Change"] * 100,
        "trend": latest["Trend"],
    }

    return np.array(list(features.values()), dtype=np.float32)


def simulate_historical_trades(df_ind: pd.DataFrame, n_trades=400):
    X_list = []
    y_list = []

    if len(df_ind) < 100:
        return np.empty((0, 0)), np.empty((0,))

    for _ in range(n_trades):
        idx = np.random.randint(50, len(df_ind) - 50)
        row = df_ind.iloc[idx]

        direction = np.random.choice(["long", "short"])
        entry = row["Close"]
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)

        if direction == "long":
            sl = entry * (1 - sl_pct / 100)
            tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100)
            tp = entry * (1 - tp_pct / 100)

        features = generate_features(df_ind.iloc[: idx + 1], entry, sl, tp, direction, 60)

        future_prices = df_ind.iloc[idx + 1 : idx + 51]["Close"].values
        if len(future_prices) == 0:
            continue

        if direction == "long":
            hit_tp = np.any(future_prices >= tp)
            hit_sl = np.any(future_prices <= sl)
        else:
            hit_tp = np.any(future_prices <= tp)
            hit_sl = np.any(future_prices >= sl)

        success = 1 if hit_tp and not hit_sl else 0
        X_list.append(features)
        y_list.append(success)

    if not X_list:
        return np.empty((0, 0)), np.empty((0,))

    return np.array(X_list), np.array(y_list)


def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def predict_success(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100.0


def get_dominant_factors(model, features):
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
    idxs = np.argsort(importances)[-5:][::-1]

    out = []
    for i in idxs:
        if i < len(feature_names):
            out.append(f"{feature_names[i]}: {features[i]:.2f} (importanza: {importances[i]:.2%})")
    return out


def get_sentiment(text: str):
    positive_words = ["rally", "up", "bullish", "gain", "positive", "strong", "rise", "surge", "boom"]
    negative_words = ["down", "bearish", "loss", "negative", "weak", "slip", "fall", "drop", "crash"]
    score = sum(w in text.lower() for w in positive_words) - sum(w in text.lower() for w in negative_words)
    if score > 0:
        return "Positive", score
    if score < 0:
        return "Negative", score
    return "Neutral", 0


def predict_price(df_ind: pd.DataFrame, steps=5):
    try:
        last = df_ind["Close"].iloc[-1]
        ema = df_ind["Close"].ewm(span=steps).mean().iloc[-1]
        values = [last + (ema - last) * (i / steps) for i in range(1, steps + 1)]
        arr = np.array(values)
        return arr.mean(), arr
    except Exception:
        return None, None


def get_investor_psychology(symbol: str, news_summary: str, sentiment_label: str, df_ind: pd.DataFrame) -> str:
    latest = df_ind.iloc[-1]
    trend = "bullish" if latest["Trend"] == 1 else "bearish"

    txt = f"""
### 🧠 Psicologia dell'investitore su {symbol}

- Trend tecnico attuale: **{trend}**
- Sentiment news: **{sentiment_label}**
- RSI: **{latest['RSI']:.1f}** → gli investitori tendono a vedere zone overbought/oversold come 'verità assoluta', spesso entrando tardi.
- Volatilità (ATR): **{latest['ATR']:.2f}** → più è alta, più dominano paura e FOMO.

Principali bias in gioco:
- **Avversione alle perdite**: si tagliano i profitti troppo presto e si lasciano correre le perdite.
- **Effetto gregge**: si entra solo quando il movimento è già evidente sui social / news.
- **Recency bias**: se le ultime giornate sono forti, molti pensano che continuerà all'infinito (e viceversa).

Un approccio più razionale:
- definire a priori **Rischio (SL)** e **Reward (TP)**
- non aumentare la size solo perché gli ultimi trade sono andati bene
- usare segnali tecnici e fondamentali come guida, non come verità assoluta.
"""
    return txt


def get_web_signals(symbol: str, df_ind: pd.DataFrame):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="5m")
        if hist.empty:
            hist = ticker.history(period="5d", interval="15m")
        if hist.empty:
            return []

        current_price = float(hist["Close"].iloc[-1])

        # News
        try:
            news = ticker.news
            if news and isinstance(news, list):
                news_titles = [n.get("title", "") for n in news[:5] if isinstance(n, dict)]
                news_summary = " | ".join(news_titles) if news_titles else "Nessuna news recente."
            else:
                news_summary = "Nessuna news recente."
        except Exception:
            news_summary = "Nessuna news recente."

        sentiment_label, sentiment_score = get_sentiment(news_summary)

        # Stagionalità semplice
        try:
            hist_monthly = yf.download(symbol, period="10y", interval="1mo", progress=False)
            if len(hist_monthly) >= 12:
                hist_monthly["Return"] = hist_monthly["Close"].pct_change()
                hist_monthly["Month"] = hist_monthly.index.month
                monthly_returns = hist_monthly.groupby("Month")["Return"].mean()
                cur_month = datetime.datetime.now().month
                avg_cur = monthly_returns.get(cur_month, 0) * 100
                seasonality_note = f"Storicamente, questo mese ha un rendimento medio del {avg_cur:.2f}%."
            else:
                seasonality_note = "Dati insufficienti per la stagionalità."
        except Exception:
            seasonality_note = "Dati insufficienti per la stagionalità."

        _, forecast_series = predict_price(df_ind, steps=5)
        if forecast_series is not None:
            forecast_note = f"Media previsione prossimi 5 step: {forecast_series.mean():.2f}"
        else:
            forecast_note = "Previsione non disponibile."

        latest = df_ind.iloc[-1]
        atr = latest["ATR"]
        trend = latest["Trend"]

        suggestions = []
        for direction in ["Long", "Short"]:
            good_dir = (direction == "Long" and (sentiment_score >= 0 and trend == 1)) or (
                direction == "Short" and (sentiment_score <= 0 and trend == 0)
            )
            prob = 70 if good_dir else 60
            entry = round(current_price, 2)
            sl_mult = 1.0 if good_dir else 1.5
            tp_mult = 2.4 if good_dir else 2.0

            if direction == "Long":
                sl = round(entry - atr * sl_mult, 2)
                tp = round(entry + atr * tp_mult, 2)
            else:
                sl = round(entry + atr * sl_mult, 2)
                tp = round(entry - atr * tp_mult, 2)

            suggestions.append(
                {
                    "Direction": direction,
                    "Entry": entry,
                    "SL": sl,
                    "TP": tp,
                    "Probability": prob,
                    "Seasonality_Note": seasonality_note,
                    "News_Summary": news_summary,
                    "Sentiment": sentiment_label,
                    "Forecast_Note": forecast_note,
                }
            )

        return suggestions
    except Exception as e:
        st.error(f"Errore nel recupero dati web: {e}")
        return []


# ==================== CARICAMENTO DATI & MODELLO ====================

@st.cache_data
def load_sample_data(symbol, interval="1h"):
    period_map = {"5m": "60d", "15m": "60d", "1h": "730d"}
    period = period_map.get(interval, "730d")
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) < 100:
            raise Exception("Dati insufficienti")
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        return data
    except Exception as e:
        st.error(f"Errore nel caricamento dati: {e}")
        return None


@st.cache_resource
def train_or_load_model(symbol, interval="1h"):
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=400)
    if X.size == 0:
        return None, None, None
    model, scaler = train_model(X, y)
    return model, scaler, df_ind


proper_names = {
    "GC=F": "XAU/USD (Gold)",
    "XAUUSD": "XAU/USD (Gold, MT4)",
    "SI=F": "XAG/USD (Silver)",
    "XAGUSD": "XAG/USD (Silver, MT4)",
    "BTC-USD": "BTC/USD",
    "BTCUSD": "BTC/USD (MT4)",
    "^GSPC": "S&P 500",
    "US500": "S&P 500 (US500 MT4)",
}

# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="Trading Predictor AI - MT4 Live",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Trading Predictor AI – prezzi live da MT4")

# Toggle prezzi live
if "live_prices_enabled" not in st.session_state:
    st.session_state["live_prices_enabled"] = True

switch_col1, switch_col2 = st.columns([1, 3])
with switch_col1:
    if st.button("📡 Attiva/Disattiva prezzo live MT4"):
        st.session_state["live_prices_enabled"] = not st.session_state["live_prices_enabled"]
with switch_col2:
    stato = "attivi ✅" if st.session_state["live_prices_enabled"] else "disattivati ⛔"
    st.caption(f"Prezzi live MT4 attualmente **{stato}**.")

st.markdown("---")

# Parametri
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    user_symbol = st.text_input(
        "🔍 Ticker",
        value="GC=F",
        help="Puoi usare GC=F, SI=F, BTC-USD, ^GSPC oppure direttamente XAUUSD, XAGUSD, BTCUSD, US500",
    )
    yahoo_symbol, mt4_symbol = resolve_symbols(user_symbol)
    proper_name = (
        proper_names.get(user_symbol)
        or proper_names.get(yahoo_symbol)
        or proper_names.get(mt4_symbol, user_symbol)
    )
    st.markdown(
        f"**Strumento:** `{proper_name}`  \n📡 Yahoo: `{yahoo_symbol}` • 🖥️ MT4: `{mt4_symbol}`"
    )

with col2:
    data_interval = st.selectbox("⏰ Timeframe dati storici", ["5m", "15m", "1h"], index=2)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Ricarica dati", use_container_width=True)

# Prezzo live
live_price, prev_close = (None, None)
if st.session_state["live_prices_enabled"]:
    live_price, prev_close = fetch_live_price(mt4_symbol)

live_col1, live_col2 = st.columns([1, 2])
with live_col1:
    if live_price is not None:
        delta_str = None
        if prev_close not in (None, 0):
            delta_pct = (live_price - prev_close) / prev_close * 100
            delta_str = f"{delta_pct:+.2f}%"
        display_price = f"{live_price:.4f}" if live_price < 10 else f"{live_price:.2f}"
        st.metric("💹 Prezzo live (MT4)", display_price, delta_str)
    else:
        st.metric("💹 Prezzo live (MT4)", "N/D")

with live_col2:
    source = st.session_state.get("last_price_source", "sorgente sconosciuta")
    path_info = st.session_state.get("mt4_file_path", "percorso sconosciuto")
    st.caption(
        f"Aggiornato alle {datetime.datetime.now().strftime('%H:%M:%S')} • Fonte: {source} • File: {path_info}"
    )

st.markdown("---")

# Modello ML
session_key = f"model_{yahoo_symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Addestramento / caricamento modello AI..."):
        model, scaler, df_ind = train_or_load_model(yahoo_symbol, data_interval)
        if model is None:
            st.error("Impossibile caricare i dati storici per questo simbolo.")
        else:
            st.session_state[session_key] = {
                "model": model,
                "scaler": scaler,
                "df_ind": df_ind,
            }
            st.success("✅ Modello pronto.")

if session_key not in st.session_state:
    st.stop()

state = st.session_state[session_key]
model = state["model"]
scaler = state["scaler"]
df_ind = state["df_ind"]

avg_forecast, forecast_series = predict_price(df_ind, steps=5)
web_signals = get_web_signals(yahoo_symbol, df_ind)

# Colonne principali
col_left, col_right = st.columns([1.4, 0.6])

with col_left:
    st.subheader("💡 Suggerimenti di trade (da web + indicatori)")
    if web_signals:
        df_sugg = pd.DataFrame(web_signals).sort_values("Probability", ascending=False)

        st.write("Clicca sulla lente per analizzare un setup con l'AI:")

        for idx, row in df_sugg.iterrows():
            emoji = (
                "🟢"
                if row["Sentiment"] == "Positive"
                else "🔴"
                if row["Sentiment"] == "Negative"
                else "🟡"
            )
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"""
<div class='trade-card'>
<strong>{row['Direction'].upper()}</strong> • Entry: <strong>{row['Entry']:.2f}</strong> • 
SL: {row['SL']:.2f} • TP: {row['TP']:.2f}<br>
Probabilità web: <strong>{row['Probability']:.0f}%</strong> • Sentiment: {emoji} <strong>{row['Sentiment']}</strong>
</div>
""",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🔍", key=f"analyze_{idx}"):
                    st.session_state["selected_trade"] = row

        with st.expander("Dettagli extra (stagionalità, news, previsione)"):
            st.markdown("**Stagionalità:**")
            st.info(df_sugg.iloc[0]["Seasonality_Note"])
            st.markdown("**News recenti:**")
            st.write(df_sugg.iloc[0]["News_Summary"])
            st.markdown("**Previsione breve termine:**")
            st.info(df_sugg.iloc[0]["Forecast_Note"])
    else:
        st.info("Nessun suggerimento web disponibile per questo strumento.")

with col_right:
    st.subheader("📊 Indicatori sintetici")
    latest = df_ind.iloc[-1]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("RSI", f"{latest['RSI']:.1f}")
        st.metric("ATR", f"{latest['ATR']:.2f}")
    with c2:
        trend_text = "Bullish 📈" if latest["Trend"] == 1 else "Bearish 📉"
        st.metric("Trend", trend_text)
        if avg_forecast is not None:
            delta_f = (avg_forecast - latest["Close"]) / latest["Close"] * 100
            st.metric("Previsione breve", f"{avg_forecast:.2f}", f"{delta_f:+.1f}%")

# Analisi trade selezionato
if "selected_trade" in st.session_state:
    trade = st.session_state["selected_trade"]
    st.markdown("---")
    st.subheader("🎯 Analisi AI del setup selezionato")

    direction = "long" if trade["Direction"].lower() in ["long", "buy"] else "short"
    entry = trade["Entry"]
    sl = trade["SL"]
    tp = trade["TP"]

    features = generate_features(df_ind, entry, sl, tp, direction, 60)
    prob_ai = predict_success(model, scaler, features)
    dom_factors = get_dominant_factors(model, features)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Probabilità web", f"{trade['Probability']:.1f}%")
    with c2:
        st.metric("Probabilità AI", f"{prob_ai:.1f}%")
    with c3:
        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        st.metric("R/R", f"{rr:.2f}x")
    with c4:
        risk_pct = abs(entry - sl) / entry * 100
        reward_pct = abs(tp - entry) / entry * 100
        st.metric("Rischio/Reward %", f"{risk_pct:.1f}% / {reward_pct:.1f}%")

    st.markdown("#### Fattori che pesano di più per l'AI")
    for i, f in enumerate(dom_factors, 1):
        st.markdown(f"{i}. {f}")

    st.markdown("#### Psicologia dell'investitore")
    st.markdown(
        get_investor_psychology(yahoo_symbol, trade["News_Summary"], trade["Sentiment"], df_ind)
    )

st.markdown("---")
st.caption(
    "Storico e indicatori da Yahoo Finance (può differire leggermente da MT4). "
    "Prezzo live **solo da MT4** via mt4_prices.csv generato da `PriceExporter.mq4`."
)
