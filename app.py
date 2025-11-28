import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import datetime

# ===================== MAPPATURE SIMBOLI =====================

# Yahoo -> MT4
YAHOO_TO_MT4 = {
    "GC=F": "XAUUSD",   # Oro future -> Oro MT4
    "SI=F": "XAGUSD",   # Argento future -> Argento MT4
    "BTC-USD": "BTCUSD",# Bitcoin USD -> Bitcoin MT4
    "^GSPC": "US500",   # S&P 500 index -> US500 MT4
}

# MT4 -> Yahoo (invertiamo la mappa sopra)
MT4_TO_YAHOO = {v: k for k, v in YAHOO_TO_MT4.items()}

def resolve_symbols(user_symbol: str):
    """
    Riceve il simbolo inserito dall'utente e restituisce:
    - yahoo_symbol: per scaricare i dati storici da Yahoo
    - mt4_symbol:   per leggere il prezzo live dal file MT4
    Esempi:
      - 'GC=F'   -> ('GC=F', 'XAUUSD')
      - 'XAUUSD' -> ('GC=F', 'XAUUSD')
      - 'US500'  -> ('^GSPC', 'US500')
      - simbolo sconosciuto -> (user_symbol, user_symbol)
    """
    user_symbol = user_symbol.strip()

    if user_symbol in YAHOO_TO_MT4:
        yahoo_symbol = user_symbol
        mt4_symbol = YAHOO_TO_MT4[user_symbol]
    elif user_symbol in MT4_TO_YAHOO:
        mt4_symbol = user_symbol
        yahoo_symbol = MT4_TO_YAHOO[user_symbol]
    else:
        yahoo_symbol = user_symbol
        mt4_symbol = user_symbol

    return yahoo_symbol, mt4_symbol


# ===================== FILE MT4 (CSV) =====================

# ⚠️ METTI QUI IL PERCORSO ESATTO DEL TUO mt4_prices.csv
#  Esempio tipico:
#  r"C:\Users\TUO_NOME\AppData\Roaming\MetaQuotes\Terminal\XXXX...\MQL4\Files\mt4_prices.csv"
MT4_PRICES_FILE = Path(
    r"C:\PERCORSO\ALLA\TUA\MQL4\Files\mt4_prices.csv"
)


def fetch_live_price(mt4_symbol: str):
    """
    Legge il prezzo live SOLO dalla tua MT4 (file mt4_prices.csv).
    Ritorna (last_price, prev_close) oppure (None, None) se qualcosa non va.
    """
    if "last_price_source" not in st.session_state:
        st.session_state["last_price_source"] = "N/D"

    try:
        df = pd.read_csv(
            MT4_PRICES_FILE,
            sep=';',
            header=None,
            names=['symbol', 'time', 'last', 'prev_close']
        )
    except Exception as e:
        st.session_state["last_price_source"] = f"ERRORE MT4: {e}"
        return None, None

    df_sym = df[df["symbol"].str.upper() == mt4_symbol.upper()]
    if df_sym.empty:
        st.session_state["last_price_source"] = f"MT4: simbolo {mt4_symbol} NON trovato in mt4_prices.csv"
        return None, None

    row = df_sym.iloc[-1]
    last_price = float(row["last"])
    prev_close = float(row["prev_close"])

    st.session_state["last_price_source"] = f"MT4 ({mt4_symbol})"
    return last_price, prev_close


# ===================== DATI STORICI (YAHOO) =====================

@st.cache_data
def load_history(yahoo_symbol: str, interval: str = "1h") -> pd.DataFrame:
    """
    Scarica dati storici da Yahoo Finance per il grafico / indicatori.
    """
    period_map = {
        "15m": "30d",
        "1h": "90d",
        "4h": "180d",
        "1d": "2y",
    }
    period = period_map.get(interval, "90d")

    try:
        data = yf.download(
            yahoo_symbol,
            period=period,
            interval=interval,
            progress=False
        )
        if data.empty:
            return pd.DataFrame()
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.reset_index(inplace=True)
        # uniformiamo il nome della colonna tempo
        if "Datetime" in data.columns:
            data.rename(columns={"Datetime": "Time"}, inplace=True)
        elif "Date" in data.columns:
            data.rename(columns={"Date": "Time"}, inplace=True)
        return data
    except Exception:
        return pd.DataFrame()


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge qualche indicatore semplice (EMA, RSI) giusto per dare contesto.
    """
    if df.empty:
        return df

    df = df.copy()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


# ===================== APP STREAMLIT =====================

def main():
    st.set_page_config(
        page_title="Trading Success Predictor AI - MT4 Live",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Trading Success Predictor AI – Live da MT4")
    st.markdown(
        """
        Questa app:
        - usa **Yahoo Finance** per lo storico (grafico e indicatori)
        - usa **SOLO MT4** (file `mt4_prices.csv`) per il **prezzo live**
        """
    )

    # === SWITCH GENERALE PER PREZZI LIVE (MT4) ===
    if "live_prices_enabled" not in st.session_state:
        st.session_state["live_prices_enabled"] = False

    col_switch1, col_switch2 = st.columns([1, 3])
    with col_switch1:
        if st.button(
            "📡 Attiva/Disattiva prezzo live MT4",
            help="Quando attivo, legge il prezzo live dal file mt4_prices.csv generato dall'EA in MT4."
        ):
            st.session_state["live_prices_enabled"] = not st.session_state["live_prices_enabled"]

    with col_switch2:
        stato = "attivi ✅" if st.session_state["live_prices_enabled"] else "disattivati ⛔"
        st.caption(f"Prezzi live MT4 attualmente **{stato}**.")

    st.markdown("---")

    # Parametri strumento
    col1, col2 = st.columns([2, 1])
    with col1:
        user_symbol = st.text_input(
            "Strumento",
            value="GC=F",
            help="Puoi scrivere GC=F, SI=F, BTC-USD, ^GSPC oppure direttamente XAUUSD, XAGUSD, BTCUSD, US500"
        )

    with col2:
        interval = st.selectbox(
            "Timeframe storico (Yahoo)",
            options=["15m", "1h", "4h", "1d"],
            index=1,
        )

    yahoo_symbol, mt4_symbol = resolve_symbols(user_symbol)
    st.caption(f"📡 Yahoo: `{yahoo_symbol}`  •  🖥️ MT4: `{mt4_symbol}`")

    # === PREZZO LIVE DA MT4 ===
    live_price, prev_close = (None, None)
    if st.session_state.get("live_prices_enabled", False):
        live_price, prev_close = fetch_live_price(mt4_symbol)

    col_lp1, col_lp2 = st.columns([1, 2])
    with col_lp1:
        if live_price is not None:
            delta_str = None
            if prev_close is not None and prev_close != 0:
                delta_pct = (live_price - prev_close) / prev_close * 100
                delta_str = f"{delta_pct:+.2f}%"
            display_price = f"{live_price:.4f}" if live_price < 10 else f"{live_price:.2f}"
            st.metric("💰 Prezzo live (MT4)", display_price, delta_str)
        else:
            st.metric("💰 Prezzo live (MT4)", "N/D")

    with col_lp2:
        source = st.session_state.get("last_price_source", "sorgente sconosciuta")
        st.caption(
            f"Aggiornato alle {datetime.datetime.now().strftime('%H:%M:%S')} • Fonte: {source}"
        )

    st.markdown("---")

    # === DATI STORICI + INDICATORI ===
    with st.spinner("Carico dati storici da Yahoo Finance..."):
        hist = load_history(yahoo_symbol, interval)

    if hist.empty:
        st.error("Nessun dato storico trovato per questo simbolo (lato Yahoo).")
        return

    hist = add_basic_indicators(hist)

    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        st.subheader("📈 Storico prezzi (Yahoo)")
        st.line_chart(
            hist.set_index("Time")[["Close"]],
            height=300
        )

    with col_info:
        st.subheader("📊 Indicatori base")
        last = hist.iloc[-1]
        st.write(f"**Ultimo close (Yahoo)**: {last['Close']:.2f}")
        if not np.isnan(last.get("EMA_20", np.nan)):
            st.write(f"**EMA 20**: {last['EMA_20']:.2f}")
        if not np.isnan(last.get("EMA_50", np.nan)):
            st.write(f"**EMA 50**: {last['EMA_50']:.2f}")
        if not np.isnan(last.get("RSI_14", np.nan)):
            st.write(f"**RSI 14**: {last['RSI_14']:.1f}")

    st.markdown("---")
    st.caption(
        "⚠️ Storico e indicatori da Yahoo Finance (può essere leggermente diverso da MT4). "
        "Prezzo live SOLO da MT4 tramite `mt4_prices.csv`."
    )


if __name__ == "__main__":
    main()
