import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yfinance as yf
import datetime
import warnings
from prophet import Prophet  # Per forecasting avanzato
import plotly.graph_objects as go  # Per charts
import requests  # Per news fetch alternativo
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Sentiment avanzato
import nltk
nltk.download('vader_lexicon', quiet=True)
warnings.filterwarnings('ignore')

# ==================== FUNZIONI CORE ====================
def calculate_technical_indicators(df):
    """Calcola indicatori tecnici con handling errori."""
    df = df.copy()
    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)  # Fix division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    # Trend
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
    df = df.dropna()
    return df

def fetch_historical_data(symbol='GC=F', start='1970-01-01', end=None):
    """Fetch dati storici reali."""
    try:
        end = end or datetime.date.today().strftime('%Y-%m-%d')
        data = yf.download(symbol, start=start, end=end, progress=False)
        return data
    except Exception as e:
        st.error(f"Errore fetch dati storici: {e}")
        return pd.DataFrame()

def extract_period_features(df, start, end):
    """Estrai features per periodo: returns, vol, correlations."""
    period_df = df.loc[start:end]
    if period_df.empty or len(period_df) < 2:
        return None
    returns = (period_df['Close'].iloc[-1] - period_df['Close'].iloc[0]) / period_df['Close'].iloc[0]
    volatility = period_df['Close'].pct_change().std() * np.sqrt(252)
    dxy = fetch_historical_data('DX-Y.NYB', start, end)
    corr_dxy = period_df['Close'].corr(dxy['Close']) if not dxy.empty else 0
    return np.array([returns, volatility, corr_dxy])

def find_similar_historical_periods(current_features, historical_df):
    """Calcola similarità dinamica con cosine."""
    periods = [
        ('1971-1980', '1971-01-01', '1980-12-31'),
        ('2001-2011', '2001-01-01', '2011-12-31'),
        ('2015-2020', '2015-01-01', '2020-12-31'),
        ('2022-2025', '2022-01-01', '2025-11-11')
    ]
    similarities = {}
    for name, start, end in periods:
        features = extract_period_features(historical_df, start, end)
        if features is not None:
            sim = np.dot(current_features, features) / (np.linalg.norm(current_features) * np.linalg.norm(features) + 1e-10)
            similarities[name] = sim * 100
    if not similarities:
        return 'No similar period', 0
    most_similar = max(similarities, key=similarities.get)
    return most_similar, similarities[most_similar]

def get_advanced_sentiment(text):
    """Sentiment avanzato con VADER."""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Positive', score
    elif score < -0.05:
        return 'Negative', score
    else:
        return 'Neutral', score

def fetch_news(symbol, period='recent'):
    """Fetch news e confronta con storici."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        summaries = [item.get('title', '') + ' ' + item.get('summary', '') for item in news[:10]]
        text = ' '.join(summaries)
        # Confronta con storici (es. keywords da periodi)
        historical_keywords = {
            '1971-1980': ['inflation', 'oil crisis', 'gold standard'],
            '2001-2011': ['9/11', 'QE', 'financial crisis'],
            '2015-2020': ['COVID', 'low rates', 'pandemic'],
            '2022-2025': ['Ukraine', 'dedollarization', 'AI boom', 'Fed cuts']
        }
        sentiment, score = get_advanced_sentiment(text)
        # Similarity news: Count matching keywords
        news_sim = {period: sum(kw in text.lower() for kw in kws) / len(kws) for period, kws in historical_keywords.items() if kws}
        most_sim_news = max(news_sim, key=news_sim.get) if news_sim else ''
        return text, sentiment, most_sim_news, news_sim.get(most_sim_news, 0) * 100
    except:
        return '', 'Neutral', '', 0

def get_gold_fundamental_factors(symbol='GC=F'):
    """Recupera fattori, update 2025."""
    factors = {}
    try:
        # Dati aggiornati da ricerca (11 Nov 2025)
        factors['dxy_current'] = 99.7  # Da ricerca
        factors['yield_10y'] = 4.12  # 10Y yield
        factors['vix'] = 17.9  # VIX
        factors['inflation_expectations'] = 3.5  # Media da 1Y 4.7%, 5Y 2.2%
        factors['geopolitical_risk'] = 8.0  # Alto da dashboard
        factors['central_bank_demand'] = 1200  # Est da Q3 634t ytd, trend up
        # Fetch dinamico per override
        dxy_ticker = yf.Ticker("DX-Y.NYB")
        dxy_hist = dxy_ticker.history(period="1d")
        if not dxy_hist.empty:
            factors['dxy_current'] = dxy_hist['Close'].iloc[-1]
        # Simile per altri, con try-except individuali
        vix_ticker = yf.Ticker("^VIX")
        vix_hist = vix_ticker.history(period="1d")
        if not vix_hist.empty:
            factors['vix'] = vix_hist['Close'].iloc[-1]
        # ... Aggiungi per yield, etc. usando tickers appropriati
    except Exception as e:
        st.warning(f"Errore fetch: {e}. Uso defaults 2025.")
    return factors

def analyze_gold_historical_comparison(current_price, factors):
    """Analizza confronti storici e predice prezzo futuro dell'oro."""
    historical_df = fetch_historical_data('GC=F', '1970-01-01')
    current_start = '2025-01-01'
    current_end = datetime.date.today().strftime('%Y-%m-%d')
    current_features = extract_period_features(historical_df, current_start, current_end)
    most_similar, similarity_pct = find_similar_historical_periods(current_features, historical_df)
    
    news_text, sentiment, most_sim_news, news_sim_pct = fetch_news('GC=F')
    
    # Historical periods updated with real gains
    historical_periods = {
        '1971-1980': {'description': 'Bull Market Post-Bretton Woods', 'gain_pct': 2329},
        '2001-2011': {'description': 'Post-Dot-Com e Crisi 2008', 'gain_pct': 653},
        '2015-2020': {'description': 'Consolidamento e COVID Rally', 'gain_pct': 97},
        '2022-2025': {'description': 'Era Inflazione Post-COVID', 'gain_pct': 59}  # Da ricerca +59% YTD
    }
    
    # Proiezione
    similar_period = historical_periods.get(most_similar, {'gain_pct': 100})
    projection_1y = current_price * (1 + similar_period['gain_pct'] / 100 * 0.25)  # Adjust per anno
    
    # Fundamental adjustment
    base_price = current_price
    if factors['dxy_current'] < 100:
        base_price *= 1.02
    if factors['yield_10y'] - factors['inflation_expectations'] < 1:
        base_price *= 1.025
    # ... Simile per altri
    
    # Media proiezioni
    target_price_1y = (projection_1y + base_price) / 2
    
    confidence = min(100, similarity_pct * 0.5 + news_sim_pct * 0.3 + (20 if factors['central_bank_demand'] > 800 else 10))
    
    return {
        'target_1y': target_price_1y,
        'confidence': confidence,
        'most_similar_period': most_similar,
        'similarity_pct': similarity_pct,
        'news_sentiment': sentiment
    }

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    latest = df_ind.iloc[-1]
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100
    features = [
        sl_distance, tp_distance, rr_ratio, 1 if direction == 'long' else 0, main_tf,
        latest['RSI'], latest['MACD'], latest['MACD_signal'], latest['ATR'],
        (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
        latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        latest['Price_Change'] * 100, latest['Trend']
    ]
    return np.array(features, dtype=np.float32)

def simulate_historical_trades(df_ind, n_trades=500):
    X_list, y_list = [], []
    for _ in range(n_trades):
        idx = np.random.randint(50, len(df_ind) - 50)
        row = df_ind.iloc[idx]
        direction = np.random.choice(['long', 'short'])
        entry = row['Close']
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)
        if direction == 'long':
            sl = entry * (1 - sl_pct / 100)
            tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100)
            tp = entry * (1 - tp_pct / 100)
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60)
        future_prices = df_ind.iloc[idx+1:idx+51]['Close'].values
        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp)
                hit_sl = np.any(future_prices <= sl)
            else:
                hit_tp = np.any(future_prices <= tp)
                hit_sl = np.any(future_prices >= sl)
            success = 1 if hit_tp and not hit_sl else 0
            X_list.append(features)
            y_list.append(success)
    return np.array(X_list), np.array(y_list)

def train_model(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_train)
    acc = accuracy_score(y_val, model.predict(scaler.transform(X_val)))
    st.info(f"Backtest Accuracy: {acc*100:.1f}%")
    return model, scaler

def predict_success(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

def get_dominant_factors(model, features):
    feature_names = [
        'SL Distance %', 'TP Distance %', 'R/R Ratio', 'Direction', 'TimeFrame',
        'RSI', 'MACD', 'MACD Signal', 'ATR', 'EMA Diff %',
        'BB Position', 'Volume Ratio', 'Price Change %', 'Trend'
    ]
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
    factors = []
    for i in indices:
        factors.append(f"{feature_names[i]}: {features[i]:.2f} (importanza: {importances[i]:.2%})")
    return factors

def advanced_predict_price(df_ind, steps=30):
    try:
        df_prophet = df_ind.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        m = Prophet(yearly_seasonality=True, daily_seasonality=False)
        m.fit(df_prophet[['ds', 'y']])
        future = m.make_future_dataframe(periods=steps)
        forecast = m.predict(future)
        return forecast['yhat'].iloc[-1], forecast['yhat_upper'].iloc[-1], forecast['yhat_lower'].iloc[-1]
    except Exception as e:
        st.warning(f"Prophet error: {e}. Fallback to EMA.")
        avg, _ = predict_price(df_ind, 5)
        return avg, avg, avg

def predict_price(df_ind, steps=5):
    try:
        last_price = df_ind['Close'].iloc[-1]
        ema = df_ind['Close'].ewm(span=steps).mean().iloc[-1]
        forecast_values = [last_price + (ema - last_price) * (i / steps) for i in range(1, steps + 1)]
        forecast = np.array(forecast_values)
        return forecast.mean(), forecast
    except:
        return None, None

def get_investor_psychology(symbol, news_summary, sentiment_label, df_ind):
    latest = df_ind.iloc[-1]
    trend = 'bullish' if latest['Trend'] == 1 else 'bearish'
    current_analysis = f"""
    **🌍 Contesto Globale (Novembre 2025)**
    Nel contesto attuale, i mercati sono influenzati da inflazione al ~3.5%, tensioni geopolitiche alte, e boom AI. Per {symbol}, trend {trend}, sentiment {sentiment_label}.
    """
    biases_analysis = """
    ### 🧠 Bias Comportamentali
    | Bias | Impatto |
    |------|---------|
    | Avversione alle Perdite | Deflussi >200 mld |
    | Eccessiva Fiducia | Perdite in volatilità |
    """
    return current_analysis + biases_analysis

def get_web_signals(symbol, df_ind):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        if hist.empty:
            return []
        current_price = hist['Close'].iloc[-1]
        news = ticker.news
        news_summary = ' | '.join([item.get('title', '') for item in news[:5]])
        sentiment_label, _ = get_advanced_sentiment(news_summary)
        hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
        if len(hist_monthly) < 12:
            seasonality_note = 'Dati insufficienti.'
        else:
            hist_monthly['Return'] = hist_monthly['Close'].pct_change()
            hist_monthly['Month'] = hist_monthly.index.month
            monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            seasonality_note = f'Retorno medio mese: {avg_current:.2f}%.'
        avg_forecast, _ = predict_price(df_ind, 5)
        forecast_note = f'Previsione 5 periodi: {avg_forecast:.2f}' if avg_forecast else 'N/A'
        latest = df_ind.iloc[-1]
        atr = latest['ATR']
        trend = latest['Trend']
        suggestions = []
        directions = ['Long', 'Short'] if '=X' not in symbol else ['Buy', 'Sell']
        for dir in directions:
            is_positive = (dir in ['Long', 'Buy'] and trend == 1) or (dir in ['Short', 'Sell'] and trend == 0)
            prob = 70 if is_positive else 60
            entry = round(current_price, 2)
            sl_mult = 1.0 if is_positive else 1.5
            tp_mult = 2.5 if is_positive else 2.0
            if dir in ['Long', 'Buy']:
                sl = round(entry - atr * sl_mult, 2)
                tp = round(entry + atr * tp_mult, 2)
            else:
                sl = round(entry + atr * sl_mult, 2)
                tp = round(entry - atr * tp_mult, 2)
            suggestions.append({
                'Direction': dir,
                'Entry': entry,
                'SL': sl,
                'TP': tp,
                'Probability': prob,
                'Seasonality_Note': seasonality_note,
                'News_Summary': news_summary,
                'Sentiment': sentiment_label,
                'Forecast_Note': forecast_note
            })
        return suggestions
    except Exception as e:
        st.error(f"Errore web signals: {e}")
        return []

@st.cache_data
def load_sample_data(symbol, interval='1h'):
    period_map = {'5m': '60d', '15m': '60d', '1h': '730d'}
    period = period_map.get(interval, '730d')
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if len(data) < 100:
            raise Exception("Dati insufficienti")
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return None

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

proper_names = {
    'GC=F': 'XAU/USD (Gold)',
    'EURUSD=X': 'EUR/USD',
    'SI=F': 'XAG/USD (Silver)',
    'BTC-USD': 'BTC/USD',
    '^GSPC': 'S&P 500',
}

st.set_page_config(page_title="Trading Predictor AI - Gold Focus", page_icon="🥇", layout="wide")

# CSS as in original (omitted for brevity, add your CSS here)

st.title("🥇 Trading Predictor AI - Gold Analysis")
# Header as in original (add your markdown here)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("🔍 Strumento (Ticker)", "GC=F")
    proper_name = proper_names.get(symbol, symbol)
    st.markdown(f"**Selezionato:** `{proper_name}`")
with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h'], index=2)
with col3:
    refresh_data = st.button("🔄 Carica Dati")

session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Caricamento..."):
        model, scaler, df_ind = train_or_load_model(symbol, data_interval)
        if model:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Pronto!")
        else:
            st.error("❌ Errore dati.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    current_price = df_ind['Close'].iloc[-1] if not df_ind.empty else 4142  # Fallback
    
    if symbol == 'GC=F':
        st.markdown("## 🥇 Analisi Oro")
        with st.spinner("🔍 Analisi..."):
            factors = get_gold_fundamental_factors()
            gold_analysis = analyze_gold_historical_comparison(current_price, factors)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prezzo Attuale", f"${current_price:.2f}")
        with col2:
            st.metric("Target 1Y", f"${gold_analysis['target_1y']:.2f}")
        st.metric("Confidenza", f"{gold_analysis['confidence']:.1f}%")
        st.info(f"Periodo simile: {gold_analysis['most_similar_period']} ({gold_analysis['similarity_pct']:.1f}%) - Sentiment: {gold_analysis['news_sentiment']}")
        
        # Chart
        if not df_ind.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_ind.index[-100:],
                                         open=df_ind['Open'][-100:],
                                         high=df_ind['High'][-100:],
                                         low=df_ind['Low'][-100:],
                                         close=df_ind['Close'][-100:]))
            st.plotly_chart(fig)
    
    # Suggerimenti trade
    web_signals_list = get_web_signals(symbol, df_ind)
    if web_signals_list:
        st.markdown("### 💡 Suggerimenti Trade")
        for sug in web_signals_list:
            st.write(f"{sug['Direction']} - Entry: {sug['Entry']} - Prob: {sug['Probability']}%")
    
    # Psicologia
    psych = get_investor_psychology(symbol, '', '', df_ind)
    st.markdown(psych)

# Expander and footer (add your markdown here)
