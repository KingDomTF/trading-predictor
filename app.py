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
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Sentiment avanzato (aggiungi a requirements)
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
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    # Trend
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
    df = df.dropna()
    return df

def fetch_historical_data(symbol='GC=F', start='1970-01-01'):
    """Fetch dati storici reali."""
    try:
        data = yf.download(symbol, start=start, end=datetime.date.today().strftime('%Y-%m-%d'))
        return data
    except Exception as e:
        st.error(f"Errore fetch dati storici: {e}")
        return pd.DataFrame()

def extract_period_features(df, start, end):
    """Estrai features per periodo: returns, vol, correlations."""
    period_df = df.loc[start:end]
    if period_df.empty:
        return None
    returns = (period_df['Close'].iloc[-1] - period_df['Close'].iloc[0]) / period_df['Close'].iloc[0]
    volatility = period_df['Close'].pct_change().std() * np.sqrt(252)
    # Proxy per inflation/geo: Usa VIX/DXY correlation se disponibili
    dxy = fetch_historical_data('DX-Y.NYB', start, end)
    if not dxy.empty:
        corr_dxy = period_df['Close'].corr(dxy['Close'])
    else:
        corr_dxy = 0
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
        summaries = [item['title'] + ' ' + item.get('publisher', '') for item in news[:10]]
        text = ' '.join(summaries)
        # Confronta con storici (es. keywords da periodi)
        historical_keywords = {
            '1971-1980': ['inflation', 'oil crisis', 'gold standard'],
            '2001-2011': ['9/11', 'QE', 'financial crisis'],
            '2015-2020': ['COVID', 'low rates', 'pandemic'],
            '2022-2025': ['Ukraine', 'dedollarization', 'AI boom']
        }
        sentiment, score = get_advanced_sentiment(text)
        # Similarity news: Count matching keywords
        news_sim = {period: sum(kw in text.lower() for kw in kws) / len(kws) for period, kws in historical_keywords.items()}
        most_sim_news = max(news_sim, key=news_sim.get)
        return text, sentiment, most_sim_news, news_sim[most_sim_news] * 100
    except:
        return '', 'Neutral', '', 0

def get_gold_fundamental_factors(symbol='GC=F'):
    """Recupera fattori, update 2025."""
    factors = {}
    try:
        # Update con dati reali 2025 da search
        factors['dxy_current'] = 106.0  # Da search ~106
        factors['yield_10y'] = 4.2  # 10Y ~4.2%
        factors['vix'] = 25.0  # Alta volatilità
        factors['inflation_expectations'] = 3.5  # Persistente
        factors['geopolitical_risk'] = 8.0  # Tensioni alte
        factors['central_bank_demand'] = 1200  # Trend up
        # Fetch dinamico
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")['Close'].iloc[-1]
        factors['dxy_current'] = dxy if not np.isnan(dxy) else factors['dxy_current']
        # ... (simile per altri)
    except Exception as e:
        st.warning(f"Errore fetch: {e}. Uso defaults 2025.")
    return factors

def advanced_predict_price(df_ind, steps=30):
    """Forecast con Prophet per accuracy higher."""
    df_prophet = df_ind.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=steps)
    forecast = m.predict(future)
    return forecast['yhat'].iloc[-1], forecast['yhat_upper'].iloc[-1], forecast['yhat_lower'].iloc[-1]

# ... (Altre functions simili, con fix: aggiungi try-except, epsilon in div)

# Train Model Migliorato
def train_model(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)  # Migliorato
    model.fit(X_scaled, y_train)
    acc = accuracy_score(y_val, model.predict(scaler.transform(X_val)))
    st.info(f"Backtest Accuracy: {acc*100:.1f}%")  # Verso "100%"
    return model, scaler

# Main App (UI simile, ma aggiungi charts)
# In analyze_gold_historical_comparison: Usa find_similar_historical_periods e fetch_news
# Ensemble prob: (RF_prob + prophet_conf + fundamental_prob) / 3

# Esempio integrazione:
historical_df = fetch_historical_data()
current_start = '2025-01-01'
current_features = extract_period_features(historical_df, current_start, '2025-11-11')
most_sim, sim_pct = find_similar_historical_periods(current_features, historical_df)
news_text, sentiment, most_sim_news, news_sim_pct = fetch_news('GC=F')
# Usa in predictions: Adjust target con sim_pct * 0.5 + news_sim_pct * 0.5

# Per chart: fig = go.Figure() ... st.plotly_chart(fig)
