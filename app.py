import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import yfinance as yf
import datetime
import logging
import warnings

# Tentativo di import Bloomberg API (xbbg wrapper per blpapi)
try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    st.warning("Bloomberg API non disponibile. Utilizzo yfinance come fallback.")

# Suppress warnings and set up logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURAZIONI ====================
ASSET_CONFIGS = {
    'BTC-USD': {'volatility_mult': 1.5, 'trend_weight': 0.7, 'volume_weight': 0.9},
    'GC=F': {'volatility_mult': 0.8, 'trend_weight': 0.6, 'volume_weight': 0.5},
    'SI=F': {'volatility_mult': 1.0, 'trend_weight': 0.6, 'volume_weight': 0.6},
    '^GSPC': {'volatility_mult': 0.7, 'trend_weight': 0.8, 'volume_weight': 0.7},
    'DEFAULT': {'volatility_mult': 1.0, 'trend_weight': 0.6, 'volume_weight': 0.6}
}

PROPER_NAMES = {
    'GC=F': 'Gold (XAU/USD)',
    'SI=F': 'Silver (XAG/USD)',
    'BTC-USD': 'Bitcoin',
    'EURUSD=X': 'EUR/USD',
    '^GSPC': 'S&P 500 Index',
    'SPY': 'S&P 500 ETF',
    'NVDA': 'Nvidia',
    'MSFT': 'Microsoft',
    'AVGO': 'Broadcom',
    'TSM': 'Taiwan Semiconductor',
    'PLTR': 'Palantir',
    'URA': 'Uranium ETF'
}

PERIOD_MAP = {
    '5m': '60d',
    '15m': '60d',
    '1h': '730d',
    '1d': '5y'
}

# ==================== FUNZIONI METRICHE AGGIUNTE ====================
def calculate_sharpe(df: pd.DataFrame, rf: float = 0.01) -> float:
    """Calcola Sharpe Ratio annualizzato."""
    returns = df['Close'].pct_change().dropna()
    if len(returns) < 2:
        return np.nan
    mean_ret = returns.mean() * 252
    std = returns.std() * np.sqrt(252)
    return (mean_ret - rf) / std if std != 0 else np.nan

def calculate_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
    """Calcola VaR storico al livello di confidenza."""
    returns = df['Close'].pct_change().dropna()
    if len(returns) < 2:
        return np.nan
    return returns.quantile(1 - confidence)

# ==================== FUNZIONI CORE ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol: str, interval: str = '1h') -> pd.DataFrame | None:
    """Carica dati con preferenza Bloomberg, fallback yfinance."""
    try:
        period = PERIOD_MAP.get(interval, '730d')
        if BLOOMBERG_AVAILABLE:
            # Usa Bloomberg API per fetch dati
            start_date = (datetime.datetime.now() - datetime.timedelta(days=int(period[:-1]) if period.endswith('d') else 365*int(period[:-1]) if period.endswith('y') else 730)).strftime('%Y-%m-%d')
            data = blp.bdh(tickers=symbol, flds=['open', 'high', 'low', 'last_price', 'volume'], start_date=start_date, end_date=datetime.datetime.now().strftime('%Y-%m-%d'))
            data.columns = data.columns.droplevel(1)  # Appiattisci multi-index
            data = data.rename(columns={'last_price': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
            logger.info(f"Dati caricati da Bloomberg per {symbol}")
        else:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            logger.info(f"Dati caricati da yfinance per {symbol}")
        
        if len(data) < 100:
            raise ValueError("Dati insufficienti per l'analisi.")
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Errore caricamento dati per {symbol}: {str(e)}")
        st.error(f"Errore caricamento dati: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def calculate_indicators(df: pd.DataFrame, symbol: str = 'DEFAULT') -> pd.DataFrame:
    config = ASSET_CONFIGS.get(symbol, ASSET_CONFIGS['DEFAULT'])
    df = df.copy()

    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_overbought'] = (df['RSI'] > 70).astype(int)

    rsi_min = df['RSI'].rolling(window=14).min()
    rsi_max = df['RSI'].rolling(window=14).max()
    df['StochRSI'] = ((df['RSI'] - rsi_min) / (rsi_max - rsi_min)) * 100

    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    df['MACD_crossover'] = ((df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)

    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_pct'] = (df['ATR'] / df['Close']) * 100 * config['volatility_mult']

    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    df['Volume_trend'] = df['Volume'].rolling(window=10).apply(lambda x: 1 if x[-1] > x[0] else 0)

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()

    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = -df['Low'].diff().clip(lower=0)
    atr = true_range.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()

    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['Senkou_B'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Cloud_position'] = np.where(df['Close'] > df['Senkou_A'], 1, np.where(df['Close'] < df['Senkou_B'], -1, 0))

    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

    df['Price_Change'] = df['Close'].pct_change()
    df['Trend_5'] = (df['Close'] > df['Close'].shift(5)).astype(int)
    df['Trend_20'] = (df['Close'] > df['Close'].shift(20)).astype(int)
    df['Trend_50'] = (df['Close'] > df['Close'].shift(50)).astype(int)
    df['Trend'] = (df['Trend_5'] + df['Trend_20'] + df['Trend_50']) / 3

    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']

    return df.dropna()

def generate_features(df_ind: pd.DataFrame, entry: float, sl: float, tp: float, direction: str, main_tf: int, symbol: str = 'DEFAULT') -> np.ndarray:
    latest = df_ind.iloc[-1]
    config = ASSET_CONFIGS.get(symbol, ASSET_CONFIGS['DEFAULT'])

    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100

    features_dict = {
        'sl_distance_pct': sl_distance,
        'tp_distance_pct': tp_distance,
        'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0,
        'main_tf': main_tf,
        'rsi': latest['RSI'],
        'rsi_oversold': latest['RSI_oversold'],
        'rsi_overbought': latest['RSI_overbought'],
        'stoch_rsi': latest['StochRSI'],
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_signal'],
        'macd_histogram': latest['MACD_histogram'],
        'macd_crossover': latest['MACD_crossover'],
        'atr': latest['ATR'],
        'atr_pct': latest['ATR_pct'],
        'bb_width': latest['BB_width'],
        'bb_pct': latest['BB_pct'],
        'ema_9_20': (latest['EMA_9'] - latest['EMA_20']) / latest['Close'] * 100,
        'ema_20_50': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'ema_50_200': (latest['EMA_50'] - latest['EMA_200']) / latest['Close'] * 100,
        'trend': latest['Trend'],
        'adx': latest['ADX'],
        'volume_ratio': latest['Volume_ratio'] * config['volume_weight'],
        'volume_trend': latest['Volume_trend'],
        'obv_signal': 1 if latest['OBV'] > latest['OBV_MA'] else 0,
        'momentum': latest['Momentum'],
        'roc': latest['ROC'],
        'cloud_position': latest['Cloud_position'],
        'price_change': latest['Price_Change'] * 100,
        'distance_to_pivot': (latest['Close'] - latest['Pivot']) / latest['Close'] * 100,
    }

    return np.array(list(features_dict.values()), dtype=np.float32)

@st.cache_data(show_spinner=False)
def simulate_trades(df_ind: pd.DataFrame, symbol: str = 'DEFAULT', n_trades: int = 800) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    config = ASSET_CONFIGS.get(symbol, ASSET_CONFIGS['DEFAULT'])

    for _ in range(n_trades):
        idx = np.random.randint(100, len(df_ind) - 100)
        row = df_ind.iloc[idx]

        rsi_signal = 1 if row['RSI'] < 40 else -1 if row['RSI'] > 60 else 0
        macd_signal = 1 if row['MACD_crossover'] == 1 else 0
        trend_signal = 1 if row['Trend'] > 0.6 else -1 if row['Trend'] < 0.4 else 0
        combined_signal = rsi_signal + macd_signal + trend_signal
        direction = 'long' if combined_signal >= 1 else 'short'

        entry = row['Close']
        atr = row['ATR']
        sl_mult = np.random.uniform(0.8, 2.0) * config['volatility_mult']
        tp_mult = np.random.uniform(1.5, 4.0) * config['volatility_mult']

        if direction == 'long':
            sl = entry - (atr * sl_mult)
            tp = entry + (atr * tp_mult)
        else:
            sl = entry + (atr * sl_mult)
            tp = entry - (atr * tp_mult)

        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60, symbol)

        future_window = min(100, len(df_ind) - idx - 1)
        future_prices = df_ind.iloc[idx+1:idx+1+future_window]['Close'].values

        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp)
                hit_sl = np.any(future_prices <= sl)
                tp_time = np.where(future_prices >= tp)[0][0] if hit_tp else future_window
                sl_time = np.where(future_prices <= sl)[0][0] if hit_sl else future_window
            else:
                hit_tp = np.any(future_prices <= tp)
                hit_sl = np.any(future_prices >= sl)
                tp_time = np.where(future_prices <= tp)[0][0] if hit_tp else future_window
                sl_time = np.where(future_prices >= sl)[0][0] if hit_sl else future_window

            success = 1 if (hit_tp and tp_time < sl_time) else 0
            X.append(features)
            y.append(success)

    return np.array(X), np.array(y)

@st.cache_resource(show_spinner=False)
def train_models(X: np.ndarray, y: np.ndarray) -> tuple[dict, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42)

    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)

    cv_score = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy').mean()
    logger.info(f"Modello addestrato con CV accuracy: {cv_score:.2f}")

    return {'rf': rf, 'gb': gb, 'cv_score': cv_score}, scaler

def predict_success(models: dict, scaler: StandardScaler, features: np.ndarray) -> float:
    features_scaled = scaler.transform(features.reshape(1, -1))
    rf_prob = models['rf'].predict_proba(features_scaled)[0][1]
    gb_prob = models['gb'].predict_proba(features_scaled)[0][1]
    return (rf_prob * 0.6 + gb_prob * 0.4) * 100

def get_dominant_factors(models: dict, features: np.ndarray) -> list[str]:
    feature_names = [
        'SL Distance %', 'TP Distance %', 'R/R Ratio', 'Direction', 'TimeFrame',
        'RSI', 'RSI Oversold', 'RSI Overbought', 'Stoch RSI',
        'MACD', 'MACD Signal', 'MACD Histogram', 'MACD Crossover',
        'ATR', 'ATR %', 'BB Width', 'BB Position',
        'EMA 9-20', 'EMA 20-50', 'EMA 50-200', 'Trend', 'ADX',
        'Volume Ratio', 'Volume Trend', 'OBV Signal',
        'Momentum', 'ROC', 'Cloud Position', 'Price Change %', 'Distance to Pivot'
    ]

    importances = models['rf'].feature_importances_
    indices = np.argsort(importances)[-7:][::-1]

    interpretations = {
        'RSI': 'Momentum del prezzo',
        'MACD': 'Convergenza medie mobili',
        'ATR': 'Volatilità mercato',
        'Volume Ratio': 'Forza volume',
        'Trend': 'Direzione trend',
        'ADX': 'Forza trend',
        'BB Position': 'Posizione bande'
    }

    factors = []
    for i in indices:
        fname = feature_names[i]
        interp = interpretations.get(fname.split()[0], 'Indicatore tecnico')
        factors.append(f"**{fname}**: {features[i]:.2f} ({interp}, peso: {importances[i]:.1%})")

    return factors

def get_sentiment(text: str, symbol: str) -> tuple[str, int]:
    positive = {'rally': 2, 'surge': 2, 'boom': 2, 'bullish': 2, 'soar': 2, 'up': 1, 'gain': 1, 'positive': 1, 'strong': 1, 'rise': 1, 'recovery': 1.5, 'breakout': 2, 'momentum': 1, 'optimistic': 1.5}
    negative = {'crash': 2, 'plunge': 2, 'bearish': 2, 'tumble': 2, 'collapse': 2, 'down': 1, 'loss': 1, 'negative': 1, 'weak': 1, 'fall': 1, 'decline': 1.5, 'breakdown': 2, 'pressure': 1, 'concern': 1.5}

    text_lower = text.lower()
    score = sum(positive.get(word, 0) for word in positive if word in text_lower) - sum(negative.get(word, 0) for word in negative if word in text_lower)

    if symbol in ['BTC-USD', 'GC=F', 'SI=F']:
        if 'institutional' in text_lower or 'adoption' in text_lower:
            score += 1
        if 'regulation' in text_lower or 'ban' in text_lower:
            score -= 1

    if score > 2: return 'Molto Positivo', score
    if score > 0: return 'Positivo', score
    if score < -2: return 'Molto Negativo', score
    if score < 0: return 'Negativo', score
    return 'Neutrale', 0

@st.cache_data(show_spinner=False)
def predict_price(df_ind: pd.DataFrame, steps: int = 10) -> tuple[float | None, np.ndarray | None]:
    try:
        last_price = df_ind['Close'].iloc[-1]
        ema_short = df_ind['EMA_9'].iloc[-1]
        ema_mid = df_ind['EMA_20'].iloc[-1]
        ema_long = df_ind['EMA_50'].iloc[-1]
        trend_strength = df_ind['Trend'].iloc[-1]

        forecast = np.array([last_price + ((ema_short * 0.5 + ema_mid * 0.3 + ema_long * 0.2) - last_price) * trend_strength * (i / steps) for i in range(1, steps + 1)])
        return forecast.mean(), forecast
    except Exception as e:
        logger.error(f"Errore previsione prezzo: {str(e)}")
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def get_web_signals(symbol: str, df_ind: pd.DataFrame) -> list[dict]:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        if hist.empty:
            raise ValueError("Dati storici vuoti")
        current_price = hist['Close'].iloc[-1]

        news = ticker.news
        news_summary = ' | '.join(item.get('title', '') for item in news[:5] if isinstance(item, dict)) or 'Nessuna news'

        sentiment_label, sentiment_score = get_sentiment(news_summary, symbol)

        hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
        if len(hist_monthly) >= 12:
            hist_monthly['Return'] = hist_monthly['Close'].pct_change()
            monthly_returns = hist_monthly.groupby(hist_monthly.index.month)['Return'].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            best_month = monthly_returns.idxmax()
            worst_month = monthly_returns.idxmin()
            seasonality_note = f"Mese corrente: {avg_current:+.2f}% medio. Migliore: {best_month}, Peggiore: {worst_month}"
        else:
            seasonality_note = 'Dati storici insufficienti'

        avg_forecast, _ = predict_price(df_ind)
        forecast_note = f"Target 10 periodi: ${avg_forecast:.2f} ({((avg_forecast - current_price) / current_price * 100 if avg_forecast else 0):+.2f}%)" if avg_forecast else 'N/A'

        latest = df_ind.iloc[-1]
        atr = latest['ATR']
        rsi = latest['RSI']
        trend = latest['Trend']
        adx = latest['ADX']

        suggestions = []

        long_score = (20 if rsi < 40 else 0) + (15 if trend > 0.6 else 0) + (10 if sentiment_score > 0 else 0) + (10 if adx > 25 else 0) + (15 if latest['MACD_crossover'] == 1 else 0)
        long_prob = min(85, max(50, 50 + long_score))
        entry = round(current_price, 2)
        sl_long = round(entry - atr * 1.2, 2)
        tp_long = round(entry + atr * 2.5, 2)
        suggestions.append({
            'Direction': 'Long' if '=X' not in symbol else 'Buy',
            'Entry': entry,
            'SL': sl_long,
            'TP': tp_long,
            'Probability': long_prob,
            'Seasonality_Note': seasonality_note,
            'News_Summary': news_summary,
            'Sentiment': sentiment_label,
            'Forecast_Note': forecast_note,
            'Technical_Score': long_score
        })

        short_score = (20 if rsi > 60 else 0) + (15 if trend < 0.4 else 0) + (10 if sentiment_score < 0 else 0) + (10 if adx > 25 else 0)
        short_prob = min(85, max(50, 50 + short_score))
        sl_short = round(entry + atr * 1.2, 2)
        tp_short = round(entry - atr * 2.5, 2)
        suggestions.append({
            'Direction': 'Short' if '=X' not in symbol else 'Sell',
            'Entry': entry,
            'SL': sl_short,
            'TP': tp_short,
            'Probability': short_prob,
            'Seasonality_Note': seasonality_note,
            'News_Summary': news_summary,
            'Sentiment': sentiment_label,
            'Forecast_Note': forecast_note,
            'Technical_Score': short_score
        })

        return sorted(suggestions, key=lambda x: x['Probability'], reverse=True)
    except Exception as e:
        logger.error(f"Errore segnali web per {symbol}: {str(e)}")
        return []

@st.cache_data(show_spinner=False)
def get_psychology(symbol: str, news_summary: str, sentiment_label: str, df_ind: pd.DataFrame) -> str:
    latest = df_ind.iloc[-1]
    trend = 'rialzista' if latest['Trend'] > 0.6 else 'ribassista' if latest['Trend'] < 0.4 else 'neutrale'
    volatility = 'alta' if latest['ATR_pct'] > 2 else 'moderata' if latest['ATR_pct'] > 1 else 'bassa'

    analysis = f"""
    ### 🧠 Contesto Psicologico 2025

    Per **{symbol}** (Trend: {trend}, Volatilità: {volatility}, Sentiment: {sentiment_label}):

    **Scenario Attuale:**
    - Mercati influenzati da tassi FED e geopolitica
    - Crypto (BTC): Volatilità post-ETF
    - Metalli (Oro/Argento): Safe haven
    - Indici (S&P 500): Rotazione settoriale

    **Bias Dominanti:**
    | Bias | Impatto | Evidenza 2025 |
    |------|---------|---------------|
    | Loss Aversion | {'Alto in bear' if trend == 'ribassista' else 'Moderato'} | ACR: -20% perf |
    | Herding | {'Critico in alta vol' if volatility == 'alta' else 'Presente'} | EPFR: $850B deflussi |
    | Recency Bias | {'Estremo con ADX alto' if latest['ADX'] > 30 else 'Moderato'} | EJBRM: buy high/sell low |
    | Overconfidence | {'Alto in crypto/tech' if 'BTC' in symbol else 'Moderato'} | JPM: +30% overtrading |

    **Raccomandazioni:**
    - DCA per timing
    - Stop-loss contro loss aversion
    - Rebalancing vs herding
    - ETF per diversificazione

    **Storico:**
    - 2008: Panic lento recovery
    - 2020: V-shape vs 2025 U-shape
    - 2000: Dot-com simile AI hype
    """
    return analysis

@st.cache_data(show_spinner=False)
def get_growth_assets() -> list[dict]:
    assets = [
        {"Nome": "Gold", "Ticker": "GC=F", "Score": 92, "Drivers": "Safe haven, banche centrali", "Risk": "Basso", "Timeframe": "3-12 mesi"},
        {"Nome": "Nvidia", "Ticker": "NVDA", "Score": 90, "Drivers": "AI/GPU, data center +50%", "Risk": "Medio-Alto", "Timeframe": "6-18 mesi"},
        {"Nome": "Bitcoin", "Ticker": "BTC-USD", "Score": 85, "Drivers": "ETF, halving, scarsità", "Risk": "Alto", "Timeframe": "12-24 mesi"},
        {"Nome": "Silver", "Ticker": "SI=F", "Score": 84, "Drivers": "Industriale (solar, EV)", "Risk": "Medio", "Timeframe": "6-18 mesi"},
        {"Nome": "Microsoft", "Ticker": "MSFT", "Score": 88, "Drivers": "Azure, AI integration", "Risk": "Basso-Medio", "Timeframe": "12-24 mesi"},
        {"Nome": "Broadcom", "Ticker": "AVGO", "Score": 86, "Drivers": "AI chips, FCF robusto", "Risk": "Medio", "Timeframe": "6-18 mesi"},
        {"Nome": "S&P 500 ETF", "Ticker": "SPY", "Score": 78, "Drivers": "Broad market, earnings", "Risk": "Medio", "Timeframe": "12-36 mesi"},
        {"Nome": "Uranium ETF", "Ticker": "URA", "Score": 82, "Drivers": "Nucleare, deficit supply", "Risk": "Alto", "Timeframe": "18-36 mesi"},
        {"Nome": "Palantir", "Ticker": "PLTR", "Score": 80, "Drivers": "AIP, gov contracts", "Risk": "Alto", "Timeframe": "12-24 mesi"},
        {"Nome": "Taiwan Semi", "Ticker": "TSM", "Score": 87, "Drivers": "Foundry, AI demand", "Risk": "Medio-Alto", "Timeframe": "12-24 mesi"}
    ]
    return sorted(assets, key=lambda x: x['Score'], reverse=True)

# ==================== APP STREAMLIT ====================
st.set_page_config(page_title="Trading Predictor AI Pro", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    .main .block-container { max-width: 1500px; padding-top: 1rem; }
    .stMetric { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stMetric label { color: white !important; }
    .stMetric [data-testid="stMetricValue"] { color: white !important; font-size: 1.8rem !important; }
    .trade-card { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; margin: 10px 0; }
    .high-prob { border-left-color: #10b981 !important; }
    .med-prob { border-left-color: #f59e0b !important; }
    .low-prob { border-left-color: #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Trading Predictor AI Pro")
st.markdown("**Analisi Avanzata Multi-Asset con Ensemble ML**")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("📈 Ticker", value="BTC-USD", help="BTC-USD, GC=F, SI=F, ^GSPC, etc.")
    proper_name = PROPER_NAMES.get(symbol, symbol)
    st.markdown(f"**Asset:** `{proper_name}`")
with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h', '1d'], index=2)
with col3:
    refresh_data = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner(f"🧠 Training AI per {proper_name}..."):
        data = load_data(symbol, data_interval)
        if data is None:
            st.stop()
        df_ind = calculate_indicators(data, symbol)
        X, y = simulate_trades(df_ind, symbol)
        models, scaler = train_models(X, y)
        st.session_state[session_key] = {'models': models, 'scaler': scaler, 'df_ind': df_ind}
        st.success(f"✅ Modello pronto! Accuracy CV: {models['cv_score']:.1%}")

if session_key in st.session_state:
    state = st.session_state[session_key]
    models = state['models']
    scaler = state['scaler']
    df_ind = state['df_ind']

    avg_forecast, _ = predict_price(df_ind)
    web_signals = get_web_signals(symbol, df_ind)

    # Aggiungi VaR e Sharpe
    sharpe = calculate_sharpe(df_ind)
    var_95 = calculate_var(df_ind, 0.95)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 💎 Setup Trading Raccomandati")
        if web_signals:
            for idx, trade in enumerate(web_signals[:3]):
                prob_class = 'high-prob' if trade['Probability'] >= 70 else 'med-prob' if trade['Probability'] >= 60 else 'low-prob'
                st.markdown(f"""
                <div class="trade-card {prob_class}">
                    <h4>🎯 Setup {idx+1}: {trade['Direction'].upper()}</h4>
                    <p><b>Entry:</b> ${trade['Entry']:.2f} | <b>SL:</b> ${trade['SL']:.2f} | <b>TP:</b> ${trade['TP']:.2f}</p>
                    <p><b>Probabilità:</b> {trade['Probability']:.0f}% | <b>Score Tecnico:</b> {trade['Technical_Score']}/70</p>
                    <p><b>Sentiment:</b> {trade['Sentiment']} | <b>R/R:</b> {abs(trade['TP']-trade['Entry'])/abs(trade['Entry']-trade['SL']):.2f}x</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"📊 Analizza con AI", key=f"analyze_{idx}"):
                    st.session_state.selected_trade = trade

            with st.expander("📰 News & Forecast"):
                st.markdown(f"**News:** {web_signals[0]['News_Summary']}")
                st.markdown(f"**Seasonality:** {web_signals[0]['Seasonality_Note']}")
                st.markdown(f"**Forecast:** {web_signals[0]['Forecast_Note']}")
        else:
            st.info("⏳ Nessun setup. Riprova.")

    with col_right:
        st.markdown("### 🏆 Top 10 Asset Growth 2025")
        for asset in get_growth_assets():
            score_color = '🟢' if asset['Score'] >= 85 else '🟡' if asset['Score'] >= 80 else '🟠'
            st.markdown(f"**{score_color} {asset['Nome']}** ({asset['Ticker']}) - Score: {asset['Score']}/100\n- 📈 {asset['Drivers']}\n- ⚠️ Risk: {asset['Risk']} | ⏰ {asset['Timeframe']}")

    st.markdown("---")
    st.markdown("### 📊 Market Snapshot")
    latest = df_ind.iloc[-1]
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1: st.metric("💰 Prezzo", f"${latest['Close']:.2f}")
    with col2: st.metric("📈 RSI", f"{latest['RSI']:.1f}", delta=f"{latest['RSI'] - 50:+.1f}")
    with col3: st.metric("💥 ATR %", f"{latest['ATR_pct']:.2f}%")
    with col4: st.metric("🎯 Trend", "Bull" if latest['Trend'] > 0.6 else "Bear" if latest['Trend'] < 0.4 else "Neutral")
    with col5: st.metric("💪 ADX", f"{latest['ADX']:.0f} ({'Strong' if latest['ADX'] > 25 else 'Weak'})")
    with col6: st.metric("🔮 Forecast", f"${avg_forecast:.2f}" if avg_forecast else "N/A", delta=f"{((avg_forecast - latest['Close']) / latest['Close'] * 100 if avg_forecast else 0):+.1f}%")
    with col7: st.metric("📊 Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
    with col8: st.metric("⚠️ VaR 95%", f"{var_95 * 100:.2f}%" if not np.isnan(var_95) else "N/A")

    if 'selected_trade' in st.session_state:
        trade = st.session_state.selected_trade
        st.markdown("---")
        st.markdown("## 🤖 Analisi AI")
        features = generate_features(df_ind, trade['Entry'], trade['SL'], trade['TP'], 'long' if trade['Direction'].lower() in ['long', 'buy'] else 'short', 60, symbol)
        ai_prob = predict_success(models, scaler, features)
        factors = get_dominant_factors(models, features)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🎲 AI Prob", f"{ai_prob:.1f}%", delta=f"{ai_prob - trade['Probability']:+.1f}% vs Web")
        with col2: st.metric("⚖️ R/R", f"{abs(trade['TP'] - trade['Entry']) / abs(trade['Entry'] - trade['SL']):.2f}x")
        with col3: st.metric("📉 Risk", f"{abs(trade['Entry'] - trade['SL']) / trade['Entry'] * 100:.2f}%")
        with col4: st.metric("📈 Reward", f"{abs(trade['TP'] - trade['Entry']) / trade['Entry'] * 100:.2f}%")

        st.markdown("### 🎯 Verdict")
        col_web, col_ai, col_final = st.columns(3)
        with col_web:
            st.markdown("**📡 Web**")
            if trade['Probability'] >= 70: st.success(f"✅ STRONG\n{trade['Probability']:.0f}%")
            elif trade['Probability'] >= 60: st.warning(f"⚠️ MODERATE\n{trade['Probability']:.0f}%")
            else: st.error(f"❌ WEAK\n{trade['Probability']:.0f}%")
        with col_ai:
            st.markdown("**🤖 AI**")
            if ai_prob >= 70: st.success(f"✅ STRONG\n{ai_prob:.1f}%")
            elif ai_prob >= 60: st.warning(f"⚠️ MODERATE\n{ai_prob:.1f}%")
            else: st.error(f"❌ WEAK\n{ai_prob:.1f}%")
        with col_final:
            st.markdown("**⚡ Consensus**")
            avg_prob = (trade['Probability'] + ai_prob) / 2
            agreement = abs(trade['Probability'] - ai_prob) <= 10
            if avg_prob >= 70 and agreement: st.success(f"🔥 ALTA\n{avg_prob:.1f}% (aligned)")
            elif avg_prob >= 60: st.warning(f"⚠️ MEDIA\n{avg_prob:.1f}%")
            else: st.error(f"❌ BASSA\n{avg_prob:.1f}%")
            if not agreement: st.info(f"ℹ️ Divergenza: {abs(ai_prob - trade['Probability']):.1f}%")

        st.markdown("### 🔬 Fattori AI")
        for factor in factors:
            st.markdown(f"- {factor}")

        st.markdown(f"**🎓 Accuracy Modello:** {models['cv_score']:.1%} (CV)")

        st.markdown("---")
        st.markdown(get_psychology(symbol, trade['News_Summary'], trade['Sentiment'], df_ind))

else:
    st.warning("⚠️ Carica dati per analisi.")

# Footer & Metodologia
st.markdown("---")
with st.expander("ℹ️ Metodologia & Disclaimer"):
    st.markdown("""
    ### 🧠 Metodologia
    - **ML Ensemble**: RF (60%) + GB (40%)
    - **Features**: 30+ indicatori (RSI, MACD, Ichimoku, ADX)
    - **Training**: 800 simulazioni realistiche
    - **Asset-Specific**: Config per crypto, metalli, indici
    - **Sentiment**: NLP contestuale
    - **Nuovo**: Sharpe Ratio & VaR per rischio/rendimento
    - **Dati**: Bloomberg API se disponibile, altrimenti yfinance

    ### ⚠️ Disclaimer
    Strumento educativo. NON consulenza finanziaria. Rischi elevati nel trading.
    """)
st.markdown("<div style='text-align: center; color: #888;'>🚀 Trading Predictor AI Pro | Powered by Ensemble ML</div>", unsafe_allow_html=True)
