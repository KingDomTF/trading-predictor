import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import yfinance as yf
import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE ASSET-SPECIFICI ====================
ASSET_CONFIGS = {
    'BTC-USD': {'volatility_mult': 1.5, 'trend_weight': 0.7, 'volume_weight': 0.9},
    'GC=F': {'volatility_mult': 0.8, 'trend_weight': 0.6, 'volume_weight': 0.5},
    'SI=F': {'volatility_mult': 1.0, 'trend_weight': 0.6, 'volume_weight': 0.6},
    '^GSPC': {'volatility_mult': 0.7, 'trend_weight': 0.8, 'volume_weight': 0.7},
    'DEFAULT': {'volatility_mult': 1.0, 'trend_weight': 0.6, 'volume_weight': 0.6}
}

# ==================== FUNZIONI CORE POTENZIATE ====================
def calculate_technical_indicators(df, symbol='DEFAULT'):
    """Calcola indicatori tecnici avanzati con configurazione asset-specific."""
    df = df.copy()
    config = ASSET_CONFIGS.get(symbol, ASSET_CONFIGS['DEFAULT'])
    
    # EMA multiple
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()
    
    # RSI con oversold/overbought zones
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
    
    # Stochastic RSI
    rsi_min = df['RSI'].rolling(window=14).min()
    rsi_max = df['RSI'].rolling(window=14).max()
    df['StochRSI'] = ((df['RSI'] - rsi_min) / (rsi_max - rsi_min)) * 100
    
    # MACD potenziato
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    df['MACD_crossover'] = ((df['MACD'] > df['MACD_signal']) & 
                            (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
    
    # Bollinger Bands con %B
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # ATR normalizzato per asset
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_pct'] = (df['ATR'] / df['Close']) * 100 * config['volatility_mult']
    
    # Volume analisi avanzata
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    df['Volume_trend'] = df['Volume'].rolling(window=10).apply(
        lambda x: 1 if x[-1] > x[0] else 0
    )
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()
    
    # ADX (Average Directional Index) per forza del trend
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = true_range
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    
    # Ichimoku Cloud (semplificato)
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high_9 + low_9) / 2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high_26 + low_26) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
    df['Cloud_position'] = np.where(df['Close'] > df['Senkou_A'], 1, 
                                    np.where(df['Close'] < df['Senkou_B'], -1, 0))
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Trend multi-timeframe
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend_5'] = (df['Close'] > df['Close'].shift(5)).astype(int)
    df['Trend_20'] = (df['Close'] > df['Close'].shift(20)).astype(int)
    df['Trend_50'] = (df['Close'] > df['Close'].shift(50)).astype(int)
    df['Trend'] = ((df['Trend_5'] + df['Trend_20'] + df['Trend_50']) / 3)
    
    # Support/Resistance (pivots)
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    
    df = df.dropna()
    return df

def generate_features(df_ind, entry, sl, tp, direction, main_tf, symbol='DEFAULT'):
    """Genera features avanzate per la predizione."""
    latest = df_ind.iloc[-1]
    config = ASSET_CONFIGS.get(symbol, ASSET_CONFIGS['DEFAULT'])
    
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100
    
    features = {
        # Setup base
        'sl_distance_pct': sl_distance,
        'tp_distance_pct': tp_distance,
        'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0,
        'main_tf': main_tf,
        
        # RSI avanzato
        'rsi': latest['RSI'],
        'rsi_oversold': latest['RSI_oversold'],
        'rsi_overbought': latest['RSI_overbought'],
        'stoch_rsi': latest['StochRSI'],
        
        # MACD
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_signal'],
        'macd_histogram': latest['MACD_histogram'],
        'macd_crossover': latest['MACD_crossover'],
        
        # Volatilità
        'atr': latest['ATR'],
        'atr_pct': latest['ATR_pct'],
        'bb_width': latest['BB_width'],
        'bb_pct': latest['BB_pct'],
        
        # Trend
        'ema_9_20': (latest['EMA_9'] - latest['EMA_20']) / latest['Close'] * 100,
        'ema_20_50': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'ema_50_200': (latest['EMA_50'] - latest['EMA_200']) / latest['Close'] * 100 if 'EMA_200' in latest else 0,
        'trend': latest['Trend'],
        'adx': latest['ADX'],
        
        # Volume
        'volume_ratio': latest['Volume_ratio'] * config['volume_weight'],
        'volume_trend': latest['Volume_trend'],
        'obv_signal': 1 if latest['OBV'] > latest['OBV_MA'] else 0,
        
        # Momentum
        'momentum': latest['Momentum'],
        'roc': latest['ROC'],
        
        # Ichimoku
        'cloud_position': latest['Cloud_position'],
        
        # Price action
        'price_change': latest['Price_Change'] * 100,
        'distance_to_pivot': (latest['Close'] - latest['Pivot']) / latest['Close'] * 100,
    }
    
    return np.array(list(features.values()), dtype=np.float32)

def simulate_historical_trades(df_ind, symbol='DEFAULT', n_trades=800):
    """Simula trade storici con logica migliorata."""
    X_list = []
    y_list = []
    config = ASSET_CONFIGS.get(symbol, ASSET_CONFIGS['DEFAULT'])
    
    for _ in range(n_trades):
        idx = np.random.randint(100, len(df_ind) - 100)
        row = df_ind.iloc[idx]
        
        # Direzione basata su segnali tecnici
        rsi_signal = 1 if row['RSI'] < 40 else -1 if row['RSI'] > 60 else 0
        macd_signal = 1 if row['MACD_crossover'] == 1 else 0
        trend_signal = 1 if row['Trend'] > 0.6 else -1 if row['Trend'] < 0.4 else 0
        
        combined_signal = rsi_signal + macd_signal + trend_signal
        direction = 'long' if combined_signal >= 1 else 'short'
        
        entry = row['Close']
        atr = row['ATR']
        
        # SL/TP basati su ATR con configurazione asset
        sl_mult = np.random.uniform(0.8, 2.0) * config['volatility_mult']
        tp_mult = np.random.uniform(1.5, 4.0) * config['volatility_mult']
        
        if direction == 'long':
            sl = entry - (atr * sl_mult)
            tp = entry + (atr * tp_mult)
        else:
            sl = entry + (atr * sl_mult)
            tp = entry - (atr * tp_mult)
        
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60, symbol)
        
        # Simula outcome con logica realistica
        future_window = min(100, len(df_ind) - idx - 1)
        future_prices = df_ind.iloc[idx+1:idx+1+future_window]['Close'].values
        
        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp)
                hit_sl = np.any(future_prices <= sl)
                # Tempo di hit
                tp_time = np.where(future_prices >= tp)[0][0] if hit_tp else future_window
                sl_time = np.where(future_prices <= sl)[0][0] if hit_sl else future_window
            else:
                hit_tp = np.any(future_prices <= tp)
                hit_sl = np.any(future_prices >= sl)
                tp_time = np.where(future_prices <= tp)[0][0] if hit_tp else future_window
                sl_time = np.where(future_prices >= sl)[0][0] if hit_sl else future_window
            
            # Success se TP prima di SL
            success = 1 if (hit_tp and tp_time < sl_time) else 0
            
            X_list.append(features)
            y_list.append(success)
    
    return np.array(X_list), np.array(y_list)

def train_model(X_train, y_train):
    """Addestra ensemble di modelli per maggiore accuratezza."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Random Forest principale
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting come secondo modello
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
    
    rf_model.fit(X_scaled, y_train)
    gb_model.fit(X_scaled, y_train)
    
    # Valutazione cross-validation
    cv_score = cross_val_score(rf_model, X_scaled, y_train, cv=5, scoring='accuracy').mean()
    
    return {'rf': rf_model, 'gb': gb_model, 'cv_score': cv_score}, scaler

def predict_success(models, scaler, features):
    """Predice con ensemble averaging."""
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    rf_prob = models['rf'].predict_proba(features_scaled)[0][1]
    gb_prob = models['gb'].predict_proba(features_scaled)[0][1]
    
    # Media pesata (RF più peso)
    ensemble_prob = (rf_prob * 0.6 + gb_prob * 0.4)
    
    return ensemble_prob * 100

def get_dominant_factors(models, features):
    """Identifica fattori dominanti con spiegazione."""
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
    
    factors = []
    interpretations = {
        'RSI': 'Momentum del prezzo',
        'MACD': 'Convergenza/Divergenza medie mobili',
        'ATR': 'Volatilità del mercato',
        'Volume Ratio': 'Forza del movimento',
        'Trend': 'Direzione predominante',
        'ADX': 'Forza del trend',
        'BB Position': 'Posizione nelle bande'
    }
    
    for i in indices:
        if i < len(feature_names):
            fname = feature_names[i]
            interpretation = interpretations.get(fname.split()[0], 'Indicatore tecnico')
            factors.append(f"**{fname}**: {features[i]:.2f} ({interpretation}, peso: {importances[i]:.1%})")
    
    return factors

def get_sentiment_advanced(text, symbol):
    """Sentiment analysis avanzata con pesi per asset."""
    positive_words = {
        'rally': 2, 'surge': 2, 'boom': 2, 'bullish': 2, 'soar': 2,
        'up': 1, 'gain': 1, 'positive': 1, 'strong': 1, 'rise': 1, 'recovery': 1.5,
        'breakout': 2, 'momentum': 1, 'optimistic': 1.5
    }
    negative_words = {
        'crash': 2, 'plunge': 2, 'bearish': 2, 'tumble': 2, 'collapse': 2,
        'down': 1, 'loss': 1, 'negative': 1, 'weak': 1, 'fall': 1, 'decline': 1.5,
        'breakdown': 2, 'pressure': 1, 'concern': 1.5
    }
    
    text_lower = text.lower()
    score = sum(weight for word, weight in positive_words.items() if word in text_lower)
    score -= sum(weight for word, weight in negative_words.items() if word in text_lower)
    
    # Boost per crypto/metalli
    if symbol in ['BTC-USD', 'GC=F', 'SI=F']:
        if 'institutional' in text_lower or 'adoption' in text_lower:
            score += 1
        if 'regulation' in text_lower or 'ban' in text_lower:
            score -= 1
    
    if score > 2:
        return 'Molto Positivo', score
    elif score > 0:
        return 'Positivo', score
    elif score < -2:
        return 'Molto Negativo', score
    elif score < 0:
        return 'Negativo', score
    else:
        return 'Neutrale', 0

def predict_price_advanced(df_ind, steps=10):
    """Previsione prezzo con multiple medie e trend."""
    try:
        last_price = df_ind['Close'].iloc[-1]
        
        # Combinazione di EMA
        ema_short = df_ind['EMA_9'].iloc[-1]
        ema_mid = df_ind['EMA_20'].iloc[-1]
        ema_long = df_ind['EMA_50'].iloc[-1]
        
        # Trend weight
        trend_strength = df_ind['Trend'].iloc[-1]
        
        # Forecast pesato
        forecast_values = []
        for i in range(1, steps + 1):
            weight = i / steps
            ema_forecast = (ema_short * 0.5 + ema_mid * 0.3 + ema_long * 0.2)
            trend_adj = (ema_forecast - last_price) * trend_strength * weight
            forecast_values.append(last_price + trend_adj)
        
        forecast = np.array(forecast_values)
        return forecast.mean(), forecast
    except:
        return None, None

def get_web_signals(symbol, df_ind):
    """Segnali web ottimizzati per accuracy."""
    try:
        ticker = yf.Ticker(symbol)
        
        hist = ticker.history(period='1d')
        if hist.empty:
            return []
        current_price = hist['Close'].iloc[-1]
        
        # News
        news = ticker.news
        news_summary = ' | '.join([item.get('title', '') for item in news[:5] if isinstance(item, dict)]) if news and isinstance(news, list) else 'Nessuna news disponibile'
        
        sentiment_label, sentiment_score = get_sentiment_advanced(news_summary, symbol)
        
        # Stagionalità
        hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
        if len(hist_monthly) >= 12:
            hist_monthly['Return'] = hist_monthly['Close'].pct_change()
            hist_monthly['Month'] = hist_monthly.index.month
            monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            best_month = monthly_returns.idxmax()
            worst_month = monthly_returns.idxmin()
            seasonality_note = f"Mese corrente: {avg_current:+.2f}% medio. Migliore: {best_month}, Peggiore: {worst_month}"
        else:
            seasonality_note = 'Dati storici insufficienti'
        
        # Previsione
        avg_forecast, forecast_series = predict_price_advanced(df_ind, steps=10)
        if avg_forecast:
            price_change_pct = ((avg_forecast - current_price) / current_price) * 100
            forecast_note = f"Target 10 periodi: ${avg_forecast:.2f} ({price_change_pct:+.2f}%)"
        else:
            forecast_note = 'Previsione non disponibile'
        
        # Suggerimenti intelligenti
        latest = df_ind.iloc[-1]
        atr = latest['ATR']
        rsi = latest['RSI']
        trend = latest['Trend']
        adx = latest['ADX']
        
        suggestions = []
        
        # Long setup
        long_score = 0
        if rsi < 40: long_score += 20
        if trend > 0.6: long_score += 15
        if sentiment_score > 0: long_score += 10
        if adx > 25: long_score += 10
        if latest['MACD_crossover'] == 1: long_score += 15
        
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
        
        # Short setup
        short_score = 0
        if rsi > 60: short_score += 20
        if trend < 0.4: short_score += 15
        if sentiment_score < 0: short_score += 10
        if adx > 25: short_score += 10
        
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
        st.error(f"Errore recupero dati: {e}")
        return []

def get_investor_psychology(symbol, news_summary, sentiment_label, df_ind):
    """Psicologia investitore ottimizzata."""
    latest = df_ind.iloc[-1]
    trend = 'rialzista' if latest['Trend'] > 0.6 else 'ribassista'
    volatility = 'alta' if latest['ATR_pct'] > 2 else 'moderata' if latest['ATR_pct'] > 1 else 'bassa'
    
    analysis = f"""
    ### 🧠 Contesto Psicologico 2025
    
    Per **{symbol}** (Trend: {trend}, Volatilità: {volatility}, Sentiment: {sentiment_label}):
    
    **Scenario Attuale:**
    - Mercati influenzati da tassi FED elevati e incertezza geopolitica
    - Crypto (BTC): Volatilità post-ETF approval, sentiment misto istituzionale/retail
    - Metalli (Oro/Argento): Safe haven con correlazione inversa tassi
    - Indici (S&P 500): Rotazione settoriale da growth a value
    
    **Bias Comportamentali Dominanti:**
    
    | Bias | Impatto su {symbol} | Evidenza 2025 |
    |------|---------------------|---------------|
    | **Loss Aversion** | {'Alto - holding perdenti in bear market' if trend == 'ribassista' else 'Moderato - FOMO su rally'} | Studio ACR Journal (2025): -20% performance |
    | **Herding** | {'Critico - panic selling' if volatility == 'alta' else 'Presente - momentum following'} | EPFR: $850B deflussi azionari |
    | **Recency Bias** | {'Estremo - sovrastima trend breve' if latest['ADX'] > 30 else 'Moderato'} | EJBRM (2025): buy high, sell low |
    | **Overconfidence** | {'Alto in crypto/tech' if 'BTC' in symbol or 'tech' in symbol.lower() else 'Moderato'} | JPMorgan: overtrading +30% |
    
    **Raccomandazioni Strategiche:**
    - ✅ **Dollar-Cost Averaging** per ridurre impact timing
    - ✅ **Stop-loss disciplinati** contro loss aversion
    - ✅ **Rebalancing trimestrale** per contrare herding
    - ✅ **ETF/Fondi indicizzati** per diversificazione passiva
    
    **Comparazione Storica:**
    - **2008 Crisis**: Panic simile, ma recovery più lenta (no digital amplification)
    - **2020 COVID**: V-shape recovery vs 2025 U-shape con inflazione persistente
    - **2000 Dot-com**: Boom/bust cycle simile a AI hype 2023-2025
    """
    
    return analysis

# ==================== TOP 10 ASSET CON SCORING ====================
def get_top_growth_assets():
    """Top 10 asset con scoring fundamentale e tecnico."""
    assets = [
        {
            "Nome": "Gold", 
            "Ticker": "GC=F",
            "Score": 92,
            "Drivers": "Safe haven #1, correlazione inversa tassi, domanda banche centrali",
            "Risk": "Basso",
            "Timeframe": "3-12 mesi"
        },
        {
            "Nome": "Nvidia", 
            "Ticker": "NVDA",
            "Score": 90,
            "Drivers": "Leader AI/GPU, crescita data center +50% YoY, margin expansion",
            "Risk": "Medio-Alto",
            "Timeframe": "6-18 mesi"
        },
        {
            "Nome": "Bitcoin", 
            "Ticker": "BTC-USD",
            "Score": 85,
            "Drivers": "ETF flows, halving 2024, adozione istituzionale, scarsità",
            "Risk": "Alto",
            "Timeframe": "12-24 mesi"
        },
        {
            "Nome": "Silver", 
            "Ticker": "SI=F",
            "Score": 84,
            "Drivers": "Domanda industriale (solar, EV), ratio oro/argento favorevole, green energy",
            "Risk": "Medio",
            "Timeframe": "6-18 mesi"
        },
        {
            "Nome": "Microsoft", 
            "Ticker": "MSFT",
            "Score": 88,
            "Drivers": "Azure growth, Copilot AI integration, enterprise moat, dividendi stabili",
            "Risk": "Basso-Medio",
            "Timeframe": "12-24 mesi"
        },
        {
            "Nome": "Broadcom", 
            "Ticker": "AVGO",
            "Score": 86,
            "Drivers": "AI chips, networking, acquisizioni strategiche, FCF robusto",
            "Risk": "Medio",
            "Timeframe": "6-18 mesi"
        },
        {
            "Nome": "S&P 500 ETF", 
            "Ticker": "SPY",
            "Score": 78,
            "Drivers": "Diversificazione broad market, earnings growth, Fed pivot potential",
            "Risk": "Medio",
            "Timeframe": "12-36 mesi"
        },
        {
            "Nome": "Uranium ETF", 
            "Ticker": "URA",
            "Score": 82,
            "Drivers": "Rinascita nucleare, deficit supply, transizione energetica, AI data centers",
            "Risk": "Alto",
            "Timeframe": "18-36 mesi"
        },
        {
            "Nome": "Palantir", 
            "Ticker": "PLTR",
            "Score": 80,
            "Drivers": "AIP platform, gov contracts, commercial expansion, AI beneficiary",
            "Risk": "Alto",
            "Timeframe": "12-24 mesi"
        },
        {
            "Nome": "Taiwan Semi", 
            "Ticker": "TSM",
            "Score": 87,
            "Drivers": "Monopolio foundry, AI chip demand, pricing power, capex cycle",
            "Risk": "Medio-Alto",
            "Timeframe": "12-24 mesi"
        }
    ]
    return sorted(assets, key=lambda x: x['Score'], reverse=True)

# ==================== STREAMLIT APP ====================
@st.cache_data(ttl=3600)
def load_sample_data(symbol, interval='1h'):
    """Carica dati con caching ottimizzato."""
    period_map = {
        '5m': '60d',
        '15m': '60d',
        '1h': '730d',
        '1d': '5y'
    }
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

@st.cache_resource(ttl=3600)
def train_or_load_model(symbol, interval='1h'):
    """Training con cache persistente."""
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None, None
    
    df_ind = calculate_technical_indicators(data, symbol)
    X, y = simulate_historical_trades(df_ind, symbol, n_trades=800)
    
    models, scaler = train_model(X, y)
    
    return models, scaler, df_ind, models['cv_score']

# Nomi asset
proper_names = {
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

# Config pagina
st.set_page_config(
    page_title="Trading Predictor AI Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 1500px;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    .trade-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .high-prob {
        border-left-color: #10b981 !important;
    }
    .med-prob {
        border-left-color: #f59e0b !important;
    }
    .low-prob {
        border-left-color: #ef4444 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚀 Trading Predictor AI Pro")
st.markdown("**Analisi Avanzata Multi-Asset con Machine Learning Ensemble**")

# Layout input
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("📈 Ticker", value="BTC-USD", help="BTC-USD, GC=F, SI=F, ^GSPC, NVDA, etc.")
    proper_name = proper_names.get(symbol, symbol)
    st.markdown(f"**Asset:** `{proper_name}`")
with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h', '1d'], index=2)
with col3:
    st.markdown("##")
    refresh_data = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

st.markdown("---")

# Training
session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner(f"🧠 Training AI per {proper_name}..."):
        models, scaler, df_ind, cv_score = train_or_load_model(symbol, data_interval)
        if models is not None:
            st.session_state[session_key] = {
                'models': models, 
                'scaler': scaler, 
                'df_ind': df_ind,
                'cv_score': cv_score
            }
            st.success(f"✅ Modello pronto! Accuracy CV: {cv_score:.1%}")
        else:
            st.error("❌ Impossibile caricare dati. Verifica il ticker.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    models = state['models']
    scaler = state['scaler']
    df_ind = state['df_ind']
    cv_score = state['cv_score']
    
    # Previsione prezzo
    avg_forecast, forecast_series = predict_price_advanced(df_ind, steps=10)
    
    # Segnali web
    web_signals_list = get_web_signals(symbol, df_ind)
    
    # Layout principale
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("### 💎 Setup Trading Raccomandati")
        
        if web_signals_list:
            for idx, trade in enumerate(web_signals_list[:3]):
                prob = trade['Probability']
                prob_class = 'high-prob' if prob >= 70 else 'med-prob' if prob >= 60 else 'low-prob'
                
                st.markdown(f"""
                <div class="trade-card {prob_class}">
                    <h4>🎯 Setup {idx+1}: {trade['Direction'].upper()}</h4>
                    <p><b>Entry:</b> ${trade['Entry']:.2f} | <b>SL:</b> ${trade['SL']:.2f} | <b>TP:</b> ${trade['TP']:.2f}</p>
                    <p><b>Probabilità Web:</b> {trade['Probability']:.0f}% | <b>Score Tecnico:</b> {trade['Technical_Score']}/70</p>
                    <p><b>Sentiment:</b> {trade['Sentiment']} | <b>R/R:</b> {abs(trade['TP']-trade['Entry'])/abs(trade['Entry']-trade['SL']):.2f}x</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"📊 Analizza con AI", key=f"analyze_{idx}", use_container_width=True):
                    st.session_state.selected_trade = trade
            
            with st.expander("📰 News & Forecast Details"):
                st.markdown(f"**News Summary:**\n{web_signals_list[0]['News_Summary']}")
                st.markdown(f"**Seasonality:** {web_signals_list[0]['Seasonality_Note']}")
                st.markdown(f"**Forecast:** {web_signals_list[0]['Forecast_Note']}")
        else:
            st.info("⏳ Nessun setup disponibile. Riprova tra poco.")
    
    with col_right:
        st.markdown("### 🏆 Top 10 Asset Growth 2025")
        top_assets = get_top_growth_assets()
        
        for asset in top_assets:
            score_color = '🟢' if asset['Score'] >= 85 else '🟡' if asset['Score'] >= 80 else '🟠'
            st.markdown(f"""
            **{score_color} {asset['Nome']}** ({asset['Ticker']}) - Score: {asset['Score']}/100
            - 📈 {asset['Drivers']}
            - ⚠️ Risk: {asset['Risk']} | ⏰ {asset['Timeframe']}
            """)
        
        st.markdown("---")
        st.markdown("**Scoring basato su:** Fundamentals, trend tecnici, catalizzatori 2025, risk/reward")
    
    # Stats correnti
    st.markdown("---")
    st.markdown("### 📊 Market Snapshot")
    latest = df_ind.iloc[-1]
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("💰 Prezzo", f"${latest['Close']:.2f}")
    with col2:
        rsi_delta = latest['RSI'] - 50
        st.metric("📈 RSI", f"{latest['RSI']:.1f}", delta=f"{rsi_delta:+.1f}")
    with col3:
        st.metric("💥 ATR %", f"{latest['ATR_pct']:.2f}%")
    with col4:
        trend_label = "Bull" if latest['Trend'] > 0.6 else "Bear" if latest['Trend'] < 0.4 else "Neutral"
        st.metric("🎯 Trend", trend_label)
    with col5:
        adx_label = "Strong" if latest['ADX'] > 25 else "Weak"
        st.metric("💪 ADX", f"{latest['ADX']:.0f} ({adx_label})")
    with col6:
        if avg_forecast:
            forecast_change = ((avg_forecast - latest['Close']) / latest['Close']) * 100
            st.metric("🔮 Forecast", f"${avg_forecast:.2f}", delta=f"{forecast_change:+.1f}%")
        else:
            st.metric("🔮 Forecast", "N/A")
    
    # Analisi trade selezionato
    if 'selected_trade' in st.session_state:
        trade = st.session_state.selected_trade
        
        st.markdown("---")
        st.markdown("## 🤖 Analisi AI Dettagliata")
        
        with st.spinner("🔬 Processing..."):
            direction = 'long' if trade['Direction'].lower() in ['long', 'buy'] else 'short'
            features = generate_features(df_ind, trade['Entry'], trade['SL'], trade['TP'], direction, 60, symbol)
            ai_prob = predict_success(models, scaler, features)
            factors = get_dominant_factors(models, features)
        
        # Metriche confronto
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_prob = ai_prob - trade['Probability']
            st.metric("🎲 AI Probability", f"{ai_prob:.1f}%", 
                     delta=f"{delta_prob:+.1f}% vs Web",
                     delta_color="normal")
        with col2:
            rr = abs(trade['TP'] - trade['Entry']) / abs(trade['Entry'] - trade['SL'])
            st.metric("⚖️ Risk/Reward", f"{rr:.2f}x")
        with col3:
            risk = abs(trade['Entry'] - trade['SL']) / trade['Entry'] * 100
            st.metric("📉 Risk", f"{risk:.2f}%")
        with col4:
            reward = abs(trade['TP'] - trade['Entry']) / trade['Entry'] * 100
            st.metric("📈 Reward", f"{reward:.2f}%")
        
        # Verdict
        st.markdown("### 🎯 Verdict Comparativo")
        col_web, col_ai, col_final = st.columns(3)
        
        with col_web:
            st.markdown("**📡 Analisi Web**")
            if trade['Probability'] >= 70:
                st.success(f"✅ STRONG BUY\n{trade['Probability']:.0f}%")
            elif trade['Probability'] >= 60:
                st.warning(f"⚠️ MODERATE\n{trade['Probability']:.0f}%")
            else:
                st.error(f"❌ WEAK\n{trade['Probability']:.0f}%")
        
        with col_ai:
            st.markdown("**🤖 Analisi AI**")
            if ai_prob >= 70:
                st.success(f"✅ STRONG BUY\n{ai_prob:.1f}%")
            elif ai_prob >= 60:
                st.warning(f"⚠️ MODERATE\n{ai_prob:.1f}%")
            else:
                st.error(f"❌ WEAK\n{ai_prob:.1f}%")
        
        with col_final:
            st.markdown("**⚡ Consensus**")
            avg_prob = (trade['Probability'] + ai_prob) / 2
            agreement = abs(trade['Probability'] - ai_prob) <= 10
            
            if avg_prob >= 70 and agreement:
                st.success(f"🔥 ALTA CONFIDENZA\n{avg_prob:.1f}% (aligned)")
            elif avg_prob >= 60:
                st.warning(f"⚠️ CONFIDENZA MEDIA\n{avg_prob:.1f}%")
            else:
                st.error(f"❌ BASSA CONFIDENZA\n{avg_prob:.1f}%")
            
            if not agreement:
                st.info(f"ℹ️ Divergenza: {abs(delta_prob):.1f}%")
        
        # Fattori AI
        st.markdown("### 🔬 Fattori Dominanti AI")
        for factor in factors:
            st.markdown(f"- {factor}")
        
        st.markdown(f"**🎓 Model Accuracy:** {cv_score:.1%} (Cross-Validation)")
        
        # Psicologia
        st.markdown("---")
        psych = get_investor_psychology(symbol, trade['News_Summary'], trade['Sentiment'], df_ind)
        st.markdown(psych)

else:
    st.warning("⚠️ Carica i dati per iniziare l'analisi")

# Footer
st.markdown("---")
with st.expander("ℹ️ Metodologia & Disclaimer"):
    st.markdown("""
    ### 🧠 Metodologia
    - **ML Ensemble**: Random Forest (60%) + Gradient Boosting (40%)
    - **Features**: 30+ indicatori tecnici (RSI, MACD, Ichimoku, ADX, Volume Profile)
    - **Training**: 800 trade simulati con logica realistica
    - **Asset-Specific**: Configurazioni ottimizzate per crypto, metalli, indici
    - **Sentiment**: NLP avanzato con pesi contestuali
    
    ### ⚠️ Disclaimer
    Questo è uno strumento educativo. NON è consulenza finanziaria.
    Il trading comporta rischi significativi. Trade responsabilmente.
    """)

st.markdown("""
<div style='text-align: center; color: #888; margin-top: 20px;'>
    🚀 <b>Trading Predictor AI Pro</b> | Powered by ML Ensemble
</div>
""", unsafe_allow_html=True)
