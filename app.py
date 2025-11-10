import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== FUNZIONI CORE ====================
def calculate_technical_indicators(df):
    """Calcola indicatori tecnici."""
    df = df.copy()
   
    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
   
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
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

def get_gold_fundamental_factors():
    """Recupera fattori fondamentali che influenzano il prezzo dell'oro."""
    factors = {}
    
    try:
        # DXY (Dollar Index)
        dxy = yf.Ticker("DX-Y.NYB")
        dxy_hist = dxy.history(period="5d")
        if not dxy_hist.empty:
            factors['dxy_current'] = dxy_hist['Close'].iloc[-1]
            factors['dxy_change'] = ((dxy_hist['Close'].iloc[-1] - dxy_hist['Close'].iloc[0]) / dxy_hist['Close'].iloc[0]) * 100
        else:
            factors['dxy_current'] = 104.5
            factors['dxy_change'] = 0.2
        
        # Tassi interesse USA (10Y Treasury)
        tnx = yf.Ticker("^TNX")
        tnx_hist = tnx.history(period="5d")
        if not tnx_hist.empty:
            factors['yield_10y'] = tnx_hist['Close'].iloc[-1]
        else:
            factors['yield_10y'] = 4.35
        
        # VIX (Volatilità/Fear)
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if not vix_hist.empty:
            factors['vix'] = vix_hist['Close'].iloc[-1]
        else:
            factors['vix'] = 18.5
        
        # S&P 500 (Risk-on/Risk-off)
        spx = yf.Ticker("^GSPC")
        spx_hist = spx.history(period="20d")
        if not spx_hist.empty:
            factors['spx_momentum'] = ((spx_hist['Close'].iloc[-1] - spx_hist['Close'].iloc[0]) / spx_hist['Close'].iloc[0]) * 100
        else:
            factors['spx_momentum'] = 2.5
        
        # Silver (correlazione metalli preziosi)
        silver = yf.Ticker("SI=F")
        silver_hist = silver.history(period="20d")
        gold = yf.Ticker("GC=F")
        gold_hist = gold.history(period="20d")
        if not silver_hist.empty and not gold_hist.empty:
            factors['gold_silver_ratio'] = gold_hist['Close'].iloc[-1] / silver_hist['Close'].iloc[-1]
        else:
            factors['gold_silver_ratio'] = 85.0
        
        # Inflazione stimata (usando TIPS spread come proxy)
        try:
            tips = yf.Ticker("^FVX")
            tips_hist = tips.history(period="5d")
            if not tips_hist.empty:
                factors['inflation_expectations'] = factors['yield_10y'] - tips_hist['Close'].iloc[-1]
            else:
                factors['inflation_expectations'] = 2.3
        except:
            factors['inflation_expectations'] = 2.3
        
    except Exception as e:
        st.warning(f"Alcuni dati di mercato non disponibili, uso valori stimati: {e}")
        factors = {
            'dxy_current': 104.5,
            'dxy_change': 0.2,
            'yield_10y': 4.35,
            'vix': 18.5,
            'spx_momentum': 2.5,
            'gold_silver_ratio': 85.0,
            'inflation_expectations': 2.3
        }
    
    # Fattori geopolitici (score 0-10, basato su analisi qualitativa 2025)
    factors['geopolitical_risk'] = 7.5  # Medio Oriente, Ucraina, tensioni USA-Cina
    
    # Domanda banche centrali (tonnellate annue stimate)
    factors['central_bank_demand'] = 1050  # 2024-2025 trend
    
    # Sentiment retail (0-10)
    factors['retail_sentiment'] = 6.8
    
    return factors

def analyze_gold_historical_comparison(current_price, factors):
    """Analizza confronti storici e predice prezzo futuro dell'oro."""
    
    # Periodi storici di riferimento
    historical_periods = {
        '1971-1980': {
            'description': 'Bull Market Post-Bretton Woods',
            'start_price': 35,
            'end_price': 850,
            'gain_pct': 2329,
            'duration_years': 9,
            'avg_inflation': 8.5,
            'geopolitical': 8,
            'dollar_weak': True,
            'key_events': 'Fine gold standard, inflazione alta, crisi petrolio'
        },
        '2001-2011': {
            'description': 'Bull Market Post-Dot-Com e Crisi 2008',
            'start_price': 255,
            'end_price': 1920,
            'gain_pct': 653,
            'duration_years': 10,
            'avg_inflation': 2.8,
            'geopolitical': 7,
            'dollar_weak': True,
            'key_events': '9/11, guerre, QE, crisi finanziaria'
        },
        '2015-2020': {
            'description': 'Consolidamento e COVID Rally',
            'start_price': 1050,
            'end_price': 2070,
            'gain_pct': 97,
            'duration_years': 5,
            'avg_inflation': 1.8,
            'geopolitical': 6,
            'dollar_weak': False,
            'key_events': 'Tassi bassi, QE, pandemia'
        },
        '2022-2025': {
            'description': 'Era Inflazione Post-COVID e Tensioni',
            'start_price': 1800,
            'end_price': current_price,
            'gain_pct': ((current_price - 1800) / 1800) * 100,
            'duration_years': 3,
            'avg_inflation': 4.5,
            'geopolitical': 7.5,
            'dollar_weak': False,
            'key_events': 'Inflazione persistente, guerra Ucraina, crisi bancarie, dedollarizzazione'
        }
    }
    
    # Analisi contesto attuale (2025)
    current_context = {
        'inflation': factors['inflation_expectations'],
        'dollar_strength': 'Forte' if factors['dxy_current'] > 103 else 'Debole',
        'real_rates': factors['yield_10y'] - factors['inflation_expectations'],
        'risk_sentiment': 'Risk-Off' if factors['vix'] > 20 else 'Neutrale' if factors['vix'] > 15 else 'Risk-On',
        'geopolitical': factors['geopolitical_risk'],
        'central_bank': 'Compratori Netti' if factors['central_bank_demand'] > 500 else 'Venditori',
        'technical_trend': 'Bullish' if current_price > 2600 else 'Neutrale'
    }
    
    # Trova periodo storico più simile
    similarity_scores = {}
    
    for period, data in historical_periods.items():
        if period == '2022-2025':
            continue
        
        score = 0
        
        # Inflazione simile
        inflation_diff = abs(data['avg_inflation'] - factors['inflation_expectations'])
        score += max(0, 10 - inflation_diff * 2)
        
        # Geopolitica simile
        geo_diff = abs(data['geopolitical'] - factors['geopolitical_risk'])
        score += max(0, 10 - geo_diff * 2)
        
        # Dollar weakness
        current_dollar_weak = factors['dxy_current'] < 100
        if data['dollar_weak'] == current_dollar_weak:
            score += 15
        
        # Domanda banche centrali
        if factors['central_bank_demand'] > 800:
            score += 10
        
        similarity_scores[period] = score
    
    most_similar = max(similarity_scores, key=similarity_scores.get)
    similarity_pct = (similarity_scores[most_similar] / 45) * 100
    
    # Calcolo prezzo target basato su multipli metodi
    
    # Metodo 1: Proiezione da periodo simile
    similar_period = historical_periods[most_similar]
    annual_return = (similar_period['gain_pct'] / 100) / similar_period['duration_years']
    projection_1y = current_price * (1 + annual_return)
    
    # Metodo 2: Modello fattori fondamentali
    base_price = current_price
    
    # Dollar impact (inverso)
    if factors['dxy_change'] < 0:
        base_price *= 1.015  # Dollar debole = oro forte
    elif factors['dxy_change'] > 1:
        base_price *= 0.985  # Dollar forte = oro debole
    
    # Real rates impact
    if current_context['real_rates'] < 1:
        base_price *= 1.025  # Tassi reali bassi favoriscono oro
    elif current_context['real_rates'] > 2:
        base_price *= 0.98
    
    # VIX/Fear impact
    if factors['vix'] > 25:
        base_price *= 1.03  # Alta volatilità = flight to safety
    elif factors['vix'] < 15:
        base_price *= 0.99
    
    # Geopolitical premium
    geo_multiplier = 1 + (factors['geopolitical_risk'] / 100)
    base_price *= geo_multiplier
    
    # Central bank demand
    cb_multiplier = 1 + ((factors['central_bank_demand'] - 500) / 10000)
    base_price *= cb_multiplier
    
    # Inflazione
    inflation_multiplier = 1 + (factors['inflation_expectations'] / 100)
    base_price *= inflation_multiplier
    
    projection_fundamental = base_price
    
    # Metodo 3: Analisi tecnica avanzata
    gold_data = yf.Ticker("GC=F").history(period="1y")
    if not gold_data.empty:
        volatility = gold_data['Close'].pct_change().std() * np.sqrt(252)
        momentum = ((gold_data['Close'].iloc[-1] - gold_data['Close'].iloc[-20]) / gold_data['Close'].iloc[-20]) * 100
        
        projection_technical = current_price * (1 + (momentum / 100) * 1.5)
    else:
        projection_technical = current_price * 1.05
    
    # Metodo 4: Gold/Silver Ratio
    historical_avg_ratio = 70
    current_ratio = factors['gold_silver_ratio']
    
    if current_ratio > historical_avg_ratio:
        ratio_adjustment = 1.02  # Oro sovraperformante, possibile correzione o continua
    else:
        ratio_adjustment = 1.01
    
    projection_ratio = current_price * ratio_adjustment
    
    # Media ponderata delle proiezioni
    weights = [0.3, 0.35, 0.25, 0.1]  # Storico, Fondamentale, Tecnico, Ratio
    projections = [projection_1y, projection_fundamental, projection_technical, projection_ratio]
    
    target_price_1y = sum(w * p for w, p in zip(weights, projections))
    
    # Range di confidenza
    std_projections = np.std(projections)
    lower_bound = target_price_1y - std_projections
    upper_bound = target_price_1y + std_projections
    
    # Targets a 3, 6, 12 mesi
    target_3m = current_price + (target_price_1y - current_price) * 0.25
    target_6m = current_price + (target_price_1y - current_price) * 0.5
    
    # Confidence score (0-100)
    confidence = min(100, similarity_pct * 0.6 + 
                     (40 if factors['central_bank_demand'] > 800 else 20) +
                     (20 if factors['geopolitical_risk'] > 6 else 10))
    
    return {
        'current_price': current_price,
        'target_3m': target_3m,
        'target_6m': target_6m,
        'target_1y': target_price_1y,
        'range_low': lower_bound,
        'range_high': upper_bound,
        'most_similar_period': most_similar,
        'similarity_pct': similarity_pct,
        'period_data': historical_periods[most_similar],
        'current_context': current_context,
        'confidence': confidence,
        'key_drivers': {
            'Dollar Index': f"${factors['dxy_current']:.2f} ({factors['dxy_change']:+.2f}%)",
            'Tassi 10Y': f"{factors['yield_10y']:.2f}%",
            'Tassi Reali': f"{current_context['real_rates']:.2f}%",
            'VIX': f"{factors['vix']:.1f}",
            'Inflazione Attesa': f"{factors['inflation_expectations']:.2f}%",
            'Rischio Geopolitico': f"{factors['geopolitical_risk']}/10",
            'Domanda BC': f"{factors['central_bank_demand']} ton/anno",
            'Gold/Silver Ratio': f"{factors['gold_silver_ratio']:.1f}"
        },
        'historical_periods': historical_periods
    }

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    """Genera features per la predizione."""
    latest = df_ind.iloc[-1]
   
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100
   
    features = {
        'sl_distance_pct': sl_distance,
        'tp_distance_pct': tp_distance,
        'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0,
        'main_tf': main_tf,
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_signal'],
        'atr': latest['ATR'],
        'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
        'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        'price_change': latest['Price_Change'] * 100,
        'trend': latest['Trend']
    }
   
    return np.array(list(features.values()), dtype=np.float32)

def simulate_historical_trades(df_ind, n_trades=500):
    """Simula trade storici per training."""
    X_list = []
    y_list = []
   
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
    """Addestra il modello Random Forest."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
   
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_train)
   
    return model, scaler

def predict_success(model, scaler, features):
    """Predice probabilità di successo."""
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

def get_dominant_factors(model, features):
    """Identifica fattori dominanti."""
    feature_names = [
        'SL Distance %', 'TP Distance %', 'R/R Ratio', 'Direction', 'TimeFrame',
        'RSI', 'MACD', 'MACD Signal', 'ATR', 'EMA Diff %',
        'BB Position', 'Volume Ratio', 'Price Change %', 'Trend'
    ]
   
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
   
    factors = []
    for i in indices:
        if i < len(feature_names):
            factors.append(f"{feature_names[i]}: {features[i]:.2f} (importanza: {importances[i]:.2%})")
   
    return factors

def get_sentiment(text):
    """Semplice analisi sentiment basata su parole chiave."""
    positive_words = ['rally', 'up', 'bullish', 'gain', 'positive', 'strong', 'rise', 'surge', 'boom']
    negative_words = ['down', 'bearish', 'loss', 'negative', 'weak', 'slip', 'fall', 'drop', 'crash']
    score = sum(word in text.lower() for word in positive_words) - sum(word in text.lower() for word in negative_words)
    if score > 0:
        return 'Positive', score
    elif score < 0:
        return 'Negative', score
    else:
        return 'Neutral', 0

def predict_price(df_ind, steps=5):
    """Previsione prezzo semplice basata su EMA."""
    try:
        last_price = df_ind['Close'].iloc[-1]
        ema = df_ind['Close'].ewm(span=steps).mean().iloc[-1]
        forecast_values = [last_price + (ema - last_price) * (i / steps) for i in range(1, steps + 1)]
        forecast = np.array(forecast_values)
        return forecast.mean(), forecast
    except:
        return None, None

def get_investor_psychology(symbol, news_summary, sentiment_label, df_ind):
    """Analisi approfondita della psicologia dell'investitore."""
    latest = df_ind.iloc[-1]
    trend = 'bullish' if latest['Trend'] == 1 else 'bearish'
    
    current_analysis = f"""
    **🌍 Contesto Globale (Novembre 2025)**
    
    Nel contesto attuale, i mercati globali sono influenzati da inflazione persistente (al 3.5% negli USA), tensioni geopolitiche (es. Medio Oriente e Ucraina) e un boom dell'IA che ha spinto il NASDAQ oltre i 20,000 punti. La psicologia degli investitori è segnata da un mix di ottimismo tecnologico e ansia macroeconomica, con il VIX a livelli elevati (intorno a 25), indicando volatilità. Per {symbol}, con trend {trend} e sentiment {sentiment_label}, gli investitori mostrano overreazioni emotive, amplificate da social media e AI-driven trading.
    """
    
    biases_analysis = """
    ### 🧠 Analisi Approfondita dei Bias Comportamentali negli Investimenti (2025)
    
    I bias comportamentali causano perdite annue del 2-3% per retail investors (Morningstar, J.P. Morgan). Nel 2025, social media e algoritmi amplificano questi effetti, con un 'gap comportamentale' stimato al 4% in mercati volatili.
    
    | Bias Cognitivo | Impatto |
    |---------------|---------|
    | **Avversione alle Perdite** | Deflussi da fondi azionari >200 mld USD |
    | **Eccessiva Fiducia** | Amplificato da app, perdite in mercati instabili |
    | **Effetto Gregge** | Flash crash virali, afflussi obbligazionari 850 mld |
    | **Bias di Conferma** | Echo chamber AI causano bolle |
    | **Recency Bias** | Comprare alto, vendere basso |
    
    💡 **Raccomandazione**: Fondi indicizzati/ETF con rebalancing automatico outperformano strategie emotive, riducendo bias del 15-25%.
    """
    
    return current_analysis + biases_analysis

def get_web_signals(symbol, df_ind):
    """Funzione dinamica per ottenere segnali web aggiornati."""
    try:
        ticker = yf.Ticker(symbol)
        
        hist = ticker.history(period='1d')
        if hist.empty:
            return []
        current_price = hist['Close'].iloc[-1]
        
        news = ticker.news
        news_summary = ' | '.join([item.get('title', '') for item in news[:5] if isinstance(item, dict)]) if news and isinstance(news, list) else 'Nessuna news recente disponibile.'
        
        sentiment_label, sentiment_score = get_sentiment(news_summary)
        
        hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
        if len(hist_monthly) < 12:
            seasonality_note = 'Dati storici insufficienti per calcolare la stagionalità.'
        else:
            hist_monthly['Return'] = hist_monthly['Close'].pct_change()
            hist_monthly['Month'] = hist_monthly.index.month
            monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            seasonality_note = f'Il mese corrente ha un ritorno medio storico di {avg_current:.2f}%.'
        
        _, forecast_series = predict_price(df_ind, steps=5)
        forecast_note = f'Previsione media per i prossimi 5 periodi: {forecast_series.mean():.2f}' if forecast_series is not None else 'Previsione non disponibile.'
        
        latest = df_ind.iloc[-1]
        atr = latest['ATR']
        trend = latest['Trend']
        suggestions = []
        directions = ['Long', 'Short'] if '=X' not in symbol else ['Buy', 'Sell']
        
        for dir in directions:
            is_positive_dir = (dir in ['Long', 'Buy'] and (sentiment_score > 0 or trend == 1)) or (dir in ['Short', 'Sell'] and (sentiment_score < 0 or trend == 0))
            prob = 70 if is_positive_dir else 60
            entry = round(current_price, 2)
            sl_mult = 1.0 if is_positive_dir else 1.5
            tp_mult = 2.5 if is_positive_dir else 2.0
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
        
        if sentiment_score == 0:
            dir = directions[0] if trend == 1 else directions[1]
            entry = round(current_price, 2)
            sl_mult = 1.2
            tp_mult = 2.2
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
                'Probability': 65,
                'Seasonality_Note': seasonality_note,
                'News_Summary': news_summary,
                'Sentiment': sentiment_label,
                'Forecast_Note': forecast_note
            })
        
        return suggestions
    except Exception as e:
        st.error(f"Errore nel recupero dati web: {e}")
        return []

@st.cache_data
def load_sample_data(symbol, interval='1h'):
    """Carica dati reali da yfinance."""
    period_map = {
        '5m': '60d',
        '15m': '60d',
        '1h': '730d'
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
        st.error(f"Errore nel caricamento dati: {e}")
        return None

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    """Addestra il modello."""
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=500)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

proper_names = {
    'GC=F': 'XAU/USD (Gold)',
    'EURUSD=X': 'EUR/USD',
    'SI=F': 'XAG/USD (Silver)',
    'BTC-USD': 'BTC/USD',
    '^GSPC': 'S&P 500',
}

st.set_page_config(
    page_title="Trading Predictor AI - Gold Focus",
    page_icon="🥇",
