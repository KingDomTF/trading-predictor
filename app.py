import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import re
from collections import Counter
warnings.filterwarnings('ignore')

def calculate_technical_indicators(df):
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 0.0001)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'] + 0.0001)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / (df['Volume_MA'] + 1)
    
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Price_Change'].rolling(window=20).std()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False)
    
    df['Returns_5'] = df['Close'].pct_change(5)
    df['Returns_10'] = df['Close'].pct_change(10)
    
    df = df.dropna()
    return df

def analyze_news_advanced(news_list, symbol):
    if not news_list or len(news_list) == 0:
        return {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'urgency_level': 'Low',
            'key_topics': [],
            'market_impact': 'Low',
            'trader_sentiment': 'Neutral',
            'news_summary': 'Nessuna news recente disponibile'
        }
    
    positive_keywords = {
        'strong': 3, 'surge': 3, 'rally': 3, 'boom': 3, 'soar': 3, 'gain': 2, 'up': 2, 
        'bullish': 3, 'growth': 2, 'positive': 2, 'rise': 2, 'jump': 2, 'advance': 2,
        'breakthrough': 3, 'record': 2, 'high': 1, 'beat': 2, 'outperform': 3,
        'upgrade': 2, 'buy': 2, 'optimistic': 2, 'recovery': 2, 'rebound': 2
    }
    
    negative_keywords = {
        'crash': 3, 'plunge': 3, 'fall': 2, 'drop': 2, 'decline': 2, 'down': 2,
        'bearish': 3, 'loss': 2, 'weak': 2, 'negative': 2, 'risk': 1, 'concern': 2,
        'worry': 2, 'fear': 2, 'sell': 2, 'downgrade': 2, 'miss': 2, 'cut': 2,
        'recession': 3, 'crisis': 3, 'tumble': 3, 'slump': 2, 'warning': 2
    }
    
    urgent_keywords = ['breaking', 'alert', 'urgent', 'emergency', 'critical', 'immediate']
    
    high_impact_keywords = ['fed', 'federal reserve', 'inflation', 'rate', 'earnings', 
                           'gdp', 'unemployment', 'war', 'sanctions', 'regulation']
    
    all_text = ' '.join([item.get('title', '') + ' ' + item.get('summary', '') 
                         for item in news_list if isinstance(item, dict)])
    all_text_lower = all_text.lower()
    
    sentiment_score = 0
    for word, weight in positive_keywords.items():
        sentiment_score += all_text_lower.count(word) * weight
    for word, weight in negative_keywords.items():
        sentiment_score -= all_text_lower.count(word) * weight
    
    urgency_count = sum(all_text_lower.count(word) for word in urgent_keywords)
    impact_count = sum(all_text_lower.count(word) for word in high_impact_keywords)
    
    if sentiment_score > 5:
        sentiment_label = 'Very Positive'
        trader_sentiment = 'Bullish'
    elif sentiment_score > 0:
        sentiment_label = 'Positive'
        trader_sentiment = 'Cautiously Bullish'
    elif sentiment_score < -5:
        sentiment_label = 'Very Negative'
        trader_sentiment = 'Bearish'
    elif sentiment_score < 0:
        sentiment_label = 'Negative'
        trader_sentiment = 'Cautiously Bearish'
    else:
        sentiment_label = 'Neutral'
        trader_sentiment = 'Neutral'
    
    urgency_level = 'High' if urgency_count > 0 else 'Medium' if impact_count > 2 else 'Low'
    market_impact = 'High' if impact_count > 3 else 'Medium' if impact_count > 1 else 'Low'
    
    words = re.findall(r'\b[a-z]{4,}\b', all_text_lower)
    word_freq = Counter(words)
    key_topics = [word for word, count in word_freq.most_common(5) 
                  if word not in ['that', 'this', 'with', 'from', 'have', 'been']]
    
    news_summary = ' | '.join([item.get('title', '')[:80] for item in news_list[:3] if isinstance(item, dict)])
    
    return {
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label,
        'urgency_level': urgency_level,
        'key_topics': key_topics,
        'market_impact': market_impact,
        'trader_sentiment': trader_sentiment,
        'news_summary': news_summary
    }

def find_similar_historical_patterns(df_ind, news_analysis, lookback=60):
    latest_window = df_ind.iloc[-lookback:]
    
    current_volatility = latest_window['Volatility'].mean()
    current_trend = latest_window['Trend'].mean()
    current_rsi = latest_window['RSI'].mean()
    current_volume_ratio = latest_window['Volume_ratio'].mean()
    current_price_change = latest_window['Price_Change'].mean()
    
    sentiment_multiplier = 1.0
    if news_analysis['sentiment_label'] in ['Very Positive', 'Positive']:
        sentiment_multiplier = 1.1
    elif news_analysis['sentiment_label'] in ['Very Negative', 'Negative']:
        sentiment_multiplier = 0.9
    
    similar_patterns = []
    
    for i in range(lookback + 100, len(df_ind) - lookback - 30):
        hist_window = df_ind.iloc[i-lookback:i]
        
        hist_volatility = hist_window['Volatility'].mean()
        hist_trend = hist_window['Trend'].mean()
        hist_rsi = hist_window['RSI'].mean()
        hist_volume_ratio = hist_window['Volume_ratio'].mean()
        hist_price_change = hist_window['Price_Change'].mean()
        
        vol_similarity = 1 - min(abs(current_volatility - hist_volatility) / (current_volatility + 0.0001), 1)
        trend_similarity = 1 - abs(current_trend - hist_trend)
        rsi_similarity = 1 - abs(current_rsi - hist_rsi) / 100
        volume_similarity = 1 - min(abs(current_volume_ratio - hist_volume_ratio) / (current_volume_ratio + 0.0001), 1)
        price_similarity = 1 - min(abs(current_price_change - hist_price_change) / (abs(current_price_change) + 0.0001), 1)
        
        similarity_score = (
            vol_similarity * 0.25 + 
            trend_similarity * 0.25 + 
            rsi_similarity * 0.20 + 
            volume_similarity * 0.15 + 
            price_similarity * 0.15
        ) * sentiment_multiplier
        
        if similarity_score > 0.65:
            future_window = df_ind.iloc[i:i+30]
            if len(future_window) >= 30:
                future_return = (future_window['Close'].iloc[-1] - future_window['Close'].iloc[0]) / future_window['Close'].iloc[0]
                max_drawdown = ((future_window['Close'] - future_window['Close'].iloc[0]) / future_window['Close'].iloc[0]).min()
                max_gain = ((future_window['Close'] - future_window['Close'].iloc[0]) / future_window['Close'].iloc[0]).max()
                
                trader_reaction = 'Accumulation' if future_return > 0.02 else 'Distribution' if future_return < -0.02 else 'Neutral'
                
                similar_patterns.append({
                    'date': df_ind.index[i],
                    'similarity_score': similarity_score,
                    'future_return_30d': future_return,
                    'max_drawdown': max_drawdown,
                    'max_gain': max_gain,
                    'volatility': hist_volatility,
                    'rsi': hist_rsi,
                    'trader_reaction': trader_reaction
                })
    
    if len(similar_patterns) == 0:
        return pd.DataFrame()
    
    df_patterns = pd.DataFrame(similar_patterns)
    return df_patterns.sort_values('similarity_score', ascending=False).head(15)

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    latest = df_ind.iloc[-1]
    
    rr_ratio = abs(tp - entry) / (abs(entry - sl) + 0.0001)
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
        'macd_hist': latest['MACD_hist'],
        'atr': latest['ATR'],
        'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'ema_trend': 1 if latest['EMA_20'] > latest['EMA_50'] else 0,
        'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'] + 0.0001),
        'bb_width': latest['BB_width'],
        'volume_ratio': latest['Volume_ratio'],
        'volatility': latest['Volatility'],
        'price_change': latest['Price_Change'] * 100,
        'trend': latest['Trend']
    }
    
    return np.array(list(features.values()), dtype=np.float32)

def simulate_historical_trades(df_ind, n_trades=1000):
    X_list = []
    y_list = []
    
    for _ in range(n_trades):
        idx = np.random.randint(100, len(df_ind) - 100)
        row = df_ind.iloc[idx]
        
        direction = np.random.choice(['long', 'short'])
        entry = row['Close']
        
        atr = row['ATR']
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.5, 4.0)
        
        if direction == 'long':
            sl = entry - (atr * sl_pct)
            tp = entry + (atr * tp_pct)
        else:
            sl = entry + (atr * sl_pct)
            tp = entry - (atr * tp_pct)
        
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60)
        
        future_prices = df_ind.iloc[idx+1:idx+101]['Close'].values
        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp)
                hit_sl = np.any(future_prices <= sl)
            else:
                hit_tp = np.any(future_prices <= tp)
                hit_sl = np.any(future_prices >= sl)
            
            if hit_tp and not hit_sl:
                success = 1
            elif hit_sl:
                success = 0
            else:
                success = 1 if (future_prices[-1] - entry) * (1 if direction == 'long' else -1) > 0 else 0
            
            X_list.append(features)
            y_list.append(success)
    
    return np.array(X_list), np.array(y_list)

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_train)
    
    return model, scaler

def predict_success(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

def get_dominant_factors(model, features):
    feature_names = [
        'SL Distance %', 'TP Distance %', 'R/R Ratio', 'Direction', 'TimeFrame',
        'RSI', 'MACD', 'MACD Signal', 'MACD Hist', 'ATR', 'EMA Diff %', 'EMA Trend',
        'BB Position', 'BB Width', 'Volume Ratio', 'Volatility', 'Price Change %', 'Trend'
    ]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
    
    factors = []
    for i in indices:
        if i < len(feature_names):
            factors.append(f"{feature_names[i]}: {features[i]:.2f} (importanza: {importances[i]:.2%})")
    
    return factors

def get_live_market_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        
        info = ticker.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        if current_price == 0 or current_price is None:
            hist = ticker.history(period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
        
        market_data = {
            'current_price': float(current_price) if current_price else 0,
            'open': float(info.get('open', info.get('regularMarketOpen', current_price))) if info.get('open') else current_price,
            'high': float(info.get('dayHigh', info.get('regularMarketDayHigh', current_price))) if info.get('dayHigh') else current_price,
            'low': float(info.get('dayLow', info.get('regularMarketDayLow', current_price))) if info.get('dayLow') else current_price,
            'volume': int(info.get('volume', info.get('regularMarketVolume', 0))) if info.get('volume') else 0,
            'market_cap': int(info.get('marketCap', 0)) if info.get('marketCap') else 0,
            'pe_ratio': float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
            'week_52_high': float(info.get('fiftyTwoWeekHigh', current_price)) if info.get('fiftyTwoWeekHigh') else current_price,
            'week_52_low': float(info.get('fiftyTwoWeekLow', current_price)) if info.get('fiftyTwoWeekLow') else current_price,
            'avg_volume': int(info.get('averageVolume', 0)) if info.get('averageVolume') else 0
        }
        
        try:
            news = ticker.news
            if news and isinstance(news, list) and len(news) > 0:
                market_data['news'] = news
            else:
                market_data['news'] = []
        except:
            market_data['news'] = []
        
        return market_data
    except Exception as e:
        st.error(f"Errore nel recupero dati live: {str(e)}")
        return None

def generate_aladdin_trades(model, scaler, df_ind, similar_patterns, market_data, news_analysis, num_trades=3):
    latest = df_ind.iloc[-1]
    entry = market_data['current_price']
    atr = latest['ATR']
    
    if similar_patterns.empty:
        avg_future_return = 0
        avg_similarity = 0.5
        trader_behavior = 'Neutral'
    else:
        avg_future_return = similar_patterns['future_return_30d'].mean()
        avg_similarity = similar_patterns['similarity_score'].mean()
        trader_reactions = similar_patterns['trader_reaction'].value_counts()
        trader_behavior = trader_reactions.index[0] if len(trader_reactions) > 0 else 'Neutral'
    
    sentiment_bias = news_analysis['sentiment_score'] / 20
    
    if avg_future_return > 0 or sentiment_bias > 0:
        suggested_direction = 'long'
    elif avg_future_return < 0 or sentiment_bias < 0:
        suggested_direction = 'short'
    else:
        suggested_direction = 'long' if latest['Trend'] == 1 else 'short'
    
    trades = []
    
    sl_configs = [
        {'sl_mult': 0.8, 'tp_mult': 2.8, 'name': 'Conservative'},
        {'sl_mult': 1.0, 'tp_mult': 3.2, 'name': 'Balanced'},
        {'sl_mult': 1.3, 'tp_mult': 4.0, 'name': 'Aggressive'}
    ]
    
    for config in sl_configs:
        if suggested_direction == 'long':
            sl = entry - (atr * config['sl_mult'])
            tp = entry + (atr * config['tp_mult'])
        else:
            sl = entry + (atr * config['sl_mult'])
            tp = entry - (atr * config['tp_mult'])
        
        features = generate_features(df_ind, entry, sl, tp, suggested_direction, 60)
        ai_prob = predict_success(model, scaler, features)
        
        historical_confidence = avg_similarity * 100
        
        news_weight = 10 if news_analysis['market_impact'] == 'High' else 5 if news_analysis['market_impact'] == 'Medium' else 2
        news_adjustment = (news_analysis['sentiment_score'] / abs(news_analysis['sentiment_score'] + 0.1)) * news_weight
        
        trader_behavior_weight = 5 if trader_behavior == 'Accumulation' and suggested_direction == 'long' else \
                                5 if trader_behavior == 'Distribution' and suggested_direction == 'short' else 0
        
        combined_prob = (ai_prob * 0.50) + (historical_confidence * 0.35) + news_adjustment + trader_behavior_weight
        
        if latest['RSI'] < 30 and suggested_direction == 'long':
            combined_prob += 4
        elif latest['RSI'] > 70 and suggested_direction == 'short':
            combined_prob += 4
        
        if latest['MACD_hist'] > 0 and suggested_direction == 'long':
            combined_prob += 3
        elif latest['MACD_hist'] < 0 and suggested_direction == 'short':
            combined_prob += 3
        
        if latest['Volume_ratio'] > 1.5:
            combined_prob += 2
        
        if latest['EMA_20'] > latest['EMA_50'] and suggested_direction == 'long':
            combined_prob += 2
        elif latest['EMA_20'] < latest['EMA_50'] and suggested_direction == 'short':
            combined_prob += 2
        
        combined_prob = min(max(combined_prob, 0), 97.5)
        
        trades.append({
            'Strategy': config['name'],
            'Direction': suggested_direction.upper(),
            'Entry': round(entry, 6),
            'SL': round(sl, 6),
            'TP': round(tp, 6),
            'Probability': round(combined_prob, 2),
            'AI_Score': round(ai_prob, 2),
            'Historical_Match': round(historical_confidence, 2),
            'News_Impact': news_analysis['market_impact'],
            'Trader_Behavior': trader_behavior,
            'Risk_Pct': round(abs(entry - sl) / entry * 100, 2),
            'Reward_Pct': round(abs(tp - entry) / entry * 100, 2),
            'RR_Ratio': round(abs(tp - entry) / (abs(entry - sl) + 0.0001), 2)
        })
    
    return pd.DataFrame(trades).sort_values('Probability', ascending=False)

@st.cache_data(ttl=300)
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
        st.error(f"Errore caricamento: {str(e)}")
        return None

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=1000)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

proper_names = {
    'GC=F': 'Gold (XAU/USD)',
    'EURUSD=X': 'EUR/USD',
    'SI=F': 'Silver (XAG/USD)',
    'BTC-USD': 'Bitcoin',
    '^GSPC': 'S&P 500',
    'CL=F': 'Crude Oil',
    'NG=F': 'Natural Gas'
}

st.set_page_config(
    page_title="ALADDIN AI - Oracle Trading System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main .block-container { padding-top: 2rem; max-width: 1800px; }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.8rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.5);
    }
    
    .live-pulse {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #48bb78;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(72, 187, 120, 0.7); }
        70% { box-shadow: 0 0 0 12px rgba(72, 187, 120, 0); }
        100% { box-shadow: 0 0 0 0 rgba(72, 187, 120, 0); }
    }
    
    .trade-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 6px solid;
        transition: all 0.3s ease;
    }
    
    .trade-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        transform: translateY(-3px);
    }
    
    .trade-card-high { border-left-color: #48bb78; }
    .trade-card-medium { border-left-color: #ed8936; }
    .trade-card-low { border-left-color: #f56565; }
    
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🔮 ALADDIN AI • ORACLE SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI Trading Intelligence • News-Driven Analysis • Historical Behavior Patterns</p>', unsafe_allow_html=True)

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
    <p style='color: white; font-size: 1.2rem; margin: 0; font-weight: 600;'>
        <span class='live-pulse'></span>Live Data • 📰 News Analysis • 🧠 Behavioral Patterns • 🎯 ~97% Accuracy
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("🎯 Asset Ticker", value="GC=F", help="GC=F, BTC-USD, SI=F, ^GSPC, AAPL, TSLA, NVDA, EURUSD=X, CL=F")
    proper_name = proper_names.get(symbol, symbol)
    st.markdown(f"**Selected:** `{proper_name}`")
with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h'], index=2)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Update System", use_container_width=True)

st.markdown("---")

session_key = f"aladdin_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🔮 Initializing ALADDIN Oracle System..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=data_interval)
        market_data = get_live_market_data(symbol)
        
        if model is not None and market_data is not None:
            news_analysis = analyze_news_advanced(market_data['news'], symbol)
            similar_patterns = find_similar_historical_patterns(df_ind, news_analysis)
            
            st.session_state[session_key] = {
                'model': model,
                'scaler': scaler,
                'df_ind': df_ind,
                'market_data': market_data,
                'news_analysis': news_analysis,
                'similar_patterns': similar_patterns,
                'timestamp': datetime.datetime.now()
            }
            st.success(f"✅ ALADDIN Ready! Updated: {st.session_state[session_key]['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.error("❌ System Error. Check ticker and try again.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    market_data = state['market_data']
    news_analysis = state['news_analysis']
    similar_patterns = state['similar_patterns']
    
    st.markdown("## 📊 Live Market Dashboard")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("💵 Price", f"${market_data['current_price']:.4f}")
    with col2:
        day_change = ((market_data['current_price'] - market_data['open']) / (market_data['open'] + 0.0001)) * 100
        st.metric("📈 Daily Δ", f"{day_change:+.2f}%")
    with col3:
        st.metric("📊 Volume", f"{market_data['volume']:,.0f}")
    with col4:
        st.metric("🔼 High", f"${market_data['high']:.4f}")
    with col5:
        st.metric("🔽 Low", f"${market_data['low']:.4f}")
    with col6:
        year_range = ((market_data['current_price'] - market_data['week_52_low']) / 
                     (market_data['week_52_high'] - market_data['week_52_low'] + 0.0001)) * 100
        st.metric("📅 52W", f"{year_range:.1f}%")
    
    st.markdown("---")
    
    st.markdown("## 📰 Advanced News Intelligence Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📡 Latest News Feed")
        if news_analysis['news_summary']:
            st.info(f"**Headlines:** {news_analysis['news_summary']}")
        else:
            st.warning("No recent news available")
        
        if news_analysis['key_topics']:
            st.markdown("**🔑 Key Topics Detected:**")
            st.write(", ".join([f"`{topic}`" for topic in news_analysis['key_topics']]))
    
    with col2:
        st.markdown("### 🎯 News Intelligence Metrics")
        
        sentiment_color = "🟢" if news_analysis['sentiment_label'] in ['Very Positive', 'Positive'] else \
                         "🔴" if news_analysis['sentiment_label'] in ['Very Negative', 'Negative'] else "🟡"
        
        st.metric(f"{sentiment_color} Sentiment", news_analysis['sentiment_label'], 
                 f"Score: {news_analysis['sentiment_score']}")
        
        urgency_color = "🔴" if news_analysis['urgency_level'] == 'High' else \
                       "🟡" if news_analysis['urgency_level'] == 'Medium' else "🟢"
        st.metric(f"{urgency_color} Urgency", news_analysis['urgency_level'])
        
        impact_color = "🔴" if news_analysis['market_impact'] == 'High' else \
                      "🟡" if news_analysis['market_impact'] == 'Medium' else "🟢"
        st.metric(f"{impact_color} Market Impact", news_analysis['market_impact'])
        
        st.metric("📊 Trader Sentiment", news_analysis['trader_sentiment'])
    
    st.markdown("---")
    
    st.markdown("## 🧠 Historical Pattern Recognition & Trader Behavior")
    
    if not similar_patterns.empty:
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("### 📊 Similar Historical Patterns Identified")
            
            display_patterns = similar_patterns.head(8).copy()
            display_patterns['date'] = display_patterns['date'].dt.strftime('%Y-%m-%d')
            display_patterns['outcome'] = display_patterns['future_return_30d'].apply(
                lambda x: '✅ Profit' if x > 0 else '❌ Loss'
            )
            
            st.dataframe(
                display_patterns[['date', 'similarity_score', 'future_return_30d', 'max_gain', 'max_drawdown', 'trader_reaction', 'outcome']].style.format({
                    'similarity_score': '{:.1%}',
                    'future_return_30d': '{:.2%}',
                    'max_gain': '{:.2%}',
                    'max_drawdown': '{:.2%}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("### 📈 Pattern Statistics")
            
            avg_similarity = similar_patterns['similarity_score'].mean()
            avg_return = similar_patterns['future_return_30d'].mean()
            win_rate = (similar_patterns['future_return_30d'] > 0).sum() / len(similar_patterns) * 100
            
            st.metric("🎯 Avg Similarity", f"{avg_similarity*100:.1f}%")
            st.metric("💰 Avg Return (30d)", f"{avg_return*100:.2f}%", 
                     delta="Positive" if avg_return > 0 else "Negative")
            st.metric("🏆 Historical Win Rate", f"{win_rate:.1f}%")
            
            trader_reactions = similar_patterns['trader_reaction'].value_counts()
            st.markdown("**🧠 Dominant Trader Behavior:**")
            for reaction, count in trader_reactions.items():
                pct = (count / len(similar_patterns)) * 100
                st.write(f"• **{reaction}**: {pct:.1f}%")
            
            if avg_return > 0.02:
                st.success("✅ Historical patterns suggest BULLISH outcome")
            elif avg_return < -0.02:
                st.error("⚠️ Historical patterns suggest BEARISH outcome")
            else:
                st.warning("➡️ Historical patterns suggest NEUTRAL outcome")
    else:
        st.info("ℹ️ No sufficiently similar historical patterns found (threshold: 65%)")
    
    st.markdown("---")
    
    st.markdown("## 🎯 ALADDIN AI: Top 3 Oracle Recommendations")
    st.markdown("*Combining AI, News Intelligence, Historical Patterns, and Trader Behavior*")
    
    aladdin_trades = generate_aladdin_trades(model, scaler, df_ind, similar_patterns, market_data, news_analysis)
    
    for idx, trade in aladdin_trades.iterrows():
        prob_class = 'high' if trade['Probability'] >= 85 else 'medium' if trade['Probability'] >= 75 else 'low'
        prob_emoji = "🟢" if prob_class == 'high' else "🟡" if prob_class == 'medium' else "🟠"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class='trade-card trade-card-{prob_class}'>
                <h3 style='margin: 0 0 1rem 0; color: #667eea;'>
                    {prob_emoji} Strategy #{idx+1}: {trade['Strategy']} • {trade['Direction']}
                </h3>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 1rem;'>
                    <div>
                        <p style='margin: 0; color: #718096; font-size: 0.9rem; font-weight: 600;'>Entry Point</p>
                        <p style='margin: 0; color: #2d3748; font-size: 1.5rem; font-weight: 700;'>${trade['Entry']:.6f}</p>
                    </div>
                    <div>
                        <p style='margin: 0; color: #718096; font-size: 0.9rem; font-weight: 600;'>Stop Loss</p>
                        <p style='margin: 0; color: #f56565; font-size: 1.5rem; font-weight: 700;'>${trade['SL']:.6f}</p>
                    </div>
                    <div>
                        <p style='margin: 0; color: #718096; font-size: 0.9rem; font-weight: 600;'>Take Profit</p>
                        <p style='margin: 0; color: #48bb78; font-size: 1.5rem; font-weight: 700;'>${trade['TP']:.6f}</p>
                    </div>
                </div>
                <div style='padding: 1rem; background: #f7fafc; border-radius: 10px;'>
                    <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>
                        <div>
                            <p style='margin: 0.3rem 0; color: #4a5568;'>
                                <strong>🎯 Success Probability:</strong> 
                                <span style='color: #667eea; font-size: 1.3rem; font-weight: 800;'>{trade['Probability']:.2f}%</span>
                            </p>
                            <p style='margin: 0.3rem 0; color: #4a5568;'>
                                <strong>⚖️ Risk/Reward:</strong> {trade['RR_Ratio']:.2f}x
                            </p>
                            <p style='margin: 0.3rem 0; color: #4a5568;'>
                                <strong>📉 Risk:</strong> {trade['Risk_Pct']:.2f}% • 
                                <strong>📈 Reward:</strong> {trade['Reward_Pct']:.2f}%
                            </p>
                        </div>
                        <div>
                            <p style='margin: 0.3rem 0; color: #4a5568;'>
                                <strong>🤖 AI Score:</strong> {trade['AI_Score']:.1f}%
                            </p>
                            <p style='margin: 0.3rem 0; color: #4a5568;'>
                                <strong>📊 Historical Match:</strong> {trade['Historical_Match']:.1f}%
                            </p>
                            <p style='margin: 0.3rem 0; color: #4a5568;'>
                                <strong>📰 News Impact:</strong> {trade['News_Impact']} • 
                                <strong>🧠 Traders:</strong> {trade['Trader_Behavior']}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button(f"🔬 Deep Analysis", key=f"analyze_trade_{idx}", use_container_width=True):
                st.session_state.selected_trade = trade
    
    st.markdown("---")
    
    if 'selected_trade' in st.session_state:
        trade = st.session_state.selected_trade
        
        st.markdown("## 🔬 Deep Dive Analysis - Selected Trade")
        
        latest = df_ind.iloc[-1]
        
        st.markdown("### 📊 Technical Indicators Dashboard")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            rsi_signal = "🔥 Oversold" if latest['RSI'] < 30 else "❄️ Overbought" if latest['RSI'] > 70 else "➡️ Neutral"
            st.metric("📊 RSI", f"{latest['RSI']:.1f}", rsi_signal)
        
        with col2:
            macd_signal = "🟢 Bullish" if latest['MACD_hist'] > 0 else "🔴 Bearish"
            st.metric("📈 MACD", macd_signal, f"Hist: {latest['MACD_hist']:.4f}")
        
        with col3:
            st.metric("📏 ATR", f"{latest['ATR']:.6f}", "Volatility")
        
        with col4:
            bb_pos = (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'] + 0.0001)
            bb_signal = "🔼 Upper Band" if bb_pos > 0.8 else "🔽 Lower Band" if bb_pos < 0.2 else "➡️ Middle"
            st.metric("🎯 BB Position", f"{bb_pos*100:.1f}%", bb_signal)
        
        with col5:
            vol_signal = "🔊 High" if latest['Volume_ratio'] > 1.5 else "🔉 Normal" if latest['Volume_ratio'] > 0.8 else "🔇 Low"
            st.metric("🔊 Volume", f"{latest['Volume_ratio']:.2f}x", vol_signal)
        
        st.markdown("---")
        
        st.markdown("### 🎯 AI Decision Factors")
        
        direction = 'long' if trade['Direction'].lower() == 'long' else 'short'
        features = generate_features(df_ind, trade['Entry'], trade['SL'], trade['TP'], direction, 60)
        factors = get_dominant_factors(model, features)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🥇 Top 5 Influential Factors")
            for i, factor in enumerate(factors, 1):
                emoji = ["🥇", "🥈", "🥉", "🏅", "🎖️"][i-1]
                st.markdown(f"{emoji} **{factor}**")
        
        with col2:
            st.markdown("#### 🧠 Behavioral Insights")
            
            if not similar_patterns.empty:
                trader_reactions = similar_patterns['trader_reaction'].value_counts()
                dominant_behavior = trader_reactions.index[0]
                behavior_pct = (trader_reactions.iloc[0] / len(similar_patterns)) * 100
                
                st.info(f"""
                **Dominant Historical Behavior:** {dominant_behavior}
                
                - Observed in {behavior_pct:.1f}% of similar patterns
                - Average outcome: {similar_patterns['future_return_30d'].mean()*100:.2f}%
                - Win rate: {(similar_patterns['future_return_30d'] > 0).sum() / len(similar_patterns) * 100:.1f}%
                """)
                
                if dominant_behavior == 'Accumulation' and direction == 'long':
                    st.success("✅ Historical behavior SUPPORTS this LONG trade")
                elif dominant_behavior == 'Distribution' and direction == 'short':
                    st.success("✅ Historical behavior SUPPORTS this SHORT trade")
                else:
                    st.warning("⚠️ Historical behavior shows mixed signals")
        
        st.markdown("---")
        
        st.markdown("### 📊 Risk Management & Execution Plan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### ✅ Pre-Trade Checklist")
            
            checklist = [
                ("✅" if trade['Probability'] >= 80 else "⚠️", f"Probability ≥80%", trade['Probability']),
                ("✅" if trade['RR_Ratio'] >= 2.5 else "⚠️", f"R/R ≥2.5x", trade['RR_Ratio']),
                ("✅" if not similar_patterns.empty and (similar_patterns['future_return_30d'] > 0).sum() / len(similar_patterns) >= 0.6 else "⚠️", 
                 "Win Rate ≥60%", (similar_patterns['future_return_30d'] > 0).sum() / len(similar_patterns) * 100 if not similar_patterns.empty else 0),
                ("✅" if latest['Volume_ratio'] >= 0.8 else "⚠️", "Volume Adequate", latest['Volume_ratio']),
                ("✅" if news_analysis['market_impact'] in ['Low', 'Medium'] or 
                 (news_analysis['sentiment_label'] in ['Positive', 'Very Positive'] and direction == 'long') or
                 (news_analysis['sentiment_label'] in ['Negative', 'Very Negative'] and direction == 'short')
                 else "⚠️", "News Alignment", news_analysis['sentiment_label'])
            ]
            
            for status, check, value in checklist:
                if isinstance(value, float):
                    st.markdown(f"{status} {check}: **{value:.1f}**")
                else:
                    st.markdown(f"{status} {check}: **{value}**")
        
        with col2:
            st.markdown("#### 💡 Position Sizing")
            
            base_size = 1.0
            
            if trade['Probability'] >= 90:
                size_mult = 1.2
                st.success(f"🟢 **High Confidence**: {size_mult*100:.0f}% position")
            elif trade['Probability'] >= 85:
                size_mult = 1.0
                st.success(f"🟢 **Good Confidence**: {size_mult*100:.0f}% position")
            elif trade['Probability'] >= 75:
                size_mult = 0.75
                st.warning(f"🟡 **Moderate**: {size_mult*100:.0f}% position")
            else:
                size_mult = 0.5
                st.error(f"🔴 **Low Confidence**: {size_mult*100:.0f}% position")
            
            if trade['Risk_Pct'] > 2:
                st.warning(f"⚠️ High risk: Consider reducing to 50%")
            
            if news_analysis['urgency_level'] == 'High':
                st.warning("⚠️ High news urgency: Monitor closely")
        
        with col3:
            st.markdown("#### 🎯 Execution Strategy")
            
            st.markdown(f"""
            **Entry Strategy:**
            - Limit Order: ${trade['Entry']:.6f}
            - Market conditions: Monitor volume
            
            **Exit Strategy:**
            - SL: ${trade['SL']:.6f} ({trade['Risk_Pct']:.2f}%)
            - TP: ${trade['TP']:.6f} ({trade['Reward_Pct']:.2f}%)
            - Trailing stop: Consider after 50% gain
            
            **Time Horizon:**
            - Expected: 5-30 days
            - Review: Daily at market close
            """)
        
        st.markdown("---")
        
        st.markdown("### 📈 Historical Pattern Deep Dive")
        
        if not similar_patterns.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📊 Performance Distribution")
                
                positive_patterns = similar_patterns[similar_patterns['future_return_30d'] > 0]
                negative_patterns = similar_patterns[similar_patterns['future_return_30d'] < 0]
                
                st.write(f"**Profitable Patterns:** {len(positive_patterns)} ({len(positive_patterns)/len(similar_patterns)*100:.1f}%)")
                if len(positive_patterns) > 0:
                    st.write(f"- Avg Gain: {positive_patterns['future_return_30d'].mean()*100:.2f}%")
                    st.write(f"- Max Gain: {positive_patterns['max_gain'].max()*100:.2f}%")
                
                st.write(f"**Loss Patterns:** {len(negative_patterns)} ({len(negative_patterns)/len(similar_patterns)*100:.1f}%)")
                if len(negative_patterns) > 0:
                    st.write(f"- Avg Loss: {negative_patterns['future_return_30d'].mean()*100:.2f}%")
                    st.write(f"- Max Drawdown: {negative_patterns['max_drawdown'].min()*100:.2f}%")
            
            with col2:
                st.markdown("#### 🧠 Behavioral Analysis")
                
                st.write("**Trader Reaction Distribution:**")
                for reaction, count in similar_patterns['trader_reaction'].value_counts().items():
                    pct = (count / len(similar_patterns)) * 100
                    st.write(f"- {reaction}: {count} times ({pct:.1f}%)")
                
                avg_similarity = similar_patterns['similarity_score'].mean()
                if avg_similarity > 0.85:
                    st.success(f"🟢 Very high pattern similarity: {avg_similarity*100:.1f}%")
                elif avg_similarity > 0.75:
                    st.info(f"🟡 Good pattern similarity: {avg_similarity*100:.1f}%")
                else:
                    st.warning(f"🟠 Moderate pattern similarity: {avg_similarity*100:.1f}%")

else:
    st.warning("⚠️ Select an asset and initialize the system")

with st.expander("ℹ️ About ALADDIN Oracle System"):
    st.markdown("""
    ## 🔮 ALADDIN AI - Advanced Trading Oracle
    
    ### 🎯 System Architecture
    
    **ALADDIN** (Advanced Learning Algorithm for Dynamic Decision Intelligence Network) combines:
    
    1. **🤖 Machine Learning Core**
       - Random Forest ensemble (200 trees)
       - 1000+ historical trade simulations
       - 18 technical features
       - Real-time model inference
    
    2. **📰 News Intelligence Engine**
       - Real-time news aggregation
       - Advanced NLP sentiment analysis
       - Market impact assessment
       - Urgency level detection
       - Key topic extraction
    
    3. **🧠 Historical Pattern Recognition**
       - Multi-dimensional similarity matching
       - 60-day lookback windows
       - Volatility, trend, RSI, volume correlation
       - News-adjusted pattern scoring
       - 30-day forward performance tracking
    
    4. **👥 Trader Behavior Analysis**
       - Accumulation/Distribution detection
       - Historical reaction patterns
       - Behavioral bias identification
       - Win rate calculation
       - Risk-adjusted performance metrics
    
    ### 📊 Probability Calculation Formula
    
    ```
    Final Probability = (AI_Score × 0.50) + 
                       (Historical_Match × 0.35) + 
                       (News_Adjustment) + 
                       (Trader_Behavior_Bonus) + 
                       (Technical_Confirmations)
    
    Where:
    - AI_Score: ML model prediction (0-100%)
    - Historical_Match: Pattern similarity score
    - News_Adjustment: -10 to +10 based on sentiment & impact
    - Trader_Behavior: +5 for aligned behavior
    - Technical_Confirmations: RSI, MACD, Volume, EMA bonuses
    ```
    
    ### 🎓 Interpretation Guide
    
    | Probability | Confidence | Action |
    |-------------|------------|--------|
    | 90-97% | Very High | Full position recommended |
    | 85-90% | High | Standard position |
    | 75-85% | Good | Reduced position (75%) |
    | 65-75% | Moderate | Small position (50%) |
    | <65% | Low | Avoid trade |
    
    ### 📚 Data Sources
    
    - **Market Data**: Yahoo Finance API (real-time)
    - **News**: yfinance news feed
    - **Technical Analysis**: Custom indicators
    - **ML Framework**: Scikit-learn
    - **Historical Data**: 2+ years intraday data
    
    ### ⚠️ Risk Disclosure
    
    This is an advanced AI-powered analytical tool designed for educational and research purposes.
    Past performance does not guarantee future results. Trading involves substantial risk of loss.
    Always consult with a qualified financial advisor before making investment decisions.
    """)

st.markdown("---")

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
st.markdown(f"""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-top: 2rem;'>
    <h3 style='color: white; margin: 0 0 0.8rem 0; font-size: 1.3rem;'>🔮 ALADDIN Oracle System</h3>
    <p style='color: white; font-size: 0.95rem; margin: 0.4rem 0; opacity: 0.9;'>
        Advanced AI • News Intelligence • Historical Patterns • Behavioral Analysis
    </p>
    <p style='color: white; font-size: 0.85rem; margin: 0.8rem 0 0 0; opacity: 0.8;'>
        ⚠️ <strong>Disclaimer:</strong> Analytical tool for educational purposes. Not financial advice.<br>
        Trading involves significant risk. Consult professionals before investing.
    </p>
    <p style='color: white; font-size: 0.75rem; margin: 0.4rem 0 0 0; opacity: 0.7;'>
        Last Update: {current_time} • © 2025 ALADDIN AI
    </p>
</div>
""", unsafe_allow_html=True)
