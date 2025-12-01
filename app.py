import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE BRIDGE MT4 ====================
# Sostituisci questo percorso con il percorso "MQL4/Files" della tua installazione MT4
# Per trovarlo in MT4: File -> Apri Scheda Data -> MQL4 -> Files
DEFAULT_MT4_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\COMMON\Files" 

def send_signal_to_mt4(symbol, direction, entry, sl, tp, volume=0.1):
    """Scrive un file CSV che l'EA di MT4 leggerà per eseguire l'ordine."""
    try:
        # Pulisce il simbolo per MT4 (es. rimuove =F o =X se necessario, dipende dal broker)
        mt4_symbol = symbol.replace("=F", "").replace("=X", "").replace("-USD", "")
        if "GBP" in mt4_symbol or "EUR" in mt4_symbol:
            pass # Va bene così
        elif mt4_symbol == "GC": mt4_symbol = "XAUUSD" # Adattamento comune
        elif mt4_symbol == "SI": mt4_symbol = "XAGUSD"
        elif mt4_symbol == "BTC": mt4_symbol = "BTCUSD"
        
        # Formato CSV: SYMBOL,OP_TYPE,VOLUME,SL,TP
        # OP_TYPE: 0 = BUY, 1 = SELL
        op_type = 0 if direction.lower() in ['long', 'buy'] else 1
        
        line = f"{mt4_symbol},{op_type},{volume},{sl},{tp}"
        
        # Percorso del file condiviso
        file_path = os.path.join(st.session_state.get('mt4_path', DEFAULT_MT4_PATH), "py_signal.csv")
        
        with open(file_path, "w") as f:
            f.write(line)
            
        return True, f"Segnale inviato a: {file_path}"
    except Exception as e:
        return False, str(e)

# ==================== FUNZIONI CORE ORIGINALI & NUOVE ====================
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
    st.info("🔄 Recupero dati di mercato in tempo reale...")
    try:
        # DXY (Dollar Index)
        dxy_tickers = ["DX-Y.NYB", "DX=F", "USDOLLAR"]
        for ticker_symbol in dxy_tickers:
            try:
                dxy = yf.Ticker(ticker_symbol)
                dxy_hist = dxy.history(period="1d", interval="1m")
                if not dxy_hist.empty and len(dxy_hist) > 0:
                    factors['dxy_current'] = float(dxy_hist['Close'].iloc[-1])
                    if len(dxy_hist) > 1:
                        factors['dxy_change'] = ((dxy_hist['Close'].iloc[-1] - dxy_hist['Close'].iloc[0]) / dxy_hist['Close'].iloc[0]) * 100
                    else:
                        factors['dxy_change'] = 0.0
                    st.success(f"✅ Dollar Index recuperato: ${factors['dxy_current']:.2f}")
                    break
            except:
                continue
        if 'dxy_current' not in factors:
            factors['dxy_current'] = 106.2
            factors['dxy_change'] = -0.3
            st.warning("⚠️ Dollar Index: uso valore stimato")
        
        # Tassi interesse USA (10Y Treasury)
        try:
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="1d", interval="1m")
            if not tnx_hist.empty and len(tnx_hist) > 0:
                factors['yield_10y'] = float(tnx_hist['Close'].iloc[-1])
                st.success(f"✅ Treasury 10Y: {factors['yield_10y']:.2f}%")
            else:
                factors['yield_10y'] = 4.42
                st.warning("⚠️ Treasury 10Y: uso valore stimato")
        except:
            factors['yield_10y'] = 4.42
            st.warning("⚠️ Treasury 10Y: uso valore stimato")
        
        # VIX (Volatilità/Fear)
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="1d", interval="1m")
            if not vix_hist.empty and len(vix_hist) > 0:
                factors['vix'] = float(vix_hist['Close'].iloc[-1])
                st.success(f"✅ VIX: {factors['vix']:.2f}")
            else:
                factors['vix'] = 16.8
                st.warning("⚠️ VIX: uso valore stimato")
        except:
            factors['vix'] = 16.8
            st.warning("⚠️ VIX: uso valore stimato")
        
        # S&P 500
        try:
            spx = yf.Ticker("^GSPC")
            spx_hist = spx.history(period="1d", interval="1m")
            if not spx_hist.empty and len(spx_hist) > 1:
                factors['spx_momentum'] = ((spx_hist['Close'].iloc[-1] - spx_hist['Close'].iloc[0]) / spx_hist['Close'].iloc[0]) * 100
                st.success(f"✅ S&P 500 Momentum: {factors['spx_momentum']:+.2f}%")
            else:
                factors['spx_momentum'] = 1.8
                st.warning("⚠️ S&P 500: uso valore stimato")
        except:
            factors['spx_momentum'] = 1.8
            st.warning("⚠️ S&P 500: uso valore stimato")
        
        # Silver e Gold
        try:
            silver = yf.Ticker("SI=F")
            silver_hist = silver.history(period="1d", interval="1m")
            gold = yf.Ticker("GC=F")
            gold_hist = gold.history(period="1d", interval="1m")
            if not silver_hist.empty and not gold_hist.empty and len(silver_hist) > 0 and len(gold_hist) > 0:
                silver_price = float(silver_hist['Close'].iloc[-1])
                gold_price = float(gold_hist['Close'].iloc[-1])
                factors['gold_silver_ratio'] = gold_price / silver_price
                st.success(f"✅ Gold/Silver Ratio: {factors['gold_silver_ratio']:.2f} (Gold: ${gold_price:.2f}, Silver: ${silver_price:.2f})")
            else:
                factors['gold_silver_ratio'] = 88.5
                st.warning("⚠️ Gold/Silver Ratio: uso valore stimato")
        except:
            factors['gold_silver_ratio'] = 88.5
            st.warning("⚠️ Gold/Silver Ratio: uso valore stimato")
        
        # Inflazione
        try:
            fvx = yf.Ticker("^FVX")
            fvx_hist = fvx.history(period="1d", interval="1m")
            if not fvx_hist.empty and len(fvx_hist) > 0:
                yield_5y = float(fvx_hist['Close'].iloc[-1])
                factors['inflation_expectations'] = max(0.5, factors['yield_10y'] - yield_5y + 2.0)
                st.success(f"✅ Inflazione Attesa: {factors['inflation_expectations']:.2f}%")
            else:
                factors['inflation_expectations'] = 2.5
                st.warning("⚠️ Inflazione: uso valore stimato")
        except:
            factors['inflation_expectations'] = 2.5
            st.warning("⚠️ Inflazione: uso valore stimato")
            
    except Exception as e:
        st.error(f"❌ Errore nel recupero dati: {str(e)}")
        factors = {
            'dxy_current': 106.2, 'dxy_change': -0.3, 'yield_10y': 4.42, 'vix': 16.8,
            'spx_momentum': 1.8, 'gold_silver_ratio': 88.5, 'inflation_expectations': 2.5
        }
        st.warning("⚠️ Uso valori stimati per tutti i parametri")
    
    factors['geopolitical_risk'] = 7.5
    factors['central_bank_demand'] = 1050
    factors['retail_sentiment'] = 6.8
    st.success("✅ Recupero dati completato!")
    return factors

def analyze_gold_historical_comparison(current_price, factors):
    """Analizza confronti storici e predice prezzo futuro dell'oro."""
    historical_periods = {
        '1971-1980': {'description': 'Bull Market Post-Bretton Woods', 'start_price': 35, 'end_price': 850, 'gain_pct': 2329, 'duration_years': 9, 'avg_inflation': 8.5, 'geopolitical': 8, 'dollar_weak': True, 'key_events': 'Fine gold standard, inflazione alta'},
        '2001-2011': {'description': 'Bull Market Post-Dot-Com e Crisi 2008', 'start_price': 255, 'end_price': 1920, 'gain_pct': 653, 'duration_years': 10, 'avg_inflation': 2.8, 'geopolitical': 7, 'dollar_weak': True, 'key_events': '9/11, QE, crisi finanziaria'},
        '2015-2020': {'description': 'Consolidamento e COVID Rally', 'start_price': 1050, 'end_price': 2070, 'gain_pct': 97, 'duration_years': 5, 'avg_inflation': 1.8, 'geopolitical': 6, 'dollar_weak': False, 'key_events': 'Tassi bassi, QE, pandemia'},
        '2022-2025': {'description': 'Era Inflazione Post-COVID', 'start_price': 1800, 'end_price': current_price, 'gain_pct': ((current_price - 1800) / 1800) * 100, 'duration_years': 3, 'avg_inflation': 4.5, 'geopolitical': 7.5, 'dollar_weak': False, 'key_events': 'Inflazione persistente, guerre'}
    }
    
    current_context = {
        'inflation': factors['inflation_expectations'],
        'dollar_strength': 'Forte' if factors['dxy_current'] > 103 else 'Debole',
        'real_rates': factors['yield_10y'] - factors['inflation_expectations'],
        'risk_sentiment': 'Risk-Off' if factors['vix'] > 20 else 'Neutrale' if factors['vix'] > 15 else 'Risk-On',
        'geopolitical': factors['geopolitical_risk'],
        'central_bank': 'Compratori Netti' if factors['central_bank_demand'] > 500 else 'Venditori',
        'technical_trend': 'Bullish' if current_price > 2600 else 'Neutrale'
    }
    
    similarity_scores = {}
    for period, data in historical_periods.items():
        if period == '2022-2025': continue
        score = 0
        inflation_diff = abs(data['avg_inflation'] - factors['inflation_expectations'])
        score += max(0, 10 - inflation_diff * 2)
        geo_diff = abs(data['geopolitical'] - factors['geopolitical_risk'])
        score += max(0, 10 - geo_diff * 2)
        current_dollar_weak = factors['dxy_current'] < 100
        if data['dollar_weak'] == current_dollar_weak: score += 15
        if factors['central_bank_demand'] > 800: score += 10
        similarity_scores[period] = score
    
    most_similar = max(similarity_scores, key=similarity_scores.get)
    similarity_pct = (similarity_scores[most_similar] / 45) * 100
    
    similar_period = historical_periods[most_similar]
    annual_return = (similar_period['gain_pct'] / 100) / similar_period['duration_years']
    projection_1y = current_price * (1 + annual_return)
    
    base_price = current_price
    if factors['dxy_change'] < 0: base_price *= 1.015
    elif factors['dxy_change'] > 1: base_price *= 0.985
    if current_context['real_rates'] < 1: base_price *= 1.025
    elif current_context['real_rates'] > 2: base_price *= 0.98
    if factors['vix'] > 25: base_price *= 1.03
    elif factors['vix'] < 15: base_price *= 0.99
    base_price *= (1 + (factors['geopolitical_risk'] / 100))
    base_price *= (1 + ((factors['central_bank_demand'] - 500) / 10000))
    base_price *= (1 + (factors['inflation_expectations'] / 100))
    projection_fundamental = base_price
    
    gold_data = yf.Ticker("GC=F").history(period="1y")
    if not gold_data.empty:
        momentum = ((gold_data['Close'].iloc[-1] - gold_data['Close'].iloc[-20]) / gold_data['Close'].iloc[-20]) * 100
        projection_technical = current_price * (1 + (momentum / 100) * 1.5)
    else:
        projection_technical = current_price * 1.05
        
    current_ratio = factors['gold_silver_ratio']
    ratio_adjustment = 1.02 if current_ratio > 70 else 1.01
    projection_ratio = current_price * ratio_adjustment
    
    weights = [0.3, 0.35, 0.25, 0.1]
    projections = [projection_1y, projection_fundamental, projection_technical, projection_ratio]
    target_price_1y = sum(w * p for w, p in zip(weights, projections))
    std_projections = np.std(projections)
    
    target_3m = current_price + (target_price_1y - current_price) * 0.25
    target_6m = current_price + (target_price_1y - current_price) * 0.5
    confidence = min(100, similarity_pct * 0.6 + (40 if factors['central_bank_demand'] > 800 else 20) + (20 if factors['geopolitical_risk'] > 6 else 10))
    
    return {
        'current_price': current_price, 'target_3m': target_3m, 'target_6m': target_6m, 'target_1y': target_price_1y,
        'range_low': target_price_1y - std_projections, 'range_high': target_price_1y + std_projections,
        'most_similar_period': most_similar, 'similarity_pct': similarity_pct, 'period_data': historical_periods[most_similar],
        'current_context': current_context, 'confidence': confidence,
        'key_drivers': {
            'Dollar Index': f"${factors['dxy_current']:.2f} ({factors['dxy_change']:+.2f}%)",
            'Tassi 10Y': f"{factors['yield_10y']:.2f}%", 'Tassi Reali': f"{current_context['real_rates']:.2f}%",
            'VIX': f"{factors['vix']:.1f}", 'Inflazione Attesa': f"{factors['inflation_expectations']:.2f}%",
            'Rischio Geopolitico': f"{factors['geopolitical_risk']}/10", 'Domanda BC': f"{factors['central_bank_demand']} ton/anno",
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
        'sl_distance_pct': sl_distance, 'tp_distance_pct': tp_distance, 'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0, 'main_tf': main_tf,
        'rsi': latest['RSI'], 'macd': latest['MACD'], 'macd_signal': latest['MACD_signal'],
        'atr': latest['ATR'], 'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
        'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        'price_change': latest['Price_Change'] * 100, 'trend': latest['Trend']
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
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_train)
    return model, scaler

def predict_success(model, scaler, features):
    """Predice probabilità di successo."""
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

def get_dominant_factors(model, features):
    """Identifica fattori dominanti."""
    feature_names = ['SL Distance %', 'TP Distance %', 'R/R Ratio', 'Direction', 'TimeFrame', 'RSI', 'MACD', 'MACD Signal', 'ATR', 'EMA Diff %', 'BB Position', 'Volume Ratio', 'Price Change %', 'Trend']
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
    factors = []
    for i in indices:
        if i < len(feature_names):
            factors.append(f"{feature_names[i]}: {features[i]:.2f} (importanza: {importances[i]:.2%})")
    return factors

def get_sentiment(text):
    positive_words = ['rally', 'up', 'bullish', 'gain', 'positive', 'strong', 'rise', 'surge', 'boom']
    negative_words = ['down', 'bearish', 'loss', 'negative', 'weak', 'slip', 'fall', 'drop', 'crash']
    score = sum(word in text.lower() for word in positive_words) - sum(word in text.lower() for word in negative_words)
    if score > 0: return 'Positive', score
    elif score < 0: return 'Negative', score
    else: return 'Neutral', 0

def predict_price(df_ind, steps=5):
    try:
        last_price = df_ind['Close'].iloc[-1]
        ema = df_ind['Close'].ewm(span=steps).mean().iloc[-1]
        forecast_values = [last_price + (ema - last_price) * (i / steps) for i in range(1, steps + 1)]
        forecast = np.array(forecast_values)
        return forecast.mean(), forecast
    except: return None, None

def get_investor_psychology(symbol, news_summary, sentiment_label, df_ind):
    latest = df_ind.iloc[-1]
    trend = 'bullish' if latest['Trend'] == 1 else 'bearish'
    current_analysis = f"""
    **🌍 Contesto Globale (Novembre 2025)**
    Nel contesto attuale, i mercati globali sono influenzati da inflazione persistente, tensioni geopolitiche e boom IA.
    La psicologia è un mix di ottimismo tecnologico e ansia macroeconomica.
    Per {symbol}, con trend {trend} e sentiment {sentiment_label}, gli investitori mostrano overreazioni emotive.
    """
    biases_analysis = """
    ### 🧠 Analisi Bias Comportamentali
    | Bias Cognitivo | Impatto |
    |---|---|
    | **Avversione alle Perdite** | Deflussi da fondi in panico |
    | **Eccessiva Fiducia** | Overtrading in mercati volatili |
    | **Effetto Gregge** | Flash crash virali |
    """
    return current_analysis + biases_analysis

def get_web_signals(symbol, df_ind):
    try:
        st.info(f"🔄 Recupero dati di mercato per {symbol}...")
        ticker = yf.Ticker(symbol)
        hist = None
        for interval in ['1m', '2m', '5m', '15m', '30m', '1h', '1d']:
            for period in ['1d', '5d', '1mo']:
                try:
                    hist = ticker.history(period=period, interval=interval)
                    if not hist.empty and len(hist) > 0: break
                except: continue
            if hist is not None and not hist.empty: break
        
        if hist is None or hist.empty:
            st.error(f"❌ Impossibile recuperare dati per {symbol}")
            return []
        
        current_price = float(hist['Close'].iloc[-1])
        st.success(f"✅ Prezzo attuale {symbol}: ${current_price:.2f}")
        
        try:
            news = ticker.news
            if news and isinstance(news, list) and len(news) > 0:
                news_summary = ' | '.join([item.get('title', '') for item in news[:5] if isinstance(item, dict)])
                st.success(f"✅ {len(news)} news recuperate")
            else:
                news_summary = 'Nessuna news recente disponibile.'
                st.warning("⚠️ Nessuna news disponibile")
        except:
            news_summary = 'Nessuna news recente disponibile.'
            st.warning("⚠️ Errore nel recupero news")
        
        sentiment_label, sentiment_score = get_sentiment(news_summary)
        
        try:
            hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
            if len(hist_monthly) >= 12:
                if isinstance(hist_monthly.columns, pd.MultiIndex):
                    hist_monthly.columns = hist_monthly.columns.droplevel(1)
                hist_monthly['Return'] = hist_monthly['Close'].pct_change()
                hist_monthly['Month'] = hist_monthly.index.month
                monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
                current_month = datetime.datetime.now().month
                avg_current = monthly_returns.get(current_month, 0) * 100
                seasonality_note = f'Il mese corrente ha un ritorno medio storico di {avg_current:.2f}%.'
                st.success("✅ Analisi stagionalità completata")
            else:
                seasonality_note = 'Dati storici insufficienti.'
                st.warning("⚠️ Dati stagionalità limitati")
        except Exception as e:
            seasonality_note = f'Errore stagionalità: {e}'
            st.warning("⚠️ Errore nel calcolo stagionalità")
        
        _, forecast_series = predict_price(
