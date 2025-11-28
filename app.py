import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
from pathlib import Path  # <--- IMPORT PER PATH MT4

warnings.filterwarnings('ignore')

# ==================== MAPPATURA YAHOO <-> MT4 + PREZZO LIVE ====================

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
    Riceve il simbolo inserito dall'utente e restituisce:
      - yahoo_symbol: per yfinance
      - mt4_symbol:   per il file MT4
    Gestisce sia GC=F, SI=F, BTC-USD, ^GSPC sia XAUUSD, XAGUSD, BTCUSD, US500.
    """
    user_symbol_clean = user_symbol.strip()

    # Match su chiavi Yahoo (case-insensitive)
    for k in YAHOO_TO_MT4.keys():
        if user_symbol_clean.upper() == k.upper():
            yahoo_symbol = k
            mt4_symbol = YAHOO_TO_MT4[k]
            return yahoo_symbol, mt4_symbol

    # Match su simboli MT4 (case-insensitive)
    for k_mt4 in MT4_TO_YAHOO.keys():
        if user_symbol_clean.upper() == k_mt4.upper():
            mt4_symbol = k_mt4
            yahoo_symbol = MT4_TO_YAHOO[k_mt4]
            return yahoo_symbol, mt4_symbol

    # Default: non mappato -> uso lo stesso
    return user_symbol_clean, user_symbol_clean

# ⚠️ METTI QUI IL PERCORSO REALE DEL TUO FILE mt4_prices.csv (cartella MQL4/Files)
MT4_PRICES_FILE = Path(
    r"C:\PERCORSO\REALE\FINO_A\MQL4\Files\mt4_prices.csv"
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
    
    st.info("🔄 Recupero dati di mercato in tempo reale...")
    
    try:
        # DXY (Dollar Index) - provo diversi ticker
        dxy_tickers = ["DX-Y.NYB", "DX=F", "USDOLLAR"]
        dxy_current = None
        
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
        
        # S&P 500 (Risk-on/Risk-off)
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
        
        # Silver e Gold (correlazione metalli preziosi)
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
        
        # Inflazione stimata (usando differenza tra 10Y e 5Y come proxy)
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
            'dxy_current': 106.2,
            'dxy_change': -0.3,
            'yield_10y': 4.42,
            'vix': 16.8,
            'spx_momentum': 1.8,
            'gold_silver_ratio': 88.5,
            'inflation_expectations': 2.5
        }
        st.warning("⚠️ Uso valori stimati per tutti i parametri")
    
    # Fattori geopolitici (score 0-10, basato su analisi qualitativa 2025)
    factors['geopolitical_risk'] = 7.5  # Medio Oriente, Ucraina, tensioni USA-Cina
    
    # Domanda banche centrali (tonnellate annue stimate)
    factors['central_bank_demand'] = 1050  # 2024-2025 trend
    
    # Sentiment retail (0-10)
    factors['retail_sentiment'] = 6.8
    
    st.success("✅ Recupero dati completato!")
    
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
        st.info(f"🔄 Recupero dati di mercato per {symbol}...")
        
        ticker = yf.Ticker(symbol)
        
        # Recupero prezzo più recente con retry multipli
        hist = None
        for interval in ['1m', '2m', '5m', '15m', '30m', '1h', '1d']:
            for period in ['1d', '5d', '1mo']:
                try:
                    hist = ticker.history(period=period, interval=interval)
                    if not hist.empty and len(hist) > 0:
                        break
                except:
                    continue
            if hist is not None and not hist.empty:
                break
        
        if hist is None or hist.empty:
            st.error(f"❌ Impossibile recuperare dati per {symbol}")
            return []
        
        current_price = float(hist['Close'].iloc[-1])
        st.success(f"✅ Prezzo attuale {symbol}: ${current_price:.2f}")
        
        # News recenti
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
        
        # Calcolo stagionalità
        try:
            hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
            if len(hist_monthly) >= 12:
                hist_monthly['Return'] = hist_monthly['Close'].pct_change()
                hist_monthly['Month'] = hist_monthly.index.month
                monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
                current_month = datetime.datetime.now().month
                avg_current = monthly_returns.get(current_month, 0) * 100
                seasonality_note = f'Il mese corrente ha un ritorno medio storico di {avg_current:.2f}%.'
                st.success("✅ Analisi stagionalità completata")
            else:
                seasonality_note = 'Dati storici insufficienti per calcolare la stagionalità.'
                st.warning("⚠️ Dati stagionalità limitati")
        except:
            seasonality_note = 'Dati storici insufficienti per calcolare la stagionalità.'
            st.warning("⚠️ Errore nel calcolo stagionalità")
        
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
        
        st.success(f"✅ {len(suggestions)} suggerimenti generati")
        return suggestions
        
    except Exception as e:
        st.error(f"❌ Errore nel recupero dati web: {str(e)}")
        return []

def fetch_real_time_prices():
    """Scova i prezzi degli strumenti incorporati in tempo reale da Yahoo (non MT4)."""
    prices = {}
    for ticker, name in proper_names.items():
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="1d", interval="1m")
            if not hist.empty:
                prices[name] = float(hist['Close'].iloc[-1])
            else:
                # Fallback to larger interval
                hist = data.history(period="1d", interval="5m")
                if not hist.empty:
                    prices[name] = float(hist['Close'].iloc[-1])
        except Exception as e:
            prices[name] = f"Errore: {str(e)}"
    return prices

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
    'XAUUSD': 'XAU/USD (Gold, MT4)',
    'EURUSD=X': 'EUR/USD',
    'SI=F': 'XAG/USD (Silver)',
    'XAGUSD': 'XAG/USD (Silver, MT4)',
    'BTC-USD': 'BTC/USD',
    'BTCUSD': 'BTC/USD (MT4)',
    '^GSPC': 'S&P 500',
    'US500': 'S&P 500 (US500 MT4)',
}

st.set_page_config(
    page_title="Trading Predictor AI - Gold Focus",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    
    h1 {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #FFA500;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #FF8C00;
        font-weight: 600;
        font-size: 1.4rem !important;
        margin-top: 1rem !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(255, 215, 0, 0.2);
        transition: transform 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 215, 0, 0.3);
    }
    
    .stMetric label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #8B4513 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #B8860B !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 165, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 165, 0, 0.4);
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #FFE4B5;
        padding: 0.6rem;
        font-size: 1rem;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FFD700;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.1);
    }
    
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #FFE4B5;
        padding: 0.6rem;
    }
    
    .stSuccess {
        background-color: #c6f6d5;
        border-left: 4px solid #48bb78;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #feebc8;
        border-left: 4px solid #ed8936;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #fed7d7;
        border-left: 4px solid #f56565;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #bee3f8;
        border-left: 4px solid #4299e1;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 100%);
        border-radius: 8px;
        padding: 0.8rem;
        font-weight: 600;
        color: #8B4513;
    }
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white !important;
        font-weight: 600;
        padding: 0.8rem;
    }
    
    .dataframe tbody tr:hover {
        background-color: #FFF8DC;
    }
    
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    .trade-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(255, 215, 0, 0.15);
        border-left: 4px solid #FFD700;
        transition: all 0.2s ease;
    }
    
    .trade-card:hover {
        box-shadow: 0 4px 8px rgba(255, 215, 0, 0.25);
        transform: translateX(4px);
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    table thead {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white;
    }
    
    table th {
        padding: 1rem;
        font-weight: 600;
        text-align: left;
    }
    
    table td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #FFE4B5;
    }
    
    table tbody tr:hover {
        background-color: #FFF8DC;
    }
    
    .gold-prediction-box {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 100%);
        border: 3px solid #FFD700;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(255, 215, 0, 0.3);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #FF6B6B 0%, #FFD700 50%, #4ECB71 100%);
        height: 30px;
        border-radius: 15px;
        position: relative;
        overflow: hidden;
    }
    
    .stSpinner > div {
        border-top-color: #FFD700 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🥇 Trading Predictor AI - Gold Analysis")
st.markdown("""
<div style='background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;'>
    <p style='color: white; font-size: 1.1rem; margin: 0; text-align: center; font-weight: 500;'>
        🥇 Analisi Oro Avanzata • 📊 Confronto Storico • 🎯 Previsioni Multi-Fattoriali • 🧠 Psicologia Investitori
    </p>
</div>
""", unsafe_allow_html=True)

# === SWITCH GENERALE PREZZI LIVE MT4 ===
if "live_prices_enabled" not in st.session_state:
    st.session_state["live_prices_enabled"] = False

switch_col1, switch_col2 = st.columns([1, 3])
with switch_col1:
    if st.button("📡 Attiva/Disattiva prezzo live MT4"):
        st.session_state["live_prices_enabled"] = not st.session_state["live_prices_enabled"]
with switch_col2:
    stato = "attivi ✅" if st.session_state["live_prices_enabled"] else "disattivati ⛔"
    st.caption(f"Prezzi live MT4 attualmente **{stato}**.")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol_input = st.text_input(
        "🔍 Seleziona Strumento (Ticker)",
        value="GC=F",
        help="GC=F (Oro), SI=F (Argento), BTC-USD, ^GSPC (S&P 500) oppure XAUUSD, BTCUSD, US500, XAGUSD da MT4"
    )
    yahoo_symbol, mt4_symbol = resolve_symbols(symbol_input)
    proper_name = (
        proper_names.get(symbol_input)
        or proper_names.get(yahoo_symbol)
        or proper_names.get(mt4_symbol, symbol_input)
    )
    st.markdown(f"**Strumento selezionato:** `{proper_name}`  \n📡 Yahoo: `{yahoo_symbol}` • 🖥️ MT4: `{mt4_symbol}`")

with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h'], index=2)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Carica Dati", use_container_width=True)

# Prezzo live MT4
live_price, prev_close = (None, None)
if st.session_state.get("live_prices_enabled", False):
    live_price, prev_close = fetch_live_price(mt4_symbol)

live_col1, live_col2 = st.columns([1, 1])
with live_col1:
    if live_price is not None:
        delta_str = None
        if prev_close is not None and prev_close != 0:
            delta_pct = (live_price - prev_close) / prev_close * 100
            delta_str = f"{delta_pct:+.2f}%"
        display_price = f"{live_price:.4f}" if live_price < 10 else f"{live_price:.2f}"
        st.metric("💹 Prezzo live (MT4)", display_price, delta_str)
    else:
        st.metric("💹 Prezzo live (MT4)", "N/D")

with live_col2:
    source = st.session_state.get("last_price_source", "sorgente sconosciuta")
    st.caption(
        f"Aggiornato alle {datetime.datetime.now().strftime('%H:%M:%S')} • Fonte: {source}"
    )

st.markdown("---")

session_key = f"model_{yahoo_symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Caricamento AI e analisi dati..."):
        model, scaler, df_ind = train_or_load_model(symbol=yahoo_symbol, interval=data_interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Sistema pronto! Modello addestrato con successo.")
        else:
            st.error("❌ Impossibile caricare dati. Verifica il ticker e riprova.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    
    # Fetch latest price: prima MT4, poi Yahoo come fallback
    if live_price is not None:
        current_price = live_price
    else:
        ticker = yf.Ticker(yahoo_symbol)
        latest_hist = ticker.history(period="1d", interval="1m")
        if not latest_hist.empty:
            current_price = float(latest_hist['Close'].iloc[-1])
        else:
            current_price = df_ind['Close'].iloc[-1]
    
    # SEZIONE SPECIALE GOLD (GC=F o XAUUSD)
    is_gold = (yahoo_symbol.upper() == "GC=F") or (mt4_symbol.upper() == "XAUUSD")
    if is_gold:
        st.markdown("## 🥇 ANALISI ORO COMPLETA - Multi-Fattoriale")
        
        with st.spinner("🔍 Analisi fondamentali e confronto storico in corso..."):
            factors = get_gold_fundamental_factors()
            gold_analysis = analyze_gold_historical_comparison(current_price, factors)
        
        st.markdown("""
        <div class='gold-prediction-box'>
            <h2 style='color: #B8860B; text-align: center; margin-bottom: 1.5rem;'>🎯 PREVISIONI PREZZO ORO</h2>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💵 Prezzo Attuale", f"${gold_analysis['current_price']:.2f}")
        with col2:
            change_3m = ((gold_analysis['target_3m'] - current_price) / current_price) * 100
            st.metric("📅 Target 3 Mesi", f"${gold_analysis['target_3m']:.2f}", f"{change_3m:+.1f}%")
        with col3:
            change_6m = ((gold_analysis['target_6m'] - current_price) / current_price) * 100
            st.metric("📅 Target 6 Mesi", f"${gold_analysis['target_6m']:.2f}", f"{change_6m:+.1f}%")
        with col4:
            change_1y = ((gold_analysis['target_1y'] - current_price) / current_price) * 100
            st.metric("📅 Target 12 Mesi", f"${gold_analysis['target_1y']:.2f}", f"{change_1y:+.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Confidence score
        st.markdown("### 🎯 Livello di Confidenza della Previsione")
        confidence = gold_analysis['confidence']
        col_conf, col_range = st.columns([1, 1])
        
        with col_conf:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem;'>
                <div class='confidence-bar' style='margin: 1rem 0;'>
                    <div style='width: {confidence}%; background: rgba(255, 255, 255, 0.3); height: 100%; display: flex; align-items: center; justify-content: center;'>
                        <span style='color: white; font-weight: bold; font-size: 1.2rem;'>{confidence:.1f}%</span>
                    </div>
                </div>
                <p style='color: #666; margin-top: 0.5rem;'>Confidenza basata su similarità storica e fattori fondamentali</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_range:
            st.markdown("#### 📊 Range di Prezzo (12 mesi)")
            st.info(f"""
            **Scenario Pessimista:** ${gold_analysis['range_low']:.2f}  
            **Scenario Base:** ${gold_analysis['target_1y']:.2f}  
            **Scenario Ottimista:** ${gold_analysis['range_high']:.2f}  
            
            Range: ${gold_analysis['range_low']:.2f} - ${gold_analysis['range_high']:.2f}
            """)
        
        st.markdown("---")
        
        # Periodo storico più simile
        st.markdown("### 📚 Confronto con Periodo Storico Più Simile")
        
        similar_period = gold_analysis['most_similar_period']
        period_data = gold_analysis['period_data']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            #### 🕰️ {similar_period}: {period_data['description']}
            
            **Similarità con contesto attuale:** {gold_analysis['similarity_pct']:.1f}%
            
            **Statistiche Periodo Storico:**
            - 💵 Prezzo Inizio: ${period_data['start_price']:.2f}
            - 💵 Prezzo Fine: ${period_data['end_price']:.2f}
            - 📈 Guadagno: {period_data['gain_pct']:.0f}%
            - ⏱️ Durata: {period_data['duration_years']} anni
            - 📊 Inflazione Media: {period_data['avg_inflation']:.1f}%
            - ⚠️ Rischio Geopolitico: {period_data['geopolitical']}/10
            
            **Eventi Chiave:** {period_data['key_events']}
            """)
        
        with col2:
            st.markdown("#### 🌍 Contesto Attuale (2025)")
            context = gold_analysis['current_context']
            
            st.markdown(f"""
            **Macro-Ambiente:**
            - 💵 Forza Dollaro: {context['dollar_strength']}
            - 📊 Tassi Reali: {context['real_rates']:.2f}%
            - 📉 Sentiment Rischio: {context['risk_sentiment']}
            - ⚠️ Rischio Geopolitico: {context['geopolitical']}/10
            - 🏦 Banche Centrali: {context['central_bank']}
            - 📈 Trend Tecnico: {context['technical_trend']}
            - 💹 Inflazione: {factors['inflation_expectations']:.2f}%
            """)
        
        st.markdown("---")
        
        # Key drivers
        st.markdown("### 🔑 Fattori Chiave che Influenzano il Prezzo dell'Oro")
        
        col1, col2, col3 = st.columns(3)
        
        drivers = gold_analysis['key_drivers']
        driver_items = list(drivers.items())
        
        with col1:
            for i in range(0, 3):
                if i < len(driver_items):
                    key, value = driver_items[i]
                    st.markdown(f"**{key}:** {value}")
        
        with col2:
            for i in range(3, 6):
                if i < len(driver_items):
                    key, value = driver_items[i]
                    st.markdown(f"**{key}:** {value}")
        
        with col3:
            for i in range(6, len(driver_items)):
                key, value = driver_items[i]
                st.markdown(f"**{key}:** {value}")
        
        st.markdown("---")
        
        # Tabella comparazione tutti i periodi
        st.markdown("### 📊 Confronto Tutti i Periodi Storici")
        
        historical_df = pd.DataFrame([
            {
                'Periodo': periodo,
                'Descrizione': data['description'],
                'Prezzo Inizio': f"${data['start_price']:.2f}",
                'Prezzo Fine': f"${data['end_price']:.2f}",
                'Guadagno %': f"{data['gain_pct']:.0f}%",
                'Durata (anni)': data['duration_years'],
                'Inflazione Media %': f"{data['avg_inflation']:.1f}%",
                'Rischio Geo': f"{data['geopolitical']}/10"
            }
            for periodo, data in gold_analysis['historical_periods'].items()
        ])
        
        st.dataframe(historical_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Interpretazione
        st.markdown("### 💡 Interpretazione e Raccomandazioni")
        
        interpretation = f"""
        Basandosi sull'analisi multi-fattoriale e sul confronto con il periodo storico più simile ({similar_period}), 
        che presenta una similarità del {gold_analysis['similarity_pct']:.1f}% con il contesto attuale:
        
        **Scenario Probabile:**
        - Il prezzo dell'oro potrebbe raggiungere **${gold_analysis['target_1y']:.2f}** entro 12 mesi
        - Questo rappresenta un potenziale guadagno del **{change_1y:+.1f}%**
        - La confidenza in questa previsione è del **{confidence:.1f}%**
        
        **Fattori Supportivi:**
        - {'✅ Tassi reali bassi favoriscono oro' if context['real_rates'] < 1.5 else '⚠️ Tassi reali elevati potrebbero limitare upside'}
        - {'✅ Dollaro debole supporta oro' if context['dollar_strength'] == 'Debole' else '⚠️ Dollaro forte potrebbe pesare'}
        - {'✅ Alta volatilità (VIX) favorisce safe-haven' if factors['vix'] > 20 else '✅ Bassa volatilità indica stabilità'}
        - {'✅ Forte domanda banche centrali' if factors['central_bank_demand'] > 800 else '⚠️ Domanda BC moderata'}
        - {'✅ Rischio geopolitico elevato supporta oro' if factors['geopolitical_risk'] > 6 else '⚠️ Contesto geopolitico stabile'}
        
        **Strategia Consigliata:**
        - **Investitori Long-Term:** Considerare accumulo graduale in range ${gold_analysis['range_low']:.2f}-${current_price:.2f}
        - **Traders:** Target tecnici a ${gold_analysis['target_3m']:.2f} (3M) e ${gold_analysis['target_6m']:.2f} (6M)
        - **Allocazione Suggerita:** 5-15% del portafoglio in oro fisico o ETF (GLD, IAU)
        - **Stop-Loss:** Sotto ${current_price * 0.92:.2f} (-8% dal prezzo attuale)
        """
        
        st.info(interpretation)
    
    # Resto del codice standard
    avg_forecast, forecast_series = predict_price(df_ind, steps=5)
    web_signals_list = get_web_signals(yahoo_symbol, df_ind)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1.2, 0.8])
   
    with col_left:
        st.markdown("### 💡 Suggerimenti Trade Intelligenti")
        if web_signals_list:
            suggestions_df = pd.DataFrame(web_signals_list)
            suggestions_df = suggestions_df.sort_values(by='Probability', ascending=False)
           
            st.markdown("**📋 Clicca su un trade per analisi approfondita AI:**")
           
            for idx, row in suggestions_df.iterrows():
                sentiment_emoji = "🟢" if row['Sentiment'] == 'Positive' else "🔴" if row['Sentiment'] == 'Negative' else "🟡"
                
                col_trade, col_btn = st.columns([5, 1])
                with col_trade:
                    st.markdown(f"""
                    <div class='trade-card'>
                        <strong style='font-size: 1.1rem; color: #FFD700;'>{row['Direction'].upper()}</strong> 
                        <span style='color: #4a5568;'>• Entry: <strong>${row['Entry']:.2f}</strong> • SL: ${row['SL']:.2f} • TP: ${row['TP']:.2f}</span><br>
                        <span style='color: #2d3748;'>📊 Probabilità: <strong>{row['Probability']:.0f}%</strong> {sentiment_emoji} Sentiment: <strong>{row['Sentiment']}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                with col_btn:
                    if st.button("🔍", key=f"analyze_{idx}", help="Analizza con AI"):
                        st.session_state.selected_trade = row
           
            with st.expander("📊 Dettagli Supplementari"):
                st.markdown("#### 📅 Analisi Stagionalità")
                st.info(suggestions_df.iloc[0]['Seasonality_Note'])
                
                st.markdown("#### 📰 News Recenti")
                st.write(suggestions_df.iloc[0]['News_Summary'])
                
                st.markdown("#### 😊 Sentiment Aggregato")
                sentiment = suggestions_df.iloc[0]['Sentiment']
                if sentiment == 'Positive':
                    st.success(f"🟢 {sentiment} - Il mercato mostra segnali positivi")
                elif sentiment == 'Negative':
                    st.error(f"🔴 {sentiment} - Il mercato mostra segnali negativi")
                else:
                    st.warning(f"🟡 {sentiment} - Il mercato è neutrale")
                
                st.markdown("#### 🔮 Previsione Prezzo")
                st.info(suggestions_df.iloc[0]['Forecast_Note'])
        else:
            st.info("ℹ️ Nessun suggerimento web disponibile.")
   
    with col_right:
        st.markdown("### 🚀 Asset con Potenziale 2025")
        
        data = [
            {"Asset": "🥇 Gold", "Ticker": "GC=F", "Score": "⭐⭐⭐⭐⭐"},
            {"Asset": "🥈 Silver", "Ticker": "SI=F", "Score": "⭐⭐⭐⭐"},
            {"Asset": "₿ Bitcoin", "Ticker": "BTC-USD", "Score": "⭐⭐⭐⭐"},
            {"Asset": "💎 Nvidia", "Ticker": "NVDA", "Score": "⭐⭐⭐⭐⭐"},
            {"Asset": "📊 S&P 500", "Ticker": "^GSPC", "Score": "⭐⭐⭐⭐"}
        ]
        growth_df = pd.DataFrame(data)
        st.dataframe(growth_df, use_container_width=True, hide_index=True)
    
    if 'selected_trade' in st.session_state:
        trade = st.session_state.selected_trade
       
        with st.spinner("🔮 Analisi AI in corso..."):
            direction = 'long' if trade['Direction'].lower() in ['long', 'buy'] else 'short'
            entry = trade['Entry']
            sl = trade['SL']
            tp = trade['TP']
           
            features = generate_features(df_ind, entry, sl, tp, direction, 60)
            success_prob = predict_success(model, scaler, features)
            factors_list = get_dominant_factors(model, features)
           
            st.markdown("---")
            
            st.markdown("### 📊 Dashboard Statistiche Real-Time")
            latest = df_ind.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("💵 Prezzo Attuale", f"${latest['Close']:.2f}")
            with col2:
                rsi_color = "🟢" if 30 <= latest['RSI'] <= 70 else "🔴"
                st.metric(f"{rsi_color} RSI", f"{latest['RSI']:.1f}")
            with col3:
                st.metric("📏 ATR", f"{latest['ATR']:.2f}")
            with col4:
                trend_emoji = "📈" if latest['Trend'] == 1 else "📉"
                trend_text = "Bullish" if latest['Trend'] == 1 else "Bearish"
                st.metric(f"{trend_emoji} Trend", trend_text)
            with col5:
                if avg_forecast is not None:
                    forecast_change = ((avg_forecast - latest['Close']) / latest['Close']) * 100
                    st.metric("🔮 Previsione", f"${avg_forecast:.2f}", f"{forecast_change:+.1f}%")
                else:
                    st.metric("🔮 Previsione", "N/A")
            
            st.markdown("---")
            st.markdown("## 🎯 Risultati Analisi AI")
           
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta = success_prob - trade['Probability']
                st.metric("🎲 Probabilità AI", f"{success_prob:.1f}%",
                         delta=f"{delta:+.1f}%" if delta != 0 else None)
            with col2:
                rr = abs(tp - entry) / abs(entry - sl)
                rr_emoji = "🟢" if rr >= 2 else "🟡" if rr >= 1.5 else "🔴"
                st.metric(f"{rr_emoji} Risk/Reward", f"{rr:.2f}x")
            with col3:
                risk_pct = abs(entry - sl) / entry * 100
                st.metric("📉 Rischio %", f"{risk_pct:.2f}%")
            with col4:
                reward_pct = abs(tp - entry) / entry * 100
                st.metric("📈 Reward %", f"{reward_pct:.2f}%")
           
            st.markdown("---")
            
            st.markdown("### 💡 Valutazione Comparativa")
            col_web, col_ai, col_final = st.columns(3)
           
            with col_web:
                st.markdown("#### 🌐 Analisi Web")
                st.markdown(f"**Probabilità:** {trade['Probability']:.0f}%")
                if trade['Probability'] >= 65:
                    st.success("✅ Setup favorevole")
                elif trade['Probability'] >= 50:
                    st.warning("⚠️ Setup neutrale")
                else:
                    st.error("❌ Setup sfavorevole")
           
            with col_ai:
                st.markdown("#### 🤖 Analisi AI")
                st.markdown(f"**Probabilità:** {success_prob:.1f}%")
                if success_prob >= 65:
                    st.success("✅ Setup favorevole")
                elif success_prob >= 50:
                    st.warning("⚠️ Setup neutrale")
                else:
                    st.error("❌ Setup sfavorevole")
            
            with col_final:
                st.markdown("#### 🎯 Verdetto Finale")
                avg_prob = (success_prob + trade['Probability']) / 2
                st.markdown(f"**Prob. Media:** {avg_prob:.1f}%")
                if abs(success_prob - trade['Probability']) > 10:
                    if success_prob > trade['Probability']:
                        st.info(f"💡 AI più ottimista (+{success_prob - trade['Probability']:.1f}%)")
                    else:
                        st.warning(f"⚠️ AI più prudente ({success_prob - trade['Probability']:.1f}%)")
                else:
                    st.success("✅ Analisi convergenti!")
           
            st.markdown("---")
            
            st.markdown("### 🔍 Fattori Chiave dell'Analisi AI")
            
            for i, factor in enumerate(factors_list, 1):
                emoji = ["🥇", "🥈", "🥉", "🏅", "🎖️"][i-1]
                st.markdown(f"{emoji} **{i}.** {factor}")
            
            st.markdown("---")
            
            st.markdown("### 🧠 Analisi Psicologica dell'Investitore")
            st.markdown("*Approfondimento comportamentale con focus su " + proper_name + "*")
            
            psych_analysis = get_investor_psychology(yahoo_symbol, trade['News_Summary'], trade['Sentiment'], df_ind)
            st.markdown(psych_analysis)
else:
    st.warning("⚠️ Seleziona uno strumento e carica i dati per iniziare l'analisi.")

with st.expander("ℹ️ Come Funziona Questo Sistema"):
    st.markdown("""
    ### 🤖 Tecnologia AI Avanzata
    
    **Machine Learning (Random Forest) analizza:**
    - 📊 **14 Indicatori Tecnici**: RSI, MACD, EMA, Bollinger Bands, ATR, Volume, Trend
    - 📈 **500+ Setup Storici**: Simulazioni basate su dati reali per training del modello
    - 🌐 **Segnali Web Real-Time**: News, sentiment, stagionalità e previsioni dinamiche
    - 🧠 **Psicologia Comportamentale**: Analisi approfondita dei bias cognitivi
    - 📚 **Comparazioni Storiche**: Pattern da crisi del 2008, COVID-19, Dot-Com
    
    ### 🥇 Analisi Speciale ORO (GC=F/XAUUSD)
    
    **Fattori Fondamentali Analizzati:**
    - 💵 Dollar Index (DXY)
    - 📊 Tassi di Interesse (Treasury 10Y, tassi reali)
    - 📉 VIX (Volatilità)
    - 📈 S&P 500 (risk-on/off)
    - 🥈 Gold/Silver Ratio
    - 💹 Aspettative Inflazione
    - 🌍 Rischio Geopolitico
    - 🏦 Domanda Banche Centrali
    - 😊 Sentiment Retail
    
    ⚠️ Questo strumento è a scopo educativo e non costituisce consiglio finanziario.
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 100%); border-radius: 12px; margin-top: 2rem;'>
    <p style='color: #8B4513; font-size: 0.95rem; margin: 0;'>
        ⚠️ <strong>Disclaimer Importante:</strong> Questo è uno strumento educativo e di ricerca. Non costituisce consiglio finanziario.<br>
        Le previsioni sono basate su modelli quantitativi e analisi storica. I risultati passati non garantiscono performance future.<br>
        Consulta sempre un professionista qualificato prima di prendere decisioni di investimento.
    </p>
    <p style='color: #A0826D; font-size: 0.85rem; margin-top: 0.5rem;'>
        🥇 Sviluppato con Machine Learning e analisi multi-fattoriale • © 2025
    </p>
</div>
""", unsafe_allow_html=True)
