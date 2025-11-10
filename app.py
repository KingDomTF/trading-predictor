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
        'sl_distance_pct': sl_distance
