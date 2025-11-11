import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.autograd import Variable

# ==================== FUNZIONI CORE ====================
# ... (tieni calculate_technical_indicators, generate_features, simulate_historical_trades, train_model, predict_success, get_dominant_factors come sono)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def train_lstm(df, epochs=200):
    """Addestra LSTM per forecasting."""
    scaler = StandardScaler()
    data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    X = []
    y = []
    for i in range(60, len(data)):
        X.append(data[i-60:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    
    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for i in range(epochs):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                             torch.zeros(1,1,model.hidden_layer_size))
        
        y_pred = model(X)
        
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    
    return model, scaler

def predict_price_lstm(model, scaler, df_ind, steps=5):
    """Previsione con LSTM."""
    last_60 = scaler.transform(df_ind['Close'].tail(60).values.reshape(-1,1))
    input_seq = torch.from_numpy(last_60.astype(np.float32))
    
    model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                         torch.zeros(1,1,model.hidden_layer_size))
    
    predictions = []
    for _ in range(steps):
        pred = model(input_seq)
        predictions.append(pred.item())
        input_seq = torch.cat((input_seq[1:], pred.unsqueeze(0)))
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    return predictions.mean(), predictions.flatten()

def get_sentiment(text):
    """NLP migliorato con more keywords and simple score."""
    positive_words = ['rally', 'up', 'bullish', 'gain', 'positive', 'strong', 'rise', 'surge', 'boom', 'recovery', 'growth', 'high', 'record']
    negative_words = ['down', 'bearish', 'loss', 'negative', 'weak', 'slip', 'fall', 'drop', 'crash', 'decline', 'low', 'crisis', 'recession']
    score = sum(word in text.lower() for word in positive_words) - sum(word in text.lower() for word in negative_words)
    intensity = abs(score) / max(1, len(text.split()))
    if score > 0:
        return 'Positive', score * intensity
    elif score < 0:
        return 'Negative', score * intensity
    else:
        return 'Neutral', 0

def get_gold_fundamental_factors():
    """Recupero fattori, aggiungi news sentiment dinamico."""
    factors = {}
    # ... (tieni il codice esistente per DXY, TNX, VIX, etc.)
    
    # Aggiungi news storiche/current
    ticker = yf.Ticker("GC=F")
    news = ticker.news
    news_summary = ' | '.join([item.get('title', '') for item in news if isinstance(item, dict)])
    sentiment_label, sentiment_score = get_sentiment(news_summary)
    factors['news_sentiment'] = sentiment_score
    
    # Per historical news, usa yfinance or external API (es. add newsapi)
    # Esempio: historical_sentiment = get_historical_sentiment() # Implementa con web search if needed
    
    return factors

def analyze_gold_historical_comparison(current_price, factors, df_ind):
    """Confronto dinamico con clustering, include news."""
    # Fetch historical macro data
    gold_hist = yf.download('GC=F', period='max')
    dxy_hist = yf.download('DX-Y.NYB', period='max')
    tnx_hist = yf.download('^TNX', period='max')
    vix_hist = yf.download('^VIX', period='max')
    
    hist_df = pd.concat([gold_hist['Close'], dxy_hist['Close'], tnx_hist['Close'], vix_hist['Close']], axis=1)
    hist_df.columns = ['Gold', 'DXY', 'TNX', 'VIX']
    hist_df = hist_df.dropna(how='all')
    
    hist_df['Gold_Return'] = hist_df['Gold'].pct_change(252) * 100
    hist_df['DXY_Change'] = hist_df['DXY'].pct_change(252) * 100
    hist_df['Real_Yield'] = hist_df['TNX'] - factors['inflation_expectations']
    
    # Current features
    current_features = np.array([factors['dxy_current'], factors['yield_10y'], factors['vix'], factors['inflation_expectations'], factors['news_sentiment']], dtype=float)
    
    # Clustering: Find similar periods using euclidean distance on annual rolling
    similarities = []
    for i in range(252, len(hist_df), 252):  # Annual steps
        past = hist_df.iloc[i-252:i].mean()
        past_features = np.array([past['DXY'], past['TNX'], past['VIX'], past['Real_Yield'], 0], dtype=float)  # 0 for sentiment historical (add if fetched)
        if np.any(np.isnan(past_features)):
            continue
        dist = np.linalg.norm(current_features - past_features)
        year_start = hist_df.index[i-252].year
        return_next = hist_df['Gold_Return'].iloc[i]
        similarities.append((year_start, dist, return_next))
    
    top_similar = sorted(similarities, key=lambda x: x[1])[:3]  # Top 3 periodi simili
    most_similar_year, similarity_dist, expected_return = top_similar[0]
    similarity_pct = 100 / (1 + similarity_dist)  # Normalized %
    
    # Previsione ensemble: RandomForest + LSTM + historical
    lstm_model, lstm_scaler = train_lstm(df_ind)
    lstm_mean, lstm_preds = predict_price_lstm(lstm_model, lstm_scaler, df_ind, steps=252)  # 1 year
    
    projection_historical = current_price * (1 + expected_return / 100)
    projection_fundamental = current_price * (1 + factors['news_sentiment'] / 10 + (factors['inflation_expectations'] / 100))  # Adjusted
    projection = (projection_historical * 0.3 + projection_fundamental * 0.3 + lstm_mean * 0.4)
    
    confidence = min(100, similarity_pct * 0.7 + (1 - similarity_dist / 10) * 30)  # Improved
    
    return {
        'target_1y': projection,
        'most_similar_period': most_similar_year,
        'similarity_pct': similarity_pct,
        'confidence': confidence,
        # ... (aggiungi altri come prima)
    }

# Nel main app:
# Addestra LSTM in train_or_load_model
model, scaler, df_ind = train_or_load_model(symbol, interval)
lstm_model, lstm_scaler = train_lstm(df_ind) if model is not None else (None, None)

# In predict_price, usa ensemble EMA + LSTM
def predict_price(df_ind, steps=5):
    ema_mean, ema_forecast = ... # Codice originale
    lstm_mean, lstm_forecast = predict_price_lstm(lstm_model, lstm_scaler, df_ind, steps)
    return (ema_mean + lstm_mean) / 2, (ema_forecast + lstm_forecast) / 2

# Per news storiche, aggiungi funzione
def get_historical_sentiment(period):
    # Usa yfinance or web: es. news = yf.Ticker("GC=F").news for current, per historical usa external (non in code)
    # Suggerimento: Integra newsapi.org with key for historical news search
    return 0  # Placeholder

# ... (resto del codice come originale, con chiamate alle nuove funzioni in gold_analysis)
