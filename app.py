import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== FUNZIONI CORE ====================

def calculate_technical_indicators(df):
    """Calcola indicatori tecnici."""
    df = df.copy()
    
    # EMA
    df['EMA_20_60'] = df['Close'].ewm(span=20).mean()
    df['EMA_50_60'] = df['Close'].ewm(span=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_60'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD_60'] = exp1 - exp2
    df['MACD_signal_60'] = df['MACD_60'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_middle_60'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper_60'] = df['BB_middle_60'] + (bb_std * 2)
    df['BB_lower_60'] = df['BB_middle_60'] - (bb_std * 2)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_60'] = true_range.rolling(14).mean()
    
    # Volume
    df['Volume_MA_60'] = df['Volume'].rolling(window=20).mean()
    
    df = df.dropna()
    return df

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    """Genera features per la predizione."""
    latest = df_ind.iloc[-1]
    
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100
    
    features = {
        'entry_price': entry,
        'sl_distance_pct': sl_distance,
        'tp_distance_pct': tp_distance,
        'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0,
        'main_tf': main_tf,
        'current_price': latest['Close'],
        'rsi': latest['RSI_60'],
        'macd': latest['MACD_60'],
        'macd_signal': latest['MACD_signal_60'],
        'atr': latest['ATR_60'],
        'ema_20': latest['EMA_20_60'],
        'ema_50': latest['EMA_50_60'],
        'bb_upper': latest['BB_upper_60'],
        'bb_lower': latest['BB_lower_60'],
        'volume_ratio': latest['Volume'] / latest['Volume_MA_60'] if latest['Volume_MA_60'] > 0 else 1.0
    }
    
    feature_vector = np.array(list(features.values()), dtype=np.float32)
    return feature_vector

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
        
        # Simula outcome
        future_prices = df_ind.iloc[idx+1:idx+51]['Close'].values
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

class TradeSuccessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TradePredictorNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, epochs=50, batch_size=32):
    """Addestra il modello."""
    dataset = TradeSuccessDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TradePredictorNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    return model

def predict_success(model, features):
    """Predice probabilità di successo."""
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        prob = model(features_tensor).item()
    return prob * 100

def get_dominant_factors(model, features):
    """Identifica fattori dominanti (semplificato)."""
    feature_names = ['Entry Price', 'SL Distance %', 'TP Distance %', 'R/R Ratio', 
                     'Direction', 'TimeFrame', 'Current Price', 'RSI', 'MACD', 
                     'MACD Signal', 'ATR', 'EMA 20', 'EMA 50', 'BB Upper', 'BB Lower', 'Volume Ratio']
    
    indices = np.argsort(np.abs(features))[-5:][::-1]
    return [f"{feature_names[i]}: {features[i]:.2f}" for i in indices if i < len(feature_names)]

# ==================== STREAMLIT APP ====================

@st.cache_data
def load_sample_data(symbol='GC=F', period='1y'):
    """Carica dati reali da yfinance."""
    try:
        data = yf.download(symbol, period=period, progress=False)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except:
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
        data = pd.DataFrame({
            'Open': 2000 + np.cumsum(np.random.randn(1000) * 0.5),
            'High': 2000 + np.cumsum(np.random.randn(1000) * 0.5) + np.random.rand(1000),
            'Low': 2000 + np.cumsum(np.random.randn(1000) * 0.5) - np.random.rand(1000),
            'Close': 2000 + np.cumsum(np.random.randn(1000) * 0.5),
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        return data

@st.cache_resource
def train_or_load_model():
    """Addestra il modello."""
    data = load_sample_data()
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=300)
    model = train_model(X, y, epochs=30)
    return model

# Configurazione pagina
st.set_page_config(
    page_title="Trading Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato per mobile
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📈 Trading Success Predictor")
st.markdown("**Stima la probabilità di successo delle tue operazioni XAU/USD**")

# Sidebar
with st.sidebar:
    st.header(⚙️ Parametri Operazione")
    
    direction = st.selectbox("📊 Direzione", ["long", "short"])
    entry = st.number_input("💰 Entry Price", value=2000.0, step=0.1, format="%.2f")
    sl = st.number_input("🛑 Stop Loss", value=1980.0, step=0.1, format="%.2f")
    tp = st.number_input("🎯 Take Profit", value=2050.0, step=0.1, format="%.2f")
    
    main_tf = st.selectbox(
        "⏰ Time Frame",
        [15, 60, 240, 1440],
        format_func=lambda x: f"{x}min" if x < 60 else (f"{x//60}H" if x < 1440 else "D1")
    )
    
    st.markdown("---")
    refresh_data = st.button("🔄 Aggiorna Dati", use_container_width=True)

# Inizializzazione dati
if 'df_ind' not in st.session_state or refresh_data:
    with st.spinner("📊 Caricamento dati..."):
        data = load_sample_data()
        st.session_state.df_ind = calculate_technical_indicators(data)
        st.success("✅ Dati aggiornati!")

df_ind = st.session_state.df_ind

# Inizializzazione modello
if 'model' not in st.session_state:
    with st.spinner("🧠 Addestramento modello AI..."):
        st.session_state.model = train_or_load_model()
        st.success("✅ Modello pronto!")

model = st.session_state.model

# Predizione
if st.button("🚀 Calcola Probabilità", type="primary", use_container_width=True):
    with st.spinner("🔮 Analisi in corso..."):
        features = generate_features(df_ind, entry, sl, tp, direction, main_tf)
        success_prob = predict_success(model, features)
        factors = get_dominant_factors(model, features)
        
        # Risultato principale
        st.markdown("### 🎯 Risultato Predizione")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Probabilità Successo", f"{success_prob:.1f}%")
        with col2:
            rr = abs(tp - entry) / abs(entry - sl)
            st.metric("Risk/Reward", f"{rr:.2f}")
        with col3:
            st.metric("Direzione", direction.upper())
        
        # Interpretazione
        if success_prob >= 70:
            st.success("✅ **Probabilità ALTA** - Setup favorevole")
        elif success_prob >= 50:
            st.warning("⚠️ **Probabilità MEDIA** - Valuta attentamente")
        else:
            st.error("❌ **Probabilità BASSA** - Setup sfavorevole")
        
        # Fattori dominanti
        st.markdown("### 📊 Fattori Principali")
        for i, factor in enumerate(factors, 1):
            st.write(f"{i}. {factor}")
        
        # Grafico
        st.markdown("### 📈 Indicatori Tecnici (ultimi 100 periodi)")
        chart_data = df_ind.tail(100)[['Close', 'EMA_20_60', 'EMA_50_60']].copy()
        chart_data.columns = ['Prezzo', 'EMA 20', 'EMA 50']
        st.line_chart(chart_data)
        
        # RSI
        st.markdown("### 📉 RSI")
        rsi_data = df_ind.tail(100)[['RSI_60']].copy()
        rsi_data.columns = ['RSI']
        st.line_chart(rsi_data)

# Upload CSV personalizzato
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📁 Dati Personalizzati")
    uploaded_file = st.file_uploader(
        "Carica CSV (Date, Open, High, Low, Close, Volume)",
        type=['csv']
    )
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            df_ind = calculate_technical_indicators(data)
            st.session_state.df_ind = df_ind
            st.success("✅ CSV caricato!")
        except Exception as e:
            st.error(f"❌ Errore: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>⚠️ Disclaimer: App per scopi educativi. Non costituisce consiglio finanziario.</small>
</div>
""", unsafe_allow_html=True)
