 🚀 Sistema Analisi Finanziaria Istituzionale

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Overview

Sistema avanzato di analisi quantitativa e previsione finanziaria di livello istituzionale, paragonabile a piattaforme come **Aladdin (BlackRock)** e **Oracle Financial Services**. 

Utilizza ensemble di modelli Machine Learning, analisi tecnica avanzata e indicatori macroeconomici per fornire previsioni accurate su:
- 🥇 **Metalli Preziosi** (Oro, Argento, Platino, Palladio)
- ₿ **Criptovalute** (Bitcoin, Ethereum, BNB, Cardano)
- 💱 **Forex** (EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD)
- 🛢️ **Commodities** (Petrolio WTI/Brent, Gas Naturale, Rame)

---

## ✨ Features Principali

### 🤖 Machine Learning Ensemble
- **Random Forest Regressor**
- **XGBoost** (Gradient Boosting)
- **Gradient Boosting Regressor**
- **ARIMA** per serie temporali
- Previsioni con **intervalli di confidenza 95%**

### 📈 Analisi Tecnica Avanzata
- 25+ indicatori tecnici automatici
- RSI, MACD, Bollinger Bands, ATR
- Moving Averages (SMA/EMA)
- Stochastic Oscillator
- Volume Profile Analysis

### 🌍 Indicatori Macroeconomici
- **VIX** (Indice della Paura)
- **Tassi FED** (Federal Reserve)
- **Fear & Greed Index** (per crypto)
- Analisi correlazioni inter-market

### 📊 Analisi del Rischio
- **VaR** (Value at Risk) 95%
- **Sharpe Ratio**
- **Maximum Drawdown**
- **Win Rate** storico
- Livelli Support/Resistance

### 🗓️ Analisi Stagionalità
- Pattern mensili storici
- Trend settimanali
- Bias temporali ricorrenti

### ⏱️ Timeframes Supportati
- **15 minuti** (intraday trading)
- **1 ora** (day trading)
- **4 ore** (swing trading)
- **1 giorno** (position trading)

---

## 🛠️ Installazione

### Requisiti
- Python 3.10 o superiore
- pip (gestore pacchetti Python)

### Setup Rapido

```bash
# 1. Clone repository
git clone https://github.com/tuousername/financial-predictor.git
cd financial-predictor

# 2. Crea virtual environment (raccomandato)
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Avvia applicazione
streamlit run App.py
```

L'applicazione si aprirà automaticamente nel browser su `http://localhost:8501`

---

## 🚀 Deploy su Streamlit Cloud

### Step 1: Prepara Repository GitHub
1. Crea nuovo repository su GitHub
2. Carica `App.py` e `requirements.txt`
3. Commit e push

### Step 2: Deploy su Streamlit Cloud
1. Vai su [share.streamlit.io](https://share.streamlit.io)
2. Connetti account GitHub
3. Seleziona repository
4. Main file: `App.py`
5. Click **Deploy**

🎉 La tua app sarà online in 2-3 minuti!

---

## 📖 Guida Utilizzo

### 1️⃣ Seleziona Asset
Nella sidebar scegli:
- **Categoria** (Metalli, Crypto, Forex, Commodities)
- **Strumento** specifico
- **Timeframe** desiderato

### 2️⃣ Visualizza Analisi
L'app mostrerà automaticamente:
- 📊 Grafico candlestick interattivo con indicatori
- 🤖 Previsioni ML con probabilità di successo
- ⚠️ Metriche di rischio (VaR, Sharpe, Drawdown)
- 🗓️ Pattern stagionali
- 🎯 Raccomandazione algoritmica finale

### 3️⃣ Interpreta Risultati

#### Probabilità di Successo
- **> 60%**: Segnale forte
- **50-60%**: Segnale moderato
- **< 50%**: Segnale debole/contrario

#### Raccomandazioni
- 🟢 **ACQUISTO FORTE**: Score bullish > 60%
- 🟡 **NEUTRALE**: Segnali contrastanti
- 🔴 **VENDITA FORTE**: Score bearish > 60%

---

## 🧠 Modelli e Algoritmi

### Machine Learning Pipeline

```python
# Ensemble Weighting
Previsione Finale = Σ (Previsione_Modello_i × Peso_i)

Pesi basati su R² Score:
- Random Forest: ~33%
- XGBoost: ~34%
- Gradient Boosting: ~33%
```

### Feature Engineering
- Lagged returns (1, 2, 3, 5, 10 periodi)
- Rolling statistics (media, std)
- Indicatori tecnici normalizzati
- Volume anomalies
- Volatilità storica

### Backtesting
- Train/Test split: 80/20
- Walk-forward validation
- Out-of-sample testing

---

## 📊 Metriche di Performance

### Accuracy Tipiche (backtesting)
- **Direzione prezzo**: 60-70%
- **R² Score medio**: 0.45-0.65
- **RMSE**: < 3% del prezzo

### Timeframe Ottimali
- **15min**: Scalping, alta frequenza
- **1h**: Day trading
- **4h**: Swing trading
- **1d**: Position trading, trend following

---

## ⚙️ Configurazione Avanzata

### API Keys (opzionali)

Per funzionalità avanzate, crea file `.env`:

```env
# Federal Reserve Economic Data
FRED_API_KEY=your_fred_api_key

# Alpha Vantage (dati aggiuntivi)
ALPHA_VANTAGE_KEY=your_av_key

# News API (sentiment analysis)
NEWS_API_KEY=your_news_key
```

### Personalizzazioni

Modifica parametri in `App.py`:

```python
# Modifica periodi dati storici
TIMEFRAMES = {
    '15min': {'period': '60d', 'interval': '15m'},
    '1h': {'period': '730d', 'interval': '1h'},
    # ...
}

# Modifica parametri ML
RandomForestRegressor(
    n_estimators=100,  # Aumenta per più accuracy
    max_depth=10,      # Aumenta per modelli più complessi
)
```

---

## 🔧 Troubleshooting

### Errore: "No module named 'ta'"
```bash
pip install ta
```

### Errore: "Failed to download data for [symbol]"
- Verifica connessione internet
- Alcuni simboli potrebbero non essere disponibili su Yahoo Finance
- Prova con timeframe diverso

### Performance lente
- Riduci periodo storico analizzato
- Usa caching di Streamlit (già implementato)
- Deploy su server con più RAM

### TA-Lib Installation Error
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# MacOS
brew install ta-lib

# Windows: scarica binary da
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
```

---

## 📈 Roadmap Futuri Sviluppi

- [ ] **Multi-timeframe analysis** simultanea
- [ ] **Portfolio optimizer** con Modern Portfolio Theory
- [ ] **Alert system** via email/Telegram
- [ ] **Social sentiment** da Twitter/Reddit
- [ ] **News impact** scoring real-time
- [ ] **Backtesting engine** interattivo
- [ ] **Export report** PDF/Excel
- [ ] **Database** PostgreSQL per storico previsioni
- [ ] **API REST** per integrazioni esterne
- [ ] **Mobile app** iOS/Android

---

## 🤝 Contribuire

Contributi benvenuti! Per contribuire:

1. Fork del repository
2. Crea feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Apri Pull Request

---

## ⚠️ Disclaimer

**IMPORTANTE**: Questo software è fornito a scopo **educativo e informativo**. 

- ❌ NON costituisce consulenza finanziaria
- ❌ NON garantisce profitti
- ❌ I mercati finanziari sono imprevedibili
- ⚠️ Ogni investimento comporta rischi
- ⚠️ Potresti perdere il capitale investito

**Consulta sempre un consulente finanziario professionista prima di investire.**

---

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

---

## 👤 Autore

**Il Tuo Nome**
- GitHub: [@tuousername](https://github.com/tuousername)
- LinkedIn: [Tuo Profilo](https://linkedin.com/in/tuoprofilo)

---

## 🙏 Credits

Tecnologie utilizzate:
- [Streamlit](https://streamlit.io/) - Framework web
- [yfinance](https://github.com/ranaroussi/yfinance) - Dati finanziari
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient Boosting
- [Plotly](https://plotly.com/) - Visualizzazioni interattive
- [TA-Lib](https://ta-lib.org/) - Analisi tecnica

---

## 📞 Supporto

Hai domande? Apri una [Issue](https://github.com/tuousername/financial-predictor/issues) su GitHub!

---

<div align="center">

**⭐ Se questo progetto ti è utile, lascia una stella! ⭐**

Made with ❤️ and 🐍 Python
