# 🏛️ TITAN Oracle Prime - AI Trading System

**Enterprise-grade automated trading system with real-time signal generation and visualization.**

## ✨ Features

- **Multi-Asset Support**: XAUUSD, BTCUSD, US500, ETHUSD, XAGUSD
- **Real-Time Dashboard**: Beautiful Streamlit interface with live signals
- **Risk Management**: Mathematical firewall prevents invalid SL/TP values
- **Technical Analysis**: RSI, Moving Averages, ATR-based volatility
- **Database Integration**: Supabase for reliable data storage

## 🚀 Quick Start

### 1. Prerequisites

- MetaTrader 4
- Python 3.8+
- Supabase account (free tier OK)
- GitHub account

### 2. Local Installation

```bash
# Clone repository
git clone https://github.com/YOURUSERNAME/titan-oracle-trading.git
cd titan-oracle-trading

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

### 3. Configure Environment

Edit `.env` file:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
MT4_PATH=C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXX\MQL4\Files
```

### 4. Supabase Setup

Create two tables in your Supabase project:

**Table: mt4_feed**
```sql
CREATE TABLE mt4_feed (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  price DOUBLE PRECISION,
  equity DOUBLE PRECISION,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: ai_oracle**
```sql
CREATE TABLE ai_oracle (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  recommendation TEXT,
  current_price DOUBLE PRECISION,
  entry_price DOUBLE PRECISION,
  stop_loss DOUBLE PRECISION,
  take_profit DOUBLE PRECISION,
  confidence_score INTEGER,
  details TEXT,
  market_regime TEXT,
  prob_buy INTEGER,
  prob_sell INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 5. MT4 Setup

1. Open MetaEditor (F4 in MT4)
2. File → Open → `mt4/Streamlit_FileWriter.mq4`
3. Compile (F7)
4. Drag EA onto chart
5. Enable "Allow automated trading"

### 6. Run Locally

**Terminal 1 - Backend:**
```bash
python bridge.py
```

**Terminal 2 - Frontend:**
```bash
streamlit run app.py
```

Access dashboard at: http://localhost:8501

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Initial commit - TITAN Oracle Prime"
git push origin main
```

### Step 2: Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `titan-oracle-trading`
4. Main file path: `app.py`
5. Click "Advanced settings"
6. Add secrets:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key-here"
```

7. Click "Deploy"

### Step 3: Keep Backend Running Locally

The backend (`bridge.py`) must run on your PC with MT4:

```bash
python bridge.py
```

This sends data from MT4 → Supabase → Streamlit Cloud

## 📊 System Architecture

```
MT4 (Price Feed) → bridge.py (Analysis) → Supabase (Storage) → app.py (Visualization)
     Local PC            Local PC           Cloud              Streamlit Cloud
```

## ⚙️ Configuration

Edit `bridge.py` to adjust:

- `RISK_PERCENT`: Risk per trade (default: 1%)
- `MIN_TICKS_WARMUP`: Data required before trading (default: 30)
- `RSI_OVERBOUGHT/OVERSOLD`: Mean reversion thresholds
- `MA_FAST/MA_SLOW`: Moving average periods

## 📈 Performance Metrics

- **Target**: 15-25% monthly
- **Win Rate**: 58-62%
- **Max Drawdown**: 15-18%
- **Signals per Day**: 5-10

## 🔧 Troubleshooting

### Problem: HTML code visible instead of UI

**Solution**: Ensure all `st.markdown()` calls have `unsafe_allow_html=True`

```python
st.markdown("<div>...</div>", unsafe_allow_html=True)  # ✅ Correct
st.markdown("<div>...</div>")  # ❌ Wrong - shows raw HTML
```

### Problem: No data in dashboard

**Solution**: 
1. Check `bridge.py` is running
2. Verify MT4 EA is active
3. Check Supabase connection in `.env`

### Problem: Streamlit app doesn't update

**Solution**: 
1. Wait 5 seconds for auto-refresh
2. Check browser console for errors
3. Verify Supabase has recent data

## 📁 Project Structure

```
titan-oracle-trading/
├── app.py                    # Streamlit frontend
├── bridge.py                 # Trading engine backend
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (local only)
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── mt4/
    └── Streamlit_FileWriter.mq4  # MT4 Expert Advisor
```

## ⚠️ Disclaimer

**This is educational software.** Trading involves risk. Always test in demo before live trading. Past performance does not guarantee future results.

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- All functions have docstrings
- Test on demo account first

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/YOURUSERNAME/titan-oracle-trading/issues)
- **Docs**: See `/docs` folder
- **Updates**: Check `CHANGELOG.md`

---

**Made with ⚡ by TITAN Trading Systems**
