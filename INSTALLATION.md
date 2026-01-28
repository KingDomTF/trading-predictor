# 📥 Complete Installation Guide

## Step 1: System Requirements

### Hardware
- CPU: 2+ cores
- RAM: 4GB minimum
- Storage: 1GB free space

### Software
- Windows 10/11 or Linux/Mac
- MetaTrader 4
- Python 3.8 or higher
- Git (optional)

## Step 2: Clone Repository
```bash
git clone https://github.com/yourusername/titan-oracle-trading.git
cd titan-oracle-trading
```

## Step 3: Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Supabase Configuration

1. Go to https://supabase.com
2. Create new project (free tier)
3. Go to Settings → API
4. Copy "Project URL" and "anon public" key
5. Paste in `.env` file

## Step 5: Create Database Tables

Run SQL queries in Supabase SQL Editor (see README.md)

## Step 6: MT4 Expert Advisor

1. Copy `mt4/Streamlit_FileWriter.mq4`
2. Open MT4 → Tools → Options → Expert Advisors
3. Enable:
   - ✅ Allow automated trading
   - ✅ Allow DLL imports
   - ✅ Allow external experts
4. MetaEditor → Open file → Compile (F7)
5. Drag onto chart → Allow automated trading

## Step 7: Verify Installation

**Test Backend:**
```bash
python backend/bridge.py
```
Should print: `ORACLE SYSTEM ONLINE`

**Test Frontend:**
```bash
streamlit run frontend/app.py
```
Should open browser at localhost:8501

## Troubleshooting

### MT4 EA not writing files
- Check: Tools → Options → Expert Advisors (permissions)
- Verify path: `C:\Users\...\MQL4\Files\`

### Backend not connecting to Supabase
- Check `.env` file credentials
- Test connection: `python -c "from supabase import create_client; ..."`

### Frontend shows no data
- Verify backend is running
- Check Supabase tables have data
- Refresh page (F5)
