import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    from supabase import create_client
except ImportError:
    st.error("⚠️ Librerie mancanti. Verifica il file requirements.txt")
    st.stop()

class AppConfig:
    PAGE_TITLE = "TITAN Oracle Prime"
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    AUTO_REFRESH_RATE = 5 

st.set_page_config(
    page_title=AppConfig.PAGE_TITLE, 
    page_icon=AppConfig.PAGE_ICON, 
    layout=AppConfig.LAYOUT, 
    initial_sidebar_state="collapsed"
)

def load_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@400;600&display=swap');
* { box-sizing: border-box; }
.main { 
    background-color: #0E1012; 
    background-image: linear-gradient(145deg, #16181C 0%, #0B0D0F 100%); 
    color: #E4E8F0; 
    font-family: 'Inter', sans-serif; 
}
#MainMenu, footer {visibility: hidden;}
.titan-header { 
    background: #1B1E23; 
    border: 1px solid #2D333B; 
    border-radius: 16px; 
    padding: 2rem; 
    margin-bottom: 2rem; 
    text-align: center; 
    border-top: 3px solid #69F0AE; 
}
.titan-title { 
    font-family: 'Rajdhani'; 
    font-size: 3rem; 
    font-weight: 700; 
    color: #FFFFFF; 
    margin: 0; 
}
.titan-subtitle { 
    font-family: 'Rajdhani'; 
    font-size: 1rem; 
    color: #69F0AE; 
    text-transform: uppercase; 
    letter-spacing: 2px; 
}
.stTabs [data-baseweb="tab-list"] { 
    background-color: #16181C; 
    padding: 8px; 
    border-radius: 12px; 
    border: 1px solid #2D333B; 
    justify-content: center; 
}
.stTabs [data-baseweb="tab"] { 
    color: #888; 
    font-family: 'Rajdhani'; 
    font-weight: 600; 
    font-size: 1.1rem; 
}
.stTabs [aria-selected="true"] { 
    background-color: #252930; 
    color: #69F0AE !important; 
    border-radius: 8px; 
}
.signal-card { 
    background: #1B1E23; 
    border: 1px solid #2D333B; 
    border-radius: 16px; 
    padding: 2.5rem; 
    margin: 1rem auto; 
    max-width: 700px; 
    box-shadow: 0 15px 35px rgba(0,0,0,0.3); 
}
.signal-card-buy { border-top: 4px solid #69F0AE; }
.signal-card-sell { border-top: 4px solid #FF5252; }
.signal-symbol { 
    font-size: 1.2rem; 
    color: #69F0AE; 
    font-weight: 600; 
    text-align: center; 
    font-family: 'Rajdhani'; 
}
.price-display { 
    font-size: 4rem; 
    font-weight: 700; 
    color: #FFF; 
    text-align: center; 
    font-family: 'Rajdhani'; 
    margin: 15px 0; 
}
.stats-grid { 
    display: grid; 
    grid-template-columns: 1fr 1fr 1fr; 
    gap: 15px; 
    margin-top: 25px; 
}
.stat-box { 
    background: #23272E; 
    border: 1px solid #2D333B; 
    border-radius: 10px; 
    padding: 15px; 
    text-align: center; 
}
.stat-label { 
    font-size: 0.7rem; 
    color: #888; 
    text-transform: uppercase; 
}
.stat-value { 
    font-size: 1.2rem; 
    font-weight: 700; 
    color: #FFF; 
    font-family: 'Rajdhani'; 
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_supabase():
    """Initialize Supabase client using secrets or env vars"""
    # Try Streamlit secrets first (for Cloud deployment)
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    except:
        # Fallback to environment variables (for local)
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        st.error("⚠️ SUPABASE_URL e SUPABASE_KEY non configurati!")
        st.info("Configura i secrets su Streamlit Cloud o il file .env locale")
        return None
    
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Errore connessione Supabase: {e}")
        return None

supabase = init_supabase()

def get_latest_signal(symbol):
    """Get most recent signal for a symbol"""
    if not supabase: 
        return None
    try:
        res = supabase.table("trading_signals")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        return res.data[0] if res.data else None
    except Exception as e:
        st.error(f"Errore caricamento {symbol}: {e}")
        return None

def get_24h_stats():
    """Get 24h trading statistics"""
    if not supabase: 
        return None
    try:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        res = supabase.table("trading_signals")\
            .select("*")\
            .gte("created_at", cutoff)\
            .in_("recommendation", ["BUY", "SELL"])\
            .execute()
        
        if not res.data: 
            return None
        
        total = len(res.data)
        buy = sum(1 for s in res.data if s['recommendation'] == 'BUY')
        sell = sum(1 for s in res.data if s['recommendation'] == 'SELL')
        avg_conf = sum(s.get('confidence_score', 0) for s in res.data) / total if total > 0 else 0
        
        return {
            'total': total, 
            'buy': buy, 
            'sell': sell, 
            'confidence': avg_conf
        }
    except Exception as e:
        st.error(f"Errore statistiche: {e}")
        return None

def render_signal_panel(symbol, signal_data):
    """Render signal card for a symbol"""
    # Check if data is stale (> 10 min old)
    is_stale = False
    time_str = "N/A"
    
    if signal_data:
        try:
            created_at = signal_data.get('created_at')
            if created_at:
                t = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                diff = (datetime.now(t.tzinfo) - t).total_seconds()
                is_stale = diff > 600  # 10 minutes
                time_str = f"{int(diff/60)}m ago" if diff < 3600 else f"{int(diff/3600)}h ago"
        except:
            is_stale = True

    # Market closed view
    if not signal_data or is_stale:
        p = signal_data.get('current_price', 0) if signal_data else 0
        st.markdown(f"""
<div class="signal-card" style="border-top: 4px solid #444;">
<div class="signal-symbol">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:2.5rem; font-weight:800; color:#555; text-align:center;">MARKET CLOSED</div>
<div class="price-display" style="color:#444;">${p:,.2f}</div>
<div style="text-align:center; color:#444; font-size:0.8rem;">Last Seen: {time_str}</div>
</div>
""", unsafe_allow_html=True)
        return

    # Active signal view
    rec = signal_data.get('recommendation', 'WAIT')
    price = signal_data.get('current_price', 0)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    conf = signal_data.get('confidence_score', 0)
    
    col = "#69F0AE" if rec == 'BUY' else "#FF5252"
    cls = "signal-card-buy" if rec == 'BUY' else "signal-card-sell"

    st.markdown(f"""
<div class="signal-card {cls}">
<div class="signal-symbol">{symbol}</div>
<div style="font-family:'Rajdhani'; font-size:3rem; font-weight:800; color:{col}; text-align:center;">{rec}</div>
<div class="price-display">${price:,.2f}</div>
<div class="stats-grid">
<div class="stat-box">
    <div class="stat-label">STOP LOSS</div>
    <div class="stat-value" style="color:#FF5252;">${sl:,.2f}</div>
</div>
<div class="stat-box">
    <div class="stat-label">CONFIDENCE</div>
    <div class="stat-value">{conf}%</div>
</div>
<div class="stat-box">
    <div class="stat-label">TARGET</div>
    <div class="stat-value" style="color:#69F0AE;">${tp:,.2f}</div>
</div>
</div>
<div style="text-align:center; color:#888; font-size:0.75rem; margin-top:15px;">{time_str}</div>
</div>
""", unsafe_allow_html=True)

def main():
    load_custom_css()
    
    # Header
    st.markdown(
        '<div class="titan-header">'
        '<div class="titan-title">TITAN ORACLE</div>'
        '<div class="titan-subtitle">Enterprise Trading Intelligence</div>'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # 24h Stats
    stats = get_24h_stats()
    c1, c2, c3, c4 = st.columns(4)
    
    with c1: 
        st.metric("Signals (24h)", stats['total'] if stats else 0)
    with c2: 
        st.metric("Buy", stats['buy'] if stats else 0)
    with c3: 
        st.metric("Sell", stats['sell'] if stats else 0)
    with c4: 
        st.metric("Avg Conf", f"{stats['confidence'] if stats else 0:.0f}%")
    
    # Asset tabs
    tabs = st.tabs(AppConfig.ASSETS)
    
    for idx, symbol in enumerate(AppConfig.ASSETS):
        with tabs[idx]:
            signal = get_latest_signal(symbol)
            render_signal_panel(symbol, signal)
    
    # Auto-refresh
    time.sleep(AppConfig.AUTO_REFRESH_RATE)
    st.rerun()

if __name__ == "__main__": 
    main()
