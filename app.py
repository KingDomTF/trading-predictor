import streamlit as st
from supabase import create_client
import time
import pandas as pd

# CONFIGURAZIONE PAGINA
st.set_page_config(page_title="NEXUS - Causal Intelligence", layout="wide", page_icon="🔮")

# STILI CSS (UI NEON/DARK)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #e0e0e0; 
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* HEADER */
    .nexus-header {
        background: linear-gradient(90deg, #16213e 0%, #0f3460 100%);
        border: 1px solid #53a8b6;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 0 30px rgba(83, 168, 182, 0.2);
    }
    
    .asset-title {
        font-size: 18px;
        color: #53a8b6;
        font-weight: bold;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .signal-display {
        font-size: 72px;
        font-weight: 900;
        margin: 10px 0;
        letter-spacing: -3px;
        text-shadow: 0 0 25px currentColor;
    }
    
    .signal-buy { color: #00ff88; }
    .signal-sell { color: #ff4757; }
    .signal-wait { color: #ffa502; }
    
    .regime-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid #53a8b6;
        border-radius: 20px;
        padding: 8px 20px;
        font-size: 12px;
        color: #53a8b6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 10px;
    }
    
    .details-text {
        font-size: 16px;
        color: #b8b8b8;
        margin-top: 15px;
        line-height: 1.6;
    }
    
    /* LEVELS GRID */
    .levels-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin: 25px 0;
    }
    
    .level-card {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid #2c3e50;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .level-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .level-label {
        font-size: 11px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    
    .level-value {
        font-size: 32px;
        font-weight: bold;
        color: #ecf0f1;
    }
    
    .entry-card { border-left: 3px solid #3498db; }
    .sl-card { border-left: 3px solid #e74c3c; }
    .tp-card { border-left: 3px solid #2ecc71; }
    
    /* METRICS */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-top: 20px;
    }
    
    .metric-box {
        background: rgba(15, 52, 96, 0.4);
        border: 1px solid #34495e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 11px;
        color: #95a5a6;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #53a8b6;
    }
    
    /* CAUSAL INSIGHT BOX */
    .causal-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f7f 100%);
        border: 1px solid #53a8b6;
        border-radius: 12px;
        padding: 20px;
        margin-top: 25px;
    }
    
    .causal-title {
        font-size: 14px;
        color: #53a8b6;
        font-weight: bold;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .causal-text {
        font-size: 14px;
        color: #ecf0f1;
        line-height: 1.8;
    }
    
    /* PROBABILITY BARS */
    .prob-container {
        margin-top: 25px;
    }
    
    .prob-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 12px;
        margin: 10px 0;
        overflow: hidden;
        position: relative;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .prob-bull { background: linear-gradient(90deg, #00ff88, #00d9ff); }
    .prob-bear { background: linear-gradient(90deg, #ff4757, #ff6348); }
    
    .prob-label {
        font-size: 12px;
        color: #95a5a6;
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }
    
    /* FOOTER */
    .nexus-footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        color: #7f8c8d;
        font-size: 11px;
        border-top: 1px solid #2c3e50;
    }
    
    /* ANIMATIONS */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# CREDENZIALI SUPABASE
# NOTA DI SICUREZZA: Non lasciare le chiavi API direttamente nel codice in produzione.
# Usa st.secrets per maggiore sicurezza.
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Errore connessione DB: {e}")
        return None

supabase = init_db()

# SIDEBAR
with st.sidebar:
    st.markdown("## 🔮 PROJECT NEXUS")
    st.markdown("---")
    
    asset = st.radio(
        "TARGET ASSET",
        ["XAUUSD", "BTCUSD", "US500", "ETHUSD"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style='font-size: 11px; color: #7f8c8d; line-height: 1.6;'>
    <b>NEXUS FRAMEWORK:</b><br>
    • Causal Analysis (10Y)<br>
    • Regime Detection<br>
    • Adaptive Execution<br>
    • Multi-Asset Intelligence
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    refresh_rate = st.slider("Refresh (sec)", 1, 10, 2)

# MAIN CONTENT
placeholder = st.empty()

while True:
    try:
        # Fetch latest signal
        if supabase:
            resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if supabase and resp.data and len(resp.data) > 0:
                d = resp.data[0]
                rec = d['recommendation']
                
                # Determina colore segnale
                signal_class = "signal-wait"
                if "BUY" in rec:
                    signal_class = "signal-buy"
                elif "SELL" in rec:
                    signal_class = "signal-sell"
                
                # HEADER PRINCIPALE
                st.markdown(f"""
                <div class="nexus-header">
                    <div class="asset-title">
                        <span class="live-indicator"></span>
                        {asset} // CAUSAL INTELLIGENCE
                    </div>
                    <div class="signal-display {signal_class}">{rec}</div>
                    <div class="regime-badge">{d.get('macro_filter', 'ANALYZING')}</div>
                    <div class="details-text">{d.get('details', 'Analyzing market structure...')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # LIVELLI OPERATIVI (solo se non WAIT)
                if "WAIT" not in rec:
                    st.markdown("""
                    <div class="levels-container">
                        <div class="level-card entry-card">
                            <div class="level-label">🎯 ENTRY PRICE</div>
                            <div class="level-value">{}</div>
                        </div>
                        <div class="level-card sl-card">
                            <div class="level-label">🛡️ STOP LOSS</div>
                            <div class="level-value">{}</div>
                        </div>
                        <div class="level-card tp-card">
                            <div class="level-label">💎 TAKE PROFIT</div>
                            <div class="level-value">{}</div>
                        </div>
                    </div>
                    """.format(
                        d.get('entry_price', '---'),
                        d.get('stop_loss', '---'),
                        d.get('take_profit', '---')
                    ), unsafe_allow_html=True)
                    
                    # METRICS
                    prob_val = max(d.get('prob_buy', 0), d.get('prob_sell', 0))
                    st.markdown(f"""
                    <div class="metrics-row">
                        <div class="metric-box">
                            <div class="metric-label">⚖️ Risk/Reward Ratio</div>
                            <div class="metric-value">1:{d.get('risk_reward', '-')}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">📊 Confidence Score</div>
                            <div class="metric-value">{prob_val:.0f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    # Modalità WAIT
                    st.markdown("""
                    <div class="causal-box">
                        <div class="causal-title">🧠 System Status</div>
                        <div class="causal-text">
                        The causal engine is analyzing macro drivers and market regime. 
                        Currently, there is no high-probability setup that aligns both 
                        technical patterns and fundamental causal factors. System remains 
                        in observation mode to avoid low-quality signals.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # PROBABILITÀ (Fixing the specific display error)
                p_buy = int(d.get('prob_buy', 0))
                p_sell = int(d.get('prob_sell', 0))
                
                st.markdown("""
                <div class="prob-container">
                    <div class="prob-label">
                        <span>BULLISH PROBABILITY</span>
                        <span><b>{}%</b></span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill prob-bull" style="width: {}%"></div>
                    </div>
                    
                    <div class="prob-label" style="margin-top: 15px;">
                        <span>BEARISH PROBABILITY</span>
                        <span><b>{}%</b></span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill prob-bear" style="width: {}%"></div>
                    </div>
                </div>
                """.format(p_buy, p_buy, p_sell, p_sell), unsafe_allow_html=True)
                
                # CAUSAL INSIGHT (Spiegazione del "Perché")
                st.markdown(f"""
                <div class="causal-box" style="margin-top: 25px;">
                    <div class="causal-title">🔬 Causal Analysis Insight</div>
                    <div class="causal-text">
                    {d.get('details', '')}<br><br>
                    This recommendation is based on a 10-year historical analysis of 
                    causal drivers that have statistically significant predictive power 
                    over {asset}. The system identified the dominant macro forces currently 
                    affecting this asset and adjusted its strategy accordingly.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # FOOTER
                st.markdown("""
                <div class="nexus-footer">
                    PROJECT NEXUS v1.0 // CAUSAL TRADING INTELLIGENCE<br>
                    Powered by 10-Year Historical Analysis • Regime Detection • Adaptive ML
                </div>
                """, unsafe_allow_html=True)
                
            else:
                # Nessun dato o Standby
                st.markdown(f"""
                <div class="nexus-header">
                    <div class="asset-title">
                        <span class="live-indicator"></span>
                        {asset} // INITIALIZING
                    </div>
                    <div class="signal-display signal-wait">STANDBY</div>
                    <div class="details-text">
                    Waiting for market data feed...<br>
                    Ensure bridge_nexus.py is running and MT4 is sending price updates.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        time.sleep(refresh_rate)
        
    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
        time.sleep(refresh_rate)
