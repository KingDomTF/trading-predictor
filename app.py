import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="Palantir Trading Desk", layout="wide", page_icon="🧿")

# CSS ISTITUZIONALE
st.markdown("""
<style>
    .stApp { background-color: #0b0d11; color: #c9d1d9; }
    
    /* KPI MACRO */
    .macro-card {
        background: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d;
        text-align: center; margin-bottom: 10px;
    }
    .macro-title { font-size: 12px; color: #8b949e; letter-spacing: 1px; }
    .macro-val { font-size: 20px; font-weight: bold; color: #f0f6fc; }
    
    /* BOX SEGNALE */
    .signal-box {
        background: rgba(22, 27, 34, 0.8); border: 2px solid #30363d;
        padding: 30px; border-radius: 12px; text-align: center; margin-top: 20px;
    }
    
    /* CARTE OPERATIVE */
    .trade-card {
        background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
        padding: 20px; text-align: center; height: 100%;
    }
    
    .buy-color { color: #2ea043 !important; border-color: #2ea043 !important; }
    .sell-color { color: #da3633 !important; border-color: #da3633 !important; }
    .wait-color { color: #d29922 !important; border-color: #d29922 !important; }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# SIDEBAR SELETTORE
with st.sidebar:
    st.title("🧿 ORACOL SYSTEM")
    asset = st.radio("ASSET TARGET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD"], index=0)
    st.divider()
    st.info("Modalità: Intraday Scalping (H1)")

# LOOP
placeholder = st.empty()

while True:
    try:
        # Prendi dati
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                sig = resp.data[0]
                rec = sig.get('recommendation', 'WAIT')
                
                # --- 1. CONTESTO MACRO (VIX & SENTIMENT) ---
                st.markdown("### 🌍 MACRO CONTEXT ANALYSIS")
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.markdown(f"<div class='macro-card'><div class='macro-title'>VIX (FEAR INDEX)</div><div class='macro-val' style='color:#da3633'>{sig.get('vix_level', 0)}</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='macro-card'><div class='macro-title'>MARKET MOOD</div><div class='macro-val'>{sig.get('market_sentiment', '---')}</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='macro-card'><div class='macro-title'>DOLLAR TREND</div><div class='macro-val' style='color:#2ea043'>{sig.get('macro_filter', '---')}</div></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"<div class='macro-card'><div class='macro-title'>LIVE PRICE</div><div class='macro-val'>{sig.get('current_price', 0)}</div></div>", unsafe_allow_html=True)

                # --- 2. IL SEGNALE OPERATIVO ---
                color_class = "wait-color"
                if "BUY" in rec: color_class = "buy-color"
                elif "SELL" in rec: color_class = "sell-color"
                
                st.markdown(f"""
                <div class="signal-box {color_class}">
                    <h1 style="font-size: 50px; margin:0;">{rec}</h1>
                    <p style="opacity:0.7; margin-top:10px;">{sig.get('details', '')}</p>
                </div>
                """, unsafe_allow_html=True)

                # --- 3. LIVELLI DI SCALPING (SOLO SE ACTIVE) ---
                if "SCALP" in rec:
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        st.markdown(f"<div class='trade-card'><div class='macro-title'>ENTRY POINT</div><div class='macro-val'>{sig.get('entry_price')}</div></div>", unsafe_allow_html=True)
                    with k2:
                        st.markdown(f"<div class='trade-card' style='border:1px solid #da3633'><div class='macro-title' style='color:#da3633'>STOP LOSS (1 ATR)</div><div class='macro-val' style='color:#da3633'>{sig.get('stop_loss')}</div></div>", unsafe_allow_html=True)
                    with k3:
                        st.markdown(f"<div class='trade-card' style='border:1px solid #2ea043'><div class='macro-title' style='color:#2ea043'>TAKE PROFIT (2 ATR)</div><div class='macro-val' style='color:#2ea043'>{sig.get('take_profit')}</div></div>", unsafe_allow_html=True)
                    
                    st.success(f"⚡ SETUP VALIDO: Risk/Reward {sig.get('risk_reward')} | Operazione Intraday")
                
                elif "WAIT" in rec and "Macro" in rec:
                    st.error("⛔ SEGNALE BLOCCATO DAL FILTRO MACRO. (Troppa volatilità o Dollaro contro). Attendi.")
                else:
                    st.info("⏳ Scansione volatilità in corso... Nessun setup chiaro al momento.")

            else:
                st.warning(f"Nessun dato per {asset}. Attiva l'EA su MT4.")
        
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
