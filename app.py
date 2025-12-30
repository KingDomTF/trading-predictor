import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="APEX Terminal", layout="wide", page_icon="🦈")

st.markdown("""
<style>
    .stApp { background-color: #000; color: #fff; font-family: 'Arial', sans-serif; }
    
    /* STATUS BAR */
    .regime-bar {
        padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; letter-spacing: 2px;
        margin-bottom: 20px; text-transform: uppercase;
    }
    .regime-trend { background: #004d00; border: 1px solid #00ff00; color: #00ff00; }
    .regime-range { background: #4d4d00; border: 1px solid #ffff00; color: #ffff00; }
    .regime-chaos { background: #4d0000; border: 1px solid #ff0000; color: #ff0000; }
    
    /* SIGNAL CARD */
    .signal-card {
        background: #111; border: 1px solid #333; padding: 30px; border-radius: 15px;
        text-align: center; box-shadow: 0 0 30px rgba(0,255,255,0.05);
    }
    .big-sig { font-size: 64px; font-weight: 900; margin: 10px 0; }
    .buy { color: #00ff00; text-shadow: 0 0 20px rgba(0,255,0,0.5); }
    .sell { color: #ff0000; text-shadow: 0 0 20px rgba(255,0,0,0.5); }
    .wait { color: #555; }
    
    /* METRICS */
    .metric-box {
        background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333; text-align: center;
    }
    .m-lbl { color: #888; font-size: 11px; text-transform: uppercase; }
    .m-val { font-size: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

with st.sidebar:
    st.title("🦈 APEX PREDATOR")
    asset = st.radio("TARGET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"])
    st.divider()
    st.write("System: Regime Switching (Trend vs Reversion)")

placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                regime = d.get('market_regime', 'ANALYZING')
                
                # STILE REGIME
                r_class = "regime-chaos"
                if regime == "TRENDING": r_class = "regime-trend"
                elif regime == "MEAN_REVERSION": r_class = "regime-range"
                
                # STILE SEGNALE
                s_class = "wait"
                if "BUY" in rec: s_class = "buy"
                elif "SELL" in rec: s_class = "sell"

                # UI
                st.markdown(f"<div class='regime-bar {r_class}'>MARKET STATE: {regime}</div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="signal-card">
                    <h4 style="margin:0; color:#666;">{asset} STRATEGY</h4>
                    <div class="big-sig {s_class}">{rec}</div>
                    <p style="color:#aaa;">{d['details']}</p>
                    <div style="margin-top:20px; font-size:12px; color:#444;">STATISTICAL CONFIDENCE (Z-Score): {d.get('confidence_score', 0)} σ</div>
                </div>
                """, unsafe_allow_html=True)
                
                if "WAIT" not in rec:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<div class='metric-box'><div class='m-lbl'>ENTRY</div><div class='m-val' style='color:#00ffff'>{d['entry_price']}</div></div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='metric-box'><div class='m-lbl'>STOP LOSS</div><div class='m-val' style='color:#ff4444'>{d['stop_loss']}</div></div>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div class='metric-box'><div class='m-lbl'>TAKE PROFIT</div><div class='m-val' style='color:#00ff00'>{d['take_profit']}</div></div>", unsafe_allow_html=True)

            else:
                st.info(f"Connecting to APEX Engine for {asset}...")
        
        time.sleep(1)
    except: time.sleep(1)
