import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="Oracle-X Trinity", layout="wide", page_icon="🔮")

st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* TERMINAL STYLE */
    .terminal-box {
        border: 1px solid #333; background: #0a0a0a; padding: 20px;
        border-radius: 5px; margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.05);
    }
    
    .signal-text { font-size: 60px; font-weight: 900; letter-spacing: -2px; }
    .buy { color: #00ff00; text-shadow: 0 0 20px rgba(0,255,0,0.4); }
    .sell { color: #ff0000; text-shadow: 0 0 20px rgba(255,0,0,0.4); }
    .wait { color: #ffff00; }
    
    .data-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 20px; }
    .grid-item { background: #111; padding: 15px; border: 1px solid #222; text-align: center; }
    .grid-label { font-size: 10px; color: #666; text-transform: uppercase; }
    .grid-val { font-size: 20px; font-weight: bold; color: #fff; }
    
    .secular-bull { color: #00ff00; border: 1px solid #00ff00; padding: 2px 8px; font-size: 12px; }
    .secular-bear { color: #ff0000; border: 1px solid #ff0000; padding: 2px 8px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

with st.sidebar:
    st.title("🔮 ORACLE-X")
    asset = st.radio("TARGET SYSTEM", ["XAUUSD", "BTCUSD", "US500", "ETHUSD"])
    st.divider()
    st.caption("Engine: 10-Year Macro + Intraday Micro")

placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # CLASSE COLORE
                color_class = "wait"
                if "BUY" in rec: color_class = "buy"
                elif "SELL" in rec: color_class = "sell"
                
                # TREND SECOLARE
                secular_html = ""
                if "BULL" in d['macro_filter']:
                    secular_html = "<span class='secular-bull'>SECULAR BULL TREND (10Y)</span>"
                elif "BEAR" in d['macro_filter']:
                    secular_html = "<span class='secular-bear'>SECULAR BEAR TREND (10Y)</span>"

                # MAIN UI
                st.markdown(f"""
                <div class="terminal-box">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0; color:#888;">{asset} // SYSTEM STATUS</h3>
                        {secular_html}
                    </div>
                    <div class="signal-text {color_class}">{rec}</div>
                    <div style="color:#aaa; font-size:14px; margin-top:-10px;">{d['details']}</div>
                </div>
                """, unsafe_allow_html=True)

                if "WAIT" not in rec:
                    st.markdown(f"""
                    <div class="data-grid">
                        <div class="grid-item" style="border-color: #444;">
                            <div class="grid-label">ENTRY EXECUTION</div>
                            <div class="grid-val">{d['entry_price']}</div>
                        </div>
                        <div class="grid-item" style="border-color: #990000;">
                            <div class="grid-label" style="color:#ff6666">STOP LOSS (ELASTIC)</div>
                            <div class="grid-val" style="color:#ff6666">{d['stop_loss']}</div>
                        </div>
                        <div class="grid-item" style="border-color: #006600;">
                            <div class="grid-label" style="color:#66ff66">TAKE PROFIT (TARGET)</div>
                            <div class="grid-val" style="color:#66ff66">{d['take_profit']}</div>
                        </div>
                    </div>
                    <div style="text-align:center; margin-top:15px; color:#555; font-size:12px;">
                        Risk/Reward: 1:{d['risk_reward']} | AI Probability: {max(d['prob_buy'], d['prob_sell']):.0f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("System idling. Waiting for high-probability setup aligned with macro drivers.")

            else:
                st.warning(f"Initializing Oracle-X link for {asset}...")
        
        time.sleep(1)
    except: time.sleep(1)
