import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="Sovereign AI", layout="wide", page_icon="💎")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    
    /* BOX PRINCIPALE */
    .main-box {
        border: 2px solid #333; border-radius: 20px; padding: 30px;
        text-align: center; margin-bottom: 30px;
        background: radial-gradient(circle at center, #1a1a1a 0%, #000 100%);
    }
    
    /* LIVELLI OPERATIVI */
    .level-card {
        background: #111; border-left: 4px solid #555; padding: 15px;
        border-radius: 8px; margin: 5px 0; text-align: left;
    }
    .lvl-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .lvl-val { font-size: 24px; font-weight: bold; color: white; }
    
    /* COLORI */
    .c-green { color: #00ff9d !important; border-color: #00ff9d !important; }
    .c-red { color: #ff4d4d !important; border-color: #ff4d4d !important; }
    .c-gold { color: #ffcc00 !important; border-color: #ffcc00 !important; }
    
    .status-badge {
        display: inline-block; padding: 5px 15px; border-radius: 20px;
        background: #222; font-size: 14px; margin-top: 10px; border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

with st.sidebar:
    st.title("💎 SOVEREIGN")
    asset = st.radio("ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD"])
    st.info("Logica: Trend H1 + Macro Filter + Dynamic ATR Levels")

placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # STILE DINAMICO
                theme = "c-gold"
                if "BUY" in rec and "WAIT" not in rec: theme = "c-green"
                elif "SELL" in rec and "WAIT" not in rec: theme = "c-red"
                
                # TITOLO E DETTAGLI
                st.markdown(f"""
                <div class="main-box" style="border-color: inherit;">
                    <h4 style="color:#888; margin:0;">{asset} ANALYSIS</h4>
                    <h1 class="{theme}" style="font-size: 64px; margin: 10px 0;">{rec}</h1>
                    <p style="font-size: 18px; color: #ccc;">{d['details']}</p>
                    <div class="status-badge">{d.get('macro_filter', 'Macro Check...')}</div>
                </div>
                """, unsafe_allow_html=True)

                # LIVELLI (VISIBILI SOLO SE C'È SEGNALE ATTIVO)
                if "WAIT" not in rec:
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.markdown(f"""
                        <div class="level-card" style="border-color: #333;">
                            <div class="lvl-label">ENTRY PRICE</div>
                            <div class="lvl-val">{d['entry_price']}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown(f"""
                        <div class="level-card c-red">
                            <div class="lvl-label" style="color:#ff4d4d">STOP LOSS</div>
                            <div class="lvl-val" style="color:#ff4d4d">{d['stop_loss']}</div>
                        </div>""", unsafe_allow_html=True)
                        
                    with c3:
                        st.markdown(f"""
                        <div class="level-card c-green">
                            <div class="lvl-label" style="color:#00ff9d">TAKE PROFIT</div>
                            <div class="lvl-val" style="color:#00ff9d">{d['take_profit']}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    st.success(f"⚡ Risk/Reward: 1:{d['risk_reward']} | Probabilità AI: {max(d['prob_buy'], d['prob_sell']):.0f}%")
                
                else:
                    st.info("Il sistema attende una confluenza migliore. Trend e Macro non sono perfettamente allineati.")

                # PROBABILITA' AI
                st.write("---")
                pb = d['prob_buy']
                ps = d['prob_sell']
                
                c_b, c_s = st.columns(2)
                c_b.progress(int(pb))
                c_b.caption(f"BULLISH: {pb:.1f}%")
                
                c_s.progress(int(ps))
                c_s.caption(f"BEARISH: {ps:.1f}%")

            else:
                st.warning(f"In attesa di dati per {asset}...")
        
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
