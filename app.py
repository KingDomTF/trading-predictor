import streamlit as st
from supabase import create_client
import time
import json

st.set_page_config(page_title="Tri-Vector Trading", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .stApp { background-color: #0b0c10; color: #c5c6c7; }
    
    /* STRATEGY CARD */
    .strat-card {
        background: #1f2833; border: 1px solid #45a29e; border-radius: 15px;
        padding: 20px; text-align: center; margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .strat-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(69, 162, 158, 0.2); }
    
    .strat-title { font-size: 20px; font-weight: bold; color: #66fcf1; margin-bottom: 10px; text-transform: uppercase; }
    .strat-prob { font-size: 14px; color: #aaa; margin-bottom: 15px; }
    
    .data-row { display: flex; justify-content: space-between; margin: 10px 0; border-bottom: 1px solid #2b3642; padding-bottom: 5px; }
    .data-label { font-size: 12px; color: #888; }
    .data-val { font-size: 16px; font-weight: bold; color: white; }
    
    .buy-tag { background: #1f4037; color: #4effbf; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .sell-tag { background: #4a0d0d; color: #ff6b6b; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    
    .main-header { text-align: center; margin-bottom: 40px; }
    .live-price { font-size: 60px; font-weight: 800; color: #fff; text-shadow: 0 0 20px rgba(102, 252, 241, 0.5); }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

with st.sidebar:
    st.title("🦅 TRI-VECTOR")
    asset = st.radio("ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD"])

placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                data = resp.data[0]
                
                # HEADER
                st.markdown(f"<div class='main-header'><h3>{asset} LIVE MARKET</h3><div class='live-price'>{data['current_price']}</div></div>", unsafe_allow_html=True)
                
                # RECUPERO LE 3 STRATEGIE (JSON)
                strategies = data.get('strategies')
                
                if strategies:
                    c1, c2, c3 = st.columns(3)
                    
                    # Funzione per disegnare la card
                    def draw_card(col, strat):
                        with col:
                            tag_class = "buy-tag" if strat['type'] == "BUY" else "sell-tag"
                            st.markdown(f"""
                            <div class='strat-card'>
                                <div class='strat-title'>{strat['name']}</div>
                                <span class='{tag_class}'>{strat['type']}</span>
                                <div class='strat-prob'>Probabilità Stimata: {strat['prob']}%</div>
                                <hr style='border-color: #45a29e; opacity: 0.3;'>
                                <div class='data-row'><span class='data-label'>ENTRY</span><span class='data-val'>{strat['entry']}</span></div>
                                <div class='data-row'><span class='data-label'>STOP LOSS</span><span class='data-val' style='color:#ff6b6b'>{strat['sl']}</span></div>
                                <div class='data-row'><span class='data-label'>TAKE PROFIT</span><span class='data-val' style='color:#4effbf'>{strat['tp']}</span></div>
                                <p style='font-size:12px; color:#888; margin-top:15px; font-style:italic;'>"{strat['desc']}"</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Disegna le 3 strategie
                    if len(strategies) >= 3:
                        draw_card(c1, strategies[0]) # Fortress
                        draw_card(c2, strategies[1]) # Tactician
                        draw_card(c3, strategies[2]) # Hunter
                    else:
                        st.warning("Strategie in elaborazione...")
                else:
                    st.info("Formattazione dati in corso... attendere il prossimo tick.")

            else:
                st.warning(f"In attesa di dati per {asset}...")
        
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
