import streamlit as st
from supabase import create_client
import time
import pandas as pd

# CONFIGURAZIONE
st.set_page_config(page_title="AI Trade Signals", layout="wide", page_icon="🦅")

# CSS OPERATIVO
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    
    .trade-card {
        background: #1c1f26; padding: 25px; border-radius: 15px;
        border: 1px solid #333; text-align: center; margin-bottom: 10px;
    }
    .card-title { font-size: 16px; color: #888; text-transform: uppercase; }
    .card-value { font-size: 36px; font-weight: bold; color: white; }
    
    .buy-text { color: #00C805; }
    .sell-text { color: #FF3B30; }
    .tp-text { color: #00BFFF; }
    .wait-text { color: #FFD700; }
    
    .status-box { padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #444; background: #222; }
</style>
""", unsafe_allow_html=True)

# CONNESSIONE
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

st.title("🦅 AI STRATEGY ROOM")

placeholder = st.empty()

while True:
    try:
        # Prendi l'ultimo segnale
        response = supabase.table("ai_oracle").select("*").order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if response.data:
                sig = response.data[0]
                rec = sig.get('recommendation', 'WAIT')
                
                # COLORE DINAMICO
                color = "#FFD700" # Giallo (Wait)
                if "BUY" in rec: color = "#00C805"
                elif "SELL" in rec: color = "#FF3B30"
                
                # --- HEADER ---
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"## {sig.get('symbol', '---')} | LIVE: {sig.get('current_price', 0)}")
                c2.markdown(f"#### SIGNAL ID: #{sig['id']}")
                
                st.markdown("---")

                # --- SEGNALE PRINCIPALE ---
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; border: 3px solid {color}; border-radius: 15px; margin-bottom: 30px; background: rgba(0,0,0,0.3);">
                    <h1 style="color: {color}; font-size: 60px; margin: 0; letter-spacing: 2px;">{rec}</h1>
                    <p style="color: #ccc; font-size: 18px; margin-top: 10px;">{sig.get('details', 'Analisi in corso...')}</p>
                </div>
                """, unsafe_allow_html=True)

                # --- CARTE OPERATIVE (SOLO SE NON È WAIT) ---
                if "WAIT" not in rec and "HOLD" not in rec:
                    k1, k2, k3 = st.columns(3)
                    
                    with k1:
                        st.markdown(f"""
                        <div class="trade-card">
                            <div class="card-title">ENTRY PRICE</div>
                            <div class="card-value">{sig.get('entry_price', 0)}</div>
                        </div>""", unsafe_allow_html=True)
                        
                    with k2:
                        st.markdown(f"""
                        <div class="trade-card" style="border-color: #FF3B30;">
                            <div class="card-title sell-text">STOP LOSS</div>
                            <div class="card-value sell-text">{sig.get('stop_loss', 0)}</div>
                        </div>""", unsafe_allow_html=True)
                        
                    with k3:
                        st.markdown(f"""
                        <div class="trade-card" style="border-color: #00BFFF;">
                            <div class="card-title tp-text">TAKE PROFIT</div>
                            <div class="card-value tp-text">{sig.get('take_profit', 0)}</div>
                        </div>""", unsafe_allow_html=True)
                
                # --- PROBABILITA' ---
                st.write("### 📊 AI Confidence")
                p_buy = sig.get('prob_buy', 0)
                p_sell = sig.get('prob_sell', 0)
                
                c_buy, c_sell = st.columns(2)
                c_buy.progress(int(p_buy))
                c_buy.caption(f"BULLISH: {p_buy}%")
                
                c_sell.progress(int(p_sell))
                c_sell.caption(f"BEARISH: {p_sell}%")

            else:
                # STATO DI ATTESA (SE DB VUOTO)
                st.warning("📡 IL SISTEMA È CONNESSO MA IN ATTESA DEL PRIMO SEGNALE...")
                st.info("Assicurati che 'bridge.py' stia girando sul tuo PC.")

        time.sleep(1)

    except Exception as e:
        time.sleep(1)
