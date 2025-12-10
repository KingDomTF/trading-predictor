import streamlit as st
from supabase import create_client
import time

# CONFIGURAZIONE
st.set_page_config(page_title="AI Trade Signals", layout="wide", page_icon="🎯")

# CSS OPERATIVO PULITO
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    
    /* CARTA SEGNALE */
    .trade-card {
        background: #1c1f26;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
        transition: transform 0.3s;
    }
    .trade-card:hover { transform: scale(1.02); }
    
    .card-title { font-size: 18px; color: #888; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
    .card-value { font-size: 32px; font-weight: bold; color: white; }
    .card-sub { font-size: 14px; color: #666; margin-top: 5px; }

    /* COLORI DINAMICI */
    .buy-text { color: #00C805 !important; }
    .sell-text { color: #FF3B30 !important; }
    .tp-text { color: #00BFFF !important; }
    
    /* BOX PROBABILITA */
    .prob-box {
        background: #262626;
        border-radius: 8px;
        padding: 10px;
        margin-top: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# CONNESSIONE
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# HEADER
st.title("🎯 AI OPERATIONAL SIGNALS")
st.markdown("Analisi istituzionale live: Entry, Stop Loss & Take Profit dinamici.")

placeholder = st.empty()

while True:
    try:
        # Prendi l'ultimo segnale operativo
        oracle = supabase.table("ai_oracle").select("*").order("id", desc=True).limit(1).execute()
        
        if oracle.data:
            sig = oracle.data[0]
            rec = sig['recommendation']
            
            with placeholder.container():
                
                # --- RIGA 1: IL VERDETTO ---
                st.markdown(f"### ASSET: {sig['symbol']} | PREZZO: {sig['current_price']}")
                
                # Colore dinamico del titolo
                title_color = "#888"
                if "BUY" in rec: title_color = "#00C805"
                elif "SELL" in rec: title_color = "#FF3B30"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {title_color}; border-radius: 10px; margin-bottom: 30px;">
                    <h1 style="color: {title_color}; font-size: 48px; margin: 0;">{rec}</h1>
                    <p style="color: #ccc;">{sig['details']}</p>
                </div>
                """, unsafe_allow_html=True)

                # --- RIGA 2: I TRE LIVELLI (LE CARTE) ---
                if "WAIT" not in rec:
                    c1, c2, c3 = st.columns(3)
                    
                    # CARTA ENTRY
                    with c1:
                        st.markdown(f"""
                        <div class="trade-card">
                            <div class="card-title">ENTRY PRICE</div>
                            <div class="card-value">{sig['entry_price']}</div>
                            <div class="card-sub">Prezzo di Mercato</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # CARTA STOP LOSS
                    with c2:
                        st.markdown(f"""
                        <div class="trade-card" style="border-color: #FF3B30;">
                            <div class="card-title sell-text">STOP LOSS</div>
                            <div class="card-value sell-text">{sig['stop_loss']}</div>
                            <div class="card-sub">Protezione Dinamica (ATR)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # CARTA TAKE PROFIT
                    with c3:
                        st.markdown(f"""
                        <div class="trade-card" style="border-color: #00BFFF;">
                            <div class="card-title tp-text">TAKE PROFIT</div>
                            <div class="card-value tp-text">{sig['take_profit']}</div>
                            <div class="card-sub">Target (RR {sig['risk_reward']})</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Mercato incerto o laterale. L'AI suggerisce di ATTENDERE un setup più chiaro.")

                # --- RIGA 3: PROBABILITA ---
                st.markdown("### 📊 Analisi Probabilistica")
                
                st.write("LONG (Rialzo)")
                st.progress(int(sig['prob_buy']))
                
                st.write("SHORT (Ribasso)")
                st.progress(int(sig['prob_sell']))
                
                st.caption(f"Ultimo aggiornamento: {time.strftime('%H:%M:%S')}")
                
        else:
            st.info("⏳ Attesa dati dal motore strategico...")
            
        time.sleep(1)
        
    except Exception as e:
        time.sleep(1)
