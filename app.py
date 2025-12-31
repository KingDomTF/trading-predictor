import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="TITAN Terminal", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .stApp { background-color: #000; color: #fff; font-family: 'Roboto', sans-serif; }
    
    /* STATUS HEADER */
    .status-ok { background: #064e3b; border: 1px solid #34d399; color: #34d399; padding: 10px; text-align: center; font-weight: bold; border-radius: 5px; }
    .status-err { background: #450a0a; border: 1px solid #f87171; color: #f87171; padding: 10px; text-align: center; font-weight: bold; border-radius: 5px; }
    
    /* SIGNAL CARD */
    .card { background: #111; border: 1px solid #333; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px; }
    .big-sig { font-size: 50px; font-weight: 900; margin: 10px 0; }
    .buy { color: #00ff00; } .sell { color: #ff0000; } .wait { color: #555; }
    
    /* METRICS */
    .metric { background: #1a1a1a; padding: 10px; border-radius: 5px; border: 1px solid #333; text-align: center; }
    .m-val { font-size: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

with st.sidebar:
    st.title("🛡️ TITAN")
    # Aggiunti tutti gli strumenti richiesti
    asset = st.radio("ASSET", ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"])
    st.divider()
    st.info("System: Data Integrity Guard + Regime Switching")

placeholder = st.empty()

while True:
    try:
        # Prendi l'ultimo segnale per l'asset selezionato
        resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                
                # --- CONTROLLO INTEGRITÀ ---
                # Se l'ultimo dato è un errore di mismatch, lo mostriamo
                if rec == "DATA ERROR":
                    st.markdown(f"""
                    <div class="status-err">
                        ⛔ ERRORE DATI CRITICO <br>
                        Il Bridge sta inviando dati sbagliati. <br>
                        {d['details']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Visualizzazione Normale
                    st.markdown(f'<div class="status-ok">🟢 SYSTEM ONLINE | ANALYZING {asset}</div>', unsafe_allow_html=True)
                    
                    s_class = "wait"
                    if "BUY" in rec: s_class = "buy"
                    elif "SELL" in rec: s_class = "sell"
                    
                    st.markdown(f"""
                    <div class="card">
                        <h3 style="color:#aaa; margin:0;">STRATEGIA {asset}</h3>
                        <div class="big-sig {s_class}">{rec}</div>
                        <p>{d['details']}</p>
                        <p style="font-size:12px; color:#555;">Confidence Z-Score: {d.get('confidence_score', 0)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "WAIT" not in rec and rec != "DATA ERROR":
                        c1, c2, c3 = st.columns(3)
                        with c1: st.markdown(f"<div class='metric'>ENTRY<br><span class='m-val' style='color:#00ffff'>{d['entry_price']}</span></div>", unsafe_allow_html=True)
                        with c2: st.markdown(f"<div class='metric'>STOP<br><span class='m-val' style='color:#ff4444'>{d['stop_loss']}</span></div>", unsafe_allow_html=True)
                        with c3: st.markdown(f"<div class='metric'>TARGET<br><span class='m-val' style='color:#00ff00'>{d['take_profit']}</span></div>", unsafe_allow_html=True)

            else:
                st.warning(f"In attesa di dati per {asset}... Assicurati che l'EA su MT4 sia attivo su questo grafico.")
        
        time.sleep(1)
    except Exception as e:
        time.sleep(1)
