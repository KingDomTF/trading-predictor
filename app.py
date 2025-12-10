import streamlit as st
from supabase import create_client
import time

# ================= CONFIGURAZIONE =================
st.set_page_config(page_title="AI Trade Command", layout="wide", page_icon="🦅")

# CSS "PALANTIR" & OPERATIVO
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    
    /* CARTA OPERATIVA */
    .trade-card {
        background: #1c1f26;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 10px;
    }
    .trade-card:hover { transform: scale(1.02); border-color: #555; }
    
    .card-label { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .card-value { font-size: 28px; font-weight: bold; color: white; margin-top: 5px; }
    
    /* COLORI LIVELLI */
    .txt-green { color: #00C805 !important; }
    .txt-red   { color: #FF3B30 !important; }
    .txt-blue  { color: #00BFFF !important; }
    .txt-gold  { color: #FFD700 !important; }
    
    /* BOX SEGNALE PRINCIPALE */
    .signal-box {
        text-align: center; padding: 25px; border-radius: 15px;
        background: rgba(255,255,255,0.03); margin-bottom: 25px;
        border: 2px solid #333;
    }
    
    /* Sidebar personalizzata */
    section[data-testid="stSidebar"] {
        background-color: #161a24;
        border-right: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# CONNESSIONE
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# ================= SIDEBAR (I PULSANTI DI SCELTA) =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
    st.title("ASSET SELECTOR")
    st.markdown("---")
    
    # IL PULSANTE DI SCELTA
    selected_asset = st.radio(
        "Scegli Strumento:",
        options=["XAUUSD", "BTCUSD", "US500", "ETHUSD", "USTEC"],
        index=0  # Default su XAUUSD
    )
    
    st.markdown("---")
    st.info(f"Visualizzando analisi per: **{selected_asset}**")
    
    st.caption("Assicurati che l'EA su MT4 stia girando sul grafico corrispondente.")

# ================= MAIN PAGE =================
st.title(f"🦅 AI STRATEGY: {selected_asset}")

placeholder = st.empty()

while True:
    try:
        # QUERY FILTRATA: Chiediamo al DB solo i dati dello strumento scelto
        response = supabase.table("ai_oracle") \
            .select("*") \
            .eq("symbol", selected_asset) \
            .order("id", desc=True) \
            .limit(1) \
            .execute()
        
        with placeholder.container():
            if response.data:
                sig = response.data[0]
                rec = sig.get('recommendation', 'WAIT')
                
                # Colori Dinamici
                main_color = "#888"
                if "BUY" in rec: main_color = "#00C805"
                elif "SELL" in rec: main_color = "#FF3B30"
                elif "WAIT" in rec: main_color = "#FFD700"

                # --- HEADER LIVELLI ---
                c1, c2 = st.columns([3, 1])
                c1.metric("Prezzo Live", f"{sig.get('current_price', 0)}")
                c2.metric("Risk/Reward", f"1:{sig.get('risk_reward', 0)}")
                
                # --- SEGNALE GIGANTE ---
                st.markdown(f"""
                <div class="signal-box" style="border-color: {main_color}; box-shadow: 0 0 20px {main_color}33;">
                    <h1 style="color: {main_color}; font-size: 56px; margin:0;">{rec}</h1>
                    <p style="color: #bbb; margin-top: 10px;">{sig.get('details', '')}</p>
                </div>
                """, unsafe_allow_html=True)

                # --- CARTE OPERATIVE (Solo se c'è un segnale attivo) ---
                if "WAIT" not in rec and "HOLD" not in rec:
                    k1, k2, k3 = st.columns(3)
                    
                    # ENTRY
                    with k1:
                        st.markdown(f"""
                        <div class="trade-card">
                            <div class="card-label">ENTRY LEVEL</div>
                            <div class="card-value">{sig.get('entry_price', 0)}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    # STOP LOSS
                    with k2:
                        st.markdown(f"""
                        <div class="trade-card" style="border-color: #FF3B30;">
                            <div class="card-label txt-red">STOP LOSS</div>
                            <div class="card-value txt-red">{sig.get('stop_loss', 0)}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    # TAKE PROFIT
                    with k3:
                        st.markdown(f"""
                        <div class="trade-card" style="border-color: #00BFFF;">
                            <div class="card-label txt-blue">TAKE PROFIT</div>
                            <div class="card-value txt-blue">{sig.get('take_profit', 0)}</div>
                        </div>""", unsafe_allow_html=True)
                
                else:
                    st.warning(f"⚠️ Nessun setup operativo chiaro per {selected_asset}. L'AI suggerisce pazienza.")

                # --- PROBABILITÀ ---
                st.write("### 🧠 AI Confidence Analysis")
                
                p_buy = sig.get('prob_buy', 0)
                p_sell = sig.get('prob_sell', 0)
                
                col_b, col_s = st.columns(2)
                
                with col_b:
                    st.progress(int(p_buy))
                    st.caption(f"BULLISH POWER: {p_buy}%")
                    
                with col_s:
                    st.progress(int(p_sell))
                    st.caption(f"BEARISH POWER: {p_sell}%")

            else:
                # SE NON CI SONO DATI PER QUEL SIMBOLO
                st.warning(f"📡 Nessun dato ricevuto per **{selected_asset}**.")
                st.markdown("""
                **Possibili Cause:**
                1. L'EA su MT4 non è attivo sul grafico di questo asset.
                2. Il mercato è chiuso.
                3. Il Bridge Python non sta girando.
                """)
                st.info("Apri il grafico su MT4 e assicurati che l'EA abbia la faccina 🙂")

        time.sleep(1)

    except Exception as e:
        time.sleep(1)
