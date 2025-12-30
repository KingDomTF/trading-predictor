import streamlit as st
from supabase import create_client
import time
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="APEX Terminal", layout="wide", page_icon="🦈")

st.markdown("""
<style>
    .stApp { background-color: #000; color: #fff; font-family: 'Arial', sans-serif; }
    
    /* STATUS BAR */
    .connection-box {
        padding: 8px; border-radius: 5px; text-align: center; font-size: 14px; font-weight: bold; margin-bottom: 10px;
    }
    .status-online { background: #064e3b; border: 1px solid #34d399; color: #34d399; }
    .status-offline { background: #450a0a; border: 1px solid #f87171; color: #f87171; }
    
    .regime-bar {
        padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; letter-spacing: 2px;
        margin-bottom: 20px; text-transform: uppercase;
    }
    .regime-trend { background: #004d00; border: 1px solid #00ff00; color: #00ff00; }
    .regime-range { background: #4d4d00; border: 1px solid #ffff00; color: #ffff00; }
    .regime-chaos { background: #4d0000; border: 1px solid #ff0000; color: #ff0000; }
    
    .signal-card {
        background: #111; border: 1px solid #333; padding: 30px; border-radius: 15px;
        text-align: center; box-shadow: 0 0 30px rgba(0,255,255,0.05);
    }
    .big-sig { font-size: 64px; font-weight: 900; margin: 10px 0; }
    .buy { color: #00ff00; text-shadow: 0 0 20px rgba(0,255,0,0.5); }
    .sell { color: #ff0000; text-shadow: 0 0 20px rgba(255,0,0,0.5); }
    .wait { color: #555; }
    
    .metric-box {
        background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333; text-align: center;
    }
    .m-lbl { color: #888; font-size: 11px; text-transform: uppercase; }
    .m-val { font-size: 20px; font-weight: bold; }
    
    .price-live { font-size: 12px; color: #aaa; margin-top: 5px; }
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
    st.write("System: Regime Switching + Live Monitor")

placeholder = st.empty()

while True:
    try:
        # 1. CONTROLLO CONNESSIONE (Heartbeat)
        # Cerchiamo l'ultimo dato grezzo inserito dal bridge
        feed_resp = supabase.table("mt4_feed").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        # 2. CONTROLLO SEGNALE AI
        ai_resp = supabase.table("ai_oracle").select("*").eq("symbol", asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            # --- LOGICA CONNESSIONE ---
            is_online = False
            last_seen = "N/A"
            live_price = 0.0
            
            if feed_resp.data:
                feed = feed_resp.data[0]
                live_price = feed['price']
                
                # Calcolo tempo trascorso (Timezone aware)
                created_at = datetime.fromisoformat(feed['created_at'].replace('Z', '+00:00'))
                now = datetime.now(pytz.utc)
                diff = (now - created_at).total_seconds()
                
                if diff < 30: # Se il dato è più giovane di 30 secondi
                    is_online = True
                    last_seen = f"{int(diff)}s ago"
                else:
                    last_seen = f"{int(diff)}s ago (LAG!)"
            
            # DISEGNA BARRA DI STATO
            if is_online:
                st.markdown(f'<div class="connection-box status-online">🟢 SYSTEM ONLINE | LIVE FEED: {last_seen} | PRICE: {live_price}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="connection-box status-offline">🔴 CONNECTION LOST | LAST SEEN: {last_seen} | CHECK BRIDGE.PY</div>', unsafe_allow_html=True)

            # --- LOGICA AI ---
            if ai_resp.data:
                d = ai_resp.data[0]
                rec = d['recommendation']
                regime = d.get('market_regime', 'ANALYZING')
                
                # Calcoliamo la distanza tra Prezzo Reale e Entry Suggerito
                dist_pct = 0
                if live_price > 0 and d['entry_price'] > 0:
                    dist_pct = ((live_price - d['entry_price']) / live_price) * 100
                
                dist_msg = ""
                if abs(dist_pct) > 0.05 and "WAIT" not in rec:
                    if dist_pct > 0: dist_msg = f"(Il prezzo è {abs(dist_pct):.2f}% SOPRA l'ingresso ideale)"
                    else: dist_msg = f"(Il prezzo è {abs(dist_pct):.2f}% SOTTO l'ingresso ideale)"

                # STILI
                r_class = "regime-chaos"
                if regime == "TRENDING": r_class = "regime-trend"
                elif regime == "MEAN_REVERSION": r_class = "regime-range"
                
                s_class = "wait"
                if "BUY" in rec: s_class = "buy"
                elif "SELL" in rec: s_class = "sell"

                st.markdown(f"<div class='regime-bar {r_class}'>MARKET STATE: {regime}</div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="signal-card">
                    <h4 style="margin:0; color:#666;">STRATEGIA {asset}</h4>
                    <div class="big-sig {s_class}">{rec}</div>
                    <p style="color:#aaa;">{d['details']}</p>
                    <div class="price-live">Prezzo MT4: {live_price} vs Entry AI: {d['entry_price']} <br> {dist_msg}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if "WAIT" not in rec:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<div class='metric-box'><div class='m-lbl'>ENTRY (LIMIT)</div><div class='m-val' style='color:#00ffff'>{d['entry_price']}</div></div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='metric-box'><div class='m-lbl'>STOP LOSS</div><div class='m-val' style='color:#ff4444'>{d['stop_loss']}</div></div>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div class='metric-box'><div class='m-lbl'>TAKE PROFIT</div><div class='m-val' style='color:#00ff00'>{d['take_profit']}</div></div>", unsafe_allow_html=True)

            else:
                st.info(f"In attesa del primo segnale AI per {asset}...")
        
        time.sleep(1)
    except Exception as e:
        time.sleep(1)
