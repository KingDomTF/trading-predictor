import os, time, streamlit as st
from datetime import datetime
from supabase import create_client

st.set_page_config(page_title="TITAN Oracle Prime", layout="wide")

# CSS Professionale allineato a sinistra per evitare errori DIV
st.markdown("""
<style>
.main { background-color: #0E1012; color: #E4E8F0; font-family: 'Inter', sans-serif; }
.signal-card { background: #1B1E23; border-radius: 16px; padding: 2rem; border-top: 4px solid #69F0AE; max-width: 600px; margin: 0 auto; }
.price-display { font-size: 4rem; font-weight: 800; text-align: center; font-family: 'Rajdhani'; }
.invalid-signal { opacity: 0.5; filter: grayscale(1); }
</style>
""", unsafe_allow_html=True)

db = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def get_data(symbol):
    try:
        s = db.table("trading_signals").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(1).execute()
        p = db.table("price_history").select("*").eq("symbol", symbol).limit(1).execute()
        return (s.data[0] if s.data else None), (p.data[0] if p.data else None)
    except: return None, None

def main():
    st.title("⚡ TITAN INSTITUTIONAL FEED")
    assets = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    tabs = st.tabs(assets)
    
    for idx, sym in enumerate(assets):
        with tabs[idx]:
            sig, price_data = get_data(sym)
            curr_p = price_data['price'] if price_data else 0
            
            # --- MOTORE DI VALIDAZIONE ---
            is_valid = False
            if sig:
                # Controlla se il prezzo attuale ha già invalidato il trade
                if sig['recommendation'] == "BUY":
                    is_valid = curr_p > sig['stop_loss'] and curr_p < sig['take_profit']
                else:
                    is_valid = curr_p < sig['stop_loss'] and curr_p > sig['take_profit']
            
            if is_valid:
                col = "#69F0AE" if sig['recommendation'] == "BUY" else "#FF5252"
                st.markdown(f"""
<div class="signal-card" style="border-top-color: {col}">
<h2 style="text-align:center; color:{col}">{sig['recommendation']}</h2>
<div class="price-display">${curr_p:,.2f}</div>
<div style="display:flex; justify-content:space-around; margin-top:20px;">
<div style="text-align:center">ENTRY<br><b>${sig['entry_price']:,.2f}</b></div>
<div style="text-align:center; color:#FF5252">STOP LOSS<br><b>${sig['stop_loss']:,.2f}</b></div>
<div style="text-align:center; color:#69F0AE">TARGET<br><b>${sig['take_profit']:,.2f}</b></div>
</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.info(f"🔍 SCANNING {sym}... No high-probability institutional setups currently active.")

    time.sleep(1)
    st.rerun()

if __name__ == "__main__": main()
