import streamlit as st
from supabase import create_client
import pandas as pd
import time
import plotly.express as px
from datetime import datetime

# ==========================================
# CONFIGURAZIONE CLOUD
# ==========================================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

# Setup Pagina
st.set_page_config(
    page_title="MT4 Live Dashboard",
    page_icon="📈",
    layout="wide"
)

# Connessione al Database
@st.cache_resource
def init_db():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    supabase = init_db()
except Exception as e:
    st.error(f"Errore connessione Database: {e}")
    st.stop()

# ==========================================
# INTERFACCIA GRAFICA
# ==========================================
st.title("📡 MT4 Institutional Monitor")

# Creiamo i contenitori vuoti
kpi_container = st.empty()
chart_container = st.empty()
raw_data_container = st.empty()

# Loop di aggiornamento
while True:
    try:
        # 1. Scarica gli ultimi 50 dati
        response = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(50).execute()
        data = response.data

        if data:
            df = pd.DataFrame(data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at')

            # Prendi l'ultimo dato
            last_tick = df.iloc[-1]
            
            # --- AGGIORNAMENTO KPI ---
            with kpi_container.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Simbolo", last_tick['symbol'])
                col2.metric("Prezzo", f"{last_tick['price']:.5f}")
                col3.metric("Equity", f"€ {last_tick['equity']:.2f}")
                # Mostra il commento dell'AI se presente
                col4.metric("AI Signal", last_tick.get('comment', 'N/A'))

            # --- AGGIORNAMENTO GRAFICO (FIX APPLICATO QUI) ---
            with chart_container.container():
                fig = px.line(df, x='created_at', y='price', title=f"Andamento {last_tick['symbol']}")
                
                # IL TRUCCO: Generiamo una chiave unica basata sull'orario esatto
                unique_key = f"chart_{datetime.now().timestamp()}"
                
                st.plotly_chart(fig, use_container_width=True, key=unique_key)

            # --- AGGIORNAMENTO TABELLA ---
            with raw_data_container.container():
                with st.expander("Vedi dati grezzi"):
                    st.dataframe(df.sort_values('created_at', ascending=False))

        else:
            st.warning("Il database è vuoto. Avvia la MT4 e il Bridge Python!")

        # Pausa di 1 secondo
        time.sleep(1)

    except Exception as e:
        # Se capita un errore momentaneo, non mostrare tutto il traceback rosso,
        # ma aspetta e riprova silenziosamente.
        time.sleep(1)
