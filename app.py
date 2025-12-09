import streamlit as st
from supabase import create_client
import pandas as pd
import time
import plotly.express as px

# ==========================================
# CONFIGURAZIONE CLOUD
# ==========================================
# Qui Streamlit legge dal Magazzino (Supabase), non tocca MT4.
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
st.markdown("### Connessione Satellite: Supabase Cloud")

# Creiamo i contenitori vuoti che aggiorneremo
kpi_container = st.empty()
chart_container = st.empty()
raw_data_container = st.empty()

# Loop di aggiornamento automatico
while True:
    try:
        # 1. Scarica gli ultimi 50 dati dal Cloud
        response = supabase.table("mt4_feed").select("*").order("id", desc=True).limit(50).execute()
        data = response.data

        if data:
            df = pd.DataFrame(data)
            # Converti timestamp
            df['created_at'] = pd.to_datetime(df['created_at'])
            # Ordina per il grafico (dal più vecchio al più nuovo)
            df = df.sort_values('created_at')

            # Prendi l'ultimo dato arrivato
            last_tick = df.iloc[-1]

            # 2. Aggiorna KPI
            with kpi_container.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("Simbolo", last_tick['symbol'])
                col2.metric("Prezzo Ask", f"{last_tick['price']:.5f}")
                col3.metric("Equity", f"€ {last_tick['equity']:.2f}")

            # 3. Aggiorna Grafico
            with chart_container.container():
                fig = px.line(df, x='created_at', y='price', title=f"Andamento {last_tick['symbol']}")
                st.plotly_chart(fig, use_container_width=True)

            # 4. Tabella Dati
            with raw_data_container.container():
                with st.expander("Vedi dati grezzi"):
                    st.dataframe(df.sort_values('created_at', ascending=False))

        else:
            st.warning("Il database è vuoto. Avvia la MT4 e il Bridge Python sul tuo PC!")

        # Pausa per non saturare la CPU del server
        time.sleep(1)

    except Exception as e:
        st.error(f"Errore durante il refresh: {e}")
        time.sleep(5)
