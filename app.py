import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader4 as mt4
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pykalman import KalmanFilter
from supabase import create_client
import os, time
from dotenv import load_dotenv

# Caricamento configurazioni [cite: 2]
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Setup Pagina (mantenendo il tuo branding TITAN) 
st.set_page_config(page_title="TITAN Oracle - Quant Terminal", layout="wide", initial_sidebar_state="collapsed")

def apply_institutional_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap');
    .main { background-color: #0E1012; color: #E4E8F0; }
    .metric-card {
        background: #1B1E23; border: 1px solid #2D333B;
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .status-live { color: #69F0AE; font-family: 'Rajdhani'; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

def get_realtime_kalman(symbol, n_ticks=150):
    """Calcola lo stato e il residuo in tempo reale via MT4."""
    ticks = mt4.copy_ticks_from(symbol, time.time(), n_ticks, mt5.COPY_TICKS_ALL)
    if ticks is None: return None
    
    df = pd.DataFrame(ticks)
    prices = df['last'].values
    
    # Parametri del Filtro: R (osservazione) e Q (transizione)
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], 
                      initial_state_mean=prices[0], observation_covariance=1, 
                      transition_covariance=0.01)
    
    state_means, state_covs = kf.filter(prices)
    state_means = state_means.flatten()
    
    # Residuo: e = Prezzo - Stima
    residuals = prices - state_means
    z_score = (residuals - np.mean(residuals)) / np.std(residuals)
    
    return df['time'], prices, state_means, z_score

def render_dashboard():
    apply_institutional_css()
    
    # Header TITAN 
    st.markdown('<h1 style="font-family:Rajdhani; text-align:center;">🏛️ TITAN QUANT TERMINAL v2</h1>', unsafe_allow_html=True)
    
    # Sidebar: System Health
    with st.sidebar:
        st.header("Core Status")
        is_mt4 = mt4.initialize()
        st.write(f"MT5 Connection: {'✅ ACTIVE' if is_mt5 else '❌ OFFLINE'}")

    # Layout principale
    col_left, col_right = st.columns([3, 1])
    
    symbol = "XAUUSD" # Asset primario
    data = get_realtime_kalman(symbol)
    
    if data:
        times, prices, k_estimate, z_scores = data
        
        with col_left:
            # Grafico combinato: Prezzo/Kalman e Residui
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Subplot 1: Price vs Kalman
            fig.add_trace(go.Scatter(x=times, y=prices, name="Raw Price", line=dict(color='#444', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=times, y=k_estimate, name="Kalman State", line=dict(color='#69F0AE', width=2)), row=1, col=1)
            
            # Subplot 2: Z-Score Residui
            fig.add_trace(go.Bar(x=times, y=z_scores, name="Z-Score Resid", marker_color='#FF5252'), row=2, col=1)
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="red", row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_dark", showlegend=False, 
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            # Metrics Istituzionali
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Current Z-Score", f"{z_scores[-1]:.2f}", delta_color="inverse")
            st.metric("Est. Volatility (ATR)", "0.0014")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Lista ultimi trade da Supabase 
            st.markdown("### Recent Executions")
            trades = supabase.table("trading_signals").select("*").order("created_at", desc=True).limit(5).execute()
            if trades.data:
                for t in trades.data:
                    st.caption(f"{t['created_at'][11:19]} | {t['recommendation']} @ {t['entry_price']}")

    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    if not mt4.initialize():
        st.error("Terminal MT4 non trovato. Assicurati che sia aperto.")
    else:
        render_dashboard()
