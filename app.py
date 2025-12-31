import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="TITAN Oracle Terminal",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

# --- CSS CUSTOM PER REPLICARE L'ESTETICA "REACT/TAILWIND" ---
st.markdown("""
<style>
    /* Sfondo Generale Nero */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Nascondere elementi standard di Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Stile Card simil-React */
    .titan-card {
        background-color: #111827; /* gray-900 */
        border: 1px solid #1f2937; /* gray-800 */
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Testi e Colori */
    .text-gray-400 { color: #9ca3af; }
    .text-emerald-400 { color: #34d399; }
    .text-emerald-500 { color: #10b981; }
    .text-red-400 { color: #f87171; }
    .text-cyan-400 { color: #22d3ee; }
    .text-purple-400 { color: #c084fc; }
    .text-yellow-400 { color: #facc15; }
    
    .font-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .font-bold { font-weight: 700; }
    
    /* Titolo Gradiente */
    .titan-title {
        background: -webkit-linear-gradient(left, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.25rem;
    }
    
    /* Pulsanti Asset Personalizzati */
    div.stButton > button {
        background-color: #1f2937;
        color: #9ca3af;
        border: none;
        border-radius: 0.5rem;
        font-family: monospace;
    }
    div.stButton > button:hover {
        background-color: #374151;
        color: white;
    }
    div.stButton > button:focus {
        background-color: #059669;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGICA DI BACKEND (Replicata dal codice React originale) ---

def calculate_ml_score(price, trend, strength):
    trend_score = 30 if trend == 'BULLISH' else 20
    momentum_score = strength * 0.4
    volume_score = np.random.random() * 20
    sentiment_score = np.random.random() * 15
    macro_score = np.random.random() * 10
    return min(100, trend_score + momentum_score + volume_score + sentiment_score + macro_score)

def detect_patterns():
    all_patterns = [
        {'name': 'Head & Shoulders', 'probability': 0.75, 'bullish': False},
        {'name': 'Double Bottom', 'probability': 0.82, 'bullish': True},
        {'name': 'Bull Flag', 'probability': 0.68, 'bullish': True},
        {'name': 'Ascending Triangle', 'probability': 0.71, 'bullish': True},
        {'name': 'Bear Flag', 'probability': 0.65, 'bullish': False},
        {'name': 'Cup & Handle', 'probability': 0.79, 'bullish': True}
    ]
    # Mescola e prendi 2 o 3 pattern
    np.random.shuffle(all_patterns)
    return all_patterns[:2 + int(np.random.random() * 1.5)]

def generate_market_data(asset_symbol):
    # Prezzi base come da codice originale
    base_prices = {
        'XAUUSD': 2650, 'BTCUSD': 95000, 'US500': 5900,
        'ETHUSD': 3400, 'XAGUSD': 31, 'EURUSD': 1.08, 'GBPUSD': 1.27
    }
    base_price = base_prices.get(asset_symbol, 100)
    
    price = base_price + (np.random.random() - 0.5) * base_price * 0.02
    trend = 'BULLISH' if np.random.random() > 0.5 else 'BEARISH'
    strength = 60 + np.random.random() * 35
    
    ml_score = calculate_ml_score(price, trend, strength)
    
    if ml_score > 70: signal = 'STRONG BUY'
    elif ml_score > 55: signal = 'BUY'
    elif ml_score < 30: signal = 'STRONG SELL'
    elif ml_score < 45: signal = 'SELL'
    else: signal = 'WAIT'
    
    is_buy = 'BUY' in signal
    
    return {
        'price': price,
        'signal': signal,
        'ml_score': ml_score,
        'trend': trend,
        'strength': strength,
        'entry': price * (1.001 if is_buy else 0.999),
        'stop_loss': price * (0.985 if is_buy else 1.015),
        'take_profit': price * (1.035 if is_buy else 0.965),
        'volatility': 0.8 + np.random.random() * 2.5,
        'sharpe_ratio': 1.2 + np.random.random() * 1.8,
        'max_drawdown': -(5 + np.random.random() * 15),
        'win_rate': 55 + np.random.random() * 25,
        'patterns': detect_patterns(),
        'order_flow': {
            'buy_volume': np.random.random() * 100,
            'sell_volume': np.random.random() * 100,
            'institutional_flow': (np.random.random() - 0.5) * 50
        },
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }

# --- STATO DELL'APP ---
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = 'XAUUSD'
if 'is_live' not in st.session_state:
    st.session_state.is_live = True

# --- INTERFACCIA UTENTE (Layout) ---

# 1. Header & Controlli
col_h1, col_h2 = st.columns([3, 1])

with col_h1:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <div style='font-size: 3rem;'>🛡️</div>
            <div>
                <h1 class='titan-title' style='margin:0; padding:0; line-height: 1.2;'>TITAN ORACLE</h1>
                <p class='text-gray-400' style='margin:0; font-size: 0.9rem;'>Institutional-Grade Trading Intelligence</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_h2:
    # Live Status Badge
    status_html = f"""
    <div style='display: flex; justify-content: flex-end; align-items: center; height: 100%;'>
        <div style='background: rgba(6, 78, 59, 0.5); border: 1px solid #10b981; padding: 0.5rem 1rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem;'>
            <div style='width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; box-shadow: 0 0 10px #10b981;'></div>
            <span class='font-mono text-emerald-400' style='font-size: 0.8rem;'>LIVE FEED</span>
        </div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)

# Asset Selector
assets = ['XAUUSD', 'BTCUSD', 'US500', 'ETHUSD', 'XAGUSD', 'EURUSD', 'GBPUSD']
st.markdown("<br>", unsafe_allow_html=True)
cols = st.columns(len(assets))
for i, asset in enumerate(assets):
    if cols[i].button(asset, key=asset, use_container_width=True):
        st.session_state.selected_asset = asset

# Generazione Dati (Simulazione Loop)
# Se 'Live' è attivo, ricarichiamo i dati. In Streamlit, usiamo un placeholder o ricarichiamo al click.
# Per semplicità qui calcoliamo i dati al render.
data = generate_market_data(st.session_state.selected_asset)

st.markdown("---")

# --- GRIGLIA PRINCIPALE ---
# Replica layout: Colonna Sinistra (Segnali), Centro (Grafici), Destra (Info)
col_left, col_mid, col_right = st.columns([3.5, 4.5, 4])

# --- COLONNA SINISTRA: SEGNALI & PREZZI ---
with col_left:
    # 1. AI SIGNAL BOX
    signal_color = "#10b981" if "BUY" in data['signal'] else "#ef4444" if "SELL" in data['signal'] else "#6b7280"
    signal_icon = "📈" if "BUY" in data['signal'] else "📉" if "SELL" in data['signal'] else "📻"
    
    st.markdown(f"""
    <div class='titan-card' style='text-align: center;'>
        <div class='text-gray-400' style='font-size: 0.8rem; margin-bottom: 0.5rem; font-weight: bold;'>AI SIGNAL</div>
        <div style='color: {signal_color}; font-size: 3.5rem; font-weight: 900; line-height: 1;'>
            {signal_icon} <br> {data['signal']}
        </div>
        <div class='text-gray-400' style='margin-top: 1rem; font-size: 0.9rem;'>
            {st.session_state.selected_asset} • H1
        </div>
        
        <div style='margin-top: 1.5rem; text-align: left;'>
            <div style='display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;'>
                <span>ML Confidence</span>
                <span>{data['ml_score']:.1f}%</span>
            </div>
            <div style='width: 100%; background-color: #374151; height: 0.5rem; border-radius: 9999px;'>
                <div style='width: {data['ml_score']}%; background: linear-gradient(90deg, #10b981, #34d399); height: 100%; border-radius: 9999px;'></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if data['signal'] != 'WAIT':
        # 2. ENTRY/STOP/TARGET
        rr_ratio = abs(data['take_profit'] - data['entry']) / abs(data['entry'] - data['stop_loss'])
        
        st.markdown(f"""
        <div class='titan-card'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>
                <span class='text-gray-400' style='font-size: 0.8rem;'>CURRENT PRICE</span>
                <span class='text-cyan-400 font-mono font-bold' style='font-size: 1.5rem;'>${data['price']:.2f}</span>
            </div>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; text-align: center;'>
                <div style='background: rgba(8, 145, 178, 0.2); border: 1px solid #0e7490; padding: 0.5rem; border-radius: 0.5rem;'>
                    <div class='text-cyan-400' style='font-size: 0.7rem;'>ENTRY</div>
                    <div class='font-mono font-bold' style='font-size: 0.9rem;'>${data['entry']:.2f}</div>
                </div>
                <div style='background: rgba(127, 29, 29, 0.2); border: 1px solid #b91c1c; padding: 0.5rem; border-radius: 0.5rem;'>
                    <div class='text-red-400' style='font-size: 0.7rem;'>STOP</div>
                    <div class='font-mono font-bold' style='font-size: 0.9rem;'>${data['stop_loss']:.2f}</div>
                </div>
                <div style='background: rgba(6, 78, 59, 0.2); border: 1px solid #047857; padding: 0.5rem; border-radius: 0.5rem;'>
                    <div class='text-emerald-400' style='font-size: 0.7rem;'>TARGET</div>
                    <div class='font-mono font-bold' style='font-size: 0.9rem;'>${data['take_profit']:.2f}</div>
                </div>
            </div>
            <div style='margin-top: 1rem; display: flex; justify-content: space-between; font-size: 0.8rem;'>
                <span class='text-gray-400'>Risk/Reward</span>
                <span class='text-emerald-400 font-bold'>1:{rr_ratio:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 3. PATTERNS
    patterns_html = ""
    for p in data['patterns']:
        p_color = "text-emerald-400" if p['bullish'] else "text-red-400"
        p_bg = "rgba(6, 78, 59, 0.3)" if p['bullish'] else "rgba(127, 29, 29, 0.3)"
        p_tag = "BULL" if p['bullish'] else "BEAR"
        patterns_html += f"""
        <div style='background: #1f2937; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                <span style='font-size: 0.9rem; font-weight: 600;'>{p['name']}</span>
                <span style='font-size: 0.7rem; background: {p_bg}; color: {p_color if p['bullish'] else '#f87171'}; padding: 2px 6px; border-radius: 4px;'>{p_tag}</span>
            </div>
            <div style='display: flex; align-items: center; gap: 0.5rem;'>
                <div style='flex: 1; height: 6px; background: #374151; border-radius: 4px;'>
                    <div style='width: {p['probability']*100}%; height: 100%; background: {'#10b981' if p['bullish'] else '#ef4444'}; border-radius: 4px;'></div>
                </div>
                <span class='text-gray-400' style='font-size: 0.75rem;'>{int(p['probability']*100)}%</span>
            </div>
        </div>
        """
    
    st.markdown(f"""
    <div class='titan-card'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>
            <h3 class='font-bold text-gray-400'>⚡ DETECTED PATTERNS</h3>
        </div>
        {patterns_html}
    </div>
    """, unsafe_allow_html=True)


# --- COLONNA CENTRALE: GRAFICI ---
with col_mid:
    # 1. CUMULATIVE P&L CHART
    st.markdown("<div class='titan-card' style='padding-bottom: 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='font-bold text-gray-400' style='margin-bottom: 1rem;'>CUMULATIVE P&L</h3>", unsafe_allow_html=True)
    
    # Dati fittizi per il grafico
    dates = [f"Day {i+1}" for i in range(30)]
    cumulative = np.cumsum(np.random.randn(30) * 200 + 50) + 1000
    
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=dates, y=cumulative, fill='tozeroy',
        mode='lines',
        line=dict(color='#10b981', width=2),
        fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    fig_area.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        xaxis=dict(showgrid=True, gridcolor='#374151', tickfont=dict(color='#6b7280', size=10)),
        yaxis=dict(showgrid=True, gridcolor='#374151', tickfont=dict(color='#6b7280', size=10))
    )
    st.plotly_chart(fig_area, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. MULTI-FACTOR ANALYSIS (RADAR)
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='font-bold text-gray-400' style='margin-bottom: 1rem;'>MULTI-FACTOR ANALYSIS</h3>", unsafe_allow_html=True)
    
    categories = ['Momentum', 'Mean Rev', 'Volatility', 'Volume', 'Sentiment', 'Macro']
    r_values = [75, 45, 68, 82, 58, 71]  # Static values from original code logic
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=r_values,
        theta=categories,
        fill='toself',
        line_color='#8b5cf6',
        fillcolor='rgba(139, 92, 246, 0.5)'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#374151'),
            angularaxis=dict(tickfont=dict(color='#9ca3af', size=10), linecolor='#374151'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=20, b=20),
        height=250
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
    
    # Grid dei fattori sotto il radar
    factors_html = "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; text-align: center; margin-top: 1rem;'>"
    for i, cat in enumerate(categories):
        factors_html += f"""
        <div>
            <div class='text-gray-400' style='font-size: 0.7rem;'>{cat}</div>
            <div class='text-purple-400 font-bold' style='font-size: 0.9rem;'>{r_values[i]}</div>
        </div>
        """
    factors_html += "</div>"
    st.markdown(factors_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 3. ORDER FLOW
    st.markdown("<div class='titan-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='font-bold text-gray-400' style='margin-bottom: 1rem;'>ORDER FLOW ANALYSIS</h3>", unsafe_allow_html=True)
    
    of = data['order_flow']
    total_vol = of['buy_volume'] + of['sell_volume']
    buy_pct = (of['buy_volume'] / total_vol) * 100
    
    # Bar Chart Orizzontale Semplice
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=['Flow'], x=[of['buy_volume']], name='Buy', orientation='h', marker_color='#10b981'
    ))
    fig_bar.add_trace(go.Bar(
        y=['Flow'], x=[of['sell_volume']], name='Sell', orientation='h', marker_color='#ef4444'
    ))
    fig_bar.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown(f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
        <div style='background: rgba(6, 78, 59, 0.2); border: 1px solid #047857; padding: 0.75rem; border-radius: 0.5rem; text-align: center;'>
            <div class='text-emerald-400' style='font-size: 0.7rem;'>Net Buy Pressure</div>
            <div class='text-emerald-400 font-bold' style='font-size: 1.2rem;'>{buy_pct:.0f}%</div>
        </div>
        <div style='background: rgba(88, 28, 135, 0.2); border: 1px solid #7e22ce; padding: 0.75rem; border-radius: 0.5rem; text-align: center;'>
            <div class='text-purple-400' style='font-size: 0.7rem;'>Smart Money</div>
            <div class='text-purple-400 font-bold' style='font-size: 1.2rem;'>{"+" if of['institutional_flow'] > 0 else ""}{of['institutional_flow']:.1f}M</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- COLONNA DESTRA: RISK & INFO ---
with col_right:
    # 1. RISK METRICS GRID
    st.markdown(f"""
    <div class='titan-card'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>
            <h3 class='font-bold text-gray-400'>🛡️ RISK METRICS</h3>
        </div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;'>
            <div style='background: #1f2937; padding: 0.75rem; border-radius: 0.5rem;'>
                <div class='text-gray-400' style='font-size: 0.7rem;'>Volatility</div>
                <div class='text-orange-400 font-bold' style='font-size: 1.1rem;'>{data['volatility']:.2f}%</div>
            </div>
            <div style='background: #1f2937; padding: 0.75rem; border-radius: 0.5rem;'>
                <div class='text-gray-400' style='font-size: 0.7rem;'>Sharpe Ratio</div>
                <div class='text-cyan-400 font-bold' style='font-size: 1.1rem;'>{data['sharpe_ratio']:.2f}</div>
            </div>
            <div style='background: #1f2937; padding: 0.75rem; border-radius: 0.5rem;'>
                <div class='text-gray-400' style='font-size: 0.7rem;'>Max DD</div>
                <div class='text-red-400 font-bold' style='font-size: 1.1rem;'>{data['max_drawdown']:.1f}%</div>
            </div>
            <div style='background: #1f2937; padding: 0.75rem; border-radius: 0.5rem;'>
                <div class='text-gray-400' style='font-size: 0.7rem;'>Win Rate</div>
                <div class='text-emerald-400 font-bold' style='font-size: 1.1rem;'>{data['win_rate']:.0f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. MARKET REGIME
    regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'CALM']
    current_regime = 'RANGING' # Statico come nell'esempio originale
    
    regime_html = ""
    for r in regimes:
        is_active = (r == current_regime)
        border_color = "#3b82f6" if is_active else "#374151"
        bg_color = "rgba(29, 78, 216, 0.3)" if is_active else "#1f2937"
        dot_html = "<div style='width: 8px; height: 8px; background: #60a5fa; border-radius: 50%; box-shadow: 0 0 8px #60a5fa;'></div>" if is_active else ""
        
        regime_html += f"""
        <div style='padding: 0.75rem; border: 1px solid {border_color}; background: {bg_color}; border-radius: 0.5rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;'>
            <span style='font-size: 0.85rem; font-weight: 600;'>{r}</span>
            {dot_html}
        </div>
        """
        
    st.markdown(f"""
    <div class='titan-card'>
        <h3 class='font-bold text-gray-400' style='margin-bottom: 1rem;'>🌊 MARKET REGIME</h3>
        {regime_html}
    </div>
    """, unsafe_allow_html=True)
    
    # 3. SYSTEM STATUS
    systems = [
        {'name': 'Data Feed', 'status': 'ACTIVE', 'color': 'emerald'},
        {'name': 'ML Engine', 'status': 'ACTIVE', 'color': 'emerald'},
        {'name': 'Risk Manager', 'status': 'ACTIVE', 'color': 'emerald'},
        {'name': 'Order Router', 'status': 'STANDBY', 'color': 'yellow'}
    ]
    
    sys_html = ""
    for sys in systems:
        c_code = "#10b981" if sys['color'] == 'emerald' else "#facc15"
        sys_html += f"""
        <div style='display: flex; justify-content: space-between; align-items: center; background: #1f2937; padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
            <span class='text-gray-400' style='font-size: 0.75rem;'>{sys['name']}</span>
            <div style='display: flex; align-items: center; gap: 0.5rem;'>
                <div style='width: 6px; height: 6px; background: {c_code}; border-radius: 50%;'></div>
                <span style='color: {c_code}; font-family: monospace; font-size: 0.75rem;'>{sys['status']}</span>
            </div>
        </div>
        """

    st.markdown(f"""
    <div class='titan-card'>
        <h3 class='font-bold text-gray-400' style='margin-bottom: 1rem;'>SYSTEM STATUS</h3>
        {sys_html}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; color: #4b5563; font-size: 0.75rem; margin-top: 2rem;'>TITAN Oracle v4.0 | Python Streamlit Port</div>", unsafe_allow_html=True)

# Auto-refresh loop (simulazione LIVE)
if st.session_state.is_live:
    time.sleep(2) # Attende 2 secondi
    st.rerun()    # Ricarica la pagina per generare nuovi dati
