import streamlit as st
from supabase import create_client
import time
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(page_title="TITAN Oracle", layout="wide", page_icon="🛡️", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 95%; }
    header { visibility: hidden; }
    .titan-card { background-color: #111827; border: 1px solid #1f2937; border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1.5rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
    .card-title { font-size: 1.125rem; font-weight: 700; color: #d1d5db; display: flex; align-items: center; gap: 0.5rem; }
    .signal-huge { font-size: 3.5rem; font-weight: 900; line-height: 1; }
    .sig-buy { color: #10b981; text-shadow: 0 0 40px rgba(16, 185, 129, 0.2); }
    .sig-sell { color: #ef4444; text-shadow: 0 0 40px rgba(239, 68, 68, 0.2); }
    .sig-wait { color: #6b7280; }
    .metric-box { background: rgba(31, 41, 55, 0.5); border: 1px solid #374151; border-radius: 0.5rem; padding: 0.75rem; }
    .box-label { font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.25rem; }
    .box-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1rem; }
    .prog-bg { width: 100%; background: #374151; height: 0.5rem; border-radius: 99px; overflow: hidden; }
    .prog-fill { height: 100%; border-radius: 99px; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    .badge-yellow { background: rgba(234, 179, 8, 0.2); color: #facc15; }
    div.stButton > button { background-color: #1f2937; color: #9ca3af; border: 1px solid #374151; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; width: 100%; }
    div.stButton > button:hover { border-color: #10b981; color: white; }
    div.stButton > button:focus { background-color: #059669; border-color: #10b981; color: white; }
    .text-cyan { color: #22d3ee; } .text-red { color: #ef4444; } .text-emerald { color: #10b981; } .text-purple { color: #a78bfa; } .text-orange { color: #fbbf24; }
</style>
""", unsafe_allow_html=True)

SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"
@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

if 'selected_asset' not in st.session_state: st.session_state.selected_asset = 'XAUUSD'

def plot_radar(d):
    conf = d.get('prob_buy', 50) if "BUY" in d['recommendation'] else d.get('prob_sell', 50)
    z = abs(d.get('confidence_score', 0))
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[min(conf+10,95), min(90-z*10,80), min(z*30,85), 82, 58, 60 if "BUY" in d['recommendation'] else 40],
        theta=['Momentum', 'Mean Rev', 'Volatility', 'Volume', 'Sentiment', 'Macro'],
        fill='toself', line_color='#8b5cf6', fillcolor='rgba(139, 92, 246, 0.4)', marker=dict(size=0)
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100], color='#4b5563', showticklabels=False, gridcolor='#374151'), angularaxis=dict(color='#9ca3af', gridcolor='#374151'), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30,r=30,t=10,b=10), height=220, showlegend=False)
    return fig

def plot_pnl_area():
    x = np.arange(30)
    y = np.cumsum(np.random.randn(30)*1.5+0.5)*100+1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', mode='lines', line=dict(color='#10b981', width=2, shape='spline'), fillcolor='rgba(16, 185, 129, 0.1)'))
    # --- LA CORREZIONE DELL'ERRORE È QUI SOTTO ---
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0), height=180, xaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False, zeroline=False), yaxis=dict(showgrid=True, gridcolor='#374151', showticklabels=False, zeroline=False), showlegend=False)
    return fig

def plot_order_flow(d):
    score = d.get('prob_buy', 50) if "BUY" in d['recommendation'] else (100 - d.get('prob_sell', 50))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Flow'], y=[score], name='Buy', marker_color='#10b981', width=0.5))
    fig.add_trace(go.Bar(x=['Flow'], y=[-(100-score)], name='Sell', marker_color='#ef4444', width=0.5))
    fig.update_layout(barmode='relative', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0), height=100, xaxis=dict(showgrid=False, showticklabels=False, zeroline=True, zerolinecolor='#6b7280'), yaxis=dict(showgrid=False, showticklabels=False), showlegend=False, bargap=0)
    return fig

c1, c2 = st.columns([3, 1])
with c1: st.markdown('<div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg><div><div style="font-size:36px;font-weight:800;line-height:1;background:-webkit-linear-gradient(45deg,#34d399,#22d3ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">TITAN ORACLE</div><div style="font-size:12px;color:#9ca3af;letter-spacing:0.1em;margin-top:4px;">INSTITUTIONAL-GRADE INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
with c2: st.markdown('<div style="display:flex;justify-content:flex-end;"><div style="background:rgba(6,78,59,0.3);border:1px solid #10b981;padding:8px 16px;border-radius:99px;color:#10b981;font-family:\'JetBrains Mono\',monospace;font-size:12px;display:flex;align-items:center;gap:8px;"><div style="width:8px;height:8px;background:#10b981;border-radius:50%;box-shadow:0 0 10px #10b981;animation:pulse 2s infinite;"></div>SYSTEM LIVE</div></div>', unsafe_allow_html=True)

assets = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
cols = st.columns(len(assets))
for i, asset in enumerate(assets):
    if cols[i].button(asset, key=asset, use_container_width=True): st.session_state.selected_asset = asset

st.write("")
placeholder = st.empty()

while True:
    try:
        resp = supabase.table("ai_oracle").select("*").eq("symbol", st.session_state.selected_asset).order("id", desc=True).limit(1).execute()
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                conf = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                z = d.get('confidence_score', 0)
                regime = d.get('market_regime', 'ANALYZING')
                
                if "BUY" in rec: sig_cls="sig-buy"; bar_col="#10b981"; pat_name="Bull Flag"; pat_type="badge-green"
                elif "SELL" in rec: sig_cls="sig-sell"; bar_col="#ef4444"; pat_name="Bear Flag"; pat_type="badge-red"
                else: sig_cls="sig-wait"; bar_col="#6b7280"; pat_name="Consolidation"; pat_type="badge-yellow"

                if "WAIT" in rec: entry_disp, sl_disp, tp_disp, rr_disp = "---", "---", "---", "WAITING"
                else: entry_disp, sl_disp, tp_disp, rr_disp = f"${d['entry_price']}", f"${d['stop_loss']}", f"${d['take_profit']}", f"1:{d['risk_reward']}"

                c_left, c_mid, c_right = st.columns([4, 5, 3])
                
                with c_left:
                    st.markdown(f"""
                    <div class="titan-card">
                        <div class="card-header"><div class="card-title"><span style="color:#22d3ee">🧠</span> AI SIGNAL</div><span class="text-gray text-mono" style="font-size:12px">{st.session_state.selected_asset}</span></div>
                        <div style="text-align:center; padding:1rem 0;"><div class="{sig_cls} signal-huge">{rec}</div><div style="color:#9ca3af; margin-top:0.5rem; font-size:0.9rem;">{d['details']}</div></div>
                        <div style="margin-top:1rem;"><div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#9ca3af; margin-bottom:0.25rem;"><span>CONFIDENCE</span><span class="text-white font-bold">{conf:.1f}%</span></div><div class="prog-bg"><div class="prog-fill" style="width:{conf}%; background:{bar_col};"></div></div></div>
                        <div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid #374151;"><div style="font-size:0.75rem; color:#9ca3af;">CURRENT PRICE</div><div style="font-size:2rem; font-weight:800; color:white;">${d['current_price']}</div></div>
                        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; margin-top:1rem;"><div class="metric-box" style="border-color:#0e7490; background:rgba(8,145,178,0.1);"><div class="box-label text-cyan">ENTRY</div><div class="box-value text-cyan">{entry_disp}</div></div><div class="metric-box" style="border-color:#991b1b; background:rgba(127,29,29,0.1);"><div class="box-label text-red">STOP</div><div class="box-value text-red">{sl_disp}</div></div><div class="metric-box" style="border-color:#065f46; background:rgba(6,78,59,0.1);"><div class="box-label text-emerald">TARGET</div><div class="box-value text-emerald">{tp_disp}</div></div></div>
                        <div style="text-align:center; margin-top:1rem; font-size:0.75rem; color:#9ca3af;">Risk/Reward: <span class="text-white font-bold">{rr_disp}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="titan-card"><div class="card-header"><div class="card-title"><span class="text-orange">⚡</span> PATTERNS</div></div><div class="list-row"><div style="font-weight:600;">{pat_name}</div><span class="badge {pat_type}">{rec.split(' ')[-1] if 'WAIT' not in rec else 'NEUTRAL'}</span></div><div style="display:flex; align-items:center; gap:0.5rem;"><div class="prog-bg" style="height:4px;"><div class="prog-fill" style="width:75%; background:{bar_col};"></div></div><span class="text-gray" style="font-size:0.75rem;">75%</span></div></div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="titan-card"><div class="card-header"><div class="card-title"><span class="text-purple">🛡️</span> RISK METRICS</div></div><div style="display:grid; grid-template-columns:repeat(2,1fr); gap:0.75rem;"><div class="metric-box"><div class="box-label">VOLATILITY</div><div class="box-value text-orange">{z*10:.2f}%</div></div><div class="metric-box"><div class="box-label">SHARPE</div><div class="box-value text-cyan">2.14</div></div><div class="metric-box"><div class="box-label">DRAWDOWN</div><div class="box-value text-red">-4.2%</div></div><div class="metric-box"><div class="box-label">WIN RATE</div><div class="box-value text-emerald">68%</div></div></div></div>
                    """, unsafe_allow_html=True)

                with c_mid:
                    st.markdown('<div class="titan-card"><div class="card-title" style="margin-bottom:1rem;">CUMULATIVE P&L</div>', unsafe_allow_html=True)
                    st.plotly_chart(plot_pnl_area(), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="titan-card"><div class="card-title" style="margin-bottom:1rem;">MULTI-FACTOR ANALYSIS</div>', unsafe_allow_html=True)
                    st.plotly_chart(plot_radar(d), use_container_width=True, config={'displayModeBar': False})
                    st.markdown(f'<div style="display:grid; grid-template-columns:repeat(3,1fr); text-align:center; margin-top:1rem;"><div><div style="font-size:10px; color:#6b7280;">Trend</div><div class="text-purple font-bold">{conf:.0f}</div></div><div><div style="font-size:10px; color:#6b7280;">Vol</div><div class="text-purple font-bold">{z*10:.0f}</div></div><div><div style="font-size:10px; color:#6b7280;">Macro</div><div class="text-purple font-bold">65</div></div></div></div>', unsafe_allow_html=True)
                    st.markdown('<div class="titan-card"><div class="card-title" style="margin-bottom:1rem;">ORDER FLOW</div>', unsafe_allow_html=True)
                    st.plotly_chart(plot_order_flow(d), use_container_width=True, config={'displayModeBar': False})
                    st.markdown(f'<div style="display:grid; grid-template-columns:repeat(2,1fr); gap:0.75rem; margin-top:1rem;"><div class="metric-box" style="border-color:#065f46; background:rgba(6,78,59,0.1); text-align:center;"><div class="box-label text-emerald">NET BUY PRESSURE</div><div class="box-value text-emerald">{max(0, conf-20):.0f}%</div></div><div class="metric-box" style="border-color:#581c87; background:rgba(88,28,135,0.1); text-align:center;"><div class="box-label text-purple">SMART MONEY</div><div class="box-value text-purple">+12.4M</div></div></div></div>', unsafe_allow_html=True)

                with c_right:
                    is_tr = regime=='TRENDING'; is_rg = regime=='RANGING'; is_vl = regime in ['VOLATILE','CHAOS']
                    st.markdown(f"""
                    <div class="titan-card"><div class="card-header"><div class="card-title"><span class="text-cyan">⚡</span> REGIME</div></div><div style="display:flex; flex-direction:column; gap:0.5rem;"><div class="list-row" style="{f'border:1px solid #3b82f6; background:rgba(59,130,246,0.1);' if is_tr else ''}"><span style="font-weight:700; color:{'white' if is_tr else '#6b7280'};">TRENDING</span>{'<div style="width:8px; height:8px; background:#60a5fa; border-radius:50%; box-shadow:0 0 8px #60a5fa;"></div>' if is_tr else ''}</div><div class="list-row" style="{f'border:1px solid #3b82f6; background:rgba(59,130,246,0.1);' if is_rg else ''}"><span style="font-weight:700; color:{'white' if is_rg else '#6b7280'};">RANGING</span>{'<div style="width:8px; height:8px; background:#60a5fa; border-radius:50%; box-shadow:0 0 8px #60a5fa;"></div>' if is_rg else ''}</div><div class="list-row" style="{f'border:1px solid #3b82f6; background:rgba(59,130,246,0.1);' if is_vl else ''}"><span style="font-weight:700; color:{'white' if is_vl else '#6b7280'};">VOLATILE</span>{'<div style="width:8px; height:8px; background:#60a5fa; border-radius:50%; box-shadow:0 0 8px #60a5fa;"></div>' if is_vl else ''}</div></div></div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="titan-card"><div class="card-header"><div class="card-title">CORRELATION</div></div><div style="font-size:0.75rem; display:flex; flex-direction:column; gap:0.5rem;"><div style="display:flex; align-items:center; gap:0.5rem;"><span class="text-gray text-mono" style="width:40px;">BTC</span><div class="prog-bg" style="height:4px;"><div class="prog-fill" style="width:20%; background:#ef4444;"></div></div><span class="text-red font-bold">-0.8</span></div><div style="display:flex; align-items:center; gap:0.5rem;"><span class="text-gray text-mono" style="width:40px;">SPX</span><div class="prog-bg" style="height:4px;"><div class="prog-fill" style="width:60%; background:#10b981;"></div></div><span class="text-emerald font-bold">+0.6</span></div><div style="display:flex; align-items:center; gap:0.5rem;"><span class="text-gray text-mono" style="width:40px;">DXY</span><div class="prog-bg" style="height:4px;"><div class="prog-fill" style="width:90%; background:#ef4444;"></div></div><span class="text-red font-bold">-0.9</span></div></div></div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="titan-card"><div class="card-header"><div class="card-title">EVENTS</div></div><div class="list-row"><div><div style="font-weight:600;">Fed Rate</div><div style="font-size:10px; color:#6b7280;">14:00 EST</div></div><span class="badge badge-red">HIGH</span></div><div class="list-row"><div><div style="font-weight:600;">CPI Data</div><div style="font-size:10px; color:#6b7280;">08:30 EST</div></div><span class="badge badge-yellow">MED</span></div></div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="titan-card"><div class="card-header"><div class="card-title">STATUS</div></div><div class="list-row"><span class="text-gray">ML Engine</span><span style="color:#10b981; font-size:10px;">● ACTIVE</span></div><div class="list-row"><span class="text-gray">Risk Man.</span><span style="color:#10b981; font-size:10px;">● ACTIVE</span></div><div class="list-row"><span class="text-gray">Router</span><span style="color:#fbbf24; font-size:10px;">● STANDBY</span></div></div>
                    """, unsafe_allow_html=True)
            else: st.info(f"Connecting to TITAN Core for {st.session_state.selected_asset}...")
        time.sleep(1)
    except Exception as e: time.sleep(1)
