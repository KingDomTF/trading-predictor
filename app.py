#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App.py — Streamlit Scenario & Analog Trading Dashboard
======================================================

Run:
  streamlit run App.py

Features
--------
- Interactive inputs for scenario (DXY %, TNX bps, VIX pts)
- Date range, horizon, k-neighbors, alpha, risk aversion
- Data fetch via yfinance (cached)
- Analog search & Ridge predictive model
- Blended view + long-only mean-variance weights
- Tables and matplotlib charts rendered in Streamlit

Notes
-----
- If DXY ticker "DX-Y.NYB" is unreliable in your region, toggle "Use UUP proxy".
- ^TNX is handled (tenths of percent -> %).

"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Data & modeling deps
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# ----------------------------- Configuration ---------------------------------

DEFAULT_ASSET_TICKERS = {
    "Gold": "XAUUSD=X",
    "Silver": "SI=F",
    "BTC": "BTC-USD",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",   # USD/JPY on Yahoo
}

DEFAULT_FACTOR_TICKERS = {
    "DXY": "DX-Y.NYB",   # consider UUP proxy if needed
    "TNX": "^TNX",       # 10y yield (tenths of a percent)
    "VIX": "^VIX",
}

# ----------------------------- Utilities -------------------------------------

def pct_change_over_horizon(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-horizon) / series - 1.0

def level_change_over_horizon(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-horizon) - series

def to_bps(x: pd.Series) -> pd.Series:
    return x * 100.0

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def mean_variance_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    n = len(mu)
    lam = 1e-3
    A = cov + lam * np.eye(n)
    try:
        raw = np.linalg.solve(A, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(A) @ mu
    raw = raw / max(risk_aversion, 1e-6)
    w = project_to_simplex(raw)
    return w

def summarize_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mean": df.mean(),
        "median": df.median(),
        "p10": df.quantile(0.10),
        "p90": df.quantile(0.90),
        "count": df.count(),
    })

# ----------------------------- Data Layer ------------------------------------

@st.cache_data(show_spinner=True, ttl=60*30)
def fetch_prices(tickers: Dict[str, str], start: str, end: Optional[str]) -> pd.DataFrame:
    data = {}
    for name, t in tickers.items():
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or "Close" not in df:
            continue
        data[name] = df["Close"].rename(name)
    out = pd.DataFrame(data).dropna(how="all")
    if out.empty:
        raise ValueError("Nessun dato scaricato. Controlla tickers e date.")
    return out

def prepare_factor_frame(factors: pd.DataFrame) -> pd.DataFrame:
    fac = factors.copy()
    if "TNX" in fac.columns:
        fac["TNX"] = fac["TNX"] / 10.0  # tenths of percent -> percent
    return fac

# ----------------------------- Scenario Model --------------------------------

@dataclass
class Scenario:
    dxy_shock: float      # % change over horizon (e.g., -0.05)
    tnx_shock_bps: float  # change in bps (e.g., -50)
    vix_shock: float      # absolute change (e.g., +10)

    def as_vector(self) -> np.ndarray:
        return np.array([self.dxy_shock, self.tnx_shock_bps, self.vix_shock], dtype=float)

class AnalogPredictor:
    def __init__(self, horizon: int, kneighbors: int, ridge_l2: float, seed: int = 42):
        self.horizon = horizon
        self.kneighbors = kneighbors
        self.ridge_l2 = ridge_l2
        self.seed = seed
        self._fitted = False

    def _build_design(self, factor_df: pd.DataFrame, asset_df: pd.DataFrame):
        dxy = factor_df["DXY"]
        tnx = factor_df["TNX"]
        vix = factor_df["VIX"]

        X = pd.DataFrame(index=factor_df.index)
        X["DXY_dh"] = pct_change_over_horizon(dxy, self.horizon)                # %
        X["TNX_dh_bps"] = to_bps(level_change_over_horizon(tnx, self.horizon))  # bps
        X["VIX_dh"] = level_change_over_horizon(vix, self.horizon)              # abs

        Y = pd.DataFrame(index=asset_df.index)
        for col in asset_df.columns:
            Y[f"{col}_fwd{self.horizon}d"] = pct_change_over_horizon(asset_df[col], self.horizon)

        XY = X.join(Y, how="inner").dropna()
        return XY[X.columns], XY[Y.columns]

    def fit(self, factor_df: pd.DataFrame, asset_df: pd.DataFrame):
        self.X, self.Y = self._build_design(factor_df, asset_df)
        self.scaler = StandardScaler()
        self.Xs = pd.DataFrame(self.scaler.fit_transform(self.X.values), index=self.X.index, columns=self.X.columns)

        self.models = {}
        for ycol in self.Y.columns:
            ridge = Ridge(alpha=self.ridge_l2, random_state=self.seed)
            ridge.fit(self.Xs.values, self.Y[ycol].values)
            self.models[ycol] = ridge

        self._fitted = True

    def find_analogs(self, scenario: Scenario) -> pd.DataFrame:
        s_std = self.scaler.transform(scenario.as_vector().reshape(1, -1))
        diffs = self.Xs.values - s_std
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        order = np.argsort(dists)[: self.kneighbors]
        df = pd.DataFrame({"date": self.Xs.index[order], "distance": dists[order]}).set_index("date").sort_values("distance")
        return df

    def analog_forward_returns(self, analogs: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.Y.columns)
        out = self.Y.loc[analogs.index, cols].copy()
        out.columns = [c.replace(f"_fwd{self.horizon}d", "") for c in out.columns]
        return out

    def model_prediction(self, scenario: Scenario) -> pd.Series:
        s_std = self.scaler.transform(scenario.as_vector().reshape(1, -1))
        preds = {}
        for ycol, model in self.models.items():
            preds[ycol.replace(f"_fwd{self.horizon}d", "")] = float(model.predict(s_std)[0])
        return pd.Series(preds).sort_index()

# ----------------------------- Streamlit UI ----------------------------------

st.set_page_config(page_title="Scenario & Analog Trading", layout="wide")

st.title("📊 Scenario & Analog Trading — Metals, FX, BTC")
st.caption("Confronta uno scenario predittivo con analoghi storici e genera una proposta di pesi di portafoglio (long-only).")

with st.sidebar:
    st.header("⚙️ Parametri")
    start = st.date_input("Start", value=pd.to_datetime("2010-01-01")).strftime("%Y-%m-%d")
    end = st.date_input("End", value=pd.to_datetime("today")).strftime("%Y-%m-%d")

    horizon = st.number_input("Horizon (giorni di borsa)", min_value=5, max_value=120, value=20, step=5)
    kneighbors = st.number_input("K (analoghi)", min_value=10, max_value=500, value=50, step=5)
    alpha = st.slider("Blend α (modello vs analoghi)", 0.0, 1.0, 0.5, 0.05)
    ridge_l2 = st.number_input("Ridge L2", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
    risk_aversion = st.number_input("Avversione al rischio (MV)", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

    st.divider()
    st.subheader("🎯 Scenario (variazioni su H)")
    dxy_shock_pct = st.number_input("DXY (%): es. -5 = -5%", value=-5.0, step=0.5) / 100.0
    tnx_shock_bps = st.number_input("TNX (bps): es. -50 = -0.50%", value=-50.0, step=5.0)
    vix_shock = st.number_input("VIX (punti): es. +8", value=8.0, step=1.0)

    st.divider()
    use_uup = st.toggle("Usa UUP come proxy del DXY (se DX-Y.NYB non scarica)")

    run_btn = st.button("Esegui analisi", type="primary")

factor_tickers = DEFAULT_FACTOR_TICKERS.copy()
if use_uup:
    factor_tickers["DXY"] = "UUP"

if run_btn:
    try:
        assets = fetch_prices(DEFAULT_ASSET_TICKERS, start, end)
        factors_raw = fetch_prices(factor_tickers, start, end)
        factors = prepare_factor_frame(factors_raw)

        df_all = assets.join(factors, how="inner").dropna()
        if df_all.empty:
            st.error("Serie vuota dopo l'allineamento date. Prova a variare l'intervallo.")
            st.stop()
        assets_al = df_all[list(DEFAULT_ASSET_TICKERS.keys())]
        factors_al = df_all[list(factor_tickers.keys())]

        predictor = AnalogPredictor(horizon=horizon, kneighbors=kneighbors, ridge_l2=ridge_l2, seed=seed)
        predictor.fit(factors_al, assets_al)

        scn = Scenario(dxy_shock=dxy_shock_pct, tnx_shock_bps=tnx_shock_bps, vix_shock=vix_shock)

        analogs = predictor.find_analogs(scn)
        analog_rets = predictor.analog_forward_returns(analogs)
        model_pred = predictor.model_prediction(scn)

        analog_mean = analog_rets.mean().reindex(model_pred.index)
        blended = alpha * model_pred + (1 - alpha) * analog_mean

        cov = analog_rets.cov().loc[blended.index, blended.index].values
        mu = blended.values
        w = mean_variance_weights(mu, cov, risk_aversion=risk_aversion)
        weights = pd.Series(w, index=blended.index, name="weight")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Analoghi selezionati (K)", value=int(kneighbors))
            st.json(asdict(scn), expanded=False)
        with c2:
            st.subheader("📈 Predizioni modello (H)")
            st.dataframe(model_pred.to_frame("pred").style.format("{:.2%}"), use_container_width=True)
        with c3:
            st.subheader("🔀 Blend (α)")
            st.dataframe(blended.to_frame("blended").style.format("{:.2%}"), use_container_width=True)

        st.subheader("📅 Top analoghi (più simili)")
        st.dataframe(analogs.head(30), use_container_width=True)

        st.subheader("🎯 Distribuzione rendimenti futuri (analoghi)")
        st.dataframe(summarize_distribution(analog_rets).style.format("{:.2%}"), use_container_width=True)

        fig1, ax1 = plt.subplots()
        analog_rets.boxplot(ax=ax1)
        ax1.set_title("Analog Forward Returns Distribution")
        ax1.set_ylabel("Return over Horizon")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        blended.sort_values().plot(kind="barh", ax=ax2)
        ax2.set_title("Blended Predicted Forward Returns")
        ax2.set_xlabel("Return over Horizon")
        st.pyplot(fig2)

        st.subheader("📦 Pesi suggeriti (long-only, somma=1)")
        st.dataframe(weights.to_frame().style.format("{:.2%}"), use_container_width=True)

        report = []
        report.append("# Scenario")
        report.append(json.dumps(asdict(scn), ensure_ascii=False))
        report.append("\n# Analog Dates (top-k)")
        report.append(analogs.to_csv())
        report.append("\n# Analog Forward Return Summary")
        report.append(summarize_distribution(analog_rets).to_csv())
        report.append("\n# Model Prediction (forward returns)")
        report.append(model_pred.to_csv(header=False))
        report.append("\n# Blended Prediction")
        report.append(blended.to_csv(header=False))
        report.append("\n# Suggested Weights")
        report.append(weights.to_csv(header=False))
        rep_str = "\n".join(report)

        st.download_button("Scarica report .csv", data=rep_str.encode("utf-8"), file_name="scenario_report.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Errore: {e}")
        st.exception(e)
else:
    st.info("Imposta i parametri nella sidebar e premi **Esegui analisi**.")
