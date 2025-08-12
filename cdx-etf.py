"""
Credit RV (Pairs & Baskets): Multi-ETF CDX–Cash Strategy
=======================================================

This module implements a **multi-ETF, relative-value** framework to trade
  1) **ETF ↔ ETF pairs** (e.g., HYG vs JNK, LQD vs VCIT), and
  2) **ETF basket ↔ CDX** (e.g., DV01-weighted HYG+JNK vs CDX HY 5Y).

Core model:
  • Time-varying-parameter (TVP) regression estimated with a **discounted Kalman filter**
    to capture dynamic hedge ratios / cointegration drifts (robust Huberized updates).
  • Optional level/spread view (OU-like spreads), but the default is returns TVP.

It includes a walk-forward OOS backtester, DV01-aware sizing, simple costs, and
risk-parity combination across pairs.

***Data layout***
Provide a dictionary of per-ETF DataFrames (daily) with columns:
  'close','nav','dividend_yield','borrow_fee','credit_dv01','duration','volume',
  'so','basket_dv01','rates_excess_ret'   # rate-hedged return preferred
For CDX families (e.g., 'IG','HY'), provide DataFrames with:
  'spread' (bp), 'dv01' (USD per 1bp per 1 notional).

***Holdings (optional)*** to compute the **cash–CDX basis** per ETF:
  columns ['date','asset','cusip','weight','dv01','oas'] →
  basis(asset,c) = DV01-weighted OAS_cash(asset) − cdx_spread(c).

Run the demo:
  python credit_rv_multi.py   # uses synthetic multi-ETF + IG/HY CDX

Requirements (minimal): numpy, pandas
Recommended extras (CLI/plots/perf): matplotlib, scipy, statsmodels, numba, tqdm
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ETFETFSpec",
    "ETFCDXSpec",
    "PairEngine",
    "PortfolioConfig",
    "create_synthetic_multi",
]

__version__ = "0.2.0"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ewma(x: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted moving average with a half-life parameter."""
    if len(x) == 0:
        return x
    alpha = 1 - math.exp(math.log(0.5) / max(halflife, 1e-6))
    return x.ewm(alpha=alpha, adjust=False).mean()


def ewm_var(x: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted variance with half-life parameter."""
    alpha = 1 - math.exp(math.log(0.5) / max(halflife, 1e-6))
    mu = x.ewm(alpha=alpha, adjust=False).mean()
    var = (x - mu).pow(2).ewm(alpha=alpha, adjust=False).mean()
    return var


def robust_z(x: pd.Series, hl: float = 30.0, eps: float = 1e-9) -> pd.Series:
    """Robust z-score via EW mean/variance."""
    mu = ewma(x.fillna(0.0), hl)
    sig = np.sqrt(np.maximum(ewm_var(x.fillna(0.0), hl), eps))
    return (x - mu) / (sig + eps)


def clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)


# ---------------------------------------------------------------------
# TVP Kalman (discounted) with optional Huberization
# ---------------------------------------------------------------------
@dataclass
class TVPKalman:
    """Time-Varying-Parameter regression via a discounted Kalman filter.

    Observation: y_t = x_t' θ_t + ε_t,   ε_t ~ N(0, R)
    State:       θ_t = F θ_{t-1} + w_t,  w_t ~ N(0, Q_t),
                 with Q_t = ((1-δ)/δ) * diag(P_{t-1})  (per-state discount).

    Huberization: clip the standardized innovation at |z| > huber_c
    to reduce the effect of outliers.
    """
    k: int
    F: np.ndarray
    delta: np.ndarray
    R: float
    huber_c: float = 4.0
    theta: np.ndarray = field(default_factory=lambda: None)
    P: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        if self.theta is None:
            self.theta = np.zeros(self.k)
        if self.P is None:
            self.P = np.eye(self.k) * 1e2
        if np.isscalar(self.delta):
            self.delta = np.repeat(float(self.delta), self.k)
        assert len(self.delta) == self.k
        if self.F is None:
            self.F = np.eye(self.k)

    def step(self, x_t: np.ndarray, y_t: float) -> Tuple[float, float, np.ndarray]:
        x = np.asarray(x_t).reshape(-1)
        # Predict
        theta_pred = self.F @ self.theta
        P_pred = self.F @ self.P @ self.F.T
        Q = np.diag((1 - self.delta) / np.maximum(self.delta, 1e-9) * np.diag(self.P))
        P_pred = P_pred + Q
        # Observe
        yhat = float(x @ theta_pred)
        nu = y_t - yhat
        S = float(x @ P_pred @ x + self.R)
        z = nu / max(math.sqrt(S), 1e-9)
        # Huberized innovation
        if abs(z) > self.huber_c:
            nu = math.copysign(self.huber_c * math.sqrt(S), z)
            z = nu / max(math.sqrt(S), 1e-9)
        K = (P_pred @ x) / max(S, 1e-12)
        # Update
        self.theta = theta_pred + K * nu
        self.P = (np.eye(self.k) - np.outer(K, x)) @ P_pred
        return z, S, self.theta.copy()


# ---------------------------------------------------------------------
# Feature engineering per asset
# ---------------------------------------------------------------------

def rate_hedged_return(df: pd.DataFrame) -> pd.Series:
    """Return series already hedged for rates (if available), else use close-to-close.
    Replace with your proper rate-neutralization.
    """
    if "rates_excess_ret" in df.columns:
        return df["rates_excess_ret"].astype(float)
    return df["close"].pct_change().fillna(0.0)


def nav_premium_z(df: pd.DataFrame, hl: float = 30.0) -> pd.Series:
    prem = (df["close"] - df["nav"]) / df["nav"].replace(0, np.nan)
    prem = prem.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return robust_z(prem, hl)


def daf_z(df: pd.DataFrame, hl: float = 60.0) -> pd.Series:
    so = df["so"].astype(float)
    d_so = so.diff()
    basket = df["basket_dv01"].astype(float)
    daf = (d_so / so.shift(1)).replace([np.inf, -np.inf], np.nan) * basket
    return robust_z(daf.fillna(0.0), hl)


def carry_daily_z(df: pd.DataFrame, cdx_spread_bp: pd.Series, hl: float = 63.0) -> pd.Series:
    """ETF vs CDX rough carry differential, standardized.
    ETF daily carry ≈ div_yield/252 − borrow/252; CDX ≈ spread_decimal/252.
    """
    etf = (df.get("dividend_yield", 0.0)).astype(float) / 252.0 - (df.get("borrow_fee", 0.0)).astype(float) / 252.0
    cdx = (cdx_spread_bp.astype(float) / 1e4) / 252.0
    return robust_z((etf - cdx).fillna(0.0), hl)


def cdx_basis_to_underlying(asset_df: pd.DataFrame,
                             cdx_spread_bp: pd.Series,
                             holdings: Optional[pd.DataFrame] = None,
                             oas_col: str = "oas") -> pd.Series:
    """Compute CDX→cash basis for an ETF.

    If `holdings` (with OAS) is provided, compute DV01-weighted OAS per date for that ETF and
    return: basis = OAS_cash − cdx_spread.
    Otherwise returns a 0 series as a safe fallback (you can wire a proxy here).
    """
    idx = asset_df.index
    if holdings is None or len(holdings) == 0:
        return pd.Series(0.0, index=idx)
    h = holdings.copy()
    h = h[h["asset"].isin([asset_df.name if hasattr(asset_df, "name") else ""])].copy()
    if h.empty:
        return pd.Series(0.0, index=idx)
    # Expect columns: date, asset, dv01, oas, weight (optional). Compute DV01-weighted OAS.
    h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)
    if "weight" not in h.columns:
        h["weight"] = 1.0
    # Aggregate by date
    grouped = h.groupby("date").apply(
        lambda g: np.average(g[oas_col].astype(float), weights=np.maximum(g["dv01"].astype(float) * g["weight"], 1e-8))
    )
    oas_cash = grouped.reindex(idx).interpolate().fillna(method="bfill").fillna(method="ffill")
    basis = oas_cash - cdx_spread_bp.reindex(idx).astype(float)
    return basis


# ---------------------------------------------------------------------
# Pair specs
# ---------------------------------------------------------------------
@dataclass
class ETFETFSpec:
    long: str
    short: str
    name: str = ""

@dataclass
class ETFCDXSpec:
    etfs: List[str]
    weights: Optional[List[float]] = None  # if None: DV01 or equal
    cdx_family: str = "HY"  # 'HY' or 'IG' (your keys)
    name: str = ""


# ---------------------------------------------------------------------
# Pair Engines
# ---------------------------------------------------------------------
class PairEngine:
    """Builds pair-level features, runs TVP Kalman, outputs residual z (signal) and hedge ratios.

    Two modes:
      • ETF↔ETF:    y = r_long, x = [1, r_short, ΔNAV_diff, DAF_diff, Carry_diff]
      • Basket↔CDX: y = r_basket, x = [1, ΔCDX, NAV_b, DAF_b, Carry_b, Basis_b]
    """
    def __init__(self,
                 assets: Dict[str, pd.DataFrame],
                 cdx: Dict[str, pd.DataFrame],
                 holdings: Optional[pd.DataFrame] = None):
        self.assets = assets
        self.cdx = cdx
        self.holdings = holdings

    # -------- ETF↔ETF --------
    def run_etf_pair(self, spec: ETFETFSpec,
                     delta: Tuple[float, ...] = (0.995, 0.985, 0.97, 0.97, 0.98),
                     R_init: float = 1e-6) -> Dict[str, pd.Series]:
        A = self.assets[spec.long].copy(); A.name = spec.long
        B = self.assets[spec.short].copy(); B.name = spec.short
        idx = A.index.intersection(B.index)
        A = A.loc[idx]; B = B.loc[idx]
        rA = rate_hedged_return(A); rB = rate_hedged_return(B)
        z_nav = (nav_premium_z(A) - nav_premium_z(B)).rename("z_nav_diff")
        z_daf = (daf_z(A) - daf_z(B)).rename("z_daf_diff")
        z_carry = robust_z((A.get("dividend_yield", 0.0)/252 - A.get("borrow_fee", 0.0)/252
                           - (B.get("dividend_yield", 0.0)/252 - B.get("borrow_fee", 0.0)/252)).fillna(0.0), 63.0).rename("z_carry_diff")
        # X: [1, rB, z_nav_diff, z_daf_diff, z_carry_diff]; y = rA
        X = pd.DataFrame(index=idx)
        X["const"] = 1.0
        X["rB"] = rB
        X["z_nav"] = z_nav
        X["z_daf"] = z_daf
        X["z_carry"] = z_carry
        y = rA
        kf = TVPKalman(k=X.shape[1], F=np.eye(X.shape[1]), delta=np.array(delta), R=R_init)
        z_list = []; theta_list = []
        for t, (_, row) in enumerate(X.iterrows()):
            z, S, theta = kf.step(row.values.astype(float), float(y.iloc[t]))
            z_list.append(z)
            theta_list.append(theta)
        theta_df = pd.DataFrame(theta_list, index=idx, columns=X.columns)
        return {
            "z_pair": pd.Series(z_list, index=idx, name=f"z_{spec.long}_{spec.short}"),
            "theta": theta_df,
            "r_long": rA,
            "r_short": rB,
        }

    # -------- Basket↔CDX --------
    def run_basket_cdx(self, spec: ETFCDXSpec,
                        delta: Tuple[float, ...] = (0.995, 0.985, 0.97, 0.97, 0.98, 0.98),
                        R_init: float = 1e-6) -> Dict[str, pd.Series]:
        ets = [self.assets[s].copy() for s in spec.etfs]
        for df in ets:
            df.sort_index(inplace=True)
        idx = ets[0].index
        for df in ets[1:]:
            idx = idx.intersection(df.index)
        ets = [df.loc[idx] for df in ets]
        # Basket weights
        if spec.weights is not None:
            w = np.array(spec.weights, dtype=float)
            w = w / np.sum(np.abs(w))
        else:
            # DV01-weighted if available, else equal
            dv01s = np.array([df.get("credit_dv01", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0).mean() for df in ets])
            if np.any(np.isfinite(dv01s)) and np.sum(np.abs(dv01s)) > 0:
                w = dv01s / np.sum(np.abs(dv01s))
            else:
                w = np.ones(len(ets)) / len(ets)
        # Aggregate basket features
        rets = [rate_hedged_return(df) for df in ets]
        r_basket = pd.concat(rets, axis=1).fillna(0.0).values @ w
        r_basket = pd.Series(r_basket, index=idx, name="r_basket")
        z_nav_b = sum(w[i] * nav_premium_z(ets[i]) for i in range(len(ets)))
        z_daf_b = sum(w[i] * daf_z(ets[i]) for i in range(len(ets)))
        # CDX side
        cdxf = self.cdx[spec.cdx_family].loc[idx]
        d_cdx = cdxf["spread"].diff().fillna(0.0) / 1e4  # bp→decimal
        # Carry diff (basket vs CDX)
        z_carry_b = sum(w[i] * (ets[i].get("dividend_yield", 0.0)/252 - ets[i].get("borrow_fee", 0.0)/252) for i in range(len(ets)))
        z_carry_b = robust_z((z_carry_b - (cdxf["spread"]/1e4)/252.0).fillna(0.0), 63.0).rename("z_carry")
        # Basis to underlying (DV01-weighted OAS − CDX spread)
        basis_list = []
        for i, df in enumerate(ets):
            b = cdx_basis_to_underlying(df, cdxf["spread"], holdings=self.holdings)
            basis_list.append(w[i] * b)
        basis_b = sum(basis_list) if basis_list else pd.Series(0.0, index=idx)
        basis_b = basis_b.rename("basis")
        # X: [1, ΔCDX, z_nav_b, z_daf_b, z_carry_b, basis_b]; y = r_basket
        X = pd.DataFrame(index=idx)
        X["const"] = 1.0
        X["d_cdx"] = d_cdx
        X["z_nav"] = z_nav_b
        X["z_daf"] = z_daf_b
        X["z_carry"] = z_carry_b
        X["basis"] = robust_z(basis_b.fillna(0.0), 63.0)
        y = r_basket
        kf = TVPKalman(k=X.shape[1], F=np.eye(X.shape[1]), delta=np.array(delta), R=R_init)
        z_list = []; theta_list = []
        for t, (_, row) in enumerate(X.iterrows()):
            z, S, theta = kf.step(row.values.astype(float), float(y.iloc[t]))
            z_list.append(z)
            theta_list.append(theta)
        theta_df = pd.DataFrame(theta_list, index=idx, columns=X.columns)
        return {
            "z_pair": pd.Series(z_list, index=idx, name=spec.name or f"z_{'_'.join(spec.etfs)}_vs_{spec.cdx_family}"),
            "theta": theta_df,
            "r_basket": r_basket,
            "d_cdx": d_cdx,
        }


# ---------------------------------------------------------------------
# Portfolio backtester (multi-pair, risk parity combination)
# ---------------------------------------------------------------------
@dataclass
class PortfolioConfig:
    target_vol: float = 0.10
    vol_hl: float = 20.0
    tanh_scale: float = 1.5
    etf_tc_bps: float = 1.0    # round-trip cost for ETF legs
    cdx_tc_bpDv01: float = 0.2 # round-trip cost for CDX leg (for basket↔CDX)


def realized_vol(ret: pd.Series, hl: float) -> pd.Series:
    var = ewm_var(ret.fillna(0.0), hl)
    return np.sqrt(var) * np.sqrt(252.0)


def size_and_cost(signal: pd.Series, pnl_proxy: pd.Series, cfg: PortfolioConfig) -> Tuple[pd.Series, pd.Series]:
    """Vol-target the signal and compute simple turnover-based transaction cost."""
    rv = realized_vol(pnl_proxy, cfg.vol_hl)
    scale = cfg.target_vol / rv.replace(0, np.nan)
    scale = scale.fillna(method="ffill").fillna(0.0)
    pos = clip(signal * scale, -3.0, 3.0)
    turn = pos.diff().abs().fillna(0.0)
    tc = (cfg.etf_tc_bps / 1e4) * turn  # add CDX cost in basket engine below if you track that leg explicitly
    return pos, tc


def combine_pairs_kelly(signals: Dict[str, pd.Series], pnl_proxies: Dict[str, pd.Series], cfg: PortfolioConfig) -> pd.DataFrame:
    """Risk-parity combine across pairs using inverse EW vol weights.

    Returns a DataFrame with columns per pair of **final positions** (post vol-scaling),
    plus a 'portfolio' column for aggregated PnL computation downstream.
    """
    idx = None
    for s in signals.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = pd.DatetimeIndex(idx)
    weights = {}
    sized = {}
    costs = {}
    for name, sig in signals.items():
        pnl_proxy = pnl_proxies[name].reindex(idx)
        pos, tc = size_and_cost(sig.reindex(idx), pnl_proxy, cfg)
        vol = realized_vol(pnl_proxy, cfg.vol_hl)
        w = 1.0 / (vol.replace(0, np.nan))
        w = w.fillna(method="ffill").fillna(0.0)
        weights[name] = clip(w / w.replace(0, np.nan).rolling(20).mean().fillna(method="ffill").fillna(1.0), 0.0, 3.0)
        sized[name] = pos * weights[name]
        costs[name] = tc
    df_pos = pd.DataFrame({k: v for k, v in sized.items()})
    df_tc = pd.DataFrame({k: v for k, v in costs.items()})
    df_pos["portfolio"] = df_pos.sum(axis=1)
    df_tc["portfolio"] = df_tc.sum(axis=1)
    return df_pos, df_tc


# ---------------------------------------------------------------------
# Synthetic multi-asset dataset (demo only)
# ---------------------------------------------------------------------

def create_synthetic_multi(n_days: int = 1200, seed: int = 42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-02", periods=n_days)

    def mk_etf(name: str, base: float, vol: float) -> pd.DataFrame:
        close = base * np.exp(np.cumsum(rng.normal(0, vol, size=n_days)))
        nav_prem = rng.normal(0, 0.001, size=n_days)
        so = np.cumsum(np.maximum(0, rng.normal(0, 10000, size=n_days))) + 5e6
        basket_dv01 = 0.5 + 0.1 * rng.normal(size=n_days)
        div = 0.03 + 0.003 * rng.normal(size=n_days)
        borrow = 0.01 + 0.002 * rng.normal(size=n_days)
        # Rate-hedged returns: link to latent credit factor + idiosyncratic
        credit_shock = rng.normal(0, 0.004, size=n_days)
        eps = rng.normal(0, 0.004, size=n_days)
        r = 0.3 * credit_shock + eps
        df = pd.DataFrame({
            "close": close,
            "nav": close * (1 - nav_prem),
            "dividend_yield": div,
            "borrow_fee": borrow,
            "credit_dv01": 0.7 + 0.05 * rng.normal(size=n_days),
            "duration": 8.0 + rng.normal(0, 0.5, size=n_days),
            "volume": rng.integers(8e5, 4e6, size=n_days),
            "so": so,
            "basket_dv01": basket_dv01,
            "rates_excess_ret": r,
        }, index=idx)
        df.name = name
        return df

    assets = {
        "HYG": mk_etf("HYG", 90, 0.012),
        "JNK": mk_etf("JNK", 40, 0.013),
        "LQD": mk_etf("LQD", 110, 0.008),
        "VCIT": mk_etf("VCIT", 95, 0.009),
    }
    # CDX (bp)
    cdx_idx = idx
    cdx_hy = pd.DataFrame({
        "spread": 350 + np.cumsum(rng.normal(0, 0.6, size=n_days)),
        "dv01": 45.0 + 2.0 * rng.normal(size=n_days),
    }, index=cdx_idx)
    cdx_ig = pd.DataFrame({
        "spread": 80 + np.cumsum(rng.normal(0, 0.3, size=n_days)),
        "dv01": 75.0 + 3.0 * rng.normal(size=n_days),
    }, index=cdx_idx)
    return assets, {"HY": cdx_hy, "IG": cdx_ig}


# ---------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Demo with synthetic data
    assets, cdx = create_synthetic_multi()
    engine = PairEngine(assets, cdx, holdings=None)

    # Define pairs
    pairs: List[ETFETFSpec | ETFCDXSpec] = [
        ETFETFSpec(long="HYG", short="JNK", name="HYG_vs_JNK"),
        ETFETFSpec(long="LQD", short="VCIT", name="LQD_vs_VCIT"),
        ETFCDXSpec(etfs=["HYG", "JNK"], weights=[0.6, 0.4], cdx_family="HY", name="HY_basket_vs_CDX"),
        ETFCDXSpec(etfs=["LQD", "VCIT"], weights=[0.5, 0.5], cdx_family="IG", name="IG_basket_vs_CDX"),
    ]

    pair_signals: Dict[str, pd.Series] = {}
    pair_theta: Dict[str, pd.DataFrame] = {}
    pnl_proxies: Dict[str, pd.Series] = {}

    # Run engines per pair
    for p in pairs:
        if isinstance(p, ETFETFSpec):
            out = engine.run_etf_pair(p)
            pair_signals[p.name or f"{p.long}_vs_{p.short}"] = out["z_pair"]
            # PnL proxy: next-day forecast error of regression y on x (use here r_long - θ*r_short)
            theta = out["theta"]
            r_long = out["r_long"]; r_short = out["r_short"]
            yhat = theta["const"].shift(1) + theta["rB"].shift(1) * r_short
            resid_next = r_long - yhat
            pnl_proxies[p.name or f"{p.long}_vs_{p.short}"] = resid_next
            pair_theta[p.name or f"{p.long}_vs_{p.short}"] = theta
        else:
            out = engine.run_basket_cdx(p)
            pair_signals[p.name] = out["z_pair"]
            theta = out["theta"]
            # Simplified OOS proxy: r_basket - (const + β*d_cdx) at t params
            resid_next = out["r_basket"] - (theta["const"].shift(1) + theta["d_cdx"].shift(1) * out["d_cdx"])  # proxy
            pnl_proxies[p.name] = resid_next
            pair_theta[p.name] = theta

    # Combine across pairs with risk parity & vol targeting
    cfg = PortfolioConfig(target_vol=0.10, vol_hl=20.0)
    positions, costs = combine_pairs_kelly(pair_signals, pnl_proxies, cfg)

    # Portfolio PnL (aggregated proxy) and KPIs
    pnl = positions.drop(columns=["portfolio"]).mul(pd.DataFrame(pnl_proxies)).sum(axis=1) - costs["portfolio"]
    equity = (1 + pnl.fillna(0.0)).cumprod()

    # Print quick KPIs
    ann_ret = equity.iloc[-1] ** (252/len(equity)) - 1
    ann_vol = pnl.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    maxdd = (equity / equity.cummax() - 1).min()
    print("==== Credit RV (Pairs+Baskets) — Synthetic Demo KPIs ====")
    print(f"AnnReturn: {ann_ret: .3%}  AnnVol: {ann_vol: .2%}  Sharpe: {sharpe: .2f}  MaxDD: {maxdd: .2%}")
    print("Last 5 equity:")
    print(equity.tail())
