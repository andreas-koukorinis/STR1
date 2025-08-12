"""
Credit RV (Pairs & Baskets): Multi-ETF CDX–Cash Strategy
=======================================================

This package implements a **multi-ETF, relative-value** framework to trade
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
  python demo.py   # uses synthetic multi-ETF + IG/HY CDX

Requirements (minimal): numpy, pandas
Recommended extras (CLI/plots/perf): matplotlib, scipy, statsmodels, numba, tqdm
"""

from .demo import run_demo
from .engine import PairEngine
from .portfolio import PortfolioConfig, combine_pairs_kelly
from .specs import ETFETFSpec, ETFCDXSpec

__all__ = [
    "ETFETFSpec",
    "ETFCDXSpec", 
    "PairEngine",
    "PortfolioConfig",
    "combine_pairs_kelly",
    "run_demo",
]

__version__ = "0.2.0" 