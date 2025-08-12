"""
Demo script for the credit RV framework.

This module demonstrates how to use the credit RV framework with synthetic data,
running both ETF-ETF pairs and basket-CDX pairs, and computing performance metrics.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from data import create_synthetic_multi
from engine import PairEngine
from portfolio import PortfolioConfig, combine_pairs_kelly
from specs import ETFETFSpec, ETFCDXSpec
from metrics import calculate_all_metrics, print_metrics_summary, calculate_and_print_advanced_drawdown_analysis


def run_demo() -> Dict[str, pd.Series | pd.DataFrame]:
    """Run the complete credit RV framework demonstration.
    
    This function demonstrates the full credit RV framework workflow:
    1. Generate synthetic data for ETFs and CDX indices
    2. Define trading pairs (ETF-ETF and basket-CDX)
    3. Run pair analysis using time-varying parameter regression
    4. Combine signals using risk-parity approach
    5. Calculate comprehensive performance metrics and KPIs
    
    The demo includes:
    - 2 ETF-ETF pairs: HYG vs JNK, LQD vs VCIT
    - 2 basket-CDX pairs: HY basket vs HY CDX, IG basket vs IG CDX
    
    Returns:
        Dictionary containing all results:
            - 'equity': Cumulative equity curve
            - 'pnl': Daily PnL series
            - 'positions': Position sizes for each strategy
            - 'costs': Transaction costs for each strategy
            - 'signals': Raw trading signals for each pair
            - 'theta': Estimated parameters for each pair
            - 'metrics': Comprehensive performance metrics
            - 'decompositions': Return decomposition components (for basket pairs)
            
    Example:
        >>> results = run_demo()
        >>> print(f"Final equity: {results['equity'].iloc[-1]:.2f}")
        >>> print(f"Sharpe ratio: {results['metrics'].sharpe_ratio:.2f}")
    """
    # Create synthetic data
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
    decompositions: Dict[str, pd.DataFrame] = {}

    # Run engines per pair
    for p in pairs:
        if isinstance(p, ETFETFSpec):
            out = engine.run_etf_pair(p)
            pair_signals[p.name or f"{p.long}_vs_{p.short}"] = out["z_pair"]
            # PnL proxy: next-day pair return
            pnl_proxies[p.name or f"{p.long}_vs_{p.short}"] = out["pair_ret_next"]
            pair_theta[p.name or f"{p.long}_vs_{p.short}"] = out["theta"]
        else:
            out = engine.run_basket_cdx(p)
            pair_signals[p.name] = out["z_pair"]
            # PnL proxy: next-day pair return
            pnl_proxies[p.name] = out["pair_ret_next"]
            pair_theta[p.name] = out["theta"]
            # Store decomposition for basket pairs
            decompositions[p.name] = out["decomp"]

    # Combine across pairs with risk parity & vol targeting
    cfg = PortfolioConfig(target_vol=0.10, vol_hl=20.0)
    positions, costs = combine_pairs_kelly(pair_signals, pnl_proxies, cfg)

    # Portfolio PnL (aggregated proxy) and KPIs
    pnl = positions.drop(columns=["portfolio"]).mul(pd.DataFrame(pnl_proxies)).sum(axis=1) - costs["portfolio"]
    equity = (1 + pnl.fillna(0.0)).cumprod()

    # Calculate comprehensive metrics
    results = {
        "equity": equity,
        "pnl": pnl,
        "positions": positions,
        "costs": costs,
        "signals": pair_signals,
        "theta": pair_theta,
        "decompositions": decompositions
    }
    
    # Calculate all performance metrics
    metrics = calculate_all_metrics(results, cdx_data=cdx)
    results["metrics"] = metrics

    # Print comprehensive metrics summary
    print_metrics_summary(metrics)
    
    # Calculate and print advanced drawdown analysis
    print("\n" + "=" * 60)
    print("ADVANCED DRAWDOWN ANALYSIS")
    print("=" * 60)
    advanced_dd_analysis = calculate_and_print_advanced_drawdown_analysis(pnl, equity)
    
    return results


if __name__ == "__main__":
    run_demo() 