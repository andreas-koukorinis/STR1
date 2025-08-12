"""
Portfolio management and risk control.

This module handles position sizing, transaction costs, and risk-parity
combination across multiple trading pairs.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from features import clip
from utils import realized_vol


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management and risk control.
    
    This class contains all the parameters needed for position sizing,
    risk management, and transaction cost modeling.
    
    Attributes:
        target_vol: Target annualized volatility for the portfolio (default: 0.10 = 10%)
        vol_hl: Half-life for volatility estimation (default: 20.0 days)
        tanh_scale: Scaling factor for tanh position sizing (default: 1.5)
        etf_tc_bps: Round-trip transaction cost for ETF legs in basis points (default: 1.0)
        cdx_tc_bpDv01: Round-trip transaction cost for CDX leg per DV01 (default: 0.2)
        
    Example:
        >>> config = PortfolioConfig(target_vol=0.15, vol_hl=30.0)
        >>> print(f"Target volatility: {config.target_vol:.1%}")
        Target volatility: 15.0%
    """
    target_vol: float = 0.10
    vol_hl: float = 20.0
    tanh_scale: float = 1.5
    etf_tc_bps: float = 1.0    # round-trip cost for ETF legs
    cdx_tc_bpDv01: float = 0.2 # round-trip cost for CDX leg (for basket↔CDX)


def size_and_cost(signal: pd.Series, pnl_proxy: pd.Series, cfg: PortfolioConfig) -> Tuple[pd.Series, pd.Series]:
    """Vol-target the signal and compute simple turnover-based transaction cost.
    
    This function implements volatility targeting by scaling the signal to achieve
    the target volatility, and computes transaction costs based on position turnover.
    
    The position sizing follows:
    1. Calculate realized volatility of the PnL proxy
    2. Scale signal by target_vol / realized_vol
    3. Clip positions to reasonable bounds (-3 to 3)
    4. Calculate transaction costs based on position changes
    
    Args:
        signal: Raw trading signal (z-score or similar)
        pnl_proxy: Proxy for PnL used to estimate volatility
        cfg: Portfolio configuration with target volatility and cost parameters
        
    Returns:
        Tuple containing:
            - positions: Volatility-targeted positions
            - costs: Transaction costs as a percentage of notional
            
    Example:
        >>> import pandas as pd
        >>> signal = pd.Series([1.0, -0.5, 0.8, -1.2])
        >>> pnl_proxy = pd.Series([0.01, -0.02, 0.015, -0.01])
        >>> config = PortfolioConfig(target_vol=0.10)
        >>> positions, costs = size_and_cost(signal, pnl_proxy, config)
        >>> print(f"Max position: {positions.max():.3f}")
        >>> print(f"Total costs: {costs.sum():.4f}")
    """
    rv = realized_vol(pnl_proxy, cfg.vol_hl)
    scale = cfg.target_vol / rv.replace(0, np.nan)
    scale = scale.ffill().fillna(0.0)
    pos = clip(signal * scale, -3.0, 3.0)
    turn = pos.diff().abs().fillna(0.0)
    tc = (cfg.etf_tc_bps / 1e4) * turn  # add CDX cost in basket engine below if you track that leg explicitly
    return pos, tc


def combine_pairs_kelly(signals: Dict[str, pd.Series], pnl_proxies: Dict[str, pd.Series], cfg: PortfolioConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Risk-parity combine across pairs using inverse EW vol weights.

    This function implements a risk-parity approach to combine multiple trading
    signals. The method:
    1. Volatility-targets each signal individually
    2. Calculates inverse volatility weights for risk parity
    3. Applies the weights to get final positions
    4. Aggregates transaction costs
    
    The risk parity approach gives more weight to strategies with lower volatility,
    aiming for equal risk contribution from each strategy.
    
    Args:
        signals: Dictionary mapping strategy names to raw signals
        pnl_proxies: Dictionary mapping strategy names to PnL proxies for volatility estimation
        cfg: Portfolio configuration for position sizing and costs
        
    Returns:
        Tuple containing:
            - positions: DataFrame with final positions for each strategy and portfolio total
            - costs: DataFrame with transaction costs for each strategy and portfolio total
            
    Example:
        >>> signals = {
        ...     'pair1': pd.Series([1.0, -0.5, 0.8]),
        ...     'pair2': pd.Series([0.5, 1.0, -0.3])
        ... }
        >>> pnl_proxies = {
        ...     'pair1': pd.Series([0.01, -0.02, 0.015]),
        ...     'pair2': pd.Series([0.005, 0.01, -0.005])
        ... }
        >>> config = PortfolioConfig(target_vol=0.10)
        >>> positions, costs = combine_pairs_kelly(signals, pnl_proxies, config)
        >>> print(f"Portfolio positions shape: {positions.shape}")
        >>> print(f"Total portfolio cost: {costs['portfolio'].sum():.4f}")
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
        w = w.ffill().fillna(0.0)
        weights[name] = clip(w / w.replace(0, np.nan).rolling(20).mean().ffill().fillna(1.0), 0.0, 3.0)
        sized[name] = pos * weights[name]
        costs[name] = tc
        
    df_pos = pd.DataFrame({k: v for k, v in sized.items()})
    df_tc = pd.DataFrame({k: v for k, v in costs.items()})
    df_pos["portfolio"] = df_pos.sum(axis=1)
    df_tc["portfolio"] = df_tc.sum(axis=1)
    
    return df_pos, df_tc 