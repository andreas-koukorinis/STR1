"""
rv_sim.py — Modular Synthetic Data Generator for Credit RV Strategies
====================================================================

This module provides a modular, functional approach to synthetic data generation
for credit RV strategies. It's designed to be easily extensible and modifiable
for more complex dynamics.

Key Features:
- Modular functions for each data component
- Configurable parameters for easy experimentation
- OOS-friendly design with separate train/test periods
- Extensible architecture for complex dynamics
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

__all__ = [
    "DataConfig",
    "generate_volatility_clustering",
    "generate_correlation_dynamics", 
    "generate_cdx_basis",
    "generate_cdx_spreads",
    "generate_etf_data",
    "create_synthetic_multi",
    "create_oss_split"
]


@dataclass
class DataConfig:
    """Configuration class for synthetic data generation."""
    # Time parameters
    n_days: int = 1600
    start_date: str = "2015-01-02"
    
    # Volatility parameters
    vol_persistence: float = 0.94
    vol_volatility: float = 0.06
    base_volatility: float = 0.004
    
    # Correlation parameters
    corr_persistence: float = 0.97
    corr_volatility: float = 0.25
    corr_min: float = 0.2
    corr_max: float = 0.8
    
    # Basis parameters
    basis_persistence: float = 0.97
    basis_volatility: float = 0.5
    jump_probability: float = 0.01
    jump_volatility: float = 5.0
    
    # CDX parameters
    hy_base_spread: float = 350.0
    ig_base_spread: float = 80.0
    hy_spread_vol: float = 0.8
    ig_spread_vol: float = 0.5
    
    # ETF parameters
    etf_configs: Dict[str, Dict] = None
    
    # Random seed
    seed: int = 11
    
    def __post_init__(self):
        if self.etf_configs is None:
            self.etf_configs = {
                "HYG": {"base_price": 90.0, "beta_to_F": 0.90, "idio_vol": 0.0035, "duration": 4.5, "rho_scale": 1.00},
                "JNK": {"base_price": 40.0, "beta_to_F": 0.85, "idio_vol": 0.0038, "duration": 4.0, "rho_scale": 1.00},
                "LQD": {"base_price": 110.0, "beta_to_F": 0.65, "idio_vol": 0.0032, "duration": 8.5, "rho_scale": 0.85},
                "VCIT": {"base_price": 95.0, "beta_to_F": 0.60, "idio_vol": 0.0034, "duration": 7.0, "rho_scale": 0.85},
            }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid function for smooth transitions."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_volatility_clustering(
    n_days: int, 
    persistence: float = 0.94, 
    volatility: float = 0.06, 
    base_vol: float = 0.004,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GARCH-like volatility clustering.
    
    Args:
        n_days: Number of days
        persistence: Volatility persistence parameter
        volatility: Volatility of volatility
        base_vol: Base volatility level
        rng: Random number generator
        
    Returns:
        Tuple of (volatility_squared, latent_factor)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    vol2 = np.zeros(n_days)
    vol2[0] = base_vol**2
    g_noise = rng.normal(0, 1, size=n_days)
    
    for t in range(1, n_days):
        vol2[t] = persistence * vol2[t-1] + volatility * (base_vol * g_noise[t-1])**2
    
    F = np.sqrt(vol2) * rng.normal(0, 1, size=n_days)
    return vol2, F


def generate_correlation_dynamics(
    n_days: int,
    persistence: float = 0.97,
    volatility: float = 0.25,
    corr_min: float = 0.2,
    corr_max: float = 0.8,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate time-varying correlation dynamics.
    
    Args:
        n_days: Number of days
        persistence: Correlation persistence
        volatility: Correlation volatility
        corr_min: Minimum correlation
        corr_max: Maximum correlation
        rng: Random number generator
        
    Returns:
        Array of time-varying correlations
    """
    if rng is None:
        rng = np.random.default_rng()
    
    u = np.zeros(n_days)
    for t in range(1, n_days):
        u[t] = persistence * u[t-1] + rng.normal(0, volatility)
    
    rho = corr_min + (corr_max - corr_min) * _sigmoid(u)
    return rho


def generate_cdx_basis(
    n_days: int,
    persistence: float = 0.97,
    volatility: float = 0.5,
    jump_prob: float = 0.01,
    jump_vol: float = 5.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate mean-reverting CDX basis with jumps.
    
    Args:
        n_days: Number of days
        persistence: Mean reversion speed
        volatility: Basis volatility
        jump_prob: Jump probability
        jump_vol: Jump volatility
        rng: Random number generator
        
    Returns:
        Array of basis values
    """
    if rng is None:
        rng = np.random.default_rng()
    
    basis = np.zeros(n_days)
    for t in range(1, n_days):
        jump = (rng.random() < jump_prob) * rng.normal(0, jump_vol)
        basis[t] = persistence * basis[t-1] + rng.normal(0, volatility) + jump
    
    return basis


def generate_cdx_spreads(
    n_days: int,
    latent_factor: np.ndarray,
    correlation: np.ndarray,
    base_spread: float,
    spread_vol: float,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate CDX spreads based on latent factor and correlation.
    
    Args:
        n_days: Number of days
        latent_factor: Latent credit factor
        correlation: Time-varying correlation
        base_spread: Base spread level
        spread_vol: Spread volatility
        rng: Random number generator
        
    Returns:
        Array of CDX spreads
    """
    if rng is None:
        rng = np.random.default_rng()
    
    eps_c = rng.normal(0, 1, n_days)
    dcdx_unit = correlation * latent_factor + np.sqrt(np.maximum(1 - correlation**2, 1e-6)) * eps_c
    spreads = base_spread + np.cumsum(spread_vol * dcdx_unit + rng.normal(0, 0.5, n_days))
    
    return spreads


def generate_etf_data(
    name: str,
    config: Dict,
    n_days: int,
    latent_factor: np.ndarray,
    correlation: np.ndarray,
    basis: np.ndarray,
    date_index: pd.DatetimeIndex,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Generate ETF data with realistic market dynamics.
    
    Args:
        name: ETF name
        config: ETF configuration dictionary
        n_days: Number of days
        latent_factor: Latent credit factor
        correlation: Time-varying correlation
        basis: CDX basis
        date_index: Date index
        rng: Random number generator
        
    Returns:
        DataFrame with ETF data
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Extract config parameters
    base_price = config["base_price"]
    beta_to_F = config["beta_to_F"]
    idio_vol = config["idio_vol"]
    duration = config["duration"]
    rho_scale = config["rho_scale"]
    
    # Generate NAV premium dynamics
    nav_prem = np.zeros(n_days)
    so = np.zeros(n_days)
    so[0] = 5e6
    
    for t in range(1, n_days):
        # NAV premium: mean-reverting with credit factor influence
        nav_prem[t] = (0.90 * nav_prem[t-1] + 
                       0.6 * (np.sign(-latent_factor[t-1]) * min(abs(latent_factor[t-1]), 0.02)) + 
                       rng.normal(0, 0.001))
        
        # Share outstanding: supply-demand effects
        dso = 2500 * (-nav_prem[t-1]) + rng.normal(0, 2000)
        so[t] = max(1e6, so[t-1] + dso)
    
    # Generate return components
    idio = idio_vol * rng.normal(0, 1, n_days)
    r_excess = (beta_to_F * (rho_scale * latent_factor) + idio + 
                0.05 * nav_prem + 
                0.02 * (np.diff(np.insert(so, 0, so[0])) / np.maximum(so, 1)) + 
                0.00005 * basis)
    
    # Price evolution
    close = base_price * np.exp(np.cumsum(r_excess))
    
    # Dividend and borrow dynamics
    div = 0.03 + 0.002 * rng.normal(size=n_days)
    borrow = 0.01 + 0.002 * rng.normal(size=n_days)
    
    # Create comprehensive DataFrame
    df = pd.DataFrame({
        "close": close,
        "nav": close * (1 - nav_prem),
        "dividend_yield": div,
        "borrow_fee": borrow,
        "credit_dv01": 0.7 + 0.05 * rng.normal(size=n_days),
        "duration": duration + rng.normal(0, 0.3, size=n_days),
        "volume": rng.integers(8e5, 4e6, size=n_days),
        "so": so,
        "basket_dv01": 0.5 + 0.1 * rng.normal(size=n_days),
        "rates_excess_ret": r_excess,
    }, index=date_index)
    
    df.name = name
    return df


def create_synthetic_multi(
    config: Optional[DataConfig] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Create sophisticated synthetic data for credit RV framework.
    
    Args:
        config: Data configuration object
        rng: Random number generator for reproducibility
        
    Returns:
        Tuple of (assets_dict, cdx_dict)
    """
    if config is None:
        config = DataConfig()
    
    if rng is None:
        rng = np.random.default_rng(config.seed)
    
    # Generate date index
    date_index = pd.bdate_range(config.start_date, periods=config.n_days)
    
    # Generate volatility clustering
    vol2, latent_factor = generate_volatility_clustering(
        config.n_days, 
        config.vol_persistence, 
        config.vol_volatility, 
        config.base_volatility,
        rng
    )
    
    # Generate correlation dynamics
    correlation = generate_correlation_dynamics(
        config.n_days,
        config.corr_persistence,
        config.corr_volatility,
        config.corr_min,
        config.corr_max,
        rng
    )
    
    # Generate CDX basis
    basis = generate_cdx_basis(
        config.n_days,
        config.basis_persistence,
        config.basis_volatility,
        config.jump_probability,
        config.jump_volatility,
        rng
    )
    
    # Generate CDX spreads
    cdx_hy_spreads = generate_cdx_spreads(
        config.n_days, latent_factor, correlation, 
        config.hy_base_spread, config.hy_spread_vol, rng
    )
    
    cdx_ig_spreads = generate_cdx_spreads(
        config.n_days, latent_factor, correlation, 
        config.ig_base_spread, config.ig_spread_vol, rng
    )
    
    # Generate ETF data
    assets = {}
    for name, etf_config in config.etf_configs.items():
        assets[name] = generate_etf_data(
            name, etf_config, config.n_days, 
            latent_factor, correlation, basis, date_index, rng
        )
    
    # Create CDX DataFrames
    cdx = {
        "HY": pd.DataFrame({
            "spread": cdx_hy_spreads,
            "dv01": 45.0 + 2.0 * rng.normal(config.n_days),
            "index_basis": basis
        }, index=date_index),
        "IG": pd.DataFrame({
            "spread": cdx_ig_spreads,
            "dv01": 75.0 + 3.0 * rng.normal(config.n_days),
            "index_basis": basis * 0.6
        }, index=date_index),
    }
    
    return assets, cdx


def create_oss_split(
    config: Optional[DataConfig] = None,
    train_ratio: float = 0.7,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], 
           Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Create out-of-sample split for train/test validation.
    
    Args:
        config: Data configuration
        train_ratio: Ratio of data for training
        rng: Random number generator
        
    Returns:
        Tuple of (train_assets, train_cdx, test_assets, test_cdx)
    """
    if config is None:
        config = DataConfig()
    
    # Generate full dataset
    assets, cdx = create_synthetic_multi(config, rng)
    
    # Split by date
    split_idx = int(len(assets[list(assets.keys())[0]]) * train_ratio)
    
    train_assets = {}
    test_assets = {}
    for name, df in assets.items():
        train_assets[name] = df.iloc[:split_idx]
        test_assets[name] = df.iloc[split_idx:]
    
    train_cdx = {}
    test_cdx = {}
    for name, df in cdx.items():
        train_cdx[name] = df.iloc[:split_idx]
        test_cdx[name] = df.iloc[split_idx:]
    
    return train_assets, train_cdx, test_assets, test_cdx


# Example usage and testing
if __name__ == "__main__":
    # Test the modular synthetic data generation
    print("Testing Modular Synthetic Data Generation")
    print("=" * 50)
    
    # Test with default config
    config = DataConfig(n_days=500, seed=42)
    assets, cdx = create_synthetic_multi(config)
    
    print(f"Generated {len(assets)} ETFs:")
    for name, df in assets.items():
        print(f"  {name}: Close range [{df['close'].min():.2f}, {df['close'].max():.2f}], "
              f"NAV prem range [{df['nav'].div(df['close']).sub(1).min():.3f}, "
              f"{df['nav'].div(df['close']).sub(1).max():.3f}]")
    
    print(f"\nGenerated {len(cdx)} CDX indices:")
    for name, df in cdx.items():
        print(f"  {name}: Spread range [{df['spread'].min():.1f}, {df['spread'].max():.1f}] bp, "
              f"Basis range [{df['index_basis'].min():.2f}, {df['index_basis'].max():.2f}] bp")
    
    print(f"\nDate range: {assets['HYG'].index[0]} to {assets['HYG'].index[-1]}")
    print(f"Total trading days: {len(assets['HYG'])}")
    
    # Test OOS split
    print("\nTesting OOS Split:")
    train_assets, train_cdx, test_assets, test_cdx = create_oss_split(config, train_ratio=0.7)
    print(f"Train period: {len(train_assets['HYG'])} days")
    print(f"Test period: {len(test_assets['HYG'])} days")
    
    print("\nModular synthetic data generation test completed successfully!") 