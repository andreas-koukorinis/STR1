"""
rv_decomposition.py — Daily return decomposition into spread / carry / roll
===========================================================================

Approximates the credit excess return of an ETF (or basket) as:
  r_excess ≈ - (DV01_price) * ΔOAS + (DV01_price) * (OAS/1e4)/252
             + (ΔDV01_price) * (OAS/1e4) + (div_yield - borrow)/252 + residual
where DV01_price = basket_dv01 / close.

This is intentionally simple and auditable. With bond‑level OAS curves you can
swap the OAS proxy for a true curve‑based roll/carry.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from features import rate_hedged_return

__all__ = [
    "decompose_etf_credit", "decompose_basket_credit"
]


def _dv01_per_price(df: pd.DataFrame) -> pd.Series:
    """Calculate DV01 per price ratio.
    
    This function calculates the ratio of basket DV01 to close price,
    which is used in return decomposition calculations.
    
    Args:
        df: DataFrame containing 'basket_dv01' and 'close' columns
        
    Returns:
        Series of DV01 per price ratios
    """
    return (df["basket_dv01"].astype(float) / df["close"].astype(float)).replace([np.inf, -np.inf], np.nan)


def decompose_etf_credit(df: pd.DataFrame,
                         oas_level_bp: pd.Series,
                         include_div_borrow: bool = True) -> pd.DataFrame:
    """Decompose ETF credit excess return into spread, carry, roll, and residual components.
    
    This function breaks down the credit excess return of an ETF into its fundamental
    components: spread return (from OAS changes), carry (from current OAS level),
    roll (from DV01 changes), and residual (unexplained component).
    
    Args:
        df: DataFrame with ETF data including 'close', 'basket_dv01', etc.
        oas_level_bp: Series of OAS levels in basis points
        include_div_borrow: Whether to include dividend/borrow component
        
    Returns:
        DataFrame with decomposition components:
        - spread_return: Return from OAS changes
        - carry: Return from current OAS level
        - roll: Return from DV01 changes
        - residual: Unexplained component
        - div_borrow: Dividend yield minus borrow cost (if include_div_borrow=True)
        
    Example:
        >>> df = pd.DataFrame({
        ...     'close': [100, 101, 102],
        ...     'basket_dv01': [0.5, 0.51, 0.52],
        ...     'dividend_yield': [5.0, 5.1, 5.2],
        ...     'borrow_fee': [1.0, 1.1, 1.2]
        ... })
        >>> oas = pd.Series([200, 210, 205])
        >>> result = decompose_etf_credit(df, oas)
        >>> print(result.columns)
        Index(['spread_return', 'carry', 'roll', 'residual', 'div_borrow'])
    """
    idx = df.index
    oas = oas_level_bp.reindex(idx).astype(float).ffill().bfill()
    doas = oas.diff().fillna(0.0)
    dv01p = _dv01_per_price(df).reindex(idx).ffill().bfill()

    # Spread return: -DV01_price * ΔOAS
    spread_ret = - dv01p * (doas)  # bp * (DV01/Price) → return
    
    # Carry: DV01_price * (OAS/1e4)/252
    carry = dv01p * (oas / 1e4) / 252.0
    
    # Roll: ΔDV01_price * (OAS/1e4)
    roll = (df["basket_dv01"].shift(-1) - df["basket_dv01"]) / df["close"] * (oas / 1e4)

    # Dividend/Borrow: (div_yield - borrow)/252
    div_borrow = (df.get("dividend_yield", 0.0) / 252.0 - df.get("borrow_fee", 0.0) / 252.0)
    
    # Actual return
    y = rate_hedged_return(df).reindex(idx).fillna(0.0)

    # Calculate residual
    parts = spread_ret + carry + roll + (div_borrow if include_div_borrow else 0.0)
    residual = y - parts

    # Create output DataFrame
    out = pd.DataFrame({
        "spread_return": spread_ret,
        "carry": carry,
        "roll": roll,
        "residual": residual,
    }, index=idx)
    
    if include_div_borrow:
        out["div_borrow"] = div_borrow
        
    return out


def decompose_basket_credit(dfs: List[pd.DataFrame],
                           weights: np.ndarray,
                           oas_level_bp: pd.Series) -> pd.DataFrame:
    """Weighted sum of per‑ETF decompositions to obtain a basket decomposition.
    
    This function creates a weighted decomposition for a basket of ETFs by
    combining the individual ETF decompositions according to the specified weights.
    
    Args:
        dfs: List of DataFrames, one for each ETF in the basket
        weights: Array of weights for each ETF (will be normalized)
        oas_level_bp: Series of OAS levels in basis points
        
    Returns:
        DataFrame with weighted basket decomposition components
        
    Example:
        >>> df1 = pd.DataFrame({'close': [100, 101], 'basket_dv01': [0.5, 0.51]})
        >>> df2 = pd.DataFrame({'close': [200, 201], 'basket_dv01': [1.0, 1.01]})
        >>> weights = np.array([0.6, 0.4])
        >>> oas = pd.Series([200, 210])
        >>> result = decompose_basket_credit([df1, df2], weights, oas)
        >>> print(result.columns)
        Index(['decomp_spread_return', 'decomp_carry', 'decomp_roll', 'decomp_residual'])
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(np.abs(weights))
    idx = dfs[0].index
    comps = []
    
    for i, df in enumerate(dfs):
        sub = decompose_etf_credit(df, oas_level_bp)
        comps.append(sub.mul(weights[i]))
    
    out = sum(comps)
    out.columns = [f"decomp_{c}" for c in out.columns]
    return out


def calculate_decomposition_metrics(decomposition: pd.DataFrame) -> dict:
    """Calculate summary metrics for return decomposition.
    
    Args:
        decomposition: DataFrame from decompose_etf_credit or decompose_basket_credit
        
    Returns:
        Dictionary with decomposition metrics including:
        - component_contributions: Relative contribution of each component
        - component_correlations: Correlations between components
        - residual_analysis: Statistics about the residual component
    """
    # Calculate component contributions
    total_variance = decomposition.var().sum()
    component_contributions = decomposition.var() / total_variance
    
    # Calculate component correlations
    component_correlations = decomposition.corr()
    
    # Residual analysis
    residual_stats = {
        'mean': decomposition['residual'].mean(),
        'std': decomposition['residual'].std(),
        'max': decomposition['residual'].max(),
        'min': decomposition['residual'].min(),
        'explained_variance_ratio': 1 - decomposition['residual'].var() / total_variance
    }
    
    return {
        'component_contributions': component_contributions,
        'component_correlations': component_correlations,
        'residual_analysis': residual_stats
    }


def plot_decomposition(decomposition: pd.DataFrame, 
                      title: str = "Return Decomposition",
                      figsize: tuple = (15, 10)) -> None:
    """Plot return decomposition components.
    
    Args:
        decomposition: DataFrame from decompose_etf_credit or decompose_basket_credit
        title: Plot title
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Time series of components
    for col in decomposition.columns:
        if col != 'residual':
            axes[0, 0].plot(decomposition.index, decomposition[col], label=col, alpha=0.7)
    axes[0, 0].set_title('Decomposition Components Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual analysis
    axes[0, 1].plot(decomposition.index, decomposition['residual'], color='red', alpha=0.7)
    axes[0, 1].set_title('Residual Component')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Component distribution
    decomposition.drop('residual', axis=1).boxplot(ax=axes[1, 0])
    axes[1, 0].set_title('Component Distributions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Cumulative decomposition
    cumulative = decomposition.cumsum()
    for col in cumulative.columns:
        if col != 'residual':
            axes[1, 1].plot(cumulative.index, cumulative[col], label=col, alpha=0.7)
    axes[1, 1].set_title('Cumulative Decomposition')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the decomposition functions
    import numpy as np
    
    # Create sample ETF data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Sample ETF data
    etf_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'basket_dv01': 0.5 + np.random.randn(100) * 0.05,
        'dividend_yield': 5.0 + np.random.randn(100) * 0.5,
        'borrow_fee': 1.0 + np.random.randn(100) * 0.2,
        'rates_excess_ret': np.random.randn(100) * 0.02
    }, index=dates)
    
    # Sample OAS data
    oas_level = pd.Series(200 + np.cumsum(np.random.randn(100) * 2), index=dates)
    
    print("Return Decomposition Test Results:")
    print("=" * 40)
    
    # Test ETF decomposition
    decomp = decompose_etf_credit(etf_data, oas_level)
    print(f"Decomposition components: {list(decomp.columns)}")
    print(f"Component means: {decomp.mean()}")
    print(f"Component stds: {decomp.std()}")
    
    # Test metrics
    metrics = calculate_decomposition_metrics(decomp)
    print(f"\nComponent contributions: {metrics['component_contributions']}")
    print(f"Explained variance ratio: {metrics['residual_analysis']['explained_variance_ratio']:.4f}")
    
    # Test basket decomposition
    etf_data2 = etf_data.copy()
    etf_data2['close'] = etf_data2['close'] * 1.1
    etf_data2['basket_dv01'] = etf_data2['basket_dv01'] * 1.2
    
    basket_decomp = decompose_basket_credit([etf_data, etf_data2], 
                                          np.array([0.6, 0.4]), oas_level)
    print(f"\nBasket decomposition components: {list(basket_decomp.columns)}")
    
    print("\nReturn decomposition test completed successfully!") 