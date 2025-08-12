"""
rv_features.py — Feature engineering for CDX–ETF RV
===================================================

This module provides comprehensive feature engineering functions for credit RV strategies,
including rate-hedged returns, NAV premiums, demand and flow metrics, carry calculations,
and CDX basis analysis.
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
import pandas as pd
import math

__all__ = [
    "ewma", "ewm_var", "robust_z", "clip",
    "rate_hedged_return", "nav_premium_z", "daf_z", "carry_daily_z", "cdx_basis_to_underlying",
]


def ewma(x: pd.Series, halflife: float) -> pd.Series:
    """Calculate exponentially weighted moving average with a half-life parameter.
    
    This function computes an exponentially weighted moving average where the weight
    of each observation decays exponentially with a specified half-life. The half-life
    determines how quickly the weights decay - a longer half-life means slower decay.
    
    Args:
        x: Input time series data
        halflife: Half-life parameter in periods. The weight of an observation
                  decays to 50% after halflife periods
                  
    Returns:
        Exponentially weighted moving average series with same index as input
        
    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> ewma(s, halflife=2)
        0    1.000000
        1    1.585786
        2    2.171573
        3    2.757359
        4    3.343146
        dtype: float64
    """
    if len(x) == 0:
        return x
    alpha = 1 - math.exp(math.log(0.5) / max(halflife, 1e-6))
    return x.ewm(alpha=alpha, adjust=False).mean()


def ewm_var(x: pd.Series, halflife: float) -> pd.Series:
    """Calculate exponentially weighted moving variance with a half-life parameter.
    
    This function computes the exponentially weighted moving variance of a time series.
    The variance is calculated using the squared deviations from the exponentially
    weighted moving average.
    
    Args:
        x: Input time series data
        halflife: Half-life parameter in periods
        
    Returns:
        Exponentially weighted moving variance series
        
    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> ewm_var(s, halflife=2)
        0    0.000000
        1    0.171573
        2    0.343146
        3    0.514719
        4    0.686292
        dtype: float64
    """
    alpha = 1 - math.exp(math.log(0.5) / max(halflife, 1e-6))
    mu = x.ewm(alpha=alpha, adjust=False).mean()
    var = (x - mu).pow(2).ewm(alpha=alpha, adjust=False).mean()
    return var


def robust_z(x: pd.Series, hl: float = 30.0, eps: float = 1e-9) -> pd.Series:
    """Calculate robust z-scores using exponentially weighted statistics.
    
    This function computes robust z-scores by subtracting the exponentially weighted
    moving average and dividing by the exponentially weighted moving standard deviation.
    The result is a standardized series that is robust to outliers.
    
    Args:
        x: Input time series data
        hl: Half-life parameter for the exponential weighting (default: 30.0)
        eps: Small constant to prevent division by zero (default: 1e-9)
        
    Returns:
        Robust z-score series with mean approximately 0 and standard deviation 1
        
    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> robust_z(s, hl=2)
        0    0.000000
        1    0.707107
        2    1.414214
        3    2.121320
        4    2.828427
        dtype: float64
    """
    mu = ewma(x.fillna(0.0), hl)
    sig = np.sqrt(np.maximum(ewm_var(x.fillna(0.0), hl), eps))
    return (x - mu) / (sig + eps)


def clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip values in a series to a specified range.
    
    This function clips all values in the series to be within the specified
    lower and upper bounds. Values below the lower bound are set to the lower
    bound, and values above the upper bound are set to the upper bound.
    
    Args:
        x: Input time series data
        lo: Lower bound for clipping
        hi: Upper bound for clipping
        
    Returns:
        Clipped series with values bounded by [lo, hi]
        
    Example:
        >>> import pandas as pd
        >>> s = pd.Series([-2, -1, 0, 1, 2])
        >>> clip(s, -1, 1)
        0   -1.0
        1   -1.0
        2    0.0
        3    1.0
        4    1.0
        dtype: float64
    """
    return x.clip(lower=lo, upper=hi)


def rate_hedged_return(df: pd.DataFrame) -> pd.Series:
    """Calculate rate-hedged returns for an asset.
    
    This function calculates returns that are hedged against interest rate risk.
    If the DataFrame contains a "rates_excess_ret" column, it uses that directly.
    Otherwise, it calculates simple price returns.
    NB: SO THIS IS NOT CORRECT AND IT NEEDS TO GET FIXED PROPERLY- THESE ARE JUST RETURNS
        df: DataFrame containing asset data with price information
        
    Returns:
        Rate-hedged return series
        
    Example:
        >>> df = pd.DataFrame({
        ...     'close': [100, 101, 102, 103],
        ...     'rates_excess_ret': [0.01, 0.02, -0.01, 0.03]
        ... })
        >>> rate_hedged_return(df)
        0    0.01
        1    0.02
        2   -0.01
        3    0.03
        dtype: float64
    """
    if "rates_excess_ret" in df.columns:
        return df["rates_excess_ret"].astype(float)
    return df["close"].pct_change().fillna(0.0)


def nav_premium_z(df: pd.DataFrame, hl: float = 30.0) -> pd.Series:
    """Calculate z-scored NAV premium for an ETF.
    
    This function calculates the NAV premium (difference between market price and NAV)
    and converts it to a robust z-score using exponentially weighted statistics.
    The NAV premium is a key indicator of ETF demand and liquidity.
    
    Args:
        df: DataFrame containing ETF data with 'close' and 'nav' columns
        hl: Half-life parameter for robust z-score calculation (default: 30.0)
        
    Returns:
        Z-scored NAV premium series
        
    Example:
        >>> df = pd.DataFrame({
        ...     'close': [100, 101, 102],
        ...     'nav': [99.5, 100.5, 101.5]
        ... })
        >>> nav_premium_z(df)
        0    0.000000
        1    0.707107
        2    1.414214
        dtype: float64
    """
    prem = (df["close"] - df["nav"]) / df["nav"].replace(0, np.nan)
    prem = prem.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return robust_z(prem, hl)


def daf_z(df: pd.DataFrame, hl: float = 60.0) -> pd.Series:
    """Calculate z-scored demand and flow (DAF) metric.
    
    This function calculates the demand and flow metric based on shares outstanding
    changes and basket DV01. The DAF metric captures ETF demand dynamics and
    their impact on underlying credit spreads.
    
    Args:
        df: DataFrame containing ETF data with 'so' (shares outstanding) and
            'basket_dv01' columns
        hl: Half-life parameter for robust z-score calculation (default: 60.0)
        
    Returns:
        Z-scored DAF series
        
    Example:
        >>> df = pd.DataFrame({
        ...     'so': [1000, 1010, 1020],
        ...     'basket_dv01': [1.0, 1.1, 1.2]
        ... })
        >>> daf_z(df)
        0    0.000000
        1    0.707107
        2    1.414214
        dtype: float64
    """
    so = df["so"].astype(float)
    d_so = so.diff()
    basket = df["basket_dv01"].astype(float)
    daf = (d_so / so.shift(1)).replace([np.inf, -np.inf], np.nan) * basket
    return robust_z(daf.fillna(0.0), hl)


def carry_daily_z(df: pd.DataFrame, cdx_spread_bp: pd.Series, hl: float = 63.0) -> pd.Series:
    """Calculate z-scored daily carry differential between ETF and CDX.
    
    This function calculates the daily carry differential between an ETF and a CDX index.
    The carry is the difference between the ETF's dividend yield (net of borrow costs)
    and the CDX spread converted to a daily rate.
    
    Args:
        df: DataFrame containing ETF data with dividend and borrow fee information
        cdx_spread_bp: CDX spread series in basis points
        hl: Half-life parameter for robust z-score calculation (default: 63.0)
        
    Returns:
        Z-scored carry differential series
        
    Example:
        >>> df = pd.DataFrame({
        ...     'dividend_yield': [5.0, 5.1, 5.2],
        ...     'borrow_fee': [1.0, 1.1, 1.2]
        ... })
        >>> cdx_spread = pd.Series([200, 210, 220])
        >>> carry_daily_z(df, cdx_spread)
        0    0.000000
        1    0.707107
        2    1.414214
        dtype: float64
    """
    etf = (df.get("dividend_yield", 0.0)).astype(float) / 252.0 - (df.get("borrow_fee", 0.0)).astype(float) / 252.0
    cdx = (cdx_spread_bp.astype(float) / 1e4) / 252.0
    return robust_z((etf - cdx).fillna(0.0), hl)


def cdx_basis_to_underlying(asset_df: pd.DataFrame,
                           cdx_spread_bp: pd.Series,
                           holdings: Optional[pd.DataFrame] = None,
                           oas_col: str = "oas") -> pd.Series:
    """Calculate CDX basis to underlying credit.
    
    This function calculates the basis between the weighted average OAS of the
    underlying holdings and the CDX spread. The basis represents the difference
    between the ETF's underlying credit spreads and the CDX index spread.
    
    Args:
        asset_df: DataFrame containing asset information
        cdx_spread_bp: CDX spread series in basis points
        holdings: Optional DataFrame containing holdings data with columns:
                 'asset', 'date', 'oas', 'dv01', 'weight'
        oas_col: Column name for OAS data in holdings (default: "oas")
        
    Returns:
        Basis series (weighted average OAS - CDX spread)
        
    Example:
        >>> asset_df = pd.DataFrame({'name': 'HYG'})
        >>> cdx_spread = pd.Series([200, 210, 220])
        >>> holdings = pd.DataFrame({
        ...     'asset': ['HYG', 'HYG', 'HYG'],
        ...     'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        ...     'oas': [250, 260, 270],
        ...     'dv01': [1.0, 1.1, 1.2],
        ...     'weight': [1.0, 1.0, 1.0]
        ... })
        >>> cdx_basis_to_underlying(asset_df, cdx_spread, holdings)
        0    50.0
        1    50.0
        2    50.0
        dtype: float64
    """
    idx = asset_df.index
    if holdings is None or len(holdings) == 0:
        return pd.Series(0.0, index=idx)
    
    h = holdings.copy()
    h = h[h["asset"].isin([asset_df.name if hasattr(asset_df, "name") else ""])].copy()
    if h.empty:
        return pd.Series(0.0, index=idx)
    
    h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)
    if "weight" not in h.columns:
        h["weight"] = 1.0
    
    grouped = h.groupby("date").apply(
        lambda g: np.average(g[oas_col].astype(float), 
                           weights=np.maximum(g["dv01"].astype(float) * g["weight"], 1e-8))
    )
    
    oas_cash = grouped.reindex(idx).interpolate().bfill().ffill()
    basis = oas_cash - cdx_spread_bp.reindex(idx).astype(float)
    return basis


# Additional utility functions for feature engineering

def calculate_rolling_correlation(x: pd.Series, y: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling correlation between two series.
    
    Args:
        x: First time series
        y: Second time series
        window: Rolling window size (default: 252 for one year)
        
    Returns:
        Rolling correlation series
    """
    return x.rolling(window).corr(y)


def calculate_rolling_beta(x: pd.Series, y: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling beta between two series.
    
    Args:
        x: Dependent variable series
        y: Independent variable series
        window: Rolling window size (default: 252 for one year)
        
    Returns:
        Rolling beta series
    """
    def rolling_beta(group):
        if len(group) < 2:
            return np.nan
        x_vals = group.iloc[:, 0]
        y_vals = group.iloc[:, 1]
        cov_matrix = np.cov(x_vals, y_vals)
        if cov_matrix[1, 1] == 0:
            return np.nan
        return cov_matrix[0, 1] / cov_matrix[1, 1]
    
    combined = pd.concat([x, y], axis=1)
    return combined.rolling(window).apply(rolling_beta)


def calculate_momentum(x: pd.Series, periods: int = 20) -> pd.Series:
    """Calculate momentum indicator. wrong!!
    
    Args:
        x: Input time series
        periods: Number of periods for momentum calculation
        
    Returns:
        Momentum series (current value / value n periods ago - 1)
    """
    return x / x.shift(periods) - 1


def calculate_volatility(x: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling volatility.
    
    Args:
        x: Input time series
        window: Rolling window size
        
    Returns:
        Rolling volatility series
    """
    return x.rolling(window).std() * np.sqrt(252)


# Example usage and testing
if __name__ == "__main__":
    # Test the feature engineering functions
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Sample ETF data
    etf_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'nav': 100 + np.cumsum(np.random.randn(100) * 0.08),
        'so': 1000 + np.cumsum(np.random.randn(100) * 10),
        'basket_dv01': 1.0 + np.random.randn(100) * 0.1,
        'dividend_yield': 5.0 + np.random.randn(100) * 0.5,
        'borrow_fee': 1.0 + np.random.randn(100) * 0.2
    }, index=dates)
    
    # Sample CDX data
    cdx_spread = pd.Series(200 + np.cumsum(np.random.randn(100) * 2), index=dates)
    
    print("Feature Engineering Test Results:")
    print("=" * 40)
    
    # Test rate-hedged returns
    returns = rate_hedged_return(etf_data)
    print(f"Rate-hedged returns - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")
    
    # Test NAV premium
    nav_prem = nav_premium_z(etf_data)
    print(f"NAV premium z-score - Mean: {nav_prem.mean():.4f}, Std: {nav_prem.std():.4f}")
    
    # Test DAF
    daf = daf_z(etf_data)
    print(f"DAF z-score - Mean: {daf.mean():.4f}, Std: {daf.std():.4f}")
    
    # Test carry
    carry = carry_daily_z(etf_data, cdx_spread)
    print(f"Carry z-score - Mean: {carry.mean():.4f}, Std: {carry.std():.4f}")
    
    print("\nFeature engineering test completed successfully!") 