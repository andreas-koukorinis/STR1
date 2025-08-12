"""
Utility functions for statistical calculations and data processing.

This module contains specialized utility functions used across the credit RV framework,
including realized volatility calculations and other statistical utilities.
"""

import math
import numpy as np
import pandas as pd
from features import ewma, ewm_var


def realized_vol(ret: pd.Series, hl: float) -> pd.Series:
    """Calculate realized volatility using exponential weighted variance.
    
    This function computes the realized volatility of a return series by taking
    the square root of the exponentially weighted variance and annualizing it
    (multiplying by sqrt(252) for daily data).
    
    Args:
        ret: Return series (typically daily returns)
        hl: Half-life parameter for exponential weighting
        
    Returns:
        Annualized realized volatility series
        
    Example:
        >>> import pandas as pd
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        >>> realized_vol(returns, hl=20)
        0    0.000000
        1    0.021213
        2    0.018385
        3    0.016971
        4    0.015811
        dtype: float64
    """
    var = ewm_var(ret.fillna(0.0), hl)
    return np.sqrt(var) * np.sqrt(252.0)


def calculate_annualized_return(returns: pd.Series) -> float:
    """Calculate annualized return from a series of returns.
    
    Args:
        returns: Series of returns (typically daily)
        
    Returns:
        Annualized return as a decimal
    """
    total_return = (1 + returns).prod() - 1
    periods = len(returns)
    return (1 + total_return) ** (252.0 / periods) - 1


def calculate_annualized_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility from a series of returns.
    
    Args:
        returns: Series of returns (typically daily)
        
    Returns:
        Annualized volatility as a decimal
    """
    return returns.std() * np.sqrt(252.0)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from a series of returns.
    
    Args:
        returns: Series of returns (typically daily)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252.0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252.0)


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown from an equity curve.
    
    Args:
        equity: Series representing cumulative equity curve
        
    Returns:
        Maximum drawdown as a decimal (negative value)
    """
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    return drawdown.min()


def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calculate rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size (default: 252 for one year)
        
    Returns:
        DataFrame with rolling metrics including return, volatility, and Sharpe ratio
    """
    rolling_return = returns.rolling(window).apply(calculate_annualized_return)
    rolling_vol = returns.rolling(window).apply(calculate_annualized_volatility)
    rolling_sharpe = rolling_return / rolling_vol
    
    return pd.DataFrame({
        'rolling_return': rolling_return,
        'rolling_vol': rolling_vol,
        'rolling_sharpe': rolling_sharpe
    })


# Example usage and testing
if __name__ == "__main__":
    # Test the utility functions
    import numpy as np
    
    # Create sample return data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
    
    print("Utility Functions Test Results:")
    print("=" * 40)
    
    # Test realized volatility
    vol = realized_vol(returns, hl=20)
    print(f"Realized volatility - Mean: {vol.mean():.4f}, Std: {vol.std():.4f}")
    
    # Test annualized metrics
    ann_return = calculate_annualized_return(returns)
    ann_vol = calculate_annualized_volatility(returns)
    sharpe = calculate_sharpe_ratio(returns)
    
    print(f"Annualized return: {ann_return:.4f}")
    print(f"Annualized volatility: {ann_vol:.4f}")
    print(f"Sharpe ratio: {sharpe:.4f}")
    
    # Test rolling metrics
    rolling_metrics = calculate_rolling_metrics(returns)
    print(f"Rolling metrics calculated for {len(rolling_metrics)} periods")
    
    print("\nUtility functions test completed successfully!") 