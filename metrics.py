"""
Comprehensive performance and risk metrics for credit RV framework.

This module provides extensive metrics calculation including traditional performance
metrics, risk metrics, drawdown analysis, and credit-specific metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from drawdown_analysis import calculate_advanced_drawdown_metrics, print_drawdown_summary


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""
    
    # Basic metrics
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    
    # Advanced drawdown metrics
    num_drawdown_episodes: int
    avg_drawdown_depth: float
    max_drawdown_depth: float
    underwater_ratio: float
    num_runup_episodes: int
    avg_runup_length: float
    max_runup_length: float
    runup_ratio: float
    
    # Additional metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    
    # Credit-specific metrics
    credit_beta: float
    spread_correlation: float
    basis_volatility: float
    
    # Transaction metrics
    total_turnover: float
    average_position_size: float
    max_position_size: float
    total_transaction_costs: float


def calculate_basic_metrics(equity: pd.Series, pnl: pd.Series) -> Dict[str, float]:
    """Calculate basic performance metrics.
    
    Args:
        equity: Cumulative equity series
        pnl: Daily PnL series
        
    Returns:
        Dictionary of basic performance metrics
    """
    # Basic return metrics
    total_return = equity.iloc[-1] - 1
    annual_return = equity.iloc[-1] ** (252/len(equity)) - 1
    annual_volatility = pnl.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = annual_return / (annual_volatility + 1e-12)
    
    # Sortino ratio (using downside deviation)
    downside_returns = pnl[pnl < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-12
    sortino_ratio = annual_return / downside_deviation
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio
    }


def calculate_drawdown_metrics(equity: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive drawdown metrics.
    
    Args:
        equity: Cumulative equity series
        
    Returns:
        Dictionary of drawdown metrics
    """
    # Calculate drawdown series
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    
    # Maximum drawdown
    max_drawdown = drawdown.min()
    
    # Drawdown duration
    max_dd_idx = drawdown.idxmin()
    recovery_idx = equity[equity.index >= max_dd_idx]
    recovery_idx = recovery_idx[recovery_idx >= rolling_max.loc[max_dd_idx]]
    
    if len(recovery_idx) > 0:
        max_drawdown_duration = (recovery_idx.index[0] - max_dd_idx).days
    else:
        max_drawdown_duration = (equity.index[-1] - max_dd_idx).days
    
    # Calmar ratio
    annual_return = equity.iloc[-1] ** (252/len(equity)) - 1
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'calmar_ratio': calmar_ratio
    }


def calculate_risk_metrics(pnl: pd.Series) -> Dict[str, float]:
    """Calculate Value at Risk and Conditional Value at Risk.
    
    Args:
        pnl: Daily PnL series
        
    Returns:
        Dictionary of risk metrics
    """
    # Value at Risk (95% and 99%)
    var_95 = np.percentile(pnl, 5)
    var_99 = np.percentile(pnl, 1)
    
    # Conditional Value at Risk (Expected Shortfall)
    cvar_95 = pnl[pnl <= var_95].mean()
    cvar_99 = pnl[pnl <= var_99].mean()
    
    return {
        'var_95': var_95,
        'cvar_95': cvar_95,
        'var_99': var_99,
        'cvar_99': cvar_99
    }


def calculate_trading_metrics(pnl: pd.Series, positions: pd.DataFrame, costs: pd.DataFrame) -> Dict[str, float]:
    """Calculate trading-specific metrics.
    
    Args:
        pnl: Daily PnL series
        positions: Position DataFrame
        costs: Transaction costs DataFrame
        
    Returns:
        Dictionary of trading metrics
    """
    # Win/loss analysis
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    
    win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0
    average_win = wins.mean() if len(wins) > 0 else 0
    average_loss = losses.mean() if len(losses) > 0 else 0
    largest_win = wins.max() if len(wins) > 0 else 0
    largest_loss = losses.min() if len(losses) > 0 else 0
    
    # Profit factor
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 1e-12
    profit_factor = total_wins / total_losses
    
    # Consecutive wins/losses
    consecutive_wins = calculate_max_consecutive(pnl > 0)
    consecutive_losses = calculate_max_consecutive(pnl < 0)
    
    # Position and turnover metrics
    portfolio_positions = positions['portfolio'] if 'portfolio' in positions.columns else positions.sum(axis=1)
    total_turnover = portfolio_positions.diff().abs().sum()
    average_position_size = portfolio_positions.abs().mean()
    max_position_size = portfolio_positions.abs().max()
    total_transaction_costs = costs['portfolio'].sum() if 'portfolio' in costs.columns else costs.sum().sum()
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'average_win': average_win,
        'average_loss': average_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses,
        'total_turnover': total_turnover,
        'average_position_size': average_position_size,
        'max_position_size': max_position_size,
        'total_transaction_costs': total_transaction_costs
    }


def calculate_credit_metrics(signals: Dict[str, pd.Series], 
                           theta: Dict[str, pd.DataFrame],
                           cdx_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, float]:
    """Calculate credit-specific metrics.
    
    Args:
        signals: Dictionary of trading signals
        theta: Dictionary of parameter estimates
        cdx_data: Optional CDX data for correlation analysis
        
    Returns:
        Dictionary of credit metrics
    """
    # Credit beta (correlation with credit factor)
    if len(signals) > 1:
        signals_df = pd.DataFrame(signals)
        # Use first principal component as credit factor proxy
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        credit_factor = pd.Series(pca.fit_transform(signals_df.fillna(0)).flatten(), index=signals_df.index)
        credit_beta = credit_factor.corr(signals_df.mean(axis=1))
    else:
        credit_beta = 0.0
    
    # Spread correlation (if CDX data available)
    if cdx_data and len(cdx_data) > 0:
        # Calculate average spread change
        spread_changes = []
        for cdx_name, cdx_df in cdx_data.items():
            if 'spread' in cdx_df.columns:
                spread_changes.append(cdx_df['spread'].pct_change())
        
        if spread_changes:
            avg_spread_change = pd.concat(spread_changes, axis=1).mean(axis=1)
            signals_mean = pd.DataFrame(signals).mean(axis=1)
            spread_correlation = avg_spread_change.corr(signals_mean)
        else:
            spread_correlation = 0.0
    else:
        spread_correlation = 0.0
    
    # Basis volatility (volatility of parameter estimates)
    basis_volatility = 0.0
    if theta:
        all_params = []
        for param_df in theta.values():
            all_params.append(param_df)
        
        if all_params:
            combined_params = pd.concat(all_params, axis=1)
            basis_volatility = combined_params.std().mean()
    
    return {
        'credit_beta': credit_beta,
        'spread_correlation': spread_correlation,
        'basis_volatility': basis_volatility
    }


def calculate_max_consecutive(condition: pd.Series) -> int:
    """Calculate maximum consecutive occurrences of a condition.
    
    Args:
        condition: Boolean series
        
    Returns:
        Maximum consecutive count
    """
    max_consecutive = 0
    current_consecutive = 0
    
    for value in condition:
        if value:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def calculate_rolling_metrics(pnl: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calculate rolling performance metrics.
    
    Args:
        pnl: Daily PnL series
        window: Rolling window size (default: 252 days)
        
    Returns:
        DataFrame with rolling metrics
    """
    rolling_metrics = pd.DataFrame(index=pnl.index)
    
    # Rolling returns
    equity = (1 + pnl.fillna(0.0)).cumprod()
    rolling_metrics['rolling_return'] = equity.rolling(window).apply(
        lambda x: (x.iloc[-1] / x.iloc[0]) ** (252/window) - 1
    )
    
    # Rolling volatility
    rolling_metrics['rolling_volatility'] = pnl.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    rolling_metrics['rolling_sharpe'] = (
        rolling_metrics['rolling_return'] / rolling_metrics['rolling_volatility']
    )
    
    # Rolling drawdown
    rolling_max = equity.rolling(window).max()
    rolling_metrics['rolling_drawdown'] = (equity - rolling_max) / rolling_max * 100
    
    # Rolling VaR
    rolling_metrics['rolling_var_95'] = pnl.rolling(window).quantile(0.05)
    
    return rolling_metrics


def calculate_all_metrics(results: Dict, 
                         cdx_data: Optional[Dict[str, pd.DataFrame]] = None) -> PerformanceMetrics:
    """Calculate all performance and risk metrics.
    
    Args:
        results: Dictionary containing all results from run_demo()
        cdx_data: Optional CDX data for correlation analysis
        
    Returns:
        PerformanceMetrics object with all calculated metrics
    """
    equity = results['equity']
    pnl = results['pnl']
    positions = results['positions']
    costs = results['costs']
    signals = results['signals']
    theta = results['theta']
    
    # Calculate all metric categories
    basic_metrics = calculate_basic_metrics(equity, pnl)
    drawdown_metrics = calculate_drawdown_metrics(equity)
    risk_metrics = calculate_risk_metrics(pnl)
    trading_metrics = calculate_trading_metrics(pnl, positions, costs)
    credit_metrics = calculate_credit_metrics(signals, theta, cdx_data)
    
    # Calculate advanced drawdown metrics
    advanced_dd_metrics = calculate_advanced_drawdown_metrics(pnl, equity)
    dd_stats = advanced_dd_metrics['drawdown_stats']
    ru_stats = advanced_dd_metrics['runup_stats']
    
    advanced_metrics = {
        'num_drawdown_episodes': dd_stats['num_episodes'],
        'avg_drawdown_depth': dd_stats['avg_depth'],
        'max_drawdown_depth': dd_stats['max_depth'],
        'underwater_ratio': dd_stats['underwater_ratio'],
        'num_runup_episodes': ru_stats['num_episodes'],
        'avg_runup_length': ru_stats['avg_length'],
        'max_runup_length': ru_stats['max_length'],
        'runup_ratio': ru_stats['runup_ratio']
    }
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **drawdown_metrics, **risk_metrics, 
                   **trading_metrics, **credit_metrics, **advanced_metrics}
    
    return PerformanceMetrics(**all_metrics)


def calculate_and_print_advanced_drawdown_analysis(pnl: pd.Series, equity: pd.Series) -> None:
    """Calculate and print detailed advanced drawdown analysis.
    
    Args:
        pnl: Daily PnL series
        equity: Cumulative equity series
    """
    # Calculate advanced drawdown metrics
    advanced_metrics = calculate_advanced_drawdown_metrics(pnl, equity)
    
    # Print detailed analysis
    print_drawdown_summary(advanced_metrics)
    
    return advanced_metrics


def print_metrics_summary(metrics: PerformanceMetrics) -> None:
    """Print a comprehensive summary of all metrics.
    
    Args:
        metrics: PerformanceMetrics object
    """
    print("=" * 60)
    print("CREDIT RV FRAMEWORK - COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print("\n📈 BASIC PERFORMANCE METRICS:")
    print(f"  Total Return:           {metrics.total_return:>10.2%}")
    print(f"  Annual Return:          {metrics.annual_return:>10.2%}")
    print(f"  Annual Volatility:      {metrics.annual_volatility:>10.2%}")
    print(f"  Sharpe Ratio:           {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:          {metrics.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:           {metrics.calmar_ratio:>10.2f}")
    
    print("\n⚠️  RISK METRICS:")
    print(f"  Maximum Drawdown:       {metrics.max_drawdown:>10.2%}")
    print(f"  Max DD Duration:        {metrics.max_drawdown_duration:>10d} days")
    print(f"  VaR (95%):              {metrics.var_95:>10.4f}")
    print(f"  CVaR (95%):             {metrics.cvar_95:>10.4f}")
    print(f"  VaR (99%):              {metrics.var_99:>10.4f}")
    print(f"  CVaR (99%):             {metrics.cvar_99:>10.4f}")
    
    print("\n🎯 TRADING METRICS:")
    print(f"  Win Rate:               {metrics.win_rate:>10.2%}")
    print(f"  Profit Factor:          {metrics.profit_factor:>10.2f}")
    print(f"  Average Win:            {metrics.average_win:>10.4f}")
    print(f"  Average Loss:           {metrics.average_loss:>10.4f}")
    print(f"  Largest Win:            {metrics.largest_win:>10.4f}")
    print(f"  Largest Loss:           {metrics.largest_loss:>10.4f}")
    print(f"  Consecutive Wins:       {metrics.consecutive_wins:>10d}")
    print(f"  Consecutive Losses:     {metrics.consecutive_losses:>10d}")
    
    print("\n💰 POSITION & COST METRICS:")
    print(f"  Total Turnover:         {metrics.total_turnover:>10.2f}")
    print(f"  Avg Position Size:      {metrics.average_position_size:>10.3f}")
    print(f"  Max Position Size:      {metrics.max_position_size:>10.3f}")
    print(f"  Total Transaction Costs: {metrics.total_transaction_costs:>10.4f}%")
    
    print("\n💳 CREDIT-SPECIFIC METRICS:")
    print(f"  Credit Beta:            {metrics.credit_beta:>10.3f}")
    print(f"  Spread Correlation:     {metrics.spread_correlation:>10.3f}")
    print(f"  Basis Volatility:       {metrics.basis_volatility:>10.4f}")
    
    print("\n📉 ADVANCED DRAWDOWN METRICS:")
    print(f"  Drawdown Episodes:      {metrics.num_drawdown_episodes:>10d}")
    print(f"  Avg Drawdown Depth:     {metrics.avg_drawdown_depth:>10.2f}%")
    print(f"  Max Drawdown Depth:     {metrics.max_drawdown_depth:>10.2f}%")
    print(f"  Underwater Ratio:       {metrics.underwater_ratio:>10.2%}")
    print(f"  Run-up Episodes:        {metrics.num_runup_episodes:>10d}")
    print(f"  Avg Run-up Length:      {metrics.avg_runup_length:>10.1f}")
    print(f"  Max Run-up Length:      {metrics.max_runup_length:>10d}")
    print(f"  Run-up Ratio:           {metrics.runup_ratio:>10.2%}")
    
    print("\n" + "=" * 60) 