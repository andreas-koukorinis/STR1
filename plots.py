"""
Plotting module for credit RV framework results.

This module provides comprehensive visualization capabilities for the credit RV framework,
including equity curves, trading signals, positions, performance metrics, and parameter
evolution over time.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Optional
from drawdown_analysis import calculate_advanced_drawdown_metrics

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_equity_curve(equity: pd.Series, 
                     title: str = "Credit RV Strategy - Equity Curve",
                     figsize: tuple = (12, 8)) -> plt.Figure:
    """Plot the cumulative equity curve with drawdown overlay.
    
    Args:
        equity: Cumulative equity series
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Plot equity curve
    ax1.plot(equity.index, equity.values, linewidth=2, color='darkblue', alpha=0.8)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Calculate and plot drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100
    
    ax2.fill_between(drawdown.index, drawdown.values, 0, 
                     color='red', alpha=0.3, label='Drawdown')
    ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_signals(signals: Dict[str, pd.Series], 
                figsize: tuple = (15, 10)) -> plt.Figure:
    """Plot trading signals for all pairs.
    
    Args:
        signals: Dictionary mapping pair names to signal series
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    n_pairs = len(signals)
    n_cols = 2
    n_rows = (n_pairs + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = sns.color_palette("husl", n_pairs)
    
    for i, (name, signal) in enumerate(signals.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        ax.plot(signal.index, signal.values, color=colors[i], linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{name} Signal', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z-Score', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_signal = signal.mean()
        std_signal = signal.std()
        ax.text(0.02, 0.98, f'Mean: {mean_signal:.3f}\nStd: {std_signal:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_pairs, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_positions(positions: pd.DataFrame, 
                  figsize: tuple = (15, 10)) -> plt.Figure:
    """Plot position sizes for all strategies and portfolio.
    
    Args:
        positions: DataFrame with position series for each strategy
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    n_strategies = len(positions.columns)
    n_cols = 2
    n_rows = (n_strategies + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = sns.color_palette("husl", n_strategies)
    
    for i, col in enumerate(positions.columns):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx]
        
        pos_series = positions[col]
        ax.plot(pos_series.index, pos_series.values, color=colors[i], linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Highlight portfolio position
        if col == 'portfolio':
            ax.set_title(f'{col} (Total)', fontsize=12, fontweight='bold', color='red')
        else:
            ax.set_title(f'{col} Position', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Position Size', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        max_pos = pos_series.max()
        min_pos = pos_series.min()
        ax.text(0.02, 0.98, f'Max: {max_pos:.3f}\nMin: {min_pos:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_strategies, n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_parameter_evolution(theta: Dict[str, pd.DataFrame], 
                           figsize: tuple = (15, 12)) -> plt.Figure:
    """Plot the evolution of estimated parameters over time.
    
    Args:
        theta: Dictionary mapping pair names to parameter DataFrames
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    n_pairs = len(theta)
    fig, axes = plt.subplots(n_pairs, 1, figsize=figsize)
    if n_pairs == 1:
        axes = [axes]
    
    colors = sns.color_palette("husl", 6)  # Max 6 parameters per pair
    
    for i, (name, param_df) in enumerate(theta.items()):
        ax = axes[i]
        
        for j, col in enumerate(param_df.columns):
            ax.plot(param_df.index, param_df[col].values, 
                   color=colors[j], linewidth=1.5, alpha=0.8, label=col)
        
        ax.set_title(f'{name} - Parameter Evolution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Parameter Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if i < n_pairs - 1:
            ax.set_xlabel('')
    
    axes[-1].set_xlabel('Date', fontsize=12)
    plt.tight_layout()
    return fig


def plot_performance_metrics(pnl: pd.Series, 
                           figsize: tuple = (15, 10)) -> plt.Figure:
    """Plot comprehensive performance metrics.
    
    Args:
        pnl: Daily PnL series
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. PnL distribution
    ax1.hist(pnl.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(pnl.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pnl.mean():.4f}')
    ax1.set_title('PnL Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Daily PnL')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling Sharpe ratio
    rolling_sharpe = pnl.rolling(252).mean() / pnl.rolling(252).std() * np.sqrt(252)
    ax2.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Rolling Sharpe Ratio (252-day)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling volatility
    rolling_vol = pnl.rolling(252).std() * np.sqrt(252)
    ax3.plot(rolling_vol.index, rolling_vol.values, color='orange', linewidth=1.5)
    ax3.set_title('Rolling Volatility (252-day)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Annualized Volatility')
    ax3.grid(True, alpha=0.3)
    
    # 4. Rolling drawdown
    equity = (1 + pnl.fillna(0.0)).cumprod()
    rolling_max = equity.expanding().max()
    rolling_dd = (equity - rolling_max) / rolling_max * 100
    ax4.fill_between(rolling_dd.index, rolling_dd.values, 0, 
                     color='red', alpha=0.3, label='Drawdown')
    ax4.plot(rolling_dd.index, rolling_dd.values, color='red', linewidth=1)
    ax4.set_title('Rolling Drawdown', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Drawdown (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(signals: Dict[str, pd.Series], 
                          figsize: tuple = (10, 8)) -> plt.Figure:
    """Plot correlation matrix of trading signals.
    
    Args:
        signals: Dictionary mapping pair names to signal series
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Create DataFrame from signals
    signals_df = pd.DataFrame(signals)
    
    # Calculate correlation matrix
    corr_matrix = signals_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    ax.set_title('Signal Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_costs_analysis(costs: pd.DataFrame, 
                       figsize: tuple = (15, 8)) -> plt.Figure:
    """Plot transaction costs analysis.
    
    Args:
        costs: DataFrame with cost series for each strategy
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Cumulative costs over time
    cumulative_costs = costs.cumsum()
    for col in costs.columns:
        ax1.plot(cumulative_costs.index, cumulative_costs[col].values, 
                linewidth=1.5, alpha=0.8, label=col)
    
    ax1.set_title('Cumulative Transaction Costs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Cost (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cost distribution
    costs_flat = costs.values.flatten()
    costs_flat = costs_flat[costs_flat > 0]  # Only positive costs
    
    ax2.hist(costs_flat, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Transaction Cost Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Daily Cost (%)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_risk_metrics(pnl: pd.Series, figsize: tuple = (15, 10)) -> plt.Figure:
    """Plot comprehensive risk metrics.
    
    Args:
        pnl: Daily PnL series
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. VaR and CVaR
    var_95 = np.percentile(pnl, 5)
    var_99 = np.percentile(pnl, 1)
    cvar_95 = pnl[pnl <= var_95].mean()
    cvar_99 = pnl[pnl <= var_99].mean()
    
    ax1.hist(pnl.values, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.4f}')
    ax1.axvline(var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.4f}')
    ax1.axvline(cvar_95, color='darkorange', linestyle=':', linewidth=2, label=f'CVaR 95%: {cvar_95:.4f}')
    ax1.axvline(cvar_99, color='darkred', linestyle=':', linewidth=2, label=f'CVaR 99%: {cvar_99:.4f}')
    ax1.set_title('Value at Risk Analysis', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Daily PnL')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling VaR
    rolling_var_95 = pnl.rolling(252).quantile(0.05)
    rolling_var_99 = pnl.rolling(252).quantile(0.01)
    ax2.plot(rolling_var_95.index, rolling_var_95.values, color='orange', linewidth=1.5, label='VaR 95%')
    ax2.plot(rolling_var_99.index, rolling_var_99.values, color='red', linewidth=1.5, label='VaR 99%')
    ax2.set_title('Rolling Value at Risk (252-day)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('VaR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win/Loss analysis
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    
    win_rate = len(wins) / len(pnl)
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 0
    
    ax3.bar(['Wins', 'Losses'], [len(wins), len(losses)], 
            color=['green', 'red'], alpha=0.7)
    ax3.set_title(f'Win/Loss Analysis (Win Rate: {win_rate:.1%})', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Trades')
    ax3.text(0.5, 0.9, f'Profit Factor: {profit_factor:.2f}', 
             transform=ax3.transAxes, ha='center', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Consecutive wins/losses
    consecutive_wins = calculate_max_consecutive(pnl > 0)
    consecutive_losses = calculate_max_consecutive(pnl < 0)
    
    ax4.bar(['Max Consecutive Wins', 'Max Consecutive Losses'], 
            [consecutive_wins, consecutive_losses], 
            color=['lightgreen', 'lightcoral'], alpha=0.7)
    ax4.set_title('Consecutive Trades', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Consecutive Trades')
    
    plt.tight_layout()
    return fig


def plot_trading_metrics(positions: pd.DataFrame, costs: pd.DataFrame, 
                        figsize: tuple = (15, 10)) -> plt.Figure:
    """Plot trading-specific metrics.
    
    Args:
        positions: Position DataFrame
        costs: Transaction costs DataFrame
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Position size distribution
    portfolio_positions = positions['portfolio'] if 'portfolio' in positions.columns else positions.sum(axis=1)
    ax1.hist(portfolio_positions.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(portfolio_positions.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {portfolio_positions.mean():.3f}')
    ax1.set_title('Position Size Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position Size')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Turnover analysis
    turnover = portfolio_positions.diff().abs()
    ax2.plot(turnover.index, turnover.values, color='purple', linewidth=1, alpha=0.7)
    ax2.set_title('Daily Turnover', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Turnover')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative transaction costs by strategy
    cumulative_costs = costs.cumsum()
    for col in costs.columns:
        ax3.plot(cumulative_costs.index, cumulative_costs[col].values, 
                linewidth=1.5, alpha=0.8, label=col)
    ax3.set_title('Cumulative Transaction Costs by Strategy', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Cost (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cost vs Turnover scatter
    ax4.scatter(turnover.values, costs['portfolio'].values if 'portfolio' in costs.columns else costs.sum(axis=1).values,
               alpha=0.6, color='orange')
    ax4.set_title('Cost vs Turnover', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Turnover')
    ax4.set_ylabel('Transaction Cost (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_credit_metrics(signals: Dict[str, pd.Series], theta: Dict[str, pd.DataFrame],
                       figsize: tuple = (15, 10)) -> plt.Figure:
    """Plot credit-specific metrics.
    
    Args:
        signals: Dictionary of trading signals
        theta: Dictionary of parameter estimates
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Signal correlation heatmap
    if len(signals) > 1:
        signals_df = pd.DataFrame(signals)
        corr_matrix = signals_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Signal Correlations', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Need multiple signals\nfor correlation analysis', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Signal Correlations', fontsize=12, fontweight='bold')
    
    # 2. Parameter stability (coefficient of variation)
    if theta:
        param_stability = {}
        for name, param_df in theta.items():
            cv = param_df.std() / param_df.mean().abs()
            param_stability[name] = cv.mean()
        
        strategies = list(param_stability.keys())
        stability_values = list(param_stability.values())
        ax2.bar(strategies, stability_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Parameter Stability (CV)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Signal distribution by strategy
    for i, (name, signal) in enumerate(signals.items()):
        ax3.hist(signal.values, bins=30, alpha=0.6, label=name, density=True)
    ax3.set_title('Signal Distribution by Strategy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Signal Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter evolution summary
    if theta:
        all_params = []
        for param_df in theta.values():
            all_params.append(param_df)
        
        if all_params:
            combined_params = pd.concat(all_params, axis=1)
            param_vol = combined_params.rolling(60).std().mean(axis=1)
            ax4.plot(param_vol.index, param_vol.values, color='green', linewidth=1.5)
            ax4.set_title('Average Parameter Volatility (60-day)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Parameter Volatility')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_advanced_drawdown_analysis(pnl: pd.Series, equity: pd.Series,
                                   figsize: tuple = (15, 12)) -> plt.Figure:
    """Plot advanced drawdown and run-up analysis.
    
    Args:
        pnl: Daily PnL series
        equity: Cumulative equity series
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Calculate advanced drawdown metrics
    advanced_metrics = calculate_advanced_drawdown_metrics(pnl, equity)
    dd_analysis = advanced_metrics['drawdown_analysis']
    ru_analysis = advanced_metrics['runup_analysis']
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. Drawdown episodes timeline
    if len(dd_analysis['episodes']) > 0:
        episodes = dd_analysis['episodes']
        depths = [ep['depth_percent'] for ep in episodes]
        durations = [ep['peak_to_recovery_duration'] for ep in episodes if ep['peak_to_recovery_duration'] is not None]
        
        axes[0, 0].bar(range(len(episodes)), depths, color='red', alpha=0.7)
        axes[0, 0].set_title('Drawdown Episodes by Depth', fontweight='bold')
        axes[0, 0].set_ylabel('Depth (%)')
        axes[0, 0].set_xlabel('Episode Number')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown duration distribution
        if durations:
            axes[0, 1].hist(durations, bins=min(10, len(durations)), color='red', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Drawdown Duration Distribution', fontweight='bold')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_xlabel('Duration (periods)')
            axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Drawdown Episodes', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Drawdown Episodes by Depth', fontweight='bold')
        axes[0, 1].text(0.5, 0.5, 'No Drawdown Episodes', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Drawdown Duration Distribution', fontweight='bold')
    
    # 3. Run-up episodes timeline
    if len(ru_analysis['episodes']) > 0:
        episodes = ru_analysis['episodes']
        lengths = [ep['length'] for ep in episodes]
        
        axes[1, 0].bar(range(len(episodes)), lengths, color='green', alpha=0.7)
        axes[1, 0].set_title('Run-up Episodes by Length', fontweight='bold')
        axes[1, 0].set_ylabel('Length (periods)')
        axes[1, 0].set_xlabel('Episode Number')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Run-up length distribution
        axes[1, 1].hist(lengths, bins=min(10, len(lengths)), color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Run-up Length Distribution', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xlabel('Length (periods)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Run-up Episodes', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Run-up Episodes by Length', fontweight='bold')
        axes[1, 1].text(0.5, 0.5, 'No Run-up Episodes', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Run-up Length Distribution', fontweight='bold')
    
    # 5. Underwater vs Run-up time ratio
    underwater_ratio = advanced_metrics['drawdown_stats']['underwater_ratio']
    runup_ratio = advanced_metrics['runup_stats']['runup_ratio']
    neutral_ratio = 1 - underwater_ratio - runup_ratio
    
    ratios = [underwater_ratio, runup_ratio, neutral_ratio]
    labels = ['Underwater', 'Run-up', 'Neutral']
    colors = ['red', 'green', 'gray']
    
    axes[2, 0].pie(ratios, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[2, 0].set_title('Time Distribution: Underwater vs Run-up', fontweight='bold')
    
    # 6. Trade analysis
    trade_analysis = advanced_metrics['trade_analysis']
    trade_counts = [trade_analysis['wins'], trade_analysis['losses'], trade_analysis['breakeven']]
    trade_labels = ['Wins', 'Losses', 'Breakeven']
    trade_colors = ['green', 'red', 'orange']
    
    axes[2, 1].pie(trade_counts, labels=trade_labels, colors=trade_colors, autopct='%1.1f%%', startangle=90)
    axes[2, 1].set_title(f'Trade Analysis (Total: {trade_analysis["total"]})', fontweight='bold')
    
    plt.tight_layout()
    return fig


def calculate_max_consecutive(condition: pd.Series) -> int:
    """Calculate maximum consecutive occurrences of a condition."""
    max_consecutive = 0
    current_consecutive = 0
    
    for value in condition:
        if value:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def create_comprehensive_dashboard(results: Dict, 
                                 save_path: Optional[str] = None,
                                 figsize: tuple = (20, 24)) -> plt.Figure:
    """Create a comprehensive dashboard with all key plots.
    
    Args:
        results: Dictionary containing all results from run_demo()
        save_path: Optional path to save the dashboard
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
    
    # 1. Equity curve (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    equity = results['equity']
    ax1.plot(equity.index, equity.values, linewidth=2, color='darkblue', alpha=0.8)
    ax1.set_title('Credit RV Strategy - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True, alpha=0.3)
    
    # 2. Signals (rows 1-2, left half)
    ax2 = fig.add_subplot(gs[1:3, :2])
    signals = results['signals']
    for name, signal in signals.items():
        ax2.plot(signal.index, signal.values, linewidth=1, alpha=0.7, label=name)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Trading Signals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Positions (rows 1-2, right half)
    ax3 = fig.add_subplot(gs[1:3, 2:])
    positions = results['positions']
    for col in positions.columns:
        if col != 'portfolio':
            ax3.plot(positions.index, positions[col].values, linewidth=1, alpha=0.7, label=col)
    ax3.plot(positions.index, positions['portfolio'].values, linewidth=2, color='red', label='Portfolio')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Position Sizes', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Position Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. PnL distribution (row 3, left)
    ax4 = fig.add_subplot(gs[3, 0])
    pnl = results['pnl']
    ax4.hist(pnl.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(pnl.mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_title('PnL Distribution', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Daily PnL')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Sharpe (row 3, right)
    ax5 = fig.add_subplot(gs[3, 1])
    rolling_sharpe = pnl.rolling(252).mean() / pnl.rolling(252).std() * np.sqrt(252)
    ax5.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=1.5)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_title('Rolling Sharpe Ratio', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation matrix (row 4, left half)
    ax6 = fig.add_subplot(gs[4, :2])
    signals_df = pd.DataFrame(signals)
    corr_matrix = signals_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6)
    ax6.set_title('Signal Correlations', fontsize=10, fontweight='bold')
    
    # 7. Costs analysis (row 4, right half)
    ax7 = fig.add_subplot(gs[4, 2:])
    costs = results['costs']
    cumulative_costs = costs.cumsum()
    for col in costs.columns:
        ax7.plot(cumulative_costs.index, cumulative_costs[col].values, 
                linewidth=1.5, alpha=0.8, label=col)
    ax7.set_title('Cumulative Transaction Costs', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Cumulative Cost (%)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance summary (row 5, full width)
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('off')
    
    # Calculate performance metrics
    ann_ret = equity.iloc[-1] ** (252/len(equity)) - 1
    ann_vol = pnl.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    maxdd = (equity / equity.cummax() - 1).min()
    total_cost = costs['portfolio'].sum()
    
    summary_text = f"""
    Performance Summary:
    • Annual Return: {ann_ret:.2%}
    • Annual Volatility: {ann_vol:.2%}
    • Sharpe Ratio: {sharpe:.2f}
    • Maximum Drawdown: {maxdd:.2%}
    • Total Transaction Costs: {total_cost:.4f}%
    • Number of Trading Days: {len(equity)}
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_all_results(results: Dict, save_dir: Optional[str] = None) -> None:
    """Generate and optionally save all plots for the credit RV results.
    
    Args:
        results: Dictionary containing all results from run_demo()
        save_dir: Optional directory to save plots
    """
    plots = {}
    
    # Generate all plots
    plots['equity_curve'] = plot_equity_curve(results['equity'])
    plots['signals'] = plot_signals(results['signals'])
    plots['positions'] = plot_positions(results['positions'])
    plots['performance_metrics'] = plot_performance_metrics(results['pnl'])
    plots['risk_metrics'] = plot_risk_metrics(results['pnl'])
    plots['trading_metrics'] = plot_trading_metrics(results['positions'], results['costs'])
    plots['credit_metrics'] = plot_credit_metrics(results['signals'], results['theta'])
    plots['correlation_matrix'] = plot_correlation_matrix(results['signals'])
    plots['costs_analysis'] = plot_costs_analysis(results['costs'])
    plots['parameter_evolution'] = plot_parameter_evolution(results['theta'])
    plots['advanced_drawdown'] = plot_advanced_drawdown_analysis(results['pnl'], results['equity'])
    plots['dashboard'] = create_comprehensive_dashboard(results)
    
    # Save plots if directory provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, fig in plots.items():
            fig.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"All plots saved to {save_dir}")
    else:
        plt.show()
    
    return plots 