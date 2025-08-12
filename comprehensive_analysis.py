"""
Comprehensive analysis script for the credit RV framework.

This script demonstrates all the enhanced capabilities including:
- Comprehensive performance metrics
- Risk analysis
- Trading analysis
- Credit-specific metrics
- Advanced visualizations
"""

import os
import numpy as np
import pandas as pd
from demo import run_demo
from plots import plot_all_results, create_comprehensive_dashboard
from metrics import calculate_all_metrics, print_metrics_summary, calculate_rolling_metrics


def run_comprehensive_analysis():
    """Run comprehensive analysis of the credit RV framework."""
    
    print("=" * 80)
    print("CREDIT RV FRAMEWORK - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Run the framework
    print("\n🚀 Running Credit RV Framework...")
    results = run_demo()
    
    # Generate comprehensive plots
    print("\n📊 Generating comprehensive visualizations...")
    plots_dir = "comprehensive_analysis_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate all plots
    plots = plot_all_results(results, save_dir=plots_dir)
    
    # Create comprehensive dashboard
    dashboard_path = os.path.join(plots_dir, "comprehensive_dashboard.png")
    dashboard = create_comprehensive_dashboard(results, save_path=dashboard_path)
    
    print(f"\n✅ All plots saved to '{plots_dir}' directory")
    
    # Additional analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS RESULTS")
    print("="*80)
    
    # 1. Basic Performance Summary
    print("\n📈 PERFORMANCE SUMMARY:")
    metrics = results['metrics']
    print(f"  Total Return:           {metrics.total_return:.2%}")
    print(f"  Annual Return:          {metrics.annual_return:.2%}")
    print(f"  Sharpe Ratio:           {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:          {metrics.sortino_ratio:.2f}")
    print(f"  Maximum Drawdown:       {metrics.max_drawdown:.2%}")
    print(f"  Win Rate:               {metrics.win_rate:.1%}")
    print(f"  Profit Factor:          {metrics.profit_factor:.2f}")
    
    # 2. Risk Analysis
    print("\n⚠️  RISK ANALYSIS:")
    print(f"  VaR (95%):              {metrics.var_95:.4f}")
    print(f"  CVaR (95%):             {metrics.cvar_95:.4f}")
    print(f"  VaR (99%):              {metrics.var_99:.4f}")
    print(f"  CVaR (99%):             {metrics.cvar_99:.4f}")
    print(f"  Max Drawdown Duration:  {metrics.max_drawdown_duration} days")
    print(f"  Consecutive Wins:       {metrics.consecutive_wins}")
    print(f"  Consecutive Losses:     {metrics.consecutive_losses}")
    
    # 3. Trading Analysis
    print("\n🎯 TRADING ANALYSIS:")
    print(f"  Total Turnover:         {metrics.total_turnover:.2f}")
    print(f"  Average Position Size:  {metrics.average_position_size:.3f}")
    print(f"  Max Position Size:      {metrics.max_position_size:.3f}")
    print(f"  Total Transaction Costs: {metrics.total_transaction_costs:.4f}%")
    print(f"  Average Win:            {metrics.average_win:.4f}")
    print(f"  Average Loss:           {metrics.average_loss:.4f}")
    print(f"  Largest Win:            {metrics.largest_win:.4f}")
    print(f"  Largest Loss:           {metrics.largest_loss:.4f}")
    
    # 4. Credit-Specific Analysis
    print("\n💳 CREDIT-SPECIFIC ANALYSIS:")
    print(f"  Credit Beta:            {metrics.credit_beta:.3f}")
    print(f"  Spread Correlation:     {metrics.spread_correlation:.3f}")
    print(f"  Basis Volatility:       {metrics.basis_volatility:.4f}")
    
    # 5. Strategy Analysis
    print("\n🔍 STRATEGY ANALYSIS:")
    signals = results['signals']
    for name, signal in signals.items():
        signal_stats = {
            'mean': signal.mean(),
            'std': signal.std(),
            'min': signal.min(),
            'max': signal.max(),
            'skew': signal.skew(),
            'kurt': signal.kurtosis()
        }
        print(f"  {name}:")
        print(f"    Mean: {signal_stats['mean']:.3f}, Std: {signal_stats['std']:.3f}")
        print(f"    Range: [{signal_stats['min']:.3f}, {signal_stats['max']:.3f}]")
        print(f"    Skew: {signal_stats['skew']:.3f}, Kurt: {signal_stats['kurt']:.3f}")
    
    # 6. Parameter Analysis
    print("\n📊 PARAMETER ANALYSIS:")
    theta = results['theta']
    for name, param_df in theta.items():
        param_vol = param_df.std().mean()
        param_mean = param_df.mean().mean()
        print(f"  {name}: Avg Vol={param_vol:.4f}, Avg Mean={param_mean:.4f}")
    
    # 7. Rolling Analysis
    print("\n📈 ROLLING METRICS ANALYSIS:")
    rolling_metrics = calculate_rolling_metrics(results['pnl'])
    
    # Calculate rolling statistics
    rolling_sharpe_mean = rolling_metrics['rolling_sharpe'].mean()
    rolling_sharpe_std = rolling_metrics['rolling_sharpe'].std()
    rolling_vol_mean = rolling_metrics['rolling_volatility'].mean()
    rolling_vol_std = rolling_metrics['rolling_volatility'].std()
    
    print(f"  Rolling Sharpe - Mean: {rolling_sharpe_mean:.3f}, Std: {rolling_sharpe_std:.3f}")
    print(f"  Rolling Vol - Mean: {rolling_vol_mean:.3f}, Std: {rolling_vol_std:.3f}")
    
    # 8. Correlation Analysis
    print("\n🔗 CORRELATION ANALYSIS:")
    signals_df = pd.DataFrame(signals)
    corr_matrix = signals_df.corr()
    
    # Find highest and lowest correlations
    corr_values = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_values.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                              corr_matrix.iloc[i, j]))
    
    if corr_values:
        corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"  Highest Correlation: {corr_values[0][0]} vs {corr_values[0][1]} = {corr_values[0][2]:.3f}")
        print(f"  Lowest Correlation: {corr_values[-1][0]} vs {corr_values[-1][1]} = {corr_values[-1][2]:.3f}")
    
    # 9. Cost Analysis
    print("\n💸 COST ANALYSIS:")
    costs = results['costs']
    total_costs = costs['portfolio'].sum()
    avg_daily_cost = costs['portfolio'].mean()
    cost_impact = total_costs / metrics.total_return if metrics.total_return != 0 else 0
    
    print(f"  Total Costs: {total_costs:.4f}%")
    print(f"  Average Daily Cost: {avg_daily_cost:.4f}%")
    print(f"  Cost Impact on Returns: {cost_impact:.2%}")
    
    # 10. Performance Attribution
    print("\n📊 PERFORMANCE ATTRIBUTION:")
    positions = results['positions']
    pnl_proxies = {}
    
    # Reconstruct PnL proxies for attribution
    for name in signals.keys():
        if name in results.get('pnl_proxies', {}):
            pnl_proxies[name] = results['pnl_proxies'][name]
    
    if pnl_proxies:
        strategy_contributions = {}
        for name in signals.keys():
            if name in positions.columns and name in pnl_proxies:
                strategy_pnl = positions[name] * pnl_proxies[name]
                strategy_contributions[name] = strategy_pnl.sum()
        
        total_contribution = sum(strategy_contributions.values())
        if total_contribution != 0:
            print("  Strategy Contributions:")
            for name, contribution in strategy_contributions.items():
                percentage = contribution / total_contribution * 100
                print(f"    {name}: {percentage:.1f}%")
    
    # 11. Summary Statistics
    print("\n📋 SUMMARY STATISTICS:")
    print(f"  Trading Period: {len(results['equity'])} days")
    print(f"  Date Range: {results['equity'].index[0].strftime('%Y-%m-%d')} to {results['equity'].index[-1].strftime('%Y-%m-%d')}")
    print(f"  Number of Strategies: {len(signals)}")
    print(f"  Number of Parameters: {sum(len(theta[name].columns) for name in theta.keys())}")
    
    # 12. Quality Metrics
    print("\n🎯 QUALITY METRICS:")
    # Signal quality (information ratio)
    signal_quality = {}
    for name, signal in signals.items():
        if name in pnl_proxies:
            ir = signal.corr(pnl_proxies[name])
            signal_quality[name] = ir
    
    if signal_quality:
        avg_ir = np.mean(list(signal_quality.values()))
        print(f"  Average Signal Information Ratio: {avg_ir:.3f}")
        for name, ir in signal_quality.items():
            print(f"    {name}: {ir:.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    run_comprehensive_analysis() 