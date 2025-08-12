"""
Plotting demonstration for the credit RV framework.

This script runs the credit RV framework and generates comprehensive visualizations
of all results using seaborn and matplotlib.
"""

import os
import numpy as np
from demo import run_demo
from plots import plot_all_results, create_comprehensive_dashboard
from metrics import print_metrics_summary


def main():
    """Run the credit RV framework and generate all plots."""
    print("Running Credit RV Framework...")
    
    # Run the demo to get results
    results = run_demo()
    
    print("\nGenerating comprehensive plots...")
    
    # Create plots directory
    plots_dir = "credit_rv_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate and save all plots
    plot_all_results(results, save_dir=plots_dir)
    
    # Create comprehensive dashboard
    dashboard_path = os.path.join(plots_dir, "comprehensive_dashboard.png")
    dashboard = create_comprehensive_dashboard(results, save_path=dashboard_path)
    
    print(f"\nAll plots saved to '{plots_dir}' directory:")
    print("• equity_curve.png - Equity curve with drawdown")
    print("• signals.png - Trading signals for all pairs")
    print("• positions.png - Position sizes for all strategies")
    print("• performance_metrics.png - PnL distribution and rolling metrics")
    print("• risk_metrics.png - VaR, CVaR, and risk analysis")
    print("• trading_metrics.png - Position and turnover analysis")
    print("• credit_metrics.png - Credit-specific metrics")
    print("• correlation_matrix.png - Signal correlation heatmap")
    print("• costs_analysis.png - Transaction costs analysis")
    print("• parameter_evolution.png - Parameter evolution over time")
    print("• comprehensive_dashboard.png - All-in-one dashboard")
    
    print(f"\nDashboard saved to: {dashboard_path}")
    
    # Display comprehensive metrics summary
    if 'metrics' in results:
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*60)
        print_metrics_summary(results['metrics'])
    
    # Additional analysis
    print(f"\n📊 ADDITIONAL ANALYSIS:")
    print(f"• Total Trading Days: {len(results['equity'])}")
    print(f"• Number of Strategies: {len(results['signals'])}")
    print(f"• Date Range: {results['equity'].index[0].strftime('%Y-%m-%d')} to {results['equity'].index[-1].strftime('%Y-%m-%d')}")
    
    # Strategy-specific analysis
    print(f"\n🎯 STRATEGY ANALYSIS:")
    for name, signal in results['signals'].items():
        signal_vol = signal.std()
        signal_mean = signal.mean()
        print(f"  {name}: Mean={signal_mean:.3f}, Vol={signal_vol:.3f}")
    
    # Position analysis
    portfolio_positions = results['positions']['portfolio']
    print(f"\n💰 POSITION ANALYSIS:")
    print(f"  Average Position Size: {portfolio_positions.abs().mean():.3f}")
    print(f"  Maximum Position Size: {portfolio_positions.abs().max():.3f}")
    print(f"  Position Volatility: {portfolio_positions.std():.3f}")
    
    # Cost analysis
    total_costs = results['costs']['portfolio'].sum()
    avg_daily_cost = results['costs']['portfolio'].mean()
    print(f"\n💸 COST ANALYSIS:")
    print(f"  Total Transaction Costs: {total_costs:.4f}%")
    print(f"  Average Daily Cost: {avg_daily_cost:.4f}%")
    print(f"  Cost Impact on Returns: {total_costs / results['metrics'].total_return:.2%}")
    
    return results


if __name__ == "__main__":
    main() 