# Credit RV Framework

A comprehensive multi-ETF, relative-value framework for trading ETF pairs and ETF basket vs CDX strategies using time-varying parameter regression with discounted Kalman filtering and sophisticated return decomposition.

## 🚀 Enhanced Features

### **Comprehensive Performance Metrics**
- **Basic Metrics**: Total return, annual return, volatility, Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk Metrics**: VaR (95% & 99%), CVaR, maximum drawdown, drawdown duration
- **Trading Metrics**: Win rate, profit factor, average win/loss, consecutive wins/losses
- **Credit Metrics**: Credit beta, spread correlation, basis volatility
- **Transaction Metrics**: Turnover, position sizes, transaction costs

### **Advanced Analysis Capabilities**
- **Risk Analysis**: Comprehensive VaR/CVaR analysis with rolling metrics
- **Trading Analysis**: Position distribution, turnover analysis, cost analysis
- **Credit Analysis**: Signal correlations, parameter stability, basis analysis
- **Performance Attribution**: Strategy contribution analysis
- **Quality Metrics**: Information ratios, signal quality assessment
- **Advanced Drawdown Analysis**: Episode analysis, run-up periods, underwater ratios, trade counting
- **Enhanced Kalman Filter**: Discounted Kalman and FLS (Flexible Least Squares) modes
- **Return Decomposition**: Daily credit return decomposition into spread, carry, roll, and residual components

### **Enhanced Visualizations**
- **Equity Curves**: With drawdown overlay
- **Risk Metrics**: VaR/CVaR distributions, rolling risk metrics
- **Trading Metrics**: Position analysis, turnover analysis, cost analysis
- **Credit Metrics**: Signal correlations, parameter evolution
- **Return Decomposition**: Component analysis and attribution
- **Comprehensive Dashboard**: All-in-one visualization

## 📁 Project Structure

### Core Modules

- **`utils.py`** - Specialized utility functions (realized volatility, performance metrics)
- **`kalman.py`** - Enhanced time-varying parameter Kalman filter with FLS mode
- **`features.py`** - Comprehensive feature engineering for credit RV strategies
- **`engine.py`** - Core signal generation engine with return decomposition capabilities
- **`specs.py`** - Data classes for pair specifications and configuration
- **`portfolio.py`** - Portfolio management, position sizing, and risk control
- **`data.py`** - Synthetic data generation for demonstration
- **`decomposition.py`** - Daily return decomposition into spread, carry, roll components

### Analysis Modules

- **`metrics.py`** - Comprehensive performance and risk metrics calculation
- **`plots.py`** - Advanced visualization capabilities with seaborn
- **`comprehensive_analysis.py`** - Complete analysis workflow
- **`drawdown_analysis.py`** - Advanced drawdown and run-up episode analysis

### Demo Scripts

- **`demo.py`** - Complete demonstration script with enhanced capabilities
- **`plot_demo.py`** - Plotting demonstration with enhanced metrics
- **`comprehensive_analysis.py`** - Full analysis with detailed reporting

## 🎯 Usage Examples

### Basic Usage
```python
from engine import PairEngine, ETFETFSpec, ETFCDXSpec, SignalConfig, PortfolioConfig
from decomposition import decompose_basket_credit

# Create engine
engine = PairEngine(assets, cdx)

# Define pairs with enhanced configuration
pairs = [
    ETFETFSpec(long="HYG", short="JNK", name="HYG_vs_JNK"),
    ETFCDXSpec(etfs=["HYG", "JNK"], weights=[0.6, 0.4], cdx_family="HY")
]

# Run analysis with feature masking
signal_config = SignalConfig(map_style="tanh", z_entry=0.8, z_exit=0.2)
results = engine.run_etf_pair(pairs[0], feature_mask={"z_carry": False})
```

### Return Decomposition
```python
# For basket vs CDX pairs, get return decomposition
basket_result = engine.run_basket_cdx(pairs[1])
decomposition = basket_result['decomp']

# Decomposition components:
# - decomp_spread_return: Return from OAS changes
# - decomp_carry: Return from current OAS level  
# - decomp_roll: Return from DV01 changes
# - decomp_residual: Unexplained component
# - decomp_div_borrow: Dividend yield minus borrow cost
```

### Comprehensive Analysis
```python
from demo import run_demo
from metrics import print_metrics_summary

# Run complete analysis with decomposition
results = run_demo()

# Display comprehensive metrics
print_metrics_summary(results['metrics'])

# Access decomposition for basket strategies
for name, decomp in results['decompositions'].items():
    print(f"{name} decomposition components: {list(decomp.columns)}")
```

### Advanced Plotting
```python
from plots import plot_all_results
from comprehensive_analysis import run_comprehensive_analysis

# Generate all plots including decomposition
plot_all_results(results, save_dir="plots")

# Run comprehensive analysis
run_comprehensive_analysis()
```

## 📊 Performance Metrics

The framework now provides **25+ comprehensive metrics**:

### 📈 Basic Performance
- Total Return, Annual Return, Annual Volatility
- Sharpe Ratio, Sortino Ratio, Calmar Ratio

### ⚠️ Risk Metrics
- Maximum Drawdown, Drawdown Duration
- VaR (95% & 99%), CVaR (95% & 99%)
- Consecutive Wins/Losses

### 🎯 Trading Metrics
- Win Rate, Profit Factor
- Average Win/Loss, Largest Win/Loss
- Total Turnover, Position Sizes

### 💳 Credit-Specific
- Credit Beta, Spread Correlation
- Basis Volatility, Parameter Stability

### 💰 Transaction Metrics
- Transaction Costs, Cost Impact
- Position Distribution, Turnover Analysis

## 📈 Visualization Features

### **11 Different Plot Types:**
1. **Equity Curve** - Cumulative returns with drawdown
2. **Signals** - Trading signals for all pairs
3. **Positions** - Position sizes and portfolio allocation
4. **Performance Metrics** - PnL distribution and rolling metrics
5. **Risk Metrics** - VaR/CVaR analysis and risk distributions
6. **Trading Metrics** - Position and turnover analysis
7. **Credit Metrics** - Credit-specific correlations and stability
8. **Correlation Matrix** - Signal correlation heatmap
9. **Costs Analysis** - Transaction cost analysis
10. **Parameter Evolution** - Parameter stability over time
11. **Comprehensive Dashboard** - All-in-one visualization

## 🔧 Installation & Requirements

### Required Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Optional Dependencies (for enhanced features)
```bash
pip install scipy statsmodels numba tqdm
```

## 📋 Data Requirements

### ETF Data
Each ETF DataFrame should contain:
- `close`, `nav` - Price and NAV data
- `dividend_yield`, `borrow_fee` - Carry components
- `credit_dv01`, `duration` - Risk metrics
- `volume`, `so`, `basket_dv01` - Liquidity and flow metrics
- `rates_excess_ret` - Rate-hedged returns (optional)

### CDX Data
Each CDX DataFrame should contain:
- `spread` - Spread in basis points
- `dv01` - Dollar duration per 1bp per 1 notional

## 🚀 Quick Start

### Run Basic Demo
```bash
python demo.py
```

### Generate All Plots
```bash
python plot_demo.py
```

### Run Comprehensive Analysis
```bash
python comprehensive_analysis.py
```

## 📊 Output Examples

### Performance Summary
```
📈 BASIC PERFORMANCE METRICS:
  Total Return:               15.23%
  Annual Return:              12.45%
  Annual Volatility:           8.91%
  Sharpe Ratio:                1.40
  Sortino Ratio:               1.85
  Calmar Ratio:                2.34

⚠️  RISK METRICS:
  Maximum Drawdown:           -5.32%
  Max DD Duration:              45 days
  VaR (95%):                -0.0234
  CVaR (95%):               -0.0312
  VaR (99%):                -0.0345
  CVaR (99%):               -0.0412

🎯 TRADING METRICS:
  Win Rate:                   58.3%
  Profit Factor:               1.85
  Average Win:                 0.0234
  Average Loss:               -0.0156
```

### Generated Plots
- `equity_curve.png` - Equity curve with drawdown
- `risk_metrics.png` - VaR/CVaR analysis
- `trading_metrics.png` - Position and turnover analysis
- `credit_metrics.png` - Credit-specific correlations
- `comprehensive_dashboard.png` - All-in-one dashboard

## 🔬 Advanced Features

### **Rolling Analysis**
- Rolling Sharpe ratios, volatility, and drawdown
- Time-varying risk metrics
- Parameter stability analysis

### **Performance Attribution**
- Strategy contribution analysis
- Risk decomposition
- Cost impact analysis

### **Quality Assessment**
- Signal information ratios
- Parameter stability metrics
- Correlation analysis

## 📈 Key Benefits

1. **Comprehensive Analysis** - 25+ performance and risk metrics
2. **Advanced Visualizations** - 11 different plot types with seaborn
3. **Credit-Specific Metrics** - Tailored for credit RV strategies
4. **Risk Management** - VaR/CVaR and drawdown analysis
5. **Transaction Analysis** - Cost and turnover analysis
6. **Modular Design** - Easy to extend and customize
7. **Professional Output** - Publication-ready plots and reports

## 🤝 Contributing

The framework is designed to be easily extensible. Key areas for enhancement:
- Additional risk metrics
- More sophisticated position sizing
- Real-time data connectors
- Additional visualization types
- Backtesting capabilities

## 📄 License

This project is open source and available under the MIT License. 