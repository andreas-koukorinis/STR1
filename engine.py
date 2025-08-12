"""
rv_engine.py — Signal engine for ETF↔ETF and Basket↔CDX
=======================================================

This module provides the core signal generation engine for credit RV strategies,
supporting both ETF-ETF pairs and ETF basket vs CDX pairs with sophisticated
return decomposition capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from kalman import TVPKalman
from features import (
    rate_hedged_return, nav_premium_z, daf_z, carry_daily_z,
    cdx_basis_to_underlying, robust_z
)
from decomposition import decompose_basket_credit

__all__ = [
    "ETFETFSpec", "ETFCDXSpec",
    "SignalConfig", "PortfolioConfig",
    "PairEngine",
]


# ----------------------------
# Specs & Configs
# ----------------------------
@dataclass
class ETFETFSpec:
    """Specification for ETF vs ETF pair trading.
    
    Attributes:
        long: Ticker symbol for long ETF
        short: Ticker symbol for short ETF
        name: Optional name for the pair (default: auto-generated)
    """
    long: str
    short: str
    name: str = ""


@dataclass
class ETFCDXSpec:
    """Specification for ETF basket vs CDX pair trading.
    
    Attributes:
        etfs: List of ETF ticker symbols in the basket
        weights: Optional weights for each ETF (default: DV01-weighted)
        cdx_family: CDX family name (e.g., "HY", "IG")
        name: Optional name for the pair (default: auto-generated)
    """
    etfs: List[str]
    weights: Optional[List[float]] = None
    cdx_family: str = "HY"
    name: str = ""


@dataclass
class SignalConfig:
    """Configuration for signal generation and mapping.
    
    Attributes:
        map_style: Signal mapping style ("tanh" or "threshold")
        tanh_scale: Scaling factor for tanh mapping
        z_entry: Z-score threshold for entry signals
        z_exit: Z-score threshold for exit signals
        max_abs_weight: Maximum absolute position weight
    """
    map_style: str = "tanh"  # "tanh" or "threshold"
    tanh_scale: float = 1.0
    z_entry: float = 0.8
    z_exit: float = 0.2
    max_abs_weight: float = 2.5


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management and risk control.
    
    Attributes:
        target_vol: Target annualized volatility
        vol_hl: Half-life for volatility estimation
        etf_tc_bps: ETF transaction costs in basis points
        cdx_tc_bpDv01: CDX transaction costs in bp per DV01
    """
    target_vol: float = 0.12
    vol_hl: float = 20.0
    etf_tc_bps: float = 1.0
    cdx_tc_bpDv01: float = 0.2


# ----------------------------
# Engine
# ----------------------------
class PairEngine:
    """Build signals via TVP Kalman; return z, θ, X, y, and (for baskets) a credit return decomposition.
    
    This class implements the core signal generation engine for credit RV strategies.
    It supports both ETF-ETF pairs and ETF basket vs CDX pairs with sophisticated
    return decomposition capabilities.
    
    Attributes:
        assets: Dictionary mapping ETF ticker symbols to DataFrames with asset data
        cdx: Dictionary mapping CDX family names to DataFrames with CDX data
        holdings: Optional DataFrame with holdings data for basis calculations
    """
    
    def __init__(self, 
                 assets: Dict[str, pd.DataFrame], 
                 cdx: Dict[str, pd.DataFrame], 
                 holdings: Optional[pd.DataFrame] = None):
        """Initialize the pair engine.
        
        Args:
            assets: Dictionary mapping ETF ticker symbols to DataFrames with asset data
            cdx: Dictionary mapping CDX family names to DataFrames with CDX data
            holdings: Optional DataFrame with holdings data for basis calculations
        """
        self.assets = assets
        self.cdx = cdx
        self.holdings = holdings

    def _apply_mask(self, X: pd.DataFrame, feature_mask: Optional[Dict[str, bool]]) -> pd.DataFrame:
        """Apply feature mask to control which features are used in the model.
        
        Args:
            X: Feature DataFrame
            feature_mask: Dictionary mapping feature names to boolean flags
            
        Returns:
            Masked feature DataFrame
        """
        if not feature_mask:
            return X
        X = X.copy()
        for col, on in feature_mask.items():
            if (col in X.columns) and (not on):
                X[col] = 0.0
        return X

    # ---------- ETF↔ETF ----------
    def run_etf_pair(self, spec: ETFETFSpec,
                     delta: Tuple[float, ...] = (0.995, 0.985, 0.97, 0.97, 0.98),
                     R_init: float = 1e-6,
                     feature_mask: Optional[Dict[str, bool]] = None) -> Dict[str, pd.Series]:
        """Run ETF vs ETF pair analysis.
        
        This method analyzes a long-short pair between two ETFs using time-varying
        parameter regression. The regression model is:
        
        r_long = α + β*r_short + γ*ΔNAV_diff + δ*DAF_diff + ε*Carry_diff + noise
        
        where the parameters are estimated using a discounted Kalman filter.
        
        Args:
            spec: ETF pair specification with long and short tickers
            delta: Discount factors for each parameter in the regression
            R_init: Initial observation noise variance for the Kalman filter
            feature_mask: Optional mask to control which features are used
            
        Returns:
            Dictionary containing:
                - pair_name: Name of the pair
                - z_pair: Trading signal (standardized regression residual)
                - theta: Time series of estimated parameters
                - X: Feature matrix used in regression
                - pair_ret_next: Next-day pair return
                - pred_pair_ret_next: Predicted next-day pair return
                
        Example:
            >>> spec = ETFETFSpec(long="HYG", short="JNK")
            >>> result = engine.run_etf_pair(spec)
            >>> signal = result['z_pair']
            >>> print(f"Signal mean: {signal.mean():.3f}")
        """
        A = self.assets[spec.long].copy(); A.name = spec.long
        B = self.assets[spec.short].copy(); B.name = spec.short
        idx = A.index.intersection(B.index)
        A = A.loc[idx]; B = B.loc[idx]
        
        rA = rate_hedged_return(A); rB = rate_hedged_return(B)
        
        # Build feature matrix
        X = pd.DataFrame(index=idx)
        X["const"] = 1.0
        X["rB"] = rB
        X["z_nav"] = (nav_premium_z(A) - nav_premium_z(B))
        X["z_daf"] = (daf_z(A) - daf_z(B))
        X["z_carry"] = robust_z((A.get("dividend_yield", 0.0)/252 - A.get("borrow_fee", 0.0)/252
                                   - (B.get("dividend_yield", 0.0)/252 - B.get("borrow_fee", 0.0)/252)).fillna(0.0), 63.0)
        
        X = self._apply_mask(X, feature_mask)
        y = rA
        
        # Run Kalman filter
        kf = TVPKalman(k=X.shape[1], F=np.eye(X.shape[1]), R=R_init, delta=np.array(delta), mode="discount")
        z_list, theta_list = [], []
        
        for t, (_, row) in enumerate(X.iterrows()):
            z, S, theta = kf.step(row.values.astype(float), float(y.iloc[t]))
            z_list.append(z); theta_list.append(theta)
            
        theta_df = pd.DataFrame(theta_list, index=idx, columns=X.columns)
        
        # Calculate next-day pair return and prediction
        beta = theta_df["rB"].shift(1)
        pair_ret_next = (rA.shift(-1) - beta * rB.shift(-1)).rename("pair_ret_next")
        
        # Calibrate prediction scale λ_t via rolling OLS of next‑day pair return on z_t
        lam = rolling_beta(pair_ret_next, pd.Series(z_list, index=idx).shift(1), win=120)
        yhat_next = (lam * pd.Series(z_list, index=idx)).rename("yhat_pair_next")
        
        return {
            "pair_name": spec.name or f"{spec.long}_vs_{spec.short}",
            "z_pair": pd.Series(z_list, index=idx),
            "theta": theta_df,
            "X": X,
            "pair_ret_next": pair_ret_next,
            "pred_pair_ret_next": yhat_next,
        }

    # ---------- Basket↔CDX ----------
    def run_basket_cdx(self, spec: ETFCDXSpec,
                       delta: Tuple[float, ...] = (0.995, 0.985, 0.97, 0.97, 0.98, 0.98),
                       R_init: float = 1e-6,
                       feature_mask: Optional[Dict[str, bool]] = None) -> Dict[str, pd.Series]:
        """Run ETF basket vs CDX pair analysis.
        
        This method analyzes a basket of ETFs against a CDX index using time-varying
        parameter regression. The regression model is:
        
        r_basket = α + β*ΔCDX + γ*NAV_basket + δ*DAF_basket + ε*Carry_basket + ζ*Basis_basket + noise
        
        where the parameters are estimated using a discounted Kalman filter.
        
        Args:
            spec: ETF basket vs CDX specification with ETF list and CDX family
            delta: Discount factors for each parameter in the regression
            R_init: Initial observation noise variance for the Kalman filter
            feature_mask: Optional mask to control which features are used
            
        Returns:
            Dictionary containing:
                - pair_name: Name of the pair
                - z_pair: Trading signal (standardized regression residual)
                - theta: Time series of estimated parameters
                - X: Feature matrix used in regression
                - pair_ret_next: Next-day pair return
                - pred_pair_ret_next: Predicted next-day pair return
                - decomp: Return decomposition components
                
        Example:
            >>> spec = ETFCDXSpec(etfs=["HYG", "JNK"], cdx_family="HY")
            >>> result = engine.run_basket_cdx(spec)
            >>> signal = result['z_pair']
            >>> print(f"Signal mean: {signal.mean():.3f}")
        """
        ets = [self.assets[s].copy() for s in spec.etfs]
        for df in ets: 
            df.sort_index(inplace=True)
        idx = ets[0].index
        for df in ets[1:]: 
            idx = idx.intersection(df.index)
        ets = [df.loc[idx] for df in ets]
        
        # Calculate basket weights
        if spec.weights is not None:
            w = np.array(spec.weights, dtype=float)
            w = w / np.sum(np.abs(w))
        else:
            # DV01-weighted if available, else equal
            dv01s = np.array([df.get("credit_dv01", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0).mean() for df in ets])
            w = dv01s / np.sum(np.abs(dv01s)) if np.sum(np.abs(dv01s)) > 0 else np.ones(len(ets))/len(ets)
                
        # Calculate basket return
        rets = [rate_hedged_return(df) for df in ets]
        r_basket = pd.concat(rets, axis=1).fillna(0.0).values @ w
        r_basket = pd.Series(r_basket, index=idx, name="r_basket")
        
        # Build basket features
        z_nav_b = sum(w[i] * nav_premium_z(ets[i]) for i in range(len(ets)))
        z_daf_b = sum(w[i] * daf_z(ets[i]) for i in range(len(ets)))
        
        # CDX features
        cdxf = self.cdx[spec.cdx_family].loc[idx]
        d_cdx = (cdxf["spread"].add(cdxf.get("index_basis", 0.0))).diff().fillna(0.0) / 1e4
        
        # Carry differential
        z_carry_b = sum(w[i] * (ets[i].get("dividend_yield", 0.0)/252 - ets[i].get("borrow_fee", 0.0)/252) for i in range(len(ets)))
        z_carry_b = robust_z((z_carry_b - (cdxf["spread"]/1e4)/252.0).fillna(0.0), 63.0)
        
        # Basis to underlying
        basis_list = []
        for i, df in enumerate(ets):
            b = cdx_basis_to_underlying(df, cdxf["spread"], holdings=self.holdings)
            basis_list.append(w[i] * b)
        basis_b = (sum(basis_list) if basis_list else pd.Series(0.0, index=idx)).rename("basis")
        
        # Build feature matrix
        X = pd.DataFrame(index=idx)
        X["const"], X["d_cdx"], X["z_nav"], X["z_daf"], X["z_carry"], X["basis"] = (
            1.0, d_cdx, z_nav_b, z_daf_b, z_carry_b, robust_z(basis_b.fillna(0.0), 63.0)
        )
        X = self._apply_mask(X, feature_mask)
        
        # Run Kalman filter
        kf = TVPKalman(k=X.shape[1], F=np.eye(X.shape[1]), R=R_init, delta=np.array(delta), mode="discount")
        z_list, theta_list = [], []
        
        for t, (_, row) in enumerate(X.iterrows()):
            z, S, theta = kf.step(row.values.astype(float), float(r_basket.iloc[t]))
            z_list.append(z); theta_list.append(theta)
            
        theta_df = pd.DataFrame(theta_list, index=idx, columns=X.columns)
        z = pd.Series(z_list, index=idx, name=spec.name or f"z_{'_'.join(spec.etfs)}_vs_{spec.cdx_family}")
        
        # Calculate next-day pair return and prediction
        beta = theta_df["d_cdx"].shift(1)
        pair_ret_next = (r_basket.shift(-1) - beta * d_cdx.shift(-1)).rename("pair_ret_next")
        lam = rolling_beta(pair_ret_next, z.shift(1), win=120)
        yhat_next = (lam * z).rename("yhat_pair_next")
        
        # Return decomposition (basket level) using OAS proxy = CDX spread + basis_b
        oas_proxy = cdxf["spread"].add(basis_b.reindex(idx).fillna(0.0))
        decomp = decompose_basket_credit(ets, w, oas_proxy)
        
        return {
            "pair_name": spec.name or f"{'_'.join(spec.etfs)}_vs_{spec.cdx_family}",
            "z_pair": z,
            "theta": theta_df,
            "X": X,
            "pair_ret_next": pair_ret_next,
            "pred_pair_ret_next": yhat_next,
            "decomp": decomp,
        }


# Local helpers
def rolling_beta(y: pd.Series, x: pd.Series, win: int = 120) -> pd.Series:
    """Calculate rolling beta using simple linear regression.
    
    Args:
        y: Dependent variable series
        x: Independent variable series
        win: Rolling window size
        
    Returns:
        Rolling beta series
    """
    x = x.fillna(0.0); y = y.fillna(0.0)
    xsq = (x**2).rolling(win).sum()
    xym = (x*y).rolling(win).sum()
    lam = xym / xsq.replace(0.0, np.nan)
    return lam.ffill().fillna(0.0)


# Example usage and testing
if __name__ == "__main__":
    # Test the engine with synthetic data
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Sample ETF data
    etf_data = {
        'HYG': pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
            'nav': 99.5 + np.cumsum(np.random.randn(100) * 0.08),
            'so': 1000 + np.cumsum(np.random.randn(100) * 10),
            'basket_dv01': 0.5 + np.random.randn(100) * 0.05,
            'dividend_yield': 5.0 + np.random.randn(100) * 0.5,
            'borrow_fee': 1.0 + np.random.randn(100) * 0.2,
            'rates_excess_ret': np.random.randn(100) * 0.02
        }, index=dates),
        'JNK': pd.DataFrame({
            'close': 95 + np.cumsum(np.random.randn(100) * 0.1),
            'nav': 94.5 + np.cumsum(np.random.randn(100) * 0.08),
            'so': 800 + np.cumsum(np.random.randn(100) * 8),
            'basket_dv01': 0.6 + np.random.randn(100) * 0.06,
            'dividend_yield': 5.5 + np.random.randn(100) * 0.5,
            'borrow_fee': 1.2 + np.random.randn(100) * 0.2,
            'rates_excess_ret': np.random.randn(100) * 0.02
        }, index=dates)
    }
    
    # Sample CDX data
    cdx_data = {
        'HY': pd.DataFrame({
            'spread': 200 + np.cumsum(np.random.randn(100) * 2),
            'index_basis': np.random.randn(100) * 5
        }, index=dates)
    }
    
    # Create engine
    engine = PairEngine(etf_data, cdx_data)
    
    print("Engine Test Results:")
    print("=" * 40)
    
    # Test ETF pair
    etf_spec = ETFETFSpec(long="HYG", short="JNK")
    etf_result = engine.run_etf_pair(etf_spec)
    print(f"ETF pair signal mean: {etf_result['z_pair'].mean():.4f}")
    print(f"ETF pair signal std: {etf_result['z_pair'].std():.4f}")
    
    # Test basket CDX
    basket_spec = ETFCDXSpec(etfs=["HYG", "JNK"], cdx_family="HY")
    basket_result = engine.run_basket_cdx(basket_spec)
    print(f"Basket CDX signal mean: {basket_result['z_pair'].mean():.4f}")
    print(f"Basket CDX signal std: {basket_result['z_pair'].std():.4f}")
    print(f"Decomposition components: {list(basket_result['decomp'].columns)}")
    
    print("\nEngine test completed successfully!") 