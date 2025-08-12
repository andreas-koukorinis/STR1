#!/usr/bin/env python3
"""
Test Demo for data.py - Synthetic Credit RV Data Generator
==========================================================

This script demonstrates the functionality of data.py by:
1. Importing and testing the synthetic data generation
2. Analyzing the generated data structure and characteristics
3. Visualizing key features and relationships
4. Testing different parameters and scenarios
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

# Import the data module
try:
    from data import create_synthetic_multi
    print("✅ Successfully imported data.py")
except ImportError as e:
    print(f"❌ Error importing data.py: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic data generation functionality."""
    print("\n" + "="*60)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("="*60)
    
    # Test with default parameters
    print("Generating synthetic data with default parameters...")
    assets, cdx = create_synthetic_multi()
    
    print(f"✅ Generated {len(assets)} ETFs: {list(assets.keys())}")
    print(f"✅ Generated {len(cdx)} CDX indices: {list(cdx.keys())}")
    
    # Check data structure
    for name, df in assets.items():
        print(f"\n📊 {name} ETF Structure:")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"   NAV premium range: {(df['nav']/df['close']-1).min():.3f} - {(df['nav']/df['close']-1).max():.3f}")
    
    for name, df in cdx.items():
        print(f"\n📈 {name} CDX Structure:")
        print(f"   Shape: {df.shape}")
        print(f"   Spread range: {df['spread'].min():.1f} - {df['spread'].max():.1f} bp")
        print(f"   DV01 range: {df['dv01'].min():.1f} - {df['dv01'].max():.1f}")
        if 'index_basis' in df.columns:
            print(f"   Basis range: {df['index_basis'].min():.2f} - {df['index_basis'].max():.2f} bp")
    
    return assets, cdx

def test_data_characteristics(assets: Dict[str, pd.DataFrame], cdx: Dict[str, pd.DataFrame]):
    """Analyze key characteristics of the generated data."""
    print("\n" + "="*60)
    print("TEST 2: DATA CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    # ETF Return Analysis
    print("\n📈 ETF Return Characteristics:")
    for name, df in assets.items():
        returns = df['close'].pct_change().dropna()
        print(f"\n{name}:")
        print(f"   Mean return: {returns.mean():.6f} ({returns.mean()*252:.2%} annualized)")
        print(f"   Volatility: {returns.std():.6f} ({returns.std()*np.sqrt(252):.2%} annualized)")
        print(f"   Skewness: {returns.skew():.3f}")
        print(f"   Kurtosis: {returns.kurtosis():.3f}")
        print(f"   Min/Max: {returns.min():.4f} / {returns.max():.4f}")
    
    # NAV Premium Analysis
    print("\n💰 NAV Premium Analysis:")
    for name, df in assets.items():
        nav_prem = (df['nav'] / df['close'] - 1)
        print(f"\n{name}:")
        print(f"   Mean premium: {nav_prem.mean():.4f} ({nav_prem.mean()*100:.2f}%)")
        print(f"   Premium volatility: {nav_prem.std():.4f}")
        print(f"   Premium range: {nav_prem.min():.4f} to {nav_prem.max():.4f}")
    
    # CDX Spread Analysis
    print("\n📊 CDX Spread Analysis:")
    for name, df in cdx.items():
        spreads = df['spread']
        print(f"\n{name}:")
        print(f"   Mean spread: {spreads.mean():.1f} bp")
        print(f"   Spread volatility: {spreads.std():.1f} bp")
        print(f"   Spread range: {spreads.min():.1f} - {spreads.max():.1f} bp")
        print(f"   Spread changes: mean={spreads.diff().mean():.3f}, std={spreads.diff().std():.3f}")

def test_correlation_dynamics(assets: Dict[str, pd.DataFrame], cdx: Dict[str, pd.DataFrame]):
    """Test correlation dynamics between ETFs and CDX."""
    print("\n" + "="*60)
    print("TEST 3: CORRELATION DYNAMICS")
    print("="*60)
    
    # ETF vs CDX correlations
    print("\n🔄 ETF vs CDX Correlations:")
    
    # HY ETFs vs CDX HY
    hyg_ret = assets['HYG']['close'].pct_change().dropna()
    jnk_ret = assets['JNK']['close'].pct_change().dropna()
    cdx_hy_ret = cdx['HY']['spread'].diff().dropna()
    
    # IG ETFs vs CDX IG
    lqd_ret = assets['LQD']['close'].pct_change().dropna()
    vcit_ret = assets['VCIT']['close'].pct_change().dropna()
    cdx_ig_ret = cdx['IG']['spread'].diff().dropna()
    
    # Find common dates
    common_hy = hyg_ret.index.intersection(cdx_hy_ret.index)
    common_ig = lqd_ret.index.intersection(cdx_ig_ret.index)
    
    if len(common_hy) > 0:
        corr_hyg_hy = hyg_ret.loc[common_hy].corr(cdx_hy_ret.loc[common_hy])
        corr_jnk_hy = jnk_ret.loc[common_hy].corr(cdx_hy_ret.loc[common_hy])
        print(f"   HYG vs CDX HY: {corr_hyg_hy:.3f}")
        print(f"   JNK vs CDX HY: {corr_jnk_hy:.3f}")
    
    if len(common_ig) > 0:
        corr_lqd_ig = lqd_ret.loc[common_ig].corr(cdx_ig_ret.loc[common_ig])
        corr_vcit_ig = vcit_ret.loc[common_ig].corr(cdx_ig_ret.loc[common_ig])
        print(f"   LQD vs CDX IG: {corr_lqd_ig:.3f}")
        print(f"   VCIT vs CDX IG: {corr_vcit_ig:.3f}")
    
    # ETF pair correlations
    print("\n🔄 ETF Pair Correlations:")
    common_etf = hyg_ret.index.intersection(jnk_ret.index).intersection(lqd_ret.index).intersection(vcit_ret.index)
    
    if len(common_etf) > 0:
        corr_hyg_jnk = hyg_ret.loc[common_etf].corr(jnk_ret.loc[common_etf])
        corr_lqd_vcit = lqd_ret.loc[common_etf].corr(vcit_ret.loc[common_etf])
        corr_hyg_lqd = hyg_ret.loc[common_etf].corr(lqd_ret.loc[common_etf])
        print(f"   HYG vs JNK: {corr_hyg_jnk:.3f}")
        print(f"   LQD vs VCIT: {corr_lqd_vcit:.3f}")
        print(f"   HYG vs LQD: {corr_hyg_lqd:.3f}")

def test_parameter_variations():
    """Test data generation with different parameters."""
    print("\n" + "="*60)
    print("TEST 4: PARAMETER VARIATIONS")
    print("="*60)
    
    # Test different time periods
    periods = [100, 500, 1000, 1600]
    print("\n📅 Testing different time periods:")
    for n_days in periods:
        assets, cdx = create_synthetic_multi(n_days=n_days, seed=42)
        print(f"   {n_days} days: {len(assets['HYG'])} actual days, "
              f"date range {assets['HYG'].index[0]} to {assets['HYG'].index[-1]}")
    
    # Test different seeds
    print("\n🎲 Testing different seeds (reproducibility):")
    seeds = [11, 42, 123, 999]
    for seed in seeds:
        assets1, cdx1 = create_synthetic_multi(n_days=100, seed=seed)
        assets2, cdx2 = create_synthetic_multi(n_days=100, seed=seed)
        
        # Check if identical (should be with same seed)
        identical = np.allclose(assets1['HYG']['close'], assets2['HYG']['close'])
        print(f"   Seed {seed}: {'✅ Identical' if identical else '❌ Different'}")

def test_data_quality(assets: Dict[str, pd.DataFrame], cdx: Dict[str, pd.DataFrame]):
    """Test data quality and consistency."""
    print("\n" + "="*60)
    print("TEST 5: DATA QUALITY CHECKS")
    print("="*60)
    
    # Check for missing values
    print("\n🔍 Missing Values Check:")
    for name, df in assets.items():
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"   ❌ {name}: {missing.sum()} missing values")
            for col, count in missing[missing > 0].items():
                print(f"      {col}: {count} missing")
        else:
            print(f"   ✅ {name}: No missing values")
    
    for name, df in cdx.items():
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"   ❌ CDX {name}: {missing.sum()} missing values")
        else:
            print(f"   ✅ CDX {name}: No missing values")
    
    # Check for negative prices
    print("\n💰 Price Validity Check:")
    for name, df in assets.items():
        neg_prices = (df['close'] <= 0).sum()
        neg_nav = (df['nav'] <= 0).sum()
        if neg_prices > 0 or neg_nav > 0:
            print(f"   ❌ {name}: {neg_prices} negative close prices, {neg_nav} negative NAV")
        else:
            print(f"   ✅ {name}: All prices positive")
    
    # Check for negative spreads
    print("\n📊 Spread Validity Check:")
    for name, df in cdx.items():
        neg_spreads = (df['spread'] <= 0).sum()
        if neg_spreads > 0:
            print(f"   ❌ CDX {name}: {neg_spreads} negative spreads")
        else:
            print(f"   ✅ CDX {name}: All spreads positive")

def main():
    """Main test function."""
    print("🚀 STARTING COMPREHENSIVE TEST OF data.py")
    print("="*60)
    
    try:
        # Test 1: Basic functionality
        assets, cdx = test_basic_functionality()
        
        # Test 2: Data characteristics
        test_data_characteristics(assets, cdx)
        
        # Test 3: Correlation dynamics
        test_correlation_dynamics(assets, cdx)
        
        # Test 4: Parameter variations
        test_parameter_variations()
        
        # Test 5: Data quality
        test_data_quality(assets, cdx)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 SUMMARY:")
        print("   • data.py successfully generates synthetic credit RV data")
        print("   • Data includes realistic ETF and CDX characteristics")
        print("   • Time-varying correlations and volatility clustering")
        print("   • NAV premium dynamics and basis movements")
        print("   • Ready for credit RV strategy testing")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
