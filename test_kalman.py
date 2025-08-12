"""
Test script for the enhanced Kalman filter with both discounted and FLS modes.

This script demonstrates the capabilities of the refactored TVPKalman class
with both discounted Kalman and Flexible Least Squares (FLS) modes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalman import TVPKalman, create_discounted_kalman, create_fls_kalman


def test_kalman_modes():
    """Test both discounted Kalman and FLS modes."""
    print("=" * 60)
    print("ENHANCED KALMAN FILTER TEST")
    print("=" * 60)
    
    # Generate synthetic time-varying parameter data
    np.random.seed(42)
    T = 1000
    
    # True parameters that change over time
    true_params = np.zeros((T, 2))
    true_params[:, 0] = 0.5 + 0.3 * np.sin(np.arange(T) * 0.01)  # Sinusoidal variation
    true_params[:, 1] = -0.3 + 0.2 * np.cos(np.arange(T) * 0.015)  # Different frequency
    
    # Generate regressors and responses
    X = np.random.randn(T, 2)
    y = np.sum(X * true_params, axis=1) + 0.1 * np.random.randn(T)
    
    print(f"Generated {T} observations with time-varying parameters")
    print(f"Parameter 1 range: [{true_params[:, 0].min():.3f}, {true_params[:, 0].max():.3f}]")
    print(f"Parameter 2 range: [{true_params[:, 1].min():.3f}, {true_params[:, 1].max():.3f}]")
    
    # Test 1: Discounted Kalman Filter
    print("\n" + "-" * 40)
    print("TEST 1: DISCOUNTED KALMAN FILTER")
    print("-" * 40)
    
    dk_kalman = create_discounted_kalman(k=2, R=0.01, delta=0.99, huber_c=4.0)
    dk_params = []
    dk_std = []
    
    for t in range(T):
        z, S, theta = dk_kalman.step(X[t], y[t])
        dk_params.append(theta.copy())
        dk_std.append(dk_kalman.get_parameter_std())
    
    dk_params = np.array(dk_params)
    dk_std = np.array(dk_std)
    
    print(f"Final parameters: {dk_params[-1]}")
    print(f"Final parameter std: {dk_std[-1]}")
    print(f"Parameter tracking error (RMSE): {np.sqrt(np.mean((dk_params - true_params)**2, axis=0))}")
    
    # Test 2: FLS Kalman Filter
    print("\n" + "-" * 40)
    print("TEST 2: FLEXIBLE LEAST SQUARES (FLS)")
    print("-" * 40)
    
    fls_kalman = create_fls_kalman(k=2, R=0.01, kappa=100.0, huber_c=4.0)
    fls_params = []
    fls_std = []
    
    for t in range(T):
        z, S, theta = fls_kalman.step(X[t], y[t])
        fls_params.append(theta.copy())
        fls_std.append(fls_kalman.get_parameter_std())
    
    fls_params = np.array(fls_params)
    fls_std = np.array(fls_std)
    
    print(f"Final parameters: {fls_params[-1]}")
    print(f"Final parameter std: {fls_std[-1]}")
    print(f"Parameter tracking error (RMSE): {np.sqrt(np.mean((fls_params - true_params)**2, axis=0))}")
    
    # Test 3: Comparison
    print("\n" + "-" * 40)
    print("COMPARISON")
    print("-" * 40)
    
    dk_rmse = np.sqrt(np.mean((dk_params - true_params)**2, axis=0))
    fls_rmse = np.sqrt(np.mean((fls_params - true_params)**2, axis=0))
    
    print("Parameter Tracking RMSE:")
    print(f"  Discounted Kalman: [{dk_rmse[0]:.4f}, {dk_rmse[1]:.4f}]")
    print(f"  FLS:               [{fls_rmse[0]:.4f}, {fls_rmse[1]:.4f}]")
    
    # Test 4: Advanced features
    print("\n" + "-" * 40)
    print("ADVANCED FEATURES TEST")
    print("-" * 40)
    
    # Test reset functionality
    dk_kalman.reset()
    print(f"After reset - parameters: {dk_kalman.get_parameters()}")
    print(f"After reset - covariance: {dk_kalman.get_covariance().diagonal()}")
    
    # Test parameter access methods
    print(f"Parameter std from method: {dk_kalman.get_parameter_std()}")
    print(f"Parameter std from covariance: {np.sqrt(np.diag(dk_kalman.get_covariance()))}")
    
    return {
        'true_params': true_params,
        'dk_params': dk_params,
        'fls_params': fls_params,
        'dk_std': dk_std,
        'fls_std': fls_std
    }


def plot_kalman_comparison(results):
    """Plot comparison of Kalman filter modes."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Parameter 1 comparison
    axes[0, 0].plot(results['true_params'][:, 0], 'k-', label='True', linewidth=2)
    axes[0, 0].plot(results['dk_params'][:, 0], 'b-', label='Discounted Kalman', alpha=0.7)
    axes[0, 0].plot(results['fls_params'][:, 0], 'r-', label='FLS', alpha=0.7)
    axes[0, 0].set_title('Parameter 1 Tracking')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parameter 2 comparison
    axes[0, 1].plot(results['true_params'][:, 1], 'k-', label='True', linewidth=2)
    axes[0, 1].plot(results['dk_params'][:, 1], 'b-', label='Discounted Kalman', alpha=0.7)
    axes[0, 1].plot(results['fls_params'][:, 1], 'r-', label='FLS', alpha=0.7)
    axes[0, 1].set_title('Parameter 2 Tracking')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Parameter uncertainty comparison
    axes[1, 0].plot(results['dk_std'][:, 0], 'b-', label='Discounted Kalman', alpha=0.7)
    axes[1, 0].plot(results['fls_std'][:, 0], 'r-', label='FLS', alpha=0.7)
    axes[1, 0].set_title('Parameter 1 Uncertainty (Std)')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(results['dk_std'][:, 1], 'b-', label='Discounted Kalman', alpha=0.7)
    axes[1, 1].plot(results['fls_std'][:, 1], 'r-', label='FLS', alpha=0.7)
    axes[1, 1].set_title('Parameter 2 Uncertainty (Std)')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'kalman_comparison.png'")
    
    return fig


if __name__ == "__main__":
    # Run tests
    results = test_kalman_modes()
    
    # Create comparison plot
    plot_kalman_comparison(results)
    
    print("\n" + "=" * 60)
    print("KALMAN FILTER TEST COMPLETE")
    print("=" * 60) 