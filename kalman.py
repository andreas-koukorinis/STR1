"""
rv_kalman.py — Time‑Varying Regression via Kalman / FLS
=======================================================

Implements a discounted Kalman filter for TVP regression with an optional
"FLS" (constant‑Q) mode. Huberized innovations for robustness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import math

__all__ = ["TVPKalman"]


@dataclass
class TVPKalman:
    """Time‑varying‑parameter regression via a discounted Kalman filter.

    This class implements a Kalman filter for time-varying parameter regression
    where the parameters evolve over time according to a state equation. The filter
    supports two modes: discounted Kalman (default) and Flexible Least Squares (FLS).
    Optional Huberization is included to make the filter robust to outliers.

    The model is:
        Observation: y_t = x_t' θ_t + ε_t,   ε_t ~ N(0, R)
        State:       θ_t = F θ_{t-1} + w_t,  w_t ~ N(0, Q_t)

    Two Q_t modes:
      • mode="discount":  Q_t = ((1-δ)/δ) * diag(P_{t-1|t-1})  (per‑state discount)
      • mode="fls":       Q_t = Q_const (derived from kappa via q = R/kappa)

    Huberization clips the standardized innovation at |z| > huber_c to reduce
    the effect of outliers.

    Attributes:
        k: Number of parameters (dimension of state vector)
        F: State transition matrix (default: identity matrix)
        R: Observation noise variance
        delta: Discount factors for each parameter (controls parameter adaptation speed)
        huber_c: Huberization constant for outlier robustness (default: 4.0)
        mode: Filter mode - "discount" for discounted Kalman or "fls" for FLS
        kappa: FLS parameter (R/q ratio) - only used in "fls" mode
        Q_const: Explicit constant Q matrix for FLS mode
        theta: Current parameter estimates (state vector)
        P: Current parameter covariance matrix
    """
    k: int
    F: np.ndarray
    R: float
    delta: np.ndarray | float = 0.99
    huber_c: float = 4.0
    mode: str = "discount"   # "discount" or "fls"
    kappa: np.ndarray | float | None = None  # only for mode="fls"
    Q_const: np.ndarray | None = None        # explicit constant Q

    theta: np.ndarray = field(default_factory=lambda: None)
    P: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        """Initialize the Kalman filter parameters and validate configuration."""
        # Initialize state vector
        if self.theta is None:
            self.theta = np.zeros(self.k)
        
        # Initialize covariance matrix
        if self.P is None:
            self.P = np.eye(self.k) * 1e2
        
        # Set default state transition matrix
        if self.F is None:
            self.F = np.eye(self.k)
        
        # Convert scalar delta to array if needed
        if np.isscalar(self.delta):
            self.delta = np.repeat(float(self.delta), self.k)
        
        # Validate mode
        if self.mode not in ("discount", "fls"):
            raise ValueError("mode must be 'discount' or 'fls'")

        # Initialize FLS mode if selected
        if self.mode == "fls" and self.Q_const is None:
            # Build constant Q from kappa (R/q = kappa  ->  q = R/kappa)
            kappa = 100.0 if self.kappa is None else self.kappa
            if np.isscalar(kappa):
                q = float(self.R) / max(float(kappa), 1e-9)
                self.Q_const = np.eye(self.k) * q
            else:
                kap = np.asarray(kappa).reshape(-1)
                q = np.maximum(self.R / np.maximum(kap, 1e-9), 1e-12)
                self.Q_const = np.diag(q)

    def _Q(self) -> np.ndarray:
        """Calculate the process noise covariance matrix Q_t.
        
        Returns:
            Process noise covariance matrix for current time step
        """
        if self.mode == "discount":
            # Discounted Kalman: Q_t = ((1-δ)/δ) * diag(P_{t-1|t-1})
            return np.diag((1 - self.delta) / np.maximum(self.delta, 1e-9) * np.diag(self.P))
        else:
            # FLS mode: constant Q
            return self.Q_const if self.Q_const is not None else np.eye(self.k) * 1e-6

    def step(self, x_t: np.ndarray, y_t: float) -> Tuple[float, float, np.ndarray]:
        """Perform one step of the Kalman filter.
        
        This method implements the standard Kalman filter recursion:
        1. Predict step: θ_t|t-1 = F θ_{t-1|t-1}, P_t|t-1 = F P_{t-1|t-1} F' + Q_t
        2. Update step: K_t = P_t|t-1 x_t / (x_t' P_t|t-1 x_t + R)
                        θ_t|t = θ_t|t-1 + K_t (y_t - x_t' θ_t|t-1)
                        P_t|t = (I - K_t x_t') P_t|t-1
        
        Args:
            x_t: Regressor vector at time t
            y_t: Observed response at time t
            
        Returns:
            Tuple containing:
                - z: Standardized innovation (for diagnostics)
                - S: Innovation variance (for diagnostics)  
                - theta: Updated parameter estimates
        """
        x = np.asarray(x_t).reshape(-1)
        
        # Predict step
        theta_pred = self.F @ self.theta
        P_pred = self.F @ self.P @ self.F.T + self._Q()
        
        # Observe step
        yhat = float(x @ theta_pred)
        nu = y_t - yhat  # Innovation
        S = float(x @ P_pred @ x + self.R)  # Innovation variance
        z = nu / max(math.sqrt(S), 1e-9)  # Standardized innovation
        
        # Huberization for robustness
        if abs(z) > self.huber_c:
            nu = math.copysign(self.huber_c * math.sqrt(S), z)
            z = nu / max(math.sqrt(S), 1e-9)
        
        # Update step
        K = (P_pred @ x) / max(S, 1e-12)  # Kalman gain
        self.theta = theta_pred + K * nu
        self.P = (np.eye(self.k) - np.outer(K, x)) @ P_pred
        
        return z, S, self.theta.copy()

    def reset(self) -> None:
        """Reset the filter to initial state.
        
        This method resets the parameter estimates and covariance matrix
        to their initial values, allowing the filter to be reused.
        """
        self.theta = np.zeros(self.k)
        self.P = np.eye(self.k) * 1e2

    def get_parameters(self) -> np.ndarray:
        """Get current parameter estimates.
        
        Returns:
            Current parameter vector θ_t
        """
        return self.theta.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current parameter covariance matrix.
        
        Returns:
            Current parameter covariance matrix P_t
        """
        return self.P.copy()

    def get_parameter_std(self) -> np.ndarray:
        """Get standard deviations of parameter estimates.
        
        Returns:
            Standard deviations of parameter estimates (sqrt of diagonal of P)
        """
        return np.sqrt(np.diag(self.P))


def create_discounted_kalman(k: int, R: float, delta: float = 0.99, 
                           huber_c: float = 4.0) -> TVPKalman:
    """Create a discounted Kalman filter for TVP regression.
    
    This is a convenience function to create a discounted Kalman filter
    with default settings suitable for most applications.
    
    Args:
        k: Number of parameters
        R: Observation noise variance
        delta: Discount factor (default: 0.99)
        huber_c: Huberization constant (default: 4.0)
        
    Returns:
        Configured TVPKalman instance in discount mode
    """
    return TVPKalman(
        k=k,
        F=np.eye(k),
        R=R,
        delta=delta,
        huber_c=huber_c,
        mode="discount"
    )


def create_fls_kalman(k: int, R: float, kappa: float = 100.0,
                     huber_c: float = 4.0) -> TVPKalman:
    """Create a Flexible Least Squares (FLS) Kalman filter.
    
    This is a convenience function to create an FLS Kalman filter
    with default settings. FLS uses a constant process noise covariance.
    
    Args:
        k: Number of parameters
        R: Observation noise variance
        kappa: FLS parameter (R/q ratio, default: 100.0)
        huber_c: Huberization constant (default: 4.0)
        
    Returns:
        Configured TVPKalman instance in FLS mode
    """
    return TVPKalman(
        k=k,
        F=np.eye(k),
        R=R,
        huber_c=huber_c,
        mode="fls",
        kappa=kappa
    )


# Example usage and testing
if __name__ == "__main__":
    # Example: TVP regression with 2 parameters
    np.random.seed(42)
    
    # Generate synthetic data
    T = 1000
    true_params = np.array([0.5, -0.3])
    X = np.random.randn(T, 2)
    y = X @ true_params + 0.1 * np.random.randn(T)
    
    # Create discounted Kalman filter
    kalman = create_discounted_kalman(k=2, R=0.01, delta=0.99)
    
    # Run filter
    params_history = []
    for t in range(T):
        z, S, theta = kalman.step(X[t], y[t])
        params_history.append(theta.copy())
    
    params_history = np.array(params_history)
    
    print("Discounted Kalman Filter Results:")
    print(f"Final parameters: {params_history[-1]}")
    print(f"True parameters:  {true_params}")
    print(f"Parameter std:    {kalman.get_parameter_std()}")
    
    # Create FLS Kalman filter
    fls_kalman = create_fls_kalman(k=2, R=0.01, kappa=100.0)
    
    # Run FLS filter
    fls_params_history = []
    for t in range(T):
        z, S, theta = fls_kalman.step(X[t], y[t])
        fls_params_history.append(theta.copy())
    
    fls_params_history = np.array(fls_params_history)
    
    print("\nFLS Kalman Filter Results:")
    print(f"Final parameters: {fls_params_history[-1]}")
    print(f"True parameters:  {true_params}")
    print(f"Parameter std:    {fls_kalman.get_parameter_std()}") 