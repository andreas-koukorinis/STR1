"""
Pair specification and configuration data classes.

This module provides a comprehensive interface for all specification and configuration
classes used in the credit RV framework. It imports from the engine module and
re-exports for convenient access.
"""

from engine import (
    ETFETFSpec, ETFCDXSpec,
    SignalConfig, PortfolioConfig
)

# Re-export all specification and configuration classes
__all__ = [
    "ETFETFSpec", "ETFCDXSpec",
    "SignalConfig", "PortfolioConfig"
] 