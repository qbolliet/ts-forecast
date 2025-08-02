"""TS-Forecast: Time Series and Panel Data Cross-Validation Library

This package provides comprehensive cross-validation tools for time series and panel data,
with support for both in-sample and out-of-sample validation strategies.

Main Features:
- Time-aware cross-validation splits that respect temporal ordering
- Support for both time series (single entity) and panel data (multiple entities)
- In-sample and out-of-sample validation strategies
- Configurable gaps between training and test periods
- sklearn-compatible API for easy integration

Modules:
    crossvals: Cross-validation classes and utilities
"""

__version__ = "0.1.0"
__author__ = "Quentin Bolliet"

# Importation des classes principales pour en favoriser l'accès
from .crossvals import (
    OutOfSampleSplit,
    InSampleSplit,
    TSOutOfSampleSplit,
    TSInSampleSplit,
    PanelOutOfSampleSplit,
    PanelInSampleSplit,
)

__all__ = [
    "OutOfSampleSplit",
    "InSampleSplit", 
    "TSOutOfSampleSplit",
    "TSInSampleSplit",
    "PanelOutOfSampleSplit",
    "PanelInSampleSplit",
]