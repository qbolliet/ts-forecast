"""Frequency detection, conversion, and imputation utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
It also provides mixed-frequency imputation capabilities.
"""

# Import des classes principales
from .detector import FrequencyDetector
from .high_frequency_imputer import HighFrequencyImputer

__all__ = [
    # Classes principales
    'FrequencyDetector',
    'HighFrequencyImputer',
]