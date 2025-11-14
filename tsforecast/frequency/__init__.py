"""Frequency detection and conversion utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
"""

# Import des classes principales
from .detector import FrequencyDetector

__all__ = [
    # Classes principales
    'FrequencyDetector',
]