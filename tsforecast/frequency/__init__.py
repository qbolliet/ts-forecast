"""Frequency detection and conversion utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
"""

# Import des classes principales
from .detector import FrequencyDetector
from .converter import FrequencyConverter

# Import des utilitaires de normalisation
from .utils import (
    FrequencyNormalizer,
    normalize_frequency,
    to_user_friendly,
    to_pandas_freq,
    to_dateoffset,
    get_base_frequency,
    get_reference_point,
    is_higher_frequency,
    validate_frequency
)

# Types exportés
from .utils import FrequencyType, UserFrequencyType
from .converter import AggregationMethod, InterpolationMethod

__all__ = [
    # Classes principales
    'FrequencyDetector',
    'FrequencyConverter',
    'FrequencyNormalizer',

    # Fonctions utilitaires
    'normalize_frequency',
    'to_user_friendly',
    'to_pandas_freq',
    'to_dateoffset',
    'get_base_frequency',
    'get_reference_point',
    'is_higher_frequency',
    'validate_frequency',

    # Types
    'FrequencyType',
    'UserFrequencyType',
    'AggregationMethod',
    'InterpolationMethod'
]