"""Frequency detection and conversion utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
"""

# Import des classes principales
from .detector import FrequencyDetector
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.normalizer import FrequencyNormalizer

# Import des utilitaires de normalisation depuis normalizer
from ..utils.frequency.normalizer import (
    normalize_frequency,
    to_literal,
    to_code,
    to_pandas_freq,
    to_dateoffset,
    is_higher_frequency,
    validate_frequency,
    get_frequency_order
)

# Types exportés
from ..utils.frequency.normalizer import FrequencyType, UserFrequencyType
from ..utils.frequency.converter import AggregationMethod, InterpolationMethod

__all__ = [
    # Classes principales
    'FrequencyDetector',
    'FrequencyConverter',
    'FrequencyNormalizer',

    # Fonctions utilitaires
    'normalize_frequency',
    'to_literal',
    'to_code',
    'to_pandas_freq',
    'to_dateoffset',
    'is_higher_frequency',
    'validate_frequency',
    'get_frequency_order',

    # Types
    'FrequencyType',
    'UserFrequencyType',
    'AggregationMethod',
    'InterpolationMethod'
]