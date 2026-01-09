"""Frequency normalization utilities for time series processing.

This module provides centralized functions to handle different frequency representations
including pandas frequency codes, DateOffsets, and user-friendly frequency names.
"""
# Import de la classe et fonctions depuis le module spécialisé
from .normalizer import (
    FrequencyNormalizer,
    FrequencyType,
    UserFrequencyType
)
from .converter import (
    FrequencyConverter,
    AggregationMethod,
    InterpolationMethod
)
from .utils import (
    normalize_frequency,
    to_literal,
    to_code,
    to_pandas_freq,
    to_dateoffset,
    is_higher_frequency,
    validate_frequency,
    get_frequency_order,
    convert_frequency
)

# Toutes les fonctions sont maintenant définies et exportées depuis normalizer.py
# Ce fichier sert de point d'entrée pour la compatibilité avec le code existant
# Note: detect_and_parse_frequency et build_frequency_string ont été déplacés vers frequency.parser
__all__ = [
    'FrequencyNormalizer',
    'FrequencyType',
    'UserFrequencyType',
    'FrequencyConverter',
    'AggregationMethod',
    'InterpolationMethod',
    'normalize_frequency',
    'to_literal',
    'to_code',
    'to_pandas_freq',
    'to_dateoffset',
    'is_higher_frequency',
    'validate_frequency',
    'get_frequency_order',
    'convert_frequency'
]