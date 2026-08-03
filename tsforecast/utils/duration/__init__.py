"""Duration normalization and conversion utilities for time series processing.

This module provides tools for normalizing and converting between different duration units,
with support for duration codes, literal names, and flexible conversion with optional rounding.
"""

# Import des classes principales
from .normalizer import DurationNormalizer
from .converter import DurationConverter

# Import des fonctions utilitaires de normalisation
from .utils import (
    normalize_duration,
    to_literal,
    to_code,
    validate_duration,
    get_duration_conversion_factor,
    convert_duration,
    get_duration_order
)

# Types exportés
from .normalizer import DurationType, UserDurationType
from .converter import RoundingType

__all__ = [
    # Classes principales
    'DurationNormalizer',
    'DurationConverter',

    # Fonctions utilitaires
    'normalize_duration',
    'to_literal',
    'to_code',
    'validate_duration',
    'get_duration_conversion_factor',
    'convert_duration',
    'get_duration_order',

    # Types
    'DurationType',
    'UserDurationType',
    'RoundingType'
]
