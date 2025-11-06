"""Duration normalization and conversion utilities for time series processing.

This module provides tools for normalizing and converting between different duration units,
with support for duration codes, literal names, and flexible conversion with optional rounding.
"""

# Import des classes principales
from .utils import DurationNormalizer, DurationConverter

# Import des fonctions utilitaires de normalisation
from .utils import (
    normalize_duration,
    to_literal,
    to_code,
    validate_duration,
    convert_duration,
    get_duration_order
)

# Types exportés
from .utils import DurationType, UserDurationType, RoundingType

__all__ = [
    # Classes principales
    'DurationNormalizer',
    'DurationConverter',

    # Fonctions utilitaires
    'normalize_duration',
    'to_literal',
    'to_code',
    'validate_duration',
    'convert_duration',
    'get_duration_order',

    # Types
    'DurationType',
    'UserDurationType',
    'RoundingType'
]
