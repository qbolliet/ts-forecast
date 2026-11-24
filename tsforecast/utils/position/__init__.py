"""Period position normalization and conversion utilities for time series processing.

This module provides tools for normalizing and converting between different period
positions (start vs end) for time series data, with support for pandas DateOffset codes.
"""


# Importation des classes principales
from .normalizer import (
    PeriodPositionNormalizer
)

from .converter import (
    PeriodPositionConverter
)

# Importation des types
from .types import (
    PositionType,
    UserPositionType
)

# Importation des fonctions utilitaires
from .utils import (
    normalize_position,
    to_literal,
    to_code,
    validate_position,
    flip_position,
    convert_position,
    convert_offset
)

# Réexport de toutes les fonctions
__all__ = [
    # Classes
    'PeriodPositionNormalizer',
    'PeriodPositionConverter',

    # Types
    'PositionType',
    'UserPositionType',

    # Fonctions de normalisation
    'normalize_position',
    'to_literal',
    'to_code',
    'validate_position',
    'flip_position',

    # Fonctions de conversion
    'convert_position',
    'convert_offset'
]