"""Period position utilities for time series processing.

This module provides utility functions to handle period position operations,
serving as a convenient interface to the normalizer and converter functionality.
"""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Union, TYPE_CHECKING
# Modules du package
from .normalizer import PeriodPositionNormalizer, PositionType, UserPositionType

# Import réservé au typage statique : voir _get_converter() pour la raison de
# l'import différé au niveau exécution
if TYPE_CHECKING:
    from .converter import PeriodPositionConverter

# Instance globale pour faciliter l'utilisation
_normalizer = PeriodPositionNormalizer()



# Fonctions de commodité pour accès direct
# Fonction de normalisation de la position
def normalize_position(position: Union[PositionType, UserPositionType]) -> PositionType:
    """Normalize position to position code.

    Args:
        position: Position string in code or literal name format.

    Returns:
        Normalized position code string ('S' or 'E').

    Examples:
        >>> normalize_position('start')
        'S'
        >>> normalize_position('E')
        'E'
    """
    return _normalizer.normalize(position)

# Fonction de conversion de la position en expression littérale
def to_literal(position: Union[PositionType, UserPositionType]) -> UserPositionType:
    """Convert position to user-friendly literal name.

    Args:
        position: Position in code or literal format to convert.

    Returns:
        User-friendly literal position name ('start' or 'end').

    Examples:
        >>> to_literal('S')
        'start'
        >>> to_literal('E')
        'end'
    """
    return _normalizer.to_literal(position)

# Fonction de conversion de la position en code
def to_code(position: Union[PositionType, UserPositionType]) -> PositionType:
    """Convert position to position code.

    Args:
        position: Position in any supported format (code or literal name).

    Returns:
        Position code string ('S' or 'E').

    Examples:
        >>> to_code('start')
        'S'
        >>> to_code('end')
        'E'
    """
    return _normalizer.to_code(position)

# Fonction de validation de la position
def validate_position(position: Union[PositionType, UserPositionType]) -> bool:
    """Validate if position is supported by the normalizer.

    Args:
        position: Position string to validate.

    Returns:
        True if position is valid and supported, False otherwise.

    Examples:
        >>> validate_position('start')
        True
        >>> validate_position('S')
        True
        >>> validate_position('invalid_pos')
        False
    """
    return _normalizer.validate(position)

# Fonction d'inversement de la position
def flip_position(position: Union[PositionType, UserPositionType]) -> str:
    """Flip position to its opposite.

    Args:
        position: Position to flip.

    Returns:
        Opposite position code.

    Examples:
        >>> flip_position('start')
        'E'
        >>> flip_position('E')
        'S'
    """
    return _normalizer.flip_position(position)

# Fonctions de conversion de la position
def convert_position(
    value: Union[pd.Series, pd.DataFrame, pd.DatetimeIndex],
    from_position: Union[PositionType, UserPositionType],
    to_position: Union[PositionType, UserPositionType],
    freq: str = None
) -> Union[pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """Convert time series data from one period position to another.

    Args:
        value: Time series data or DatetimeIndex to convert
        from_position: Source position
        to_position: Target position
        freq: Frequency of the time series (if None, will attempt to infer)

    Returns:
        Converted time series data with adjusted index

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=3, freq='MS')
        >>> series = pd.Series([1, 2, 3], index=dates)
        >>> end_series = convert_position(series, 'start', 'end', freq='M')
    """
    # Importation locale (pour éviter les imports circulaires)
    from .converter import PeriodPositionConverter
    # Instanciation de la classe
    _converter = PeriodPositionConverter()
    
    return _converter.convert(value, from_position, to_position, freq=freq)

# Fonction de conversion de l'offset
def convert_offset(offset_str: str, to_position: Union[PositionType, UserPositionType]) -> str:
    """Convert pandas DateOffset to a different position.

    Args:
        offset_str: Source pandas DateOffset
        to_position: Target position

    Returns:
        Converted pandas DateOffset string

    Examples:
        >>> convert_offset('MS', 'end')
        'ME'
        >>> convert_offset('QE', 'start')
        'QS'
    """
    # Importation locale (pour éviter les imports circulaires)
    from .converter import PeriodPositionConverter
    # Instanciation de la classe
    _converter = PeriodPositionConverter()
    
    return _converter.convert_offset(offset_str, to_position)
