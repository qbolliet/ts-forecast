"""Duration normalization and conversion utilities for time series processing.

This module provides utility functions to handle different duration representations
including codes, literal names, and conversions between different duration units with
optional rounding.
"""
# Importation des modules
from typing import Union

# Import des classes depuis les modules spécialisés
from .normalizer import DurationNormalizer
from .types import DurationType, UserDurationType, RoundingType


# Instance globale pour faciliter l'utilisation
_normalizer = DurationNormalizer()


# Fonctions de commodité pour accès direct
# Normalisation de la durée
def normalize_duration(duration: Union[DurationType, UserDurationType]) -> str:
    """Normalize duration to duration code.

    Args:
        duration: Duration string in code or literal name format.
            Supported codes: 'ns', 'us', 'ms', 's', 'min', 'h', 'D', 'B',
            'W', 'SM', 'M', 'Q', 'Y'. Supported literal names: 'nanosecond',
            'microsecond', 'millisecond', 'second', 'minute', 'hour', 'day',
            'business_day', 'week', 'semi_month', 'month', 'quarter', 'year'.

    Returns:
        Normalized duration code string.

    Raises:
        ValueError: If duration format is not supported.

    Examples:
        >>> normalize_duration('hour')
        'h'
        >>> normalize_duration('D')
        'D'
        >>> normalize_duration('business_day')
        'B'
    """
    return _normalizer.normalize(duration)


# Conversion d'une durée dans son expression littérale
def to_literal(duration: Union[DurationType, UserDurationType]) -> str:
    """Convert duration to user-friendly literal name.

    Args:
        duration: Duration in code or literal format to convert.

    Returns:
        User-friendly literal duration name.

    Examples:
        >>> to_literal('h')
        'hour'
        >>> to_literal('Q')
        'quarter'
        >>> to_literal('day')
        'day'
    """
    return _normalizer.to_literal(duration)


# Conversion d'une durée dans son expression code
def to_code(duration: Union[DurationType, UserDurationType]) -> str:
    """Convert duration to duration code.

    Args:
        duration: Duration in any supported format (code or literal name).

    Returns:
        Duration code string.

    Examples:
        >>> to_code('hour')
        'h'
        >>> to_code('quarter')
        'Q'
        >>> to_code('D')
        'D'
    """
    return _normalizer.to_code(duration)


# Vérification de la validité d'une durée
def validate_duration(duration: Union[DurationType, UserDurationType]) -> bool:
    """Validate if duration is supported by the normalizer.

    Args:
        duration: Duration string to validate.

    Returns:
        True if duration is valid and supported, False otherwise.

    Examples:
        >>> validate_duration('day')
        True
        >>> validate_duration('h')
        True
        >>> validate_duration('invalid_dur')
        False
    """
    return _normalizer.validate(duration)


# Extraction du facteur de conversion entre deux durées
def get_duration_conversion_factor(
    from_duration: Union[DurationType, UserDurationType],
    to_duration: Union[DurationType, UserDurationType]
) -> Union[int, float]:
    """Get the conversion factor between two duration units.
    
    Implementation of TemporalConverter.get_conversion_factor() for durations.

    Args:
        from_unit: Source duration unit
        to_unit: Target duration unit

    Returns:
        Conversion factor to multiply by source value

    Raises:
        ValueError: If duration units are not supported

    Examples:
        >>> get_duration_conversion_factor('hour', 'minute')
        60.0
        >>> get_duration_conversion_factor('day', 'hour')
        24.0
    """
    # Import local pour éviter l'import circulaire
    from .converter import DurationConverter

    # Initialisation de l'instance de la classe
    _converter = DurationConverter()

    return _converter.get_conversion_factor(from_duration, to_duration)


# Conversion entre durées
def convert_duration(
    value: float,
    from_duration: Union[DurationType, UserDurationType],
    to_duration: Union[DurationType, UserDurationType],
    rounding: RoundingType = None
) -> float:
    """Convert a value from one duration unit to another.

    Args:
        value: Numeric value to convert
        from_duration: Source duration unit
        to_duration: Target duration unit
        rounding: Optional rounding mode ('floor', 'ceil', or None)

    Returns:
        Converted value, optionally rounded

    Raises:
        ValueError: If duration units are not supported

    Examples:
        >>> convert_duration(2.5, 'hour', 'minute')
        150.0
        >>> convert_duration(90, 'minute', 'hour')
        1.5
        >>> convert_duration(90, 'minute', 'hour', rounding='ceil')
        2
        >>> convert_duration(1.5, 'day', 'hour')
        36.0
    """
    # Import local pour éviter l'import circulaire
    from .converter import DurationConverter

    # Initialisation de l'instance de la classe
    _converter = DurationConverter()

    return _converter.convert(value, from_duration, to_duration, rounding)


# Extraction de l'ordre des durées
def get_duration_order(duration: Union[DurationType, UserDurationType]) -> float:
    """Get the numerical order of a duration for comparison purposes.

    The order represents the length where lower numbers indicate
    shorter duration (more granular) and higher numbers indicate longer
    duration (less granular).

    Args:
        duration: Duration to get order for. Can be code or literal name.

    Returns:
        Duration order as float. Higher number means longer duration.
        Returns 0 if duration is not found in the order mapping.

    Examples:
        >>> get_duration_order('day')
        7.0
        >>> get_duration_order('month')
        9.0
        >>> get_duration_order('hour')
        6.0
        >>> get_duration_order('quarter') > get_duration_order('month')
        True
    """
    duration_code = normalize_duration(duration)
    return _normalizer._duration_order.get(duration_code, 0)
