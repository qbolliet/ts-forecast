# Importation des modules
# Modules de base
from typing import Union
import pandas as pd

# Importation des modules du package
from .normalizer import FrequencyNormalizer, FrequencyType, UserFrequencyType
from .converter import FrequencyConverter

# Instance globale pour faciliter l'utilisation
_normalizer = FrequencyNormalizer()
_converter = FrequencyConverter()

# Fonctions de commodité pour accès direct
# Fonction de normalisation de la fréquence
def normalize_frequency(frequency: Union[FrequencyType, UserFrequencyType]) -> str:
    """Normalize frequency to pandas frequency code.

    Args:
        frequency: Frequency string in pandas code or literal name format.

    Returns:
        Normalized pandas frequency code string.

    Examples:
        >>> normalize_frequency('monthly')
        'M'
        >>> normalize_frequency('D')
        'D'
    """
    return _normalizer.normalize(frequency)

# Fonction de conversion en expression littéraire
def to_literal(frequency: Union[FrequencyType, UserFrequencyType]) -> str:
    """Convert frequency to user-friendly literal name.

    Args:
        frequency: Frequency in pandas code or literal format to convert.

    Returns:
        User-friendly literal frequency name.

    Examples:
        >>> to_literal('M')
        'monthly'
        >>> to_literal('Q')
        'quarterly'
    """
    return _normalizer.to_literal(frequency)

# Fonction de conversion en code
def to_code(frequency: Union[FrequencyType, UserFrequencyType]) -> str:
    """Convert frequency to frequency code.

    Args:
        frequency: Frequency in any supported format (code or literal name).

    Returns:
        Frequency code string.

    Examples:
        >>> to_code('monthly')
        'M'
        >>> to_code('D')
        'D'
    """
    return _normalizer.to_code(frequency)

# Fonction de conversion en code de fréquence pandas
def to_pandas_freq(frequency: Union[FrequencyType, UserFrequencyType]) -> str:
    """Convert frequency to pandas frequency code.

    Args:
        frequency: Frequency in any supported format (pandas code or literal name).

    Returns:
        Pandas frequency code string.

    Examples:
        >>> to_pandas_freq('monthly')
        'M'
        >>> to_pandas_freq('D')
        'D'
    """
    return _normalizer.to_pandas_freq(frequency)

# Fonction de conversion en dateoffset
def to_dateoffset(frequency: Union[FrequencyType, UserFrequencyType]) -> pd.DateOffset:
    """Convert frequency to pandas DateOffset object.

    Args:
        frequency: Frequency string to convert to DateOffset.

    Returns:
        Pandas DateOffset object corresponding to the frequency.

    Examples:
        >>> offset = to_dateoffset('monthly')
        >>> isinstance(offset, pd.DateOffset)
        True
    """
    return _normalizer.to_dateoffset(frequency)

# Fonction de détection d'une fréquence plus élevée
def is_higher_frequency(freq1: Union[FrequencyType, UserFrequencyType],
                       freq2: Union[FrequencyType, UserFrequencyType]) -> bool:
    """Check if freq1 is higher frequency than freq2.

    Higher frequency means more granular time periods (e.g., daily vs monthly).

    Args:
        freq1: First frequency to compare.
        freq2: Second frequency to compare.

    Returns:
        True if freq1 has higher frequency (more granular) than freq2.

    Examples:
        >>> is_higher_frequency('daily', 'monthly')
        True
        >>> is_higher_frequency('quarterly', 'weekly')
        False
    """
    return _normalizer.is_higher_frequency(freq1, freq2)

# Fonction de validation de la fréquence
def validate_frequency(frequency: Union[FrequencyType, UserFrequencyType]) -> bool:
    """Validate if frequency is supported by the normalizer.

    Args:
        frequency: Frequency string to validate.

    Returns:
        True if frequency is valid and supported, False otherwise.

    Examples:
        >>> validate_frequency('daily')
        True
        >>> validate_frequency('M')
        True
        >>> validate_frequency('invalid_freq')
        False
    """
    return _normalizer.validate(frequency)

# Fonction de détection de l'ordre des fréquences
def get_frequency_order(frequency: Union[FrequencyType, UserFrequencyType]) -> float:
    """Get the numerical order of a frequency for comparison purposes.

    The order represents the granularity level where lower numbers indicate
    higher frequency (more granular) and higher numbers indicate lower
    frequency (less granular).

    Args:
        frequency: Frequency to get order for. Can be pandas code or literal name.

    Returns:
        Frequency order as float. Higher number means lower frequency/granularity.
        Returns 0 if frequency is not found in the order mapping.

    Examples:
        >>> get_frequency_order('daily')
        7.0
        >>> get_frequency_order('monthly')
        9.0
        >>> get_frequency_order('quarterly') > get_frequency_order('monthly')
        True
    """
    base_freq = normalize_frequency(frequency)
    return _normalizer._frequency_order.get(base_freq, 0)

# Conversion entre fréquences
def convert_frequency(
    value: Union[pd.Series, pd.DataFrame],
    to_unit: str,
    **kwargs
) -> float:
    """Convert data from one frequency to another.

    Args:
        value: Time series data to convert (Series or DataFrame)
        to_unit: Target frequency
        **kwargs: Additional conversion parameters (method, fill_method, etc.)

    Returns:
        Converted time series data

    Raises:
        ValueError: If conversion parameters are invalid

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> monthly = convert_frequency(series, 'daily', 'monthly', method='mean')
        >>> len(monthly)
        1
    """
    return _converter.convert(value, to_unit, **kwargs)
