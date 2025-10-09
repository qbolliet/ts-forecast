"""Frequency normalization utilities for time series processing.

This module provides centralized functions to handle different frequency representations
including pandas frequency codes, DateOffsets, and user-friendly frequency names.
"""
# Importation des modules
# Modules de base
import pandas as pd
from pandas.tseries.frequencies import to_offset
from typing import Union, Literal, Optional, Dict, List

# Types supportés pour les fréquences
FrequencyType = Literal['ns', 'us', 'ms', 's', 'min', 'h', 'D', 'B', 'W', 'SM', 'M', 'Q', 'Y']
UserFrequencyType = Literal[
    'daily', 'weekly', 'monthly', 'quarterly', 'annual', 'business_daily'
]

# Classe de normalisation des fréquences
class FrequencyNormalizer:
    """Centralized frequency normalization and conversion utility.

    This class handles conversions between different frequency representations:
    - Pandas frequency codes (D, W, M, Q, A, etc.)
    - DateOffset objects
    - User-friendly names with reference points

    Examples:
        >>> normalizer = FrequencyNormalizer()
        >>> normalizer.to_pandas_freq('monthly_start')
        'MS'
        >>> normalizer.to_user_friendly('Q')
        'quarterly'
        >>> normalizer.normalize_frequency('daily')
        'D'
    """

    # Initialisation
    def __init__(self):
        """Initialize frequency mappings."""
        # Mapping des fréquences pandas vers des noms littéraux
        self._pandas_to_literal = {
            'ns': 'nanosecond',
            'us': 'microsecond',
            'ms': 'millisecond',
            's': 'second',
            'min': 'minute',
            'h': 'hourly',
            'D': 'daily',
            'B': 'business_daily',
            'W': 'weekly',
            'SM': 'semi_monthly',
            'M': 'monthly',
            'Q': 'quarterly',
            'Y': 'annual'
        }

        # Mapping inverse pour les conversions
        self._literal_to_pandas = {
            'nanosecond': 'ns',
            'microsecond': 'us',
            'millisecond': 'ms',
            'second': 's',
            'minute': 'min',
            'hourly': 'h',
            'daily': 'D',
            'business_daily': 'B',
            'weekly': 'W',
            'semi_monthly': 'SM',
            'monthly': 'M',
            'quarterly': 'Q',
            'annual': 'Y'
        }

        # Ordre des fréquences pour les comparaisons (du plus granulaire au moins granulaire)
        self._frequency_order = {
            'ns' : 1,
            'us': 2,
            'ms': 3,
            's': 4,
            'min': 5,
            'h': 6,
            'D': 7,
            'B': 7.5,
            'W': 8,
            'SM': 8.5,
            'M': 9,
            'Q': 10,
            'Y': 11
        }

    # Méthode de normalisation des fréquences dans leur expression pandas
    def normalize_frequency(self, frequency: Union[FrequencyType, UserFrequencyType]) -> FrequencyType:
        """Normalize any frequency representation to pandas frequency code.

        Args:
            frequency: Frequency string (pandas code or litteral name)

        Returns:
            Pandas frequency code string

        Raises:
            ValueError: If frequency format is not supported

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.normalize_frequency('monthly')
            'M'
            >>> normalizer.normalize_frequency('D')
            'D'
        """
        # Vérification que la fréquence est du type spécifié
        if not isinstance(frequency, str):
            raise ValueError(f"Frequency must be a string, got {type(frequency)}")

        # Retourne un code pandas inchangé
        if frequency in self._pandas_to_literal:
            return frequency

        # Conversion du nom littéral en pandas
        if frequency in self._literal_to_pandas:
            return self._literal_to_pandas[frequency]

        # Renvoie une erreur si le code est inconnu
        raise ValueError(f"Unsupported frequency: {frequency}. Supported frequencies: {list(self._literal_to_pandas.keys())} or pandas codes: {list(self._pandas_to_literal.keys())}")

    # Méthode de conversion de la fréquence dans son expression littérale
    def to_literal(self, frequency: FrequencyType) -> UserFrequencyType:
        """Convert frequency to literal name.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Literal frequency name

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.to_user_friendly('M')
            'monthly'
            >>> normalizer.to_user_friendly('Q')
            'quarterly'
        """
        # Normalisation de la fréquence
        pandas_freq = self.normalize_frequency(frequency)
        # COnversion dans son nom littéral
        return self._pandas_to_literal.get(pandas_freq, pandas_freq)

    # Conversion d'une fréquence dans son expression pandas
    def to_pandas_freq(self, frequency: Union[FrequencyType, UserFrequencyType]) -> FrequencyType:
        """Convert frequency to pandas frequency code.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Pandas frequency code

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.to_pandas_freq('monthly')
            'M'
        """
        return self.normalize_frequency(frequency)

    # Conversion d'une fréquence en DateOffset
    def to_dateoffset(self, frequency: FrequencyType) -> pd.DateOffset:
        """Convert frequency to pandas DateOffset object.

        Args:
            frequency: Frequency string

        Returns:
            Pandas DateOffset object

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> offset = normalizer.to_dateoffset('monthly')
            >>> isinstance(offset, pd.DateOffset)
            True
        """
        # Normalisation de la fréquence
        pandas_freq = self.normalize_frequency(frequency)
        # Conversion en OffSet
        return to_offset(pandas_freq)

    # Méthode déterminant si une fréquence est plus élevée d'une autre
    def is_higher_frequency(self, freq1: FrequencyType, freq2: FrequencyType) -> bool:
        """Check if freq1 is a higher frequency than freq2.

        Args:
            freq1: First frequency
            freq2: Second frequency

        Returns:
            True if freq1 is higher frequency than freq2

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.is_higher_frequency('daily', 'monthly')
            True
            >>> normalizer.is_higher_frequency('quarterly', 'weekly')
            False
        """
        literal_freq1 = self.to_literal(freq1)
        literal_freq2 = self.to_literal(freq2)

        order1 = self._frequency_order.get(literal_freq1, 0)
        order2 = self._frequency_order.get(literal_freq2, 0)

        return order1 < order2

    # Méthode de vérification que deux expressions de fréquences sont compatibles
    def are_compatible_frequencies(self, freq1: FrequencyType, freq2: FrequencyType) -> bool:
        """Check if two frequencies are compatible for conversion.

        Args:
            freq1: First frequency
            freq2: Second frequency

        Returns:
            True if frequencies can be converted between each other

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.are_compatible_frequencies('daily', 'monthly')
            True
            >>> normalizer.are_compatible_frequencies('business_daily', 'monthly')
            True
        """
        try:
            # Tentative de normalisation de chaque fréquence
            self.normalize_frequency(freq1)
            self.normalize_frequency(freq2)
            return True
        except ValueError:
            return False

    # Méthode de vérification qu'une fréquence est supportée
    def validate_frequency(self, frequency: FrequencyType) -> bool:
        """Validate if a frequency is supported.

        Args:
            frequency: Frequency to validate

        Returns:
            True if frequency is valid and supported

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.validate_frequency('daily')
            True
            >>> normalizer.validate_frequency('invalid_freq')
            False
        """
        try:
            # Tentative de normalisation de la fréquence
            self.normalize_frequency(frequency)
            return True
        except ValueError:
            return False


# Instance globale pour faciliter l'utilisation
_normalizer = FrequencyNormalizer()

# Fonctions de commodité pour accès direct
# Normalisation de la fréquence
def normalize_frequency(frequency: FrequencyType) -> str:
    """Normalize frequency to pandas frequency code.

    Args:
        frequency: Frequency string in pandas code or literal name format.
            Supported pandas codes: 'ns', 'us', 'ms', 's', 'min', 'h', 'D', 'B',
            'W', 'SM', 'M', 'Q', 'Y'. Supported literal names: 'nanosecond',
            'microsecond', 'millisecond', 'second', 'minute', 'hourly', 'daily',
            'business_daily', 'weekly', 'semi_monthly', 'monthly', 'quarterly', 'annual'.

    Returns:
        Normalized pandas frequency code string.

    Raises:
        ValueError: If frequency format is not supported.

    Examples:
        >>> normalize_frequency('monthly')
        'M'
        >>> normalize_frequency('D')
        'D'
        >>> normalize_frequency('business_daily')
        'B'
    """
    return _normalizer.normalize_frequency(frequency)

# Conversion d'une fréquence dans son expression littérale
def to_literal(frequency: FrequencyType) -> str:
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
        >>> to_literal('daily')
        'daily'
    """
    return _normalizer.to_literal(frequency)

# Conversion d'une fréquence dans son expression pandas
def to_pandas_freq(frequency: FrequencyType) -> str:
    """Convert frequency to pandas frequency code.

    Args:
        frequency: Frequency in any supported format (pandas code or literal name).

    Returns:
        Pandas frequency code string.

    Examples:
        >>> to_pandas_freq('monthly')
        'M'
        >>> to_pandas_freq('quarterly')
        'Q'
        >>> to_pandas_freq('D')
        'D'
    """
    return _normalizer.to_pandas_freq(frequency)

# Conversion d'une fréquence en DateOffset
def to_dateoffset(frequency: FrequencyType) -> pd.DateOffset:
    """Convert frequency to pandas DateOffset object.

    Args:
        frequency: Frequency string to convert to DateOffset.

    Returns:
        Pandas DateOffset object corresponding to the frequency.

    Examples:
        >>> offset = to_dateoffset('monthly')
        >>> isinstance(offset, pd.DateOffset)
        True
        >>> to_dateoffset('D')
        <Day>
        >>> to_dateoffset('quarterly')
        <QuarterEnd: startingMonth=12>
    """
    return _normalizer.to_dateoffset(frequency)

# Détermination de la plus haute fréquence
def is_higher_frequency(freq1: FrequencyType, freq2: FrequencyType) -> bool:
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
        >>> is_higher_frequency('hourly', 'daily')
        True
    """
    return _normalizer.is_higher_frequency(freq1, freq2)

# Vérification de la validité d'une fréquence
def validate_frequency(frequency: FrequencyType) -> bool:
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
    return _normalizer.validate_frequency(frequency)

# Extraction de l'ordre des fréquences
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
        >>> get_frequency_order('hourly')
        6.0
        >>> get_frequency_order('quarterly') > get_frequency_order('monthly')
        True
    """
    base_freq = normalize_frequency(frequency)
    return _normalizer._frequency_order.get(base_freq, 0)