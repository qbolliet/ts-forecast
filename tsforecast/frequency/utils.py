"""Frequency normalization utilities for time series processing.

This module provides centralized functions to handle different frequency representations
including pandas frequency codes, DateOffsets, and user-friendly frequency names.
"""
# Importation des modules
import pandas as pd
from pandas.tseries.frequencies import to_offset
from typing import Union, Optional, Dict, Literal, Tuple
import re


# Types supportés pour les fréquences
FrequencyType = Union[str, pd.DateOffset]
UserFrequencyType = Literal[
    'daily', 'weekly', 'monthly', 'quarterly', 'annual',
    'daily_start', 'weekly_start', 'monthly_start', 'quarterly_start', 'annual_start',
    'daily_end', 'weekly_end', 'monthly_end', 'quarterly_end', 'annual_end'
]


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

    def __init__(self):
        """Initialize frequency mappings."""
        # Mapping des fréquences pandas vers des noms conviviaux
        self._pandas_to_friendly = {
            'D': 'daily',
            'B': 'business_daily',
            'W': 'weekly',
            'W-SUN': 'weekly_end',
            'W-MON': 'weekly_start',
            'M': 'monthly_end',
            'MS': 'monthly_start',
            'Q': 'quarterly_end',
            'QS': 'quarterly_start',
            'A': 'annual_end',
            'AS': 'annual_start',
            'Y': 'annual_end',
            'YS': 'annual_start'
        }

        # Mapping inverse pour les conversions
        self._friendly_to_pandas = {
            'daily': 'D',
            'business_daily': 'B',
            'weekly': 'W',
            'weekly_start': 'W-MON',
            'weekly_end': 'W-SUN',
            'monthly': 'M',
            'monthly_start': 'MS',
            'monthly_end': 'M',
            'quarterly': 'Q',
            'quarterly_start': 'QS',
            'quarterly_end': 'Q',
            'annual': 'A',
            'annual_start': 'AS',
            'annual_end': 'A'
        }

        # Ordre des fréquences pour les comparaisons (du plus granulaire au moins granulaire)
        self._frequency_order = {
            'daily': 1,
            'business_daily': 1.5,
            'weekly': 2,
            'monthly': 3,
            'quarterly': 4,
            'annual': 5
        }

    def normalize_frequency(self, frequency: FrequencyType) -> str:
        """Normalize any frequency representation to pandas frequency code.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Pandas frequency code string

        Raises:
            ValueError: If frequency format is not supported

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.normalize_frequency('monthly_start')
            'MS'
            >>> normalizer.normalize_frequency(pd.DateOffset(months=1))
            'M'
        """
        if frequency is None:
            raise ValueError("Frequency cannot be None")

        # Gestion des DateOffset
        if isinstance(frequency, pd.DateOffset):
            return self._dateoffset_to_pandas_freq(frequency)

        # Gestion des chaînes de caractères
        if isinstance(frequency, str):
            # Si c'est déjà un code pandas valide
            if frequency in self._pandas_to_friendly:
                return frequency

            # Si c'est un nom convivial
            if frequency in self._friendly_to_pandas:
                return self._friendly_to_pandas[frequency]

            # Tentative d'inférence avec pandas
            try:
                offset = to_offset(frequency)
                return self._dateoffset_to_pandas_freq(offset)
            except ValueError:
                pass

        raise ValueError(f"Unsupported frequency format: {frequency}")

    def to_user_friendly(self, frequency: FrequencyType) -> str:
        """Convert frequency to user-friendly name.

        Args:
            frequency: Frequency in any supported format

        Returns:
            User-friendly frequency name

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.to_user_friendly('MS')
            'monthly_start'
            >>> normalizer.to_user_friendly('Q')
            'quarterly_end'
        """
        pandas_freq = self.normalize_frequency(frequency)
        return self._pandas_to_friendly.get(pandas_freq, pandas_freq)

    def to_pandas_freq(self, frequency: FrequencyType) -> str:
        """Convert frequency to pandas frequency code.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Pandas frequency code

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.to_pandas_freq('monthly_start')
            'MS'
        """
        return self.normalize_frequency(frequency)

    def to_dateoffset(self, frequency: FrequencyType) -> pd.DateOffset:
        """Convert frequency to pandas DateOffset object.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Pandas DateOffset object

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> offset = normalizer.to_dateoffset('monthly')
            >>> isinstance(offset, pd.DateOffset)
            True
        """
        pandas_freq = self.normalize_frequency(frequency)
        return to_offset(pandas_freq)

    def get_base_frequency(self, frequency: FrequencyType) -> str:
        """Extract base frequency without reference point.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Base frequency name (daily, weekly, monthly, quarterly, annual)

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.get_base_frequency('monthly_start')
            'monthly'
            >>> normalizer.get_base_frequency('Q')
            'quarterly'
        """
        friendly = self.to_user_friendly(frequency)

        # Extraction de la fréquence de base
        base_freq = friendly.split('_')[0]
        if base_freq == 'business':
            return 'business_daily'
        return base_freq

    def get_reference_point(self, frequency: FrequencyType) -> Optional[str]:
        """Extract reference point from frequency.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Reference point ('start', 'end') or None if not specified

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.get_reference_point('monthly_start')
            'start'
            >>> normalizer.get_reference_point('Q')
            'end'
        """
        friendly = self.to_user_friendly(frequency)

        if '_start' in friendly:
            return 'start'
        elif '_end' in friendly:
            return 'end'
        elif friendly in ['monthly', 'quarterly', 'annual']:
            # Par défaut, les fréquences sans référence explicite sont "end"
            return 'end'
        return None

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
        base1 = self.get_base_frequency(freq1)
        base2 = self.get_base_frequency(freq2)

        order1 = self._frequency_order.get(base1, 0)
        order2 = self._frequency_order.get(base2, 0)

        return order1 < order2

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
            self.normalize_frequency(freq1)
            self.normalize_frequency(freq2)
            return True
        except ValueError:
            return False

    def _dateoffset_to_pandas_freq(self, offset: pd.DateOffset) -> str:
        """Convert DateOffset to pandas frequency string.

        Args:
            offset: Pandas DateOffset object

        Returns:
            Pandas frequency string
        """
        # Utilisation de la méthode pandas pour récupérer le code de fréquence
        freq_str = offset.freqstr

        # Normalisation des codes de fréquence courants
        if freq_str == '1D':
            return 'D'
        elif freq_str == '1W':
            return 'W'
        elif freq_str == '1M':
            return 'M'
        elif freq_str == '1MS':
            return 'MS'
        elif freq_str == '1Q':
            return 'Q'
        elif freq_str == '1QS':
            return 'QS'
        elif freq_str in ['1A', '1Y']:
            return 'A'
        elif freq_str in ['1AS', '1YS']:
            return 'AS'

        return freq_str

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
            self.normalize_frequency(frequency)
            return True
        except ValueError:
            return False


# Instance globale pour faciliter l'utilisation
_normalizer = FrequencyNormalizer()

# Fonctions de commodité pour accès direct
def normalize_frequency(frequency: FrequencyType) -> str:
    """Normalize frequency to pandas frequency code."""
    return _normalizer.normalize_frequency(frequency)

def to_user_friendly(frequency: FrequencyType) -> str:
    """Convert frequency to user-friendly name."""
    return _normalizer.to_user_friendly(frequency)

def to_pandas_freq(frequency: FrequencyType) -> str:
    """Convert frequency to pandas frequency code."""
    return _normalizer.to_pandas_freq(frequency)

def to_dateoffset(frequency: FrequencyType) -> pd.DateOffset:
    """Convert frequency to pandas DateOffset."""
    return _normalizer.to_dateoffset(frequency)

def get_base_frequency(frequency: FrequencyType) -> str:
    """Get base frequency without reference point."""
    return _normalizer.get_base_frequency(frequency)

def get_reference_point(frequency: FrequencyType) -> Optional[str]:
    """Get reference point from frequency."""
    return _normalizer.get_reference_point(frequency)

def is_higher_frequency(freq1: FrequencyType, freq2: FrequencyType) -> bool:
    """Check if freq1 is higher frequency than freq2."""
    return _normalizer.is_higher_frequency(freq1, freq2)

def validate_frequency(frequency: FrequencyType) -> bool:
    """Validate if frequency is supported."""
    return _normalizer.validate_frequency(frequency)