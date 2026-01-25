"""Frequency normalization utilities for time series processing.

This module provides the FrequencyNormalizer class to handle different frequency
representations including pandas frequency codes, DateOffsets, and user-friendly names.
"""
# Importation des modules
import re
import pandas as pd
from pandas.tseries.frequencies import to_offset
from typing import Union, Literal

# Import de la classe parente
from ..abc.normalizer import TemporalNormalizer

# Types supportés pour les fréquences
FrequencyType = Literal['ns', 'us', 'ms', 's', 'min', 'T', 'h', 'D', 'B', 'W', 'SM', 'M', 'Q', 'Y']
UserFrequencyType = Literal[
    'daily', 'weekly', 'monthly', 'quarterly', 'annual', 'business_daily'
]


# Classe de normalisation des fréquences
class FrequencyNormalizer(TemporalNormalizer):
    """Centralized frequency normalization and conversion utility.

    This class handles conversions between different frequency representations:
    - Pandas frequency codes (D, W, M, Q, A, etc.)
    - DateOffset objects
    - User-friendly names with reference points

    Examples:
        >>> normalizer = FrequencyNormalizer()
        >>> normalizer.to_pandas_freq('monthly')
        'M'
        >>> normalizer.to_literal('Q')
        'quarterly'
        >>> normalizer.normalize('daily')
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
            'T': 'minute',
            'h': 'hourly',
            'D': 'daily',
            'B': 'business_daily',
            'W': 'weekly',
            'SM': 'semi_monthly',
            'M': 'monthly',
            'Q': 'quarterly',
            'Y': 'annual'
        }

        # Mapping inverse pour les conversions (utilisation de la méthode héritée)
        self._literal_to_pandas = self._build_reverse_mapping(self._pandas_to_literal)

        # Ordre des fréquences pour les comparaisons (du plus granulaire au moins granulaire)
        self._frequency_order = {
            'ns' : 1,
            'us': 2,
            'ms': 3,
            's': 4,
            'min': 5,
            'T': 5,
            'h': 6,
            'D': 7,
            'B': 7.5,
            'W': 8,
            'SM': 8.5,
            'M': 9,
            'Q': 10,
            'Y': 11
        }

    # Méthode auxiliaire d'extraction de la fréquence de base
    def _extract_base_frequency(self, freq_string: str) -> Union[FrequencyType, None]:
        """Extract base frequency from complex pandas frequency string.

        Handles pandas DateOffset strings with positions and anchors:
        - Position suffixes: 'MS', 'ME', 'QS', 'QE', 'AS', 'AE'
        - Anchor suffixes: 'QE-DEC', 'QS-JAN', 'AS-MAR'

        Args:
            freq_string: Complex pandas frequency string

        Returns:
            Base frequency code if successful, None otherwise

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer._extract_base_frequency('QE-DEC')
            'Q'
            >>> normalizer._extract_base_frequency('MS')
            'M'
            >>> normalizer._extract_base_frequency('AS-JAN')
            'Y'
        """
        # Pattern: [FREQ_BASE][S|E]?[-ANCHOR]?
        match = re.match(r"^([A-Z]+?)([SE])?(-(.*?))?$", freq_string)

        if not match:
            return None

        # Extraction de la fréquence de base (premier groupe)
        base = match.group(1)

        # Normalisation de 'A' (annual) vers 'Y' (year) pour cohérence
        if base == 'A':
            base = 'Y'

        # Vérification que la base extraite est valide
        if base in self._pandas_to_literal:
            return base

        return None

    # Méthode de normalisation de l'expression de la fréquence
    def normalize(self, value: Union[FrequencyType, UserFrequencyType]) -> FrequencyType:
        """Normalize any frequency representation to pandas frequency code.

        Automatically extracts base frequencies from complex pandas frequency strings
        (e.g., 'QE-DEC' → 'Q', 'MS' → 'M', 'AS-JAN' → 'Y').

        Args:
            value: Frequency string (pandas code, literal name, or complex pandas string)

        Returns:
            Pandas frequency code string

        Raises:
            ValueError: If value format is not supported

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.normalize('monthly')
            'M'
            >>> normalizer.normalize('D')
            'D'
            >>> normalizer.normalize('QE-DEC')
            'Q'
            >>> normalizer.normalize('MS')
            'M'
            >>> normalizer.normalize('AS-JAN')
            'Y'
        """
        # Vérification que la fréquence est du type spécifié
        if not isinstance(value, str):
            raise ValueError(f"Frequency must be a string, got {type(value)}")

        # Retourne un code pandas inchangé
        if value in self._pandas_to_literal:
            return value

        # Conversion du nom littéral en pandas
        if value in self._literal_to_pandas:
            return self._literal_to_pandas[value]

        # Tentative d'extraction de la fréquence de base
        base_freq = self._extract_base_frequency(value)
        if base_freq is not None:
            return base_freq

        # Renvoie une erreur si le code est inconnu
        raise ValueError(
            f"Unsupported frequency: {value}. "
            f"Supported frequencies: {list(self._literal_to_pandas.keys())} "
            f"or pandas codes: {list(self._pandas_to_literal.keys())}"
        )

    # Méthode de conversion en nom littéraire
    def to_literal(self, frequency: FrequencyType) -> UserFrequencyType:
        """Convert frequency to literal name.

        Implementation of TemporalNormalizer.to_literal() for frequencies.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Literal frequency name

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.to_literal('M')
            'monthly'
            >>> normalizer.to_literal('Q')
            'quarterly'
        """
        # Normalisation de la fréquence
        pandas_freq = self.normalize(frequency)
        # Conversion dans son nom littéral
        return self._pandas_to_literal.get(pandas_freq, pandas_freq)

    # Conversion d'une fréquence dans son expression code
    def to_code(self, frequency: Union[FrequencyType, UserFrequencyType]) -> FrequencyType:
        """Convert frequency to frequency code.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Frequency code

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.to_code('daily')
            'D'
        """
        return self.normalize(frequency)

    # Méthode de validation de la fréquence
    def validate(self, value: FrequencyType) -> bool:
        """Validate if a frequency is supported.

        Args:
            frequency: Frequency to validate

        Returns:
            True if frequency is valid and supported

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.validate('daily')
            True
            >>> normalizer.validate('invalid_freq')
            False
        """
        try:
            # Tentative de normalisation de la fréquence
            self.normalize(value)
            return True
        except ValueError:
            return False

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
        return self.normalize(frequency)

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
        pandas_freq = self.normalize(frequency)
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
        # Normalisation des fréquences
        code1 = self.to_code(freq1)
        code2 = self.to_code(freq2)

        # Extraction de l'ordre associé à chaque fréquence
        order1 = self._frequency_order.get(code1, 0)
        order2 = self._frequency_order.get(code2, 0)

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
            self.normalize(freq1)
            self.normalize(freq2)
            return True
        except ValueError:
            return False

