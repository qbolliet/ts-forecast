"""Frequency normalization utilities for time series processing.

This module provides the FrequencyNormalizer class to handle different frequency
representations including pandas frequency codes, DateOffsets, and user-friendly names.
"""
# Importation des modules
import pandas as pd
from pandas.tseries.frequencies import to_offset
from typing import Union

# Import de la classe parente
from ..abc.normalizer import TemporalNormalizer
from .._constants import CONVERSION_FACTORS_TO_SECONDS, MONTH_ABBREVIATIONS, WEEKDAY_ABBREVIATIONS

# Import de l'utilitaire du package
from .types import FrequencyType, UserFrequencyType
from ..parse.utils import (
    ParsedFrequency, parse_frequency, build_frequency_string,
)

# Durées nominales (en secondes) servant à comparer des fréquences multipliées. Celles des
# conversions, sauf le jour ouvré : ses 5 observations par semaine espacent en moyenne les
# dates de 7/5 jours calendaires, si bien que 'kB' est une fréquence plus basse que 'kD'
# (cohérence avec l'ordre sans multiplicateur, où 'D' est plus élevée que 'B')
_NOMINAL_SECONDS = {**CONVERSION_FACTORS_TO_SECONDS, 'B': CONVERSION_FACTORS_TO_SECONDS['D'] * 7 / 5}


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
        'ME'
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
            'h': 6,
            'D': 7,
            'B': 7.5,
            'W': 8,
            'SM': 8.5,
            'M': 9,
            'Q': 10,
            'Y': 11
        }

    # Méthode de normalisation de l'expression de la fréquence
    def normalize(self, value: Union[FrequencyType, UserFrequencyType]) -> FrequencyType:
        """Normalize any frequency representation to pandas frequency code.

        Automatically extracts base frequencies from complex pandas frequency strings
        (e.g., 'QE-DEC' → 'Q', 'MS' → 'M', 'YS-JAN' → 'Y'). A leading multiplier
        is accepted and dropped like the position and the anchor ('2MS' → 'M');
        use :meth:`normalize_with_multiplier` to keep it. The anchor is dropped
        but checked: ``'QS-JAN'`` and ``'W-MON'`` are valid, ``'QS-XYZ'``,
        ``'MS-JAN'`` and ``'D-MON'`` are not (see :meth:`_validate_anchor`).

        Args:
            value: Frequency string (pandas code, literal name, or complex pandas string)

        Returns:
            Pandas frequency code string

        Raises:
            ValueError: If value format is not supported, or if its anchor is
                not valid for its base frequency

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
            >>> normalizer.normalize('YS-JAN')
            'Y'
        """
        # Vérification que la fréquence est du type spécifié
        if not isinstance(value, str):
            raise ValueError(f"Frequency must be a string, got {type(value)}")

        # Retourne un code pandas inchangé
        if value in self._pandas_to_literal:
            return value

        # Conversion du nom littéral en pandas (avec résolution d'alias)
        if value in self._literal_to_pandas:
            code = self._literal_to_pandas[value]
            return code

        # Tentative d'extraction de la fréquence de base via parse_frequency
        code = None
        try:
            parsed = parse_frequency(value)
            # Récursion seulement si la base est différente de la valeur d'entrée (évite boucle infinie)
            if parsed.freq != value:
                code = self.normalize(parsed.freq)
        except ValueError:
            pass

        # Validation de l'ancre hors du try : son message précis ne doit pas être
        # remplacé par l'erreur générique ci-dessous
        if code is not None:
            self._validate_anchor(value, code, parsed)
            return code

        # Renvoie une erreur si le code est inconnu
        raise ValueError(
            f"Unsupported frequency: {value}. "
            f"Supported frequencies: {list(self._literal_to_pandas.keys())} "
            f"or pandas codes: {list(self._pandas_to_literal.keys())}"
        )

    # Méthode de validation de l'ancre d'une chaîne de fréquence pandas
    @staticmethod
    def _validate_anchor(value: str, code: str, parsed: ParsedFrequency) -> None:
        """Check that the anchor of a parsed frequency string exists in pandas.

        The rules are those of ``pandas.tseries.frequencies.to_offset``:
        quarterly and yearly anchors are months (``'QS-JAN'``, ``'YE-DEC'``,
        ``'Q-DEC'``), weekly anchors are weekdays (``'W-MON'``), semi-monthly
        anchors are a day of the month behind a position (``'SMS-15'``: 2 to 27;
        ``'SME-15'``: 1 to 27), and every other frequency takes no anchor
        (``'MS-JAN'`` and ``'D-MON'`` are not pandas frequencies).

        Args:
            value: Original frequency string, quoted in the error message.
            code: Normalized base code of ``value``.
            parsed: Components of ``value`` given by ``parse_frequency``.

        Raises:
            ValueError: If the anchor is not valid for the base frequency.
        """
        # Extraction de l'ancre
        anchor = parsed.suffix
        if anchor is None:
            return

        # Ancres admises, par fréquence de base
        if code in ('Q', 'Y'):
            valid = anchor in MONTH_ABBREVIATIONS
            expected = f"a month among {', '.join(MONTH_ABBREVIATIONS)}"
        elif code == 'W':
            valid = anchor in WEEKDAY_ABBREVIATIONS
            expected = f"a weekday among {', '.join(WEEKDAY_ABBREVIATIONS)}"
        elif code == 'SM':
            # Jour du mois, borné par pandas : 2 à 27 pour un début, 1 à 27 pour une fin
            lowest = {'S': 2, 'E': 1}.get(parsed.position)
            valid = lowest is not None and anchor.isdigit() and lowest <= int(anchor) <= 27
            expected = "a day of the month (2 to 27 after 'SMS', 1 to 27 after 'SME')"
        else:
            # Aucune ancre admise
            raise ValueError(
                f"Unsupported frequency: {value}. "
                f"Base frequency '{code}' does not accept an anchor ('-{anchor}')."
            )

        # Message d'erreur quand l'ancre est invalide
        if not valid:
            raise ValueError(
                f"Unsupported frequency: {value}. Invalid anchor '{anchor}' "
                f"for base frequency '{code}': expected {expected}."
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
    def to_pandas_freq(self, frequency: str) -> str:
        """Convert frequency to a pandas frequency alias.

        The result is an alias pandas accepts (``pd.date_range``, ``to_offset``)
        without deprecation warning. Position, anchor and multiplier are kept.
        A frequency that has start and end variants (month, quarter, year,
        semi-month) but no position is given its **end** variant, as pandas did
        with the bare aliases that pandas 2.2 deprecated and pandas 3 removes:
        ``'M'`` → ``'ME'``, ``'Q'`` → ``'QE'``, ``'Y'`` → ``'YE'``, ``'SM'`` →
        ``'SME'``. To get the base code (``'M'``) rather than a pandas alias, use
        :meth:`normalize` or :meth:`to_code`.

        Args:
            frequency: Frequency in any supported format

        Returns:
            Pandas frequency alias

        Raises:
            ValueError: If frequency is not a string, is not supported, or has an
                anchor that does not exist in pandas

        Examples:
            >>> normalizer = FrequencyNormalizer()
            >>> normalizer.to_pandas_freq('monthly')
            'ME'
            >>> normalizer.to_pandas_freq('MS')
            'MS'
            >>> normalizer.to_pandas_freq('Q-DEC')
            'QE-DEC'
            >>> normalizer.to_pandas_freq('daily')
            'D'
        """
        # Vérification du type avant tout accès à un dictionnaire (une liste n'est pas hachable)
        if not isinstance(frequency, str):
            raise ValueError(f"Frequency must be a string, got {type(frequency)}")

        # Distinction suivant la nature de l'entrée
        if frequency in self._literal_to_pandas:
            parsed = ParsedFrequency(self.normalize(frequency), None, None, 1)
        elif frequency in self._pandas_to_literal:
            parsed = ParsedFrequency(frequency, None, None, 1)
        else:
            # Validation de la chaîne complète (base, ancre, multiplicateur), puis
            # parsing : la base normalisée remplace celle de l'entrée
            code = self.normalize(frequency)
            parsed = parse_frequency(frequency)._replace(freq=code)

        # Réassemblage (multiplicateur compris), variante fin par défaut : les alias
        # nus 'M', 'Q', 'Y' et 'SM' sont dépréciés
        return build_frequency_string(*parsed, default_position='E')

    # Conversion d'une fréquence en DateOffset
    def to_dateoffset(self, frequency: str) -> pd.DateOffset:
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
        # Conversion en fréquence pandas
        pandas_freq = self.to_pandas_freq(frequency=frequency)
        # Conversion en OffSet
        return to_offset(pandas_freq)

    # Méthode déterminant si une fréquence est plus élevée d'une autre
    def is_higher_frequency(self, freq1: FrequencyType, freq2: FrequencyType) -> bool:
        """Check if freq1 is a higher frequency than freq2.

        A multiplier lengthens the period: ``'MS'`` is a higher frequency than
        ``'2MS'``, and ``'2MS'`` a higher one than ``'QS'``. Without any
        multiplier the comparison follows the granularity order of the codes.
        With multipliers, nominal durations are compared, a business day
        counting for 7/5 of a calendar day: ``'2D'`` is a higher frequency than
        ``'2B'``, like ``'D'`` than ``'B'``. Two different codes of equal
        nominal duration (``'24h'`` and ``'D'``) are neither higher nor lower.

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
            >>> normalizer.is_higher_frequency('MS', '2MS')
            True
            >>> normalizer.is_higher_frequency('2MS', 'QS')
            True
            >>> normalizer.is_higher_frequency('2D', '2B')
            True
        """
        # Normalisation des fréquences et extraction de leurs multiplicateurs
        code1, multiplier1 = self.normalize_with_multiplier(freq1)
        code2, multiplier2 = self.normalize_with_multiplier(freq2)

        # Sans multiplicateur : ordre de granularité des codes
        if multiplier1 == multiplier2 == 1:
            return self._frequency_order.get(code1, 0) < self._frequency_order.get(code2, 0)

        # Même code : le plus petit multiplicateur donne la période la plus courte
        if code1 == code2:
            return multiplier1 < multiplier2

        # Codes différents : comparaison des durées nominales des périodes
        return multiplier1 * _NOMINAL_SECONDS[code1] < multiplier2 * _NOMINAL_SECONDS[code2]

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

