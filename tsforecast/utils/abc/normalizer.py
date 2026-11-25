"""Base classes for temporal data normalization.

This module provides abstract base classes for normalizing temporal expressions
like frequencies, durations, and period positions. Common functionality is shared across different
temporal types.
"""
# Importation des modules
from abc import ABC, abstractmethod
from typing import Dict, Tuple

# Import de la fonction de parsing des chaînes de fréquence / durée
from ..parse.utils import parse_frequency
from .._constants import CALENDAR_SUBPERIODS, EXACT_CALENDAR_SUBPERIOD_PAIRS

# Classe abstraite de normalisation 
class TemporalNormalizer(ABC):
    """Abstract base class for normalizing temporal expressions.

    This class provides a common interface for normalizing different types of
    temporal expressions (frequencies, durations, etc.) between various
    representation formats (codes, literal names, etc.).

    Subclasses must implement the abstract methods to define specific normalization
    behavior for their temporal type.

    Examples:
        >>> class MyNormalizer(TemporalNormalizer):
        ...     def __init__(self):
        ...         self._code_to_literal = {'D': 'daily', 'W': 'weekly'}
        ...         self._literal_to_code = self._build_reverse_mapping(self._code_to_literal)
        ...     def normalize(self, value: str) -> str:
        ...         if value in self._code_to_literal:
        ...             return value
        ...         if value in self._literal_to_code:
        ...             return self._literal_to_code[value]
        ...         raise ValueError(f"Unknown value: {value}")
        ...     def to_literal(self, value: str) -> str:
        ...         code = self.normalize(value)
        ...         return self._code_to_literal[code]
        ...     def validate(self, value: str) -> bool:
        ...         try:
        ...             self.normalize(value)
        ...             return True
        ...         except ValueError:
        ...             return False
    """

    # Méthode de normalisation de la représentation de la valeur temporelle
    @abstractmethod
    def normalize(self, value: str) -> str:
        """Normalize any representation to standard format.

        Args:
            value: Value to normalize (code or literal name)

        Returns:
            Normalized value in standard format (typically a code)

        Raises:
            ValueError: If value format is not supported
        """
        pass

    # Méthode de conversion de la valeur temporelle dans son expression littéraire
    @abstractmethod
    def to_literal(self, value: str) -> str:
        """Convert value to user-friendly literal name.

        Args:
            value: Value in any supported format

        Returns:
            User-friendly literal name

        Raises:
            ValueError: If value format is not supported
        """
        pass

    # Méthode de validation du support par la classe de la valeur temporelle
    @abstractmethod
    def validate(self, value: str) -> bool:
        """Validate if a value is supported.

        Args:
            value: Value to validate

        Returns:
            True if value is valid and supported, False otherwise
        """
        try:
            # Tentative de normalisation de la position
            self.normalize(value)
            return True
        except ValueError:
            return False

    # Méthode de normalisation d'une valeur avec extraction de son multiplicateur
    def normalize_with_multiplier(self, value: str) -> Tuple[str, int]:
        """Normalize a value and extract its leading multiplier.

        ``normalize`` returns the code alone (``'2MS'`` -> ``'M'``); this
        method keeps the multiplier next to it, for the callers that compare
        or scale quantities (``'2D'`` is twice ``'D'``).

        Args:
            value: Value to normalize, possibly multiplied (``'2D'``, ``'3QS-FEB'``)
                or a literal name without multiplier (``'day'``).

        Returns:
            Tuple ``(code, multiplier)``, the multiplier being 1 when absent.

        Raises:
            ValueError: If value format is not supported

        Examples:
            >>> from tsforecast.utils.frequency.normalizer import FrequencyNormalizer
            >>> FrequencyNormalizer().normalize_with_multiplier('2MS')
            ('M', 2)
            >>> FrequencyNormalizer().normalize_with_multiplier('daily')
            ('D', 1)
        """
        # Extraction du multiplier
        try:
            multiplier = parse_frequency(value).multiplier
        except (ValueError, TypeError):
            # Nom littéral ne se parsant pas ('business_day') ou valeur invalide,
            # cette dernière étant rejetée par normalize
            multiplier = 1
        return self.normalize(value), multiplier

    # Méthode auxiliaire de comparaison des longueurs de deux unités multipliées de codes différents
    @staticmethod
    def _compare_multiplied_lengths(
        code1: str, multiplier1: int, code2: str, multiplier2: int, seconds: Dict[str, float]
    ) -> int:
        """Compare the lengths of two multiplied units of different codes.

        Nested calendar pairs (3 months in a quarter, 7 days in a week, ...) are
        compared exactly, in sub-periods: ``'3M'`` and ``'Q'`` have the same
        length. Other pairs are compared by their nominal lengths in seconds.

        Args:
            code1: Normalized code of the first unit.
            multiplier1: Multiplier of the first unit.
            code2: Normalized code of the second unit, different from ``code1``.
            multiplier2: Multiplier of the second unit.
            seconds: Nominal length of each code, in seconds.

        Returns:
            -1 if the first unit is shorter, 1 if it is longer, 0 if both have
            the same length.

        Examples:
            >>> from tsforecast.utils._constants import CONVERSION_FACTORS_TO_SECONDS as seconds
            >>> TemporalNormalizer._compare_multiplied_lengths('M', 3, 'Q', 1, seconds)
            0
            >>> TemporalNormalizer._compare_multiplied_lengths('D', 2, 'W', 1, seconds)
            -1
        """
        # Paire calendaire emboîtée : comparaison exacte en sous-périodes du code le plus fin
        if (code2, code1) in EXACT_CALENDAR_SUBPERIOD_PAIRS:
            length1, length2 = multiplier1, multiplier2 * CALENDAR_SUBPERIODS[(code2, code1)]
        elif (code1, code2) in EXACT_CALENDAR_SUBPERIOD_PAIRS:
            length1, length2 = multiplier1 * CALENDAR_SUBPERIODS[(code1, code2)], multiplier2
        # Autres paires : durées nominales
        else:
            length1, length2 = multiplier1 * seconds[code1], multiplier2 * seconds[code2]
        return (length1 > length2) - (length1 < length2)

    # Méthode auxiliaire de construction du mapping inverse entre les codes temporels et leur expression littéraire
    @staticmethod
    def _build_reverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
        """Build reverse mapping from a given dictionary.

        This is a utility method to create the inverse mapping between
        codes and literal names.

        Args:
            mapping: Original dictionary to reverse

        Returns:
            Reversed dictionary where keys become values and vice versa

        Examples:
            >>> mapping = {'D': 'daily', 'W': 'weekly'}
            >>> TemporalNormalizer._build_reverse_mapping(mapping)
            {'daily': 'D', 'weekly': 'W'}
        """
        return {v: k for k, v in mapping.items()}