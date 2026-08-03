"""Duration normalization utilities for time series processing.

This module provides the DurationNormalizer class to handle different duration
representations including codes and literal names.
"""
# Importation des modules
from typing import Union

# Import de la classe parente
from ..abc.normalizer import TemporalNormalizer
# Importation des types
from .types import DurationType, UserDurationType
# Importation de la fonction de parsing des fréquences pandas / durées
from ..parse import parse_frequency


# Classe de normalisation des durées
class DurationNormalizer(TemporalNormalizer):
    """Centralized duration normalization and conversion utility.

    This class handles conversions between different duration representations:
    - Duration codes (D, W, M, Q, Y, etc.)
    - User-friendly literal names

    Examples:
        >>> normalizer = DurationNormalizer()
        >>> normalizer.normalize('hour')
        'h'
        >>> normalizer.to_literal('D')
        'day'
        >>> normalizer.normalize_duration('monthly')
        'M'
    """

    # Initialisation
    def __init__(self):
        """Initialize duration mappings."""
        # Mapping des codes de durées vers des noms littéraux
        self._code_to_literal = {
            'ns': 'nanosecond',
            'us': 'microsecond',
            'ms': 'millisecond',
            's': 'second',
            'min': 'minute',
            'h': 'hour',
            'D': 'day',
            'B': 'business_day',
            'W': 'week',
            'SM': 'semi_month',
            'M': 'month',
            'Q': 'quarter',
            'Y': 'year'
        }

        # Mapping inverse pour les conversions (utilisation de la méthode héritée)
        self._literal_to_code = self._build_reverse_mapping(self._code_to_literal)

        # Ordre des durées pour les comparaisons (du plus court au plus long)
        self._duration_order = {
            'ns': 1,
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

    # Méthode de normalisation de la durée
    def normalize(self, value: Union[DurationType, UserDurationType]) -> DurationType:
        """Normalize any duration representation to duration code.

        Args:
            value: Duration string (code or literal name)

        Returns:
            Duration code string

        Raises:
            ValueError: If value format is not supported

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.normalize('hour')
            'h'
            >>> normalizer.normalize('D')
            'D'
        """
        # Vérification que la durée est du type spécifié
        if not isinstance(value, str):
            raise ValueError(f"Duration must be a string, got {type(value)}")

        # Retourne un code inchangé
        if value in self._code_to_literal:
            return value

        # Conversion du nom littéral en code
        if value in self._literal_to_code:
            return self._literal_to_code[value]

        # Tentative d'extraction de la fréquence de base via parse_frequency
        try:
            base, _, _ = parse_frequency(value)
            # Récursion seulement si la base est différente de la valeur d'entrée (évite boucle infinie)
            if base != value:
                return self.normalize(base)
        except ValueError:
            pass

        # Renvoie une erreur si le code est inconnu
        raise ValueError(
            f"Unsupported duration: {value}. "
            f"Supported durations: {list(self._literal_to_code.keys())} "
            f"or duration codes: {list(self._code_to_literal.keys())}"
        )

    # Méthode de conversion en expression littéraire
    def to_literal(self, duration: DurationType) -> UserDurationType:
        """Convert duration to literal name.

        Implementation of TemporalNormalizer.to_literal() for durations.

        Args:
            duration: Duration in any supported format

        Returns:
            Literal duration name

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.to_literal('h')
            'hour'
            >>> normalizer.to_literal('Q')
            'quarter'
        """
        # Normalisation de la durée
        duration_code = self.normalize(duration)
        # Conversion dans son nom littéral
        return self._code_to_literal.get(duration_code, duration_code)

    # Méthode de validation du support de la durée
    def validate(self, value: Union[DurationType, UserDurationType]) -> bool:
        """Validate if a duration is supported.

        Args:
            duration: Duration to validate

        Returns:
            True if duration is valid and supported

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.validate('day')
            True
            >>> normalizer.validate('invalid_dur')
            False
        """
        try:
            # Tentative de normalisation de la durée
            self.normalize(value)
            return True
        except ValueError:
            return False

    # Conversion d'une durée dans son expression code
    def to_code(self, duration: Union[DurationType, UserDurationType]) -> DurationType:
        """Convert duration to duration code.

        Args:
            duration: Duration in any supported format

        Returns:
            Duration code

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.to_code('hour')
            'h'
        """
        return self.normalize(duration)

    # Méthode déterminant si une durée est plus longue qu'une autre
    def is_longer_duration(self, dur1: DurationType, dur2: DurationType) -> bool:
        """Check if dur1 is a longer duration than dur2.

        Args:
            dur1: First duration
            dur2: Second duration

        Returns:
            True if dur1 is longer duration than dur2

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.is_longer_duration('month', 'day')
            True
            >>> normalizer.is_longer_duration('week', 'quarter')
            False
        """
        # Conversion en code
        code1 = self.to_code(dur1)
        code2 = self.to_code(dur2)

        # Extraction de l'ordre associé à chaque durée
        order1 = self._duration_order.get(code1, 0)
        order2 = self._duration_order.get(code2, 0)

        return order1 > order2

    # Méthode de vérification que deux expressions de durées sont compatibles
    def are_compatible_durations(self, dur1: DurationType, dur2: DurationType) -> bool:
        """Check if two durations are compatible for conversion.

        Args:
            dur1: First duration
            dur2: Second duration

        Returns:
            True if durations can be converted between each other

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.are_compatible_durations('day', 'month')
            True
            >>> normalizer.are_compatible_durations('business_day', 'month')
            True
        """
        try:
            # Tentative de normalisation de chaque durée
            self.normalize(dur1)
            self.normalize(dur2)
            return True
        except ValueError:
            return False
