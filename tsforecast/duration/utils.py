"""Duration normalization and conversion utilities for time series processing.

This module provides centralized functions to handle different duration representations
including codes, literal names, and conversions between different duration units with
optional rounding.
"""
# Importation des modules
import math
from typing import Union, Literal, Optional, Dict
from ..common.base import TemporalNormalizer

# Types supportés pour les durées
DurationType = Literal['ns', 'us', 'ms', 's', 'min', 'h', 'D', 'B', 'W', 'SM', 'M', 'Q', 'Y']
UserDurationType = Literal[
    'nanosecond', 'microsecond', 'millisecond', 'second', 'minute', 'hour',
    'day', 'business_day', 'week', 'semi_month', 'month', 'quarter', 'year'
]
RoundingType = Optional[Literal['floor', 'ceil']]


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

    # Implémentation des méthodes abstraites de TemporalNormalizer
    def normalize(self, value: str) -> str:
        """Normalize any duration representation to duration code.

        Implementation of TemporalNormalizer.normalize() for durations.

        Args:
            value: Duration string (code or literal name)

        Returns:
            Duration code string

        Raises:
            ValueError: If duration format is not supported

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.normalize('hour')
            'h'
            >>> normalizer.normalize('D')
            'D'
        """
        return self.normalize_duration(value)

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
        duration_code = self.normalize_duration(duration)
        # Conversion dans son nom littéral
        return self._code_to_literal.get(duration_code, duration_code)

    def validate(self, value: str) -> bool:
        """Validate if a duration is supported.

        Implementation of TemporalNormalizer.validate() for durations.

        Args:
            value: Duration to validate

        Returns:
            True if duration is valid and supported

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.validate('day')
            True
            >>> normalizer.validate('invalid_duration')
            False
        """
        return self.validate_duration(value)

    # Méthode de normalisation des durées dans leur expression code
    def normalize_duration(self, duration: Union[DurationType, UserDurationType]) -> DurationType:
        """Normalize any duration representation to duration code.

        Args:
            duration: Duration string (code or literal name)

        Returns:
            Duration code string

        Raises:
            ValueError: If duration format is not supported

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.normalize_duration('hour')
            'h'
            >>> normalizer.normalize_duration('D')
            'D'
        """
        # Vérification que la durée est du type spécifié
        if not isinstance(duration, str):
            raise ValueError(f"Duration must be a string, got {type(duration)}")

        # Retourne un code inchangé
        if duration in self._code_to_literal:
            return duration

        # Conversion du nom littéral en code
        if duration in self._literal_to_code:
            return self._literal_to_code[duration]

        # Renvoie une erreur si le code est inconnu
        raise ValueError(
            f"Unsupported duration: {duration}. "
            f"Supported durations: {list(self._literal_to_code.keys())} "
            f"or duration codes: {list(self._code_to_literal.keys())}"
        )

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
        return self.normalize_duration(duration)

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
        code1 = self.to_code(dur1)
        code2 = self.to_code(dur2)

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
            self.normalize_duration(dur1)
            self.normalize_duration(dur2)
            return True
        except ValueError:
            return False

    # Méthode de vérification qu'une durée est supportée
    def validate_duration(self, duration: DurationType) -> bool:
        """Validate if a duration is supported.

        Args:
            duration: Duration to validate

        Returns:
            True if duration is valid and supported

        Examples:
            >>> normalizer = DurationNormalizer()
            >>> normalizer.validate_duration('day')
            True
            >>> normalizer.validate_duration('invalid_dur')
            False
        """
        try:
            # Tentative de normalisation de la durée
            self.normalize_duration(duration)
            return True
        except ValueError:
            return False


# Classe de conversion entre durées
class DurationConverter:
    """Handle conversions between different duration units.

    This class manages duration conversions using conversion factors relative
    to a reference unit (seconds). Supports optional rounding to floor or ceiling.

    Examples:
        >>> converter = DurationConverter()
        >>> converter.convert(2.5, 'hour', 'minute')
        150.0
        >>> converter.convert(90, 'minute', 'hour', rounding='ceil')
        2
        >>> converter.convert(1.5, 'day', 'hour')
        36.0
    """

    # Initialisation
    def __init__(self):
        """Initialize conversion factors and normalizer."""
        # Instance du normaliseur de durées
        self._normalizer = DurationNormalizer()

        # Facteurs de conversion vers l'unité de référence (secondes)
        self._conversion_factors = {
            'ns': 1e-9,
            'us': 1e-6,
            'ms': 1e-3,
            's': 1,
            'min': 60,
            'h': 3600,
            'D': 86400,  # 24 * 3600
            'B': 86400,  # Même que 'D' pour la conversion
            'W': 604800,  # 7 * 24 * 3600
            'SM': 1296000,  # 15 * 24 * 3600 (approximation)
            'M': 2592000,  # 30 * 24 * 3600 (approximation)
            'Q': 7776000,  # 90 * 24 * 3600 (approximation)
            'Y': 31536000,  # 365 * 24 * 3600 (approximation)
        }

    # Méthode principale de conversion
    def convert(
        self,
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
            rounding: Optional rounding mode ('floor', 'ceil', or None for no rounding)

        Returns:
            Converted value, optionally rounded

        Raises:
            ValueError: If duration units are not supported

        Examples:
            >>> converter = DurationConverter()
            >>> converter.convert(2.5, 'hour', 'minute')
            150.0
            >>> converter.convert(90, 'minute', 'hour')
            1.5
            >>> converter.convert(90, 'minute', 'hour', rounding='ceil')
            2
            >>> converter.convert(90, 'minute', 'hour', rounding='floor')
            1
        """
        # Normalisation des durées
        from_code = self._normalizer.normalize_duration(from_duration)
        to_code = self._normalizer.normalize_duration(to_duration)

        # Récupération du facteur de conversion
        conversion_factor = self.get_conversion_factor(from_code, to_code)

        # Application de la conversion
        result = value * conversion_factor

        # Application de l'arrondi si spécifié
        if rounding is not None:
            result = self._round_result(result, rounding)

        return result

    # Méthode de récupération du facteur de conversion
    def get_conversion_factor(
        self,
        from_duration: Union[DurationType, UserDurationType],
        to_duration: Union[DurationType, UserDurationType]
    ) -> float:
        """Get the conversion factor between two duration units.

        Args:
            from_duration: Source duration unit
            to_duration: Target duration unit

        Returns:
            Conversion factor to multiply by source value

        Raises:
            ValueError: If duration units are not supported

        Examples:
            >>> converter = DurationConverter()
            >>> converter.get_conversion_factor('hour', 'minute')
            60.0
            >>> converter.get_conversion_factor('day', 'hour')
            24.0
        """
        # Normalisation des durées
        from_code = self._normalizer.normalize_duration(from_duration)
        to_code = self._normalizer.normalize_duration(to_duration)

        # Vérification que les facteurs existent
        if from_code not in self._conversion_factors:
            raise ValueError(f"No conversion factor for duration: {from_code}")
        if to_code not in self._conversion_factors:
            raise ValueError(f"No conversion factor for duration: {to_code}")

        # Calcul du facteur via l'unité de référence (secondes)
        from_to_seconds = self._conversion_factors[from_code]
        to_to_seconds = self._conversion_factors[to_code]

        return from_to_seconds / to_to_seconds

    # Méthode auxiliaire d'application de l'arrondi
    def _round_result(self, value: float, rounding: Literal['floor', 'ceil']) -> float:
        """Apply rounding to the conversion result.

        Args:
            value: Value to round
            rounding: Rounding mode ('floor' or 'ceil')

        Returns:
            Rounded value

        Examples:
            >>> converter = DurationConverter()
            >>> converter._round_result(1.5, 'floor')
            1
            >>> converter._round_result(1.5, 'ceil')
            2
        """
        if rounding == 'floor':
            return math.floor(value)
        elif rounding == 'ceil':
            return math.ceil(value)
        else:
            return value


# Instance globale pour faciliter l'utilisation
_normalizer = DurationNormalizer()
_converter = DurationConverter()


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
    return _normalizer.normalize_duration(duration)


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
    return _normalizer.validate_duration(duration)


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
