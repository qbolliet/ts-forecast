"""Duration conversion utilities for time series processing.

This module provides the DurationConverter class to handle conversions between
different duration units with optional rounding.
"""
# Importation des modules
import math
from typing import Union, Literal, Optional

# Import de la classe parente
from ..abc.converter import TemporalConverter

# Import du normalizer
from .normalizer import DurationNormalizer, DurationType, UserDurationType

# Types pour le rounding
RoundingType = Optional[Literal['floor', 'ceil']]


# Classe de conversion entre durées
class DurationConverter(TemporalConverter):
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
        from_unit: Union[DurationType, UserDurationType],
        to_unit: Union[DurationType, UserDurationType],
        rounding: RoundingType = None,
        **kwargs
    ) -> float:
        """Convert a value from one duration unit to another.

        Implementation of TemporalConverter.convert() for durations.

        Args:
            value: Numeric value to convert
            from_unit: Source duration unit
            to_unit: Target duration unit
            rounding: Optional rounding mode ('floor', 'ceil', or None for no rounding)
            **kwargs: Additional parameters (unused for duration conversion)

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
        from_code = self._normalizer.normalize_duration(from_unit)
        to_code = self._normalizer.normalize_duration(to_unit)

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
        from_unit: Union[DurationType, UserDurationType],
        to_unit: Union[DurationType, UserDurationType]
    ) -> float:
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
            >>> converter = DurationConverter()
            >>> converter.get_conversion_factor('hour', 'minute')
            60.0
            >>> converter.get_conversion_factor('day', 'hour')
            24.0
        """
        # Normalisation des durées
        from_code = self._normalizer.normalize_duration(from_unit)
        to_code = self._normalizer.normalize_duration(to_unit)

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
