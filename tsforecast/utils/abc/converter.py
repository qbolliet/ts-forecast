
"""Base classes for temporal data normalization conversion.

This module provides abstract base classes for converting values
between different temporal units. Common functionality is shared across different
temporal types.
"""
# Importation des modules
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

# Facteurs de conversion
_CONVERSION_FACTORS_TO_SECONDS: Dict[str, float] = {
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
# Table des nombres de sous-périodes calendaires (clé : fréquence basse,
# fréquence haute). Les paires emboîtées (Y/Q/M, W/D) y sont exactes ; les
# autres (jours dans un mois, semaines dans une année) portent la valeur
# conventionnelle, aucune valeur exacte constante n'existant. Dans les deux
# cas la table corrige le ratio de durées, inexact par construction
# (365/30 = 12.17 mois dans une année).
_CALENDAR_SUBPERIODS: Dict[Tuple[str, str], int] = {
    ('Y', 'Q'): 4, ('Y', 'M'): 12, ('Y', 'SM'): 24, ('Y', 'W'): 52, ('Y', 'D'): 365,
    ('Q', 'M'): 3, ('Q', 'SM'): 6, ('Q', 'W'): 13, ('Q', 'D'): 91,
    ('M', 'SM'): 2, ('M', 'D'): 30,
    ('W', 'D'): 7,
}

# Classe abstraite de conversion
class TemporalConverter(ABC):
    """Abstract base class for temporal value conversions.

    This class provides a common interface for converting values between different
    temporal units (e.g., duration units, frequency units, period positions).

    Subclasses must implement the abstract methods to define specific conversion
    behavior for their temporal type.

    Examples:
        >>> class MyConverter(TemporalConverter):
        ...     def __init__(self):
        ...         self._conversion_factors = {'s': 1, 'min': 60, 'h': 3600}
        ...     def convert(self, value: float, from_unit: str, to_unit: str, **kwargs) -> float:
        ...         factor = self.get_conversion_factor(from_unit, to_unit)
        ...         return value * factor
        ...     def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        ...         return self._conversion_factors[to_unit] / self._conversion_factors[from_unit]
    """

    # Méthode de conversion d'une valeur entre deux unités temporelles
    @abstractmethod
    def convert(self, value: Any, from_unit: str, to_unit: str, **kwargs) -> Any:
        """Convert a value from one temporal unit to another.

        Args:
            value: Value to convert (can be numeric, Series, DataFrame, etc.)
            from_unit: Source temporal unit
            to_unit: Target temporal unit
            **kwargs: Additional conversion parameters (rounding, methods, etc.)

        Returns:
            Converted value in the target unit

        Raises:
            ValueError: If units are not supported or conversion is not possible
        """
        pass

    # Méthode de calcul du facteur de conversion entre deux unités temporelles
    @abstractmethod
    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """Get the conversion factor between two temporal units.

        This method calculates the multiplicative factor needed to convert
        from one unit to another. For example, converting from hours to seconds
        would return 3600.

        Args:
            from_unit: Source temporal unit
            to_unit: Target temporal unit

        Returns:
            Conversion factor (multiply value by this factor to convert)

        Raises:
            ValueError: If units are not supported or conversion is not possible
        """
        pass