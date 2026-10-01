
"""Base classes for temporal data normalization conversion.

This module provides abstract base classes for converting values
between different temporal units. Common functionality is shared across different
temporal types.
"""
# Importation des modules
from abc import ABC, abstractmethod
from typing import Any

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