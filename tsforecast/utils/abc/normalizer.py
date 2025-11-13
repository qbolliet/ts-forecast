"""Base classes for temporal data normalization.

This module provides abstract base classes for normalizing temporal expressions
like frequencies, durations, and period positions. Common functionality is shared across different
temporal types.
"""
# Importation des modules
from abc import ABC, abstractmethod
from typing import Dict

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