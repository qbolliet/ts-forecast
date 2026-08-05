"""Period position normalization utilities for time series processing.

This module provides the PeriodPositionNormalizer class to handle different period
position representations (start vs end of period) for time series data.
"""
# Importation des modules
from typing import Union

# Import de la classe parente
from ..abc.normalizer import TemporalNormalizer
# Importation des types
from .types import PositionType, UserPositionType


# Classe de normalisation des positions de période
class PeriodPositionNormalizer(TemporalNormalizer):
    """Centralized period position normalization utility.

    This class handles conversions between different period position representations:
    - Position codes (S for start, E for end)
    - User-friendly literal names (start, end)
    - Pandas DateOffset codes with position suffixes (MS, ME, QS, QE, etc.)

    Examples:
        >>> normalizer = PeriodPositionNormalizer()
        >>> normalizer.normalize('end')
        'E'
        >>> normalizer.to_literal('S')
        'start'
    """

    # Initialisation
    def __init__(self):
        """Initialize position mappings."""
        # Mapping des codes de position vers des noms littéraux
        self._code_to_literal = {
            'S': 'start',
            'E': 'end'
        }

        # Mapping inverse pour les conversions (utilisation de la méthode héritée)
        self._literal_to_code = self._build_reverse_mapping(self._code_to_literal)

    # Implémentation des méthodes abstraites de TemporalNormalizer
    # Méthode de normalisation des positions dans leur expression code
    def normalize(self, value: Union[PositionType, UserPositionType]) -> PositionType:
        """Normalize any position representation to position code.

        Implementation of TemporalNormalizer.normalize() for period positions.

        Args:
            value: Position string (code or literal name)

        Returns:
            Position code string ('S' or 'E')

        Raises:
            ValueError: If position format is not supported

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.normalize('start')
            'S'
            >>> normalizer.normalize('E')
            'E'
        """
        # Vérification que la position est du type spécifié
        if not isinstance(value, str):
            raise ValueError(f"Position must be a string, got {type(value)}")

        # Retourne un code inchangé
        if value in self._code_to_literal:
            return value

        # Conversion du nom littéral en code
        if value in self._literal_to_code:
            return self._literal_to_code[value]

        # Renvoie une erreur si le code est inconnu
        raise ValueError(
            f"Unsupported position: {value}. "
            f"Supported positions: {list(self._literal_to_code.keys())} "
            f"or position codes: {list(self._code_to_literal.keys())}"
        )

    # Méthode de conversion en nom littéraire
    def to_literal(self, position: PositionType) -> UserPositionType:
        """Convert position to literal name.

        Implementation of TemporalNormalizer.to_literal() for positions.

        Args:
            position: Position in any supported format

        Returns:
            Literal position name ('start' or 'end')

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.to_literal('S')
            'start'
            >>> normalizer.to_literal('E')
            'end'
        """
        # Normalisation de la position
        position_code = self.normalize(position)
        # Conversion dans son nom littéral
        return self._code_to_literal.get(position_code, position_code)

    # Conversion d'une position dans son expression code
    def to_code(self, position: Union[PositionType, UserPositionType]) -> PositionType:
        """Convert position to position code.

        Args:
            position: Position in any supported format

        Returns:
            Position code ('S' or 'E')

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.to_code('start')
            'S'
            >>> normalizer.to_code('end')
            'E'
        """
        return self.normalize(position)

    # Méthode de validation de la position
    def validate(self, value: Union[PositionType, UserPositionType]) -> bool:
        """Validate if a position is supported.

        Args:
            position: Position to validate

        Returns:
            True if position is valid and supported

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.validate('start')
            True
            >>> normalizer.validate('invalid_pos')
            False
        """
        try:
            # Tentative de normalisation de la position
            self.normalize(value)
            return True
        except ValueError:
            return False

    # Méthode de conversion d'une position vers son opposé
    def flip_position(self, position: Union[PositionType, UserPositionType]) -> PositionType:
        """Flip position to its opposite (start <-> end).

        Args:
            position: Position to flip

        Returns:
            Opposite position code

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.flip_position('start')
            'E'
            >>> normalizer.flip_position('E')
            'S'
        """
        # Normalisation de la position
        position_code = self.normalize(position)

        # Retour de l'opposé
        if position_code == 'S':
            return 'E'
        else:
            return 'S'

