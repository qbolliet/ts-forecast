"""Period position normalization utilities for time series processing.

This module provides the PeriodPositionNormalizer class to handle different period
position representations (start vs end of period) for time series data.
"""
# Importation des modules
from typing import Union, Literal, Optional, Tuple
import re
import warnings

# Import de la classe parente
from ..abc.normalizer import TemporalNormalizer

# Types supportés pour les positions de période
PositionType = Literal['S', 'E']
UserPositionType = Literal['start', 'end']


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
        >>> normalizer.extract_position_from_offset('MS')
        'S'
        >>> normalizer.combine_frequency_position('M', 'start')
        'MS'
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

        # Fréquences qui supportent les suffixes de position S/E
        self._position_aware_frequencies = ['M', 'Q', 'Y', 'W', 'B']

        # Mapping des anciens codes pandas vers les nouveaux (pandas a changé ME->M, MS->M, etc.)
        self._legacy_offset_mapping = {
            'MS': ('M', 'S'),
            'ME': ('M', 'E'),
            'M': ('M', 'E'),  # Par défaut M = end
            'QS': ('Q', 'S'),
            'QE': ('Q', 'E'),
            'Q': ('Q', 'E'),  # Par défaut Q = end
            'YS': ('Y', 'S'),
            'YE': ('Y', 'E'),
            'Y': ('Y', 'E'),  # Par défaut Y = end
            'AS': ('Y', 'S'),
            'AE': ('Y', 'E'),
            'A': ('Y', 'E'),  # A = alias pour Y
            'WS': ('W', 'S'),
            'WE': ('W', 'E'),
            'W': ('W', 'E'),  # Par défaut W = end
            'BS': ('B', 'S'),
            'BE': ('B', 'E'),
            'B': ('B', 'E')   # Par défaut B = end
        }

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

    # Méthode d'extraction de la position depuis un DateOffset pandas
    def extract_position_from_offset(self, offset_str: str) -> PositionType:
        """Extract period position from pandas DateOffset string.

        Extracts the position component (start/end) from pandas frequency codes
        like 'MS', 'ME', 'QS', 'QE', etc.

        Args:
            offset_str: Pandas DateOffset string (e.g., 'MS', 'QE', 'M')

        Returns:
            Position code ('S' or 'E')

        Raises:
            ValueError: If offset string doesn't contain position information

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.extract_position_from_offset('MS')
            'S'
            >>> normalizer.extract_position_from_offset('QE')
            'E'
            >>> normalizer.extract_position_from_offset('M')
            'E'
        """
        # Nettoyage de l'offset (suppression des multiplicateurs comme '2MS')
        clean_offset = re.sub(r'^\d+', '', offset_str)

        # Recherche dans le mapping des offsets connus
        if clean_offset in self._legacy_offset_mapping:
            _, position = self._legacy_offset_mapping[clean_offset]
            return position

        # Si pas de suffixe explicite, retourne 'E' par défaut
        return 'E'

    # Méthode de décomposition d'un DateOffset en fréquence et position
    def decompose_offset(self, offset_str: str) -> Tuple[str, PositionType]:
        """Decompose pandas DateOffset into frequency and position components.

        Args:
            offset_str: Pandas DateOffset string (e.g., 'MS', 'QE', 'M')

        Returns:
            Tuple of (frequency_code, position_code)

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.decompose_offset('MS')
            ('M', 'S')
            >>> normalizer.decompose_offset('QE')
            ('Q', 'E')
            >>> normalizer.decompose_offset('M')
            ('M', 'E')
        """
        # Nettoyage de l'offset (suppression des multiplicateurs comme '2MS')
        multiplier_match = re.match(r'^(\d+)(.+)', offset_str)
        if multiplier_match:
            clean_offset = multiplier_match.group(2)
        else:
            clean_offset = offset_str

        # Recherche dans le mapping des offsets connus
        if clean_offset in self._legacy_offset_mapping:
            freq, position = self._legacy_offset_mapping[clean_offset]
            return (freq, position)

        # Si pas trouvé, retourne l'offset tel quel avec position 'E' par défaut
        return (clean_offset, 'E')

    # Méthode de combinaison d'une fréquence et d'une position en DateOffset
    def combine_frequency_position(
        self,
        frequency: str,
        position: Optional[Union[PositionType, UserPositionType]]
    ) -> str:
        """Combine frequency and position into pandas DateOffset string.

        Args:
            frequency: Frequency code (e.g., 'M', 'Q', 'Y')
            position: Position code or literal ('S', 'E', 'start', 'end')

        Returns:
            Pandas DateOffset string (e.g., 'MS', 'QE')

        Examples:
            >>> normalizer = PeriodPositionNormalizer()
            >>> normalizer.combine_frequency_position('M', 'start')
            'MS'
            >>> normalizer.combine_frequency_position('Q', 'E')
            'QE'
            >>> normalizer.combine_frequency_position('D', 'end')
            'D'
        """
        
        # Si la fréquence ne supporte pas les positions, retourne la fréquence telle quelle
        if frequency not in self._position_aware_frequencies:
            return frequency
        else:
            # Si la position n'est pas spécifié, retourne la fréquence telle quelle
            if position is None:
                # Warning
                warnings.warn(f"No 'position' specified with 'frequency' : '{frequency}'")
                return frequency
            else:
                # Normalisation de la position
                position_code = self.normalize(position)
                # Combinaison de la fréquence et de la position
                return f"{frequency}{position_code}"

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

