# Importation des modules
# Modules de base
from typing import Union, Literal, Tuple, Optional
import pandas as pd

# Importation des modules du package
from .normalizer import FrequencyNormalizer
from .types import FrequencyType, UserFrequencyType
from ..parse.utils import parse_frequency

# Instance globale pour faciliter l'utilisation
_normalizer = FrequencyNormalizer()

# Fonctions de commodité pour accès direct
# Fonction de normalisation de la fréquence
def normalize_frequency(
    frequency: Union[str, FrequencyType, UserFrequencyType],
    return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
) -> Union[str, Tuple[str, Optional[str], Optional[str]]]:
    """Normalize frequency with configurable output format.

    This function provides flexible frequency normalization with multiple
    output formats to support different use cases while maintaining
    backward compatibility.

    Args:
        frequency: Frequency to normalize (pandas code, literal name, or complex string)
        return_format: Output format level (default: 'base' for backward compatibility):
            - 'base': Base frequency only → 'Q'
            - 'with_position': Base + position if present → 'QE'
            - 'full': Complete validated string → 'QE-DEC'
            - 'components': Tuple (base, position, anchor) → ('Q', 'E', 'DEC')

    Returns:
        Normalized frequency in requested format:
        - 'base', 'with_position', 'full': str
        - 'components': Tuple[str, Optional[str], Optional[str]]

    Raises:
        ValueError: If frequency is invalid or return_format is unsupported

    Examples:
        >>> # Default: backward compatible base extraction
        >>> normalize_frequency('QE-DEC')
        'Q'
        >>> normalize_frequency('monthly')
        'M'

        >>> # With position: useful for resampling
        >>> normalize_frequency('QE-DEC', return_format='with_position')
        'QE'
        >>> normalize_frequency('MS', return_format='with_position')
        'MS'

        >>> # Full: validated original string
        >>> normalize_frequency('QE-DEC', return_format='full')
        'QE-DEC'

        >>> # Components: complete parsing
        >>> normalize_frequency('QE-DEC', return_format='components')
        ('Q', 'E', 'DEC')
        >>> normalize_frequency('MS', return_format='components')
        ('M', 'S', None)
        >>> normalize_frequency('D', return_format='components')
        ('D', None, None)
    """
    if return_format == 'base':
        # Comportement actuel (backward compatible)
        return _normalizer.normalize(frequency)

    elif return_format == 'components':
        # Décomposition complète via parse_frequency
        try:
            base, position, suffix = parse_frequency(frequency)
            normalized_base = _normalizer.normalize(base)
            return (normalized_base, position, suffix)
        except ValueError:
            # Fallback pour les noms littéraux ('daily', 'monthly', etc.)
            normalized_base = _normalizer.normalize(frequency)
            return (normalized_base, None, None)

    elif return_format == 'with_position':
        # Base + position si présente
        try:
            base, position, _ = parse_frequency(frequency)
            normalized_base = _normalizer.normalize(base)
            if position:
                return f"{normalized_base}{position}"
            return normalized_base
        except ValueError:
            return _normalizer.normalize(frequency)

    elif return_format == 'full':
        # Validation + retour de la chaîne complète
        try:
            base, _, _ = parse_frequency(frequency)
            _normalizer.normalize(base)  # Validation seulement
        except ValueError:
            _normalizer.normalize(frequency)  # Validation via normalize
        return frequency

    else:
        raise ValueError(
            f"Invalid return_format: {return_format}. "
            f"Must be one of: 'base', 'with_position', 'full', 'components'"
        )

# Fonction de conversion en expression littéraire
def to_literal(frequency: Union[FrequencyType, UserFrequencyType]) -> str:
    """Convert frequency to user-friendly literal name.

    Args:
        frequency: Frequency in pandas code or literal format to convert.

    Returns:
        User-friendly literal frequency name.

    Examples:
        >>> to_literal('M')
        'monthly'
        >>> to_literal('Q')
        'quarterly'
    """
    return _normalizer.to_literal(frequency)

# Fonction de conversion en code
def to_code(frequency: Union[FrequencyType, UserFrequencyType]) -> str:
    """Convert frequency to frequency code.

    Args:
        frequency: Frequency in any supported format (code or literal name).

    Returns:
        Frequency code string.

    Examples:
        >>> to_code('monthly')
        'M'
        >>> to_code('D')
        'D'
    """
    return _normalizer.to_code(frequency)

# Fonction de conversion en code de fréquence pandas
def to_pandas_freq(frequency: Union[FrequencyType, UserFrequencyType]) -> str:
    """Convert frequency to pandas frequency code.

    Args:
        frequency: Frequency in any supported format (pandas code or literal name).

    Returns:
        Pandas frequency code string.

    Examples:
        >>> to_pandas_freq('monthly')
        'M'
        >>> to_pandas_freq('D')
        'D'
    """
    return _normalizer.to_pandas_freq(frequency)

# Fonction de conversion en dateoffset
def to_dateoffset(frequency: Union[FrequencyType, UserFrequencyType]) -> pd.DateOffset:
    """Convert frequency to pandas DateOffset object.

    Args:
        frequency: Frequency string to convert to DateOffset.

    Returns:
        Pandas DateOffset object corresponding to the frequency.

    Examples:
        >>> offset = to_dateoffset('monthly')
        >>> isinstance(offset, pd.DateOffset)
        True
    """
    return _normalizer.to_dateoffset(frequency)

# Fonction de détection d'une fréquence plus élevée
def is_higher_frequency(freq1: Union[FrequencyType, UserFrequencyType],
                       freq2: Union[FrequencyType, UserFrequencyType]) -> bool:
    """Check if freq1 is higher frequency than freq2.

    Higher frequency means more granular time periods (e.g., daily vs monthly).

    Args:
        freq1: First frequency to compare.
        freq2: Second frequency to compare.

    Returns:
        True if freq1 has higher frequency (more granular) than freq2.

    Examples:
        >>> is_higher_frequency('daily', 'monthly')
        True
        >>> is_higher_frequency('quarterly', 'weekly')
        False
    """
    return _normalizer.is_higher_frequency(freq1, freq2)

# Fonction de validation de la fréquence
def validate_frequency(frequency: Union[FrequencyType, UserFrequencyType]) -> bool:
    """Validate if frequency is supported by the normalizer.

    Args:
        frequency: Frequency string to validate.

    Returns:
        True if frequency is valid and supported, False otherwise.

    Examples:
        >>> validate_frequency('daily')
        True
        >>> validate_frequency('M')
        True
        >>> validate_frequency('invalid_freq')
        False
    """
    return _normalizer.validate(frequency)

# Fonction de détection de l'ordre des fréquences
def get_frequency_order(frequency: Union[FrequencyType, UserFrequencyType]) -> float:
    """Get the numerical order of a frequency for comparison purposes.

    The order represents the granularity level where lower numbers indicate
    higher frequency (more granular) and higher numbers indicate lower
    frequency (less granular).

    Args:
        frequency: Frequency to get order for. Can be pandas code or literal name.

    Returns:
        Frequency order as float. Higher number means lower frequency/granularity.
        Returns 0 if frequency is not found in the order mapping.

    Examples:
        >>> get_frequency_order('daily')
        7.0
        >>> get_frequency_order('monthly')
        9.0
        >>> get_frequency_order('quarterly') > get_frequency_order('monthly')
        True
    """
    base_freq = normalize_frequency(frequency)
    return _normalizer._frequency_order.get(base_freq, 0)

# Conversion entre fréquences
def convert_frequency(
    value: Union[pd.Series, pd.DataFrame],
    to_unit: str,
    **kwargs
) -> float:
    """Convert data from one frequency to another.

    Args:
        value: Time series data to convert (Series or DataFrame)
        to_unit: Target frequency
        **kwargs: Additional conversion parameters (method, fill_method, etc.)

    Returns:
        Converted time series data

    Raises:
        ValueError: If conversion parameters are invalid

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> monthly = convert_frequency(series, 'daily', 'monthly', method='mean')
        >>> len(monthly)
        1
    """
    # Import local pour éviter l'import circulaire
    from .converter import FrequencyConverter

    converter = FrequencyConverter()
    return converter.convert(value, to_unit, **kwargs)
