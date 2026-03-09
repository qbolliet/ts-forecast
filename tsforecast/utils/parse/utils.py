"""Frequency parsing utilities for time series index manipulation.

This module provides utilities for detecting and parsing DatetimeIndex frequencies
into their component parts (frequency code, position, suffix) and for building
complete frequency strings suitable for pandas operations.
"""
# Importation des modules
import re
import pandas as pd
from typing import Tuple, Optional

# Importation des modules du package
from ..frequency.utils import FrequencyType
from ..position import combine_frequency_position

# Fonction de parsing d'une chaîne de caractères contenant une fréquence
def parse_frequency(frequency_str : str) -> Tuple[FrequencyType, str, str]:
    """Parse a frequency string into its component parts.

    Extracts the base frequency code, position indicator, and suffix from
    a pandas-style frequency string. This is useful for decomposing frequency
    specifications before normalization or conversion operations.

    Args:
        frequency_str: Pandas frequency string to parse (e.g., 'MS', 'QE-DEC', 'D')
            Must follow the format: [FREQ][S|E?]-[SUFFIX?]
            - FREQ: Base frequency code (uppercase letters like 'M', 'Q', 'W')
            - S|E: Optional position indicator ('S' for start, 'E' for end)
            - SUFFIX: Optional suffix after '-' (e.g., month names for quarters)

    Returns:
        Tuple containing:
            - freq_ind (FrequencyType): Base frequency code ('M', 'Q', 'W', etc.)
            - position (str): Position indicator ('S', 'E', or None)
            - suffix (str): Suffix component (e.g., 'DEC') or None

    Raises:
        ValueError: If frequency_str is None (no frequency detected)
        ValueError: If frequency_str doesn't match expected format

    Examples:
        >>> # Monthly start frequency
        >>> parse_frequency('MS')
        ('M', 'S', None)
        >>>
        >>> # Quarterly end with December anchor
        >>> parse_frequency('QE-DEC')
        ('Q', 'E', 'DEC')
        >>>
        >>> # Daily frequency (no position or suffix)
        >>> parse_frequency('D')
        ('D', None, None)
        >>>
        >>> # Weekly frequency
        >>> parse_frequency('W')
        ('W', None, None)
    """
    # Levée d'une erreur si aucune fréquence n'est détectée
    if frequency_str is None:
        raise ValueError(
            "Could not detect index frequency. "
            "Index may be irregular or have insufficient observations."
        )

    # Séparation de la fréquence de sa position et de son suffixe
    # Expression régulière pour matcher: indicateur [S|E] optionnel [-suffixe] optionnel
    match = re.match(r"([A-Z]+?)([SE])?(-(.*?))?$", frequency_str)

    # Extraction des éléments si un appariement est trouvé
    if match:
        freq_ind, position, _, suffix = match.groups()
        return freq_ind, position, suffix
    else:
        raise ValueError(
            f"Unable to parse frequency, position and suffix in '{frequency_str}'. "
            "Should follow the format: [FREQ][S|E?]-[SUFFIX?]"
        )

# Fonction de construction d'une chaine de caractère de fréquence à partir de la fréquence, de la position et du suffixe
def build_frequency_string(
    frequency: str,
    position: Optional[str] = None,
    suffix: Optional[str] = None
) -> str:
    """Build complete pandas frequency string from components.

    Constructs a complete frequency string suitable for pandas operations
    by combining a base frequency code with optional position and suffix
    indicators. This is useful for extending DatetimeIndex with proper
    frequency specification.

    Args:
        frequency: Base frequency code ('D', 'M', 'Q', 'W', 'h', 'T', 'S', etc.)
        position: Optional position indicator:
            - 'S': Start of period (e.g., 'MS' for month start, 'QS' for quarter start)
            - 'E': End of period (e.g., 'ME' for month end, 'QE' for quarter end)
            - None: No position specification
        suffix: Optional suffix for anchoring:
            - For quarterly: 'JAN', 'FEB', 'MAR', ..., 'DEC'
            - For other frequencies: may be ignored by pandas
            - None: No suffix

    Returns:
        Complete frequency string (e.g., 'D', 'MS', 'QE-DEC')

    Raises:
        ValueError: If position is invalid (not in ['S', 'E', None])

    Examples:
        >>> # Daily frequency (no position/suffix)
        >>> build_frequency_string('D')
        'D'
        >>>
        >>> # Monthly start
        >>> build_frequency_string('M', position='S')
        'MS'
        >>>
        >>> # Quarterly end with December anchor
        >>> build_frequency_string('Q', position='E', suffix='DEC')
        'QE-DEC'
        >>>
        >>> # Weekly frequency
        >>> build_frequency_string('W')
        'W'
    """
    from ..frequency.utils import normalize_frequency

    # Fréquence de base
    base_freq = normalize_frequency(frequency=frequency)

    # Validité du paramètre position
    if position is not None and position not in ['S', 'E']:
        raise ValueError(f"position must be 'S' (start), 'E' (end), or None, got '{position}'")

    # Si pas de position, retourne la fréquence de base
    if position is None:
        return base_freq

    # Combinaison avec la position
    freq_with_position = combine_frequency_position(base_freq, position)

    # Ajout du suffixe si présent
    if suffix is not None:
        return f"{freq_with_position}-{suffix}"

    return freq_with_position
