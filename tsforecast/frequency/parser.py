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
from .detector import detect_frequency
from ..utils.frequency.utils import normalize_frequency
from ..utils.position import combine_frequency_position


# Fonction de détection et de parsing de la fréquence d'un index
def detect_and_parse_frequency(
    index: pd.DatetimeIndex
) -> Tuple[str, Optional[str], Optional[str]]:
    """Detect and parse DatetimeIndex frequency into components.

    Analyzes a DatetimeIndex to detect its frequency and parses it into
    three components: the base frequency code, the position indicator
    (S for start, E for end), and an optional suffix (e.g., DEC for
    quarterly data anchored to December).

    Args:
        index: DatetimeIndex to analyze. Must have at least 2 observations
            and a regular frequency pattern.

    Returns:
        Tuple of (frequency_code, position, suffix) where:
        - frequency_code: Pandas frequency indicator ('D', 'M', 'Q', etc.)
        - position: Position indicator ('S' for start, 'E' for end, or None)
        - suffix: Anchor suffix (e.g., 'DEC', 'JAN', or None)

    Raises:
        ValueError: If frequency cannot be detected or if the index has
            insufficient observations (<2) or irregular spacing.

    Examples:
        >>> import pandas as pd
        >>> # Daily frequency
        >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('D', None, None)
        >>>
        >>> # Monthly start
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('M', 'S', None)
        >>>
        >>> # Quarterly end with December anchor
        >>> dates = pd.date_range('2024-01-31', periods=4, freq='QE-DEC')
        >>> freq, pos, suffix = detect_and_parse_frequency(dates)
        >>> (freq, pos, suffix)
        ('Q', 'E', 'DEC')
    """
    # Création d'une série dummy avec des valeurs pour detect_frequency
    dummy = pd.Series(range(len(index)), index=index, dtype=float)
    freq = detect_frequency(dummy, literal=False)

    # Levée d'une erreur si aucune fréquence n'est détectée
    if freq is None:
        raise ValueError(
            "Could not detect index frequency. "
            "Index may be irregular or have insufficient observations."
        )

    # Séparation de la fréquence de sa position et de son suffixe
    # Expression régulière pour matcher: indicateur [S|E] optionnel [-suffixe] optionnel
    match = re.match(r"([A-Z]+?)([SE])?(-(.*?))?$", freq)

    # Extraction des éléments si un appariement est trouvé
    if match:
        freq_ind, position, _, suffix = match.groups()
        return freq_ind, position, suffix
    else:
        raise ValueError(
            f"Unable to parse frequency, position and suffix in '{freq}'. "
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
