"""Frequency parsing utilities for time series index manipulation.

This module provides utilities for detecting and parsing DatetimeIndex frequencies
into their component parts (frequency code, position, suffix) and for building
complete frequency strings suitable for pandas operations.
"""
# Report de l'évaluation des annotations (évite l'import circulaire avec ..frequency)
from __future__ import annotations

# Importation des modules
import re
import pandas as pd
from typing import NamedTuple, Optional, TYPE_CHECKING

# Constantes transverses
from .._constants import POSITION_AWARE_FREQUENCIES

# Import réservé au typage statique : ..frequency et ..position importent ce
# module au niveau package (via frequency/normalizer.py et position/utils.py),
# un import réel ici créerait un cycle
if TYPE_CHECKING:
    from ..frequency.types import FrequencyType

# Composants d'une chaîne de fréquence pandas
class ParsedFrequency(NamedTuple):
    """Components of a pandas frequency string ``[N]FREQ[S|E][-SUFFIX]``.

    Attributes:
        freq: Base frequency code ('M', 'Q', 'W', 'h', ...).
        position: Position indicator ('S', 'E') or None.
        suffix: Anchor after the dash ('DEC', 'MON', ...) or None.
        multiplier: Leading integer multiplier (1 when absent), the ``n`` of
            ``to_offset('2MS').n``.

    Examples:
        >>> ParsedFrequency('M', 'S', None, 2)
        ParsedFrequency(freq='M', position='S', suffix=None, multiplier=2)
    """

    freq: FrequencyType
    position: Optional[str]
    suffix: Optional[str]
    multiplier: int = 1


# Fonction de parsing d'une chaîne de caractères contenant une fréquence
def parse_frequency(frequency_str : str) -> ParsedFrequency:
    """Parse a frequency string into its component parts.

    Extracts the multiplier, base frequency code, position indicator, and
    suffix from a pandas-style frequency string. This is useful for
    decomposing frequency specifications before normalization or conversion
    operations.

    Args:
        frequency_str: Pandas frequency string to parse (e.g., 'MS', '2MS',
            'QE-DEC', 'D'). Must follow the format
            ``[N][FREQ][S|E?]-[SUFFIX?]`` where ``N`` is an optional positive
            integer multiplier, ``FREQ`` is the base frequency code (letters
            like 'M', 'Q', 'W'), ``S``/``E`` is an optional position
            indicator ('S' for start, 'E' for end), and ``SUFFIX`` is an
            optional suffix after ``-`` (e.g. month names for quarters).

    Returns:
        ParsedFrequency, a named tuple ``(freq, position, suffix, multiplier)``:
            - freq (FrequencyType): Base frequency code ('M', 'Q', 'W', etc.)
            - position (str): Position indicator ('S', 'E', or None)
            - suffix (str): Suffix component (e.g., 'DEC') or None
            - multiplier (int): Leading multiplier, 1 if absent

    Raises:
        ValueError: If frequency_str is None (no frequency detected)
        ValueError: If frequency_str doesn't match expected format, or if its
            multiplier is zero

    Examples:
        >>> # Monthly start frequency
        >>> parse_frequency('MS')
        ParsedFrequency(freq='M', position='S', suffix=None, multiplier=1)
        >>>
        >>> # Quarterly end with December anchor
        >>> parse_frequency('QE-DEC')
        ParsedFrequency(freq='Q', position='E', suffix='DEC', multiplier=1)
        >>>
        >>> # Multiplied frequency (every two months, at month start)
        >>> parse_frequency('2MS')
        ParsedFrequency(freq='M', position='S', suffix=None, multiplier=2)
        >>>
        >>> # Daily frequency (no position or suffix)
        >>> parse_frequency('D')
        ParsedFrequency(freq='D', position=None, suffix=None, multiplier=1)
    """
    # Levée d'une erreur si aucune fréquence n'est détectée
    if frequency_str is None:
        raise ValueError(
            "Could not detect index frequency. "
            "Index may be irregular or have insufficient observations."
        )

    # Séparation du multiplicateur, de la fréquence, de sa position et de son suffixe
    # Expression régulière pour matcher: [multiplicateur] optionnel, indicateur, [S|E] optionnel, [-suffixe] optionnel
    # L'indicateur autorise aussi les minuscules (ex: 'h', 'min', 'ms', 'us', 'ns')
    # pour couvrir les fréquences infra-journalières ; S/E restent en majuscules,
    # seules les fréquences position-aware (M/Q/Y/SM) les utilisant réellement
    match = re.match(r"(\d*)([A-Za-z]+?)([SE])?(-(.*?))?$", frequency_str)

    # Extraction des éléments si un appariement est trouvé
    if match:
        multiplier, freq_ind, position, _, suffix = match.groups()
        # Multiplicateur absent -> 1 ; un multiplicateur nul n'a pas de sens
        n = int(multiplier) if multiplier else 1
        if n < 1:
            raise ValueError(
                f"Frequency multiplier must be a positive integer, got '{multiplier}' in '{frequency_str}'"
            )
        return ParsedFrequency(freq_ind, position, suffix, n)
    else:
        raise ValueError(
            f"Unable to parse frequency, position and suffix in '{frequency_str}'. "
            "Should follow the format: [N][FREQ][S|E?]-[SUFFIX?]"
        )

# Fonction de construction d'une chaine de caractère de fréquence à partir de la fréquence, de la position et du suffixe
def build_frequency_string(
    frequency: str,
    position: Optional[str] = None,
    suffix: Optional[str] = None,
    multiplier: int = 1,
    default_position: Optional[str] = None
) -> str:
    """Build complete pandas frequency string from components.

    Constructs a complete frequency string by combining a base frequency
    code with optional position and suffix indicators. Without
    ``default_position`` the string is a faithful assembly of its components:
    a bare ``'M'`` stays ``'M'``, which is a base code (fit for a duration or
    a comparison) but not a pandas alias. When the string goes to pandas (``pd.date_range``,
    ``resample``, ``to_offset``), pass ``default_position='E'``: a bare month,
    quarter, year or semi-month then becomes ``'ME'``, ``'QE'``, ``'YE'``,
    ``'SME'``, the period end that the bare aliases designated.

    Args:
        frequency: Base frequency code ('D', 'M', 'Q', 'W', 'h', 'T', 'S', etc.)
        position: Optional position indicator:
            - 'S': Start of period (e.g., 'MS' for month start, 'QS' for quarter start)
            - 'E': End of period (e.g., 'ME' for month end, 'QE' for quarter end)
            - None: No position specification
            Only applied to frequencies with a start/end variant in pandas
            ('M', 'Q', 'Y', 'SM'); silently ignored otherwise ('D', 'W',
            'B', 'h', ...), so the result is always a valid pandas alias.
        suffix: Optional suffix for anchoring:
            - For quarterly: 'JAN', 'FEB', 'MAR', ..., 'DEC'
            - For other frequencies: may be ignored by pandas
            - None: No suffix
        multiplier: Positive integer multiplier put in front of the string
            (default 1, omitted from the result).
        default_position: Position ('S', 'E' or None) used when ``position`` is
            None, for the frequencies that have start / end variants (``'M'``,
            ``'Q'``, ``'Y'``, ``'SM'``); ignored for the others and when
            ``position`` is given. None (default) leaves the base code bare.

    Returns:
        Complete frequency string (e.g., 'D', 'MS', 'QE-DEC', '2MS')

    Raises:
        ValueError: If position or default_position is invalid (not in
            ['S', 'E', None]) or if multiplier is not a positive integer

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
        >>>
        >>> # Weekly anchored on Monday (no S/E position for weekly anchors)
        >>> build_frequency_string('W', suffix='MON')
        'W-MON'
        >>>
        >>> # Every two months, at month start (inverse of parse_frequency)
        >>> build_frequency_string('M', position='S', multiplier=2)
        '2MS'
        >>>
        >>> # Bare base code, then pandas alias (period end by default)
        >>> build_frequency_string('M')
        'M'
        >>> build_frequency_string('M', default_position='E')
        'ME'
        >>> build_frequency_string('M', position='S', default_position='E')
        'MS'
    """
    from ..frequency.utils import normalize_frequency

    # Fréquence de base
    base_freq = normalize_frequency(frequency=frequency)

    # Validité du paramètre position
    if position is not None and position not in ['S', 'E']:
        raise ValueError(f"position must be 'S' (start), 'E' (end), or None, got '{position}'")

    # Validité de la position par défaut
    if default_position not in (None, 'S', 'E'):
        raise ValueError(f"default_position must be 'S' (start), 'E' (end), or None, got '{default_position}'")

    # Validité du multiplicateur (bool exclu : True == 1 passerait sinon)
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
        raise ValueError(f"multiplier must be a positive integer, got {multiplier!r}")

    # Ajout de la position uniquement si elle est fournie et si la fréquence la
    # supporte (ex: 'D' n'a pas de déclinaison S/E) ; sinon la position est
    # silencieusement ignorée. Le suffixe, lui, reste ajouté indépendamment de
    # la position : certaines fréquences portent une ancre significative sans
    # notion de position (ex: 'W-MON', jour de la semaine, pas de S/E en pandas).
    # La position par défaut ne sert qu'en l'absence de position explicite
    effective_position = position if position is not None else default_position
    freq_with_position = (
        f"{base_freq}{effective_position}"
        if effective_position is not None and base_freq in POSITION_AWARE_FREQUENCIES
        else base_freq
    )

    # Ajout du suffixe si présent, puis du multiplicateur (omis lorsqu'il vaut 1)
    if suffix is not None:
        freq_with_position = f"{freq_with_position}-{suffix}"

    return f"{multiplier}{freq_with_position}" if multiplier != 1 else freq_with_position
