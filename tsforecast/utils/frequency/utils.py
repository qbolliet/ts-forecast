# Importation des modules
# Modules de base
from typing import Union, Literal, Tuple, Optional, Dict, List, TYPE_CHECKING
import pandas as pd

# Importation des modules du package
from .normalizer import FrequencyNormalizer
from .types import FrequencyType, UserFrequencyType
from ..parse.utils import parse_frequency, build_frequency_string
from ...panel.utils import normalize_entity_key

# Import réservé au typage statique : .detector importe des noms définis dans
# ce module (normalize_frequency, get_frequency_order), un import global ici
# créerait un cycle
if TYPE_CHECKING:
    from .detector import FrequencyDetector

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

# Fonction de détection de la fréquence d'une série
def detect_frequency(data: Union[pd.Series, pd.DataFrame],
                    time_col: Optional[str] = None,
                    panel_cols: Optional[List[str]] = None,
                    return_format: Literal['base', 'with_position', 'full', 'components'] = 'base',
                    check_consistency: bool = False,
                    consistency_mode: str = 'modal',
                    strict: bool = True) -> Optional[Union[FrequencyType, UserFrequencyType, Dict]]:
    """Detect the frequency of time series data using FrequencyDetector.

    This function handles both Series and DataFrame inputs. For Series with MultiIndex,
    it returns a dictionary mapping panel IDs to frequencies. For DataFrames, it can
    detect frequencies for all columns and optionally return a consistent frequency.

    Args:
        data: Time series data (Series or DataFrame) with datetime index or MultiIndex
        time_col: Name of the time column (if None, uses index). Only for DataFrame.
        panel_cols: List of columns identifying panel dimensions. Only for DataFrame.
        return_format: Output format for the detected frequency:
            - 'base': Base frequency code (e.g. 'M', 'Q', 'D')
            - 'with_position': Frequency with position (e.g. 'MS', 'QE')
            - 'full': Full pandas frequency string (e.g. 'QE-DEC')
            - 'components': Tuple of (base, position, suffix)
        check_consistency: If True, check frequency consistency across panel groups or columns
        consistency_mode: Mode for determining consistent frequency ('modal' or 'highest')
            - 'modal': Returns the most common frequency (default)
            - 'highest': Returns the highest frequency (most granular)
        strict: If True, all frequencies must be identical (used with check_consistency)

    Returns:
        For Series without MultiIndex: Detected frequency as string, or None if detection fails
        For Series with MultiIndex: Dictionary mapping panel_id to frequencies, or single frequency if check_consistency=True
        For DataFrame without check_consistency: Dictionary mapping columns to frequencies
        For DataFrame with check_consistency: Single consistent frequency or None

    Examples:
        >>> import pandas as pd
        >>> # Simple series detection
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> detect_frequency(series)
        'D'

        >>> # Series with MultiIndex (panel data)
        >>> idx = pd.MultiIndex.from_arrays([
        ...     ['A', 'A', 'A', 'B', 'B', 'B'],
        ...     pd.date_range('2023-01-01', periods=3, freq='D').tolist() * 2
        ... ], names=['panel_id', 'date'])
        >>> series = pd.Series([1, 2, 3, 4, 5, 6], index=idx)
        >>> detect_frequency(series)
        {('A',): 'D', ('B',): 'D'}

        >>> # Series with MultiIndex and consistency check
        >>> detect_frequency(series, check_consistency=True)
        'D'

        >>> # DataFrame detection with consistency check (modal)
        >>> df = pd.DataFrame({'a': series, 'b': series})
        >>> detect_frequency(df, check_consistency=True, consistency_mode='modal')
        'D'
    """
    # Cas 1: Entrée Series
    if isinstance(data, pd.Series):
        if time_col is not None or panel_cols:
            raise ValueError("time_col and panel_cols parameters are only valid for DataFrame inputs")

        # Importation du locale du détector (pour éviter les imports circulaires)
        from .detector import FrequencyDetector
        # Instanciation de la classe
        _detector = FrequencyDetector()

        # Détection de la fréquence (peut retourner un dict pour MultiIndex)
        freq_result = _detector.detect_frequency(data, return_format)

        # Si check_consistency demandé et résultat est un dict (MultiIndex détecté)
        if check_consistency and isinstance(freq_result, dict):
            # Application de la logique de validation de consistance
            if consistency_mode == 'modal':
                # Mode modal: fréquence la plus commune
                _, frequency = _detector.validate_frequency_consistency(frequency_map=freq_result, strict=strict)
                return frequency
            elif consistency_mode == 'highest':
                # Mode highest: fréquence la plus haute (plus granulaire)
                return _get_highest_frequency(freq_result)
            else:
                raise ValueError(f"Invalid consistency_mode: {consistency_mode}. Must be 'modal' or 'highest'")

        # Sinon, retour direct du résultat
        return freq_result

    # Cas 2: Entrée DataFrame
    elif isinstance(data, pd.DataFrame):
        return detect_dataset_frequency(
            df=data,
            time_col=time_col,
            panel_cols=panel_cols,
            return_format=return_format,
            check_consistency=check_consistency,
            consistency_mode=consistency_mode,
            strict=strict
        )

    else:
        raise ValueError("Input must be a pandas Series or DataFrame")

# Fonction de détection de la fréquence d'un jeu de données
def detect_dataset_frequency(df: pd.DataFrame,
                           time_col: Optional[str] = None,
                           panel_cols: Optional[List[str]] = None,
                           return_format: Literal['base', 'with_position', 'full', 'components'] = 'base',
                           check_consistency: bool = False,
                           consistency_mode: str = 'modal',
                           strict: bool = True) -> Union[Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]], FrequencyType, UserFrequencyType]:
    """Detect frequencies for all series in a dataset using FrequencyDetector.

    Args:
        df: DataFrame containing time series data
        time_col: Name of the time column (if None, uses index)
        panel_cols: List of columns identifying panel dimensions
        return_format: Output format for the detected frequency:
            - 'base': Base frequency code (e.g. 'M', 'Q', 'D')
            - 'with_position': Frequency with position (e.g. 'MS', 'QE')
            - 'full': Full pandas frequency string (e.g. 'QE-DEC')
            - 'components': Tuple of (base, position, suffix)
        check_consistency: If True, check the frequency consistency across columns
        consistency_mode: Mode for determining consistent frequency ('modal' or 'highest')
            - 'modal': Returns the most common frequency (default)
            - 'highest': Returns the highest frequency (most granular)
        strict: If True, all frequencies must be identical

    Returns:
        Dictionary mapping column names (or (panel_id, column) tuples) to frequencies or consistent frequency
    """
    # Importation du locale du détector (pour éviter les imports circulaires)
    from .detector import FrequencyDetector
    # Instanciation de la classe
    _detector = FrequencyDetector()
    
    # Détection des fréquences
    frequency_map = _detector.detect_dataset_frequency(df, time_col, panel_cols, return_format)
    # Vérification de la consistence des fréquences si demandé
    if check_consistency :
        if consistency_mode == 'modal':
            # Mode modal: fréquence la plus commune
            _, frequency = _detector.validate_frequency_consistency(frequency_map=frequency_map, strict=strict)
        elif consistency_mode == 'highest':
            # Mode highest: fréquence la plus haute (plus granulaire)
            frequency = _get_highest_frequency(frequency_map)
        else:
            raise ValueError(f"Invalid consistency_mode: {consistency_mode}. Must be 'modal' or 'highest'")
        return frequency
    else :
        return frequency_map

# Fonction de détection de la fréquence d'un index
def detect_index_frequency(
    index: Union[pd.DatetimeIndex, pd.MultiIndex],
    return_format: Literal['base', 'with_position', 'full', 'components'] = 'base'
) -> Union[FrequencyType, Dict[Tuple, FrequencyType]]:
    """Detect and parse DatetimeIndex or MultiIndex frequency.

    For a DatetimeIndex, analyzes the frequency and returns it in the
    requested format. With 'components' format, returns a tuple of
    (base, position, suffix).

    For a MultiIndex with dates as the last level, detects frequency
    for each unique entity (combination of non-date levels) and returns
    a dictionary mapping entity keys to their frequencies.

    Args:
        index: DatetimeIndex or MultiIndex to analyze. For MultiIndex,
            the last level must be a DatetimeIndex. Must have at least 2
            observations and a regular frequency pattern.
        return_format: Output format for the detected frequency:
            - 'base': Base frequency code (e.g. 'M', 'Q', 'D')
            - 'with_position': Frequency with position (e.g. 'MS', 'QE')
            - 'full': Full pandas frequency string (e.g. 'QE-DEC')
            - 'components': Tuple of (base, position, suffix)

    Returns:
        For DatetimeIndex: Frequency in the requested format
        For MultiIndex: Dict mapping entity keys to frequencies. Entity keys
            are ALWAYS tuples, even for a single entity level (``('FR',)``,
            never ``'FR'``), so that they match the keys produced by
            ``get_unique_panel_entities`` and ``detect_dataset_frequency``.

    Raises:
        ValueError: If frequency cannot be detected or if the index has
            insufficient observations (<2) or irregular spacing.

    Examples:
        >>> import pandas as pd
        >>> # DatetimeIndex - Daily frequency
        >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
        >>> detect_index_frequency(dates)
        'D'
        >>>
        >>> # With components format
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        >>> detect_index_frequency(dates, return_format='components')
        ('M', 'S', None)
        >>>
        >>> # MultiIndex panel data
        >>> idx = pd.MultiIndex.from_product([
        ...     ['entity_1', 'entity_2'],
        ...     pd.date_range('2024-01-01', periods=5, freq='MS')
        ... ], names=['entity', 'date'])
        >>> detect_index_frequency(idx)
        {('entity_1',): 'M', ('entity_2',): 'M'}
    """
    # Vérification du type d'index : MultiIndex ou DatetimeIndex
    if isinstance(index, pd.MultiIndex):
        # Traitement du MultiIndex avec date au dernier niveau
        result = {}
        date_level_idx = index.nlevels - 1

        # Extraction des entités uniques (n-1 premiers niveaux)
        entity_index = index.droplevel(date_level_idx)

        # Parcours des entités
        for entity_key in entity_index.unique():
            # Création du masque pour cette entité
            mask = entity_index == entity_key
            # Extraction du DatetimeIndex pour cette entité
            dates = index.get_level_values(date_level_idx)[mask]
            # Appel récursif pour déterminer la fréquence
            freq = detect_index_frequency(dates, return_format=return_format)
            # Clé d'entité TOUJOURS normalisée en tuple : "entity_index.unique()"
            # rend des scalaires quand le panel n'a qu'un seul niveau d'entité
            result[normalize_entity_key(entity_key)] = freq

        return result
    else:
        # Traitement du DatetimeIndex
        # Inférence de la fréquence de l'index
        freq = index.inferred_freq
        # Normalisation au format demandé
        return normalize_frequency(frequency=freq, return_format=return_format)


# Fonction de construction d'un offset cible ancré comme un index source
def target_offset_for_index(
    index: pd.DatetimeIndex,
    target_frequency: str,
) -> str:
    """Build a pandas offset for target_frequency anchored like the index.

    Detects whether the source index is anchored at period start or end and
    combines that position with the target frequency, so that aggregated
    labels land on dates that exist in the source index (e.g. a
    month-start-anchored index converted to a quarterly frequency yields
    'QS', not 'QE'). Shared by every consumer that resamples data to a
    lower frequency while needing the result to stay reindexable against
    the original, position-anchored data.

    Args:
        index: Source DatetimeIndex whose position (start/end) is detected.
        target_frequency: Target frequency, with or without an explicit
            position (e.g. 'Q' or 'QE'). When a source position IS detected,
            any existing target position is stripped and overridden by it,
            so a caller passing an end-anchored frequency against a
            start-anchored index still gets rebased correctly instead of
            keeping the mismatched position as-is.

    Returns:
        Pandas offset alias anchored like ``index`` (e.g. 'QS') when a
        position can be detected on ``index``. Falls back to
        ``target_frequency`` unchanged when the source position cannot be
        detected at all (e.g. a position-less source frequency like daily,
        or an index too short to detect a frequency): there is then nothing
        to anchor on, so the caller's own choice — positioned or not — is
        preserved rather than defaulting to an arbitrary position.

    Examples:
        >>> ms_index = pd.date_range('2024-01-01', periods=12, freq='MS')
        >>> target_offset_for_index(ms_index, 'Q')
        'QS'
        >>> target_offset_for_index(ms_index, 'QE')
        'QS'
        >>> daily_index = pd.date_range('2024-01-01', periods=59, freq='D')
        >>> target_offset_for_index(daily_index, 'ME')
        'ME'
    """
    # Détection de la position (début/fin) de l'index source
    try:
        _, position, _ = detect_index_frequency(index, return_format='components')
    except Exception:
        position = None

    # Position source indétectable (fréquence sans position, ex. journalière,
    # ou index trop court) : rien à ancrer, la fréquence cible fournie par
    # l'appelant est conservée telle quelle, positionnée ou non
    if position is None:
        return target_frequency

    # Une position cible déjà présente (ex. 'QE') serait sinon transmise telle
    # quelle par build_frequency_string, qui ignore silencieusement les
    # fréquences déjà suffixées : on repart de la base pour l'écraser par la
    # position détectée sur la source
    try:
        target_base = normalize_frequency(target_frequency, return_format='base')
    except Exception:
        return target_frequency

    try:
        return build_frequency_string(target_base, position)
    except Exception:
        return target_frequency


# Fonction auxiliaire d'extraction de la fréquence la plus haute d'un dictionnaire de fréquences
def _get_highest_frequency(frequency_map: Dict[Union[str, tuple], Union[FrequencyType, UserFrequencyType]]) -> Optional[Union[FrequencyType, UserFrequencyType]]:
    """Get the highest (most granular) frequency from a frequency map.

    Args:
        frequency_map: Dictionary mapping columns to frequencies

    Returns:
        Highest frequency or None if map is empty

    Examples:
        >>> freq_map = {'col1': 'D', 'col2': 'M', 'col3': 'D'}
        >>> _get_highest_frequency(freq_map)
        'D'
    """
    # Vérification que le dictionnaire n'est pas vide
    if not frequency_map:
        return None

    # Extraction des fréquences uniques
    unique_frequencies = set(frequency_map.values())

    # Si une seule fréquence, la retourner
    if len(unique_frequencies) == 1:
        return list(unique_frequencies)[0]

    # Normalisation des fréquences avec ancrages (ex: 'W-SUN' -> 'W', 'Q-DEC' -> 'Q')
    # Calcul de l'ordre de chaque fréquence et sélection de la plus haute (ordre le plus bas)
    freq_orders = {}
    for freq in unique_frequencies:
        try:
            freq_orders[freq] = get_frequency_order(freq)
        except ValueError:
            # Si la normalisation échoue, essayer avec la fréquence complète
            try:
                freq_orders[freq] = get_frequency_order(freq)
            except ValueError:
                # Si toujours échec, ignorer cette fréquence
                continue

    if not freq_orders:
        return None

    # Retour de la fréquence avec l'ordre le plus bas (plus granulaire)
    highest_freq = min(freq_orders.keys(), key=lambda x: freq_orders[x])

    return highest_freq
