"""Publication delay calculation utilities.

This module provides functions to calculate applicable publication delays
by converting frequencies and aggregating delays across time series.
"""
# Importation des modules
# Modules de base
from fractions import Fraction

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal

# Modules du package
from ..utils.frequency import normalize_frequency, is_higher_frequency, to_pandas_freq
from ..utils.time.utils import get_period_boundaries
from ..utils.duration import to_code as duration_to_code, get_duration_nanoseconds

# Fonction de calcul du délai applicable
def calculate_applicable_delay(
    publication_delays: pd.DataFrame,
    reference_point: Literal['start', 'end'],
    frequency: Union[str, Dict[Union[str, tuple], str]],
    unit: Optional[Literal['us', 's', 'D', 'microsecond', 'second', 'day']] = None,
    indicators: Optional[List[str]] = None,
    aggregate_by_panel: bool = False,
    aggregation_method: Union[str, callable] = 'median'
) -> pd.DataFrame:
    """Calculate applicable publication delay for specified indicators.

    This function recalculates publication delays by converting to a target
    frequency and reference point, then aggregates delays according to the
    specified method.

    **Important behavior for higher frequency conversions:**

    When converting to a higher frequency (e.g., quarterly to monthly), the
    sub-period used as reference is determined by identifying which sub-period
    contains the observation_date.

    For example, when converting quarterly data (Q1 = Jan-Feb-Mar) to monthly:
    - If observation_date is in January: The target period is January
    - If observation_date is in February: The target period is February
    - If observation_date is in March: The target period is March

    This ensures that the delay is calculated relative to the specific
    sub-period when the observation actually occurred.

    **Unknown delays.** A row whose delay is ``NaN`` or whose frequency is
    undetermined (``None``), as returned by ``compare_and_detect_delays`` for a
    couple whose frequency cannot be detected, is kept with a ``NaN`` converted
    delay: it is not counted in ``n_observations``, and a group made only of such
    rows gets a ``NaN`` delay and ``n_observations=0``.

    Args:
        publication_delays: DataFrame returned by compare_and_detect_delays()
            containing columns: observation_date, download_date, frequency,
            period_start, period_end, reference_point, delay, unit. The last
            level of its index is the indicator (the level names are free, or
            absent); any other level identifies a panel entity. The ``unit``
            values must be duration names or codes (``'day'``/``'D'``,
            ``'second'``/``'s'``, ``'microsecond'``/``'us'``, ...): plural forms
            such as ``'days'`` are rejected. The rows aggregated together must share
            the same unit.
        reference_point: Reference point for delay calculation ('start' or 'end'): the
            delay is counted from the start of the target period, or from its end,
            which is exclusive (the period of March ends on April 1st).
        frequency: Target frequency for delay calculation. Either a single
            frequency applied to all indicators ('monthly', 'M', 'quarterly',
            'Q', ...), or a dict whose keys are indicators (``{'GDP': 'monthly'}``)
            and/or full index keys of a panel, ``(entity, ..., indicator)`` tuples
            (``{('FR', 'GDP'): 'monthly'}``). The tuple key of a row takes
            precedence over its indicator key; every row of the (filtered) data must
            be covered. Without ``aggregate_by_panel``, the entities of an indicator
            must share the same target frequency.
        unit: Unit for output delays. If None, keeps the unit of the input data
            (its label is then unchanged, e.g. ``'day'``). Otherwise any duration
            supported by ``convert_duration`` ('us'/'microsecond', 's'/'second',
            'D'/'day', 'h'/'hour', 'W'/'week', ...): delays are converted with
            ceiling rounding and the ``unit`` column then holds the duration
            *code* (``'D'``, ``'h'``, ...).
        indicators: List of indicators to calculate delays for. If None, uses all
            indicators found in the data.
        aggregate_by_panel: If True, calculate separate delays for each (panel_entity, indicator)
            combination. If False, aggregate across all panel entities for each indicator.
        aggregation_method: Method to aggregate delays. Can be:
            - String: any method supported by pandas.agg ('mean', 'median', 'max', 'min', etc.)
            - Callable: custom aggregation function (reported by its ``__name__``)

    Returns:
        DataFrame with calculated applicable delays, indexed by indicator
        (or by panel_entity and indicator if aggregate_by_panel=True).
        Contains columns:
        - delay: The calculated delay value
        - unit: Unit of the delay
        - frequency: The target frequency used, as given (literal name or code)
        - reference_point: The target reference point used
        - n_observations: Number of observations (known delays) used in aggregation
        - aggregation_method: The aggregation method used

    Raises:
        ValueError: If a required column is missing, if ``publication_delays`` has no
            row, if reference_point is not 'start' or 'end', if none of the requested
            indicators is found, if a frequency or a unit is not supported, if
            ``frequency`` is a dict that does not cover every row (the uncovered keys
            are named), if the entities of an indicator have different target
            frequencies without ``aggregate_by_panel``, if aggregated rows have
            different units, or if ``aggregation_method`` is an unknown method name
        TypeError: If frequency is neither a string nor a dict, or if
            aggregation_method is neither a string nor a callable

    Examples:
        >>> # Quarterly GDP of two countries and monthly CPI, all observed in March 2024
        >>> index = pd.MultiIndex.from_tuples(
        ...     [('FR', 'GDP'), ('DE', 'GDP'), ('FR', 'CPI')], names=['country', 'indicator'])
        >>> delays_df = pd.DataFrame({
        ...     'observation_date': pd.to_datetime(['2024-03-15'] * 3),
        ...     'download_date': pd.to_datetime(['2024-05-15', '2024-05-25', '2024-04-10']),
        ...     'frequency': ['Q', 'Q', 'M'],
        ...     'period_start': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-03-01']),
        ...     'period_end': pd.to_datetime(['2024-03-31', '2024-03-31', '2024-03-31']),
        ...     'reference_point': ['end', 'end', 'end'],
        ...     'delay': [45, 55, 10],
        ...     'unit': ['day', 'day', 'day'],
        ... }, index=index)

        >>> # Monthly delay from the start of the month, median over the countries:
        >>> # GDP observed in March, downloaded May 15 (FR) and May 25 (DE), gives
        >>> # 75 and 85 days after March 1st, hence a median of 80
        >>> applicable = calculate_applicable_delay(
        ...     publication_delays=delays_df,
        ...     reference_point='start',
        ...     frequency='monthly',
        ...     aggregation_method='median'
        ... )
        >>> applicable['delay'].to_dict()
        {'CPI': 40.0, 'GDP': 80.0}

        >>> # Panel-level aggregation, delays counted from the end of the quarter
        >>> # (April 1st, exclusive)
        >>> applicable = calculate_applicable_delay(
        ...     publication_delays=delays_df,
        ...     reference_point='end',
        ...     frequency='quarterly',
        ...     aggregate_by_panel=True,
        ...     aggregation_method='mean'
        ... )
        >>> applicable['delay'].to_dict()
        {('DE', 'GDP'): 54.0, ('FR', 'CPI'): 9.0, ('FR', 'GDP'): 44.0}

        >>> # One target frequency per indicator, delays expressed in hours
        >>> applicable = calculate_applicable_delay(
        ...     publication_delays=delays_df,
        ...     reference_point='start',
        ...     frequency={'GDP': 'monthly', 'CPI': 'quarterly'},
        ...     unit='hour'
        ... )
        >>> applicable['delay'].to_dict(), applicable['unit'].unique().tolist()
        ({'CPI': 2400.0, 'GDP': 1920.0}, ['h'])

        >>> # One target frequency per (entity, indicator) couple: German GDP in
        >>> # quarterly (Jan 1st to May 25th: 145 days), French GDP and CPI in monthly
        >>> applicable = calculate_applicable_delay(
        ...     publication_delays=delays_df,
        ...     reference_point='start',
        ...     frequency={('DE', 'GDP'): 'quarterly', 'GDP': 'monthly', 'CPI': 'monthly'},
        ...     aggregate_by_panel=True
        ... )
        >>> applicable['delay'].to_dict()
        {('DE', 'GDP'): 145.0, ('FR', 'CPI'): 40.0, ('FR', 'GDP'): 75.0}
    """
    # Validation des colonnes requises dans le DataFrame
    _validate_columns(publication_delays)

    # Validation des arguments
    if reference_point not in ['start', 'end']:
        raise ValueError("reference_point must be 'start' or 'end'")
    if len(publication_delays) == 0:
        raise ValueError("Cannot calculate applicable delays on empty data: publication_delays has no row")
    _validate_aggregation_method(aggregation_method)

    # Copie indépendante des données
    delays = publication_delays.copy()

    # Position du niveau de l'indicateur (dernier niveau de l'index) : les niveaux
    # sont repérés par leur position, leurs noms étant libres, voire absents
    indicator_level = delays.index.nlevels - 1

    # Filtrage sur les indicateurs si spécifié
    if indicators is not None:
        delays = delays[delays.index.get_level_values(indicator_level).isin(indicators)]
        if len(delays) == 0:
            raise ValueError(f"No data found for specified indicators: {indicators}")

    # Fréquence cible de chaque ligne
    if isinstance(frequency, str):
        # Fréquence unique pour tous les indicateurs
        target_frequencies = [frequency] * len(delays)
    elif isinstance(frequency, dict):
        target_frequencies = _resolve_target_frequencies(delays.index, frequency)
    else:
        raise TypeError(f"'frequency' should be a string or a dict, got a {type(frequency).__name__}")

    # Conversion des délais au point de référence et à la fréquence cibles
    delays = _convert_to_target_frequency_and_reference(
        delays=delays,
        target_frequencies=target_frequencies,
        target_reference_point=reference_point
    )

    # Fréquence cible unique par indicateur quand les entités sont agrégées
    if not aggregate_by_panel:
        _validate_single_frequency_per_indicator(delays, indicator_level)

    # Conversion de l'unité si nécessaire
    if unit is not None:
        delays = _convert_delay_unit(delays, unit)

    # Agrégation des délais
    result = _aggregate_delays(
        delays=delays,
        aggregate_by_panel=aggregate_by_panel,
        aggregation_method=aggregation_method,
        target_reference_point=reference_point
    )

    return result

# Fonction auxiliaire de validation des colonnes
def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If any required column is missing

    Examples:
        >>> _validate_columns(pd.DataFrame({'delay': [45]}))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Missing required columns in publication_delays DataFrame: ['observation_date', ...
    """
    # Colonnes requises pour le calcul des délais
    required_columns = [
        'observation_date',
        'download_date',
        'frequency',
        'period_start',
        'period_end',
        'reference_point',
        'delay',
        'unit'
    ]

    # Vérification de la présence de toutes les colonnes
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in publication_delays DataFrame: {missing_columns}. "
            f"Required columns are: {required_columns}"
        )


# Fonction auxiliaire de validation de la méthode d'agrégation
def _validate_aggregation_method(aggregation_method: Union[str, callable]) -> None:
    """Validate the aggregation method before any computation.

    A method name is tried on a minimal grouped series, so that an unknown name is
    reported as an invalid argument rather than as an obscure pandas error.

    Args:
        aggregation_method: Method name supported by ``pandas.agg``, or callable

    Raises:
        ValueError: If a name is not a pandas aggregation method
        TypeError: If the method is neither a string nor a callable

    Examples:
        >>> _validate_aggregation_method('median')
        >>> _validate_aggregation_method('nope')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Unsupported aggregation_method 'nope': ...
    """
    if isinstance(aggregation_method, str):
        # Essai de la méthode sur un groupe minimal
        try:
            pd.Series([0.0, 1.0]).groupby([0, 0]).agg(aggregation_method)
        except Exception as error:
            raise ValueError(
                f"Unsupported aggregation_method {aggregation_method!r}: it is not a pandas "
                f"aggregation method ({type(error).__name__}: {error})"
            ) from error
    elif not callable(aggregation_method):
        raise TypeError(
            f"'aggregation_method' should be a string or a callable, got a {type(aggregation_method).__name__}"
        )


# Fonction auxiliaire de résolution de la fréquence cible de chaque ligne
def _resolve_target_frequencies(index: pd.Index, frequency: Dict[Union[str, tuple], str]) -> List[str]:
    """Resolve the target frequency of each row from a dictionary.

    The key of a row is its full index entry (a tuple for a ``MultiIndex``: entity
    levels, then indicator) and takes precedence; the indicator alone is the fallback.

    Args:
        index: Index of the delays (last level = indicator)
        frequency: ``{indicator: frequency}`` and/or ``{(entity, ..., indicator): frequency}``

    Returns:
        One target frequency per row, in the order of ``index``

    Raises:
        ValueError: If some rows are covered by neither of their keys (the keys are named)

    Examples:
        >>> index = pd.MultiIndex.from_tuples([('FR', 'GDP'), ('DE', 'GDP')])
        >>> _resolve_target_frequencies(index, {('DE', 'GDP'): 'Q', 'GDP': 'M'})
        ['M', 'Q']
    """
    # Liste des fréquences valides
    targets = []
    # Liste des entités sans fréquences associées
    uncovered = []
    # Parcours de l'index
    for key in index:
        # Extraction de l'indicateur en dernière position de l'index
        indicator = key[-1] if isinstance(key, tuple) else key
        # Recherche de la fréquence associée à l'indicateur
        if key in frequency:
            targets.append(frequency[key])
        elif indicator in frequency:
            targets.append(frequency[indicator])
        # Cas où la fréquence n'est pas trouvée
        else:
            targets.append(None)
            uncovered.append(key)

    # Cas avec des entités non couvertes
    if uncovered:
        raise ValueError(
            f"'frequency' does not give a target frequency for: {list(dict.fromkeys(uncovered))}. "
            f"Give one per indicator, or per (entity, ..., indicator) index key"
        )

    return targets


# Fonction auxiliaire de validation de l'unicité de la fréquence cible par indicateur
def _validate_single_frequency_per_indicator(delays: pd.DataFrame, indicator_level: int) -> None:
    """Validate that each indicator has a single target frequency.

    Delays counted against different target frequencies cannot be aggregated across
    entities.

    Args:
        delays: Converted delays, holding ``target_frequency_normalized``
        indicator_level: Position of the indicator level in the index

    Raises:
        ValueError: If the entities of an indicator have different target frequencies
    """
    # Extraction des fréquences
    frequencies = delays['target_frequency_normalized'].groupby(level=indicator_level).nunique()
    # Vérification de l'unicité
    ambiguous = frequencies[frequencies > 1].index.tolist()
    if ambiguous:
        raise ValueError(
            f"The target frequency differs between the entities of the indicators {ambiguous}: "
            f"use aggregate_by_panel=True to aggregate per (entity, indicator)"
        )


# Fonction auxiliaire de conversion à la fréquence cible et au point de référence cible des délais de publication
def _convert_to_target_frequency_and_reference(
    delays: pd.DataFrame,
    target_frequencies: List[str],
    target_reference_point: str
) -> pd.DataFrame:
    """Convert delays to target frequency and reference point.

    The columns ``target_frequency``, ``target_frequency_normalized`` and
    ``current_frequency_normalized`` are added to ``delays`` in place; the
    conversion of each row is then delegated to ``_calculate_converted_delay``.

    Args:
        delays: Publication delays DataFrame
        target_frequencies: Target frequency of each row, in the order of ``delays``
        target_reference_point: Target reference point ('start' or 'end')

    Returns:
        DataFrame with the input columns, the three frequency columns above and
        ``converted_delay``, ``target_period_start``, ``target_period_end`` and
        ``target_reference_point``

    Raises:
        ValueError: If a frequency is not supported
    """
    # Ajout d'une colonne pour la fréquence cible
    delays['target_frequency'] = target_frequencies

    # Normalisation des fréquences (une fréquence source indéterminée reste telle quelle)
    delays['target_frequency_normalized'] = delays['target_frequency'].apply(normalize_frequency)
    delays['current_frequency_normalized'] = delays['frequency'].apply(
        lambda current: normalize_frequency(current) if pd.notna(current) else np.nan
    )

    # Calcul de la nouvelle date de référence selon la fréquence et le point de référence cibles
    delays = delays.apply(
        lambda row: _calculate_converted_delay(row, target_reference_point),
        axis=1
    )

    return delays


# Fonction auxiliaire de conversion d'une valeur en nanosecondes entières
def _to_nanoseconds(value: float, nanoseconds_per_unit: int) -> int:
    """Convert a delay value to whole nanoseconds, exactly.

    Args:
        value: Delay, in a unit of ``nanoseconds_per_unit`` nanoseconds (integer or float)
        nanoseconds_per_unit: Length of the unit, in nanoseconds

    Returns:
        The delay in nanoseconds (a fraction of a nanosecond is rounded)

    Examples:
        >>> _to_nanoseconds(129_600_000_123_457, 1_000)
        129600000123457000
        >>> _to_nanoseconds(1.5, 10**9)
        1500000000
    """
    # Cas d'un entier
    if isinstance(value, (int, np.integer)):
        return int(value) * nanoseconds_per_unit
    # Fraction d'un flottant : valeur binaire exacte, sans erreur d'arrondi du produit
    return round(Fraction(float(value)) * nanoseconds_per_unit)


# Fonction auxiliaire de conversion des délais de publication à la bonne fréquence et à la bonne référence
def _calculate_converted_delay(row: pd.Series, target_reference_point: str) -> pd.Series:
    """Calculate delay with converted frequency and reference point for a single row.

    This function performs a three-step conversion process:

    1. **Reconstruction of download date**: Converts the relative delay
       (e.g., "45 days after period end") into an absolute download date
       by adding the delay to the original reference date.

    2. **Determination of target period**: Calculates the period boundaries
       at the target frequency, the end being **exclusive** (the period of
       March ends on April 1st). The logic differs based on frequency conversion:

       - **Higher frequency** (e.g., quarterly → monthly): Identifies the sub-period
         that contains the observation_date. For example, if converting Q1 (Jan-Mar)
         to monthly:
         * If observation_date is in January: Uses January as target period
         * If observation_date is in February: Uses February as target period
         * If observation_date is in March: Uses March as target period

       - **Equal/lower frequency** (e.g., monthly → quarterly): Uses the period
         containing the observation_date at the target frequency.

    3. **Calculation of converted delay**: Computes the delay between the
       download date and the new reference date (start or end of target period),
       then converts back to the original time unit using ceiling rounding.
       The calculation is made with integer nanoseconds, hence exact.

    A row whose delay or source frequency is unknown (``NaN`` / ``None``) is returned
    with a ``NaN`` converted delay and ``NaT`` target period bounds.

    Args:
        row: Row from delays DataFrame containing:
            - observation_date: Date the observation refers to
            - period_start, period_end: Original period boundaries
            - reference_point: Original reference ('start' or 'end')
            - delay: Numeric delay value
            - unit: Duration name or code of the delay ('day'/'D', 'second'/'s',
              'microsecond'/'us', ...; plural forms such as 'days' are rejected)
            - target_frequency_normalized: Target frequency, as a base code ('M')
            - current_frequency_normalized: Current frequency, as a base code ('Q')
        target_reference_point: Target reference point ('start' or 'end')

    Returns:
        Updated row with:
            - converted_delay: Delay value in original unit, rounded up
            - target_period_start: Start of target period
            - target_period_end: End of target period (exclusive)
            - target_reference_point: Target reference point used

    Raises:
        ValueError: If the unit of a row with a known delay is not a supported duration

    Examples:
        >>> # Q1 2024 data (Jan 1 - Mar 31), published 45 days after quarter end
        >>> # Observation occurred in March, convert to monthly with 'start' reference
        >>> row = pd.Series({
        ...     'observation_date': pd.Timestamp('2024-03-15'),
        ...     'period_start': pd.Timestamp('2024-01-01'),
        ...     'period_end': pd.Timestamp('2024-03-31'),
        ...     'reference_point': 'end',
        ...     'delay': 45,
        ...     'unit': 'day',
        ...     'target_frequency_normalized': 'M',
        ...     'current_frequency_normalized': 'Q'
        ... })
        >>> result = _calculate_converted_delay(row, 'start')
        >>> # Download date: Mar 31 + 45 days = May 15
        >>> # Target period: March 2024 because observation_date is in March
        >>> result['target_period_start'], result['target_period_end']
        (Timestamp('2024-03-01 00:00:00'), Timestamp('2024-04-01 00:00:00'))
        >>> # Converted delay: May 15 - Mar 1 = 75 days
        >>> result['converted_delay']
        75
    """
    # Extraction du point de référence ("start"/"end")
    row['target_reference_point'] = target_reference_point

    # Délai ou fréquence source inconnus : la ligne est conservée avec un délai inconnu
    if pd.isna(row['delay']) or pd.isna(row['current_frequency_normalized']):
        row['converted_delay'] = np.nan
        row['target_period_start'] = pd.NaT
        row['target_period_end'] = pd.NaT
        return row

    # Reconstruction de la date de téléchargement originale
    # download_date = reference_date + delay
    if row['reference_point'] == 'start':
        original_reference_date = row['period_start']
    else:
        original_reference_date = row['period_end']

    # Durée de l'unité du délai en nanosecondes entières (calcul exact, sans secondes flottantes)
    nanoseconds_per_unit = get_duration_nanoseconds(row['unit'])
    delay_timedelta = pd.Timedelta(_to_nanoseconds(row['delay'], nanoseconds_per_unit), unit='ns')

    # Calcul de la date de téléchargement
    download_date = original_reference_date + delay_timedelta

    # Calcul de la nouvelle période de référence selon la fréquence cible
    # Si la fréquence cible est plus élevée que la fréquence actuelle,
    # on prend la première sous-période
    target_freq_normalized = row['target_frequency_normalized']
    current_freq_normalized = row['current_frequency_normalized']

    # Détermination de la période de référence pour la fréquence cible
    if is_higher_frequency(target_freq_normalized, current_freq_normalized):
        # Fréquence plus élevée : on identifie la sous-période qui coïncide avec observation_date
        # Génération de toutes les sous-périodes dans la période originale
        target_freq_pandas = to_pandas_freq(target_freq_normalized)
        subperiods = pd.date_range(
            start=row['period_start'],
            end=row['period_end'],
            freq=target_freq_pandas
        )

        # Identification de la sous-période qui contient observation_date
        # On cherche la sous-période dont les bornes englobent observation_date
        observation_date = row['observation_date']
        reference_date_for_subperiod = None

        for subperiod_date in subperiods:
            subperiod_start, subperiod_end = get_period_boundaries(
                date=subperiod_date,
                frequency=target_freq_normalized
            )
            # Vérification si observation_date se trouve dans cette sous-période
            if subperiod_start <= observation_date < subperiod_end:
                reference_date_for_subperiod = subperiod_date
                break

        # Si aucune sous-période ne correspond (cas limite), on utilise la dernière
        if reference_date_for_subperiod is None:
            reference_date_for_subperiod = subperiods[-1]

        target_period_start, target_period_end = get_period_boundaries(
            date=reference_date_for_subperiod,
            frequency=target_freq_normalized
        )
    else:
        # Fréquence égale ou inférieure : on garde la même période
        target_period_start, target_period_end = get_period_boundaries(
            date=row['observation_date'],
            frequency=target_freq_normalized
        )

    # Calcul de la nouvelle date de référence selon le point de référence cible
    if target_reference_point == 'start':
        new_reference_date = target_period_start
    else:
        new_reference_date = target_period_end

    # Calcul du nouveau délai, en nanosecondes entières
    new_delay_nanoseconds = (download_date - new_reference_date) // pd.Timedelta(1, unit='ns')

    # Conversion dans l'unité d'origine, arrondie au supérieur (division entière par excès)
    row['converted_delay'] = -(-new_delay_nanoseconds // nanoseconds_per_unit)

    # Ajout des informations sur la nouvelle période de référence
    row['target_period_start'] = target_period_start
    row['target_period_end'] = target_period_end

    return row

# Fonction auxiliaire de conversion de l'unité du délai
def _convert_delay_unit(delays: pd.DataFrame, target_unit: str) -> pd.DataFrame:
    """Convert delay values to target unit.

    The conversion is made with integer nanoseconds, hence exact before the ceiling
    rounding.

    Args:
        delays: DataFrame with delays, holding ``converted_delay`` and ``unit``
        target_unit: Target duration, name or code: 'us'/'microsecond',
            's'/'second', 'D'/'day', 'h'/'hour', 'W'/'week', ... (any duration
            supported by ``convert_duration``)

    Returns:
        DataFrame with ``converted_delay`` converted (ceiling rounding; rows
        already in the target unit, and unknown delays, are left as they are) and
        ``unit`` set to the duration *code* of the target unit (``'D'``, not ``'day'``)

    Raises:
        ValueError: If a unit is not a supported duration
    """
    # Normalisation de l'unité cible avec la fonction utilitaire to_code
    # qui gère déjà tous les formats possibles
    target_unit_code = duration_to_code(target_unit)
    target_nanoseconds = get_duration_nanoseconds(target_unit_code)

    # Fonction de conversion en nanosecondes entières
    def convert_value(row):
        value = row['converted_delay']

        # Délai inconnu : conservé tel quel
        if pd.isna(value):
            return value

        # Extraction du code de l'unité courante
        current_unit_code = duration_to_code(row['unit'])

        # Pas de conversion nécessaire si les unités sont identiques
        if current_unit_code == target_unit_code:
            return value

        # Conversion exacte, arrondie au supérieur
        nanoseconds = _to_nanoseconds(value, get_duration_nanoseconds(current_unit_code))
        return -(-nanoseconds // target_nanoseconds)

    # Conversion
    delays['converted_delay'] = delays.apply(convert_value, axis=1)
    delays['unit'] = target_unit_code

    return delays

# Fonction auxiliaire d'aggrégation des délais
def _aggregate_delays(
    delays: pd.DataFrame,
    aggregate_by_panel: bool,
    aggregation_method: Union[str, callable],
    target_reference_point: str
) -> pd.DataFrame:
    """Aggregate delays by indicator or by (panel, indicator).

    Args:
        delays: DataFrame with converted delays
        aggregate_by_panel: Whether to aggregate by panel
        aggregation_method: Aggregation method
        target_reference_point: Target reference point

    Returns:
        DataFrame indexed by the grouping levels (indicator, or panel entities and
        indicator) with the columns ``delay``, ``unit`` (first of the group),
        ``frequency`` (target frequency as given, first of the group),
        ``reference_point``, ``n_observations`` (known delays) and ``aggregation_method``

    Raises:
        ValueError: If the rows of a group have different units
    """
    # Détermination des niveaux de groupement (par position)
    last_level = delays.index.nlevels - 1
    if aggregate_by_panel:
        # Groupement par tous les niveaux de l'index (panel + indicateur)
        group_levels = list(range(delays.index.nlevels))
    else:
        # Groupement uniquement par indicateur
        group_levels = [last_level]

    # Les délais agrégés ensemble doivent être exprimés dans la même unité
    unit_codes = delays['unit'].map(duration_to_code)
    mixed = unit_codes.groupby(level=group_levels).nunique()
    mixed_groups = mixed[mixed > 1].index.tolist()
    if mixed_groups:
        raise ValueError(
            f"The delays of {mixed_groups} are expressed in different units and cannot be aggregated "
            f"as they are: give a common 'unit'"
        )

    # Agrégation des délais
    agg_result = delays.groupby(level=group_levels).agg({
        'converted_delay': aggregation_method,
        'unit': 'first',  # L'unité doit être la même pour tous
        'target_frequency': 'first'  # La fréquence cible doit être la même pour chaque groupe (sera renommée en 'frequency')
    })

    # Comptage du nombre d'observations dont le délai est connu
    n_obs = delays['converted_delay'].notna().groupby(level=group_levels).sum().astype('int64')
    agg_result['n_observations'] = n_obs

    # Un groupe sans aucun délai connu a un délai inconnu (une somme vide vaudrait 0)
    agg_result.loc[agg_result['n_observations'] == 0, 'converted_delay'] = np.nan

    # Renommage de la colonne du délai
    agg_result = agg_result.rename(columns={'converted_delay': 'delay', 'target_frequency': 'frequency'})

    # Ajout des métadonnées
    agg_result['reference_point'] = target_reference_point
    agg_result['aggregation_method'] = (
        aggregation_method if isinstance(aggregation_method, str)
        else getattr(aggregation_method, '__name__', type(aggregation_method).__name__)
    )

    # Réorganisation des colonnes
    column_order = [
        'delay',
        'unit',
        'frequency',
        'reference_point',
        'n_observations',
        'aggregation_method'
    ]
    agg_result = agg_result[column_order]

    return agg_result
