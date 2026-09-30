"""Publication delay calculation utilities.

This module provides functions to calculate applicable publication delays
by converting frequencies and aggregating delays across time series.
"""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Dict, List, Optional, Union, Literal

# Modules du package
from ..utils.frequency import normalize_frequency, is_higher_frequency, to_pandas_freq
from ..utils.time.utils import get_period_boundaries
from ..utils.duration import to_code as duration_to_code, convert_duration

# Fonction de calcul du délai applicable
def calculate_applicable_delay(
    publication_delays: pd.DataFrame,
    reference_point: Literal['start', 'end'],
    frequency: Union[str, Dict[str, str]],
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

    Args:
        publication_delays: DataFrame returned by compare_and_detect_delays()
            containing columns: observation_date, download_date, frequency,
            period_start, period_end, reference_point, delay, unit. The last
            level of its index is the indicator; any other level identifies a
            panel entity. The ``unit`` values must be duration names or codes
            (``'day'``/``'D'``, ``'second'``/``'s'``, ``'microsecond'``/``'us'``,
            ...): plural forms such as ``'days'`` are rejected. All rows of an
            indicator are expected to share the same unit (the aggregated
            ``unit`` is the first one of the group).
        reference_point: Reference point for delay calculation ('start' or 'end'): the
            delay is counted from the start of the target period, or from its end,
            which is exclusive (the period of March ends on April 1st).
        frequency: Target frequency for delay calculation. Either a single
            frequency applied to all indicators ('monthly', 'M', 'quarterly',
            'Q', ...), or a dict ``{indicator: frequency}`` that must cover every
            indicator of the (filtered) data.
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
        - n_observations: Number of observations used in aggregation
        - aggregation_method: The aggregation method used

    Raises:
        ValueError: If a required column is missing, if reference_point is not
            'start' or 'end', if none of the requested indicators is found, or if a
            frequency or a unit is not supported (this includes an indicator that
            ``frequency`` does not cover when it is a dict)
        TypeError: If frequency is neither a string nor a dict

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
    """
    # Validation des colonnes requises dans le DataFrame
    _validate_columns(publication_delays)

    # Validation des arguments
    if reference_point not in ['start', 'end']:
        raise ValueError("reference_point must be 'start' or 'end'")
    
    # Copie indépendante des données
    delays = publication_delays.copy()

    # Identification du niveau de l'indicateur (dernier niveau de l'index)
    indicator_level_name = delays.index.names[-1]
    
    # Filtrage sur les indicateurs si spécifié
    if indicators is not None:
        delays = delays[delays.index.get_level_values(indicator_level_name).isin(indicators)]
        if len(delays) == 0:
            raise ValueError(f"No data found for specified indicators: {indicators}")
    
    # Création du mapping indicateur -> fréquence cible
    if isinstance(frequency, str):
        # Fréquence unique pour tous les indicateurs
        unique_indicators = delays.index.get_level_values(indicator_level_name).unique()
        target_freq_map = {ind: frequency for ind in unique_indicators}
    elif isinstance(frequency, dict):
        target_freq_map = frequency.copy()
    else:
        raise TypeError(f"'frequency' should be a string or a dict, got a {type(frequency).__name__}")
    
    # Conversion des délais au point de référence et à la fréquence cibles
    delays = _convert_to_target_frequency_and_reference(
        delays=delays,
        target_freq_map=target_freq_map,
        target_reference_point=reference_point,
        indicator_level_name=indicator_level_name
    )
    
    # Conversion de l'unité si nécessaire
    if unit is not None:
        delays = _convert_delay_unit(delays, unit)
    
    # Agrégation des délais
    result = _aggregate_delays(
        delays=delays,
        indicator_level_name=indicator_level_name,
        aggregate_by_panel=aggregate_by_panel,
        aggregation_method=aggregation_method,
        target_reference_point=reference_point,
        target_freq_map=target_freq_map
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


# Fonction auxiliaire de conversion à la fréquence cible et au point de référence cible des délais de publication
def _convert_to_target_frequency_and_reference(
    delays: pd.DataFrame,
    target_freq_map: dict,
    target_reference_point: str,
    indicator_level_name: str
) -> pd.DataFrame:
    """Convert delays to target frequency and reference point.

    The columns ``target_frequency``, ``target_frequency_normalized`` and
    ``current_frequency_normalized`` are added to ``delays`` in place; the
    conversion of each row is then delegated to ``_calculate_converted_delay``.

    Args:
        delays: Publication delays DataFrame
        target_freq_map: Mapping of indicator to target frequency
        target_reference_point: Target reference point ('start' or 'end')
        indicator_level_name: Name of the indicator level in the index

    Returns:
        DataFrame with the input columns, the three frequency columns above and
        ``converted_delay``, ``target_period_start``, ``target_period_end`` and
        ``target_reference_point``

    Raises:
        ValueError: If an indicator has no target frequency in ``target_freq_map``
            (its frequency is then NaN), or if a frequency is not supported
    """
    # Ajout d'une colonne pour la fréquence cible
    delays['target_frequency'] = delays.index.get_level_values(indicator_level_name).map(target_freq_map)
    
    # Normalisation des fréquences
    delays['target_frequency_normalized'] = delays['target_frequency'].apply(normalize_frequency)
    delays['current_frequency_normalized'] = delays['frequency'].apply(normalize_frequency)
    
    # Calcul de la nouvelle date de référence selon la fréquence et le point de référence cibles
    delays = delays.apply(
        lambda row: _calculate_converted_delay(row, target_reference_point),
        axis=1
    )
    
    return delays

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
    # Reconstruction de la date de téléchargement originale
    # download_date = reference_date + delay
    if row['reference_point'] == 'start':
        original_reference_date = row['period_start']
    else:
        original_reference_date = row['period_end']

    # Conversion du délai en secondes pour créer un timedelta en utilisant la fonction utilitaire
    delay_seconds = convert_duration(
        value=row['delay'],
        from_duration=row['unit'],
        to_duration='s',
        rounding=None
    )
    delay_timedelta = pd.Timedelta(seconds=delay_seconds)

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
    
    # Calcul du nouveau délai
    new_delay_timedelta = download_date - new_reference_date

    # Conversion du nouveau délai dans l'unité d'origine en utilisant la fonction utilitaire
    new_delay_seconds = new_delay_timedelta.total_seconds()
    row['converted_delay'] = convert_duration(
        value=new_delay_seconds,
        from_duration='s',
        to_duration=row['unit'],
        rounding='ceil'
    )
    
    # Ajout des informations sur la nouvelle période de référence
    row['target_period_start'] = target_period_start
    row['target_period_end'] = target_period_end
    row['target_reference_point'] = target_reference_point
    
    return row

# Fonction auxiliaire de conversion de l'unité du délai
def _convert_delay_unit(delays: pd.DataFrame, target_unit: str) -> pd.DataFrame:
    """Convert delay values to target unit using convert_duration utility.

    Args:
        delays: DataFrame with delays, holding ``converted_delay`` and ``unit``
        target_unit: Target duration, name or code: 'us'/'microsecond',
            's'/'second', 'D'/'day', 'h'/'hour', 'W'/'week', ... (any duration
            supported by ``convert_duration``)

    Returns:
        DataFrame with ``converted_delay`` converted (ceiling rounding; rows
        already in the target unit are left as they are) and ``unit`` set to the duration
        *code* of the target unit (``'D'``, not ``'day'``)

    Raises:
        ValueError: If a unit is not a supported duration
    """
    # Normalisation de l'unité cible avec la fonction utilitaire to_code
    # qui gère déjà tous les formats possibles
    target_unit_code = duration_to_code(target_unit)

    # Fonction de conversion utilisant la fonction utilitaire convert_duration
    def convert_value(row):
        value = row['converted_delay']
        current_unit = row['unit']

        # Extraction du code de l'unité courante
        current_unit_code = duration_to_code(current_unit)

        # Pas de conversion nécessaire si les unités sont identiques
        if current_unit_code == target_unit_code:
            return value

        # Conversion avec la fonction utilitaire convert_duration
        return convert_duration(
            value=value,
            from_duration=current_unit_code,
            to_duration=target_unit_code,
            rounding='ceil'
        )

    delays['converted_delay'] = delays.apply(convert_value, axis=1)
    delays['unit'] = target_unit_code

    return delays

# Fonction auxiliaire d'aggrégation des délais
def _aggregate_delays(
    delays: pd.DataFrame,
    indicator_level_name: str,
    aggregate_by_panel: bool,
    aggregation_method: Union[str, callable],
    target_reference_point: str,
    target_freq_map: dict
) -> pd.DataFrame:
    """Aggregate delays by indicator or by (panel, indicator).

    Args:
        delays: DataFrame with converted delays
        indicator_level_name: Name of indicator level in index
        aggregate_by_panel: Whether to aggregate by panel
        aggregation_method: Aggregation method
        target_reference_point: Target reference point
        target_freq_map: Mapping of indicator to target frequency

    Returns:
        DataFrame indexed by the grouping levels (indicator, or panel entities and
        indicator) with the columns ``delay``, ``unit`` (first of the group),
        ``frequency`` (target frequency as given, first of the group),
        ``reference_point``, ``n_observations`` and ``aggregation_method``
    """
    # Détermination des niveaux de groupement
    if aggregate_by_panel:
        # Groupement par tous les niveaux de l'index (panel + indicateur)
        group_levels = list(delays.index.names)
    else:
        # Groupement uniquement par indicateur
        group_levels = [indicator_level_name]
    
    # Agrégation des délais
    agg_result = delays.groupby(level=group_levels).agg({
        'converted_delay': aggregation_method,
        'unit': 'first',  # L'unité doit être la même pour tous
        'target_frequency': 'first'  # La fréquence cible doit être la même pour chaque groupe (sera renommée en 'frequency')
    })
    
    # Comptage du nombre d'observations
    n_obs = delays.groupby(level=group_levels).size()
    agg_result['n_observations'] = n_obs
    
    # Renommage de la colonne du délai
    agg_result = agg_result.rename(columns={'converted_delay': 'delay', 'target_frequency': 'frequency'})

    # Ajout des métadonnées
    agg_result['reference_point'] = target_reference_point
    agg_result['aggregation_method'] = str(aggregation_method) if isinstance(aggregation_method, str) else aggregation_method.__name__

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