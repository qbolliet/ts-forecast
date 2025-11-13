"""Publication delay calculation utilities.

This module provides functions to calculate applicable publication delays
by converting frequencies and aggregating delays across time series.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Literal
from datetime import datetime

# Modules du package
from ..utils.frequency import normalize_frequency, is_higher_frequency, to_pandas_freq
from ..utils.time import get_period_boundaries
from ..utils.duration import to_code as duration_to_code, convert_duration

# Fonction auxiliaire de validation des colonnes
def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If any required column is missing

    Examples:
        >>> _validate_columns(publication_delays_df)
        # Raises ValueError if columns are missing
    """
    # Colonnes requises pour le calcul des délais
    required_columns = [
        'observation_date',
        'download_date',
        'frequency',
        'period_start',
        'period_end',
        'reference_point',
        'release_delay',
        'unit'
    ]

    # Vérification de la présence de toutes les colonnes
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in publication_delays DataFrame: {missing_columns}. "
            f"Required columns are: {required_columns}"
        )

# Fonction de calcul du délai applicable
def calculate_applicable_delay(
    publication_delays: pd.DataFrame,
    target_reference_point: Literal['start', 'end'],
    target_frequency: Union[str, List[str]],
    indicators: Optional[List[str]] = None,
    aggregate_by_panel: bool = False,
    aggregation_method: Union[str, callable] = 'median',
    output_unit: Optional[Literal['us', 's', 'D', 'microsecond', 'second', 'day']] = None
) -> pd.DataFrame:
    """Calculate applicable publication delay for specified indicators.

    This function recalculates publication delays by converting to a target
    frequency and reference point, then aggregates delays according to the
    specified method.

    **Important behavior for higher frequency conversions:**

    When converting to a higher frequency (e.g., quarterly to monthly), the
    sub-period used as reference depends on the original reference_point:

    - If original reference_point='start': Uses the first sub-period that
      coincides with the start of the original period (e.g., January for Q1)
    - If original reference_point='end': Uses the last sub-period that
      coincides with the end of the original period (e.g., March for Q1)

    This ensures consistency: if a quarterly series references the end of
    the quarter, the monthly conversion will reference the end of the last
    month of that quarter.

    Args:
        publication_delays: DataFrame returned by compare_and_detect_delays()
            containing columns: observation_date, download_date, frequency,
            period_start, period_end, reference_point, release_delay, unit
        target_reference_point: Reference point for delay calculation ('start' or 'end')
        target_frequency: Target frequency for delay calculation. Can be:
            - A single frequency string to apply to all indicators
            - A list of frequency strings (one per indicator in same order)
        indicators: List of indicators to calculate delays for. If None, uses all
            indicators found in the data.
        aggregate_by_panel: If True, calculate separate delays for each (panel_entity, indicator)
            combination. If False, aggregate across all panel entities for each indicator.
        aggregation_method: Method to aggregate delays. Can be:
            - String: any method supported by pandas.agg ('mean', 'median', 'max', 'min', etc.)
            - Callable: custom aggregation function
        output_unit: Unit for output delays. If None, uses the unit from input data.
            Options: 'us'/'microsecond', 's'/'second', 'D'/'day'

    Returns:
        DataFrame with calculated applicable delays, indexed by indicator
        (or by panel_entity and indicator if aggregate_by_panel=True).
        Contains columns:
        - applicable_delay: The calculated delay value
        - unit: Unit of the delay
        - target_frequency: The target frequency used
        - target_reference_point: The target reference point used
        - n_observations: Number of observations used in aggregation
        - aggregation_method: The aggregation method used

    Raises:
        ValueError: If parameters are invalid or incompatible

    Examples:
        >>> # Calculate monthly delay from start for quarterly data
        >>> applicable = calculate_applicable_delay(
        ...     publication_delays=delays_df,
        ...     target_reference_point='start',
        ...     target_frequency='monthly',
        ...     indicators=['GDP', 'Unemployment'],
        ...     aggregation_method='median'
        ... )
        
        >>> # Calculate with panel-level aggregation
        >>> applicable = calculate_applicable_delay(
        ...     publication_delays=delays_df,
        ...     target_reference_point='end',
        ...     target_frequency='quarterly',
        ...     aggregate_by_panel=True,
        ...     aggregation_method='mean'
        ... )
    """
    # Validation des colonnes requises dans le DataFrame
    _validate_columns(publication_delays)

    # Validation des arguments
    if target_reference_point not in ['start', 'end']:
        raise ValueError("target_reference_point must be 'start' or 'end'")
    
    # Copie indépendante des données
    delays = publication_delays.copy()
    
    # Extraction de l'index (indicateur et éventuellement panel)
    # L'index doit contenir au moins un niveau (l'indicateur)
    if not isinstance(delays.index, pd.MultiIndex):
        raise ValueError("publication_delays must have a MultiIndex with at least indicator level")
    
    # Identification du niveau de l'indicateur (dernier niveau de l'index)
    indicator_level_name = delays.index.names[-1]
    
    # Filtrage sur les indicateurs si spécifié
    if indicators is not None:
        delays = delays[delays.index.get_level_values(indicator_level_name).isin(indicators)]
        if len(delays) == 0:
            raise ValueError(f"No data found for specified indicators: {indicators}")
    
    # Gestion de la fréquence cible
    if isinstance(target_frequency, list):
        # Vérification que le nombre de fréquences correspond au nombre d'indicateurs
        unique_indicators = delays.index.get_level_values(indicator_level_name).unique()
        if len(target_frequency) != len(unique_indicators):
            raise ValueError(
                f"Number of target frequencies ({len(target_frequency)}) must match "
                f"number of indicators ({len(unique_indicators)})"
            )
        # Création d'un mapping indicateur -> fréquence cible
        target_freq_map = dict(zip(unique_indicators, target_frequency))
    else:
        # Fréquence unique pour tous les indicateurs
        unique_indicators = delays.index.get_level_values(indicator_level_name).unique()
        target_freq_map = {ind: target_frequency for ind in unique_indicators}
    
    # Conversion des délais au point de référence et à la fréquence cibles
    delays = _convert_to_target_frequency_and_reference(
        delays=delays,
        target_freq_map=target_freq_map,
        target_reference_point=target_reference_point,
        indicator_level_name=indicator_level_name
    )
    
    # Conversion de l'unité si nécessaire
    if output_unit is not None:
        delays = _convert_delay_unit(delays, output_unit)
    
    # Agrégation des délais
    result = _aggregate_delays(
        delays=delays,
        indicator_level_name=indicator_level_name,
        aggregate_by_panel=aggregate_by_panel,
        aggregation_method=aggregation_method,
        target_reference_point=target_reference_point,
        target_freq_map=target_freq_map
    )
    
    return result

# Fonction auxiliaire de conversion à la fréquence cible et au point de référence cible des délais de publication
def _convert_to_target_frequency_and_reference(
    delays: pd.DataFrame,
    target_freq_map: dict,
    target_reference_point: str,
    indicator_level_name: str
) -> pd.DataFrame:
    """Convert delays to target frequency and reference point.

    Args:
        delays: Publication delays DataFrame
        target_freq_map: Mapping of indicator to target frequency
        target_reference_point: Target reference point ('start' or 'end')
        indicator_level_name: Name of the indicator level in the index

    Returns:
        DataFrame with converted delays
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
       at the target frequency. The logic differs based on frequency conversion:

       - **Higher frequency** (e.g., quarterly → monthly): Selects the sub-period
         that coincides with the original reference point:
         * If reference_point='start': First sub-period (e.g., January for Q1)
         * If reference_point='end': Last sub-period (e.g., March for Q1)

       - **Equal/lower frequency** (e.g., monthly → quarterly): Uses the period
         containing the observation_date at the target frequency.

    3. **Calculation of converted delay**: Computes the delay between the
       download date and the new reference date (start or end of target period),
       then converts back to the original time unit using ceiling rounding.

    Args:
        row: Row from delays DataFrame containing:
            - period_start, period_end: Original period boundaries
            - reference_point: Original reference ('start' or 'end')
            - release_delay: Numeric delay value
            - unit: Time unit ('days', 'seconds', 'microseconds')
            - target_frequency_normalized: Target frequency
            - current_frequency_normalized: Current frequency
        target_reference_point: Target reference point ('start' or 'end')

    Returns:
        Updated row with:
            - converted_delay: Delay value in original unit
            - target_period_start: Start of target period
            - target_period_end: End of target period
            - target_reference_point: Target reference point used

    Examples:
        >>> # Q1 2024 data (Jan 1 - Mar 31), published 45 days after quarter end
        >>> # Convert to monthly with 'start' reference
        >>> row = pd.Series({
        ...     'period_start': pd.Timestamp('2024-01-01'),
        ...     'period_end': pd.Timestamp('2024-03-31'),
        ...     'reference_point': 'end',
        ...     'release_delay': 45,
        ...     'unit': 'days',
        ...     'target_frequency_normalized': 'M',
        ...     'current_frequency_normalized': 'Q'
        ... })
        >>> result = _calculate_converted_delay(row, 'start')
        # Download date: Mar 31 + 45 days = May 15
        # Target period: March 2024 (Mar 1 - Mar 31) because reference_point='end'
        # Converted delay: May 15 - Mar 1 = 75 days
    """
    # Reconstruction de la date de téléchargement originale
    # download_date = reference_date + release_delay
    if row['reference_point'] == 'start':
        original_reference_date = row['period_start']
    else:
        original_reference_date = row['period_end']

    # Conversion du délai en secondes pour créer un timedelta en utilisant la fonction utilitaire
    delay_seconds = convert_duration(
        value=row['release_delay'],
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
        # Fréquence plus élevée : on choisit la sous-période selon le point de référence original
        # Si reference_point='start' : on prend la première sous-période (ex: janvier pour Q1)
        # Si reference_point='end' : on prend la dernière sous-période (ex: mars pour Q1)
        if row['reference_point'] == 'start':
            # Première sous-période : on utilise period_start
            reference_date_for_subperiod = row['period_start']
        else:
            # Dernière sous-période : on utilise period_end
            reference_date_for_subperiod = row['period_end']

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
        delays: DataFrame with delays
        target_unit: Target unit ('us'/'microsecond', 's'/'second', 'D'/'day')

    Returns:
        DataFrame with converted delays
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
        Aggregated DataFrame
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
        'target_frequency': 'first'  # La fréquence cible doit être la même pour chaque groupe
    })
    
    # Comptage du nombre d'observations
    n_obs = delays.groupby(level=group_levels).size()
    agg_result['n_observations'] = n_obs
    
    # Renommage de la colonne du délai
    agg_result = agg_result.rename(columns={'converted_delay': 'applicable_delay'})
    
    # Ajout des métadonnées
    agg_result['target_reference_point'] = target_reference_point
    agg_result['aggregation_method'] = str(aggregation_method) if isinstance(aggregation_method, str) else aggregation_method.__name__
    
    # Réorganisation des colonnes
    column_order = [
        'applicable_delay',
        'unit',
        'target_frequency',
        'target_reference_point',
        'n_observations',
        'aggregation_method'
    ]
    agg_result = agg_result[column_order]
    
    return agg_result