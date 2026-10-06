"""Data manager for comparing and storing publication delays information.

This module provides tools for comparing new datasets with existing data
and managing publication delay information using pandas DataFrames.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any, Literal, cast, overload
from datetime import datetime, timedelta
import logging
import warnings
# Module de détection de la fréquence des séries et de conversion en littéral
from ..utils.frequency import detect_frequency, to_literal
# Module de validation des données temporelles
from ..utils.validation import validate_temporal_data
# Module de manipulation temporelle
from ..utils.time import resolve_date, get_period_boundaries
# Module de conversion des durées (le calcul des délais se fait en nanosecondes entières :
# un produit flottant par 1e6 fausserait le dernier chiffre au-delà de ~800 jours)
from ..utils.duration import to_code as duration_to_code, to_literal as duration_to_literal, get_duration_nanoseconds
# Module de manipulation des structures de panel
from ..panel.utils import is_panel_data, get_entity_levels, get_unique_panel_entities
# Module de rapport de détection
from .report import DelayDetectionReport, delay_statistics

# Journalisation : aucun handler n'est configuré ici, c'est à l'application d'en fournir un
logger = logging.getLogger(__name__)

# Codes des unités de délai acceptées (sous-ensemble des durées supportées par utils.duration)
_DELAY_UNIT_CODES = frozenset({'D', 's', 'us'})

# Signatures typées de la fonction de comparaison : le type de retour dépend de return_report
@overload
def compare_and_detect_delays(new_data: pd.DataFrame, existing_data: Optional[pd.DataFrame] = None, download_date: Union[str, datetime, None] = None, detection_mode: str = 'new_only', reference_point: str = 'start', delay_unit: Literal['us', 's', 'D', 'microsecond', 'second', 'day'] = 'day', time_col: Optional[str] = None, panel_cols: Optional[List[str]] = None, *, return_report: Literal[False] = False) -> pd.DataFrame: ...
@overload
def compare_and_detect_delays(new_data: pd.DataFrame, existing_data: Optional[pd.DataFrame] = None, download_date: Union[str, datetime, None] = None, detection_mode: str = 'new_only', reference_point: str = 'start', delay_unit: Literal['us', 's', 'D', 'microsecond', 'second', 'day'] = 'day', time_col: Optional[str] = None, panel_cols: Optional[List[str]] = None, *, return_report: Literal[True]) -> Tuple[pd.DataFrame, DelayDetectionReport]: ...

# Fonction de comparaison et d'inférence des délais de publication
def compare_and_detect_delays(new_data: pd.DataFrame, existing_data: Optional[pd.DataFrame] = None, download_date: Union[str, datetime, None] = None, detection_mode: str = 'new_only', reference_point: str = 'start', delay_unit: Literal['us', 's', 'D', 'microsecond', 'second', 'day'] = 'day', time_col: Optional[str] = None, panel_cols: Optional[List[str]] = None, *, return_report: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, DelayDetectionReport]]:
    """Compare new data with existing data and detect publication delays.

    This function identifies new or changed observations by comparing new_data with
    existing_data (if provided), then calculates publication delays for these observations.

    When existing_data is None, the function identifies the most recent non-null observation
    for each variable (and each panel entity if panel data is provided).

    The delay of an observation is always counted from the same point of its period
    (``reference_point``) to ``download_date``, whatever the detection mode: a revised
    value detected in ``'all_changes'`` mode gets the same kind of delay as a first
    publication. The delay between the first publication and the revision is the
    difference between the two results.

    Values that disappear (non-null in ``existing_data``, ``NaN`` in ``new_data``) are
    not reported by either mode: there is no value left to attach a delay to.

    A negative delay is legitimate and kept as is (for instance an observation that is
    itself a forecast, published before the period it covers).

    Dates without time zone (the index of the data, the period bounds) are read as UTC
    when compared with a time-zone-aware ``download_date``; conversely a naive
    ``download_date`` is read as UTC when the data are time-zone-aware. Time-zone-aware
    values of different zones are compared as instants.

    Args:
        new_data: New data DataFrame to analyze, with a time index (``DatetimeIndex``,
            ``PeriodIndex`` or dates convertible to datetime), a ``MultiIndex`` whose last
            level is the time for panel data, or time / entity columns given by
            ``time_col`` / ``panel_cols``
        existing_data: Existing data DataFrame for comparison. If None, identifies the most
            recent non-null observation for each variable/entity. Only the columns common
            to both datasets are compared
        download_date: Date when the data was downloaded: a string, a ``datetime`` (naive
            or time-zone-aware) or ``'today'``. If None, uses current datetime
        detection_mode: Detection mode - 'new_only' (only null→non-null transitions) or
            'all_changes' (all value changes, revisions included). Only used when
            existing_data is provided
        reference_point: Reference point for delay calculation - 'start' or 'end' of the period.
            The end of a period is its exclusive bound (the first instant of the next period)
        delay_unit: Unit for delay calculation - 'day'/'D', 'second'/'s', or 'microsecond'/'us'
        time_col: Name of the time column (optional)
        panel_cols: List of panel column names for panel data (optional)
        return_report: If True, also return a :class:`~tsforecast.delays.DelayDetectionReport`
            describing what was compared and detected (counts, ignored columns, vanished
            values, frequencies, delay statistics), ready to be logged. Keyword-only

    Returns:
        DataFrame containing detected observations with publication delay information
        (or the tuple ``(DataFrame, DelayDetectionReport)`` if ``return_report`` is True).
        Its index is the entity levels of the data (none for a time series) followed by a
        level named ``'column'`` holding the variable name; a variable can appear on several
        rows. It contains the following columns, in this order:
        - 'observation_date': The date of the observation
        - 'has_changes': Always True (marks the detected observations)
        - 'download_date': Date when data was downloaded
        - 'frequency': Detected frequency of the (entity, column) couple, as a literal
          ('daily', 'weekly', 'monthly', 'quarterly', 'annual', ...), None if it could not
          be detected
        - 'period_start': Start boundary of the period (NaT if the frequency is None)
        - 'period_end': Exclusive end boundary of the period (NaT if the frequency is None)
        - 'reference_point': Reference point used ('start' or 'end')
        - 'delay': Calculated publication delay (ceil-rounded to the unit; NaN if the
          frequency is None)
        - 'unit': Unit of the delay value ('day', 'second' or 'microsecond')

        The frame is empty (with the same columns and index levels) when nothing is detected:
        no change since ``existing_data``, no observation at all, or no column.

    Raises:
        TypeError: If new_data or existing_data is not a DataFrame
        ValueError: If invalid parameters are provided, if new_data has no row, or if
            download_date cannot be resolved to a date

    Warns:
        UserWarning: If the frequency of some (entity, column) couples cannot be detected
            (for instance a single observation). They are still returned, with
            ``frequency=None`` and NaN period bounds and delay.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        >>> existing = pd.DataFrame({'GDP': [1.0, 2.0, 3.0, np.nan]}, index=dates)
        >>> new = pd.DataFrame({'GDP': [1.0, 2.0, 3.0, 4.0]}, index=dates)
        >>> result = compare_and_detect_delays(new, existing, download_date='2023-06-15')
        >>> result.index.tolist(), result['frequency'].tolist(), result['delay'].tolist(), result['unit'].tolist()
        (['GDP'], ['monthly'], [75.0], ['day'])
        >>> result['observation_date'].dt.strftime('%Y-%m-%d').tolist()
        ['2023-04-01']
        >>> compare_and_detect_delays(new, existing, '2023-06-15', reference_point='end')['delay'].tolist()
        [45.0]
    """
    # Validation des arguments
    # Validation du type des jeux de données
    for name, data in (('new_data', new_data), ('existing_data', existing_data)):
        if data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame, got {type(data).__name__}")

    # Validation du point de référence
    if reference_point not in ['start', 'end']:
        raise ValueError("reference_point must be 'start' or 'end'")
    # Validation du mode de détection
    if detection_mode not in ['new_only', 'all_changes']:
        raise ValueError("detection_mode must be 'new_only' or 'all_changes'")
    # Validation de l'unité du délai (avant tout calcul)
    _resolve_delay_unit(delay_unit)

    # Validation des jeux de données
    new_data = _validate_input_data(data=new_data, time_col=time_col, panel_cols=panel_cols)
    if existing_data is not None:
        existing_data = _validate_input_data(data=existing_data, time_col=time_col, panel_cols=panel_cols)

    # Un jeu sans aucune ligne n'a rien à analyser : refus explicite (un jeu avec des lignes mais
    # sans valeur observée, ou sans colonne, donne en revanche un résultat vide)
    if len(new_data) == 0:
        raise ValueError("Cannot detect publication delays on empty data: new_data has no row")

    # Validation de la date de téléchargement
    if download_date is None:
        download_date = datetime.now()
    download_date = resolve_date(date=download_date)

    # Logging
    logger.debug(
        "Detecting publication delays: mode=%s, reference_point=%s, unit=%s, existing_data=%s, %d rows",
        detection_mode, reference_point, delay_unit, existing_data is not None, len(new_data)
    )

    # Identification des nouvelles observations
    new_observations, comparison = _identify_new_observations(
        new_data=new_data,
        existing_data=existing_data,
        detection_mode=detection_mode
    )

    # Fonction de calcul des délais associés aux observations nouvellement publiées
    new_observations, frequency_map = _calculate_publication_delays(
        new_observations=new_observations,
        new_data=new_data,
        download_date=download_date,
        reference_point=reference_point,
        unit=delay_unit
    )

    # Rapport : toujours construit (coût négligeable), car il alimente aussi le journal
    report = _build_detection_report(
        delays=new_observations,
        comparison=comparison,
        frequency_map=frequency_map,
        new_data=new_data,
        existing_data=existing_data,
        download_date=download_date,
        detection_mode=detection_mode,
        reference_point=reference_point,
        unit_label=_resolve_delay_unit(delay_unit)[0]
    )
    # Logging
    logger.info(report.summary())

    if return_report:
        return new_observations, report
    return new_observations


# Fonction auxiliaire de résolution de l'unité du délai
def _resolve_delay_unit(unit: str) -> Tuple[str, int]:
    """Resolve a delay unit into its output label and its duration in nanoseconds.

    Args:
        unit: 'day'/'D', 'second'/'s' or 'microsecond'/'us'

    Returns:
        Tuple ``(label, nanoseconds)`` where label is 'day', 'second' or 'microsecond'

    Raises:
        ValueError: If unit is not one of 'us', 's', 'D', 'microsecond', 'second', 'day'
    """
    # Initialisation du message d'erreur
    error = ValueError(f"Unit must be one of 'us', 's', 'D', 'microsecond', 'second', 'day', got {unit}")
    # Une unité non textuelle ou inconnue des utilitaires de durée est refusée avec le même message
    if not isinstance(unit, str):
        raise error
    try:
        code = duration_to_code(unit)
    except ValueError:
        raise error from None
    # Les durées supportées au-delà de ces trois unités (heure, semaine, ...) ne sont pas des unités de délai
    if code not in _DELAY_UNIT_CODES:
        raise error
    return duration_to_literal(code), get_duration_nanoseconds(code)


# Fonction auxiliaire de validation des jeux de données en entrée
def _validate_input_data(data: pd.DataFrame, time_col: Optional[str] = None, panel_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Validate and prepare input data for analysis.

    This function uses the validate_temporal_data() function to validate and prepare
    time series or panel data structures.

    Args:
        data: Input pandas DataFrame to validate
        time_col: Name of the time column (optional)
        panel_cols: List of panel column names (optional)

    Returns:
        Validated and sorted DataFrame

    Raises:
        ValueError: If validation fails or invalid parameter combination
    """
    # Utilisation de validate_temporal_data pour valider les données
    data_validated = validate_temporal_data(
        data=data,
        time_col=time_col,
        panel_cols=panel_cols,
        strict=True,
        sort_data=True,
        return_metadata=False  # Pas besoin de métadonnées pour la reversion ici
    )

    return data_validated


# Fonction auxiliaire de marquage des observations détectées
def _flag_observations(index: pd.Index, column: Any) -> pd.DataFrame:
    """Build the detection frame of one column: one row per detected observation.

    Args:
        index: Index labels (dates, or entities and dates) of the detected observations
        column: Name of the variable the observations belong to

    Returns:
        DataFrame indexed by ``index`` with the columns 'column' and 'has_changes' (always True)
    """
    return pd.DataFrame(
        {'column': [column] * len(index), 'has_changes': np.ones(len(index), dtype=bool)},
        index=index
    )


# Fonction auxiliaire d'identification des nouvelles observations
def _identify_new_observations(new_data: pd.DataFrame, existing_data: Optional[pd.DataFrame] = None, detection_mode: str = 'new_only') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Identify new or changed observations.

    When existing_data is provided, compares the two DataFrames to identify changes.
    When existing_data is None, identifies the most recent non-null observation for
    each variable (and each panel entity if panel data).

    Args:
        new_data: New data DataFrame, validated and sorted
        existing_data: Existing data DataFrame for comparison. If None, identifies the
            most recent non-null observation for each variable/entity
        detection_mode: 'new_only' or 'all_changes'. Only used when existing_data is provided

    Returns:
        Tuple ``(observations, comparison)``: the DataFrame containing only new or changed
        observations with columns 'column' and 'has_changes', indexed by the dates (and
        entities) of the observations; and the counters of the comparison
        (``columns_compared``, ``columns_new_only``, ``columns_existing_only``,
        ``n_new_values``, ``n_revisions``, ``n_vanished_values``)
    """
    # Liste des résultats, un jeu par colonne (ordre des colonnes puis des index)
    frames = []
    # Compteurs de la comparaison, remontés au rapport
    comparison: Dict[str, Any] = {
        'columns_compared': tuple(new_data.columns),
        'columns_new_only': (),
        'columns_existing_only': (),
        'n_new_values': 0,
        'n_revisions': 0,
        'n_vanished_values': 0,
    }

    # Cas où existing_data n'est pas fourni : identification des observations les plus récentes
    if existing_data is None:
        # Niveaux d'entité du panel : toutes les dimensions sauf la dernière (le temps)
        is_panel = is_panel_data(new_data)
        entity_levels = get_entity_levels(new_data)

        # Parcours des colonnes
        for col in new_data.columns:
            # Observations non nulles, triées : la dernière de chaque entité est la plus récente
            observed = new_data[col].dropna()
            last = observed.groupby(level=entity_levels).tail(1) if is_panel else observed.tail(1)
            frames.append(_flag_observations(last.index, col))

        # Sans jeu de référence, la dernière observation de chaque entité est comptée comme nouvelle valeur
        comparison['n_new_values'] = sum(len(frame) for frame in frames)

        reference_index = new_data.index
    else:
        # Cas où existing_data est fourni
        # Colonnes communes, dans l'ordre de new_data (ordre déterministe)
        common_columns = [col for col in new_data.columns if col in existing_data.columns]

        # Restriction aux colonnes communes
        new_common_data = new_data[common_columns]
        existing_common_data = existing_data[common_columns]

        # Alignement des DataFrames sur l'index commun
        aligned_new, aligned_existing = new_common_data.align(existing_common_data, fill_value=np.nan)

        # Création des masques booléens pour les valeurs nulles
        new_isnull = aligned_new.isnull()
        existing_isnull = aligned_existing.isnull()

        # Détection des changements
        # Valeurs null → non-null : valeur existante null ET nouvelle valeur non nulle
        appeared_mask = existing_isnull & ~new_isnull
        if detection_mode == 'new_only':
            # Détection uniquement des valeurs null → non-null
            changes_mask = appeared_mask
            comparison['n_revisions'] = 0
        else:  # 'all_changes'
            # Valeurs révisées : valeur → autre valeur (calculées seulement dans ce mode)
            revised_mask = ~new_isnull & ~existing_isnull & (aligned_new != aligned_existing)
            # Détection de tous les changements (null → non-null ET valeur → nouvelle valeur)
            changes_mask = appeared_mask | revised_mask
            comparison['n_revisions'] = int(revised_mask.to_numpy().sum())

        # Mise en forme : une ligne par changement
        for col in common_columns:
            frames.append(_flag_observations(changes_mask.index[changes_mask[col].to_numpy()], col))

        # Compteurs de la comparaison
        comparison['columns_compared'] = tuple(common_columns)
        comparison['columns_new_only'] = tuple(col for col in new_data.columns if col not in existing_data.columns)
        comparison['columns_existing_only'] = tuple(col for col in existing_data.columns if col not in new_data.columns)
        comparison['n_new_values'] = int(appeared_mask.to_numpy().sum())
        # Valeurs disparues : non nulles dans existing_data, nulles ou absentes de new_data (jamais reportées)
        comparison['n_vanished_values'] = int((~existing_isnull & new_isnull).to_numpy().sum())

        reference_index = changes_mask.index

    # Concaténation, ou jeu vide d'index de même structure quand rien n'est détecté
    if frames:
        return pd.concat(frames), comparison
    return _flag_observations(reference_index[:0], None), comparison


# Fonction auxiliaire de construction du résultat vide
def _empty_publication_delays(new_observations: pd.DataFrame, download_date: datetime, reference_point: str, unit_label: str) -> pd.DataFrame:
    """Build the result frame when no observation is detected.

    Args:
        new_observations: Empty detection frame, indexed like the data (dates, or entities and dates)
        download_date: Date when the data was downloaded
        reference_point: 'start' or 'end'
        unit_label: Label of the delay unit

    Returns:
        Empty DataFrame with the output columns and the index levels of a non-empty result
        (entity levels, then 'column')
    """
    # Niveaux d'index du résultat : entités éventuelles, puis nom de la variable
    names = list(new_observations.index.names[:-1]) + ['column']
    if len(names) == 1:
        index = pd.Index([], name='column', dtype=object)
    else:
        index = pd.MultiIndex.from_arrays([[] for _ in names], names=names)

    # Colonnes de sortie, avec leurs types
    return pd.DataFrame(
        {
            'observation_date': pd.Series([], dtype='datetime64[ns]'),
            'has_changes': pd.Series([], dtype=bool),
            'download_date': pd.Series([], dtype='datetime64[ns]'),
            'frequency': pd.Series([], dtype=object),
            'period_start': pd.Series([], dtype='datetime64[ns]'),
            'period_end': pd.Series([], dtype='datetime64[ns]'),
            'reference_point': pd.Series([], dtype=object),
            'delay': pd.Series([], dtype=float),
            'unit': pd.Series([], dtype=object),
        },
        index=index
    )


# Fonction de calcul des délais de publication
def _calculate_publication_delays(new_observations: pd.DataFrame,
                                new_data: pd.DataFrame,
                                download_date: datetime,
                                reference_point: Literal['start', 'end'],
                                unit: Literal['us', 's', 'D', 'microsecond', 'second', 'day']
                                ) -> Tuple[pd.DataFrame, Dict[Any, Optional[str]]]:
    """Calculate publication delays for observed data points.

    Computes the delay between the observation date (period start or end) and the
    download date. Also enriches the observations with metadata including frequency,
    period boundaries, and delay values in the specified unit. The delay is computed
    with integer arithmetic, so a delay in microseconds is exact.

    Naive dates are read as UTC when the other operand is time-zone-aware.

    Args:
        new_observations: DataFrame with detected observations, indexed by datetime/multi-level
            and containing 'column' and 'has_changes' columns
        new_data: Original DataFrame containing the time series data
        download_date: Date when the data was downloaded (as datetime object)
        reference_point: Point used for delay calculation - 'start' (period start) or 'end'
            (exclusive period end)
        unit: Unit for delay values - 'day'/'D', 'second'/'s', or 'microsecond'/'us'

    Returns:
        Tuple ``(delays, frequency_map)``. ``delays`` is the DataFrame with computed
        publication delays containing columns:
        - 'observation_date': The date of the observation
        - 'has_changes': Always True
        - 'download_date': Date when data was downloaded
        - 'frequency': Detected frequency (literal), None if it could not be detected
        - 'period_start': Start boundary of the period (NaT without frequency)
        - 'period_end': Exclusive end boundary of the period (NaT without frequency)
        - 'reference_point': Reference point used ('start' or 'end')
        - 'delay': Calculated publication delay (ceil-rounded, NaN without frequency)
        - 'unit': Unit of the delay value

        ``frequency_map`` maps each (entity, column) key of ``new_data`` to its detected
        frequency literal (None if undetectable); it is empty when nothing is detected.

    Raises:
        ValueError: If unit is not one of 'us', 's', 'D', 'microsecond', 'second', 'day'

    Warns:
        UserWarning: If the frequency of some (entity, column) couples cannot be detected
    """
    # Résolution de l'unité du délai
    unit_label, unit_nanoseconds = _resolve_delay_unit(unit)

    # Cas où aucune observation n'est détectée : résultat vide de même structure
    if new_observations.empty:
        return _empty_publication_delays(new_observations, download_date, reference_point, unit_label), {}

    # Copie indépendante du jeu de données
    publication_delays = new_observations.copy()

    # Ajout de l'indicateur à l'index et suppression de la date
    publication_delays.set_index("column", drop=True, append=True, inplace=True)
    publication_delays.reset_index(
        level=publication_delays.index.nlevels-2,
        drop=False,
        inplace=True,
        names=[
            "observation_date" if i == publication_delays.index.nlevels-2 else i
            for i in range(publication_delays.index.nlevels)
        ],
    )

    # Ajout d'informations d'intérêt
    # Date de téléchargement
    publication_delays["download_date"] = download_date

    # Fréquence
    # Détection de la fréquence de chaque couple (entité, colonne) dans new_data : pour un DataFrame
    # le résultat est toujours un dictionnaire, la valeur None marquant un couple indétectable
    frequency_map_raw = cast(Dict[Any, Any], detect_frequency(new_data))
    frequency_map = {k: to_literal(v) if v is not None else None for k, v in frequency_map_raw.items()}

    # Conversion du dictionnaire en DataFrame (clés de panel aplaties en tuples : MultiIndex)
    freq_df = pd.Series(frequency_map, name="frequency", dtype=object).to_frame()

    # Ajout de la fréquence au jeu de données
    publication_delays = publication_delays.join(freq_df, on=publication_delays.index.names)
    publication_delays["frequency"] = publication_delays["frequency"].astype(object).where(
        publication_delays["frequency"].notna(), None
    )

    # Couples sans fréquence détectable : conservés, sans période ni délai, avec un avertissement
    detected = publication_delays["frequency"].notna().to_numpy()
    if not detected.all():
        # Couple non détectés
        undetected = list(dict.fromkeys(publication_delays.index[~detected]))
        # Warning
        warnings.warn(
            f"The frequency could not be detected for {undetected}: their period bounds and "
            f"delay are set to NaN",
            UserWarning,
            stacklevel=3
        )

    # Dates de début et de fin de la période sur laquelle porte l'observation
    # Calcul des bornes de période en fonction de la fréquence
    boundaries = [
        get_period_boundaries(date=date, frequency=frequency) if is_detected else (pd.NaT, pd.NaT)
        for date, frequency, is_detected in zip(
            publication_delays["observation_date"], publication_delays["frequency"], detected
        )
    ]
    publication_delays["period_start"] = pd.DatetimeIndex([bounds[0] for bounds in boundaries])
    publication_delays["period_end"] = pd.DatetimeIndex([bounds[1] for bounds in boundaries])

    # Point de référence
    publication_delays["reference_point"] = reference_point

    # Délai de publication
    # Le délai de publication est toujours arrondi à l'entier supérieur
    # Calcul du délai selon le point de référence
    if reference_point == "start":
        reference_dates = publication_delays["period_start"]
    else:  # "end"
        reference_dates = publication_delays["period_end"]

    # Mise en cohérence des fuseaux horaires : une date sans fuseau est lue en UTC face à une date avec fuseau
    download_timestamp = pd.Timestamp(download_date)
    if download_timestamp.tzinfo is None and reference_dates.dt.tz is not None:
        download_timestamp = download_timestamp.tz_localize('UTC')
    elif download_timestamp.tzinfo is not None and reference_dates.dt.tz is None:
        reference_dates = reference_dates.dt.tz_localize('UTC')

    # Calcul des délais en entiers de nanosecondes, puis arrondi à l'entier supérieur de l'unité
    # (division entière par excès : -(-n // u)) ; NaN pour les périodes inconnues
    delays_timedelta = download_timestamp - reference_dates
    known = delays_timedelta.notna().to_numpy()
    delays = np.full(len(publication_delays), np.nan)
    elapsed_nanoseconds = delays_timedelta[known].dt.as_unit('ns').astype('int64').to_numpy()
    delays[known] = -(-elapsed_nanoseconds // unit_nanoseconds)
    publication_delays["delay"] = delays
    publication_delays["unit"] = unit_label

    # Ordre des colonnes de sortie
    output_columns = [
        'observation_date', 'has_changes', 'download_date', 'frequency',
        'period_start', 'period_end', 'reference_point', 'delay', 'unit',
    ]
    return publication_delays[output_columns], frequency_map


# Fonction auxiliaire de construction du rapport de détection
def _build_detection_report(
    delays: pd.DataFrame,
    comparison: Dict[str, Any],
    frequency_map: Dict[Any, Optional[str]],
    new_data: pd.DataFrame,
    existing_data: Optional[pd.DataFrame],
    download_date: datetime,
    detection_mode: str,
    reference_point: str,
    unit_label: str,
) -> DelayDetectionReport:
    """Build the report of a detection from its result and its comparison counters.

    Args:
        delays: Result of ``_calculate_publication_delays``
        comparison: Counters returned by ``_identify_new_observations``
        frequency_map: Detected frequency of each (entity, column) key of ``new_data``
        new_data: Validated new dataset
        existing_data: Validated existing dataset, or None
        download_date: Resolved download date
        detection_mode: 'new_only' or 'all_changes'
        reference_point: 'start' or 'end'
        unit_label: Label of the delay unit

    Returns:
        The immutable :class:`DelayDetectionReport` of the call
    """
    # Nombre de détections par variable : la variable est le dernier niveau de l'index du résultat
    counts = delays.index.get_level_values(-1).value_counts() if len(delays) else pd.Series(dtype='int64')
    compared = comparison['columns_compared']
    n_detected_by_column = {col: int(counts.get(col, 0)) for col in compared}

    # Couples détectés dont la fréquence n'a pas pu être déterminée (clés uniques, ordre d'apparition)
    undetected = delays.index[delays['frequency'].isna().to_numpy()]
    undetected_keys = tuple(dict.fromkeys(undetected))

    # Statistiques des délais connus, dans l'unité du résultat
    stats = delay_statistics(delays['delay'])

    return DelayDetectionReport(
        detection_mode=detection_mode,
        reference_point=reference_point,
        delay_unit=unit_label,
        download_date=download_date,
        has_existing_data=existing_data is not None,
        n_rows_new=len(new_data),
        n_rows_existing=None if existing_data is None else len(existing_data),
        n_entities=len(get_unique_panel_entities(new_data)) if is_panel_data(new_data) else 0,
        n_columns=new_data.shape[1],
        columns_compared=compared,
        columns_new_only=comparison['columns_new_only'],
        columns_existing_only=comparison['columns_existing_only'],
        n_detected=len(delays),
        n_new_values=comparison['n_new_values'],
        n_revisions=comparison['n_revisions'],
        n_vanished_values=comparison['n_vanished_values'],
        n_detected_by_column=n_detected_by_column,
        columns_without_detection=tuple(col for col, n in n_detected_by_column.items() if n == 0),
        frequencies=dict(frequency_map),
        undetected_keys=undetected_keys,
        n_known=stats['n_known'],
        n_negative=stats['n_negative'],
        delay_min=stats['min'],
        delay_max=stats['max'],
        delay_mean=stats['mean'],
        delay_median=stats['median'],
    )
