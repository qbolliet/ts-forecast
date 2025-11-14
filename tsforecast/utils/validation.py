"""Temporal data validation utilities.

This module provides functions for validating and preparing time series
and panel data structures, including validation of datetime indexes,
uniqueness checks, and proper index configuration.
"""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Any, Tuple
import warnings


# Fonction de validation 
def validate_temporal_data(
    data: Union[pd.Series, pd.DataFrame],
    time_col: Optional[str] = None,
    panel_cols: Optional[List[str]] = None,
    strict: bool = True,
    sort_data: bool = True,
    return_metadata: bool = False
) -> Union[pd.DataFrame, pd.Series, Tuple[Union[pd.DataFrame, pd.Series], Dict[str, Any]]]:
    """Validate and prepare time series or panel data.

    This function validates temporal data structures (time series or panel data)
    and optionally converts time and panel columns into a proper index. It supports
    both simple time series and panel data with flexible validation modes.

    Args:
        data: Time series or panel data to validate (Series or DataFrame)
        time_col: Name of the time column (if None, uses index for validation)
        panel_cols: List of panel identifier column names (optional)
        strict: If True, raises errors on validation failures; if False, attempts corrections
        sort_data: If True, sorts data by index after validation
        return_metadata: If True, returns (data, metadata) tuple for later structure restoration

    Returns:
        Validated data (Series or DataFrame), or tuple (validated_data, metadata) if
        return_metadata=True. Metadata contains information needed to restore original structure.

    Raises:
        ValueError: If validation fails and strict=True, or if parameters are invalid

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.utils.validation import validate_temporal_data
        >>>
        >>> # Example 1: Simple time series with datetime index
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> validated = validate_temporal_data(series)
        >>>
        >>> # Example 2: DataFrame with time_col
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        ...     'value': [10, 20, 30, 40, 50]
        ... })
        >>> validated = validate_temporal_data(df, time_col='date')
        >>>
        >>> # Example 3: Panel data with time_col and panel_cols
        >>> panel_df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=6, freq='D').tolist() * 2,
        ...     'country': ['US']*6 + ['FR']*6,
        ...     'value': range(12)
        ... })
        >>> validated = validate_temporal_data(panel_df, time_col='date', panel_cols=['country'])
        >>>
        >>> # Example 4: With metadata for later restoration
        >>> validated, metadata = validate_temporal_data(df, time_col='date', return_metadata=True)
        >>> # ... processing ...
        >>> from tsforecast.utils.validation import restore_original_structure
        >>> original = restore_original_structure(validated, metadata)
    """
    # Validation des paramètres d'entrée
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Input data must be a pandas Series or DataFrame")

    # Vérification de la cohérence des paramètres
    if panel_cols and time_col is None:
        raise ValueError("Cannot specify panel_cols without time_col")

    # Conversion Series en DataFrame pour traitement uniforme
    is_series_input = isinstance(data, pd.Series)
    data_work = data.to_frame() if is_series_input else data.copy()

    # Construction des métadonnées si demandé
    if return_metadata:
        metadata = _build_metadata(data_work, time_col, panel_cols, sort_data)
    else:
        metadata = None

    # Détermination du mode de validation
    if time_col is None and not panel_cols:
        # Mode 1: Validation basée sur l'index
        data_validated = _validate_index_based(data_work, strict)
    else:
        # Mode 2: Validation basée sur les colonnes
        data_validated = _validate_column_based(data_work, time_col, panel_cols, strict)

    # Tri des données si demandé
    if sort_data:
        data_validated = data_validated.sort_index()

    # Retour au format Series si l'entrée était une Series
    if is_series_input:
        data_validated = data_validated.iloc[:, 0]

    # Retour avec ou sans métadonnées
    if return_metadata:
        return data_validated, metadata
    else:
        return data_validated

# Méthode de restauration de la structure originale du jeu de données (i.e. avant validation)
def restore_original_structure(
    data: Union[pd.Series, pd.DataFrame],
    metadata: Dict[str, Any]
) -> Union[pd.Series, pd.DataFrame]:
    """Restore original data structure from validation metadata.

    This function reverses the index transformations applied by validate_temporal_data(),
    restoring the original structure of the data including column positions and index names.

    Args:
        data: Validated data with modified index
        metadata: Metadata dictionary returned by validate_temporal_data()

    Returns:
        Data with original structure restored

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.utils.validation import validate_temporal_data, restore_original_structure
        >>>
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        ...     'value': [10, 20, 30, 40, 50]
        ... })
        >>> validated, metadata = validate_temporal_data(df, time_col='date', return_metadata=True)
        >>> # After processing
        >>> original = restore_original_structure(validated, metadata)
    """
    # Conversion Series en DataFrame pour traitement uniforme
    is_series = isinstance(data, pd.Series)
    data_work = data.to_frame() if is_series else data.copy()

    # ÉTAPE 1: Si l'index a été modifié (colonnes converties en index), restaurer en colonnes D'ABORD
    # Cette étape doit être faite EN PREMIER pour éviter les incompatibilités de types d'index
    # (par exemple DatetimeIndex vs RangeIndex) lors de reindex()
    if metadata.get('index_was_replaced', False):
        # Restauration de l'index en colonnes (time_col et panel_cols redeviennent des colonnes)
        data_work = data_work.reset_index()

    # ÉTAPE 2: Restauration de l'index original
    # Maintenant que les colonnes sont restaurées, on peut réassigner l'index original
    if 'original_index' in metadata:
        # Vérification de la compatibilité des tailles
        if len(data_work) == len(metadata['original_index']):
            data_work.index = metadata['original_index']

            # Restauration du nom de l'index si applicable
            if metadata.get('index_name'):
                data_work.index.name = metadata['index_name']
            elif metadata.get('index_names'):
                data_work.index.names = metadata['index_names']

    # Retour au format Series si l'entrée était une Series
    if is_series:
        return data_work.iloc[:, 0]

    return data_work

# Fonctions de validation spécialisées pour les données de panel
# Fonction de vérification que les observations du panel sont groupées par entité
def validate_entities_grouped(
    data: Union[pd.Series, pd.DataFrame],
    panel_cols: Optional[List[str]] = None
) -> bool:
    """Check if panel entities are grouped (contiguous, no discontinuities).

    For panel data, entities should appear in contiguous blocks without interleaving.
    This function validates that each entity's observations are adjacent in the data,
    which is important for efficient time-aware operations and cross-validation.

    Args:
        data: Input data (Series or DataFrame). Should have MultiIndex for panel data
            or be a DataFrame with panel_cols specified.
        panel_cols: List of panel identifier column names. If None, assumes MultiIndex
            with entities at level 0. For MultiIndex data, this parameter is ignored.

    Returns:
        True if entities are properly grouped (all observations for each entity are
        contiguous), False otherwise.

    Raises:
        ValueError: If data structure is incompatible with panel data validation
            (e.g., not a MultiIndex and no panel_cols specified).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from tsforecast.utils.validation import validate_entities_grouped
        >>>
        >>> # Example 1: Valid grouping with MultiIndex
        >>> entities = ['A', 'A', 'A', 'B', 'B', 'B']
        >>> dates = pd.date_range('2020-01-01', periods=6, freq='D')
        >>> idx = pd.MultiIndex.from_arrays([entities, dates], names=['entity', 'date'])
        >>> data = pd.DataFrame({'value': range(6)}, index=idx)
        >>> validate_entities_grouped(data)
        True
        >>>
        >>> # Example 2: Invalid grouping (interleaved entities)
        >>> entities_bad = ['A', 'B', 'A', 'B', 'A', 'B']
        >>> idx_bad = pd.MultiIndex.from_arrays([entities_bad, dates], names=['entity', 'date'])
        >>> data_bad = pd.DataFrame({'value': range(6)}, index=idx_bad)
        >>> validate_entities_grouped(data_bad)
        False
        >>>
        >>> # Example 3: Series with MultiIndex
        >>> series = pd.Series(range(6), index=idx)
        >>> validate_entities_grouped(series)
        True
    """
    # Vérification que les données sont bien de type Series ou DataFrame
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Input data must be a pandas Series or DataFrame")

    # Cas 1: Données avec MultiIndex (données de panel)
    if isinstance(data.index, pd.MultiIndex):
        # Extraction de tous les niveaux sauf le dernier (qui représente le temps)
        # Si un seul niveau d'entité (nlevels=2), extraction directe du niveau 0
        # Si plusieurs niveaux d'entité (nlevels>2), création de tuples pour chaque combinaison
        if data.index.nlevels == 2:
            entities = data.index.get_level_values(0)
        else:
            # Extraction de tous les niveaux sauf le dernier et création de tuples
            entity_levels = [data.index.get_level_values(i) for i in range(data.index.nlevels - 1)]
            entities = pd.Series(list(zip(*entity_levels)))
    # Cas 2: Données avec panel_cols spécifiés
    elif panel_cols is not None:
        if isinstance(data, pd.Series):
            raise ValueError("Cannot use panel_cols with Series input. Use MultiIndex instead.")
        if not all(col in data.columns for col in panel_cols):
            missing = set(panel_cols) - set(data.columns)
            raise ValueError(f"Panel columns not found in data: {missing}")
        # Si un seul panel_col, extraction directe
        if len(panel_cols) == 1:
            entities = data[panel_cols[0]]
        else:
            # Pour plusieurs panel_cols, création d'un tuple pour chaque ligne
            entities = pd.Series(list(zip(*[data[col] for col in panel_cols])))
    else:
        raise ValueError("Data must have MultiIndex or panel_cols must be specified for panel data validation")

    # Conversion en codes catégoriels pour comparaisons efficaces
    entity_codes = pd.Categorical(entities).codes
    unique_entities = np.unique(entity_codes)

    # Vérification vectorisée : pour chaque entité, ses positions doivent être consécutives
    for entity_code in unique_entities:
        # Extraction des positions de l'entité
        entity_positions = np.where(entity_codes == entity_code)[0]
        if len(entity_positions) > 1:
            # Vérification que les positions sont consécutives
            position_diffs = np.diff(entity_positions)
            if not np.all(position_diffs == 1):
                return False

    return True

# Fonction de vérification que les données du panel sont ordonnées au sein de chaque entité
def validate_sorted_within_groups(
    data: Union[pd.Series, pd.DataFrame],
    panel_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None
) -> bool:
    """Check if data is sorted by time within each panel group.

    For panel data, time values should be monotonically increasing within each
    entity group. This function validates that temporal ordering is preserved
    within each panel entity, which is critical for time-aware operations.

    Args:
        data: Input data (Series or DataFrame). Should have DatetimeIndex for time series
            or MultiIndex with time at the last level for panel data.
        panel_cols: List of panel identifier column names. If None, assumes MultiIndex
            with entities at level 0. For MultiIndex data, this parameter is ignored.
        time_col: Name of the time column. If None, uses the last level of MultiIndex
            for panel data or the index for time series data.

    Returns:
        True if dates are sorted within each group (monotonically increasing),
        False otherwise.

    Raises:
        ValueError: If data structure is incompatible with validation requirements
            or if time column cannot be identified.

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.utils.validation import validate_sorted_within_groups
        >>>
        >>> # Example 1: Valid sorting with MultiIndex
        >>> entities = ['A', 'A', 'A', 'B', 'B', 'B']
        >>> dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03',
        ...                         '2020-01-01', '2020-01-02', '2020-01-03'])
        >>> idx = pd.MultiIndex.from_arrays([entities, dates], names=['entity', 'date'])
        >>> data = pd.DataFrame({'value': range(6)}, index=idx)
        >>> validate_sorted_within_groups(data)
        True
        >>>
        >>> # Example 2: Invalid sorting (dates not sorted within entity B)
        >>> dates_bad = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03',
        ...                             '2020-01-03', '2020-01-01', '2020-01-02'])
        >>> idx_bad = pd.MultiIndex.from_arrays([entities, dates_bad], names=['entity', 'date'])
        >>> data_bad = pd.DataFrame({'value': range(6)}, index=idx_bad)
        >>> validate_sorted_within_groups(data_bad)
        False
        >>>
        >>> # Example 3: Time series (single entity) - always returns True for monotonic index
        >>> dates_ts = pd.date_range('2020-01-01', periods=5, freq='D')
        >>> series_ts = pd.Series(range(5), index=dates_ts)
        >>> validate_sorted_within_groups(series_ts)
        True
    """
    # Vérification que les données sont bien de type Series ou DataFrame
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Input data must be a pandas Series or DataFrame")

    # Cas 1: Données avec MultiIndex (données de panel)
    if isinstance(data.index, pd.MultiIndex):
        # Extraction de tous les niveaux sauf le dernier (qui représente le temps)
        # Si un seul niveau d'entité (nlevels=2), extraction directe du niveau 0
        # Si plusieurs niveaux d'entité (nlevels>2), création de tuples pour chaque combinaison
        if data.index.nlevels == 2:
            entities = data.index.get_level_values(0)
        else:
            # Extraction de tous les niveaux sauf le dernier et création de tuples
            entity_levels = [data.index.get_level_values(i) for i in range(data.index.nlevels - 1)]
            entities = pd.Series(list(zip(*entity_levels)))
        # Extraction du dernier niveau (dates/temps)
        dates = data.index.get_level_values(-1)
    # Cas 2: Données avec panel_cols et time_col spécifiés
    elif panel_cols is not None and time_col is not None:
        if isinstance(data, pd.Series):
            raise ValueError("Cannot use panel_cols and time_col with Series input. Use MultiIndex instead.")
        if not all(col in data.columns for col in panel_cols):
            missing = set(panel_cols) - set(data.columns)
            raise ValueError(f"Panel columns not found in data: {missing}")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")

        # Extraction des entités
        if len(panel_cols) == 1:
            entities = data[panel_cols[0]]
        else:
            # Pour plusieurs panel_cols, création d'un tuple pour chaque ligne
            entities = pd.Series(list(zip(*[data[col] for col in panel_cols])))

        # Extraction des dates
        dates = data[time_col]
    # Cas 3: Données de série temporelle simple (pas de panel)
    elif panel_cols is None and time_col is None:
        # Pour une série temporelle simple, vérification directe de l'index
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index.is_monotonic_increasing
        else:
            # Si ce n'est pas un DatetimeIndex, on ne peut pas valider
            raise ValueError("Time series data must have DatetimeIndex for temporal validation")
    else:
        raise ValueError("For panel data, both panel_cols and time_col must be specified, or data must have MultiIndex")

    # Utilisation de groupby pour vérification vectorisée
    try:
        grouped_dates = pd.Series(dates.values, index=entities).groupby(level=0)

        # Vérification que chaque groupe est monotone croissant
        for _, group_dates in grouped_dates:
            if not pd.Series(group_dates.values).is_monotonic_increasing:
                return False

        return True
    except Exception:
        # Fallback en cas d'erreur avec la méthode vectorisée
        return False


# Fonctions auxiliaires
# Fonction de validation sur l'index
def _validate_index_based(data: pd.DataFrame, strict: bool) -> pd.DataFrame:
    """Validate data using index as time reference.

    Args:
        data: Input DataFrame
        strict: Whether to raise errors or attempt corrections

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If index validation fails and strict=True
    """
    # Cas 1: Index simple
    if not isinstance(data.index, pd.MultiIndex):
        # Vérification et conversion en DatetimeIndex
        if not _check_datetime_convertible(data.index):
            if strict:
                raise ValueError("Index cannot be converted to datetime")
            else:
                warnings.warn("Index conversion to datetime failed")
                return data

        # Conversion si nécessaire
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                if strict:
                    raise ValueError(f"Failed to convert index to datetime: {e}")
                else:
                    warnings.warn(f"Index conversion failed: {e}")
                    return data

        # Vérification de l'unicité
        if not _check_uniqueness(data.index):
            if strict:
                raise ValueError("Index contains duplicate values")
            else:
                warnings.warn("Index contains duplicates. Keeping first occurrence.")
                data = data[~data.index.duplicated(keep='first')]

    # Cas 2: MultiIndex
    else:
        # Vérification que le dernier niveau est datetime
        last_level = data.index.get_level_values(-1)

        if not isinstance(last_level, pd.DatetimeIndex):
            if not _check_datetime_convertible(last_level):
                if strict:
                    raise ValueError("Last level of MultiIndex cannot be converted to datetime")
                else:
                    warnings.warn("MultiIndex last level conversion failed")
                    return data

            # Tentative de conversion
            try:
                new_levels = list(data.index.levels)
                new_levels[-1] = pd.to_datetime(new_levels[-1])
                new_codes = data.index.codes
                data.index = pd.MultiIndex(levels=new_levels, codes=new_codes, names=data.index.names)
            except Exception as e:
                if strict:
                    raise ValueError(f"Failed to convert MultiIndex last level: {e}")
                else:
                    warnings.warn(f"MultiIndex conversion failed: {e}")
                    return data

        # Vérification de l'unicité
        if not _check_uniqueness(data.index):
            if strict:
                raise ValueError("MultiIndex contains duplicate combinations")
            else:
                warnings.warn("MultiIndex contains duplicates. Keeping first occurrence.")
                data = data[~data.index.duplicated(keep='first')]

    return data

# Méthode de validation sur la base de colonnes de dates et d'entités
def _validate_column_based(
    data: pd.DataFrame,
    time_col: Optional[str],
    panel_cols: Optional[List[str]],
    strict: bool
) -> pd.DataFrame:
    """Validate data using time_col and panel_cols, then set as index.

    Args:
        data: Input DataFrame
        time_col: Time column name
        panel_cols: Panel identifier columns
        strict: Whether to raise errors or attempt corrections

    Returns:
        Validated DataFrame with new index

    Raises:
        ValueError: If validation fails and strict=True
    """
    # Vérification de la présence de time_col
    if time_col and time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")

    # Vérification de la présence des panel_cols
    if panel_cols:
        missing_cols = set(panel_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Panel columns not found in data: {missing_cols}")

    # Conversion de la colonne temporelle
    if time_col:
        if not _check_datetime_convertible(data[time_col]):
            raise ValueError(f"Column '{time_col}' cannot be converted to datetime")

        try:
            data[time_col] = pd.to_datetime(data[time_col])
        except Exception as e:
            raise ValueError(f"Failed to convert time column '{time_col}' to datetime: {e}")

    # Création des colonnes d'index
    index_cols = []
    if panel_cols:
        index_cols.extend(panel_cols)
    if time_col:
        index_cols.append(time_col)

    # Vérification de l'unicité
    if data.duplicated(subset=index_cols).any():
        if strict:
            raise ValueError(f"Duplicate rows found for combination of columns: {index_cols}")
        else:
            warnings.warn(f"Duplicates found for {index_cols}. Keeping first occurrence.")
            data = data.drop_duplicates(subset=index_cols, keep='first')

    # Définition du nouvel index
    data = data.set_index(index_cols)

    # Message d'avertissement sur le remplacement de l'index
    warnings.warn(
        f"Index replaced with {index_cols}. Use return_metadata=True and restore_original_structure() to revert.",
        UserWarning
    )

    return data

# Fonction auxiliaire de construction des méta-données associées à la transformation
def _build_metadata(
    data: pd.DataFrame,
    time_col: Optional[str],
    panel_cols: Optional[List[str]],
    sort_data: bool
) -> Dict[str, Any]:
    """Build metadata dictionary for later structure restoration.

    Args:
        data: Input DataFrame
        time_col: Time column name
        panel_cols: Panel identifier columns
        sort_data: Whether data will be sorted

    Returns:
        Dictionary containing original structure information
    """
    # Construction du dictionnaire de méta-données
    metadata = {
        'original_index': data.index.copy(),
        'original_columns': data.columns.tolist(),
        'index_type': type(data.index).__name__,
        'had_time_col_in_columns': time_col is not None and time_col in data.columns,
        'had_panel_cols_in_columns': bool(panel_cols) and all(col in data.columns for col in panel_cols),
        'index_was_replaced': time_col is not None or bool(panel_cols),
        'time_col': time_col,
        'panel_cols': panel_cols.copy() if panel_cols else None,
        'was_sorted': sort_data
    }

    # Stockage des noms d'index
    if isinstance(data.index, pd.MultiIndex):
        metadata['index_names'] = data.index.names
    else:
        metadata['index_name'] = data.index.name

    return metadata

# Fonction de vérification que l'index est convertible en datetime
def _check_datetime_convertible(index_or_series: Union[pd.Index, pd.Series]) -> bool:
    """Check if an index or series can be converted to datetime.

    Args:
        index_or_series: Index or Series to check

    Returns:
        True if convertible to datetime, False otherwise
    """
    # Déjà un DatetimeIndex
    if isinstance(index_or_series, pd.DatetimeIndex):
        return True

    # Tentative de conversion
    try:
        pd.to_datetime(index_or_series)
        return True
    except (ValueError, TypeError):
        return False

# Fonction de vérification de l'unicité des index
def _check_uniqueness(index: pd.Index) -> bool:
    """Check if an index contains only unique values.

    Args:
        index: Index to check

    Returns:
        True if all values are unique, False otherwise
    """
    return not index.duplicated().any()
