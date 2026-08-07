# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

# Fonction de détection de la structure de panel
def is_panel_data(data: Union[pd.DataFrame, pd.Series]) -> bool:
    """Check if data has a MultiIndex (panel) structure with 2+ levels.

    Panel data is identified by a MultiIndex whose first n-1 levels represent
    entities and whose last level represents time.

    Args:
        data: Input data (DataFrame or Series).

    Returns:
        True if data has a MultiIndex with at least 2 levels, False otherwise.

    Examples:
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [('FR', '2023-01'), ('FR', '2023-02')], names=['entity', 'date']
        ... )
        >>> is_panel_data(pd.Series([1, 2], index=idx))
        True
        >>> is_panel_data(pd.Series([1, 2], index=pd.date_range('2023', periods=2)))
        False
    """
    return isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 2

# Fonction de détection de la structure de panel d'un DataFrame
def detect_panel_structure(
    df: pd.DataFrame,
    panel_cols: Optional[List[str]],
) -> Tuple[Optional[List[str]], bool]:
    """Detect the panel structure of a DataFrame (columns or index).

    Args:
        df: DataFrame to analyze.
        panel_cols: List of specified panel columns (can be None).

    Returns:
        Tuple (panel_cols_final, panel_in_index) where:
            - panel_cols_final: List of detected panel columns (None if no panel)
            - panel_in_index: True if panel is in the index, False if in columns
    """
    # Initialisation de l'indicateur que le panel est en index
    panel_in_index = False
    # Initialisation des noms des colonnes de panel
    panel_cols_final = panel_cols

    # Auto-détection si panel_cols non spécifié et index MultiIndex
    if panel_cols is None and isinstance(df.index, pd.MultiIndex):
        if df.index.nlevels >= 2:
            # Extraction des noms des niveaux de panel (tous sauf le dernier)
            panel_cols_final = df.index.names[:-1]
            # Conversion en liste si nécessaire
            if not isinstance(panel_cols_final, list):
                panel_cols_final = list(panel_cols_final)
            panel_in_index = True
    # Vérification si panel_cols spécifiés sont dans l'index
    elif panel_cols is not None and isinstance(df.index, pd.MultiIndex):
        if all(col in df.index.names for col in panel_cols):
            panel_cols_final = panel_cols
            panel_in_index = True

    return panel_cols_final, panel_in_index

# Fonction d'extraction de la série temporelle (dernier niveau) d'un MultiIndex
def extract_time_series_from_multiindex(series: pd.Series) -> pd.Series:
    """Extract a simple time series from a series with MultiIndex.

    Args:
        series: Series with MultiIndex where the last level is the date.

    Returns:
        Simple time series with only the time index.
    """
    # Extraction de l'index temporel (dernier niveau)
    time_index = series.index.get_level_values(-1)
    # Création d'une série temporelle simple avec l'index temporel
    return pd.Series(series.values, index=time_index)

# Fonction de normalisation de l'entité du panel
def normalize_entity_key(key) -> tuple:
    """Normalize entity key to tuple format.

    Args:
        key: Entity key (scalar or tuple).

    Returns:
        Normalized entity key as a tuple.
    """
    if isinstance(key, tuple):
        return key
    return (key,)

# Fonction de décomposition d'une clé de variable en (entité, colonne)
def split_variable_key(key: Union[str, Tuple]) -> Tuple[tuple, str]:
    """Split a variable key into its entity tuple and its column name.

    A panel variable key is an ``(entity..., column)`` tuple, as produced by
    ``detect_dataset_frequency``; a time series key is a plain column name.
    The entity part is ALWAYS returned as a tuple — ``('FR',)`` for a
    single-level panel, ``()`` for a time series — so callers never have to
    branch on the number of entity levels, and the returned entity always
    matches the keys of :func:`get_unique_panel_entities`.

    Args:
        key: Variable identifier: a column name, or an ``(entity..., column)``
            tuple.

    Returns:
        Tuple ``(entity_tuple, column)``.

    Examples:
        >>> split_variable_key(('FR', 'gdp'))
        (('FR',), 'gdp')
        >>> split_variable_key(('FR', 'manufacturing', 'gdp'))
        (('FR', 'manufacturing'), 'gdp')
        >>> split_variable_key('gdp')
        ((), 'gdp')
    """
    # Cas d'une clé de série temporelle : aucune entité
    if not isinstance(key, tuple):
        return (), key

    # Cas d'une clé de panel : l'entité est toujours le préfixe de la clé
    return tuple(key[:-1]), key[-1]

# Fonction d'identification des unités d'un panel
def get_unique_panel_entities(panel_data: Union[pd.DataFrame, pd.Series]) -> List[tuple]:
    """Extract unique entities from panel data.

    Assumes panel_data has a MultiIndex where the first n-1 levels represent
    entities and the last (n-th) level represents time/date. This function
    extracts all unique entity combinations and normalizes them as tuples.

    Args:
        panel_data: Panel data (DataFrame or Series) with MultiIndex structure
            where entity levels come first, followed by a time/date level.

    Returns:
        List of unique normalized entity keys as tuples.

    Raises:
        IndexError: If panel_data has fewer than 2 index levels.
    """
    # Extraction des levels d'entité (tous sauf le dernier qui est la date)
    entity_index = panel_data.index.droplevel(-1)
    
    # Récupération des combinaisons uniques d'entités
    unique_entities = entity_index.unique()
    
    # Normalisation des clés d'entité
    normalized_entities = [normalize_entity_key(entity) for entity in unique_entities]

    return normalized_entities

# Fonction de regroupement des variables par entité
def group_keys_by_entity_and_variable(
    keys: List[Union[str, Tuple]],
) -> Dict[tuple, List[str]]:
    """Group variable keys by entity, extracting column names.

    Args:
        keys: List of (entity..., variable) tuples or plain column names.

    Returns:
        Dict mapping entity tuples to lists of column names.
        For time series (non-panel), returns {(): [col_names]}.

    Examples:
        >>> group_keys_by_entity_and_variable([('FR', 'gdp'), ('FR', 'cpi'), ('DE', 'gdp')])
        {('FR',): ['gdp', 'cpi'], ('DE',): ['gdp']}
        >>> group_keys_by_entity_and_variable(['gdp', 'cpi'])
        {(): ['gdp', 'cpi']}
    """
    # Initialisation du dictionnaire résultat
    grouped: Dict[tuple, List[str]] = {}

    # Parcours des clés
    for key in keys:
        # Décomposition de la clé : entité toujours en tuple, () hors panel
        entity, col = split_variable_key(key)
        # Initialisation de l'entité si elle n'existe pas déjà
        if entity not in grouped:
            grouped[entity] = []
        # Ajout de la colonne à l'entité
        if col not in grouped[entity]:
            grouped[entity].append(col)

    return grouped

# Fonction d'extraction des noms de colonnes à partir des clés de variables
def extract_column_names(
    keys: List[Union[str, Tuple]],
) -> List[str]:
    """Extract unique column names from variable keys.

    Args:
        keys: List of variable identifiers (column names or
            (entity..., column) tuples).

    Returns:
        List of unique column names.

    Examples:
        >>> sorted(extract_column_names([('FR', 'gdp'), ('DE', 'gdp'), 'cpi']))
        ['cpi', 'gdp']
    """
    # Initialisation des noms de colonnes
    column_names = set()
    # Parcours des clés
    for key in keys:
        # Cas où la clé est un tuple : extraction du dernier élément
        if isinstance(key, tuple):
            column_names.add(key[-1])
        else:
            column_names.add(key)
    return list(column_names)

# Fonction de construction du masque des observations associées à une entité
def get_entity_mask(
    X: Union[pd.DataFrame, pd.Series],
    entity: tuple,
) -> np.ndarray:
    """Get a boolean mask for rows belonging to a specific entity.

    Args:
        X: DataFrame or Series with MultiIndex (entity levels + time level).
        entity: Entity tuple to filter on.

    Returns:
        Boolean numpy array of shape (len(X),).
    """
    # Extraction de l'index
    index = X.index
    # Extraction des niveaux associés aux entités
    n_entity_levels = index.nlevels - 1

    # Cas où les entités sont sur un seul niveau
    if n_entity_levels == 1:
        return (index.get_level_values(0) == entity[0])
    else:
        # Construction d'un masque pour les entités multi-niveaux
        # Initialisation du masque
        mask = np.ones(len(X), dtype=bool)
        # Parcours des niveaux associés aux entités
        for i in range(n_entity_levels):
            # Mise à jour du masque
            mask &= (index.get_level_values(i) == entity[i])
        return mask

# Fonction d'extraction de la fréquence cible d'une entité
def get_entity_target_frequency(
    entity: tuple,
    target_frequency: Union[str, Dict],
) -> str:
    """Get the target frequency for a specific entity.

    Args:
        entity: Entity tuple (e.g., ('FR',) or ('FR', 'GDP')).
        target_frequency: Either a single frequency string or a dict
            mapping entities to frequencies.

    Returns:
        Target frequency string for the entity.

    Raises:
        ValueError: If no target frequency found for the entity.

    Examples:
        >>> get_entity_target_frequency(('FR',), 'M')
        'M'
        >>> get_entity_target_frequency(('FR',), {'FR': 'M', 'DE': 'Q'})
        'M'
    """
    # Cas où la fréquence cible est un dictionnaire
    if isinstance(target_frequency, dict):
        # Extraction de la fréquence cible
        freq = target_frequency.get(entity)
        # Recherche en extrayant l'entité du tuple
        if freq is None and len(entity) == 1:
            freq = target_frequency.get(entity[0])
        # Cas où la fréquence n'est pas trouvée
        if freq is None:
            raise ValueError(
                f"No target frequency found for entity {entity}"
            )
        return freq
    # Dans le cas où la fréquence cible est une chaîne de caractère, elle s'applique à toutes les entités
    return target_frequency

# Fonction de résolution des fréquences cibles par colonne pour une entité
def resolve_entity_column_frequencies(
    entity: tuple,
    columns: List[str],
    target_frequency: Union[str, Dict],
) -> Dict[str, str]:
    """Resolve the target frequency of each column for a given panel entity.

    Supports a target specification whose keys may be mixed: an
    ``(entity..., column)`` tuple, an ``(entity...)`` tuple, or a plain column
    name. For each column the most specific key wins, following the precedence
    ``(entity, column)`` > ``(entity,)`` > ``column``. Columns without any
    matching key are omitted from the result (they are meant to be left
    unchanged by the caller).

    Args:
        entity: Entity tuple (e.g., ``('FR',)`` or ``('FR', 'sector1')``).
        columns: Column names to resolve a frequency for.
        target_frequency: Either a single frequency string (applied to every
            column) or a dict mapping mixed keys to frequencies.

    Returns:
        Dict mapping each covered column to its resolved frequency string.
        Columns with no matching key are absent from the dict.

    Examples:
        >>> resolve_entity_column_frequencies(('FR',), ['gdp', 'cpi'], 'M')
        {'gdp': 'M', 'cpi': 'M'}
        >>> resolve_entity_column_frequencies(
        ...     ('FR',), ['gdp', 'cpi'],
        ...     {('FR', 'gdp'): 'Q', ('FR',): 'M'}
        ... )
        {'gdp': 'Q', 'cpi': 'M'}
        >>> resolve_entity_column_frequencies(
        ...     ('DE',), ['gdp', 'cpi'], {'cpi': 'ME'}
        ... )
        {'cpi': 'ME'}
    """
    # Cas d'une chaîne : même fréquence pour toutes les colonnes
    if not isinstance(target_frequency, dict):
        return {col: target_frequency for col in columns}

    # Résolution par colonne selon la précédence (entité,col) > (entité,) > col
    resolved: Dict[str, str] = {}
    for col in columns:
        # Clé la plus spécifique : entité + colonne
        entity_col_key = (*entity, col)
        if entity_col_key in target_frequency:
            resolved[col] = target_frequency[entity_col_key]
        # Clé d'entité (tuple), avec repli sur le scalaire si entité mono-niveau
        elif entity in target_frequency:
            resolved[col] = target_frequency[entity]
        elif len(entity) == 1 and entity[0] in target_frequency:
            resolved[col] = target_frequency[entity[0]]
        # Clé de colonne globale
        elif col in target_frequency:
            resolved[col] = target_frequency[col]
        # Sinon : colonne non couverte, omise (laissée inchangée)

    return resolved

# Fonction d'itération sur les blocs d'entités d'un jeu de données
def iter_entity_blocks(
    data: Union[pd.DataFrame, pd.Series],
    is_panel: Optional[bool] = None,
) -> Iterator[Tuple[tuple, np.ndarray, Union[pd.DataFrame, pd.Series]]]:
    """Iterate over the entity blocks of a dataset, each indexed by date only.

    Unifies the time series and panel cases: a time series is yielded as a
    degenerate panel holding a single entity ``()`` covering every row, so
    callers can share one processing loop instead of branching on the data
    structure. Panel blocks have their entity levels dropped, hence carry a
    plain DatetimeIndex just like a time series.

    Args:
        data: Time series (DatetimeIndex) or panel data (MultiIndex whose
            first n-1 levels are entities and last level is time).
        is_panel: Force the structure interpretation. If None, it is inferred
            with :func:`is_panel_data`.

    Yields:
        Tuple ``(entity, mask, block)`` where ``entity`` is the entity key
        (``()`` for a time series), ``mask`` a boolean array selecting the
        entity's rows in ``data``, and ``block`` the entity's data indexed by
        date only.

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=2, freq='MS')
        >>> idx = pd.MultiIndex.from_product([['FR', 'DE'], dates])
        >>> df = pd.DataFrame({'x': range(4)}, index=idx)
        >>> [entity for entity, _, _ in iter_entity_blocks(df)]
        [('FR',), ('DE',)]
        >>> ts = pd.DataFrame({'x': range(2)}, index=dates)
        >>> [entity for entity, _, _ in iter_entity_blocks(ts)]
        [()]
    """
    # Résolution de la structure des données
    if is_panel is None:
        is_panel = is_panel_data(data)

    # Cas des séries temporelles : un unique bloc d'entité vide couvrant tout
    if not is_panel:
        yield (), np.ones(len(data), dtype=bool), data
        return

    # Niveaux d'entité à retirer pour ramener chaque bloc à un index daté
    entity_levels = list(range(data.index.nlevels - 1))

    # Parcours des entités du panel
    for entity in get_unique_panel_entities(data):
        # Masque des observations de l'entité
        mask = get_entity_mask(data, entity)
        # Extraction du bloc, ramené à son seul index temporel
        yield entity, mask, data[mask].droplevel(entity_levels)

# Fonction de reconstruction d'un MultiIndex de panel pour une entité
def build_panel_index(
    entity: tuple,
    times: pd.Index,
    names: Optional[Sequence] = None,
) -> pd.MultiIndex:
    """Rebuild a panel MultiIndex from an entity key and its dates.

    Inverse of the block extraction performed by :func:`iter_entity_blocks`:
    re-attaches the entity levels dropped there.

    Args:
        entity: Entity key as a tuple (e.g. ``('FR',)``).
        times: Dates of the entity.
        names: Names of the resulting index levels (entity levels then time).

    Returns:
        MultiIndex pairing ``entity`` with each date of ``times``.

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=2, freq='MS')
        >>> build_panel_index(('FR',), dates, names=['entity', 'date'])
        MultiIndex([('FR', '2023-01-01'),
                    ('FR', '2023-02-01')],
                   names=['entity', 'date'])
    """
    # Réplication de chaque niveau d'entité sur la longueur de l'index temporel
    arrays = [np.full(len(times), level) for level in entity]
    # Reconstruction du MultiIndex (niveaux d'entité puis niveau temporel)
    return pd.MultiIndex.from_arrays([*arrays, times], names=names)
