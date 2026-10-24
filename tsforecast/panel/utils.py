"""Helpers for panel data: entity keys, panel structure, entity selection and frequencies.

A panel is a ``MultiIndex`` whose last level is the time and whose other levels identify the
entity. Entity keys are normalized to tuples (``'FR'`` and ``('FR',)`` designate the same
entity); the functions here detect the structure, list the entities, select their rows,
rebuild panel indexes and resolve per-entity (and per-column) target frequencies.
"""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

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

# Fonction d'identification des niveaux d'index associés aux entités
def get_entity_levels(data: Union[pd.DataFrame, pd.Series]) -> List[int]:
    """Get the positions of the index levels holding the entities of a panel.

    The entities are all the levels but the last one, which is the time.

    Args:
        data: Input data (DataFrame or Series).

    Returns:
        Positions of the entity levels, in order; an empty list when ``data`` is
        not panel data (see :func:`is_panel_data`).

    Examples:
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [('FR', 'a', '2023-01'), ('FR', 'b', '2023-01')], names=['c', 's', 'date']
        ... )
        >>> get_entity_levels(pd.Series([1, 2], index=idx))
        [0, 1]
        >>> get_entity_levels(pd.Series([1, 2], index=pd.date_range('2023', periods=2)))
        []
    """
    # Cas des données sans structure de panel : aucun niveau d'entité
    if not is_panel_data(data):
        return []
    return list(range(data.index.nlevels - 1))

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
            - panel_cols_final: List of detected panel columns (None if no panel).
              Detected from the entity levels of the index, it is a new plain list.
            - panel_in_index: True if panel is in the index, False if in columns

    Examples:
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [('FR', '2023-01'), ('DE', '2023-01')], names=['country', 'date']
        ... )
        >>> detect_panel_structure(pd.DataFrame({'x': [1, 2]}, index=idx), None)
        (['country'], True)
        >>> detect_panel_structure(pd.DataFrame({'x': [1, 2]}), ['country'])
        (['country'], False)
    """
    # Initialisation de l'indicateur que le panel est en index
    panel_in_index = False
    # Initialisation des noms des colonnes de panel
    panel_cols_final = panel_cols

    # Auto-détection si panel_cols non spécifié et index MultiIndex
    if panel_cols is None and isinstance(df.index, pd.MultiIndex):
        if df.index.nlevels >= 2:
            # Noms des niveaux de panel (tous sauf le dernier), en liste ordinaire
            panel_cols_final = list(df.index.names[:-1])
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

    Only the last level (the dates) is kept, so the function neither separates
    nor aggregates entities: call it on the rows of one entity. With several
    entities, every date appears once per entity and the index is not unique.

    Args:
        series: Series with MultiIndex where the last level is the date. A
            Series is required: a DataFrame is refused.

    Returns:
        Time series carrying the values and the name of ``series``, indexed by
        the dates only.

    Raises:
        ValueError: If ``series`` is two-dimensional.

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_product([['FR'], pd.date_range('2023-01-01', periods=2)])
        >>> series = extract_time_series_from_multiindex(pd.Series([1, 2], index=idx, name='gdp'))
        >>> series.index.strftime('%Y-%m-%d').tolist(), series.tolist(), series.name
        (['2023-01-01', '2023-01-02'], [1, 2], 'gdp')
    """
    # Refus des données à deux dimensions : seule une Series porte un nom et des valeurs uniques
    if series.ndim != 1:
        raise ValueError(f"Data must be 1-dimensional (a Series), got {series.ndim} dimensions")
    # Extraction de l'index temporel (dernier niveau)
    time_index = series.index.get_level_values(-1)
    # Création d'une série temporelle simple avec l'index temporel
    return pd.Series(series.values, index=time_index, name=series.name)

# Fonction de normalisation de l'entité du panel
def normalize_entity_key(key: Any) -> tuple:
    """Normalize entity key to tuple format.

    A scalar label designates a single-level entity: ``'FR'`` and ``('FR',)``
    are the same key. A list is read as the tuple of its labels, as a key coming
    from a JSON or YAML configuration is.

    Args:
        key: Entity key (scalar, tuple or list of labels).

    Returns:
        Normalized entity key as a hashable tuple.

    Raises:
        TypeError: If the key cannot be hashed (a dict, a set, an array, or a
            tuple holding one), hence cannot index a dictionary.

    Examples:
        >>> normalize_entity_key('FR')
        ('FR',)
        >>> normalize_entity_key(('EU', 'FR'))
        ('EU', 'FR')
        >>> normalize_entity_key(['EU', 'FR'])
        ('EU', 'FR')
    """
    # Une liste désigne les étiquettes d'une entité à plusieurs niveaux
    if isinstance(key, list):
        key = tuple(key)
    normalized = key if isinstance(key, tuple) else (key,)
    # Une clé d'entité sert de clé de dictionnaire : elle doit être hachable
    try:
        hash(normalized)
    except TypeError:
        raise TypeError(f"Entity key must be hashable, got {key!r}") from None
    return normalized

# Fonction de codage entier de clés de lignes composites
def factorize_row_keys(*keys: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Encode composite row keys as integer codes, in order of first appearance.

    Two rows share a code when they are equal on every key. Each key is
    factorized on its own, then the codes are combined key after key, so that
    no tuple is ever built per row. A missing value (NaN, NaT, None) is a value
    like any other: the rows that hold it share a code.

    Args:
        *keys: One-dimensional array-likes of the same length (arrays, Series,
            Index, MultiIndex level codes), one per component of the key.

    Returns:
        Tuple ``(codes, first_rows)``: ``codes[i]`` is the code of row ``i``
        (``0`` for the first distinct key, ``1`` for the next one, ...) and
        ``first_rows[k]`` the position of the first row of code ``k``. Without
        any key, both arrays are empty.

    Raises:
        ValueError: If the keys do not have the same length.

    Examples:
        >>> codes, first_rows = factorize_row_keys(['FR', 'FR', 'DE', 'FR'], ['M', 'Q', 'M', 'M'])
        >>> codes.tolist(), first_rows.tolist()
        ([0, 1, 2, 0], [0, 1, 2])
        >>> factorize_row_keys(pd.Series([1.0, None, None]))[0].tolist()
        [0, 1, 1]
    """
    # Absence de clé : aucune ligne
    if not keys:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)

    # Contrôle des longueurs
    n_rows = len(keys[0])
    if any(len(key) != n_rows for key in keys):
        raise ValueError(f"All keys must have the same length, got {[len(key) for key in keys]}")

    # Combinaison clé après clé : le code combiné reste borné par le nombre de lignes
    codes = np.zeros(n_rows, dtype=np.int64)
    for key in keys:
        # Une liste n'est pas factorisable directement par pandas
        values = np.asarray(key) if isinstance(key, list) else key
        key_codes, uniques = pd.factorize(values, use_na_sentinel=False)
        codes, _ = pd.factorize(codes * max(len(uniques), 1) + key_codes)

    # Première ligne de chaque code : les codes suivent l'ordre de première apparition
    _, first_rows = np.unique(codes, return_index=True)
    return codes.astype(np.intp, copy=False), first_rows

# Fonction de codage entier des entités de chaque ligne
def get_entity_codes(index: pd.Index) -> Tuple[np.ndarray, List[tuple]]:
    """Return the entity code of each row and the entity keys, in order of first appearance.

    The codes are read from the integer codes of the entity levels (all levels
    of a panel ``MultiIndex`` but the last), without building any tuple per
    row. A simple index is a time series: a single degenerate entity ``()``.

    Args:
        index: Index of the data: ``(entity..., date)`` ``MultiIndex`` for a
            panel, any simple index for a time series.

    Returns:
        Tuple ``(codes, entities)``: ``codes[i]`` is the position in
        ``entities`` of the entity of row ``i``, and ``entities`` the
        normalized entity keys (tuples). A missing entity label is a label like
        any other.

    Examples:
        >>> idx = pd.MultiIndex.from_product([['FR', 'DE'], pd.date_range('2023-01-01', periods=2)])
        >>> codes, entities = get_entity_codes(idx)
        >>> codes.tolist(), entities
        ([0, 0, 1, 1], [('FR',), ('DE',)])
        >>> get_entity_codes(pd.date_range('2023-01-01', periods=2))[1]
        [()]
    """
    # Série temporelle : une seule entité dégénérée
    if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
        return np.zeros(len(index), dtype=np.intp), [()]

    # Codes des niveaux d'entité, combinés sans construire de tuple par ligne
    n_entity_levels = index.nlevels - 1
    codes, first_rows = factorize_row_keys(*index.codes[:n_entity_levels])

    # Étiquettes de la première ligne de chaque entité, niveau par niveau
    labels = [index.get_level_values(level)[first_rows].tolist() for level in range(n_entity_levels)]
    return codes, [normalize_entity_key(tuple(entity)) for entity in zip(*labels)]

# Fonction auxiliaire de comparaison de deux valeurs déclarées
def _same_value(first: Any, second: Any) -> bool:
    """Tell whether two declared values are equal, a failing comparison meaning no.

    Args:
        first: First value.
        second: Second value.

    Returns:
        True if ``first == second`` evaluates to a true boolean.

    Examples:
        >>> _same_value('Q', 'Q'), _same_value('Q', 'M')
        (True, False)
        >>> _same_value(np.array([1, 2]), np.array([1, 2]))
        False
    """
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False

# Fonction de normalisation des clés d'une liaison par entité
def normalize_entity_mapping(mapping: Mapping[Any, Any]) -> Dict[tuple, Any]:
    """Re-key an entity -> value mapping on normalized entity keys.

    A scalar key designates a single-level entity, exactly like an entity
    argument passed to :func:`normalize_entity_key`: ``{'FR': 'Q'}`` and
    ``{('FR',): 'Q'}`` are the same binding. The single crossing point for
    the per-entity bindings read by the components of ``tsforecast.imputation``.

    Args:
        mapping: Mapping entity key -> value, keys scalar or tuple.

    Returns:
        A new dict holding the same values, keyed by tuples, in the order of
        ``mapping``. When a scalar key and its tuple form are both present with
        equal values, they make a single binding.

    Raises:
        ValueError: If a scalar key and its tuple form (or two notations of the
            same entity) hold different values: the binding is ambiguous.
        TypeError: If a key cannot be hashed once normalized.

    Examples:
        >>> normalize_entity_mapping({'FR': 'Q', ('DE',): 'M'})
        {('FR',): 'Q', ('DE',): 'M'}
        >>> normalize_entity_mapping({('Zone euro', 'IT'): 'Y'})
        {('Zone euro', 'IT'): 'Y'}
        >>> normalize_entity_mapping({'FR': 'Q', ('FR',): 'M'})
        Traceback (most recent call last):
        ...
        ValueError: Conflicting values for entity ('FR',): 'Q' and 'M'
    """
    # Initialisation du dictionnaire normalisé
    normalized: Dict[tuple, Any] = {}
    # Parcours des couples clé-valeur du mapping
    for key, value in mapping.items():
        # Normalisation de l'entité
        entity = normalize_entity_key(key)
        # Deux notations d'une même entité doivent porter la même valeur
        if entity in normalized and not _same_value(normalized[entity], value):
            raise ValueError(
                f"Conflicting values for entity {entity}: {normalized[entity]!r} and {value!r}"
            )
        normalized.setdefault(entity, value)
    return normalized

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
        List of unique normalized entity keys as tuples, in order of first
        appearance.

    Raises:
        ValueError: If panel_data has fewer than 2 index levels, or if an entity
            label is missing (NaN, None): such rows could be selected by no
            entity.

    Examples:
        >>> idx = pd.MultiIndex.from_product(
        ...     [['FR', 'DE'], pd.date_range('2023-01-01', periods=2)]
        ... )
        >>> get_unique_panel_entities(pd.Series([1, 2, 3, 4], index=idx))
        [('FR',), ('DE',)]
    """
    # Refus des données sans niveau d'entité ou aux étiquettes manquantes
    _check_panel_entity_labels(panel_data)

    # Combinaisons uniques d'entités, lues sur les codes des niveaux d'entité
    return get_entity_codes(panel_data.index)[1]


# Fonction auxiliaire de validation des étiquettes d'entité d'un panel
def _check_panel_entity_labels(panel_data: Union[pd.DataFrame, pd.Series]) -> None:
    """Reject data without entity level, or whose entity labels are missing.

    Args:
        panel_data: Panel data (DataFrame or Series).

    Raises:
        ValueError: If ``panel_data`` has fewer than 2 index levels, or if an
            entity label is missing (NaN, None): such rows could be selected
            by no entity.
    """
    # Refus des données sans niveau d'entité
    if not is_panel_data(panel_data):
        raise ValueError(
            "Panel data must have a MultiIndex with at least 2 levels "
            f"(entity levels then time), got {panel_data.index.nlevels} level(s)"
        )

    # Refus des étiquettes d'entité manquantes : aucun masque ne les sélectionne
    check_entity_labels(panel_data)

# Fonction de refus des étiquettes d'entité manquantes
def check_entity_labels(
    data: Union[pd.DataFrame, pd.Series, pd.Index],
    levels: Optional[Sequence[int]] = None,
    columns: Optional[Sequence[Any]] = None,
) -> None:
    """Reject entity labels that are missing (NaN, None).

    A row whose entity label is missing belongs to no entity: grouping would
    drop it silently, and no entity key could select it. Entity labels are
    read on index levels, on columns, or both.

    Args:
        data: Data (DataFrame or Series) or its index.
        levels: Positions of the entity levels of the index. ``None`` with
            ``columns`` also ``None``: every level but the last of a panel
            ``MultiIndex``, no level for a simple index.
        columns: Names of the entity columns of a DataFrame.

    Raises:
        ValueError: If an entity level or column holds a missing value.

    Examples:
        >>> idx = pd.MultiIndex.from_arrays(
        ...     [['FR', None], pd.date_range('2023-01-01', periods=2)], names=['country', 'date']
        ... )
        >>> check_entity_labels(idx)
        Traceback (most recent call last):
        ...
        ValueError: Entity labels must not be missing, level 'country' holds missing values
        >>> check_entity_labels(pd.DataFrame({'country': ['FR', 'DE']}), columns=['country'])
    """
    # Index des données
    index = data if isinstance(data, pd.Index) else data.index
    # Niveaux d'entité par défaut : tous sauf le dernier d'un panel
    if levels is None and columns is None:
        levels = range(index.nlevels - 1) if isinstance(index, pd.MultiIndex) else ()

    # Niveaux d'index : une étiquette manquante d'un MultiIndex a le code -1
    for position in levels or ():
        if isinstance(index, pd.MultiIndex):
            missing = bool((index.codes[position] == -1).any())
        else:
            missing = bool(index.isna().any())
        if missing:
            name = index.names[position]
            raise ValueError(
                f"Entity labels must not be missing, level "
                f"{name if name is not None else position!r} holds missing values"
            )

    # Colonnes d'entité
    for column in columns or ():
        if data[column].isna().any():
            raise ValueError(
                f"Entity labels must not be missing, column {column!r} holds missing values"
            )

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
        List of unique column names, in order of first appearance.

    Examples:
        >>> extract_column_names([('FR', 'gdp'), ('DE', 'gdp'), 'cpi'])
        ['gdp', 'cpi']
    """
    # Noms de colonnes distincts, dans l'ordre d'apparition (un dict garde l'ordre d'insertion)
    column_names: Dict[Any, None] = {}
    # Parcours des clés
    for key in keys:
        # Cas où la clé est un tuple : extraction du dernier élément
        column_names[key[-1] if isinstance(key, tuple) else key] = None
    return list(column_names)

# Fonction de normalisation d'une clé d'entité à sélectionner ou à créer
def _normalize_present_entity_key(entity) -> tuple:
    """Normalize an entity key and refuse missing labels.

    A missing label (NaN, None) is equal to nothing, itself included: an entity
    carrying one can neither be selected nor listed.

    Args:
        entity: Entity key (scalar or tuple).

    Returns:
        Normalized entity key as a tuple.

    Raises:
        ValueError: If a label of the key is missing.

    Examples:
        >>> _normalize_present_entity_key('FR')
        ('FR',)
        >>> _normalize_present_entity_key(('FR', float('nan')))
        Traceback (most recent call last):
        ...
        ValueError: Entity key ('FR', nan) holds a missing label
    """
    # Clé associée à l'entité
    key = normalize_entity_key(entity)
    # Cas invalides (label nul comme clé)
    if any(pd.api.types.is_scalar(label) and pd.isna(label) for label in key):
        raise ValueError(f"Entity key {key} holds a missing label")
    return key

# Fonction de construction du masque des observations associées à une entité
def get_entity_mask(
    X: Union[pd.DataFrame, pd.Series],
    entity: Union[tuple, Any],
) -> np.ndarray:
    """Get a boolean mask for rows belonging to a specific entity.

    The key is normalized first, so a scalar label selects a single-level
    entity: ``'FR'`` and ``('FR',)`` give the same mask. A time series (simple
    index) has no entity level: its only entity is ``()``, which selects every
    row.

    Args:
        X: DataFrame or Series with MultiIndex (entity levels + time level).
        entity: Entity key, one label per entity level (scalar or tuple).

    Returns:
        Boolean numpy array of shape (len(X),).

    Raises:
        ValueError: If the number of labels of the key differs from the number
            of entity levels of ``X``, or if a label of the key is missing.

    Examples:
        >>> idx = pd.MultiIndex.from_product(
        ...     [['FR', 'DE'], pd.date_range('2023-01-01', periods=2)]
        ... )
        >>> get_entity_mask(pd.Series([1, 2, 3, 4], index=idx), 'DE')
        array([False, False,  True,  True])
        >>> get_entity_mask(pd.Series([1, 2, 3, 4], index=idx), ('FR', 'DE'))
        Traceback (most recent call last):
        ...
        ValueError: Entity key ('FR', 'DE') has 2 label(s), the data has 1 entity level(s)
    """
    # Normalisation de la clé (un scalaire désigne une entité à un niveau)
    entity = _normalize_present_entity_key(entity)
    # Extraction de l'index
    index = X.index
    # Extraction des niveaux associés aux entités
    n_entity_levels = index.nlevels - 1

    # Contrôle du nombre d'étiquettes de la clé
    if len(entity) != n_entity_levels:
        raise ValueError(
            f"Entity key {entity} has {len(entity)} label(s), "
            f"the data has {n_entity_levels} entity level(s)"
        )

    # Construction du masque : une égalité par niveau d'entité (tout vrai sans niveau)
    mask = np.ones(len(X), dtype=bool)
    for level, label in enumerate(entity):
        mask &= (index.get_level_values(level) == label)
    return mask

# Fonction de lecture de la valeur déclarée pour une entité
def _entity_binding(mapping: Mapping[Any, Any], entity: tuple) -> Any:
    """Read the value a mapping declares for an entity, under any of its notations.

    A single-level entity may be declared by its tuple (``('FR',)``) or by its
    scalar label (``'FR'``). A value of None counts as undeclared.

    Args:
        mapping: Mapping whose keys may be entity tuples or scalar labels.
        entity: Entity key as a tuple.

    Returns:
        The declared value, or None when the entity is not declared.

    Raises:
        ValueError: If both notations are declared with different values.

    Examples:
        >>> _entity_binding({'FR': 'M'}, ('FR',))
        'M'
        >>> _entity_binding({('EU', 'FR'): 'Q'}, ('EU', 'FR'))
        'Q'
        >>> _entity_binding({'FR': 'M', ('FR',): 'Q'}, ('FR',))
        Traceback (most recent call last):
        ...
        ValueError: Conflicting values for entity ('FR',): 'Q' and 'M'
    """
    # Extraction de la valeur associée à l'entité
    tuple_value = mapping.get(entity)
    scalar_value = mapping.get(entity[0]) if len(entity) == 1 else None
    # Valeur non renseignée
    if tuple_value is None:
        return scalar_value
    # Valeur déjà présente pour l'entité
    if scalar_value is not None and not _same_value(tuple_value, scalar_value):
        raise ValueError(
            f"Conflicting values for entity {entity}: {tuple_value!r} and {scalar_value!r}"
        )
    return tuple_value

# Fonction d'extraction de la fréquence cible d'une entité
def get_entity_target_frequency(
    entity: tuple,
    target_frequency: Union[str, Dict],
) -> str:
    """Get the target frequency for a specific entity.

    A dict declares the entities by their tuple or, for a single-level entity,
    by their scalar label (``{'FR': 'M'}`` is ``{('FR',): 'M'}``). A value of
    None counts as undeclared.

    Args:
        entity: Entity tuple (e.g., ('FR',) or ('FR', 'GDP')).
        target_frequency: Either a single frequency string or a dict
            mapping entities to frequencies.

    Returns:
        Target frequency string for the entity.

    Raises:
        ValueError: If no target frequency found for the entity, or if both
            notations of the entity are declared with different frequencies.

    Examples:
        >>> get_entity_target_frequency(('FR',), 'M')
        'M'
        >>> get_entity_target_frequency(('FR',), {'FR': 'M', 'DE': 'Q'})
        'M'
    """
    # Cas où la fréquence cible est un dictionnaire
    if isinstance(target_frequency, dict):
        # Lecture de la déclaration de l'entité, sous la notation tuple ou scalaire
        freq = _entity_binding(target_frequency, normalize_entity_key(entity))
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
    ``(entity..., column)`` tuple, an entity (a tuple, or a scalar label for a
    single-level entity), or a plain column name. For each column the most
    specific key wins, following the precedence ``(entity, column)`` > entity >
    ``column``. A scalar key is read as the entity before being read as a
    column: with the entity ``('FR',)``, ``{'FR': 'Q'}`` covers every column.
    Columns without any matching key are omitted from the result (they are
    meant to be left unchanged by the caller). A value of None counts as
    undeclared.

    Args:
        entity: Entity tuple (e.g., ``('FR',)`` or ``('FR', 'sector1')``).
        columns: Column names to resolve a frequency for.
        target_frequency: Either a single frequency string (applied to every
            column) or a dict mapping mixed keys to frequencies.

    Returns:
        Dict mapping each covered column to its resolved frequency string.
        Columns with no matching key are absent from the dict.

    Raises:
        ValueError: If a single-level entity is declared both by its tuple and
            by its scalar label, with different frequencies.

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

    # Déclaration de l'entité (tuple ou scalaire), commune à toutes les colonnes
    entity_freq = _entity_binding(target_frequency, normalize_entity_key(entity))

    # Résolution par colonne selon la précédence (entité,col) > entité > col
    resolved: Dict[str, str] = {}
    for col in columns:
        # Clé la plus spécifique : entité + colonne
        pair_freq = target_frequency.get((*entity, col))
        if pair_freq is not None:
            resolved[col] = pair_freq
        elif entity_freq is not None:
            resolved[col] = entity_freq
        # Clé de colonne globale
        elif target_frequency.get(col) is not None:
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

    Raises:
        ValueError: On the first iteration, if the data is read as a panel but
            has a simple index, or if an entity label is missing (see
            :func:`get_unique_panel_entities`).

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
    entity_levels = get_entity_levels(data)

    # Entités du panel, codées une fois : le masque d'une entité compare des
    # codes entiers, sans relire les niveaux d'entité
    _check_panel_entity_labels(data)
    entity_codes, entities = get_entity_codes(data.index)

    # Parcours des entités du panel
    for code, entity in enumerate(entities):
        # Masque des observations de l'entité
        mask = entity_codes == code
        # Extraction du bloc, ramené à son seul index temporel
        yield entity, mask, data[mask].droplevel(entity_levels)

# Fonction de reconstruction d'un MultiIndex de panel pour une entité
def build_panel_index(
    entity: Union[tuple, Any],
    times: pd.Index,
    names: Optional[Sequence] = None,
) -> pd.MultiIndex:
    """Rebuild a panel MultiIndex from an entity key and its dates.

    Inverse of the block extraction performed by :func:`iter_entity_blocks`:
    re-attaches the entity levels dropped there. The key is normalized first:
    ``'FR'`` builds the same one-level entity as ``('FR',)``.

    Args:
        entity: Entity key (scalar or tuple, e.g. ``'FR'`` or ``('EU', 'FR')``).
        times: Dates of the entity.
        names: Names of the resulting index levels (entity levels then time).
            If None, the entity levels are unnamed and the time level keeps
            the name of ``times``.

    Returns:
        MultiIndex pairing ``entity`` with each date of ``times``.

    Raises:
        ValueError: If a label of the key is missing, or if ``names`` does not
            hold one name per level.

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=2, freq='MS')
        >>> build_panel_index(('FR',), dates, names=['entity', 'date'])
        MultiIndex([('FR', '2023-01-01'),
                    ('FR', '2023-02-01')],
                   names=['entity', 'date'])
        >>> build_panel_index('FR', dates.rename('date')).names
        FrozenList([None, 'date'])
    """
    # Normalisation de la clé (un scalaire désigne une entité à un niveau)
    entity = _normalize_present_entity_key(entity)
    # Noms par défaut : niveaux d'entité anonymes, niveau temporel nommé comme les dates
    if names is None:
        names = [*([None] * len(entity)), getattr(times, 'name', None)]
    # Réplication de chaque niveau d'entité sur la longueur de l'index temporel
    arrays = [np.full(len(times), level) for level in entity]
    # Reconstruction du MultiIndex (niveaux d'entité puis niveau temporel)
    return pd.MultiIndex.from_arrays([*arrays, times], names=names)
