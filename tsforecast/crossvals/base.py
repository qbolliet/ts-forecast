"""Base time-aware cross-validators: ``OutOfSampleSplit`` and ``InSampleSplit``.

``OutOfSampleSplit`` trains strictly before each test period (optional ``gap``
and rolling window); ``InSampleSplit`` yields one fold whose training set
contains the test period. Both split a time series, or panel data when
``groups`` are given, and share private helpers that validate the parameters
and the data, sort unsorted rows and resolve the requested ``test_indices``.
A panel has a MultiIndex whose last level is the date and whose other levels
identify the entity.
"""
# Importation des modules
# Modules de base
import datetime
import warnings
from numbers import Integral
from typing import Any, Optional

import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

# Utilitaires de panel et de validation
from ..panel.utils import get_entity_levels, is_panel_data
from ..utils.validation import validate_entities_grouped, validate_sorted_within_groups

# Natures possibles des indices de test demandés
_POSITIONS = 'positions'
_DATES = 'dates'
_PAIRS = 'pairs'

# Types reconnus comme des dates (Timestamp et datetime dérivent de datetime.date)
_DATE_TYPES = (str, datetime.date, np.datetime64, pd.Period)

# Conteneurs lus élément par élément ; toute autre valeur est un indice unique
_CONTAINER_TYPES = (list, tuple, np.ndarray, pd.Index, pd.Series)


# =============================================================================
# Validation des paramètres et des entrées
# =============================================================================

# Fonction auxiliaire de vérification s'il s'agit d'une Series ou d'un DataFrame pandas
def _is_pandas(X: Any) -> bool:
    """Return whether ``X`` is a pandas Series or DataFrame."""
    # Test de type explicite : une liste possède aussi un attribut 'index' (méthode)
    return isinstance(X, (pd.Series, pd.DataFrame))


# Vérification qu'une valeur est un entier positif ou None
def _check_positive_or_none(name: str, value: Any) -> None:
    """Raise a ValueError unless ``value`` is None or an integer of at least 1."""
    # Paramètre non renseigné : valeur par défaut calculée par le splitter
    if value is None:
        return
    # Refus des booléens (entiers pour Python), des non-entiers et des valeurs inférieures à 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")


# Vérification que le 'gap' est un entier positif ou nul
def _check_gap(gap: Any) -> None:
    """Raise a ValueError unless ``gap`` is a non-negative integer."""
    # Un écart négatif placerait des lignes de test dans l'apprentissage
    if isinstance(gap, (bool, np.bool_)) or not isinstance(gap, Integral) or gap < 0:
        raise ValueError(f"gap must be a non-negative integer, got {gap!r}")


# Vérification de l'alignement de X et y
def _check_target(X: Any, y: Any) -> None:
    """Raise a ValueError when X and y are both pandas objects with different indexes."""
    # Contrôle possible seulement si X et y portent tous deux un index pandas
    if y is not None and _is_pandas(X) and _is_pandas(y) and not X.index.equals(y.index):
        raise ValueError("'X' and 'y' should have the same indexes")


# Fonction de conversion d'une suite d'étiquettes en tableau à une dimension
def _labels_array(values: Any) -> np.ndarray:
    """Return ``values`` as a one-dimensional object array, tuples kept whole.

    ``np.asarray`` would turn a list of tuples into a two-dimensional array;
    each label is therefore placed element by element.
    """
    labels = list(values)
    result = np.empty(len(labels), dtype=object)
    result[:] = labels
    return result


# Fonction d'extraction de l'entité de chaque ligne d'un panel
def _entity_labels(X: Any) -> np.ndarray:
    """Return the entity of each row of panel data.

    The entity is the value of the entity level when there is one, the tuple
    of the entity levels (all levels but the last) when there are several.

    Examples:
        >>> idx = pd.MultiIndex.from_product([['FR'], ['a', 'b'], [2020, 2021]])
        >>> _entity_labels(pd.Series(range(4), index=idx)).tolist()
        [('FR', 'a'), ('FR', 'a'), ('FR', 'b'), ('FR', 'b')]
    """
    # Suppression du niveau temporel (dernier niveau) : il reste les niveaux d'entité
    return _labels_array(X.index.droplevel(-1))


# Fonction d'extraction des dates de chaque ligne d'un panel
def _time_level(X: Any) -> pd.Index:
    """Return the date of each row of panel data (last index level)."""
    return X.index.get_level_values(-1)


# =============================================================================
# Lecture des indices de test demandés
# =============================================================================

# Fonction de détermination de la nature d'un indice de test
def _kind_of(item: Any) -> str:
    """Return the nature of one requested test index.

    Raises:
        ValueError: If the item is a boolean or neither an integer position, a
            date nor an ``(entity..., date)`` tuple.
    """
    # Un booléen est un entier pour Python : refus explicite avant le test d'entier
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(
            f"test_indices must hold integer positions, dates or (entity, date) tuples, "
            f"got the boolean {item!r}"
        )
    # Entier Python ou numpy : position d'une ligne
    if isinstance(item, Integral):
        return _POSITIONS
    # Tuple : ligne d'une entité précise d'un panel
    if isinstance(item, tuple):
        return _PAIRS
    # Chaîne, date, Timestamp, datetime64 ou Period : date à rechercher dans l'index
    if isinstance(item, _DATE_TYPES):
        return _DATES
    # Tout autre type (flottant, None, ...) : refus en nommant le type reçu
    raise ValueError(
        f"test_indices must hold integer positions, dates or (entity, date) tuples, "
        f"got {item!r} of type {type(item).__name__}"
    )


# Fonction auxiliaire de parsing des indices de test
def _parse_test_indices(test_indices: Any) -> Optional[tuple[str, list]]:
    """Split ``test_indices`` into its nature and the list of its items.

    Args:
        test_indices: None, a single index, or a list, tuple, array, Index or
            Series of indices.

    Returns:
        None when no index is given (None or an empty container), otherwise
        ``(kind, items)`` with ``kind`` one of ``'positions'``, ``'dates'``,
        ``'pairs'``.

    Raises:
        ValueError: If an item is not a valid test index, or if positions,
            dates and pairs are mixed.

    Examples:
        >>> _parse_test_indices([3, 7])
        ('positions', [3, 7])
        >>> _parse_test_indices('2024-04-30')
        ('dates', ['2024-04-30'])
        >>> _parse_test_indices([]) is None
        True
    """
    # Aucun indice demandé : découpage par défaut
    if test_indices is None:
        return None
    # Conteneur lu élément par élément ; valeur isolée traitée comme un indice unique
    items = list(test_indices) if isinstance(test_indices, _CONTAINER_TYPES) else [test_indices]
    # Conteneur vide : équivalent à l'absence d'indices
    if not items:
        return None
    # Détermination de la nature de chaque élément (et non du seul premier)
    kinds = {_kind_of(item) for item in items}
    # Refus d'un mélange de natures, ambigu à résoudre
    if len(kinds) > 1:
        raise ValueError(
            f"test_indices must not mix integer positions, dates and (entity, date) tuples, "
            f"got {sorted(kinds)}"
        )
    return kinds.pop(), items


# Normalisation d'une date au type de l'index dans lequel elle est recherchée
def _normalize_label(index: pd.Index, label: Any) -> Any:
    """Convert a date to the type of the index it is looked up in.

    Strings are left as they are (pandas parses them against a datetime
    index); other dates become a ``Timestamp`` (localized to the index time
    zone) on a ``DatetimeIndex`` and a ``Period`` on a ``PeriodIndex``.
    """
    # Chaîne : interprétée par pandas selon le type de l'index
    if isinstance(label, str):
        return label
    # Index de dates : conversion en Timestamp, dans le fuseau horaire de l'index
    if isinstance(index, pd.DatetimeIndex):
        stamp = pd.Timestamp(label)
        if index.tz is not None and stamp.tz is None:
            stamp = stamp.tz_localize(index.tz)
        return stamp
    # Index de périodes : conversion en période de même fréquence
    if isinstance(index, pd.PeriodIndex) and not isinstance(label, pd.Period):
        return pd.Period(label, freq=index.freq)
    # Autre index (libellés) : comparaison directe
    return label


# Recherche de la position d'une date dans l'index d'une série temporelle
def _locate_in_series(index: pd.Index, item: Any) -> int:
    """Return the position of the date ``item`` in the index of a time series.

    Raises:
        ValueError: If the date is absent, or matches several rows (partial
            date string).
    """
    # Recherche en pandas ; une date absente ou incomparable devient une ValueError
    try:
        loc = index.get_loc(_normalize_label(index, item))
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Time series index {item} not found in data") from None
    # Cas où une position unique est retournée
    if isinstance(loc, (int, np.integer)):
        return int(loc)
    # Cas où une tranche ou un masque est retourné (date partielle, ex. '2024-04')
    matches = np.arange(len(index))[loc]
    # Vérification que la date ne désigne qu'une seule ligne
    if len(matches) != 1:
        raise ValueError(f"Time series index {item} matches {len(matches)} rows, expected one")
    return int(matches[0])


# Construction de la clé d'index complète d'un tuple (entité..., date)
def _pair_key(index: pd.MultiIndex, item: tuple) -> tuple:
    """Return the full MultiIndex key of an ``(entity..., date)`` test index.

    The entity is given either level by level, ``('FR', 'industry', date)``,
    or as one tuple, ``(('FR', 'industry'), date)``; the date is converted to
    the type of the date level. With a single entity level, the first item
    is the whole entity, even when the keys of that level are tuples.

    Raises:
        ValueError: If the entity does not have one value per entity level.

    Examples:
        >>> idx = pd.MultiIndex.from_arrays([pd.Index([('FR', 'a')], tupleize_cols=False), [2024]])
        >>> _pair_key(idx, (('FR', 'a'), 2024))
        (('FR', 'a'), 2024)
    """
    # Nombre de niveaux d'entité (tous les niveaux sauf le dernier)
    n_entity_levels = index.nlevels - 1
    # Un seul niveau d'entité : le premier élément est l'entité entière (tuple compris)
    if n_entity_levels == 1 and len(item) == 2:
        entity = (item[0],)
    # Entité donnée d'un bloc (tuple en première position) ou niveau par niveau
    elif len(item) == 2 and isinstance(item[0], tuple):
        entity = item[0]
    else:
        entity = tuple(item[:-1])
    # Vérification d'une valeur par niveau d'entité
    if len(entity) != n_entity_levels:
        raise ValueError(
            f"Panel index {item} does not match the index: expected {n_entity_levels} entity "
            f"value(s) followed by a date"
        )
    return (*entity, _normalize_label(index.get_level_values(-1), item[-1]))


# Recherche des positions de test
def _resolve_test_positions(test_indices: Any, X: Any) -> Optional[np.ndarray]:
    """Resolve ``test_indices`` to row positions of ``X``.

    Integers are positions. Dates are looked up in the index of a time
    series, or in the date level (last level) of a panel, where a date selects
    the row of every entity holding it. ``(entity..., date)`` tuples select
    one row of a panel.

    Args:
        test_indices: Requested test indices (see :func:`_parse_test_indices`).
        X: Data whose rows the indices refer to.

    Returns:
        Integer array of positions, or None when no test index is given.

    Raises:
        ValueError: If an index is invalid, absent from the data, or of a kind
            the data cannot resolve (dates without a pandas index, tuples on a
            time series).

    Examples:
        >>> X = pd.Series(range(10), index=pd.date_range('2020-01-01', periods=10, freq='D'))
        >>> _resolve_test_positions(['2020-01-05', pd.Timestamp('2020-01-08')], X).tolist()
        [4, 7]
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], pd.date_range('2020-01-01', periods=5, freq='D')])
        >>> panel = pd.DataFrame({'value': range(10)}, index=idx)
        >>> _resolve_test_positions([('A', '2020-01-03'), ('B', '2020-01-04')], panel).tolist()
        [2, 8]
        >>> _resolve_test_positions(['2020-01-03', '2020-01-04'], panel).tolist()
        [2, 7, 3, 8]
    """
    # Parsing des indices de test
    parsed = _parse_test_indices(test_indices)
    # Aucun indice demandé : rien à résoudre
    if parsed is None:
        return None
    kind, items = parsed
    # Positions entières : utilisées telles quelles (bornes contrôlées par l'appelant)
    if kind == _POSITIONS:
        return np.asarray(items, dtype=int)

    # Vérification que X porte un index pandas, indispensable pour chercher des dates
    if not _is_pandas(X):
        raise ValueError("Cannot use string/date/tuple indices without pandas index")
    # Extraction de l'index
    index = X.index
    # Structure de panel : MultiIndex (entité..., date)
    is_panel = is_panel_data(X)

    # Initialisation de la liste des positions
    positions: list[int] = []
    # Cas des tuples (entité..., date) : une ligne précise d'un panel
    if kind == _PAIRS:
        # Un tuple n'a pas de sens sur une série temporelle
        if not is_panel:
            raise ValueError("Index type mismatch: time series data cannot use tuple indices")
        for item in items:
            # Recherche de la clé complète, la date convertie au type du niveau temporel
            key = _pair_key(index, item)
            if key not in index:
                raise ValueError(f"Panel index {item} not found in data")
            positions.append(int(index.get_loc(key)))
    # Cas d'une date sur un panel : la ligne de chaque entité qui la porte
    elif is_panel:
        # Extraction des dates
        dates = _time_level(X)
        for item in items:
            matching = np.flatnonzero(dates == _normalize_label(dates, item))
            if len(matching) == 0:
                raise ValueError(f"Date {item} not found in panel data")
            positions.extend(matching.tolist())
    # Cas d'une date sur une série temporelle : une seule ligne
    else:
        positions = [_locate_in_series(index, item) for item in items]
    return np.asarray(positions, dtype=int)


# =============================================================================
# Groupes (entités), tri et contrôle des données
# =============================================================================

# Fonction de pré-calcul des positions de chaque groupe
def _precompute_group_mappings(groups: Any) -> dict:
    """Map each group to the positions of its rows.

    Groups may be scalars or tuples (entities defined on several levels).

    Args:
        groups: Group label of each row.

    Returns:
        Dictionary ``{group: positions}``, groups in order of first
        appearance (sorted order on data sorted by entity).

    Examples:
        >>> mappings = _precompute_group_mappings(np.array(['A', 'A', 'B', 'B', 'A']))
        >>> mappings['A'].tolist(), mappings['B'].tolist()
        ([0, 1, 4], [2, 3])
    """
    # Codage des étiquettes (scalaires ou tuples) par ordre de première apparition
    codes, uniques = pd.factorize(_labels_array(groups), sort=False)
    # Positions des lignes de chaque groupe, calculées une seule fois
    return {group: np.flatnonzero(codes == code) for code, group in enumerate(uniques)}


# Vérification de l'unicité des dates (des couples entité-date pour un panel)
def _check_unique_index(index: pd.Index, is_panel: bool) -> None:
    """Raise a ValueError when a date (an entity-date pair for a panel) repeats."""
    # Une date répétée serait coupée par position : apprise et testée dans le même pli
    if index.has_duplicates:
        repeated = list(index[index.duplicated()].unique()[:5])
        what = "(entity, date) pairs" if is_panel else "dates"
        raise ValueError(
            f"X has duplicated {what}: {repeated}; each date must appear once"
            f"{' per entity' if is_panel else ''}"
        )


# Vérification de la cohérence entre groups et les niveaux d'entité du panel
def _check_groups(X: Any, groups: np.ndarray) -> None:
    """Check that ``groups`` repeats the entity of each row of a panel.

    Raises:
        ValueError: If an entity label is missing, if ``groups`` does not
            hold one label per row, or if it differs from the entity levels.
    """
    # Étiquette d'entité manquante, dans l'index ou dans groups
    entity_missing = any(X.index.get_level_values(level).isna().any() for level in get_entity_levels(X))
    if entity_missing or pd.isna(groups).any():
        raise ValueError("Entity labels must not be missing: the entity levels or groups hold a missing value")
    # Une étiquette par ligne
    if len(groups) != len(X):
        raise ValueError(f"groups must hold one label per row: got {len(groups)} labels for {len(X)} rows")
    # Même entité que celle de l'index, ligne par ligne (les positions de test en dépendent)
    if not np.array_equal(groups, _entity_labels(X)):
        raise ValueError("groups must match the entity levels of X (all index levels but the last), row by row")


# Fonction de vérification que l'ordre d'un index est chronologique
def _is_chronologically_sortable(index: pd.Index) -> bool:
    """Return whether sorting ``index`` puts its rows in chronological order.

    True for dates, periods, durations and numbers; False for labels
    (strings, objects), whose sorted order is alphabetical: ``'02/01/2024'``
    sorts before ``'12/01/2023'``.

    Examples:
        >>> _is_chronologically_sortable(pd.period_range('2024-01', periods=2, freq='M'))
        True
        >>> _is_chronologically_sortable(pd.Index(['2024-01-31', '2024-02-29']))
        False
    """
    # Dates, périodes, durées : ordre chronologique
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        return True
    # Nombres (années, numéros de période, positions) : ordre numérique ; booléens exclus
    return pd.api.types.is_numeric_dtype(index.dtype) and not pd.api.types.is_bool_dtype(index.dtype)


# Fonction de construction du message de refus d'un index de libellés non trié
def _unsortable_message(what: str, index: pd.Index) -> str:
    """Return the error message for unsorted labels that cannot be sorted by date."""
    return (
        f"{what} is not sorted and holds labels of dtype {index.dtype}, whose sorted order is not "
        f"chronological; convert it to dates (pd.to_datetime) or periods, or sort the rows by date"
    )


# Méthode auxiliaire de vérification et de tri des données par entité et par date
def _verify_and_sort_data(X: Any, groups: Any = None) -> tuple[Any, Optional[np.ndarray], np.ndarray]:
    """Validate the data and sort it by date (by entity, then date, for a panel).

    Args:
        X: Data to split. Without ``groups``, a pandas index of dates,
            periods or numbers is sorted when it is not increasing, an index
            of labels must already be increasing, and arrays are taken in
            their row order. With ``groups``, ``X`` must have a MultiIndex
            ``(entity..., date)``, sorted by the same rule within each entity.
        groups: Entity label of each row, or None for a time series.

    Returns:
        ``(X_sorted, groups_sorted, sort_indices)`` where ``sort_indices[k]``
        is the position, in ``X``, of the k-th row of ``X_sorted``.

    Raises:
        ValueError: If dates repeat, if an index of labels is not sorted, if
            a panel has no MultiIndex, or if ``groups`` is inconsistent with
            it.

    Examples:
        >>> dates = pd.date_range('2020-01-01', periods=3, freq='D')
        >>> X = pd.Series([0, 1, 2], index=dates[[2, 0, 1]])
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter('ignore')
        ...     X_sorted, _, sort_indices = _verify_and_sort_data(X)
        >>> sort_indices.tolist()
        [1, 2, 0]
        >>> X_sorted.tolist()
        [1, 2, 0]
    """
    # Cas d'une série temporelle (pas de groupes)
    if groups is None:
        # Tableau sans index : pris dans l'ordre de ses lignes
        if not _is_pandas(X):
            return X, None, np.arange(_num_samples(X))
        # Refus des dates répétées, avant tout tri
        _check_unique_index(X.index, is_panel=False)
        # Index déjà croissant : aucun tri, l'ordre des lignes est pris tel quel
        if X.index.is_monotonic_increasing:
            return X, None, np.arange(len(X))
        # Index de libellés non trié : ordre chronologique impossible à déduire
        if not _is_chronologically_sortable(X.index):
            raise ValueError(_unsortable_message("Time series index", X.index))
        # Index temporel ou numérique non trié : tri annoncé par un avertissement
        warnings.warn("Time series data is not sorted by date. Sorting automatically.")
        sort_indices = np.asarray(X.index.argsort(), dtype=int)
        return X.iloc[sort_indices], None, sort_indices

    # Cas d'un panel : MultiIndex (entité..., date) obligatoire
    if not _is_pandas(X) or not is_panel_data(X):
        raise ValueError("Panel data requires MultiIndex with at least 2 levels (group, date)")
    # Contrôles de groups et de l'unicité des couples (entité, date)
    groups = _labels_array(groups)
    _check_groups(X, groups)
    _check_unique_index(X.index, is_panel=True)

    # Entités contiguës et dates croissantes dans chaque entité : aucun tri
    entities_grouped = validate_entities_grouped(X)
    dates_sorted = validate_sorted_within_groups(X)
    if entities_grouped and dates_sorted:
        return X, groups, np.arange(len(X))

    # Niveau des dates fait de libellés
    date_sortable = _is_chronologically_sortable(_time_level(X))
    # Dates non triées dans une entité et sans ordre chronologique connu : refus
    if not dates_sorted and not date_sortable:
        raise ValueError(_unsortable_message("Panel date level", _time_level(X)))

    # Panel non trié : tri annoncé par un avertissement
    warnings.warn("Panel data is not sorted by group then by date. Sorting automatically.")
    # Clés de tri : niveaux d'entité, puis date si son ordre est chronologique ;
    # sinon, le tri stable regroupe les entités en gardant l'ordre de leurs lignes
    levels = pd.DataFrame({level: X.index.get_level_values(level) for level in range(X.index.nlevels)})
    keys = list(levels.columns) if date_sortable else list(levels.columns[:-1])
    sort_indices = levels.sort_values(keys, kind='mergesort').index.to_numpy(dtype=int)
    return X.iloc[sort_indices], groups[sort_indices], sort_indices


# Vérification que les positions de test sont dans les bornes des données
def _check_positions_in_range(positions: np.ndarray, n_samples: int) -> None:
    """Raise a ValueError when a test position is outside ``[0, n_samples)``."""
    # Position au-delà de la dernière ligne
    if np.any(positions >= n_samples):
        raise ValueError("test_indices contains indices beyond data length")
    # Position négative (pas de lecture depuis la fin, contrairement à numpy)
    if np.any(positions < 0):
        raise ValueError("test_indices contains negative indices")


# Fonction d'extraction des dates distinctes d'un panel
def _sorted_time_points(X: Any) -> pd.Index:
    """Return the distinct dates of a panel (last index level), in increasing order."""
    # Tri explicite : l'ordre d'apparition n'est pas chronologique si les entités sont décalées
    return _time_level(X).unique().sort_values()


# Fonction de sélection des lignes d'une entité à une date donnée
def _rows_at(X: Any, rows: np.ndarray, date: Any) -> np.ndarray:
    """Return the positions, among ``rows``, of the rows dated ``date``."""
    dates = _time_level(X)
    return rows[np.asarray(dates[rows] == _normalize_label(dates, date))]


# Fonction de construction des tests de chaque entité à partir d'une date de début
def _entity_runs_from(X: Any, mappings: dict, start_date: Any, test_size: int) -> list[int]:
    """Return, for each entity holding ``start_date``, ``test_size`` rows from it.

    Each run is cut at the end of its entity; an entity lacking the date is
    warned about and left out of the test (it may still train in the fold).
    """
    fold: list[int] = []
    for group, group_rows in mappings.items():
        # Recherche de la ligne de l'entité à la date de début
        starts = _rows_at(X, group_rows, start_date)
        # Entité sans cette date : signalée, puis absente du test de ce pli
        if len(starts) == 0:
            warnings.warn(
                f"Cannot find test period '{start_date}' for entity '{group}'; the entity is not tested in this fold"
            )
            continue
        # test_size lignes à partir de la date, sans dépasser la fin de l'entité
        start = int(starts[0])
        fold.extend(range(start, min(start + test_size, int(group_rows.max()) + 1)))
    return fold


# =============================================================================
# Validation croisée hors échantillon
# =============================================================================

# Classe de base pour la validation croisée out-of-sample
class OutOfSampleSplit(_BaseKFold):
    """Out-of-sample cross-validator for time series and panel data.

    Each test period is excluded from training, and training data comes
    strictly before it, as when forecasting in real time. Folds are cut by
    position on data sorted by date (by entity, then date, for a panel):

    - without ``test_indices``, ``n_splits`` adjacent test sets of
      ``test_size`` rows (default ``n_samples // (n_splits + 1)``, 1 for a
      panel) end at the last date, like sklearn's ``TimeSeriesSplit``;
    - with ``test_indices``, each index opens one fold of ``test_size`` rows
      (default 1), cut at the end of the data or of its entity.

    Training holds the rows before the test period minus ``gap`` rows (an
    expanding window), or the last ``max_train_size`` of them (a rolling
    window). For a panel (``groups`` given, MultiIndex ``(entity..., date)``),
    every rule applies entity by entity. A date selects the row of every
    entity; an entity that does not hold the test date of a fold is not
    tested in it and trains on its rows before that date (minus ``gap``,
    within the window), so that no training date of any entity reaches the
    test period. An integer position or an ``(entity..., date)`` tuple opens
    a fold that tests and trains one entity only.

    Args:
        n_splits: Number of folds when ``test_indices`` is None (at least 2).
        test_indices: Test starts: integer positions (never booleans), dates
            (strings, timestamps, datetimes, dates, numpy datetimes,
            periods, a DatetimeIndex) or, for a panel, ``(entity..., date)``
            tuples. A single value is one test start.
        max_train_size: Maximum number of training rows (per entity), or
            None for an expanding window.
        test_size: Number of rows (dates per entity) of each test set.
        gap: Number of rows (dates per entity) skipped between training and
            test.

    Raises:
        ValueError: At construction if ``n_splits`` is not an integer of at
            least 2; at ``split`` for an invalid parameter or data, see
            :meth:`split`.

    Examples:
        >>> X = pd.DataFrame({'v': range(12)}, index=pd.date_range('2024-01-31', periods=12, freq='ME'))
        >>> for train, test in OutOfSampleSplit(n_splits=3, gap=1).split(X):
        ...     print(train.tolist(), test.tolist())
        [0, 1] [3, 4, 5]
        [0, 1, 2, 3, 4] [6, 7, 8]
        [0, 1, 2, 3, 4, 5, 6, 7] [9, 10, 11]
        >>> splitter = OutOfSampleSplit(test_indices=['2024-06-30'], test_size=2, max_train_size=3)
        >>> [(train.tolist(), test.tolist()) for train, test in splitter.split(X)]
        [([2, 3, 4], [5, 6])]
        >>> splitter.get_n_splits()
        1
    """

    # Initialisation
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None, gap=0):
        """Initialize the cross-validator; parameters are stored unchanged.

        Args:
            n_splits: Number of folds when ``test_indices`` is None.
            test_indices: Explicit test starts, or None.
            max_train_size: Maximum number of training rows, or None.
            test_size: Number of rows of each test set, or None.
            gap: Number of rows between training and test.

        Raises:
            ValueError: If ``n_splits`` is not an integer of at least 2.

        Examples:
            >>> OutOfSampleSplit(n_splits=3, test_size=10, gap=2).gap
            2
        """
        # Initialisation du parent (contrôle de n_splits), sans mélange des lignes
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Stockage des paramètres tels quels (validés à l'appel de split)
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    # Méthode de calcul du nombre de plis
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of folds, computed from the parameters alone.

        Like sklearn's splitters, the count does not read the data: it is the
        number of test indices when some are given, ``n_splits`` otherwise.
        ``split`` produces exactly this number of folds on any data it accepts.

        Args:
            X: Ignored, present for sklearn compatibility.
            y: Ignored, present for sklearn compatibility.
            groups: Ignored, present for sklearn compatibility.

        Returns:
            Number of folds.

        Raises:
            ValueError: If ``test_indices`` holds an invalid index.

        Examples:
            >>> OutOfSampleSplit(n_splits=4).get_n_splits()
            4
            >>> OutOfSampleSplit(test_indices=['2024-03-31', '2024-06-30']).get_n_splits()
            2
        """
        # Parsing des indices de test
        parsed = _parse_test_indices(self.test_indices)
        # Un pli par indice de test demandé, sinon n_splits plis
        return self.n_splits if parsed is None else len(parsed[1])

    # Méthode de séparation des indices d'entrainement et de test
    def split(self, X, y=None, groups=None):
        """Generate out-of-sample ``(train, test)`` position arrays.

        Parameters, ``y`` and ``groups`` are checked when ``split`` is called;
        the data-dependent checks run when the first fold is requested.

        Args:
            X: Array-like, Series or DataFrame. A pandas index of dates,
                periods or numbers that is not increasing is sorted (with a
                warning); an index of labels (strings) must already be
                increasing; rows of an array are taken in their order.
            y: Target; when both are pandas objects, its index must equal
                that of ``X``.
            groups: Entity label of each row, for a panel with a MultiIndex
                ``(entity..., date)``; must repeat its entity levels.

        Returns:
            Generator of ``(train_positions, test_positions)`` integer arrays,
            positions of the rows of ``X`` as given.

        Raises:
            ValueError: If a parameter is invalid (``gap`` negative,
                ``test_size`` or ``max_train_size`` below 1), if ``X`` and
                ``y`` have different indexes, if dates repeat, if an index of
                labels is not sorted, if ``groups``
                is inconsistent with ``X``, if ``test_indices`` cannot be
                resolved, if there are too many folds for the data, or if a
                fold would have no training data.

        Examples:
            >>> X = pd.Series(range(6), index=pd.date_range('2024-01-31', periods=6, freq='ME'))
            >>> [(train.tolist(), test.tolist()) for train, test in OutOfSampleSplit(n_splits=2).split(X)]
            [([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])]
        """
        # Validation des paramètres et de la cible, dès l'appel (avant toute itération)
        _check_gap(self.gap)
        _check_positive_or_none('test_size', self.test_size)
        _check_positive_or_none('max_train_size', self.max_train_size)
        _check_target(X, y)
        return self._split(X, y, groups)

    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X, y=None, groups=None):
        """Yield the folds on sorted data, mapped back to the positions of ``X``."""
        # Conversion en objet indexable (listes comprises)
        (X,) = indexable(X)
        # Contrôle et tri des données ; sort_mapping ramène les positions triées aux positions de X
        X_sorted, groups_sorted, sort_mapping = _verify_and_sort_data(X, groups)
        parsed = _parse_test_indices(self.test_indices)
        # Une position ou un couple (entité, date) ne concerne que son entité ;
        # une date ou le découpage par défaut concernent toutes les entités
        every_group = parsed is None or parsed[0] == _DATES
        # Parcours des ensembles de test, chacun associé à son apprentissage
        for test_indices in self._iter_test_indices(X_sorted, y, groups_sorted):
            train_indices = self._get_train_indices(X_sorted, test_indices, groups_sorted, every_group)
            # Positions du jeu trié ramenées aux positions de X
            yield sort_mapping[train_indices], sort_mapping[test_indices]

    # Méthode auxiliaire d'extraction des indices d'entraînement correspondant aux indices de test
    def _get_train_indices(self, X, test_indices, groups, every_group=True):
        """Return the training positions of one fold (sorted data)."""
        # Série temporelle : une seule fenêtre d'apprentissage
        if groups is None:
            return self._get_timeseries_train_indices(test_indices)
        # Panel : une fenêtre par entité
        return self._get_group_train_indices(X, test_indices, groups, every_group)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les séries temporelles
    def _get_timeseries_train_indices(self, test_indices):
        """Return the rows before the test set, ``gap`` rows apart, within the window.

        Raises:
            ValueError: If no row is left for training.
        """
        # Fin (exclue) de l'apprentissage : début du test moins l'écart
        first_test = int(np.min(test_indices))
        train_end = first_test - self.gap
        # Aucune ligne avant le test : configuration refusée
        if train_end <= 0:
            raise ValueError(
                f"No training data before the test set starting at row {first_test} "
                f"(gap={self.gap}); reduce gap or the number of splits"
            )
        # Fenêtre extensible depuis la première ligne, ou glissante de max_train_size lignes
        train_start = 0 if self.max_train_size is None else max(0, train_end - self.max_train_size)
        return np.arange(train_start, train_end)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les données de panel
    def _get_group_train_indices(self, X, test_indices, groups, every_group=True):
        """Return, entity by entity, the rows before its test rows, within the window.

        Args:
            X: Sorted panel data, read for the dates of the rows.
            test_indices: Test positions of the fold (sorted data).
            groups: Entity label of each row (sorted data).
            every_group: If True, the fold tests common dates: an entity
                without test rows trains on its rows dated before the first
                test date of the fold. Otherwise the fold belongs to other
                entities and an entity without test rows is left out.

        Raises:
            ValueError: If a tested entity has no training row left.
        """
        # Première date de test du pli : borne de l'apprentissage des entités non testées
        dates = _time_level(X)
        fold_start = dates[test_indices].min() if every_group and len(test_indices) else None
        # Initialisation de la liste des indices d'entraînement
        train_indices: list[int] = []
        for group, group_rows in _precompute_group_mappings(groups).items():
            # Lignes de l'entité contiguës (données triées par entité puis date)
            group_start = int(group_rows.min())
            # Lignes de test de l'entité dans ce pli
            group_test = test_indices[np.isin(test_indices, group_rows)]
            # Entité sans ligne de test
            if len(group_test) == 0:
                # Pli propre à une autre entité : celle-ci n'y participe pas
                if fold_start is None:
                    continue
                # Pli sur des dates communes : apprentissage seul, sur les dates antérieures
                # à la première date de test du pli, moins l'écart
                train_end = group_start + int(np.sum(dates[group_rows] < fold_start)) - self.gap
                # Aucune date antérieure : l'entité ne participe pas au pli
                if train_end <= group_start:
                    continue
            else:
                train_end = int(group_test.min()) - self.gap
                # Aucune ligne de l'entité avant son test : configuration refusée
                if train_end <= group_start:
                    raise ValueError(
                        f"No training data for the group {group!r} before its test set "
                        f"(gap={self.gap}); reduce gap or the number of splits"
                    )
            # Fenêtre extensible depuis le début de l'entité, ou glissante
            train_start = group_start if self.max_train_size is None else max(
                group_start, train_end - self.max_train_size
            )
            train_indices.extend(range(train_start, train_end))
        return np.asarray(train_indices, dtype=int)

    # Méthode auxiliaire d'identification des indices de test
    def _iter_test_indices(self, X, y=None, groups=None):
        """Yield the test positions of each fold (sorted data)."""
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        # Conversion des indices de test demandés en positions
        positions = _resolve_test_positions(self.test_indices, X)
        # Aucun indice demandé : dernières portions des données
        if positions is None:
            yield from self._iter_default_test_indices(X, groups)
            return

        # Indices demandés : une ligne de test par défaut, positions dans les bornes
        test_size = self.test_size if self.test_size is not None else 1
        _check_positions_in_range(positions, n_samples)
        # Dates sur un panel : un pli par date, pour toutes les entités
        if groups is not None and _parse_test_indices(self.test_indices)[0] == _DATES:
            yield from self._iter_group_test_indices(X, positions, groups)
        # Positions, tuples, ou dates d'une série : un pli par début de test
        else:
            yield from self._iter_position_test_indices(positions, test_size, n_samples, groups)

    # Méthode auxiliaire d'identification des indices de test à partir de positions de début
    def _iter_position_test_indices(self, positions, test_size, n_samples, groups):
        """Yield, for each start in date order, ``test_size`` rows cut at the end of its block.

        The block is the whole data for a time series, the entity of the start
        for a panel.
        """
        # Fin (exclue) de chaque entité, pour couper les tests d'un panel
        entity_end = None
        if groups is not None:
            mappings = _precompute_group_mappings(groups)
            entity_end = {group: int(rows.max()) + 1 for group, rows in mappings.items()}
        # Parcours des débuts de test dans l'ordre des positions (donc des dates)
        for test_start in sorted(positions):
            block_end = n_samples if entity_end is None else entity_end[groups[test_start]]
            yield np.arange(test_start, min(test_start + test_size, block_end))

    # Méthode auxiliaire d'identification des indices de test au sein de chaque groupe
    def _iter_group_test_indices(self, X, resolved_test_positions, groups):
        """Yield, for each test date, ``test_size`` rows of every entity holding it.

        An entity lacking the date is warned about; the fold is still built
        for the other entities. The dates are known to exist: they were
        resolved before.
        """
        # Taille des périodes de test
        test_size = self.test_size if self.test_size is not None else 1
        # Mapping des groupes à leurs positions
        mappings = _precompute_group_mappings(groups)
        # Un pli par date demandée, dans l'ordre donné
        for test_time in _parse_test_indices(self.test_indices)[1]:
            yield np.asarray(_entity_runs_from(X, mappings, test_time, test_size), dtype=int)

    # Méthode auxiliaire d'identification des indices de test par défaut (dernières portions)
    def _iter_default_test_indices(self, X, groups):
        """Yield ``n_splits`` adjacent test sets ending at the last date.

        Raises:
            ValueError: If the data cannot hold ``n_splits`` test sets and
                some training data.
        """
        # Nombre d'observations
        n_samples = _num_samples(X)
        # Cas d'une série temporelle : découpage sur les lignes
        if groups is None:
            # n_splits + 1 segments, le premier réservé à l'apprentissage
            n_folds = self.n_splits + 1
            test_size = self.test_size if self.test_size is not None else n_samples // n_folds
            # Plus de segments que de lignes
            if n_folds > n_samples:
                raise ValueError(
                    f"Cannot have number of folds={n_folds} greater than the number of samples={n_samples}"
                )
            # Tests couvrant toutes les lignes : plus rien pour l'apprentissage
            if n_samples - test_size * self.n_splits <= 0:
                raise ValueError(
                    f"Too many splits={self.n_splits} for number of samples={n_samples} with test_size={test_size}"
                )
            # Débuts de test : les n_splits derniers blocs de test_size lignes
            for test_start in range(n_samples - self.n_splits * test_size, n_samples, test_size):
                yield np.arange(test_start, test_start + test_size)
            return

        # Cas d'un panel : découpage sur les dates, appliqué à chaque entité
        test_size = self.test_size if self.test_size is not None else 1
        time_points = _sorted_time_points(X)
        n_time_points = len(time_points)
        # Tests couvrant toutes les dates : plus rien pour l'apprentissage
        if n_time_points - test_size * self.n_splits <= 0:
            raise ValueError(
                f"Too many splits={self.n_splits} for number of time points={n_time_points} "
                f"with test_size={test_size}"
            )
        mappings = _precompute_group_mappings(groups)
        # Dates de début : les n_splits derniers blocs de test_size dates
        for i in range(self.n_splits):
            test_time = time_points[n_time_points - self.n_splits * test_size + i * test_size]
            yield np.asarray(_entity_runs_from(X, mappings, test_time, test_size), dtype=int)


# =============================================================================
# Validation croisée dans l'échantillon
# =============================================================================

# Classe de base pour la validation croisée in-sample
class InSampleSplit(_BaseKFold):
    """In-sample cross-validator for time series and panel data.

    It yields a single fold whose training set contains the test period, to
    evaluate a historical fit (backcast, revised nowcast). The test period is:

    - without ``test_indices``, the last ``test_size`` rows (default
      ``n_samples // (n_splits + 1)``), or the last ``test_size`` dates of
      a panel (default 1), for every entity holding them;
    - with one test index, ``test_size`` rows from it (default 1), cut at the
      end of the data or of its entity;
    - with several test indices, exactly those rows, one per index
      (``test_size`` is not used).

    Training holds the rows up to the last test row; with ``max_train_size``,
    only the last ``max(max_train_size, span of the test)`` of them, so that
    every test row stays a training row. For a panel (``groups`` given), every
    rule applies entity by entity; an entity holding none of the test dates is
    not tested and trains on its rows up to the last test date (the last
    ``max_train_size`` of them). Integer positions and ``(entity..., date)``
    tuples test and train their own entities only.

    Args:
        n_splits: Only sets the default ``test_size`` of a time series; the
            split always has one fold (at least 2, as for sklearn splitters).
        test_indices: Test rows: integer positions (never booleans), dates or,
            for a panel, ``(entity..., date)`` tuples (see ``OutOfSampleSplit``).
        max_train_size: Maximum number of training rows (per entity), or
            None for all the history.
        test_size: Number of rows (dates per entity) of the test period.

    Raises:
        ValueError: At construction if ``n_splits`` is not an integer of at
            least 2; at ``split`` for an invalid parameter or data.

    Examples:
        >>> X = pd.Series(range(12), index=pd.date_range('2024-01-31', periods=12, freq='ME'))
        >>> [(train.tolist(), test.tolist()) for train, test in InSampleSplit(test_size=3, max_train_size=5).split(X)]
        [([7, 8, 9, 10, 11], [9, 10, 11])]
        >>> splitter = InSampleSplit(test_indices=['2024-02-29', '2024-05-31'])
        >>> [(train.tolist(), test.tolist()) for train, test in splitter.split(X)]
        [([0, 1, 2, 3, 4], [1, 4])]
        >>> splitter.get_n_splits()
        1
    """

    # Initialisation de la classe
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None):
        """Initialize the cross-validator; parameters are stored unchanged.

        Args:
            n_splits: Sets the default ``test_size`` of a time series.
            test_indices: Explicit test rows, or None.
            max_train_size: Maximum number of training rows, or None.
            test_size: Number of rows of the test period, or None.

        Raises:
            ValueError: If ``n_splits`` is not an integer of at least 2.

        Examples:
            >>> InSampleSplit(test_size=5, max_train_size=100).max_train_size
            100
        """
        # Initialisation du parent (contrôle de n_splits), sans mélange des lignes
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Stockage des paramètres tels quels (validés à l'appel de split)
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size

    # Méthode de calcul du nombre de plis
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return 1: an in-sample split has a single fold.

        Args:
            X: Ignored, present for sklearn compatibility.
            y: Ignored, present for sklearn compatibility.
            groups: Ignored, present for sklearn compatibility.

        Returns:
            Number of folds, always 1.

        Examples:
            >>> InSampleSplit(n_splits=5).get_n_splits()
            1
        """
        return 1

    # Méthode principale de séparation des données
    def split(self, X, y=None, groups=None):
        """Generate the in-sample ``(train, test)`` position arrays.

        Args:
            X: Array-like, Series or DataFrame (see ``OutOfSampleSplit.split``).
            y: Target; when both are pandas objects, its index must equal
                that of ``X``.
            groups: Entity label of each row, for a panel.

        Returns:
            Generator of one ``(train_positions, test_positions)`` pair.

        Raises:
            ValueError: If a parameter is invalid (``test_size`` or
                ``max_train_size`` below 1), if ``X`` and ``y`` have different
                indexes, if the data is invalid (see ``OutOfSampleSplit``), or
                if the test period would be empty.

        Examples:
            >>> X = pd.Series(range(6), index=pd.date_range('2024-01-31', periods=6, freq='ME'))
            >>> [(train.tolist(), test.tolist()) for train, test in InSampleSplit(test_size=2).split(X)]
            [([0, 1, 2, 3, 4, 5], [4, 5])]
        """
        # Validation des paramètres et de la cible, dès l'appel (avant toute itération)
        _check_positive_or_none('test_size', self.test_size)
        _check_positive_or_none('max_train_size', self.max_train_size)
        _check_target(X, y)
        return self._split(X, y, groups)

    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X, y=None, groups=None):
        """Yield the fold on sorted data, mapped back to the positions of ``X``."""
        # Conversion en objet indexable (listes comprises)
        (X,) = indexable(X)
        # Contrôle et tri des données ; sort_mapping ramène les positions triées aux positions de X
        X_sorted, groups_sorted, sort_mapping = _verify_and_sort_data(X, groups)
        parsed = _parse_test_indices(self.test_indices)
        # Une position ou un couple (entité, date) ne concerne que son entité
        every_group = parsed is None or parsed[0] == _DATES
        # Un seul pli : l'ensemble de test et son apprentissage
        for test_indices in self._iter_test_indices(X_sorted, y, groups_sorted):
            train_indices = self._get_train_indices(X_sorted, test_indices, groups_sorted, every_group)
            # Positions du jeu trié ramenées aux positions de X
            yield sort_mapping[train_indices], sort_mapping[test_indices]

    # Méthode auxiliaire d'extraction des indices d'entraînement incluant la période de test
    def _get_train_indices(self, X, test_indices, groups, every_group=True):
        """Return the training positions of the fold (sorted data)."""
        # Série temporelle : une seule fenêtre d'apprentissage
        if groups is None:
            return self._get_timeseries_train_indices(test_indices)
        # Panel : une fenêtre par entité
        return self._get_group_train_indices(X, test_indices, groups, every_group)

    # Méthode auxiliaire de calcul de la fenêtre d'apprentissage d'un bloc
    def _window(self, test_indices, block_start):
        """Return the training range ending at the last test row, starting no earlier than ``block_start``."""
        # La période de test étant incluse, l'apprentissage s'arrête à la dernière ligne de test
        first_test, last_test = int(np.min(test_indices)), int(np.max(test_indices))
        train_end = last_test + 1
        # Sans limite : tout l'historique du bloc
        if self.max_train_size is None:
            return range(block_start, train_end)
        # Avec limite : la fenêtre couvre au moins toute l'étendue du test
        size = max(self.max_train_size, last_test - first_test + 1)
        return range(max(block_start, train_end - size), train_end)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les séries temporelles
    def _get_timeseries_train_indices(self, test_indices):
        """Return the rows up to the last test row, within the window."""
        return np.asarray(self._window(test_indices, 0), dtype=int)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les données de panel
    def _get_group_train_indices(self, X, test_indices, groups, every_group=True):
        """Return, entity by entity, the rows up to its last test row, within the window.

        Args:
            X: Sorted panel data, read for the dates of the rows.
            test_indices: Test positions of the fold (sorted data).
            groups: Entity label of each row (sorted data).
            every_group: If True, the fold tests common dates: an entity
                without test rows trains on its rows dated up to the last test
                date (the last ``max_train_size`` of them). Otherwise the
                fold belongs to other entities and such an entity is left out.
        """
        # Dernière date de test du pli : borne de l'apprentissage des entités non testées
        dates = _time_level(X)
        fold_end = dates[test_indices].max() if every_group and len(test_indices) else None
        # Initialisation de la liste des indices d'entraînement
        train_indices: list[int] = []
        for group, group_rows in _precompute_group_mappings(groups).items():
            group_start = int(group_rows.min())
            # Lignes de test de l'entité
            group_test = test_indices[np.isin(test_indices, group_rows)]
            # Entité testée : fenêtre qui ne remonte pas avant son début
            if len(group_test) > 0:
                train_indices.extend(self._window(group_test, group_start))
                continue
            # Test propre à d'autres entités : celle-ci n'y participe pas
            if fold_end is None:
                continue
            # Test sur des dates communes : apprentissage seul, jusqu'à la dernière date de test
            train_end = group_start + int(np.sum(dates[group_rows] <= fold_end))
            train_start = group_start if self.max_train_size is None else max(
                group_start, train_end - self.max_train_size
            )
            train_indices.extend(range(train_start, train_end))
        return np.asarray(train_indices, dtype=int)

    # Méthode auxiliaire de génération des indices de test pour la validation in-sample
    def _iter_test_indices(self, X, y=None, groups=None):
        """Yield the test positions of the single fold (sorted data).

        Raises:
            ValueError: If test indices are out of range, or if the default
                test period is empty.
        """
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        # Conversion des indices de test demandés en positions
        positions = _resolve_test_positions(self.test_indices, X)
        # Aucun indice demandé : dernière portion des données
        if positions is None:
            # Série temporelle : les test_size dernières lignes
            if groups is None:
                yield from self._iter_default_timeseries_test_indices(X, n_samples)
            # Panel : les test_size dernières dates de chaque entité
            else:
                yield from self._iter_default_group_test_indices(X, groups)
            return

        # Indices demandés : une ligne de test par défaut, positions dans les bornes
        test_size = self.test_size if self.test_size is not None else 1
        _check_positions_in_range(positions, n_samples)
        # Dates sur un panel : recherche de chaque date dans chaque entité
        if groups is not None and _parse_test_indices(self.test_indices)[0] == _DATES:
            yield from self._iter_group_test_indices(X, positions, groups)
            return
        # Plusieurs indices : une ligne par indice, test_size non utilisé
        if len(positions) > 1:
            yield np.unique(positions)
            return
        # Un seul indice : test_size lignes, sans dépasser la fin des données ou de l'entité
        block_end = n_samples
        if groups is not None:
            block_end = int(_precompute_group_mappings(groups)[groups[positions[0]]].max()) + 1
        yield np.arange(positions[0], min(positions[0] + test_size, block_end))

    # Méthode auxiliaire d'identification des indices de test au sein de chaque groupe
    def _iter_group_test_indices(self, X, resolved_test_positions, groups):
        """Yield the test rows of a panel for test dates.

        One date: ``test_size`` rows of each entity from it. Several dates:
        the row of each entity at each date. Missing dates are warned about.
        """
        # Taille de l'échantillon de test
        test_size = self.test_size if self.test_size is not None else 1
        test_times = _parse_test_indices(self.test_indices)[1]
        # Une date : test_size lignes ; plusieurs dates : une ligne par date
        length = test_size if len(test_times) == 1 else 1
        # Assocation des entités à leur position
        mappings = _precompute_group_mappings(groups)
        fold: list[int] = []
        for test_time in test_times:
            fold.extend(_entity_runs_from(X, mappings, test_time, length))
        # Tri et suppression des doublons (dates données deux fois)
        yield np.unique(np.asarray(fold, dtype=int))

    # Méthode auxiliaire d'identification des indices de test par défaut pour les séries temporelles
    def _iter_default_timeseries_test_indices(self, X, n_samples):
        """Yield the last ``test_size`` rows.

        Raises:
            ValueError: If the data is empty or the test period has no row.
        """
        # Jeu vide : aucun test possible
        if n_samples == 0:
            raise ValueError("Cannot split an empty dataset: number of samples=0")
        # Taille de test par défaut : une part des données fixée par n_splits
        test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)
        # Trop peu de lignes pour n_splits : taille de test nulle, refusée
        if test_size < 1:
            raise ValueError(
                f"Empty test period: test_size={test_size} for number of samples={n_samples} "
                f"with n_splits={self.n_splits}; set test_size or reduce n_splits"
            )
        # Les test_size dernières lignes (toutes si test_size dépasse les données)
        yield np.arange(max(0, n_samples - test_size), n_samples)

    # Méthode auxiliaire d'identification des indices de test par défaut pour les données de panel
    def _iter_default_group_test_indices(self, X, groups):
        """Yield the rows of each entity at the last ``test_size`` dates of the panel.

        An entity lacking one of these dates is warned about.

        Raises:
            ValueError: If the panel is empty.
        """
        # Panel vide : aucune date à tester (même refus que pour une série vide)
        if _num_samples(X) == 0:
            raise ValueError("Cannot split an empty dataset: number of samples=0")
        # Taille de l'échantillon de test
        test_size = self.test_size if self.test_size is not None else 1
        # Les test_size dernières dates du panel, dans l'ordre chronologique
        test_times = _sorted_time_points(X)[-test_size:]
        # Association de chaque groupe à ses positions
        mappings = _precompute_group_mappings(groups)
        fold: list[int] = []
        # Une ligne par entité et par date de test
        for test_time in test_times:
            fold.extend(_entity_runs_from(X, mappings, test_time, 1))
        yield np.unique(np.asarray(fold, dtype=int))
