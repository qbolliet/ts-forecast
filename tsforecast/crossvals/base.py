"""Base time-aware cross-validators: ``OutOfSampleSplit`` and ``InSampleSplit``.

``OutOfSampleSplit`` trains strictly before each test period (optional ``gap``
and rolling window); ``InSampleSplit`` yields one fold whose training set
contains the test period. Both split a time series, or panel data when
``groups`` are given, and share private helpers that validate the parameters
and the data, sort unsorted rows and resolve the requested ``test_indices``.
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

# Import des utilitaires de validation
from ..utils.validation import validate_entities_grouped, validate_sorted_within_groups

# Natures possibles des indices de test demandés
_POSITIONS = 'positions'
_DATES = 'dates'
_PAIRS = 'pairs'

# Types reconnus comme des dates (Timestamp et datetime dérivent de datetime.date)
_DATE_TYPES = (str, datetime.date, np.datetime64, pd.Period)

# Conteneurs lus élément par élément ; toute autre valeur est un indice unique
_CONTAINER_TYPES = (list, tuple, np.ndarray, pd.Index, pd.Series)


def _is_pandas(X: Any) -> bool:
    """Return whether ``X`` is a pandas Series or DataFrame."""
    return isinstance(X, (pd.Series, pd.DataFrame))


def _check_positive_or_none(name: str, value: Any) -> None:
    """Raise a ValueError unless ``value`` is None or an integer of at least 1."""
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")


def _check_gap(gap: Any) -> None:
    """Raise a ValueError unless ``gap`` is a non-negative integer."""
    if isinstance(gap, (bool, np.bool_)) or not isinstance(gap, Integral) or gap < 0:
        raise ValueError(f"gap must be a non-negative integer, got {gap!r}")


def _check_target(X: Any, y: Any) -> None:
    """Raise a ValueError when X and y are both pandas objects with different indexes."""
    if y is not None and _is_pandas(X) and _is_pandas(y) and not X.index.equals(y.index):
        raise ValueError("'X' and 'y' should have the same indexes")


def _kind_of(item: Any) -> str:
    """Return the nature of one requested test index.

    Raises:
        ValueError: If the item is a boolean or neither an integer position, a
            date nor an ``(entity, date)`` tuple.
    """
    # Un booléen est un entier pour Python : refus explicite avant le test d'entier
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(
            f"test_indices must hold integer positions, dates or (entity, date) tuples, "
            f"got the boolean {item!r}"
        )
    if isinstance(item, Integral):
        return _POSITIONS
    if isinstance(item, tuple):
        return _PAIRS
    if isinstance(item, _DATE_TYPES):
        return _DATES
    raise ValueError(
        f"test_indices must hold integer positions, dates or (entity, date) tuples, "
        f"got {item!r} of type {type(item).__name__}"
    )


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
    if test_indices is None:
        return None
    items = list(test_indices) if isinstance(test_indices, _CONTAINER_TYPES) else [test_indices]
    if not items:
        return None
    kinds = {_kind_of(item) for item in items}
    if len(kinds) > 1:
        raise ValueError(
            f"test_indices must not mix integer positions, dates and (entity, date) tuples, "
            f"got {sorted(kinds)}"
        )
    return kinds.pop(), items


def _normalize_label(index: pd.Index, label: Any) -> Any:
    """Convert a date to the type of the index it is looked up in.

    Strings are left as they are (pandas parses them against a datetime
    index); other dates become a ``Timestamp`` (localized to the index time
    zone) on a ``DatetimeIndex`` and a ``Period`` on a ``PeriodIndex``.
    """
    if isinstance(label, str):
        return label
    if isinstance(index, pd.DatetimeIndex):
        stamp = pd.Timestamp(label)
        if index.tz is not None and stamp.tz is None:
            stamp = stamp.tz_localize(index.tz)
        return stamp
    if isinstance(index, pd.PeriodIndex) and not isinstance(label, pd.Period):
        return pd.Period(label, freq=index.freq)
    return label


def _locate_in_series(index: pd.Index, item: Any) -> int:
    """Return the position of the date ``item`` in the index of a time series.

    Raises:
        ValueError: If the date is absent, or matches several rows (partial
            date string).
    """
    try:
        loc = index.get_loc(_normalize_label(index, item))
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Time series index {item} not found in data") from None
    if isinstance(loc, (int, np.integer)):
        return int(loc)
    matches = np.arange(len(index))[loc]
    if len(matches) != 1:
        raise ValueError(f"Time series index {item} matches {len(matches)} rows, expected one")
    return int(matches[0])


def _resolve_test_positions(test_indices: Any, X: Any) -> Optional[np.ndarray]:
    """Resolve ``test_indices`` to row positions of ``X``.

    Integers are positions. Dates are looked up in the index of a time
    series, or in the date level (second level) of a panel, where a date
    selects the row of every entity holding it. ``(entity, date)`` tuples
    select one row of a panel.

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
    parsed = _parse_test_indices(test_indices)
    if parsed is None:
        return None
    kind, items = parsed
    if kind == _POSITIONS:
        return np.asarray(items, dtype=int)

    # Vérification que X porte un index pandas
    if not _is_pandas(X):
        raise ValueError("Cannot use string/date/tuple indices without pandas index")
    index = X.index
    is_panel = isinstance(index, pd.MultiIndex)

    positions: list[int] = []
    if kind == _PAIRS:
        if not is_panel:
            raise ValueError("Index type mismatch: time series data cannot use tuple indices")
        dates = index.get_level_values(1)
        for item in items:
            # Recherche du couple (entité, date), la date convertie au type du niveau
            key = (item[0], _normalize_label(dates, item[1])) if len(item) == 2 else item
            if key not in index:
                raise ValueError(f"Panel index {item} not found in data")
            positions.append(int(index.get_loc(key)))
    elif is_panel:
        dates = index.get_level_values(1)
        for item in items:
            # Une date seule sélectionne la ligne de chaque entité qui la porte
            matching = np.flatnonzero(dates == _normalize_label(dates, item))
            if len(matching) == 0:
                raise ValueError(f"Date {item} not found in panel data")
            positions.extend(matching.tolist())
    else:
        positions = [_locate_in_series(index, item) for item in items]
    return np.asarray(positions, dtype=int)


def _precompute_group_mappings(groups: Any) -> dict:
    """Map each group to the positions of its rows.

    Args:
        groups: Group label of each row.

    Returns:
        Dictionary ``{group: positions}``, groups in sorted order.

    Examples:
        >>> mappings = _precompute_group_mappings(np.array(['A', 'A', 'B', 'B', 'A']))
        >>> mappings['A'].tolist(), mappings['B'].tolist()
        ([0, 1, 4], [2, 3])
    """
    groups = np.asarray(groups)
    return {group: np.flatnonzero(groups == group) for group in np.unique(groups)}


def _check_unique_index(index: pd.Index, is_panel: bool) -> None:
    """Raise a ValueError when a date (an entity-date pair for a panel) repeats."""
    if index.has_duplicates:
        repeated = list(index[index.duplicated()].unique()[:5])
        what = "(entity, date) pairs" if is_panel else "dates"
        raise ValueError(
            f"X has duplicated {what}: {repeated}; each date must appear once"
            f"{' per entity' if is_panel else ''}"
        )


def _check_groups(X: Any, groups: np.ndarray) -> None:
    """Check that ``groups`` repeats the entity level of a panel, row by row.

    Raises:
        ValueError: If an entity label is missing, if ``groups`` does not
            hold one label per row, or if it differs from the first index
            level.
    """
    entities = X.index.get_level_values(0)
    if entities.isna().any() or pd.isna(groups).any():
        raise ValueError("Entity labels must not be missing: the entity level or groups holds a missing value")
    if len(groups) != len(X):
        raise ValueError(f"groups must hold one label per row: got {len(groups)} labels for {len(X)} rows")
    if not np.array_equal(groups.astype(object), entities.to_numpy(dtype=object)):
        raise ValueError("groups must match the first index level (entity) of X, row by row")


def _verify_and_sort_data(X: Any, groups: Any = None) -> tuple[Any, Optional[np.ndarray], np.ndarray]:
    """Validate the data and sort it by date (by entity, then date, for a panel).

    Args:
        X: Data to split. Without ``groups``, a pandas index is sorted when it
            is not increasing, and arrays are taken in their row order. With
            ``groups``, ``X`` must have a MultiIndex ``(entity, date, ...)``.
        groups: Entity label of each row, or None for a time series.

    Returns:
        ``(X_sorted, groups_sorted, sort_indices)`` where ``sort_indices[k]``
        is the position, in ``X``, of the k-th row of ``X_sorted``.

    Raises:
        ValueError: If dates repeat, if a panel has no MultiIndex, or if
            ``groups`` is inconsistent with it.

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
    if groups is None:
        # Série temporelle : un tableau est pris dans l'ordre de ses lignes
        if not _is_pandas(X):
            return X, None, np.arange(_num_samples(X))
        _check_unique_index(X.index, is_panel=False)
        if X.index.is_monotonic_increasing:
            return X, None, np.arange(len(X))
        warnings.warn("Time series data is not sorted by date. Sorting automatically.")
        sort_indices = np.asarray(X.index.argsort(), dtype=int)
        return X.iloc[sort_indices], None, sort_indices

    # Données de panel : MultiIndex (entité, date) obligatoire
    if not _is_pandas(X) or X.index.nlevels < 2:
        raise ValueError("Panel data requires MultiIndex with at least 2 levels (group, date)")
    groups = np.asarray(groups)
    _check_groups(X, groups)
    _check_unique_index(X.index, is_panel=True)

    # Vérification du regroupement des entités et du tri des dates dans chaque entité
    if validate_entities_grouped(X) and validate_sorted_within_groups(X):
        return X, groups, np.arange(len(X))

    warnings.warn("Panel data is not sorted by group then by date. Sorting automatically.")
    # Tri stable sur tous les niveaux, entité d'abord
    levels = pd.DataFrame({level: X.index.get_level_values(level) for level in range(X.index.nlevels)})
    sort_indices = levels.sort_values(list(levels.columns), kind='mergesort').index.to_numpy(dtype=int)
    return X.iloc[sort_indices], groups[sort_indices], sort_indices


def _check_positions_in_range(positions: np.ndarray, n_samples: int) -> None:
    """Raise a ValueError when a test position is outside ``[0, n_samples)``."""
    if np.any(positions >= n_samples):
        raise ValueError("test_indices contains indices beyond data length")
    if np.any(positions < 0):
        raise ValueError("test_indices contains negative indices")


def _sorted_time_points(X: Any) -> pd.Index:
    """Return the distinct dates of a panel (second index level), in increasing order."""
    return X.index.get_level_values(1).unique().sort_values()


def _rows_at(X: Any, rows: np.ndarray, date: Any) -> np.ndarray:
    """Return the positions, among ``rows``, of the rows dated ``date``."""
    dates = X.index.get_level_values(1)
    return rows[np.asarray(dates[rows] == _normalize_label(dates, date))]


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
    window). For a panel (``groups`` given), every rule applies entity by
    entity. A date selects the row of every entity; an integer position or an
    ``(entity, date)`` tuple opens a fold that tests and trains one entity
    only.

    Args:
        n_splits: Number of folds when ``test_indices`` is None (at least 2).
        test_indices: Test starts: integer positions (never booleans), dates
            (strings, timestamps, datetimes, dates, numpy datetimes,
            periods, a DatetimeIndex) or, for a panel, ``(entity, date)``
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
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

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
        parsed = _parse_test_indices(self.test_indices)
        return self.n_splits if parsed is None else len(parsed[1])

    # Méthode de séparation des indices d'entrainement et de test
    def split(self, X, y=None, groups=None):
        """Generate out-of-sample ``(train, test)`` position arrays.

        Parameters, ``y`` and ``groups`` are checked when ``split`` is called;
        the data-dependent checks run when the first fold is requested.

        Args:
            X: Array-like, Series or DataFrame. A pandas index that is not
                increasing is sorted (with a warning); rows of an array are
                taken in their order.
            y: Target; when both are pandas objects, its index must equal
                that of ``X``.
            groups: Entity label of each row, for a panel with a MultiIndex
                ``(entity, date)``; must repeat its first level.

        Returns:
            Generator of ``(train_positions, test_positions)`` integer arrays,
            positions of the rows of ``X`` as given.

        Raises:
            ValueError: If a parameter is invalid (``gap`` negative,
                ``test_size`` or ``max_train_size`` below 1), if ``X`` and
                ``y`` have different indexes, if dates repeat, if ``groups``
                is inconsistent with ``X``, if ``test_indices`` cannot be
                resolved, if there are too many folds for the data, or if a
                fold would have no training data.

        Examples:
            >>> X = pd.Series(range(6), index=pd.date_range('2024-01-31', periods=6, freq='ME'))
            >>> [(train.tolist(), test.tolist()) for train, test in OutOfSampleSplit(n_splits=2).split(X)]
            [([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])]
        """
        # Validation des paramètres et de la cible
        _check_gap(self.gap)
        _check_positive_or_none('test_size', self.test_size)
        _check_positive_or_none('max_train_size', self.max_train_size)
        _check_target(X, y)
        return self._split(X, y, groups)

    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X, y=None, groups=None):
        """Yield the folds on sorted data, mapped back to the positions of ``X``."""
        (X,) = indexable(X)
        X_sorted, groups_sorted, sort_mapping = _verify_and_sort_data(X, groups)
        parsed = _parse_test_indices(self.test_indices)
        # Une position ou un couple (entité, date) ne concerne que son entité
        every_group = parsed is None or parsed[0] == _DATES
        for test_indices in self._iter_test_indices(X_sorted, y, groups_sorted):
            train_indices = self._get_train_indices(X_sorted, test_indices, groups_sorted, every_group)
            # Positions du jeu trié ramenées aux positions de X
            yield sort_mapping[train_indices], sort_mapping[test_indices]

    # Méthode auxiliaire d'extraction des indices d'entraînement correspondant aux indices de test
    def _get_train_indices(self, X, test_indices, groups, every_group=True):
        """Return the training positions of one fold (sorted data)."""
        if groups is None:
            return self._get_timeseries_train_indices(test_indices)
        return self._get_group_train_indices(test_indices, groups, every_group)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les séries temporelles
    def _get_timeseries_train_indices(self, test_indices):
        """Return the rows before the test set, ``gap`` rows apart, within the window.

        Raises:
            ValueError: If no row is left for training.
        """
        first_test = int(np.min(test_indices))
        train_end = first_test - self.gap
        if train_end <= 0:
            raise ValueError(
                f"No training data before the test set starting at row {first_test} "
                f"(gap={self.gap}); reduce gap or the number of splits"
            )
        train_start = 0 if self.max_train_size is None else max(0, train_end - self.max_train_size)
        return np.arange(train_start, train_end)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les données de panel
    def _get_group_train_indices(self, test_indices, groups, every_group=True):
        """Return, entity by entity, the rows before its test rows, within the window.

        Args:
            test_indices: Test positions of the fold (sorted data).
            groups: Entity label of each row (sorted data).
            every_group: If True, every entity must have test rows in the
                fold; otherwise entities without test rows are left out.

        Raises:
            ValueError: If an entity has no test row while ``every_group`` is
                True, or if an entity has no training row left.
        """
        train_indices: list[int] = []
        for group, group_rows in _precompute_group_mappings(groups).items():
            group_test = test_indices[np.isin(test_indices, group_rows)]
            if len(group_test) == 0:
                if every_group:
                    raise ValueError(f"Cannot find test indices for the group : {group}")
                continue
            # Lignes de l'entité contiguës (données triées par entité puis date)
            group_start = int(group_rows.min())
            train_end = int(group_test.min()) - self.gap
            if train_end <= group_start:
                raise ValueError(
                    f"No training data for the group {group!r} before its test set "
                    f"(gap={self.gap}); reduce gap or the number of splits"
                )
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
        positions = _resolve_test_positions(self.test_indices, X)
        if positions is None:
            yield from self._iter_default_test_indices(X, groups)
            return

        test_size = self.test_size if self.test_size is not None else 1
        _check_positions_in_range(positions, n_samples)
        if groups is not None and _parse_test_indices(self.test_indices)[0] == _DATES:
            yield from self._iter_group_test_indices(X, positions, groups)
        else:
            yield from self._iter_position_test_indices(positions, test_size, n_samples, groups)

    def _iter_position_test_indices(self, positions, test_size, n_samples, groups):
        """Yield, for each start in date order, ``test_size`` rows cut at the end of its block.

        The block is the whole data for a time series, the entity of the start
        for a panel.
        """
        entity_end = None
        if groups is not None:
            mappings = _precompute_group_mappings(groups)
            entity_end = {group: int(rows.max()) + 1 for group, rows in mappings.items()}
        for test_start in sorted(positions):
            block_end = n_samples if entity_end is None else entity_end[groups[test_start]]
            yield np.arange(test_start, min(test_start + test_size, block_end))

    # Méthode auxiliaire d'identification des indices de test au sein de chaque groupe
    def _iter_group_test_indices(self, X, resolved_test_positions, groups):
        """Yield, for each test date, ``test_size`` rows of every entity holding it.

        An entity lacking the date is warned about; the fold is still built
        for the other entities.
        """
        test_size = self.test_size if self.test_size is not None else 1
        mappings = _precompute_group_mappings(groups)
        for test_time in _parse_test_indices(self.test_indices)[1]:
            fold: list[int] = []
            for group, group_rows in mappings.items():
                starts = _rows_at(X, group_rows, test_time)
                if len(starts) == 0:
                    warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                    continue
                start = int(starts[0])
                fold.extend(range(start, min(start + test_size, int(group_rows.max()) + 1)))
            yield np.asarray(fold, dtype=int)

    # Méthode auxiliaire d'identification des indices de test par défaut (dernières portions)
    def _iter_default_test_indices(self, X, groups):
        """Yield ``n_splits`` adjacent test sets ending at the last date.

        Raises:
            ValueError: If the data cannot hold ``n_splits`` test sets and
                some training data.
        """
        n_samples = _num_samples(X)
        if groups is None:
            # Série temporelle : n_splits + 1 segments, le premier réservé à l'apprentissage
            n_folds = self.n_splits + 1
            test_size = self.test_size if self.test_size is not None else n_samples // n_folds
            if n_folds > n_samples:
                raise ValueError(
                    f"Cannot have number of folds={n_folds} greater than the number of samples={n_samples}"
                )
            if n_samples - test_size * self.n_splits <= 0:
                raise ValueError(
                    f"Too many splits={self.n_splits} for number of samples={n_samples} with test_size={test_size}"
                )
            for test_start in range(n_samples - self.n_splits * test_size, n_samples, test_size):
                yield np.arange(test_start, test_start + test_size)
            return

        # Panel : découpage sur les dates, appliqué à chaque entité
        test_size = self.test_size if self.test_size is not None else 1
        time_points = _sorted_time_points(X)
        n_time_points = len(time_points)
        if n_time_points - test_size * self.n_splits <= 0:
            raise ValueError(
                f"Too many splits={self.n_splits} for number of time points={n_time_points} "
                f"with test_size={test_size}"
            )
        mappings = _precompute_group_mappings(groups)
        for i in range(self.n_splits):
            test_time = time_points[n_time_points - self.n_splits * test_size + i * test_size]
            fold: list[int] = []
            for group, group_rows in mappings.items():
                starts = _rows_at(X, group_rows, test_time)
                if len(starts) == 0:
                    warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                    continue
                start = int(starts[0])
                fold.extend(range(start, min(start + test_size, int(group_rows.max()) + 1)))
            yield np.asarray(fold, dtype=int)


# Classe de base pour la validation croisée in-sample
class InSampleSplit(_BaseKFold):
    """In-sample cross-validator for time series and panel data.

    It yields a single fold whose training set contains the test period, to
    evaluate a historical fit (backcast, revised nowcast). The test period is:

    - without ``test_indices``, the last ``test_size`` rows (default
      ``n_samples // (n_splits + 1)``), or the last ``test_size`` dates of
      each entity of a panel (default 1);
    - with one test index, ``test_size`` rows from it (default 1), cut at the
      end of the data or of its entity;
    - with several test indices, exactly those rows, one per index
      (``test_size`` is not used).

    Training holds the rows up to the last test row; with ``max_train_size``,
    only the last ``max(max_train_size, span of the test)`` of them, so that
    every test row stays a training row. For a panel (``groups`` given), every
    rule applies entity by entity; a single integer position or ``(entity,
    date)`` tuple tests and trains its entity only.

    Args:
        n_splits: Only sets the default ``test_size`` of a time series; the
            split always has one fold (at least 2, as for sklearn splitters).
        test_indices: Test rows: integer positions (never booleans), dates or,
            for a panel, ``(entity, date)`` tuples (see ``OutOfSampleSplit``).
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
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size

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
        # Validation des paramètres et de la cible
        _check_positive_or_none('test_size', self.test_size)
        _check_positive_or_none('max_train_size', self.max_train_size)
        _check_target(X, y)
        return self._split(X, y, groups)

    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X, y=None, groups=None):
        """Yield the fold on sorted data, mapped back to the positions of ``X``."""
        (X,) = indexable(X)
        X_sorted, groups_sorted, sort_mapping = _verify_and_sort_data(X, groups)
        parsed = _parse_test_indices(self.test_indices)
        # Une position ou un couple (entité, date) ne concerne que son entité
        every_group = parsed is None or parsed[0] == _DATES
        for test_indices in self._iter_test_indices(X_sorted, y, groups_sorted):
            train_indices = self._get_train_indices(X_sorted, test_indices, groups_sorted, every_group)
            yield sort_mapping[train_indices], sort_mapping[test_indices]

    # Méthode auxiliaire d'extraction des indices d'entraînement incluant la période de test
    def _get_train_indices(self, X, test_indices, groups, every_group=True):
        """Return the training positions of the fold (sorted data)."""
        if groups is None:
            return self._get_timeseries_train_indices(test_indices)
        return self._get_group_train_indices(test_indices, groups, every_group)

    def _window(self, test_indices, block_start):
        """Return the training range ending at the last test row, starting no earlier than ``block_start``."""
        first_test, last_test = int(np.min(test_indices)), int(np.max(test_indices))
        train_end = last_test + 1
        if self.max_train_size is None:
            return range(block_start, train_end)
        # La fenêtre couvre au moins toute l'étendue du test
        size = max(self.max_train_size, last_test - first_test + 1)
        return range(max(block_start, train_end - size), train_end)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les séries temporelles
    def _get_timeseries_train_indices(self, test_indices):
        """Return the rows up to the last test row, within the window."""
        return np.asarray(self._window(test_indices, 0), dtype=int)

    # Méthode auxiliaire d'extraction des indices d'entraînement pour les données de panel
    def _get_group_train_indices(self, test_indices, groups, every_group=True):
        """Return, entity by entity, the rows up to its last test row, within the window.

        Raises:
            ValueError: If an entity has no test row while ``every_group`` is
                True.
        """
        train_indices: list[int] = []
        for group, group_rows in _precompute_group_mappings(groups).items():
            group_test = test_indices[np.isin(test_indices, group_rows)]
            if len(group_test) == 0:
                if every_group:
                    raise ValueError(f"Cannot find test indices for the group : {group}")
                continue
            train_indices.extend(self._window(group_test, int(group_rows.min())))
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
        positions = _resolve_test_positions(self.test_indices, X)
        if positions is None:
            if groups is None:
                yield from self._iter_default_timeseries_test_indices(X, n_samples)
            else:
                yield from self._iter_default_group_test_indices(X, groups)
            return

        test_size = self.test_size if self.test_size is not None else 1
        _check_positions_in_range(positions, n_samples)
        if groups is not None and _parse_test_indices(self.test_indices)[0] == _DATES:
            yield from self._iter_group_test_indices(X, positions, groups)
            return
        if len(positions) > 1:
            # Plusieurs indices : une ligne par indice, test_size non utilisé
            yield np.unique(positions)
            return
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
        test_size = self.test_size if self.test_size is not None else 1
        test_times = _parse_test_indices(self.test_indices)[1]
        fold: list[int] = []
        for group, group_rows in _precompute_group_mappings(groups).items():
            for test_time in test_times:
                starts = _rows_at(X, group_rows, test_time)
                if len(starts) == 0:
                    warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                    continue
                start = int(starts[0])
                length = test_size if len(test_times) == 1 else 1
                fold.extend(range(start, min(start + length, int(group_rows.max()) + 1)))
        yield np.unique(np.asarray(fold, dtype=int))

    # Méthode auxiliaire d'identification des indices de test par défaut pour les séries temporelles
    def _iter_default_timeseries_test_indices(self, X, n_samples):
        """Yield the last ``test_size`` rows.

        Raises:
            ValueError: If the data is empty or the test period has no row.
        """
        if n_samples == 0:
            raise ValueError("Cannot split an empty dataset: number of samples=0")
        test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)
        if test_size < 1:
            raise ValueError(
                f"Empty test period: test_size={test_size} for number of samples={n_samples} "
                f"with n_splits={self.n_splits}; set test_size or reduce n_splits"
            )
        yield np.arange(max(0, n_samples - test_size), n_samples)

    # Méthode auxiliaire d'identification des indices de test par défaut pour les données de panel
    def _iter_default_group_test_indices(self, X, groups):
        """Yield the rows of each entity at the last ``test_size`` dates of the panel.

        An entity lacking one of these dates is warned about.
        """
        test_size = self.test_size if self.test_size is not None else 1
        test_times = _sorted_time_points(X)[-test_size:]
        fold: list[int] = []
        for group, group_rows in _precompute_group_mappings(groups).items():
            for test_time in test_times:
                rows = _rows_at(X, group_rows, test_time)
                if len(rows) == 0:
                    warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                    continue
                fold.extend(rows.tolist())
        yield np.unique(np.asarray(fold, dtype=int))
