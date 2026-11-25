"""Property checks shared by the cross-validation tests.

Provides ``collect_folds``, ``as_lists``, ``time_values``, ``entity_values``,
``split_folds_by_entity`` and the assertions ``assert_valid_folds``,
``assert_no_leakage``, ``assert_train_window``, ``assert_in_sample``,
``assert_folds_cover`` and ``assert_n_splits_consistent``. Each assertion
checks one safety or coverage property of a splitter on every fold, on the
dates of the rows and entity by entity for panel data.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

Fold = tuple[np.ndarray, np.ndarray]
ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]

# Valeur d'entité unique attribuée à toutes les lignes d'une série temporelle
_SINGLE_ENTITY = None


def collect_folds(
    splitter: Any,
    X: Any,
    y: Any = None,
    groups: Any = None,
) -> list[Fold]:
    """Run ``splitter.split`` and return every fold as a pair of arrays.

    Args:
        splitter: Object exposing the sklearn ``split(X, y, groups)`` method.
        X: Data passed to ``split``.
        y: Target passed to ``split``.
        groups: Group labels passed to ``split``.

    Returns:
        List of ``(train_positions, test_positions)`` pairs, in the order the
        splitter yields them.

    Raises:
        Exception: Whatever ``splitter.split`` raises.

    Examples:
        >>> from sklearn.model_selection import TimeSeriesSplit
        >>> folds = collect_folds(TimeSeriesSplit(n_splits=2), np.arange(6))
        >>> [(train.tolist(), test.tolist()) for train, test in folds]
        [([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])]
    """
    return [(np.asarray(train), np.asarray(test)) for train, test in splitter.split(X, y, groups)]


def as_lists(folds: Sequence[Fold]) -> list[tuple[list[int], list[int]]]:
    """Convert folds to plain lists, for comparison with hand-written folds.

    Args:
        folds: Folds returned by :func:`collect_folds`.

    Returns:
        List of ``(train_list, test_list)`` pairs of Python integers.

    Examples:
        >>> as_lists([(np.array([0, 1]), np.array([2]))])
        [([0, 1], [2])]
    """
    return [(np.asarray(train).tolist(), np.asarray(test).tolist()) for train, test in folds]


def time_values(X: Any) -> np.ndarray:
    """Return the time label of each row of ``X``.

    The time label is the last level of a ``MultiIndex`` (panel convention
    ``(entity..., date)``), the index of any other pandas object, and the row
    position of an array.

    Args:
        X: DataFrame, Series or array-like.

    Returns:
        Array of time labels, one per row.

    Examples:
        >>> idx = pd.MultiIndex.from_product([['A'], [2020, 2021]])
        >>> time_values(pd.Series([1.0, 2.0], index=idx)).tolist()
        [2020, 2021]
        >>> time_values(np.zeros((3, 2))).tolist()
        [0, 1, 2]
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        index = X.index.get_level_values(-1) if isinstance(X.index, pd.MultiIndex) else X.index
        # Conversion en objets pandas (Timestamp, Period) : messages lisibles et
        # comparaisons sans conversion numpy
        return index.astype(object).to_numpy()
    return np.arange(len(X))


def entity_values(X: Any, groups: Any = None) -> np.ndarray:
    """Return the entity of each row of ``X``.

    Explicit ``groups`` win; otherwise the entity is read from every level of
    a ``MultiIndex`` but the last (a tuple when there are several). Rows of a
    time series all belong to one entity, ``None``.

    Args:
        X: DataFrame, Series or array-like.
        groups: Optional explicit group label of each row.

    Returns:
        Object array of entity labels, one per row.

    Examples:
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], [1, 2]])
        >>> entity_values(pd.Series(range(4), index=idx)).tolist()
        ['A', 'A', 'B', 'B']
        >>> entity_values(np.zeros(2)).tolist()
        [None, None]
    """
    if groups is not None:
        labels = list(groups)
    elif isinstance(X, (pd.DataFrame, pd.Series)) and isinstance(X.index, pd.MultiIndex):
        labels = X.index.droplevel(-1).to_list()
    else:
        labels = [_SINGLE_ENTITY] * len(X)
    # Remplissage élément par élément : une liste de tuples passée à np.asarray
    # produirait un tableau à deux dimensions
    result = np.empty(len(labels), dtype=object)
    result[:] = labels
    return result


def _check(condition: Any, message: str) -> None:
    """Raise ``AssertionError(message)`` when ``condition`` is false.

    An explicit raise keeps the message intact whatever the assertion
    rewriting of the caller and survives ``python -O``.
    """
    if not condition:
        raise AssertionError(message)


def _rows_of(entities: np.ndarray, entity: Hashable) -> np.ndarray:
    """Return the positions of the rows belonging to ``entity``."""
    return np.flatnonzero([label == entity for label in entities])


def split_folds_by_entity(
    X: Any,
    folds: Sequence[Fold],
    groups: Any = None,
) -> dict[Hashable, list[Fold]]:
    """Restrict every fold to the rows of each entity.

    Args:
        X: Data the folds index into.
        folds: Folds returned by :func:`collect_folds`.
        groups: Optional explicit group label of each row.

    Returns:
        Mapping entity → list (one item per fold, same order) of the
        ``(train_positions, test_positions)`` of that entity; positions stay
        positions of ``X``.

    Examples:
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], [1, 2]])
        >>> X = pd.Series(range(4), index=idx)
        >>> by_entity = split_folds_by_entity(X, [(np.array([0, 2]), np.array([1, 3]))])
        >>> [(train.tolist(), test.tolist()) for train, test in by_entity['B']]
        [([2], [3])]
    """
    entities = entity_values(X, groups)
    result: dict[Hashable, list[Fold]] = {}
    for entity in dict.fromkeys(entities):
        rows = _rows_of(entities, entity)
        result[entity] = [
            (np.asarray(train)[np.isin(train, rows)], np.asarray(test)[np.isin(test, rows)])
            for train, test in folds
        ]
    return result


def assert_valid_folds(X: Any, folds: Sequence[Fold]) -> None:
    """Assert that every fold holds integer, in-range, duplicate-free positions.

    Args:
        X: Data the folds index into.
        folds: Folds returned by :func:`collect_folds`.

    Raises:
        AssertionError: If a position array is not of integer dtype, holds a
            position outside ``[0, len(X))`` or repeats a position.

    Examples:
        >>> assert_valid_folds(np.zeros(4), [(np.array([0, 1]), np.array([2]))])
        >>> assert_valid_folds(np.zeros(4), [(np.array([]), np.array([2]))])
        Traceback (most recent call last):
        ...
        AssertionError: fold 0: train positions have dtype float64, not an integer dtype
    """
    n_samples = len(X)
    for k, (train, test) in enumerate(folds):
        for name, positions in (("train", train), ("test", test)):
            _check(positions.dtype.kind in "iu", (
                f"fold {k}: {name} positions have dtype {positions.dtype}, not an integer dtype"
            ))
            _check(np.all((positions >= 0) & (positions < n_samples)), (
                f"fold {k}: {name} positions outside [0, {n_samples}): {positions.tolist()}"
            ))
            _check(len(np.unique(positions)) == len(positions), (
                f"fold {k}: {name} positions repeat: {positions.tolist()}"
            ))


def _entity_dates(times: np.ndarray, rows: np.ndarray) -> pd.Index:
    """Return the sorted distinct time labels of the given rows."""
    return pd.Index(times[rows]).unique().sort_values()


def assert_no_leakage(
    X: Any,
    folds: Sequence[Fold],
    gap: int = 0,
    *,
    groups: Any = None,
    exact_gap: bool = True,
) -> None:
    """Assert that training dates strictly precede test dates, entity by entity.

    For each fold and each entity that has both training and test rows, every
    training date is strictly earlier than every test date, and the number of
    the entity's dates lying strictly between the last training date and the
    first test date equals ``gap`` (at least ``gap`` when ``exact_gap`` is
    False). Rows sharing a date count as one date.

    Args:
        X: Data the folds index into.
        folds: Folds returned by :func:`collect_folds`.
        gap: Number of dates expected between training and test.
        groups: Optional explicit group label of each row.
        exact_gap: If True, the gap must be met exactly; otherwise it is a
            minimum.

    Raises:
        AssertionError: On the first fold and entity breaking the rule.

    Examples:
        >>> X = pd.Series(range(5), index=pd.date_range('2024-01-31', periods=5, freq='ME'))
        >>> assert_no_leakage(X, [(np.array([0, 1]), np.array([3, 4]))], gap=1)
        >>> assert_no_leakage(X, [(np.array([0, 3]), np.array([2]))])
        Traceback (most recent call last):
        ...
        AssertionError: fold 0, entity None: training date 2024-04-30 00:00:00 is not before test date 2024-03-31 00:00:00
    """
    times = time_values(X)
    entities = entity_values(X, groups)
    for entity, entity_folds in split_folds_by_entity(X, folds, groups).items():
        entity_dates = _entity_dates(times, _rows_of(entities, entity))
        for k, (train, test) in enumerate(entity_folds):
            if len(train) == 0 or len(test) == 0:
                continue
            last_train, first_test = max(times[train]), min(times[test])
            _check(last_train < first_test, (
                f"fold {k}, entity {entity}: training date {last_train} is not before test date "
                f"{first_test}"
            ))
            # Nombre de dates de l'entité strictement comprises entre les deux bornes
            between = entity_dates.get_loc(first_test) - entity_dates.get_loc(last_train) - 1
            if exact_gap:
                _check(between == gap, (
                    f"fold {k}, entity {entity}: {between} dates between training and test, "
                    f"expected {gap}"
                ))
            else:
                _check(between >= gap, (
                    f"fold {k}, entity {entity}: {between} dates between training and test, "
                    f"expected at least {gap}"
                ))


def assert_train_window(
    X: Any,
    folds: Sequence[Fold],
    *,
    gap: int = 0,
    max_train_size: Optional[int] = None,
    groups: Any = None,
) -> None:
    """Assert the out-of-sample training window of every fold, entity by entity.

    The training rows of an entity are the contiguous run of its dates that
    ends ``gap`` dates before its first test date; the run starts at the
    entity's first date (expanding window) or holds ``max_train_size`` dates
    when that much history exists (rolling window).

    Args:
        X: Data the folds index into, sorted by entity then date.
        folds: Folds returned by :func:`collect_folds`.
        gap: Number of dates skipped between training and test.
        max_train_size: Rolling window length, or None for an expanding window.
        groups: Optional explicit group label of each row.

    Raises:
        AssertionError: On the first fold and entity whose training window
            differs from the expected one.

    Examples:
        >>> X = pd.Series(range(6), index=pd.date_range('2024-01-31', periods=6, freq='ME'))
        >>> assert_train_window(X, [(np.array([1, 2]), np.array([4]))], gap=1, max_train_size=2)
        >>> assert_train_window(X, [(np.array([2]), np.array([4]))], gap=1)
        Traceback (most recent call last):
        ...
        AssertionError: fold 0, entity None: training positions [2], expected [0, 1, 2]
    """
    entities = entity_values(X, groups)
    for entity, entity_folds in split_folds_by_entity(X, folds, groups).items():
        rows = _rows_of(entities, entity)
        for k, (train, test) in enumerate(entity_folds):
            if len(test) == 0:
                continue
            # Position, dans l'entité, de la première ligne de test
            first_test = int(np.flatnonzero(rows == test.min())[0])
            end = max(first_test - gap, 0)
            start = 0 if max_train_size is None else max(end - max_train_size, 0)
            expected = rows[start:end]
            _check(np.array_equal(np.sort(train), expected), (
                f"fold {k}, entity {entity}: training positions {np.sort(train).tolist()}, "
                f"expected {expected.tolist()}"
            ))


def assert_in_sample(X: Any, folds: Sequence[Fold], *, groups: Any = None) -> None:
    """Assert the in-sample contract of every fold, entity by entity.

    For each fold and each entity with test rows: the test rows belong to the
    training rows, no training date is later than the entity's last test
    date, and the training dates form a contiguous run of the entity's dates.

    Args:
        X: Data the folds index into.
        folds: Folds returned by :func:`collect_folds`.
        groups: Optional explicit group label of each row.

    Raises:
        AssertionError: On the first fold and entity breaking the contract.

    Examples:
        >>> X = pd.Series(range(5), index=pd.date_range('2024-01-31', periods=5, freq='ME'))
        >>> assert_in_sample(X, [(np.array([1, 2, 3]), np.array([2, 3]))])
        >>> assert_in_sample(X, [(np.array([3, 4]), np.array([1, 4]))])
        Traceback (most recent call last):
        ...
        AssertionError: fold 0, entity None: test positions [1] are not in the training set
    """
    times = time_values(X)
    entities = entity_values(X, groups)
    for entity, entity_folds in split_folds_by_entity(X, folds, groups).items():
        entity_dates = _entity_dates(times, _rows_of(entities, entity))
        for k, (train, test) in enumerate(entity_folds):
            if len(test) == 0:
                continue
            missing = np.setdiff1d(test, train)
            _check(len(missing) == 0, (
                f"fold {k}, entity {entity}: test positions {missing.tolist()} are not in the "
                f"training set"
            ))
            _check(max(times[train]) == max(times[test]), (
                f"fold {k}, entity {entity}: training ends at {max(times[train])}, after the last "
                f"test date {max(times[test])}"
            ))
            train_dates = _entity_dates(times, train)
            first = entity_dates.get_loc(train_dates[0])
            _check(train_dates.equals(entity_dates[first:first + len(train_dates)]), (
                f"fold {k}, entity {entity}: training dates are not contiguous"
            ))


def assert_folds_cover(
    folds: Sequence[Fold],
    *,
    test_size: Optional[int] = None,
    n_samples: Optional[int] = None,
    contiguous: bool = True,
) -> None:
    """Assert that test sets are disjoint and chronologically successive.

    Works on positions, for a time series sorted by date (or the folds of one
    entity of a grouped panel, see :func:`split_folds_by_entity`): each test
    set lies entirely after the previous one, adjacent to it when
    ``contiguous`` is True; each holds ``test_size`` positions when given; the
    last one ends at position ``n_samples - 1`` when given.

    Args:
        folds: Folds returned by :func:`collect_folds`.
        test_size: Expected size of every test set.
        n_samples: Number of rows; when given, the last test set must end at
            the last row.
        contiguous: If True, consecutive test sets must be adjacent.

    Raises:
        AssertionError: On the first test set breaking a rule.

    Examples:
        >>> folds = [(np.array([0]), np.array([1, 2])), (np.array([0, 1, 2]), np.array([3, 4]))]
        >>> assert_folds_cover(folds, test_size=2, n_samples=5)
        >>> assert_folds_cover(folds, n_samples=6)
        Traceback (most recent call last):
        ...
        AssertionError: last test set ends at 4, expected 5
    """
    tests = [np.sort(test) for _, test in folds]
    for k, test in enumerate(tests):
        if test_size is not None:
            _check(len(test) == test_size, f"test set {k} holds {len(test)} rows, expected {test_size}")

        if k == 0:
            continue
        previous = tests[k - 1]
        _check(previous.max() < test.min(), (
            f"test set {k} starts at {test.min()}, not after test set {k - 1} (ends at "
            f"{previous.max()})"
        ))
        if contiguous:
            _check(previous.max() + 1 == test.min(), (
                f"test set {k} starts at {test.min()}, expected {previous.max() + 1}"
            ))
    if n_samples is not None and tests:
        _check(tests[-1].max() == n_samples - 1, (
            f"last test set ends at {tests[-1].max()}, expected {n_samples - 1}"
        ))


def assert_n_splits_consistent(
    splitter: Any,
    X: Any,
    y: Any = None,
    groups: Any = None,
) -> None:
    """Assert that ``get_n_splits`` announces exactly the folds ``split`` produces.

    Follows the sklearn convention: the count must not depend on the data, so
    ``get_n_splits()`` without arguments and ``get_n_splits(X, y, groups)``
    must both equal the number of folds. Data that ``split`` refuses is not
    covered: the error of ``split`` propagates.

    Args:
        splitter: Object exposing ``split`` and ``get_n_splits``.
        X: Data passed to both methods.
        y: Target passed to both methods.
        groups: Group labels passed to both methods.

    Raises:
        AssertionError: If an announced number differs from the number of
            folds produced.
        Exception: Whatever ``splitter.split`` raises.

    Examples:
        >>> from sklearn.model_selection import TimeSeriesSplit
        >>> assert_n_splits_consistent(TimeSeriesSplit(n_splits=3), np.arange(8))
    """
    n_folds = len(collect_folds(splitter, X, y, groups))
    with_data = splitter.get_n_splits(X, y, groups)
    without_data = splitter.get_n_splits()
    _check(
        with_data == without_data == n_folds,
        f"get_n_splits announced {with_data} folds (with data) and {without_data} (without), "
        f"split produced {n_folds}",
    )
