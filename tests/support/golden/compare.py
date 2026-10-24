"""Strict comparison of two canonical records of the golden master.

:func:`assert_same` walks two objects produced by
:func:`tests.support.golden.scenarios.canonicalize` and raises on the first
difference, naming its path: pandas objects through ``pandas.testing`` with
``check_exact=True`` and every type check enabled, numpy arrays on dtype and
values, floats bit for bit, containers on their type, length, key order and
items.
"""
# Modules de base
import math
from typing import Any

import numpy as np
import pandas as pd


def _same_float(expected: float, actual: float) -> bool:
    """Tell whether two floats are identical bit for bit, NaN equal to NaN."""
    if math.isnan(expected) and math.isnan(actual):
        return True
    return float(expected).hex() == float(actual).hex()


def _is_missing_scalar(value: Any) -> bool:
    """Tell whether a scalar is a missing value (NaN, NaT, ``pd.NA``)."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def assert_same(expected: Any, actual: Any, path: str = 'record') -> None:
    """Assert that two canonical objects are identical.

    Args:
        expected: Object read from a golden file.
        actual: Object produced by the current code.
        path: Location of the objects inside the record, quoted in the error.

    Raises:
        AssertionError: On the first difference: type, dtype, shape, index,
            key order or value. The message starts with the path.

    Examples:
        >>> assert_same({'a': [1.0, float('nan')]}, {'a': [1.0, float('nan')]})
        >>> try:
        ...     assert_same({'a': (1, 2)}, {'a': (1, 3)})
        ... except AssertionError as error:
        ...     print(error)
        record['a'][1]: 2 != 3
    """
    if type(expected) is not type(actual):
        raise AssertionError(
            f'{path}: type {type(expected).__name__} != {type(actual).__name__}'
        )

    # Objets pandas : comparaison exacte, tous les contrôles de type actifs
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(
            expected, actual, check_exact=True, check_index_type=True,
            check_column_type=True, check_freq=True, check_flags=True, obj=path,
        )
        return
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(
            expected, actual, check_exact=True, check_index_type=True,
            check_series_type=True, check_freq=True, check_flags=True, obj=path,
        )
        return
    if isinstance(expected, pd.Index):
        pd.testing.assert_index_equal(expected, actual, exact=True, check_exact=True, obj=path)
        if getattr(expected, 'freq', None) != getattr(actual, 'freq', None):
            raise AssertionError(f'{path}: freq {expected.freq} != {actual.freq}')
        return

    # Tableaux numpy : dtype, forme, puis valeurs (élément par élément si object)
    if isinstance(expected, np.ndarray):
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(
                f'{path}: array {expected.dtype}{expected.shape} != {actual.dtype}{actual.shape}'
            )
        if expected.dtype == object:
            for position, (left, right) in enumerate(zip(expected.ravel(), actual.ravel())):
                assert_same(left, right, f'{path}[{position}]')
            return
        if not np.array_equal(expected, actual, equal_nan=expected.dtype.kind in 'fcmM'):
            raise AssertionError(f'{path}: arrays differ\n{expected!r}\n{actual!r}')
        return

    # Conteneurs : ordre des clés et des éléments compris
    if isinstance(expected, dict):
        if list(expected) != list(actual):
            raise AssertionError(f'{path}: keys {list(expected)} != {list(actual)}')
        for key in expected:
            assert_same(expected[key], actual[key], f'{path}[{key!r}]')
        return
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError(f'{path}: length {len(expected)} != {len(actual)}')
        for position, (left, right) in enumerate(zip(expected, actual)):
            assert_same(left, right, f'{path}[{position}]')
        return
    if isinstance(expected, frozenset):
        if expected != actual:
            raise AssertionError(f'{path}: {sorted(map(repr, expected))} != {sorted(map(repr, actual))}')
        return

    # Scalaires : flottants au bit près, manquants égaux entre eux
    if isinstance(expected, (float, np.floating)):
        if not _same_float(expected, actual):
            raise AssertionError(f'{path}: {expected!r} != {actual!r}')
        return
    if _is_missing_scalar(expected) and _is_missing_scalar(actual):
        return
    if expected != actual:
        raise AssertionError(f'{path}: {expected!r} != {actual!r}')
