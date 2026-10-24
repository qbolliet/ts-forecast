"""Tests of the golden master tooling of ``tests/support/golden``.

Covers the strict comparison (``assert_same``: bit-level floats, dtypes, date
units, index frequency, categories, key order, container types, warning
lists), the canonical form (``canonicalize``: enum members, dataclasses,
estimators, refusal of unknown objects), the store (refusal of a package
object, reproducible bytes) and one cheap case of ``run_case`` (no package
class in the record, same record twice).
"""
# Modules de base
import dataclasses
import pickle
from enum import Enum

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Outils testés
from tests.support.golden.compare import assert_same
from tests.support.golden.scenarios import GoldenCase, canonicalize, run_case
from tests.support.golden.store import read_record, write_record

# Membre d'énumération du paquet, pour les contrôles de sérialisation
from tsforecast.frequency.provenance import ProvenanceType


def _frame(**overrides) -> pd.DataFrame:
    """Return a small dated frame with a NaN, optionally overriding columns."""
    dates = pd.date_range('2021-01-31', periods=3, freq='ME')
    columns = {'a': [1.0, np.nan, 3.0], 'b': ['x', 'y', 'z']}
    columns.update(overrides)
    return pd.DataFrame(columns, index=dates)


# =============================================================================
# Comparaison stricte
# =============================================================================
class TestAssertSameAccepts:
    """Identical canonical objects pass, NaN included."""

    def test_nested_record_with_nan(self):
        """A nested record equals its rebuilt copy, NaN and NaT equal to themselves."""
        record = {
            'frame': _frame(),
            'pairs': [('UserWarning', 'message')],
            'scalars': (np.float64('nan'), pd.NaT, None, 'Q'),
        }
        rebuilt = {
            'frame': _frame(),
            'pairs': [('UserWarning', 'message')],
            'scalars': (np.float64('nan'), pd.NaT, None, 'Q'),
        }
        assert_same(record, rebuilt)


class TestAssertSameRejects:
    """Every kind of difference raises, naming its path."""

    def test_last_bit_of_a_float(self):
        """``0.1 + 0.2`` and ``0.3`` differ by their last bit."""
        with pytest.raises(AssertionError, match=r"record\['x'\]"):
            assert_same({'x': 0.3}, {'x': 0.1 + 0.2})

    def test_last_bit_inside_a_frame(self):
        """A frame cell off by one ulp fails the exact comparison."""
        shifted = _frame(a=[1.0, np.nan, np.nextafter(3.0, 4.0)])
        with pytest.raises(AssertionError):
            assert_same(_frame(), shifted)

    def test_column_dtype(self):
        """The same values in ``float32`` do not equal ``float64`` ones."""
        with pytest.raises(AssertionError):
            assert_same(_frame(), _frame().astype({'a': 'float32'}))

    def test_date_unit(self):
        """The same instants stored in ``ns`` and in ``us`` differ."""
        frame = _frame()
        other = frame.copy()
        other.index = other.index.as_unit('ns' if frame.index.unit != 'ns' else 'us')
        with pytest.raises(AssertionError):
            assert_same(frame, other)

    def test_index_frequency(self):
        """An index that lost its ``freq`` differs from the original one."""
        index = pd.date_range('2021-01-31', periods=3, freq='ME')
        with pytest.raises(AssertionError, match='freq'):
            assert_same(index, pd.DatetimeIndex(list(index)))

    def test_categories(self):
        """Two categoricals with the same values but other categories differ."""
        left = pd.Series(pd.Categorical(['a'], categories=['a', 'b']))
        right = pd.Series(pd.Categorical(['a'], categories=['a', 'c']))
        with pytest.raises(AssertionError):
            assert_same(left, right)

    def test_key_order(self):
        """Two dicts with the same items in another order differ."""
        with pytest.raises(AssertionError, match='keys'):
            assert_same({'a': 1, 'b': 2}, {'b': 2, 'a': 1})

    def test_container_type(self):
        """A tuple never equals a list holding the same items."""
        with pytest.raises(AssertionError, match='type'):
            assert_same((1, 2), [1, 2])

    def test_warning_order(self):
        """The same warnings emitted in another order differ."""
        first, second = ('UserWarning', 'one'), ('UserWarning', 'two')
        with pytest.raises(AssertionError):
            assert_same([first, second], [second, first])

    def test_integer_and_float(self):
        """``1`` and ``1.0`` differ: types are compared before values."""
        with pytest.raises(AssertionError, match='type'):
            assert_same(1, 1.0)


# =============================================================================
# Forme canonique
# =============================================================================
class _Colour(Enum):
    """Local enumeration for the canonical form tests."""

    RED = 'red'


@dataclasses.dataclass(frozen=True)
class _Pair:
    """Local dataclass for the canonical form tests."""

    left: object
    right: object


class TestCanonicalize:
    """The canonical form holds pandas, numpy and builtin objects only."""

    def test_enum_member_becomes_its_value(self):
        """A ``str`` enum member becomes a plain ``str``, keys of mappings included."""
        result = canonicalize({ProvenanceType.ORIGINAL: _Colour.RED})
        assert [(type(key), type(value)) for key, value in result.items()] == [(str, str)]

    def test_dataclass_becomes_a_dict(self):
        """A dataclass becomes its class name followed by its fields, in order."""
        assert canonicalize(_Pair(1, (_Colour.RED,))) == {
            '__class__': '_Pair', 'left': 1, 'right': ('red',)
        }

    def test_fitted_estimator_keeps_its_coefficients(self):
        """A fitted estimator keeps its public fitted attributes."""
        model = LinearRegression().fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))
        np.testing.assert_allclose(canonicalize(model)['coef_'], [2.0])

    def test_unknown_object_is_refused(self):
        """A function has no canonical form: the generation stops on it."""
        with pytest.raises(TypeError, match='function'):
            canonicalize(len)


# =============================================================================
# Stockage
# =============================================================================
class TestStore:
    """Golden files hold no package class and are reproducible to the byte."""

    def test_package_object_is_refused(self, tmp_path):
        """A raw enum member of the package would not load after a renaming."""
        with pytest.raises(TypeError, match='tsforecast'):
            write_record(tmp_path / 'case.pkl.gz', {'cell': ProvenanceType.ORIGINAL})

    def test_two_writes_give_the_same_bytes(self, tmp_path):
        """The gzip timestamp is null: identical records give identical files."""
        record = {'frame': _frame(), 'warnings': [('UserWarning', 'w')]}
        write_record(tmp_path / 'first.pkl.gz', record)
        write_record(tmp_path / 'second.pkl.gz', record)
        assert (tmp_path / 'first.pkl.gz').read_bytes() == (tmp_path / 'second.pkl.gz').read_bytes()

    def test_round_trip(self, tmp_path):
        """A record read back equals the record written."""
        record = {'frame': _frame(), 'nested': {('FR', 'q1'): 'Q'}}
        write_record(tmp_path / 'case.pkl.gz', record)
        assert_same(record, read_record(tmp_path / 'case.pkl.gz'))


# =============================================================================
# Exécution d'un cas
# =============================================================================
class TestRunCase:
    """One cheap case of the matrix: a regular time series, interpolation."""

    CASE = GoldenCase('ts', 'interpolate', False, 'frequency', 'sum', False, False)

    def test_record_references_no_package_class(self):
        """The four calls are recorded and the pickled record names no package class."""
        record = run_case(self.CASE)
        assert list(record) == [
            'fit_transform', 'state_after_fit_transform', 'inverse_fit_transform',
            'transform', 'state_after_transform', 'inverse_transform',
        ]
        assert b'tsforecast' not in pickle.dumps(record)

    def test_same_record_twice(self):
        """Two runs of one case in one process give identical records."""
        assert_same(run_case(self.CASE), run_case(self.CASE))
