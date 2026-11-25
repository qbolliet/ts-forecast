"""Tests of the cross-validation property checks of ``tests/support/splits.py``.

Covers ``collect_folds``, ``time_values``, ``entity_values``,
``split_folds_by_entity`` and each assertion (``assert_valid_folds``,
``assert_no_leakage``, ``assert_train_window``, ``assert_in_sample``,
``assert_folds_cover``, ``assert_n_splits_consistent``): every check accepts
hand-built valid folds and rejects one hand-built violation of its rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.support.splits import (
    assert_folds_cover,
    assert_in_sample,
    assert_n_splits_consistent,
    assert_no_leakage,
    assert_train_window,
    assert_valid_folds,
    collect_folds,
    entity_values,
    split_folds_by_entity,
    time_values,
)


def _fold(train: list[int], test: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Build a fold from two lists of integer positions."""
    return np.array(train, dtype=int), np.array(test, dtype=int)


@pytest.fixture
def series_8() -> pd.Series:
    """Eight month-end dates of 2024, value equal to the position."""
    return pd.Series(range(8), index=pd.date_range('2024-01-31', periods=8, freq='ME'))


@pytest.fixture
def panel_2x4() -> pd.DataFrame:
    """Two entities ``A`` (positions 0-3) and ``B`` (positions 4-7), four month-ends each."""
    dates = pd.date_range('2024-01-31', periods=4, freq='ME')
    index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
    return pd.DataFrame({'v': range(8)}, index=index)


class _FixedSplitter:
    """Splitter double yielding fixed folds and announcing a fixed count."""

    def __init__(self, folds, announced, fail_with=None):
        self.folds = folds
        self.announced = announced
        self.fail_with = fail_with

    def split(self, X, y=None, groups=None):
        if self.fail_with is not None:
            raise self.fail_with
        yield from self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.announced


class TestReaders:
    """Fold collection and per-row time and entity labels."""

    def test_collect_folds_returns_arrays_in_order(self):
        """Folds come back as numpy arrays, in the order of the splitter."""
        splitter = _FixedSplitter([([0], [1]), ([0, 1], [2])], announced=2)
        folds = collect_folds(splitter, np.zeros(3))
        assert [(tr.tolist(), te.tolist()) for tr, te in folds] == [([0], [1]), ([0, 1], [2])]

    def test_time_values_of_a_panel_is_the_last_level(self, panel_2x4):
        """The time label of a panel row is its last index level."""
        assert list(time_values(panel_2x4)[:2]) == [pd.Timestamp('2024-01-31'), pd.Timestamp('2024-02-29')]

    def test_time_values_of_a_period_index_keeps_periods(self):
        """Periods stay periods."""
        X = pd.Series([1.0, 2.0], index=pd.period_range('2024-01', periods=2, freq='M'))
        assert list(time_values(X)) == [pd.Period('2024-01', 'M'), pd.Period('2024-02', 'M')]

    def test_entity_values_of_a_three_level_index_are_tuples(self):
        """With three levels, the entity is the tuple of the first two."""
        index = pd.MultiIndex.from_product([['FR'], ['x', 'y'], [1, 2]])
        assert entity_values(pd.Series(range(4), index=index)).tolist() == [
            ('FR', 'x'), ('FR', 'x'), ('FR', 'y'), ('FR', 'y'),
        ]

    def test_explicit_groups_win_over_the_index(self, panel_2x4):
        """Explicit groups replace the entity read from the index."""
        assert entity_values(panel_2x4, groups=[1] * 8).tolist() == [1] * 8

    def test_split_folds_by_entity_keeps_global_positions(self, panel_2x4):
        """Each entity receives its own rows of every fold, as positions of X."""
        by_entity = split_folds_by_entity(panel_2x4, [_fold([0, 1, 4, 5], [2, 6])])
        assert {e: [(tr.tolist(), te.tolist()) for tr, te in f] for e, f in by_entity.items()} == {
            'A': [([0, 1], [2])],
            'B': [([4, 5], [6])],
        }


class TestAssertValidFolds:
    """Integer, in-range, duplicate-free positions."""

    def test_accepts_valid_folds(self, series_8):
        """Valid folds pass."""
        assert_valid_folds(series_8, [_fold([0, 1], [2, 3])])

    @pytest.mark.parametrize(
        ('fold', 'message'),
        [
            ((np.array([]), np.array([2])), 'dtype float64'),
            (_fold([0, 8], [2]), 'outside'),
            (_fold([0], [2, 2]), 'repeat'),
        ],
        ids=['float-dtype', 'out-of-range', 'duplicate'],
    )
    def test_rejects_invalid_positions(self, series_8, fold, message):
        """A float array, an out-of-range or a repeated position is rejected."""
        with pytest.raises(AssertionError, match=message):
            assert_valid_folds(series_8, [fold])


class TestAssertNoLeakage:
    """Training dates strictly before test dates, gap measured on dates."""

    def test_accepts_an_exact_gap(self, series_8):
        """Two dates between the last training and the first test date match gap=2."""
        assert_no_leakage(series_8, [_fold([0, 1], [4, 5])], gap=2)

    def test_rejects_a_different_gap(self, series_8):
        """One date between training and test is not gap=2."""
        with pytest.raises(AssertionError, match='1 dates between training and test, expected 2'):
            assert_no_leakage(series_8, [_fold([0, 1, 2], [4])], gap=2)

    def test_minimum_gap_when_not_exact(self, series_8):
        """With exact_gap=False, a larger gap passes and a smaller one fails."""
        assert_no_leakage(series_8, [_fold([0], [4])], gap=2, exact_gap=False)
        with pytest.raises(AssertionError, match='expected at least 2'):
            assert_no_leakage(series_8, [_fold([0, 1, 2], [4])], gap=2, exact_gap=False)

    def test_rejects_a_training_date_after_the_test(self, series_8):
        """A training row dated after a test row is a leak."""
        with pytest.raises(AssertionError, match='is not before test date'):
            assert_no_leakage(series_8, [_fold([0, 5], [3])])

    def test_rejects_a_date_shared_by_training_and_test(self):
        """A date present on both sides of the split is a leak."""
        dates = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-02-29', '2024-03-31'])
        with pytest.raises(AssertionError, match='is not before test date'):
            assert_no_leakage(pd.Series(range(4), index=dates), [_fold([0, 1], [2, 3])])

    def test_is_checked_entity_by_entity(self, panel_2x4):
        """Entity B training on dates later than entity A's test is no leak."""
        # Valeur d'or : A apprend sur 01-02, teste 03 ; B apprend sur 01-03, teste 04
        assert_no_leakage(panel_2x4, [_fold([0, 1, 4, 5, 6], [2, 7])], exact_gap=False)

    def test_detects_a_leak_inside_one_entity(self, panel_2x4):
        """A leak within entity B is reported with the entity."""
        # Entité A saine (apprentissage 01, test 03) ; entité B : apprentissage 04, test 02
        with pytest.raises(AssertionError, match='entity B'):
            assert_no_leakage(panel_2x4, [_fold([0, 7], [2, 5])], exact_gap=False)

    def test_skips_empty_training_sets(self, series_8):
        """A fold without training rows cannot leak."""
        assert_no_leakage(series_8, [(np.array([], dtype=int), np.array([0]))], gap=3)


class TestAssertTrainWindow:
    """Expanding and rolling out-of-sample windows."""

    def test_expanding_window(self, series_8):
        """Expanding: training starts at the first date and stops gap dates before the test."""
        assert_train_window(series_8, [_fold([0, 1, 2], [4, 5])], gap=1)

    def test_rolling_window(self, series_8):
        """Rolling: training holds max_train_size dates."""
        assert_train_window(series_8, [_fold([2, 3], [4])], max_train_size=2)

    def test_rolling_window_shorter_than_the_history(self, series_8):
        """With less history than max_train_size, training holds all of it."""
        assert_train_window(series_8, [_fold([0], [1])], max_train_size=3)

    def test_rejects_a_truncated_expanding_window(self, series_8):
        """A window that does not start at the first date is not expanding."""
        with pytest.raises(AssertionError, match=r'expected \[0, 1, 2, 3\]'):
            assert_train_window(series_8, [_fold([1, 2, 3], [4])])

    def test_panel_windows_per_entity(self, panel_2x4):
        """Each entity has its own window."""
        assert_train_window(panel_2x4, [_fold([1, 5], [2, 6])], max_train_size=1)


class TestAssertInSample:
    """Test rows included in a contiguous training run ending at the last test date."""

    def test_accepts_the_contract(self, series_8):
        """Training [2..5] contains test [4, 5] and stops at the last test date."""
        assert_in_sample(series_8, [_fold([2, 3, 4, 5], [4, 5])])

    def test_rejects_training_after_the_last_test_date(self, series_8):
        """Training must not go beyond the last test date."""
        with pytest.raises(AssertionError, match='after the last test date'):
            assert_in_sample(series_8, [_fold([2, 3, 4, 5, 6], [4, 5])])

    def test_rejects_non_contiguous_training(self, series_8):
        """Training dates must form one run."""
        with pytest.raises(AssertionError, match='not contiguous'):
            assert_in_sample(series_8, [_fold([1, 4, 5], [4, 5])])


class TestAssertFoldsCover:
    """Disjoint, successive, sized test sets."""

    def test_accepts_contiguous_test_sets_ending_at_the_last_row(self):
        """Test sets [4, 5] and [6, 7] cover the end of eight rows."""
        assert_folds_cover([_fold([0], [4, 5]), _fold([0], [6, 7])], test_size=2, n_samples=8)

    @pytest.mark.parametrize(
        ('folds', 'kwargs', 'message'),
        [
            ([_fold([0], [4, 5]), _fold([0], [5, 6])], {}, 'not after'),
            ([_fold([0], [2, 3]), _fold([0], [6, 7])], {}, 'expected 4'),
            ([_fold([0], [4, 5, 6])], {'test_size': 2}, 'holds 3 rows'),
            ([_fold([0], [4, 5])], {'n_samples': 8}, 'ends at 5'),
        ],
        ids=['overlapping', 'hole', 'size', 'not-at-the-end'],
    )
    def test_rejects_each_violation(self, folds, kwargs, message):
        """Overlap, hole, wrong size and early end are each rejected."""
        with pytest.raises(AssertionError, match=message):
            assert_folds_cover(folds, **kwargs)

    def test_hole_allowed_when_not_contiguous(self):
        """With contiguous=False, successive test sets may leave a hole."""
        assert_folds_cover([_fold([0], [2, 3]), _fold([0], [6, 7])], contiguous=False)


class TestAssertNSplitsConsistent:
    """Announced fold count against produced folds, sklearn convention."""

    def test_accepts_a_matching_count(self):
        """Two folds announced, two produced."""
        assert_n_splits_consistent(_FixedSplitter([([0], [1]), ([0], [2])], announced=2), np.zeros(3))

    def test_rejects_a_different_count(self):
        """Five announced, one produced."""
        with pytest.raises(AssertionError, match=r'announced 5 folds \(with data\).*split produced 1'):
            assert_n_splits_consistent(_FixedSplitter([([0], [1])], announced=5), np.zeros(3))

    def test_split_errors_propagate(self):
        """Data refused by split is outside the check: the error of split surfaces."""
        splitter = _FixedSplitter([], announced=5, fail_with=ValueError('too many splits'))
        with pytest.raises(ValueError, match='too many splits'):
            assert_n_splits_consistent(splitter, np.zeros(3))
