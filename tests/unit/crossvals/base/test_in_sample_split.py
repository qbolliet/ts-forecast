"""Tests of the ``InSampleSplit`` cross-validator, golden values on small data.

Covers ``InSampleSplit`` (and ``TSInSampleSplit`` for the parameter
contract): parameters, the single default fold (``test_size`` derived from
``n_splits``, ``max_train_size``), explicit ``test_indices`` (one start or
several points), ``y`` handling, degenerate data and the panel branches
reached with explicit ``groups``. The in-sample contract tested here: one
fold whose training set holds the test rows and the dates before them, up
to the last test date and never beyond it.
"""
# Importation des modules
import numpy as np
import pandas as pd
import pytest

from tests.support.splits import as_lists, assert_in_sample, collect_folds
from tsforecast.crossvals.base import InSampleSplit
from tsforecast.crossvals.time_series import TSInSampleSplit


def _upto(k: int) -> list[int]:
    """Return the positions ``0 .. k-1``."""
    return list(range(k))


class TestParameters:
    """Constructor contract shared by the in-sample splitters."""

    @pytest.mark.parametrize('cls', [InSampleSplit, TSInSampleSplit], ids=['base', 'ts'])
    def test_defaults(self, cls):
        """Defaults: n_splits 5, no explicit test indices, no limit; no gap parameter."""
        splitter = cls()
        assert (splitter.n_splits, splitter.test_indices, splitter.max_train_size,
                splitter.test_size) == (5, None, None, None)
        assert not hasattr(splitter, 'gap')

    @pytest.mark.parametrize('cls', [InSampleSplit, TSInSampleSplit], ids=['base', 'ts'])
    def test_values_are_stored_as_given(self, cls):
        """Constructor arguments are stored unchanged."""
        dates = ['2020-01-15']
        splitter = cls(n_splits=3, test_indices=dates, max_train_size=50, test_size=10)
        assert (splitter.n_splits, splitter.max_train_size, splitter.test_size) == (3, 50, 10)
        assert splitter.test_indices is dates

    def test_gap_is_not_a_parameter(self):
        """In-sample splits have no gap."""
        with pytest.raises(TypeError):
            InSampleSplit(gap=1)

    def test_n_splits_below_two_is_refused_even_for_one_fold(self):
        """n_splits=1 is refused although an in-sample split always has one fold."""
        with pytest.raises(ValueError, match='n_splits=2 or more'):
            InSampleSplit(n_splits=1)


class TestDefaultFold:
    """The single fold taken from the end of the data."""

    def test_test_size_derived_from_n_splits(self, cv_frame_12):
        """test_size = 12 // (3 + 1) = 3 last dates; training is the whole series."""
        assert as_lists(collect_folds(InSampleSplit(n_splits=3), cv_frame_12)) == [(_upto(12), [9, 10, 11])]

    def test_default_n_splits(self, cv_frame_12):
        """With the default n_splits=5, test_size = 12 // 6 = 2."""
        assert as_lists(collect_folds(InSampleSplit(), cv_frame_12)) == [(_upto(12), [10, 11])]

    def test_explicit_test_size(self, cv_frame_12):
        """test_size sets the length of the final test period."""
        assert as_lists(collect_folds(InSampleSplit(test_size=5), cv_frame_12)) == [(_upto(12), [7, 8, 9, 10, 11])]

    @pytest.mark.parametrize('n_splits', [2, 3, 7])
    def test_always_one_fold(self, cv_frame_12, n_splits):
        """Whatever n_splits, an in-sample split yields one fold."""
        assert len(collect_folds(InSampleSplit(n_splits=n_splits), cv_frame_12)) == 1

    def test_max_train_size(self, cv_frame_12):
        """max_train_size keeps that many dates, ending at the last test date."""
        folds = collect_folds(InSampleSplit(test_size=3, max_train_size=5), cv_frame_12)
        assert as_lists(folds) == [([7, 8, 9, 10, 11], [9, 10, 11])]

    def test_max_train_size_smaller_than_the_test(self, cv_frame_12):
        """Training never holds less than the test period: here exactly the test period."""
        folds = collect_folds(InSampleSplit(test_size=5, max_train_size=3), cv_frame_12)
        assert as_lists(folds) == [([7, 8, 9, 10, 11], [7, 8, 9, 10, 11])]

    def test_test_size_larger_than_the_data(self, cv_frame_12):
        """A test period longer than the data covers all of it."""
        assert as_lists(collect_folds(InSampleSplit(test_size=20), cv_frame_12)) == [(_upto(12), _upto(12))]


class TestExplicitTestIndices:
    """One test start with test_size, or several test points."""

    def test_single_date(self, cv_frame_12):
        """One date opens a test period of test_size; training stops at its end."""
        folds = collect_folds(InSampleSplit(test_indices=['2024-04-30'], test_size=3), cv_frame_12)
        assert as_lists(folds) == [(_upto(6), [3, 4, 5])]

    @pytest.mark.parametrize(
        'indices', [['2024-04-30'], [pd.Timestamp('2024-04-30')], [3], 3],
        ids=['string', 'timestamp', 'position', 'scalar'],
    )
    def test_single_index_forms(self, cv_frame_12, indices):
        """A date string, timestamp or position designate the same single test row."""
        assert as_lists(collect_folds(InSampleSplit(test_indices=indices), cv_frame_12)) == [(_upto(4), [3])]

    def test_single_date_with_max_train_size(self, cv_frame_12):
        """The window keeps max(max_train_size, test length) dates up to the test end."""
        folds = collect_folds(InSampleSplit(test_indices=[3], test_size=3, max_train_size=2), cv_frame_12)
        assert as_lists(folds) == [([3, 4, 5], [3, 4, 5])]

    def test_single_date_cut_at_the_last_date(self, cv_frame_12):
        """A test period running past the end is truncated."""
        folds = collect_folds(InSampleSplit(test_indices=['2024-11-30'], test_size=3), cv_frame_12)
        assert as_lists(folds) == [(_upto(12), [10, 11])]

    def test_several_dates_form_one_test_set(self, cv_frame_12):
        """Several dates are tested together, one row each: test_size is not used."""
        splitter = InSampleSplit(test_indices=['2024-02-29', '2024-08-31', '2024-11-30'], test_size=3)
        assert as_lists(collect_folds(splitter, cv_frame_12)) == [(_upto(11), [1, 7, 10])]

    def test_several_positions_are_sorted(self, cv_frame_12):
        """Positions given out of order are tested in date order."""
        assert as_lists(collect_folds(InSampleSplit(test_indices=[10, 1, 7]), cv_frame_12)) == [(_upto(11), [1, 7, 10])]

    def test_several_points_with_max_train_size_stay_in_training(self, cv_frame_12):
        """The window covers at least the span of the test points, so all stay trained on."""
        # Valeur d'or : étendue 10 - 1 + 1 = 10 > max_train_size = 4 : apprentissage 1..10
        folds = collect_folds(InSampleSplit(test_indices=[1, 7, 10], max_train_size=4), cv_frame_12)
        assert as_lists(folds) == [(list(range(1, 11)), [1, 7, 10])]
        assert_in_sample(cv_frame_12, folds)

    def test_several_points_with_a_wider_window(self, cv_frame_12):
        """A window wider than the span of the test points keeps max_train_size rows."""
        folds = collect_folds(InSampleSplit(test_indices=[8, 10], max_train_size=5), cv_frame_12)
        assert as_lists(folds) == [([6, 7, 8, 9, 10], [8, 10])]

    def test_repeated_test_points_are_tested_once(self, cv_frame_12):
        """A test point given twice is one test row."""
        assert as_lists(collect_folds(InSampleSplit(test_indices=[3, 7, 3]), cv_frame_12)) == [(_upto(8), [3, 7])]

    @pytest.mark.parametrize('indices', [[True], [2, True]], ids=['boolean', 'mixed'])
    def test_booleans_are_refused(self, cv_frame_12, indices):
        """Booleans are not positions."""
        with pytest.raises(ValueError, match='got the boolean True'):
            collect_folds(InSampleSplit(test_indices=indices), cv_frame_12)

    @pytest.mark.parametrize(
        ('indices', 'message'),
        [
            ([3, 12], 'test_indices contains indices beyond data length'),
            ([-1], 'test_indices contains negative indices'),
            (['2030-01-31'], 'Time series index 2030-01-31 not found in data'),
        ],
        ids=['beyond', 'negative', 'unknown-date'],
    )
    def test_invalid_indices_are_refused(self, cv_frame_12, indices, message):
        """Out-of-range, negative or unknown indices raise a ValueError."""
        with pytest.raises(ValueError, match=message):
            collect_folds(InSampleSplit(test_indices=indices), cv_frame_12)

    def test_empty_list_falls_back_to_the_default_fold(self, cv_frame_12):
        """An empty list of test indices behaves like None."""
        explicit = collect_folds(InSampleSplit(test_indices=[], test_size=2), cv_frame_12)
        assert as_lists(explicit) == [(_upto(12), [10, 11])]


class TestTarget:
    """Validation of y by split."""

    def test_y_with_another_index_is_refused_at_the_call(self, cv_frame_12):
        """The index check happens when split is called."""
        with pytest.raises(ValueError, match="'X' and 'y' should have the same indexes"):
            InSampleSplit().split(cv_frame_12, pd.Series(range(12)))

    def test_y_without_index_is_not_checked(self, cv_frame_12):
        """An array target is accepted as is."""
        assert len(collect_folds(InSampleSplit(test_size=2), cv_frame_12, np.arange(12))) == 1


class TestDegenerateData:
    """Empty data, a single observation, a test size of zero."""

    def test_single_observation_with_test_size(self):
        """One row with test_size=1 is both the training and the test set."""
        assert as_lists(collect_folds(InSampleSplit(test_size=1), pd.Series([1.0]))) == [([0], [0])]

    @pytest.mark.parametrize(
        ('X', 'kwargs', 'message'),
        [
            (pd.Series([], dtype=float), {'test_size': 1}, 'Cannot split an empty dataset: number of samples=0'),
            (pd.Series([], dtype=float), {}, 'Cannot split an empty dataset: number of samples=0'),
            (pd.Series([1.0, 2.0, 3.0]), {},
             'Empty test period: test_size=0 for number of samples=3 with n_splits=5'),
            (pd.Series([1.0]), {}, 'Empty test period: test_size=0 for number of samples=1 with n_splits=5'),
            (pd.Series(np.arange(12.0)), {'test_size': 0}, 'test_size must be a positive integer or None, got 0'),
        ],
        ids=['empty', 'empty-default', 'fewer-rows-than-folds', 'single-default', 'test_size-zero'],
    )
    def test_empty_test_period_is_refused_with_a_clear_message(self, X, kwargs, message):
        """A split without test rows is refused with a message naming the cause."""
        with pytest.raises(ValueError, match=message):
            collect_folds(InSampleSplit(**kwargs), X)


class TestInvalidParameters:
    """Parameter values that cannot describe a valid in-sample split."""

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'test_size': -1}, 'test_size must be a positive integer or None, got -1'),
            ({'test_size': 1.5}, 'test_size must be a positive integer or None, got 1.5'),
            ({'max_train_size': 0}, 'max_train_size must be a positive integer or None, got 0'),
            ({'max_train_size': False}, 'max_train_size must be a positive integer or None, got False'),
        ],
        ids=['test_size-negative', 'test_size-float', 'max_train_size-zero', 'max_train_size-bool'],
    )
    def test_is_refused_with_its_name(self, cv_frame_12, kwargs, message):
        """An invalid value raises a ValueError naming the parameter, when split is called."""
        splitter = InSampleSplit(**kwargs)
        with pytest.raises(ValueError, match=message):
            splitter.split(cv_frame_12)

    def test_count_is_always_one(self):
        """get_n_splits returns 1, the number of folds of an in-sample split."""
        assert InSampleSplit(n_splits=7, test_indices=[1, 2, 3]).get_n_splits() == 1


class TestPanelGroups:
    """Panel branches of the base splitter, reached with explicit groups.

    Entity ``A`` holds positions 0-5 and ``B`` positions 6-11, both on the
    month-ends 2024-01-31 to 2024-06-30.
    """

    @staticmethod
    def _groups(X: pd.DataFrame) -> np.ndarray:
        """Return the entity of each row of the panel."""
        return X.index.get_level_values(0).to_numpy()

    def test_default_fold_is_the_last_date(self, cv_panel_2x6):
        """Without test_size, the default test is the last date of each entity."""
        # Valeur d'or : dernière date (positions 5 et 11) ; fenêtre max(3, 1) = 3 dates
        folds = collect_folds(InSampleSplit(max_train_size=3), cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([3, 4, 5, 9, 10, 11], [5, 11])]

    def test_default_fold_holds_the_last_test_size_dates(self, cv_panel_2x6):
        """test_size sets the number of final dates tested for each entity, as on a series."""
        # Valeur d'or : deux dernières dates par entité ; fenêtre max(3, 2) = 3 dates
        folds = collect_folds(InSampleSplit(test_size=2, max_train_size=3), cv_panel_2x6,
                              groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([3, 4, 5, 9, 10, 11], [4, 5, 10, 11])]

    def test_several_dates(self, cv_panel_2x6):
        """Several dates are tested for every entity, in one fold."""
        splitter = InSampleSplit(test_indices=['2024-02-29', '2024-05-31'])
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], [1, 4, 7, 10])]
        assert_in_sample(cv_panel_2x6, folds)

    def test_single_date_with_test_size_and_window(self, cv_panel_2x6):
        """One date opens test_size dates per entity; the window ends at the test end."""
        splitter = InSampleSplit(test_indices=['2024-03-31'], test_size=2, max_train_size=3)
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([1, 2, 3, 7, 8, 9], [2, 3, 8, 9])]

    def test_several_positions(self, cv_panel_2x6):
        """Positions on every entity are tested together."""
        folds = collect_folds(InSampleSplit(test_indices=[2, 8]), cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([0, 1, 2, 6, 7, 8], [2, 8])]

    def test_entity_date_pairs(self, cv_panel_2x6):
        """(entity, date) pairs select a different date per entity."""
        dates = pd.date_range('2024-01-31', periods=6, freq='ME')
        splitter = InSampleSplit(test_indices=[('A', dates[2]), ('B', dates[3])])
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([0, 1, 2, 6, 7, 8, 9], [2, 9])]

    def test_several_dates_with_one_missing_for_an_entity(self, cv_panel_2x6):
        """A date missing for one entity is warned about and skipped for it."""
        X = cv_panel_2x6.drop(('B', pd.Timestamp('2024-06-30')))
        splitter = InSampleSplit(test_indices=['2024-05-31', '2024-06-30'])
        with pytest.warns(UserWarning, match="Cannot find test period '2024-06-30' for entity 'B'"):
            folds = collect_folds(splitter, X, groups=self._groups(X))
        assert as_lists(folds) == [(_upto(11), [4, 5, 10])]

    @pytest.mark.parametrize(
        'kwargs', [{'test_indices': ['2024-06-30']}, {}], ids=['single-date', 'default'],
    )
    def test_entity_without_the_test_date(self, cv_panel_2x6, kwargs):
        """An entity lacking the only test date is warned about, then the split fails."""
        X = cv_panel_2x6.drop(('B', pd.Timestamp('2024-06-30')))
        with pytest.warns(UserWarning, match="Cannot find test period '2024-06-30.*' for entity 'B'"):
            with pytest.raises(ValueError, match='Cannot find test indices for the group : B'):
                collect_folds(InSampleSplit(**kwargs), X, groups=self._groups(X))

    @pytest.mark.parametrize(
        ('indices', 'expected'),
        [
            ([2], [([0, 1, 2, 3], [2, 3])]),
            ([('B', '2024-05-31')], [([6, 7, 8, 9, 10, 11], [10, 11])]),
        ],
        ids=['position', 'tuple'],
    )
    def test_single_entity_specific_index(self, cv_panel_2x6, indices, expected):
        """A single position or tuple tests test_size rows of its entity, cut at its end."""
        folds = collect_folds(InSampleSplit(test_indices=indices, test_size=2), cv_panel_2x6,
                              groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == expected

    def test_tuples_of_one_entity(self, cv_panel_2x6):
        """Several tuples of the same entity test and train that entity only."""
        splitter = InSampleSplit(test_indices=[('A', '2024-02-29'), ('A', '2024-04-30')])
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([0, 1, 2, 3], [1, 3])]

    @pytest.mark.parametrize(
        'kwargs',
        [{}, {'test_indices': ['2024-03-31']}, {'test_indices': ['2024-03-31', '2024-04-30']}],
        ids=['default', 'single-date', 'several-dates'],
    )
    def test_groups_inconsistent_with_the_index_are_refused(self, cv_panel_2x6, kwargs):
        """Groups that do not match the entity level are refused."""
        groups = np.char.lower(self._groups(cv_panel_2x6).astype(str))
        with pytest.raises(ValueError, match='groups must match the first index level'):
            collect_folds(InSampleSplit(**kwargs), cv_panel_2x6, groups=groups)
