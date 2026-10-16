"""Tests of the ``OutOfSampleSplit`` cross-validator, golden values on small data.

Covers ``OutOfSampleSplit`` (and ``TSOutOfSampleSplit`` for the parameter
contract): parameters, default folds (expanding and rolling windows, ``gap``,
``test_size``, ``n_splits``), explicit ``test_indices`` (dates, timestamps,
positions, scalars), ``y`` and ``groups`` handling, degenerate data and
invalid parameters, and the panel branches reached with explicit ``groups``.
Expected folds are written by hand on 12 month-end dates whose value equals
the row position.
"""
# Importation des modules
import datetime

import numpy as np
import pandas as pd
import pytest

from tests.support.splits import as_lists, assert_no_leakage, collect_folds
from tsforecast.crossvals.base import OutOfSampleSplit
from tsforecast.crossvals.time_series import TSOutOfSampleSplit


# Raccourcis de lecture : positions 0..k-1
def _upto(k: int) -> list[int]:
    """Return the positions ``0 .. k-1``."""
    return list(range(k))


class TestParameters:
    """Constructor contract shared by the out-of-sample splitters."""

    @pytest.mark.parametrize('cls', [OutOfSampleSplit, TSOutOfSampleSplit], ids=['base', 'ts'])
    def test_defaults(self, cls):
        """Defaults: five splits, no explicit test indices, no limit, no gap."""
        splitter = cls()
        assert (splitter.n_splits, splitter.test_indices, splitter.max_train_size,
                splitter.test_size, splitter.gap) == (5, None, None, None, 0)

    @pytest.mark.parametrize('cls', [OutOfSampleSplit, TSOutOfSampleSplit], ids=['base', 'ts'])
    def test_values_are_stored_as_given(self, cls):
        """Constructor arguments are stored unchanged."""
        dates = ['2020-01-05']
        splitter = cls(n_splits=3, test_indices=dates, max_train_size=50, test_size=10, gap=2)
        assert (splitter.n_splits, splitter.test_indices, splitter.max_train_size,
                splitter.test_size, splitter.gap) == (3, dates, 50, 10, 2)
        assert splitter.test_indices is dates

    def test_parameters_after_n_splits_are_keyword_only(self):
        """Only n_splits may be passed positionally."""
        with pytest.raises(TypeError):
            OutOfSampleSplit(3, None)

    def test_repr_lists_the_parameters(self):
        """The repr shows every constructor parameter."""
        text = repr(TSOutOfSampleSplit(n_splits=3, gap=1))
        assert text.startswith('TSOutOfSampleSplit(') and 'gap=1' in text and 'n_splits=3' in text

    @pytest.mark.parametrize('n_splits', [1, 0, -1])
    def test_fewer_than_two_splits_is_refused(self, n_splits):
        """n_splits below 2 is refused at construction."""
        with pytest.raises(ValueError, match='n_splits=2 or more'):
            OutOfSampleSplit(n_splits=n_splits)

    @pytest.mark.parametrize('n_splits', [2.5, '3'], ids=['float', 'str'])
    def test_non_integer_n_splits_is_refused(self, n_splits):
        """n_splits must be an integer."""
        with pytest.raises(ValueError, match='must be of Integral type'):
            OutOfSampleSplit(n_splits=n_splits)


class TestDefaultFolds:
    """Folds taken from the end of the data when no test indices are given."""

    def test_expanding_window(self, cv_frame_12):
        """Three folds of three dates; training grows from the first date."""
        # Valeur d'or : test_size = 12 // (3 + 1) = 3 ; débuts de test 12 - 3*3 = 3, 6, 9
        folds = collect_folds(OutOfSampleSplit(n_splits=3), cv_frame_12)
        assert as_lists(folds) == [
            (_upto(3), [3, 4, 5]),
            (_upto(6), [6, 7, 8]),
            (_upto(9), [9, 10, 11]),
        ]

    def test_explicit_test_size(self, cv_frame_12):
        """test_size fixes the test length; folds still end at the last date."""
        # Valeur d'or : débuts 12 - 3*2 = 6, 8, 10
        folds = collect_folds(OutOfSampleSplit(n_splits=3, test_size=2), cv_frame_12)
        assert as_lists(folds) == [(_upto(6), [6, 7]), (_upto(8), [8, 9]), (_upto(10), [10, 11])]

    def test_default_test_size_rounds_down(self, cv_frame_12):
        """With 12 rows and 4 splits, test_size = 12 // 5 = 2; the first rows are never tested."""
        folds = collect_folds(OutOfSampleSplit(n_splits=4), cv_frame_12)
        assert [test for _, test in as_lists(folds)] == [[4, 5], [6, 7], [8, 9], [10, 11]]

    def test_gap(self, cv_frame_12):
        """gap=2 drops the two dates preceding each test set from training."""
        folds = collect_folds(OutOfSampleSplit(n_splits=3, gap=2), cv_frame_12)
        assert [train for train, _ in as_lists(folds)] == [[0], _upto(4), _upto(7)]

    def test_rolling_window(self, cv_frame_12):
        """max_train_size=2 keeps the two dates just before each test set."""
        folds = collect_folds(OutOfSampleSplit(n_splits=3, max_train_size=2), cv_frame_12)
        assert [train for train, _ in as_lists(folds)] == [[1, 2], [4, 5], [7, 8]]

    def test_rolling_window_longer_than_the_history(self, cv_frame_12):
        """With less history than max_train_size, the first fold keeps all of it."""
        # Valeur d'or : historique de 3 dates < 4 au premier pli ; fenêtres de 4 ensuite
        folds = collect_folds(OutOfSampleSplit(n_splits=3, max_train_size=4), cv_frame_12)
        assert [train for train, _ in as_lists(folds)] == [[0, 1, 2], [2, 3, 4, 5], [5, 6, 7, 8]]

    def test_rolling_window_with_gap(self, cv_frame_12):
        """The rolling window ends gap dates before the test set."""
        # Valeur d'or : fins d'apprentissage 3-1=2, 6-1=5, 9-1=8 ; fenêtres de 2
        folds = collect_folds(OutOfSampleSplit(n_splits=3, gap=1, max_train_size=2), cv_frame_12)
        assert [train for train, _ in as_lists(folds)] == [[0, 1], [3, 4], [6, 7]]

    def test_largest_number_of_splits(self, cv_frame_12):
        """n_splits = n - 1 tests every date but the first, one at a time."""
        folds = collect_folds(OutOfSampleSplit(n_splits=11), cv_frame_12)
        assert as_lists(folds) == [(_upto(k), [k]) for k in range(1, 12)]

    def test_gap_longer_than_the_history_is_refused(self, cv_frame_12):
        """A fold without training data is refused, naming its first test row and the gap."""
        # Valeur d'or : premier pli, test à partir de la ligne 3, fin d'apprentissage 3 - 7 < 0
        with pytest.raises(ValueError, match=r'No training data before the test set starting at row 3 \(gap=7\)'):
            collect_folds(OutOfSampleSplit(n_splits=3, gap=7), cv_frame_12)

    def test_gap_leaving_one_training_row(self, cv_frame_12):
        """The largest gap still leaving a training row is accepted."""
        # Valeur d'or : fins d'apprentissage 3-2 = 1, 6-2 = 4, 9-2 = 7
        folds = collect_folds(OutOfSampleSplit(n_splits=3, gap=2), cv_frame_12)
        assert [train for train, _ in as_lists(folds)][0] == [0]

    def test_n_splits_times_test_size_equal_to_the_data_is_refused(self, cv_frame_12):
        """Folds covering every row would leave no training data: refused."""
        with pytest.raises(ValueError, match='Too many splits=3 for number of samples=12 with test_size=4'):
            collect_folds(OutOfSampleSplit(n_splits=3, test_size=4), cv_frame_12)

    def test_too_many_folds_is_refused(self, cv_frame_12):
        """n_splits + 1 segments cannot exceed the number of rows."""
        with pytest.raises(ValueError, match='Cannot have number of folds=13 greater than the number of samples=12'):
            collect_folds(OutOfSampleSplit(n_splits=12), cv_frame_12)

    def test_test_size_larger_than_the_data_is_refused(self, cv_frame_12):
        """A test set longer than the data is refused."""
        with pytest.raises(ValueError, match='Too many splits=2 for number of samples=12 with test_size=13'):
            collect_folds(OutOfSampleSplit(n_splits=2, test_size=13), cv_frame_12)


class TestInvalidParameters:
    """Parameter values that cannot describe a valid split."""

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'gap': -1}, 'gap must be a non-negative integer, got -1'),
            ({'gap': 1.5}, 'gap must be a non-negative integer, got 1.5'),
            ({'gap': True}, 'gap must be a non-negative integer, got True'),
            ({'test_size': 0}, 'test_size must be a positive integer or None, got 0'),
            ({'test_size': -1}, 'test_size must be a positive integer or None, got -1'),
            ({'test_size': 2.0}, 'test_size must be a positive integer or None, got 2.0'),
            ({'max_train_size': 0}, 'max_train_size must be a positive integer or None, got 0'),
            ({'max_train_size': -1}, 'max_train_size must be a positive integer or None, got -1'),
        ],
        ids=['gap-negative', 'gap-float', 'gap-bool', 'test_size-zero', 'test_size-negative',
             'test_size-float', 'max_train_size-zero', 'max_train_size-negative'],
    )
    def test_is_refused_with_its_name(self, cv_frame_12, kwargs, message):
        """An invalid value raises a ValueError naming the parameter, when split is called."""
        splitter = OutOfSampleSplit(n_splits=3, **kwargs)
        with pytest.raises(ValueError, match=message):
            splitter.split(cv_frame_12)

    def test_numpy_integers_are_accepted(self, cv_frame_12):
        """Integers built by numpy are valid parameter values."""
        splitter = OutOfSampleSplit(n_splits=3, gap=np.int64(2), test_size=np.int64(3), max_train_size=np.int64(2))
        assert [train for train, _ in as_lists(collect_folds(splitter, cv_frame_12))] == [[0], [2, 3], [5, 6]]


class TestExplicitTestIndices:
    """Folds starting at the given test dates or positions."""

    def test_date_strings(self, cv_frame_12):
        """Each date opens a fold of test_size rows; training stops just before."""
        splitter = OutOfSampleSplit(test_indices=['2024-04-30', '2024-08-31'], test_size=2)
        assert as_lists(collect_folds(splitter, cv_frame_12)) == [
            (_upto(3), [3, 4]),
            (_upto(7), [7, 8]),
        ]

    @pytest.mark.parametrize(
        'indices',
        [
            [pd.Timestamp('2024-04-30'), pd.Timestamp('2024-08-31')],
            [datetime.datetime(2024, 4, 30), datetime.datetime(2024, 8, 31)],
            [3, 7],
            np.array([3, 7]),
            (3, 7),
        ],
        ids=['timestamps', 'datetimes', 'positions', 'position-array', 'position-tuple'],
    )
    def test_equivalent_forms(self, cv_frame_12, indices):
        """Timestamps, datetimes and positions designate the same folds as date strings."""
        splitter = OutOfSampleSplit(test_indices=indices, test_size=2)
        assert [test for _, test in as_lists(collect_folds(splitter, cv_frame_12))] == [[3, 4], [7, 8]]

    def test_scalar_position(self, cv_frame_12):
        """A single integer is one test position."""
        assert as_lists(collect_folds(OutOfSampleSplit(test_indices=5), cv_frame_12)) == [(_upto(5), [5])]

    def test_default_test_size_is_one(self, cv_frame_12):
        """Without test_size, each fold tests a single row."""
        folds = collect_folds(OutOfSampleSplit(test_indices=['2024-04-30']), cv_frame_12)
        assert as_lists(folds) == [(_upto(3), [3])]

    def test_n_splits_is_ignored(self, cv_frame_12):
        """The number of folds is the number of test indices, not n_splits."""
        splitter = OutOfSampleSplit(n_splits=2, test_indices=[3, 6, 9])
        assert len(collect_folds(splitter, cv_frame_12)) == 3

    def test_folds_follow_the_chronological_order(self, cv_frame_12):
        """Test indices given out of order produce folds in date order."""
        folds = collect_folds(OutOfSampleSplit(test_indices=[8, 3]), cv_frame_12)
        assert [test for _, test in as_lists(folds)] == [[3], [8]]

    def test_test_set_is_cut_at_the_last_date(self, cv_frame_12):
        """A test set running past the end is truncated."""
        folds = collect_folds(OutOfSampleSplit(test_indices=['2024-11-30'], test_size=3), cv_frame_12)
        assert as_lists(folds) == [(_upto(10), [10, 11])]

    def test_test_sets_may_overlap(self, cv_frame_12):
        """Close origins with a long test_size give overlapping test sets."""
        folds = collect_folds(OutOfSampleSplit(test_indices=[3, 4], test_size=3), cv_frame_12)
        assert as_lists(folds) == [(_upto(3), [3, 4, 5]), (_upto(4), [4, 5, 6])]

    def test_repeated_index_repeats_the_fold(self, cv_frame_12):
        """A repeated test index yields the same fold twice."""
        folds = collect_folds(OutOfSampleSplit(test_indices=[3, 3]), cv_frame_12)
        assert as_lists(folds) == [(_upto(3), [3]), (_upto(3), [3])]

    @pytest.mark.parametrize(
        'indices', [[True], [1, False], np.array([True, False]), True, np.bool_(True)],
        ids=['list', 'mixed-with-integers', 'boolean-array', 'scalar', 'numpy-scalar'],
    )
    def test_booleans_are_refused(self, cv_frame_12, indices):
        """Booleans are not positions, although Python counts them as integers."""
        with pytest.raises(ValueError, match='test_indices must hold integer positions, dates or .* got the boolean'):
            collect_folds(OutOfSampleSplit(test_indices=indices), cv_frame_12)

    def test_booleans_are_refused_by_get_n_splits(self):
        """The count of folds refuses booleans too."""
        with pytest.raises(ValueError, match='got the boolean True'):
            OutOfSampleSplit(test_indices=[True]).get_n_splits()

    def test_numpy_integer_positions(self, cv_frame_12):
        """Numpy integers are positions like Python integers."""
        folds = collect_folds(OutOfSampleSplit(test_indices=[np.int32(3), np.int64(7)]), cv_frame_12)
        assert [test for _, test in as_lists(folds)] == [[3], [7]]

    def test_positions_and_dates_cannot_be_mixed(self, cv_frame_12):
        """A list mixing positions and dates is refused."""
        with pytest.raises(ValueError, match='test_indices must not mix integer positions, dates'):
            collect_folds(OutOfSampleSplit(test_indices=[3, '2024-08-31']), cv_frame_12)

    def test_gap(self, cv_frame_12):
        """gap applies before each explicit test date."""
        folds = collect_folds(OutOfSampleSplit(test_indices=['2024-06-30'], gap=2), cv_frame_12)
        assert as_lists(folds) == [(_upto(3), [5])]

    def test_rolling_window(self, cv_frame_12):
        """max_train_size applies before each explicit test date."""
        folds = collect_folds(OutOfSampleSplit(test_indices=[9], max_train_size=3), cv_frame_12)
        assert as_lists(folds) == [([6, 7, 8], [9])]

    def test_first_position_is_refused(self, cv_frame_12):
        """Testing the first date leaves no training data: refused."""
        with pytest.raises(ValueError, match='No training data before the test set starting at row 0'):
            collect_folds(OutOfSampleSplit(test_indices=[0, 4]), cv_frame_12)

    def test_gap_reaching_the_first_date_is_refused(self, cv_frame_12):
        """A gap that swallows the whole history of an explicit test date is refused."""
        with pytest.raises(ValueError, match=r'starting at row 3 \(gap=3\)'):
            collect_folds(OutOfSampleSplit(test_indices=['2024-04-30'], gap=3), cv_frame_12)

    def test_empty_list_falls_back_to_the_default_folds(self, cv_frame_12):
        """An empty list of test indices behaves like None."""
        explicit = collect_folds(OutOfSampleSplit(n_splits=3, test_indices=[]), cv_frame_12)
        default = collect_folds(OutOfSampleSplit(n_splits=3), cv_frame_12)
        assert as_lists(explicit) == as_lists(default)

    @pytest.mark.parametrize(
        ('indices', 'message'),
        [
            ([3, 12], 'test_indices contains indices beyond data length'),
            ([-1], 'test_indices contains negative indices'),
            (['2030-01-31'], 'Time series index 2030-01-31 not found in data'),
            ([('A', '2024-01-31')], 'Index type mismatch: time series data cannot use tuple indices'),
        ],
        ids=['beyond', 'negative', 'unknown-date', 'tuple-on-series'],
    )
    def test_invalid_indices_are_refused(self, cv_frame_12, indices, message):
        """Out-of-range, negative, unknown or tuple indices raise a ValueError."""
        with pytest.raises(ValueError, match=message):
            collect_folds(OutOfSampleSplit(test_indices=indices), cv_frame_12)

    def test_dates_on_an_array_are_refused(self, cv_frame_12):
        """Dates cannot be resolved without a pandas index."""
        with pytest.raises(ValueError, match='Cannot use string/date/tuple indices without pandas index'):
            collect_folds(OutOfSampleSplit(test_indices=['2024-04-30']), cv_frame_12.to_numpy())

    @pytest.mark.parametrize(
        'make_indices',
        [
            lambda X: X.index[[3, 7]],
            lambda X: pd.Series(X.index[[3, 7]]),
            lambda X: [np.datetime64('2024-04-30'), np.datetime64('2024-08-31')],
            lambda X: X.index[[3, 7]].to_numpy(),
            lambda X: [datetime.date(2024, 4, 30), datetime.date(2024, 8, 31)],
        ],
        ids=['DatetimeIndex', 'Series', 'numpy-datetime64', 'datetime64-array', 'datetime.date'],
    )
    def test_other_date_forms_are_resolved(self, cv_frame_12, make_indices):
        """Every common representation of dates designates the same folds."""
        splitter = OutOfSampleSplit(test_indices=make_indices(cv_frame_12))
        assert [test for _, test in as_lists(collect_folds(splitter, cv_frame_12))] == [[3], [7]]
        assert splitter.get_n_splits() == 2

    @pytest.mark.parametrize(
        'single', ['2024-04-30', pd.Timestamp('2024-04-30'), np.datetime64('2024-04-30')],
        ids=['string', 'timestamp', 'datetime64'],
    )
    def test_single_date_is_one_test_start(self, cv_frame_12, single):
        """A single date designates one test date, like a single position."""
        splitter = OutOfSampleSplit(test_indices=single)
        assert [test for _, test in as_lists(collect_folds(splitter, cv_frame_12))] == [[3]]
        assert splitter.get_n_splits() == 1

    def test_dates_on_a_time_zone_aware_index(self, cv_frame_12):
        """A naive date is read in the time zone of the index."""
        X = cv_frame_12.tz_localize('Europe/Paris')
        folds = collect_folds(OutOfSampleSplit(test_indices=[datetime.date(2024, 4, 30)]), X)
        assert [test for _, test in as_lists(folds)] == [[3]]

    @pytest.mark.parametrize('indices', [[3.0], [np.float64(3.0)], [None]], ids=['float', 'numpy-float', 'none'])
    def test_other_types_are_refused(self, cv_frame_12, indices):
        """A float or None is neither a position nor a date: refused, naming the type."""
        with pytest.raises(ValueError, match=r'test_indices must hold integer positions, .* of type'):
            collect_folds(OutOfSampleSplit(test_indices=indices), cv_frame_12)

    def test_partial_date_matching_several_rows_is_refused(self):
        """A month written without its day designates several daily rows: refused."""
        X = pd.Series(np.arange(60.0), index=pd.date_range('2024-01-01', periods=60, freq='D'))
        with pytest.raises(ValueError, match='Time series index 2024-02 matches 29 rows, expected one'):
            collect_folds(OutOfSampleSplit(test_indices=['2024-02']), X)


class TestTargetAndGroups:
    """Validation of y and groups by split."""

    def test_y_with_the_same_index_is_accepted(self, cv_frame_12):
        """A target on the same index gives the same folds as no target."""
        y = cv_frame_12['v'] * 2
        splitter = OutOfSampleSplit(n_splits=3)
        assert as_lists(collect_folds(splitter, cv_frame_12, y)) == as_lists(collect_folds(splitter, cv_frame_12))

    def test_y_with_another_index_is_refused_at_the_call(self, cv_frame_12):
        """The index check happens when split is called, before any iteration."""
        y = pd.Series(range(12))
        with pytest.raises(ValueError, match="'X' and 'y' should have the same indexes"):
            OutOfSampleSplit(n_splits=3).split(cv_frame_12, y)

    def test_y_without_index_is_not_checked(self, cv_frame_12):
        """An array target is not compared with X, even with another length."""
        folds = collect_folds(OutOfSampleSplit(n_splits=3), cv_frame_12, np.arange(5))
        assert len(folds) == 3

    @pytest.mark.parametrize('kind', ['Series', 'DataFrame'])
    def test_groups_require_a_multiindex(self, cv_frame_12, kind):
        """Groups on data without a MultiIndex are refused."""
        X = cv_frame_12['v'] if kind == 'Series' else cv_frame_12
        with pytest.raises(ValueError, match='Panel data requires MultiIndex with at least 2 levels'):
            collect_folds(OutOfSampleSplit(n_splits=3), X, groups=np.zeros(12))


class TestDegenerateData:
    """Empty data and a single observation."""

    @pytest.mark.parametrize(
        'X',
        [pd.Series([], dtype=float), pd.DataFrame({'v': []}), np.empty((0, 1))],
        ids=['Series', 'DataFrame', 'ndarray'],
    )
    def test_empty_data_is_refused(self, X):
        """Empty data cannot be split."""
        with pytest.raises(ValueError, match='Cannot have number of folds=3 greater than the number of samples=0'):
            collect_folds(OutOfSampleSplit(n_splits=2, test_size=1), X)

    def test_single_observation_is_refused(self):
        """One row cannot be split into two folds."""
        with pytest.raises(ValueError, match='Cannot have number of folds=3 greater than the number of samples=1'):
            collect_folds(OutOfSampleSplit(n_splits=2, test_size=1), pd.Series([1.0]))

    def test_single_observation_as_explicit_test_index(self):
        """Testing the only row leaves no training data: refused."""
        with pytest.raises(ValueError, match='No training data before the test set starting at row 0'):
            collect_folds(OutOfSampleSplit(test_indices=[0]), pd.Series([1.0]))


class TestPanelGroups:
    """Panel branches of the base splitter, reached with explicit groups.

    The panel holds entity ``A`` on positions 0-5 and ``B`` on 6-11, both on
    the same six month-ends; training and test are built entity by entity.
    """

    @staticmethod
    def _groups(X: pd.DataFrame) -> np.ndarray:
        """Return the entity of each row of the panel."""
        return X.index.get_level_values(0).to_numpy()

    def test_default_folds(self, cv_panel_2x6):
        """Each fold tests the same date for every entity; test_size defaults to 1."""
        # Valeur d'or : 6 dates, débuts de test 6 - 2*1 + i = 4, 5 (dates de mai et juin)
        folds = collect_folds(OutOfSampleSplit(n_splits=2), cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [
            ([0, 1, 2, 3, 6, 7, 8, 9], [4, 10]),
            ([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], [5, 11]),
        ]

    def test_gap_and_rolling_window_per_entity(self, cv_panel_2x6):
        """gap and max_train_size apply inside each entity."""
        # Valeur d'or : tests aux dates 2-3 puis 4-5 ; fin d'apprentissage = début - 1 ;
        # fenêtre de 2 seulement quand l'historique le permet
        splitter = OutOfSampleSplit(n_splits=2, test_size=2, gap=1, max_train_size=2)
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([0, 6], [2, 3, 8, 9]), ([1, 2, 7, 8], [4, 5, 10, 11])]

    def test_gap_longer_than_an_entity_history_is_refused(self, cv_panel_2x6):
        """A gap leaving an entity without training data is refused, naming the entity."""
        with pytest.raises(ValueError, match=r"No training data for the group 'A' before its test set \(gap=5\)"):
            collect_folds(OutOfSampleSplit(n_splits=2, gap=5), cv_panel_2x6, groups=self._groups(cv_panel_2x6))

    def test_too_many_splits_for_the_dates(self, cv_panel_2x6):
        """The number of dates, not rows, bounds the number of splits."""
        with pytest.raises(ValueError, match='Too many splits=6 for number of time points=6 with test_size=1'):
            collect_folds(OutOfSampleSplit(n_splits=6), cv_panel_2x6, groups=self._groups(cv_panel_2x6))

    def test_explicit_dates(self, cv_panel_2x6):
        """A date selects that date for every entity."""
        splitter = OutOfSampleSplit(test_indices=['2024-03-31', '2024-05-31'], test_size=2)
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([0, 1, 6, 7], [2, 3, 8, 9]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5, 10, 11])]
        assert_no_leakage(cv_panel_2x6, folds)

    def test_explicit_positions_on_a_single_entity(self, cv_panel_2x6):
        """Positions open folds cut at the end of their entity."""
        X = cv_panel_2x6.loc[['A']]
        splitter = OutOfSampleSplit(test_indices=[2, 4], test_size=3)
        folds = collect_folds(splitter, X, groups=self._groups(X))
        assert as_lists(folds) == [([0, 1], [2, 3, 4]), (_upto(4), [4, 5])]

    def test_unknown_entity_date_pair_is_refused(self, cv_panel_2x6):
        """A tuple naming an absent entity is refused."""
        splitter = OutOfSampleSplit(test_indices=[('C', pd.Timestamp('2024-01-31'))])
        with pytest.raises(ValueError, match=r"Panel index \('C', .*\) not found in data"):
            collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))

    def test_unknown_date_is_refused(self, cv_panel_2x6):
        """A date absent from every entity is refused."""
        splitter = OutOfSampleSplit(test_indices=['2030-01-31'])
        with pytest.raises(ValueError, match='Date 2030-01-31 not found in panel data'):
            collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))

    def test_default_date_missing_for_one_entity(self, cv_panel_2x6):
        """An entity lacking a default test date is warned about, then the split fails."""
        X = cv_panel_2x6.drop(('B', pd.Timestamp('2024-06-30')))
        with pytest.warns(UserWarning, match="Cannot find test period '2024-06-30 00:00:00' for entity 'B'"):
            with pytest.raises(ValueError, match='Cannot find test indices for the group : B'):
                collect_folds(OutOfSampleSplit(n_splits=2), X, groups=self._groups(X))

    def test_date_missing_for_one_entity(self, cv_panel_2x6):
        """An entity without the test date is warned about, then the split fails."""
        X = cv_panel_2x6.drop(('B', pd.Timestamp('2024-06-30')))
        splitter = OutOfSampleSplit(test_indices=['2024-06-30'])
        with pytest.warns(UserWarning, match="Cannot find test period '2024-06-30' for entity 'B'"):
            with pytest.raises(ValueError, match='Cannot find test indices for the group : B'):
                collect_folds(splitter, X, groups=self._groups(X))

    @pytest.mark.parametrize(
        ('indices', 'expected'),
        [
            ([('A', pd.Timestamp('2024-04-30')), ('B', '2024-05-31')], [([0, 1, 2], [3, 4]), ([6, 7, 8, 9], [10, 11])]),
            ([3, 10], [([0, 1, 2], [3, 4]), ([6, 7, 8, 9], [10, 11])]),
            ([9], [([6, 7, 8], [9, 10])]),
        ],
        ids=['tuples', 'positions', 'position-of-the-second-entity'],
    )
    def test_entity_specific_test_indices(self, cv_panel_2x6, indices, expected):
        """Each (entity, date) tuple or position opens a fold that tests and trains its entity only."""
        # Valeur d'or : A teste avril-mai (positions 3-4) après janvier-mars ; B teste mai-juin
        # (positions 10-11) après janvier-avril ; test_size coupé à la fin de l'entité
        splitter = OutOfSampleSplit(test_indices=indices, test_size=2)
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == expected
        assert_no_leakage(cv_panel_2x6, folds)
        assert splitter.get_n_splits() == len(indices)

    def test_entity_specific_fold_with_gap_and_window(self, cv_panel_2x6):
        """gap and max_train_size apply to the entity of the fold."""
        splitter = OutOfSampleSplit(test_indices=[('B', '2024-06-30')], gap=1, max_train_size=2)
        folds = collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))
        assert as_lists(folds) == [([8, 9], [11])]

    def test_tuple_on_a_panel_with_an_unknown_date(self, cv_panel_2x6):
        """An (entity, date) tuple whose date is absent is refused."""
        splitter = OutOfSampleSplit(test_indices=[('A', '2030-01-31')])
        with pytest.raises(ValueError, match=r"Panel index \('A', '2030-01-31'\) not found in data"):
            collect_folds(splitter, cv_panel_2x6, groups=self._groups(cv_panel_2x6))

    def test_groups_of_another_length_are_refused(self, cv_panel_2x6):
        """Groups must hold one label per row."""
        with pytest.raises(ValueError, match='groups must hold one label per row: got 6 labels for 12 rows'):
            collect_folds(OutOfSampleSplit(test_indices=[9]), cv_panel_2x6, groups=np.array(['A'] * 6))

    def test_duplicated_entity_date_pairs_are_refused(self, cv_panel_2x6):
        """A panel row repeated for the same entity and date is refused."""
        X = pd.concat([cv_panel_2x6, cv_panel_2x6.iloc[[2]]]).sort_index()
        with pytest.raises(ValueError, match=r"X has duplicated \(entity, date\) pairs: .*'A'.*once per entity"):
            collect_folds(OutOfSampleSplit(n_splits=2), X, groups=self._groups(X))

    @pytest.mark.parametrize(
        'kwargs', [{'n_splits': 2}, {'test_indices': ['2024-03-31']}], ids=['default', 'dates'],
    )
    def test_groups_inconsistent_with_the_index_are_refused(self, cv_panel_2x6, kwargs):
        """Groups that do not match the entity level are refused."""
        groups = np.char.lower(self._groups(cv_panel_2x6).astype(str))
        with pytest.raises(ValueError, match='groups must match the entity levels'):
            collect_folds(OutOfSampleSplit(**kwargs), cv_panel_2x6, groups=groups)
