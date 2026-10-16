"""Tests of the time series splitters ``TSOutOfSampleSplit`` and ``TSInSampleSplit``.

Covers what the two classes add to their bases (``groups`` ignored with a
warning, inheritance), their folds on time series with golden values
(expanding and rolling windows, gap, explicit test dates and positions,
degenerate data), a panel given by mistake, and the sklearn contract:
``cross_validate``, ``cross_val_score`` and ``GridSearchCV`` with
``cv=splitter``.
"""
# Importation des modules
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate

from tests.support.splits import as_lists, assert_folds_cover, assert_in_sample, assert_no_leakage, collect_folds
from tsforecast.crossvals.base import InSampleSplit, OutOfSampleSplit
from tsforecast.crossvals.time_series import TSInSampleSplit, TSOutOfSampleSplit

GROUPS_WARNING = (
    r'The groups parameter is ignored by {name}\. '
    r'Time series splits treat data as a single temporal sequence\.'
)


def _upto(k: int) -> list[int]:
    """Return the positions ``0 .. k-1``."""
    return list(range(k))


@pytest.fixture
def target_12(cv_frame_12) -> pd.Series:
    """Target equal to twice the row position, on the index of ``cv_frame_12``."""
    return cv_frame_12['v'] * 2


class TestTSOutOfSampleSplit:
    """Out-of-sample splits of a single time series."""

    def test_inherits_from_the_base_splitter(self):
        """TSOutOfSampleSplit is an OutOfSampleSplit."""
        assert isinstance(TSOutOfSampleSplit(), OutOfSampleSplit)

    def test_gap_and_test_size(self):
        """Three folds of five days, one day of gap."""
        # Valeur d'or : débuts de test 30 - 3*5 = 15, 20, 25 ; fin d'apprentissage = début - 1
        X = pd.DataFrame({'value': range(30)}, index=pd.date_range('2020-01-01', periods=30, freq='D'))
        folds = collect_folds(TSOutOfSampleSplit(n_splits=3, test_size=5, gap=1), X)
        assert as_lists(folds) == [
            (_upto(14), list(range(15, 20))),
            (_upto(19), list(range(20, 25))),
            (_upto(24), list(range(25, 30))),
        ]
        assert_no_leakage(X, folds, gap=1)

    def test_rolling_window(self):
        """max_train_size=15 keeps the 15 days before every test set."""
        # Valeur d'or : débuts de test 50 - 5*5 = 25, 30, ..., 45
        X = pd.Series(np.arange(50.0))
        folds = collect_folds(TSOutOfSampleSplit(n_splits=5, test_size=5, max_train_size=15), X)
        assert [(train[0], len(train)) for train, _ in folds] == [(10, 15), (15, 15), (20, 15), (25, 15), (30, 15)]
        assert_folds_cover(folds, test_size=5, n_samples=50)

    def test_expanding_window(self):
        """Without a limit, training grows by test_size at each fold."""
        folds = collect_folds(TSOutOfSampleSplit(n_splits=3, test_size=5), pd.Series(np.arange(25.0)))
        assert [len(train) for train, _ in folds] == [10, 15, 20]

    def test_specific_test_dates_with_gap(self):
        """Each test date opens a fold; two days of gap before it."""
        X = pd.DataFrame({'value': range(30)}, index=pd.date_range('2020-01-01', periods=30, freq='D'))
        splitter = TSOutOfSampleSplit(test_indices=['2020-01-10', '2020-01-20'], test_size=3, gap=2)
        assert as_lists(collect_folds(splitter, X)) == [(_upto(7), [9, 10, 11]), (_upto(17), [19, 20, 21])]

    def test_specific_test_positions(self):
        """Integer test indices are positions."""
        folds = collect_folds(TSOutOfSampleSplit(test_indices=[5, 15], test_size=2), pd.Series(np.arange(20.0)))
        assert as_lists(folds) == [(_upto(5), [5, 6]), (_upto(15), [15, 16])]

    @pytest.mark.parametrize('index', [
        pytest.param(pd.date_range('2020-01-01', periods=10, freq='D'), id='datetime'),
        pytest.param(pd.RangeIndex(10), id='range'),
        pytest.param(pd.Index([f'item_{i}' for i in range(10)]), id='labels'),
    ])
    def test_index_kind_does_not_matter(self, index):
        """The folds of ten sorted rows are the same whatever the index."""
        folds = collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=2), pd.Series(np.arange(10.0), index=index))
        assert as_lists(folds) == [(_upto(6), [6, 7]), (_upto(8), [8, 9])]

    def test_numpy_array_input(self):
        """An array gives integer position arrays."""
        folds = collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=3), np.arange(20))
        assert as_lists(folds) == [(_upto(14), [14, 15, 16]), (_upto(17), [17, 18, 19])]
        assert all(train.dtype.kind == 'i' and test.dtype.kind == 'i' for train, test in folds)

    def test_y_slices_like_x(self, cv_frame_12, target_12):
        """The folds index X and y alike: the target stays aligned on the dates."""
        for train, test in TSOutOfSampleSplit(n_splits=3).split(cv_frame_12, target_12):
            assert target_12.iloc[test].index.equals(cv_frame_12.iloc[test].index)

    def test_too_many_splits(self):
        """Five folds of three rows do not fit in eight rows."""
        with pytest.raises(ValueError, match='Too many splits=5 for number of samples=8 with test_size=3'):
            collect_folds(TSOutOfSampleSplit(n_splits=5, test_size=3), pd.Series(np.arange(8.0)))

    def test_test_size_larger_than_the_data(self):
        """A test set longer than the data is refused."""
        with pytest.raises(ValueError, match='Too many splits=2 for number of samples=5 with test_size=10'):
            collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=10), pd.Series(np.arange(5.0)))

    def test_gap_leaving_no_training_data(self):
        """A gap longer than the history is refused."""
        # Valeur d'or : premier test à la ligne 10 - 2*3 = 4 ; fin d'apprentissage 4 - 15 < 0
        with pytest.raises(ValueError, match=r'No training data before the test set starting at row 4 \(gap=15\)'):
            collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=3, gap=15), pd.Series(np.arange(10.0)))

    def test_single_observation(self):
        """One row cannot be split into two folds."""
        with pytest.raises(ValueError, match='Cannot have number of folds=3 greater than the number of samples=1'):
            collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=1), pd.Series([42.0]))

    def test_empty_series(self):
        """An empty series cannot be split."""
        with pytest.raises(ValueError, match='Cannot have number of folds=3 greater than the number of samples=0'):
            collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=1), pd.Series([], dtype=float))

    def test_unknown_test_date(self):
        """A test date absent from the series is refused."""
        X = pd.Series(np.arange(10.0), index=pd.date_range('2020-01-01', periods=10, freq='D'))
        with pytest.raises(ValueError, match='Time series index 2020-02-01 not found in data'):
            collect_folds(TSOutOfSampleSplit(test_indices=['2020-02-01']), X)


class TestTSInSampleSplit:
    """In-sample splits of a single time series."""

    def test_inherits_from_the_base_splitter(self):
        """TSInSampleSplit is an InSampleSplit."""
        assert isinstance(TSInSampleSplit(), InSampleSplit)

    def test_last_test_size_dates(self):
        """One fold: the last ten days tested, the whole series trained on."""
        X = pd.DataFrame({'value': range(30)}, index=pd.date_range('2020-01-01', periods=30, freq='D'))
        folds = collect_folds(TSInSampleSplit(test_size=10), X)
        assert as_lists(folds) == [(_upto(30), list(range(20, 30)))]
        assert_in_sample(X, folds)

    def test_specific_test_date(self):
        """A test date opens test_size days; training stops at their end."""
        X = pd.DataFrame({'value': range(30)}, index=pd.date_range('2020-01-01', periods=30, freq='D'))
        folds = collect_folds(TSInSampleSplit(test_indices=['2020-01-15'], test_size=5), X)
        assert as_lists(folds) == [(_upto(19), list(range(14, 19)))]

    def test_multiple_test_dates(self):
        """Several dates are tested together, one row each, in one fold."""
        X = pd.DataFrame({'value': range(30)}, index=pd.date_range('2020-01-01', periods=30, freq='D'))
        splitter = TSInSampleSplit(test_indices=['2020-01-10', '2020-01-15', '2020-01-20'], test_size=2)
        assert as_lists(collect_folds(splitter, X)) == [(_upto(20), [9, 14, 19])]

    def test_max_train_size(self):
        """Training keeps the 25 days ending at the last test date."""
        folds = collect_folds(TSInSampleSplit(test_size=10, max_train_size=25), pd.Series(np.arange(50.0)))
        assert as_lists(folds) == [(list(range(25, 50)), list(range(40, 50)))]

    def test_max_train_size_smaller_than_the_test(self):
        """Training cannot be shorter than the test period: here it equals it."""
        folds = collect_folds(TSInSampleSplit(test_size=15, max_train_size=10), pd.Series(np.arange(30.0)))
        assert as_lists(folds) == [(list(range(15, 30)), list(range(15, 30)))]

    def test_default_test_size(self):
        """Without test_size, the test period holds n // (n_splits + 1) rows."""
        # Valeur d'or : 21 // (5 + 1) = 3
        assert as_lists(collect_folds(TSInSampleSplit(), pd.Series(np.arange(21.0)))) == [(_upto(21), [18, 19, 20])]

    def test_numpy_array_input(self):
        """An array gives integer position arrays."""
        folds = collect_folds(TSInSampleSplit(test_size=5), np.arange(20))
        assert as_lists(folds) == [(_upto(20), list(range(15, 20)))]

    def test_single_observation(self):
        """One row with test_size=1 is both trained on and tested."""
        assert as_lists(collect_folds(TSInSampleSplit(test_size=1), pd.Series([42.0]))) == [([0], [0])]

    def test_empty_series(self):
        """An empty series is refused with a message naming the cause."""
        with pytest.raises(ValueError, match='Cannot split an empty dataset: number of samples=0'):
            collect_folds(TSInSampleSplit(test_size=1), pd.Series([], dtype=float))

    def test_unknown_test_date(self):
        """A test date absent from the series is refused."""
        X = pd.Series(np.arange(10.0), index=pd.date_range('2020-01-01', periods=10, freq='D'))
        with pytest.raises(ValueError, match='Time series index 2020-02-01 not found in data'):
            collect_folds(TSInSampleSplit(test_indices=['2020-02-01']), X)


class TestGroupsAreIgnored:
    """Groups passed to a time series splitter."""

    @pytest.mark.parametrize('cls', [TSOutOfSampleSplit, TSInSampleSplit], ids=['oos', 'is'])
    def test_warning_and_unchanged_folds(self, cv_frame_12, cls):
        """Groups raise a warning naming the class and do not change the folds."""
        splitter = cls(n_splits=3)
        with pytest.warns(UserWarning, match=GROUPS_WARNING.format(name=cls.__name__)):
            with_groups = collect_folds(splitter, cv_frame_12, groups=np.array(['A'] * 6 + ['B'] * 6))
        assert as_lists(with_groups) == as_lists(collect_folds(splitter, cv_frame_12))

    def test_groups_of_another_length_are_ignored_too(self, cv_frame_12):
        """Ignored groups are not even checked against X."""
        with pytest.warns(UserWarning, match='groups parameter is ignored'):
            assert len(collect_folds(TSOutOfSampleSplit(n_splits=3), cv_frame_12, groups=np.zeros(3))) == 3


class TestPanelGivenToATimeSeriesSplitter:
    """Panel data refused by the time series splitters."""

    @pytest.mark.parametrize(('cls', 'panel_class'), [
        (TSOutOfSampleSplit, 'PanelOutOfSampleSplit'),
        (TSInSampleSplit, 'PanelInSampleSplit'),
    ], ids=['oos', 'is'])
    @pytest.mark.parametrize('kind', ['DataFrame', 'Series'])
    def test_multiindex_is_refused(self, cv_panel_2x6, cls, panel_class, kind):
        """A MultiIndex is refused when split is called, pointing to the panel splitter."""
        X = cv_panel_2x6 if kind == 'DataFrame' else cv_panel_2x6['v']
        message = (rf'{cls.__name__} splits a single time series, but X has a MultiIndex '
                   rf'\(panel data\); use {panel_class} instead')
        with pytest.raises(ValueError, match=message):
            cls(n_splits=2).split(X)

    def test_refusal_precedes_the_groups_warning(self, cv_panel_2x6):
        """With groups, the refusal comes without the warning about ignored groups."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ValueError, match='MultiIndex'):
                TSOutOfSampleSplit(n_splits=2).split(cv_panel_2x6, groups=np.zeros(12))


class TestSklearnModelSelection:
    """The splitters as the cv argument of sklearn model selection tools."""

    def test_cross_validate_uses_the_folds(self, cv_frame_12, target_12):
        """cross_validate evaluates exactly the folds of split."""
        splitter = TSOutOfSampleSplit(n_splits=3)
        result = cross_validate(DummyRegressor(), cv_frame_12, target_12, cv=splitter,
                                scoring='neg_mean_absolute_error', return_indices=True)
        produced = list(zip(result['indices']['train'], result['indices']['test']))
        assert as_lists(produced) == as_lists(collect_folds(splitter, cv_frame_12))
        # Valeur d'or : moyenne d'apprentissage 2, 5, 8 ; cibles de test 6-10, 12-16, 18-22
        np.testing.assert_allclose(result['test_score'], [-6.0, -9.0, -12.0])

    def test_cross_val_score_in_sample(self, cv_frame_12, target_12):
        """One in-sample fold: one score."""
        # Valeur d'or : moyenne des 12 cibles = 11 ; écarts 7, 9, 11 sur les trois dernières
        scores = cross_val_score(DummyRegressor(), cv_frame_12, target_12, cv=TSInSampleSplit(n_splits=3),
                                 scoring='neg_mean_absolute_error')
        np.testing.assert_allclose(scores, [-9.0])

    def test_cross_validate_on_arrays(self, cv_frame_12, target_12):
        """Arrays give the same scores as pandas objects."""
        splitter = TSOutOfSampleSplit(n_splits=3)
        scores = cross_validate(DummyRegressor(), cv_frame_12.to_numpy(), target_12.to_numpy(), cv=splitter,
                                scoring='neg_mean_absolute_error')['test_score']
        np.testing.assert_allclose(scores, [-6.0, -9.0, -12.0])

    def test_grid_search_with_default_folds(self, cv_frame_12, target_12):
        """GridSearchCV runs one evaluation per fold and candidate."""
        search = GridSearchCV(Ridge(), {'alpha': [0.1, 1.0]}, cv=TSOutOfSampleSplit(n_splits=3))
        search.fit(cv_frame_12, target_12)
        assert search.n_splits_ == 3 and 'split2_test_score' in search.cv_results_

    @pytest.mark.parametrize(('splitter', 'n_folds'), [
        pytest.param(TSOutOfSampleSplit(test_indices=['2024-06-30', '2024-09-30'], test_size=3), 2, id='test-indices'),
        pytest.param(TSInSampleSplit(test_size=3), 1, id='in-sample'),
    ])
    def test_grid_search_with_other_folds(self, cv_frame_12, target_12, splitter, n_folds):
        """GridSearchCV accepts explicit test dates and in-sample splits."""
        search = GridSearchCV(Ridge(), {'alpha': [0.1, 1.0]}, cv=splitter).fit(cv_frame_12, target_12)
        assert search.n_splits_ == n_folds

    def test_fold_without_training_data_is_reported(self, cv_frame_12, target_12):
        """A configuration leaving a fold without training data stops cross_validate with its cause."""
        with pytest.raises(ValueError, match='No training data before the test set starting at row 3'):
            cross_validate(DummyRegressor(), cv_frame_12, target_12, cv=TSOutOfSampleSplit(n_splits=3, gap=7))

    def test_cross_validate_on_lists(self):
        """Lists are accepted like arrays."""
        values = [float(i) for i in range(12)]
        scores = cross_validate(DummyRegressor(), [[v] for v in values], [2 * v for v in values],
                                cv=TSOutOfSampleSplit(n_splits=3), scoring='neg_mean_absolute_error')['test_score']
        np.testing.assert_allclose(scores, [-6.0, -9.0, -12.0])
