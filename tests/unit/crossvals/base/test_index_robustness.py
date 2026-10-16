"""Robustness of ``OutOfSampleSplit`` and ``InSampleSplit`` to the index and the values of X.

Covers the index kinds (``DatetimeIndex`` start or end anchored, with a
time zone, ``RangeIndex``, ``PeriodIndex``, labels, date strings), the
resolution of ``test_indices`` on each, unsorted data (automatic sort and
the dates of the folds it returns), duplicated dates, missing values in
``X`` and ``y``, and the containers accepted for ``X``.
"""
# Importation des modules
import warnings

import numpy as np
import pandas as pd
import pytest

from tests.support.splits import as_lists, assert_no_leakage, collect_folds
from tsforecast.crossvals.base import InSampleSplit, OutOfSampleSplit

# Plis attendus sur 12 lignes, quel que soit l'index (le découpage est positionnel)
# Valeur d'or : test_size = 12 // 4 = 3, débuts de test 3, 6, 9
EXPECTED_OOS = [(list(range(3)), [3, 4, 5]), (list(range(6)), [6, 7, 8]), (list(range(9)), [9, 10, 11])]
EXPECTED_IS = [(list(range(12)), [9, 10, 11])]

# Permutation sans involution : la ligne i porte la date numéro PERM[i]
PERM = [5, 0, 9, 2, 11, 7, 1, 10, 3, 8, 6, 4]

INDEX_KINDS = {
    'month-end': pd.date_range('2024-01-31', periods=12, freq='ME'),
    'month-start': pd.date_range('2024-01-01', periods=12, freq='MS'),
    'tz-aware': pd.date_range('2024-01-31', periods=12, freq='ME', tz='Europe/Paris'),
    'range': pd.RangeIndex(12),
    'range-from-100': pd.RangeIndex(100, 112),
    'period': pd.period_range('2024-01', periods=12, freq='M'),
    'labels': pd.Index([f'p{i:02d}' for i in range(12)]),
    'date-strings': pd.Index(pd.date_range('2024-01-31', periods=12, freq='ME').strftime('%Y-%m-%d')),
    'named': pd.date_range('2024-01-31', periods=12, freq='ME', name='Date de référence'),
}


class TestIndexKinds:
    """The folds depend on the row order only, not on the kind of index."""

    @pytest.mark.parametrize('index', [pytest.param(v, id=k) for k, v in INDEX_KINDS.items()])
    @pytest.mark.parametrize(('cls', 'expected'), [
        pytest.param(OutOfSampleSplit, EXPECTED_OOS, id='oos'),
        pytest.param(InSampleSplit, EXPECTED_IS, id='is'),
    ])
    def test_default_folds(self, index, cls, expected):
        """Every kind of sorted index gives the positional golden folds."""
        X = pd.DataFrame({'v': np.arange(12.0)}, index=index)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert as_lists(collect_folds(cls(n_splits=3), X)) == expected

    @pytest.mark.parametrize(('index', 'test_date'), [
        pytest.param(INDEX_KINDS['month-end'], '2024-04-30', id='month-end'),
        pytest.param(INDEX_KINDS['month-start'], '2024-04-01', id='month-start'),
        pytest.param(INDEX_KINDS['tz-aware'], '2024-04-30', id='tz-aware'),
        pytest.param(INDEX_KINDS['period'], '2024-04', id='period'),
        pytest.param(INDEX_KINDS['labels'], 'p03', id='labels'),
        pytest.param(INDEX_KINDS['date-strings'], '2024-04-30', id='date-strings'),
    ])
    def test_test_index_given_as_a_string(self, index, test_date):
        """A string is matched against the index labels: the fourth row here."""
        X = pd.DataFrame({'v': np.arange(12.0)}, index=index)
        assert as_lists(collect_folds(OutOfSampleSplit(test_indices=[test_date]), X)) == [([0, 1, 2], [3])]

    def test_integers_are_positions_not_labels(self):
        """On an integer index, test_indices are positions: label 103 is out of range."""
        X = pd.DataFrame({'v': np.arange(12.0)}, index=INDEX_KINDS['range-from-100'])
        assert as_lists(collect_folds(OutOfSampleSplit(test_indices=[3]), X)) == [([0, 1, 2], [3])]
        with pytest.raises(ValueError, match='beyond data length'):
            collect_folds(OutOfSampleSplit(test_indices=[103]), X)

    def test_timestamp_on_an_index_of_date_strings_is_not_found(self):
        """Labels are compared as they are: a Timestamp does not match a date string."""
        X = pd.DataFrame({'v': np.arange(12.0)}, index=INDEX_KINDS['date-strings'])
        with pytest.raises(ValueError, match='Time series index 2024-04-30 00:00:00 not found in data'):
            collect_folds(OutOfSampleSplit(test_indices=[pd.Timestamp('2024-04-30')]), X)

    @pytest.mark.parametrize('label', [pd.Period('2024-04', 'M'), pd.Timestamp('2024-04-15')],
                             ids=['period', 'timestamp-inside-the-period'])
    def test_test_index_on_a_period_index(self, label):
        """A Period, or a timestamp falling inside it, designates the matching row of a PeriodIndex."""
        X = pd.DataFrame({'v': np.arange(12.0)}, index=INDEX_KINDS['period'])
        assert as_lists(collect_folds(OutOfSampleSplit(test_indices=[label]), X)) == [([0, 1, 2], [3])]

    def test_partial_date_matching_one_row(self):
        """A month without its day designates the only row of that month."""
        X = pd.DataFrame({'v': np.arange(12.0)}, index=INDEX_KINDS['month-end'])
        assert as_lists(collect_folds(OutOfSampleSplit(test_indices=['2024-04']), X)) == [([0, 1, 2], [3])]


class TestUnsortedData:
    """Unsorted rows are sorted by index before splitting."""

    @pytest.fixture
    def unsorted(self, cv_frame_12) -> pd.DataFrame:
        """The twelve month-ends in the order of ``PERM`` (value = date number)."""
        return cv_frame_12.iloc[PERM]

    @pytest.mark.parametrize('cls', [OutOfSampleSplit, InSampleSplit], ids=['oos', 'is'])
    def test_sort_is_announced(self, unsorted, cls):
        """A warning announces the automatic sort."""
        with pytest.warns(UserWarning, match='Time series data is not sorted by date. Sorting automatically.'):
            collect_folds(cls(n_splits=3), unsorted)

    @pytest.mark.parametrize(('cls', 'expected'), [
        pytest.param(OutOfSampleSplit, EXPECTED_OOS, id='oos'),
        pytest.param(InSampleSplit, EXPECTED_IS, id='is'),
    ])
    def test_folds_select_the_same_dates_as_on_sorted_data(self, cv_frame_12, unsorted, cls, expected):
        """Each fold holds the same dates as the fold of the sorted series."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            folds = collect_folds(cls(n_splits=3), unsorted)
        dates = cv_frame_12.index
        assert [(sorted(unsorted.index[tr]), sorted(unsorted.index[te])) for tr, te in folds] == [
            (list(dates[tr]), list(dates[te])) for tr, te in expected
        ]

    @pytest.mark.parametrize('kwargs', [
        {'n_splits': 3}, {'n_splits': 2, 'gap': 1, 'max_train_size': 4}, {'test_indices': ['2024-07-31']},
    ], ids=['expanding', 'rolling-gap', 'test-indices'])
    def test_no_leakage(self, unsorted, kwargs):
        """Training dates precede test dates on unsorted data too."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            folds = collect_folds(OutOfSampleSplit(**kwargs), unsorted)
        assert_no_leakage(unsorted, folds, gap=kwargs.get('gap', 0))

    def test_no_leakage_on_an_interleaved_panel(self, cv_panel_2x6):
        """A panel sorted by date then entity gives leak-free folds testing the last dates."""
        # Construction : lignes entrelacées (B, A, B, A, ...) date par date
        panel = cv_panel_2x6.iloc[[6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5]]
        groups = panel.index.get_level_values(0).to_numpy()
        with pytest.warns(UserWarning, match='Panel data is not sorted by group then by date'):
            folds = collect_folds(OutOfSampleSplit(n_splits=2), panel, groups=groups)
        assert_no_leakage(panel, folds, groups=groups)
        assert [sorted(set(panel.index[te].get_level_values(1))) for _, te in folds] == [
            [pd.Timestamp('2024-05-31')], [pd.Timestamp('2024-06-30')],
        ]

    def test_reversed_order(self, cv_frame_12):
        """Rows given latest first are split like the sorted series."""
        reversed_frame = cv_frame_12.iloc[::-1]
        with pytest.warns(UserWarning, match='not sorted by date'):
            folds = collect_folds(OutOfSampleSplit(n_splits=3), reversed_frame)
        assert [sorted(reversed_frame['v'].iloc[te]) for _, te in folds] == [[3, 4, 5], [6, 7, 8], [9, 10, 11]]

    def test_labels_are_ordered_as_strings(self):
        """A label index is sorted lexicographically: 'period_10' comes before 'period_2'."""
        X = pd.Series(np.arange(12.0), index=[f'period_{i}' for i in range(12)])
        with pytest.warns(UserWarning, match='not sorted by date'):
            collect_folds(OutOfSampleSplit(n_splits=3), X)


class TestDuplicatedDates:
    """Several rows sharing a date."""

    @pytest.fixture
    def duplicated(self) -> pd.DataFrame:
        """Six rows on five month-ends; rows 1 and 2 share 2024-02-29."""
        dates = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-02-29', '2024-03-31', '2024-04-30', '2024-05-31'])
        return pd.DataFrame({'v': np.arange(6.0)}, index=dates)

    @pytest.mark.parametrize('kwargs', [{'n_splits': 2, 'test_size': 2}, {'test_indices': ['2024-02-29']}],
                             ids=['default', 'repeated-date-as-test-index'])
    @pytest.mark.parametrize('cls', [OutOfSampleSplit, InSampleSplit], ids=['oos', 'is'])
    def test_is_refused_naming_the_date(self, duplicated, cls, kwargs):
        """A date shared by several rows of a time series is refused, whatever the folds asked."""
        with pytest.raises(ValueError, match=r"X has duplicated dates: \[Timestamp\('2024-02-29.*each date must appear once$"):
            collect_folds(cls(**kwargs), duplicated)

    def test_unsorted_duplicates_are_refused_before_sorting(self, duplicated):
        """The check precedes the automatic sort: no sort warning is emitted."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ValueError, match='X has duplicated dates'):
                collect_folds(OutOfSampleSplit(n_splits=2), duplicated.iloc[::-1])


class TestMissingValues:
    """The values of X and y are never read."""

    @pytest.mark.parametrize('cls', [OutOfSampleSplit, InSampleSplit], ids=['oos', 'is'])
    def test_nan_do_not_change_the_folds(self, cv_frame_12, cls):
        """NaN in X (scattered or a whole column) and in y leave the folds unchanged."""
        X = cv_frame_12.assign(empty=np.nan)
        X.iloc[[0, 5, 11], 0] = np.nan
        y = pd.Series(np.nan, index=X.index)
        assert as_lists(collect_folds(cls(n_splits=3), X, y)) == as_lists(collect_folds(cls(n_splits=3), cv_frame_12))


class TestMissingEntityLabel:
    """A panel whose entity level holds a missing value."""

    @pytest.mark.parametrize('splitter', [
        pytest.param(OutOfSampleSplit(n_splits=2), id='oos-default'),
        pytest.param(OutOfSampleSplit(test_indices=['2024-03-31']), id='oos-date'),
        pytest.param(InSampleSplit(), id='is-default'),
        pytest.param(InSampleSplit(test_indices=['2024-03-31']), id='is-date'),
    ])
    def test_is_refused_with_a_clear_message(self, splitter):
        """A missing entity label is refused with a message saying so."""
        dates = pd.date_range('2024-01-31', periods=4, freq='ME')
        panel = pd.DataFrame({'v': np.arange(8.0)}, index=pd.MultiIndex.from_product([[1.0, np.nan], dates]))
        groups = panel.index.get_level_values(0).to_numpy()
        with pytest.raises(ValueError, match='Entity labels must not be missing'):
            collect_folds(splitter, panel, groups=groups)


class TestContainers:
    """Containers accepted for X."""

    @pytest.mark.parametrize('cls', [OutOfSampleSplit, InSampleSplit], ids=['oos', 'is'])
    def test_list_gives_the_folds_of_the_array(self, cls):
        """A list of rows is split like the equivalent array."""
        values = list(range(12))
        assert as_lists(collect_folds(cls(n_splits=3), values)) == as_lists(
            collect_folds(cls(n_splits=3), np.array(values))
        )
