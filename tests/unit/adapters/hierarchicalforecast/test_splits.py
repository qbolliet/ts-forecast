"""Unit tests of ``tsforecast.adapters.hierarchicalforecast.splits``: ``period_splits``.

Folds of complete calendar years on a monthly temporal hierarchy (series,
balanced and unbalanced panels, cross-temporal data), aggregates dated at the
start or at the end of their period, several periods per fold, the gap in fine
steps (an aggregate overlapping the gap never trains), ``temporal_origin``,
shuffled rows, and invalid parameters.
"""
import numpy as np
import pandas as pd
import pytest

from tests.support.datasets import TEMPORAL_HIERARCHY, build_cross_temporal_hierarchy, build_temporal_hierarchy
from tests.support.perturbations import shuffle_rows
from tsforecast.adapters import period_splits


def _years(X, positions):
    """Calendar years of the periods holding the rows at ``positions`` (sorted, distinct)."""
    dates = X.index.get_level_values('date')[positions]
    return sorted(set(dates.year))


def _levels(X, positions):
    """Number of rows of each temporal level at ``positions``."""
    return pd.Series(X.index.get_level_values('level')[positions]).value_counts().to_dict()


class TestFolds:
    """Test blocks of complete periods, training before them."""

    @pytest.mark.parametrize('kind', ['series', 'panel', 'unbalanced_panel'])
    def test_last_years_tested_completely(self, kind):
        X, _ = build_temporal_hierarchy(kind, n_years=4)
        folds = period_splits(X, TEMPORAL_HIERARCHY, n_splits=2)
        assert [_years(X, test) for _, test in folds] == [[2022], [2023]]
        assert [_years(X, train) for train, _ in folds] == [[2020, 2021], [2020, 2021, 2022]]
        # Chaque série testée a ses 12 mois, 4 trimestres et 1 année
        n_series = 1 if kind == 'series' else 3
        assert _levels(X, folds[-1][1]) == {'month': 12 * n_series, 'quarter': 4 * n_series, 'year': n_series}

    @pytest.mark.parametrize('freq', ['MS', 'ME'])
    def test_start_and_end_labelled_aggregates(self, freq):
        X, _ = build_temporal_hierarchy('series', n_years=3, freq=freq)
        train, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1)[0]
        assert _levels(X, test) == {'month': 12, 'quarter': 4, 'year': 1}
        assert len(train) + len(test) == len(X)

    def test_several_periods_per_fold(self):
        X, _ = build_temporal_hierarchy('series', n_years=5)
        folds = period_splits(X, TEMPORAL_HIERARCHY, n_splits=2, test_periods=2)
        assert [_years(X, test) for _, test in folds] == [[2021, 2022], [2023, 2024]]

    def test_cross_temporal_rows(self):
        X, _ = build_cross_temporal_hierarchy('panel', n_years=3)
        _, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1)[0]
        # Tous les nœuds (5) de toutes les copies (3), à tous les niveaux temporels
        assert len(test) == 3 * 5 * 17

    def test_positions_follow_the_rows_of_X(self):
        X, _ = build_temporal_hierarchy('panel', n_years=3)
        shuffled = shuffle_rows(X)
        for (train, test), (train_s, test_s) in zip(
            period_splits(X, TEMPORAL_HIERARCHY), period_splits(shuffled, TEMPORAL_HIERARCHY)
        ):
            assert set(shuffled.index[test_s]) == set(X.index[test])
            assert set(shuffled.index[train_s]) == set(X.index[train])

    def test_series_input_and_origin(self):
        _, y = build_temporal_hierarchy('series', n_years=3)
        months = y[y.index.get_level_values('level') == 'month']
        # Années comptées d'octobre à septembre : la dernière période commence en octobre 2022
        train, test = period_splits(months, TEMPORAL_HIERARCHY, n_splits=1, temporal_origin='2019-10-01')[0]
        dates = months.index.get_level_values('date')
        assert dates[test].min() == pd.Timestamp('2022-10-01')
        assert dates[train].max() == pd.Timestamp('2022-09-01')


class TestGap:
    """Fine steps left out between training and test."""

    def test_gap_drops_the_last_fine_steps_and_overlapping_aggregates(self):
        X, _ = build_temporal_hierarchy('series', n_years=3)
        train, _ = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1, gap=2)[0]
        trained = X.index[train]
        months = trained[trained.get_level_values('level') == 'month'].get_level_values('date')
        quarters = trained[trained.get_level_values('level') == 'quarter'].get_level_values('date')
        years = trained[trained.get_level_values('level') == 'year'].get_level_values('date')
        # Mois jusqu'à octobre 2021 ; le 4e trimestre 2021 et l'année 2021 chevauchent le gap
        assert months.max() == pd.Timestamp('2021-10-01')
        assert quarters.max() == pd.Timestamp('2021-07-01')
        assert years.max() == pd.Timestamp('2020-01-01')

    def test_no_gap_keeps_every_previous_row(self):
        X, _ = build_temporal_hierarchy('series', n_years=3)
        train, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1, gap=0)[0]
        assert len(train) == 2 * 17 and len(test) == 17


class TestErrors:
    """Invalid parameters and data."""

    @pytest.mark.parametrize(
        'params',
        [{'n_splits': 0}, {'n_splits': 1.5}, {'test_periods': 0}, {'gap': -1}, {'gap': True}],
    )
    def test_invalid_parameters(self, params):
        X, _ = build_temporal_hierarchy('series', n_years=3)
        with pytest.raises(ValueError, match='must be an integer'):
            period_splits(X, TEMPORAL_HIERARCHY, **params)

    def test_too_few_periods(self):
        X, _ = build_temporal_hierarchy('series', n_years=2)
        with pytest.raises(ValueError, match='need more than 2 periods'):
            period_splits(X, TEMPORAL_HIERARCHY, n_splits=2)

    def test_fold_without_training_row(self):
        X, _ = build_temporal_hierarchy('series', n_years=2)
        with pytest.raises(ValueError, match='no training row'):
            period_splits(X, TEMPORAL_HIERARCHY, n_splits=1, gap=12)

    def test_unknown_temporal_label(self):
        X, _ = build_temporal_hierarchy('series', n_years=2)
        with pytest.raises(ValueError, match='not levels of temporal_hierarchy'):
            period_splits(X, {'year': 12, 'month': 1})

    def test_period_labels_refused(self):
        X, _ = build_temporal_hierarchy('series', n_years=2)
        index = X.index.set_levels(X.index.levels[-1].to_period('M'), level='date')
        with pytest.raises(TypeError, match='Period labels'):
            period_splits(X.set_axis(index), TEMPORAL_HIERARCHY)

    def test_folds_are_integer_arrays(self):
        X, _ = build_temporal_hierarchy('series', n_years=3)
        for train, test in period_splits(X, TEMPORAL_HIERARCHY):
            assert train.dtype.kind == 'i' and test.dtype.kind == 'i'
            assert np.intersect1d(train, test).size == 0
