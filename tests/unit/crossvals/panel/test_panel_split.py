"""Tests of the global panel splitters ``PanelOutOfSampleSplit`` and ``PanelInSampleSplit``.

Covered: parameters and inheritance of the four panel splitters, hand-written
folds (default folds, ``gap``, rolling window, explicit dates, positions and
``(entity, date)`` tuples), degenerate panels (single entity, entity with a
single observation, entity shorter than ``gap`` + ``test_size``, empty panel,
data that is not a panel), and the safety properties of every fold (no
leakage, exact ``gap``, training window, in-sample contract) on the staggered
panel and on the realistic heterogeneous-coverage panel.
"""
# Importation des modules
import numpy as np
import pandas as pd
import pytest

from tests.support.splits import (
    as_lists,
    assert_folds_cover,
    assert_in_sample,
    assert_no_leakage,
    assert_train_window,
    assert_valid_folds,
    collect_folds,
    split_folds_by_entity,
)
from tsforecast.crossvals.base import InSampleSplit, OutOfSampleSplit
from tsforecast.crossvals.panel import (
    PanelInSampleSplit,
    PanelInSampleSplitPerEntity,
    PanelOutOfSampleSplit,
    PanelOutOfSampleSplitPerEntity,
)

from .conftest import ALL_SPLITTERS


def _upto(stop: int, start: int = 0) -> list[int]:
    """Return the positions ``start`` to ``stop`` (excluded) as a list."""
    return list(range(start, stop))


@pytest.fixture
def panel_with_single_observation(panel_3x6) -> pd.DataFrame:
    """``panel_3x6`` plus an entity ``D`` holding only June 2024 (position 18)."""
    extra = pd.DataFrame(
        {'v': [18.0]}, index=pd.MultiIndex.from_tuples([('D', pd.Timestamp('2024-06-30'))], names=['entity', 'date'])
    )
    return pd.concat([panel_3x6, extra])


class TestParameters:
    """Inheritance and stored parameters of the four panel splitters."""

    @pytest.mark.parametrize(
        ('cls', 'parent'),
        [
            pytest.param(PanelOutOfSampleSplit, OutOfSampleSplit, id='global-out'),
            pytest.param(PanelInSampleSplit, InSampleSplit, id='global-in'),
            pytest.param(PanelOutOfSampleSplitPerEntity, PanelOutOfSampleSplit, id='per-entity-out'),
            pytest.param(PanelInSampleSplitPerEntity, PanelInSampleSplit, id='per-entity-in'),
        ],
    )
    def test_inheritance(self, cls, parent):
        """Each panel splitter extends the splitter it specializes."""
        assert issubclass(cls, parent)

    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_defaults(self, cls):
        """Defaults: five splits, no explicit test index, no window, default test size."""
        splitter = cls()
        assert (splitter.n_splits, splitter.test_indices, splitter.max_train_size, splitter.test_size) == (
            5, None, None, None
        )

    @pytest.mark.parametrize('cls', [PanelOutOfSampleSplit, PanelOutOfSampleSplitPerEntity], ids=['global', 'per-entity'])
    def test_out_of_sample_gap_defaults_to_zero(self, cls):
        """Out-of-sample splitters skip no date by default."""
        assert cls().gap == 0


class TestPanelOutOfSampleSplit:
    """Hand-written folds of PanelOutOfSampleSplit, groups deduced from the index."""

    def test_default_folds(self, panel_3x6):
        """Default folds test the last dates, one date (test_size=1) per entity and per fold."""
        # Valeur d'or : 6 dates, 3 plis → tests en avril, mai, juin (positions 3, 4, 5 de chaque entité)
        folds = collect_folds(PanelOutOfSampleSplit(n_splits=3, test_size=1), panel_3x6)
        assert as_lists(folds) == [
            ([0, 1, 2, 6, 7, 8, 12, 13, 14], [3, 9, 15]),
            ([0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15], [4, 10, 16]),
            ([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16], [5, 11, 17]),
        ]

    def test_default_test_size_is_one_date(self, panel_3x6):
        """Without test_size, each fold tests one date per entity."""
        folds = collect_folds(PanelOutOfSampleSplit(n_splits=2), panel_3x6)
        assert [test.tolist() for _, test in folds] == [[4, 10, 16], [5, 11, 17]]

    def test_gap_is_counted_in_dates_of_each_entity(self, panel_3x6):
        """gap dates are skipped before the test of each entity."""
        # Valeur d'or : tests en mai puis juin ; écart 2 → apprentissage jusqu'en février puis mars
        folds = collect_folds(PanelOutOfSampleSplit(n_splits=2, gap=2), panel_3x6)
        assert as_lists(folds) == [
            ([0, 1, 6, 7, 12, 13], [4, 10, 16]),
            ([0, 1, 2, 6, 7, 8, 12, 13, 14], [5, 11, 17]),
        ]

    def test_rolling_window_per_entity(self, panel_3x6):
        """max_train_size keeps the last dates of each entity before its test."""
        folds = collect_folds(PanelOutOfSampleSplit(n_splits=2, max_train_size=2), panel_3x6)
        assert as_lists(folds) == [
            ([2, 3, 8, 9, 14, 15], [4, 10, 16]),
            ([3, 4, 9, 10, 15, 16], [5, 11, 17]),
        ]

    def test_explicit_dates(self, panel_3x6):
        """Each date opens a fold testing test_size dates of every entity from it."""
        splitter = PanelOutOfSampleSplit(test_indices=['2024-03-31', '2024-05-31'], test_size=2)
        assert as_lists(collect_folds(splitter, panel_3x6)) == [
            ([0, 1, 6, 7, 12, 13], [2, 3, 8, 9, 14, 15]),
            ([0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15], [4, 5, 10, 11, 16, 17]),
        ]

    def test_entity_date_tuple_with_gap_and_window(self, staggered_panel):
        """An (entity, date) tuple trains and tests its entity only, with gap and window."""
        # Valeur d'or : B juin = position 15 ; test 15-16 ; écart 1 → fin d'apprentissage 14 ;
        # fenêtre de 2 → 12, 13
        splitter = PanelOutOfSampleSplit(test_indices=[('B', '2024-06-30')], test_size=2, gap=1, max_train_size=2)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [([12, 13], [15, 16])]

    def test_positions_open_one_fold_per_entity(self, staggered_panel):
        """A position opens a fold limited to the entity of that row."""
        # Valeur d'or : position 5 = A juin, position 15 = B juin
        splitter = PanelOutOfSampleSplit(test_indices=[15, 5], test_size=2)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            ([0, 1, 2, 3, 4], [5, 6]),
            ([12, 13, 14], [15, 16]),
        ]

    def test_entity_shorter_than_gap_and_test_size_is_refused(self, staggered_panel):
        """C starts in July: an August test with a gap of 1 leaves it no training date."""
        splitter = PanelOutOfSampleSplit(test_indices=['2024-08-31'], test_size=2, gap=1)
        with pytest.raises(ValueError, match=r"No training data for the group 'C' before its test set \(gap=1\)"):
            collect_folds(splitter, staggered_panel)

    def test_too_many_splits_for_the_dates_is_refused(self, panel_3x6):
        """The number of distinct dates bounds n_splits * test_size."""
        with pytest.raises(ValueError, match='Too many splits=3 for number of time points=6 with test_size=2'):
            collect_folds(PanelOutOfSampleSplit(n_splits=3, test_size=2), panel_3x6)

    def test_date_absent_from_every_entity_is_refused(self, panel_3x6):
        """A requested date that no entity holds is refused."""
        with pytest.raises(ValueError, match='Date 2025-01-31 not found in panel data'):
            collect_folds(PanelOutOfSampleSplit(test_indices=['2025-01-31']), panel_3x6)


class TestPanelInSampleSplit:
    """Hand-written fold of PanelInSampleSplit, groups deduced from the index."""

    def test_default_fold(self, panel_3x6):
        """The last test_size dates of every entity, training on all their history."""
        folds = collect_folds(PanelInSampleSplit(test_size=2), panel_3x6)
        assert as_lists(folds) == [(_upto(18), [4, 5, 10, 11, 16, 17])]

    def test_default_fold_with_a_window(self, panel_3x6):
        """max_train_size keeps the last dates of each entity, the test included."""
        folds = collect_folds(PanelInSampleSplit(test_size=2, max_train_size=3), panel_3x6)
        assert as_lists(folds) == [([3, 4, 5, 9, 10, 11, 15, 16, 17], [4, 5, 10, 11, 16, 17])]

    def test_window_longer_than_the_history(self, panel_3x6):
        """A window longer than an entity keeps all of it."""
        folds = collect_folds(PanelInSampleSplit(test_size=2, max_train_size=10), panel_3x6)
        assert as_lists(folds)[0][0] == _upto(18)

    def test_one_date(self, staggered_panel):
        """One date tests test_size dates of every entity; training stops at the last test row."""
        splitter = PanelInSampleSplit(test_indices=['2024-08-31'], test_size=2)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            (_upto(9) + _upto(19, 12) + _upto(23, 20), [7, 8, 17, 18, 21, 22]),
        ]

    def test_several_dates(self, staggered_panel):
        """Several dates test one row per entity and per date."""
        # Valeur d'or : juillet et septembre ; A 6 et 8, B 16 et 18, C 20 et 22
        splitter = PanelInSampleSplit(test_indices=['2024-07-31', '2024-09-30'])
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            (_upto(9) + _upto(19, 12) + _upto(23, 20), [6, 8, 16, 18, 20, 22]),
        ]

    def test_entity_date_tuple(self, staggered_panel):
        """An (entity, date) tuple tests test_size rows of its entity only."""
        # Valeur d'or : B août = 17 ; trois lignes 17-19 (B finit en octobre)
        splitter = PanelInSampleSplit(test_indices=[('B', '2024-08-31')], test_size=3)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [(_upto(20, 12), [17, 18, 19])]


class TestDegenerateEntities:
    """Single entity, entity with a single observation, empty panel, data that is not a panel."""

    @pytest.mark.parametrize(
        ('cls', 'expected'),
        [
            pytest.param(PanelOutOfSampleSplit, [([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])], id='out'),
            pytest.param(PanelInSampleSplit, [(_upto(6), [4, 5])], id='in'),
        ],
    )
    def test_single_entity_is_split_like_a_time_series(self, panel_3x6, cls, expected):
        """A panel of one entity gives the folds of its time series."""
        kwargs = {'n_splits': 2, 'test_size': 2} if cls is PanelOutOfSampleSplit else {'test_size': 2}
        assert as_lists(collect_folds(cls(**kwargs), panel_3x6.loc[['A']])) == expected

    def test_single_observation_entity_in_sample(self, panel_with_single_observation):
        """In sample, an entity of one row is tested and trained on that row."""
        folds = collect_folds(PanelInSampleSplit(), panel_with_single_observation)
        assert as_lists(folds) == [(_upto(19), [5, 11, 17, 18])]

    def test_single_observation_entity_out_of_sample_is_refused(self, panel_with_single_observation):
        """Out of sample, an entity of one row has nothing to train on."""
        splitter = PanelOutOfSampleSplit(test_indices=['2024-06-30'])
        with pytest.raises(ValueError, match="No training data for the group 'D' before its test set"):
            collect_folds(splitter, panel_with_single_observation)

    def test_single_observation_entity_in_the_default_folds_is_refused(self, panel_with_single_observation):
        """D holds June only: left out of the May fold, it is tested in June with nothing to train on."""
        with pytest.warns(UserWarning, match="Cannot find test period '2024-05-31 00:00:00' for entity 'D'"):
            with pytest.raises(ValueError, match="No training data for the group 'D' before its test set"):
                collect_folds(PanelOutOfSampleSplit(n_splits=2), panel_with_single_observation)

    @pytest.mark.parametrize(
        ('X', 'message'),
        [
            pytest.param(pd.DataFrame({'v': range(6)}), r'Panel data requires MultiIndex \(entity, time\)', id='frame'),
            pytest.param(pd.Series(range(6), index=pd.date_range('2024-01-31', periods=6, freq='ME')),
                         r'Panel data requires MultiIndex \(entity, time\)', id='series'),
            pytest.param(np.zeros((6, 1)), 'Panel data requires pandas DataFrame/Series with MultiIndex', id='array'),
        ],
    )
    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_data_that_is_not_a_panel_is_refused(self, cls, X, message):
        """Without a MultiIndex (entity..., date), the entities cannot be deduced."""
        with pytest.raises(ValueError, match=message):
            collect_folds(cls(n_splits=2), X)

    def test_empty_panel_is_refused_in_sample(self, panel_3x6):
        """An empty panel is refused, as an empty time series is."""
        with pytest.raises(ValueError, match='Cannot split an empty dataset: number of samples=0'):
            collect_folds(PanelInSampleSplit(), panel_3x6.iloc[:0])

    def test_empty_panel_is_refused_out_of_sample(self, panel_3x6):
        """Out of sample, an empty panel has no date to split."""
        with pytest.raises(ValueError, match='Too many splits=2 for number of time points=0'):
            collect_folds(PanelOutOfSampleSplit(n_splits=2), panel_3x6.iloc[:0])


# Configurations hors échantillon pour les propriétés : (fixture, paramètres)
OUT_OF_SAMPLE_CONFIGS = {
    'balanced-default': ('panel_3x6', {'n_splits': 3, 'test_size': 1}),
    'balanced-gap-window': ('panel_3x6', {'n_splits': 2, 'test_size': 1, 'gap': 1, 'max_train_size': 2}),
    'staggered-dates': ('staggered_panel', {'test_indices': ['2024-08-31', '2024-10-31'], 'test_size': 2}),
    'staggered-dates-gap': ('staggered_panel', {'test_indices': ['2024-09-30', '2024-10-31'], 'gap': 1,
                                                'max_train_size': 3}),
    'staggered-tuples': ('staggered_panel', {'test_indices': [('A', '2024-05-31'), ('C', '2024-10-31')],
                                             'test_size': 2, 'gap': 2}),
}

# Configurations dans l'échantillon pour les propriétés : (fixture, paramètres)
IN_SAMPLE_CONFIGS = {
    'balanced-default': ('panel_3x6', {'test_size': 3}),
    'balanced-window': ('panel_3x6', {'test_size': 2, 'max_train_size': 3}),
    'staggered-date': ('staggered_panel', {'test_indices': ['2024-08-31'], 'test_size': 3}),
    'staggered-dates-window': ('staggered_panel', {'test_indices': ['2024-07-31', '2024-09-30'], 'max_train_size': 2}),
}


def _configs(configs: dict) -> list:
    """Turn a configuration dictionary into ``pytest.param`` items with readable ids."""
    return [pytest.param(fixture, kwargs, id=key) for key, (fixture, kwargs) in configs.items()]


class TestFoldProperties:
    """Safety properties of every fold, checked entity by entity on the dates."""

    @pytest.mark.parametrize(('fixture', 'kwargs'), _configs(OUT_OF_SAMPLE_CONFIGS))
    def test_out_of_sample_no_leakage_and_exact_gap(self, request, fixture, kwargs):
        """Training dates precede test dates by exactly gap dates, in every entity."""
        X = request.getfixturevalue(fixture)
        folds = collect_folds(PanelOutOfSampleSplit(**kwargs), X)
        assert_valid_folds(X, folds)
        assert_no_leakage(X, folds, gap=kwargs.get('gap', 0))

    @pytest.mark.parametrize(('fixture', 'kwargs'), _configs(OUT_OF_SAMPLE_CONFIGS))
    def test_out_of_sample_training_window(self, request, fixture, kwargs):
        """Training is the run of an entity's dates ending gap dates before its test."""
        X = request.getfixturevalue(fixture)
        folds = collect_folds(PanelOutOfSampleSplit(**kwargs), X)
        assert_train_window(X, folds, gap=kwargs.get('gap', 0), max_train_size=kwargs.get('max_train_size'))

    @pytest.mark.parametrize(('fixture', 'kwargs'), _configs(IN_SAMPLE_CONFIGS))
    def test_in_sample_contract(self, request, fixture, kwargs):
        """Test rows are training rows; training ends at the last test date of each entity."""
        X = request.getfixturevalue(fixture)
        folds = collect_folds(PanelInSampleSplit(**kwargs), X)
        assert_valid_folds(X, folds)
        assert_in_sample(X, folds)

    @pytest.mark.parametrize(('n_splits', 'test_size'), [(3, 1), (2, 2), (5, 1)])
    def test_default_tests_tile_the_last_dates_of_each_entity(self, panel_3x6, n_splits, test_size):
        """Default test sets of an entity are adjacent, of test_size dates, and end at its last date."""
        folds = collect_folds(PanelOutOfSampleSplit(n_splits=n_splits, test_size=test_size), panel_3x6)
        for k, (entity, entity_folds) in enumerate(split_folds_by_entity(panel_3x6, folds).items()):
            # Positions de l'entité : 6k à 6k + 5 ; son dernier pli doit finir à 6k + 5
            assert_folds_cover(entity_folds, test_size=test_size)
            assert entity_folds[-1][1].max() == 6 * k + 5, entity

    def test_in_sample_default_test_is_the_last_dates_of_each_entity(self, panel_3x6):
        """Without test_indices, the in-sample test holds the last test_size dates of every entity."""
        (_, test), = collect_folds(PanelInSampleSplit(test_size=2), panel_3x6)
        dates = panel_3x6.index.get_level_values(-1)
        assert sorted(set(dates[test])) == list(pd.to_datetime(['2024-05-31', '2024-06-30']))


class TestRealisticPanel:
    """Properties on the realistic panel whose entities cover different, irregular periods."""

    # Dates portées par les trois entités (grille mensuelle commune à partir de 2019-01)
    DATES = ['2022-01-01', '2023-01-01']

    def test_out_of_sample_no_leakage_gap_and_window(self, heterogeneous_coverage_panel):
        """Every entity trains on its 24 dates ending one date before its 3 test dates."""
        X = heterogeneous_coverage_panel
        splitter = PanelOutOfSampleSplit(test_indices=self.DATES, test_size=3, gap=1, max_train_size=24)
        folds = collect_folds(splitter, X)
        assert_no_leakage(X, folds, gap=1)
        assert_train_window(X, folds, gap=1, max_train_size=24)

    def test_out_of_sample_tests_every_entity_at_each_date(self, heterogeneous_coverage_panel):
        """A date fold holds the three entities, three rows each."""
        X = heterogeneous_coverage_panel
        folds = collect_folds(PanelOutOfSampleSplit(test_indices=self.DATES, test_size=3), X)
        assert [len(test) for _, test in folds] == [9, 9]

    def test_in_sample_contract(self, heterogeneous_coverage_panel):
        """In sample, each entity trains on its history up to its last test date."""
        X = heterogeneous_coverage_panel
        folds = collect_folds(PanelInSampleSplit(test_indices=self.DATES), X)
        assert_in_sample(X, folds)

    def test_default_folds_when_an_entity_ends_earlier(self, heterogeneous_coverage_panel):
        """Allemagne ends in April 2024: not tested in the last two folds, it trains on all its dates."""
        X = heterogeneous_coverage_panel
        with pytest.warns(UserWarning, match="for entity 'Allemagne'; the entity is not tested"):
            folds = collect_folds(PanelOutOfSampleSplit(n_splits=4), X)
        # Valeur d'or : tests en avril, mai, juin, juillet 2024 ; Allemagne testée seulement en avril
        tested = [sorted(set(X.index[test].get_level_values(0))) for _, test in folds]
        assert tested == [['Allemagne', 'France', 'Italie']] + [['France', 'Italie']] * 3
        assert_no_leakage(X, folds)
        # Apprentissage de l'Allemagne dans les plis où elle n'est pas testée : toutes ses dates
        n_germany = int((X.index.get_level_values(0) == 'Allemagne').sum())
        assert [int(np.sum(X.index[train].get_level_values(0) == 'Allemagne')) for train, _ in folds[1:]] == [n_germany] * 3
