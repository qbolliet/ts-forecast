"""Contract of the global panel splitters against their per-entity variants.

``PanelOutOfSampleSplit`` / ``PanelInSampleSplit`` cut every entity at the same
dates and yield one fold holding all entities, an entity without the test dates
only training; the ``PerEntity`` variants cut each entity on its own dates.
Covered: when every entity holds the test dates, the per-entity folds partition
the global ones and equal the folds of each entity split alone; common start
dates of a global fold; one entity per ``PerEntity`` fold; entities whose
coverage differs (staggered and realistic panels); ``get_n_splits``.
"""
# Importation des modules
import warnings

import pandas as pd
import pytest

from tests.support.splits import as_lists, assert_no_leakage, collect_folds, split_folds_by_entity
from tsforecast.crossvals import TSInSampleSplit, TSOutOfSampleSplit
from tsforecast.crossvals.panel import (
    PanelInSampleSplit,
    PanelInSampleSplitPerEntity,
    PanelOutOfSampleSplit,
    PanelOutOfSampleSplitPerEntity,
)

from .conftest import entities_of

# Configurations hors échantillon : (fixture du panel, paramètres)
OUT_OF_SAMPLE = {
    'balanced-gap': ('panel_3x6', {'n_splits': 2, 'test_size': 2, 'gap': 1}),
    'balanced-rolling': ('panel_3x6', {'n_splits': 3, 'max_train_size': 2}),
    'staggered-dates': ('staggered_panel', {'test_indices': ['2024-08-31', '2024-10-31'], 'test_size': 2}),
    'staggered-positions': ('staggered_panel', {'test_indices': [5, 15, 23], 'test_size': 2}),
    'staggered-tuples': ('staggered_panel', {'test_indices': [('B', '2024-06-30'), ('C', '2024-10-31')], 'gap': 1}),
}

# Configurations dans l'échantillon : (fixture du panel, paramètres)
IN_SAMPLE = {
    'balanced-default': ('panel_3x6', {'test_size': 2}),
    'balanced-window': ('panel_3x6', {'test_size': 2, 'max_train_size': 3}),
    'staggered-one-date': ('staggered_panel', {'test_indices': ['2024-08-31'], 'test_size': 2}),
    'staggered-dates': ('staggered_panel', {'test_indices': ['2024-07-31', '2024-09-30']}),
    'staggered-tuples': ('staggered_panel', {'test_indices': [('B', '2024-08-31'), ('C', '2024-12-31')]}),
}

# Couples (variante globale, variante par entité) et leurs configurations
PAIRS = [
    pytest.param(PanelOutOfSampleSplit, PanelOutOfSampleSplitPerEntity, fixture, kwargs, id=f'out-{key}')
    for key, (fixture, kwargs) in OUT_OF_SAMPLE.items()
] + [
    pytest.param(PanelInSampleSplit, PanelInSampleSplitPerEntity, fixture, kwargs, id=f'in-{key}')
    for key, (fixture, kwargs) in IN_SAMPLE.items()
]


def _expected_per_entity(X: pd.DataFrame, global_folds: list) -> list:
    """Return the global folds cut entity by entity, entities without test rows left out.

    Order: fold by fold, entities in order of first appearance in ``X``.
    """
    by_entity = split_folds_by_entity(X, global_folds)
    return [
        by_entity[entity][k]
        for k in range(len(global_folds))
        for entity in by_entity
        if len(by_entity[entity][k][1]) > 0
    ]


class TestPerEntityFoldsPartitionTheGlobalFolds:
    """When every entity holds the test dates, PerEntity yields each global fold entity by entity."""

    @pytest.mark.parametrize(('global_cls', 'per_entity_cls', 'fixture', 'kwargs'), PAIRS)
    def test_per_entity_folds_are_the_global_folds_cut_by_entity(
        self, request, global_cls, per_entity_cls, fixture, kwargs
    ):
        """PerEntity yields, fold after fold, the rows of each entity tested in the global fold."""
        X = request.getfixturevalue(fixture)
        # Valeur attendue : plis globaux restreints à chaque entité qui y est testée
        expected = as_lists(_expected_per_entity(X, collect_folds(global_cls(**kwargs), X)))
        assert as_lists(collect_folds(per_entity_cls(**kwargs), X)) == expected

    @pytest.mark.parametrize(('global_cls', 'per_entity_cls', 'fixture', 'kwargs'), PAIRS)
    def test_each_per_entity_fold_holds_one_entity(self, request, global_cls, per_entity_cls, fixture, kwargs):
        """Training and test rows of a PerEntity fold belong to one and the same entity."""
        X = request.getfixturevalue(fixture)
        folds = collect_folds(per_entity_cls(**kwargs), X)
        # Entités vues par chaque pli : une seule, commune à l'apprentissage et au test
        seen = [(entities_of(X, train), entities_of(X, test)) for train, test in folds]
        assert all(train == test and len(test) == 1 for train, test in seen)


class TestGoldenFolds:
    """Hand-written folds of both variants on the same panel and parameters."""

    def test_global_fold_on_the_staggered_panel(self, staggered_panel):
        """One date: one fold holding every entity, each cut from its own history."""
        # Valeur d'or : août = position 7 (A), 17 (B), 21 (C) ; deux lignes de test chacune ;
        # apprentissage depuis le début de chaque entité jusqu'à la veille du test
        folds = collect_folds(PanelOutOfSampleSplit(test_indices=['2024-08-31'], test_size=2), staggered_panel)
        assert as_lists(folds) == [([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 20], [7, 8, 17, 18, 21, 22])]

    def test_per_entity_folds_on_the_staggered_panel(self, staggered_panel):
        """The same date gives one fold per entity, in the order of the entities."""
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=['2024-08-31'], test_size=2)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            ([0, 1, 2, 3, 4, 5, 6], [7, 8]),
            ([12, 13, 14, 15, 16], [17, 18]),
            ([20], [21, 22]),
        ]

    def test_global_default_folds_on_a_balanced_panel(self, panel_3x6):
        """Default folds test the last dates of every entity at once."""
        # Valeur d'or : 6 dates, 2 plis de 2 dates, écart 1 → tests mars-avril puis mai-juin
        splitter = PanelOutOfSampleSplit(n_splits=2, test_size=2, gap=1)
        assert as_lists(collect_folds(splitter, panel_3x6)) == [
            ([0, 6, 12], [2, 3, 8, 9, 14, 15]),
            ([0, 1, 2, 6, 7, 8, 12, 13, 14], [4, 5, 10, 11, 16, 17]),
        ]

    def test_per_entity_default_folds_on_a_balanced_panel(self, panel_3x6):
        """PerEntity yields fold by fold, entity by entity."""
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=2, gap=1)
        assert as_lists(collect_folds(splitter, panel_3x6)) == [
            ([0], [2, 3]), ([6], [8, 9]), ([12], [14, 15]),
            ([0, 1, 2], [4, 5]), ([6, 7, 8], [10, 11]), ([12, 13, 14], [16, 17]),
        ]

    def test_in_sample_per_entity_folds_on_the_staggered_panel(self, staggered_panel):
        """In sample, each entity trains up to its own last test row."""
        splitter = PanelInSampleSplitPerEntity(test_indices=['2024-08-31'], test_size=2)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            ([0, 1, 2, 3, 4, 5, 6, 7, 8], [7, 8]),
            ([12, 13, 14, 15, 16, 17, 18], [17, 18]),
            ([20, 21, 22], [21, 22]),
        ]


class TestBalancedPanelEqualsEntitiesSplitAlone:
    """On a balanced panel, PerEntity cuts each entity as its own time series would be cut."""

    @pytest.mark.parametrize(
        ('per_entity_cls', 'ts_cls', 'kwargs'),
        [
            pytest.param(PanelOutOfSampleSplitPerEntity, TSOutOfSampleSplit,
                         {'n_splits': 2, 'test_size': 2, 'gap': 1}, id='out-gap'),
            pytest.param(PanelOutOfSampleSplitPerEntity, TSOutOfSampleSplit,
                         {'n_splits': 3, 'test_size': 1, 'max_train_size': 2}, id='out-rolling'),
            pytest.param(PanelInSampleSplitPerEntity, TSInSampleSplit, {'test_size': 2}, id='in'),
        ],
    )
    def test_folds_of_each_entity(self, panel_3x6, per_entity_cls, ts_cls, kwargs):
        """The folds of entity e are those of the time series of e, shifted to its rows."""
        by_entity = split_folds_by_entity(panel_3x6, collect_folds(per_entity_cls(**kwargs), panel_3x6))
        # Valeur attendue : découpage de la série de chaque entité, décalé de sa première position
        expected = {
            entity: [(train + 6 * k, test + 6 * k) for train, test in collect_folds(ts_cls(**kwargs), panel_3x6.loc[entity])]
            for k, entity in enumerate(['A', 'B', 'C'])
        }
        # Plis de chaque entité, sans les plis où elle n'est pas testée
        actual = {entity: [fold for fold in folds if len(fold[1])] for entity, folds in by_entity.items()}
        assert {e: as_lists(f) for e, f in actual.items()} == {e: as_lists(f) for e, f in expected.items()}


class TestGlobalFoldsShareTheirDates:
    """A global fold starts the test of every entity at the same date."""

    @pytest.mark.parametrize(
        ('fixture', 'splitter'),
        [
            pytest.param('panel_3x6', PanelOutOfSampleSplit(n_splits=2, test_size=2), id='out-default'),
            pytest.param('staggered_panel', PanelOutOfSampleSplit(test_indices=['2024-08-31', '2024-10-31'], test_size=3),
                         id='out-dates'),
            pytest.param('panel_3x6', PanelInSampleSplit(test_size=3), id='in-default'),
            pytest.param('staggered_panel', PanelInSampleSplit(test_indices=['2024-10-31'], test_size=2), id='in-date'),
        ],
    )
    def test_first_test_date_is_common(self, request, fixture, splitter):
        """Every entity of a fold has the same first test date."""
        X = request.getfixturevalue(fixture)
        dates = X.index.get_level_values(-1)
        by_entity = split_folds_by_entity(X, collect_folds(splitter, X))
        # Première date de test de chaque entité, pli par pli : un seul élément par pli
        firsts = [
            {dates[folds[k][1]].min() for folds in by_entity.values()}
            for k in range(len(next(iter(by_entity.values()))))
        ]
        assert all(len(first) == 1 for first in firsts)

    def test_test_run_is_cut_at_the_end_of_each_entity(self, staggered_panel):
        """A common start date tests test_size dates, fewer for an entity ending earlier."""
        # Valeur d'or : octobre, 3 dates ; A octobre-décembre (9-11), B seulement octobre (19),
        # C octobre-décembre (23-25)
        splitter = PanelOutOfSampleSplit(test_indices=['2024-10-31'], test_size=3)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22], [9, 10, 11, 19, 23, 24, 25]),
        ]


def _upto(stop: int, start: int = 0) -> list[int]:
    """Return the positions ``start`` to ``stop`` (excluded) as a list."""
    return list(range(start, stop))


class TestGlobalFoldsWithDifferentCoverage:
    """Global folds when an entity does not hold the common test dates.

    The entity is warned about and not tested; it trains on its dates before
    the test of the fold (up to the last test date in sample), so that a
    model pooling the entities never trains on the test period.
    """

    def test_default_folds_out_of_sample(self, staggered_panel):
        """B ends in October: it trains on all its dates in the November and December folds."""
        # Valeur d'or : tests en novembre (A 10, C 24) puis décembre (A 11, C 25) ;
        # B (12-19, mars-octobre) entière en apprentissage dans les deux plis
        with pytest.warns(UserWarning, match="Cannot find test period '2024-11-30.*' for entity 'B'; the entity is not tested"):
            folds = collect_folds(PanelOutOfSampleSplit(n_splits=2), staggered_panel)
        assert as_lists(folds) == [
            (_upto(10) + _upto(20, 12) + _upto(24, 20), [10, 24]),
            (_upto(11) + _upto(20, 12) + _upto(25, 20), [11, 25]),
        ]

    def test_requested_date_with_gap_and_window(self, staggered_panel):
        """An untested entity trains before the test date, minus gap, within the window."""
        # Valeur d'or : décembre ; A apprend août-octobre (7-9), C août-octobre (21-23) ;
        # B : dates avant décembre = 12-19, écart 1 → jusqu'à septembre (18), fenêtre 3 → 16-18
        splitter = PanelOutOfSampleSplit(test_indices=['2024-12-31'], gap=1, max_train_size=3)
        with pytest.warns(UserWarning, match="Cannot find test period '2024-12-31' for entity 'B'"):
            folds = collect_folds(splitter, staggered_panel)
        assert as_lists(folds) == [([7, 8, 9, 16, 17, 18, 21, 22, 23], [11, 25])]

    def test_entity_starting_after_the_test_date_is_left_out(self, staggered_panel):
        """C starts in July: it neither trains nor tests in a May fold."""
        folds = collect_folds(PanelOutOfSampleSplit(test_indices=['2024-05-31']), staggered_panel)
        assert as_lists(folds) == [([0, 1, 2, 3, 12, 13], [4, 14])]

    def test_default_fold_in_sample(self, staggered_panel):
        """In sample, B is not tested in December and trains on all its dates."""
        with pytest.warns(UserWarning, match="for entity 'B'"):
            folds = collect_folds(PanelInSampleSplit(), staggered_panel)
        assert as_lists(folds) == [(_upto(26), [11, 25])]

    def test_in_sample_window_of_an_untested_entity(self, staggered_panel):
        """In sample, an untested entity keeps its last max_train_size dates up to the test date."""
        # Valeur d'or : décembre, fenêtre 2 ; A 10-11, C 24-25, B : ses deux dernières dates 18-19
        with pytest.warns(UserWarning, match="for entity 'B'"):
            folds = collect_folds(PanelInSampleSplit(max_train_size=2), staggered_panel)
        assert as_lists(folds) == [([10, 11, 18, 19, 24, 25], [11, 25])]

    @pytest.mark.parametrize(
        ('fixture', 'splitter'),
        [
            pytest.param('staggered_panel', PanelOutOfSampleSplit(n_splits=3, gap=1), id='staggered-default'),
            pytest.param('staggered_panel', PanelOutOfSampleSplit(test_indices=['2024-11-30', '2024-12-31'],
                                                                  max_train_size=4), id='staggered-dates'),
            pytest.param('heterogeneous_coverage_panel', PanelOutOfSampleSplit(n_splits=3, test_size=2, gap=1),
                         id='realistic-default'),
        ],
    )
    def test_no_training_date_reaches_the_test_period(self, request, fixture, splitter):
        """Out of sample, every training date of every entity precedes the first test date of the fold."""
        X = request.getfixturevalue(fixture)
        dates = X.index.get_level_values(-1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            folds = collect_folds(splitter, X)
        assert all(dates[train].max() < dates[test].min() for train, test in folds)

    @pytest.mark.parametrize(
        'splitter',
        [
            pytest.param(PanelInSampleSplit(test_size=3), id='default'),
            pytest.param(PanelInSampleSplit(test_indices=['2023-01-01', '2024-07-01']), id='dates'),
        ],
    )
    def test_in_sample_training_stops_at_the_last_test_date(self, heterogeneous_coverage_panel, splitter):
        """In sample, no training date of any entity is later than the last test date."""
        X = heterogeneous_coverage_panel
        dates = X.index.get_level_values(-1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            (train, test), = collect_folds(splitter, X)
        assert dates[train].max() == dates[test].max()


class TestPerEntityFoldsWithDifferentCoverage:
    """Per-entity folds cut each entity on its own dates."""

    def test_default_folds_use_each_entity_dates(self, staggered_panel):
        """Each entity is tested on its own last dates (B in September and October)."""
        # Valeur d'or : deux plis d'une date sur les dernières dates de chaque entité ;
        # A novembre-décembre (10, 11), B septembre-octobre (18, 19), C novembre-décembre (24, 25)
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(n_splits=2), staggered_panel)
        assert as_lists(folds) == [
            (_upto(10), [10]), (_upto(18, 12), [18]), (_upto(24, 20), [24]),
            (_upto(11), [11]), (_upto(19, 12), [19]), (_upto(25, 20), [25]),
        ]

    def test_entity_too_short_for_the_default_folds_is_skipped(self, staggered_panel):
        """C (6 dates) cannot hold three folds of two dates and a training date: warned, no fold."""
        with pytest.warns(UserWarning, match="Entity 'C' is too short for the default folds and gets none"):
            folds = collect_folds(PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=2), staggered_panel)
        # Valeur d'or : A teste juillet-août, septembre-octobre, novembre-décembre ; B mai-juin, juillet-août,
        # septembre-octobre ; aucun pli pour C
        assert as_lists(folds) == [
            (_upto(6), [6, 7]), (_upto(14, 12), [14, 15]),
            (_upto(8), [8, 9]), (_upto(16, 12), [16, 17]),
            (_upto(10), [10, 11]), (_upto(18, 12), [18, 19]),
        ]

    def test_entity_without_the_requested_date_gets_no_fold(self, staggered_panel):
        """B does not hold December: only A and C are tested, B is warned about."""
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=['2024-12-31'])
        with pytest.warns(UserWarning, match="Cannot find test period '2024-12-31' for entity 'B'; the entity gets no fold"):
            folds = collect_folds(splitter, staggered_panel)
        assert as_lists(folds) == [(_upto(11), [11]), (_upto(25, 20), [25])]

    def test_entity_without_history_before_the_date_gets_no_fold(self, staggered_panel):
        """C starts in July: a July test leaves it no training date; warned, no fold."""
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=['2024-07-31'])
        with pytest.warns(UserWarning, match="Entity 'C' gets no fold for '2024-07-31'"):
            folds = collect_folds(splitter, staggered_panel)
        assert as_lists(folds) == [(_upto(6), [6]), (_upto(16, 12), [16])]

    def test_folds_of_each_requested_date(self, staggered_panel):
        """Several dates: one fold per date and per entity holding it, date by date."""
        # Valeur d'or : avril (A 3, B 13 ; C absente) puis novembre (A 10, C 24 ; B absente)
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=['2024-04-30', '2024-11-30'])
        with pytest.warns(UserWarning):
            folds = collect_folds(splitter, staggered_panel)
        assert as_lists(folds) == [(_upto(3), [3]), ([12], [13]), (_upto(10), [10]), (_upto(24, 20), [24])]

    def test_in_sample_default_uses_each_entity_last_date(self, staggered_panel):
        """Each entity is tested on its own last date, training on its whole history."""
        folds = collect_folds(PanelInSampleSplitPerEntity(), staggered_panel)
        assert as_lists(folds) == [(_upto(12), [11]), (_upto(20, 12), [19]), (_upto(26, 20), [25])]

    def test_in_sample_several_dates_with_one_held(self, staggered_panel):
        """Several requested dates: an entity holding only one is tested on that row alone."""
        # Valeur d'or : A porte février (1) et août (7) ; B et C seulement août (17, 21) ;
        # test_size n'est pas utilisé quand plusieurs dates sont demandées
        splitter = PanelInSampleSplitPerEntity(test_indices=['2024-02-29', '2024-08-31'], test_size=3)
        with pytest.warns(UserWarning, match="'2024-02-29' for entity 'B'"):
            folds = collect_folds(splitter, staggered_panel)
        assert as_lists(folds) == [(_upto(8), [1, 7]), (_upto(18, 12), [17]), ([20, 21], [21])]

    def test_in_sample_entity_without_any_requested_date(self, staggered_panel):
        """An entity holding none of the requested dates gets no fold."""
        splitter = PanelInSampleSplitPerEntity(test_indices=['2024-01-31', '2024-02-29'])
        with pytest.warns(UserWarning, match="for entity 'C'"):
            folds = collect_folds(splitter, staggered_panel)
        assert as_lists(folds) == [([0, 1], [0, 1])]

    @pytest.mark.parametrize('cls', [PanelOutOfSampleSplitPerEntity, PanelInSampleSplitPerEntity], ids=['out', 'in'])
    def test_empty_panel_is_refused(self, staggered_panel, cls):
        """An empty panel has no entity to split."""
        with pytest.raises(ValueError, match='Cannot split an empty dataset: number of samples=0'):
            collect_folds(cls(), staggered_panel.iloc[:0])

    def test_realistic_panel_default_folds(self, heterogeneous_coverage_panel):
        """Allemagne ends in April 2024: its default folds are its own last dates, without leakage."""
        X = heterogeneous_coverage_panel
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=3, gap=1), X)
        dates = X.index.get_level_values(-1)
        assert_no_leakage(X, folds, gap=1)
        # Dernière date testée de chaque entité : sa propre dernière date
        last_tested = {entities_of(X, test)[0]: dates[test].max() for _, test in folds}
        assert last_tested == {entity: block.index.get_level_values(-1).max() for entity, block in X.groupby(level=0)}


class TestNumberOfSplits:
    """get_n_splits announces the folds that split produces."""

    @pytest.mark.parametrize(
        'splitter',
        [
            pytest.param(PanelOutOfSampleSplit(n_splits=3), id='out-default'),
            pytest.param(PanelOutOfSampleSplit(test_indices=['2024-08-31', '2024-10-31']), id='out-dates'),
            pytest.param(PanelOutOfSampleSplit(test_indices=[5, 15, 23]), id='out-positions'),
            pytest.param(PanelInSampleSplit(test_indices=['2024-08-31']), id='in'),
        ],
    )
    def test_global_count(self, staggered_panel, splitter):
        """Global variants yield as many folds as get_n_splits announces, with or without data."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            n_folds = len(collect_folds(splitter, staggered_panel))
        assert splitter.get_n_splits(staggered_panel) == splitter.get_n_splits() == n_folds

    @pytest.mark.parametrize(
        ('fixture', 'splitter', 'n_folds'),
        [
            pytest.param('panel_3x6', PanelOutOfSampleSplitPerEntity(n_splits=2), 6, id='out-default'),
            pytest.param('panel_3x6', PanelOutOfSampleSplitPerEntity(test_indices=['2024-03-31']), 3, id='out-date'),
            pytest.param('panel_3x6', PanelInSampleSplitPerEntity(test_size=2), 3, id='in'),
            pytest.param('staggered_panel', PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=2), 6,
                         id='out-entity-skipped'),
            pytest.param('staggered_panel', PanelOutOfSampleSplitPerEntity(test_indices=[5, 15]), 2, id='out-positions'),
            pytest.param('staggered_panel', PanelInSampleSplitPerEntity(test_indices=['2024-01-31']), 1,
                         id='in-entities-skipped'),
        ],
    )
    def test_per_entity_count_with_the_data(self, request, fixture, splitter, n_folds):
        """Given the data, get_n_splits counts one fold per tested entity and per split."""
        X = request.getfixturevalue(fixture)
        # Valeur d'or : plis par entité écrits à la main (3 entités × 2 plis, 3 × 1, A et B × 3 plis, ...)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            produced = len(collect_folds(splitter, X))
        assert (splitter.get_n_splits(X), produced) == (n_folds, n_folds)

    def test_per_entity_count_does_not_repeat_the_warnings(self, staggered_panel, recwarn):
        """Counting the folds emits none of the warnings of split."""
        PanelOutOfSampleSplitPerEntity(test_indices=['2024-12-31']).get_n_splits(staggered_panel)
        assert len(recwarn) == 0

    @pytest.mark.parametrize('cls', [PanelOutOfSampleSplitPerEntity, PanelInSampleSplitPerEntity], ids=['out', 'in'])
    def test_per_entity_count_without_the_data_is_refused(self, cls):
        """Without the data, the entities are unknown: get_n_splits raises."""
        with pytest.raises(ValueError, match=f'{cls.__name__}.get_n_splits requires X'):
            cls(n_splits=2).get_n_splits()


def test_per_entity_order_follows_the_first_appearance_of_entities(staggered_panel):
    """Entities are yielded in the order of their first row, not in alphabetical order."""
    # Construction d'un panel dont les blocs d'entités sont dans l'ordre C, A, B
    X = pd.concat([staggered_panel.loc[['C']], staggered_panel.loc[['A']], staggered_panel.loc[['B']]])
    folds = collect_folds(PanelOutOfSampleSplitPerEntity(test_indices=['2024-08-31']), X)
    assert [entities_of(X, test)[0] for _, test in folds] == ['C', 'A', 'B']
