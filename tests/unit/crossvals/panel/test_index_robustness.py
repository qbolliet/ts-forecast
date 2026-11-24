"""Robustness of the four panel splitters to the shape and order of the panel index.

Covered: entities on two index levels (three-level ``MultiIndex``), index
levels named differently or unnamed, unsorted panels (entities interleaved
by date, dates reversed within each entity, shuffled rows, entity blocks in
reverse order), the date held in a column instead of the index, entity keys
that are integers, strings with spaces and accents, mixed types or tuples,
non-mutation of the input, and reproducibility of the folds.
"""
# Importation des modules
import copy

import pandas as pd
import pytest

from tests.support.perturbations import reverse_entities, shuffle_rows, to_three_level_index, with_index_names
from tests.support.splits import as_lists, assert_no_leakage, collect_folds
from tsforecast.crossvals.panel import (
    PanelInSampleSplit,
    PanelInSampleSplitPerEntity,
    PanelOutOfSampleSplit,
    PanelOutOfSampleSplitPerEntity,
)

from .conftest import ALL_SPLITTERS, AUGUST_TEST, labelled_folds


def _interleave(X: pd.DataFrame) -> pd.DataFrame:
    """Sort the rows by date then entity: entity blocks are no longer contiguous."""
    return X.swaplevel().sort_index().swaplevel()


def _reverse_dates_within_entities(X: pd.DataFrame) -> pd.DataFrame:
    """Keep entity blocks contiguous but put each entity's dates in decreasing order."""
    return pd.concat([X.loc[[entity]].iloc[::-1] for entity in dict.fromkeys(X.index.get_level_values(0))])


class TestSeveralEntityLevels:
    """The entity is made of every index level but the last."""

    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        """Entities ('FR', 'a') on positions 0-3 and ('FR', 'b') on 4-7, four month-ends each."""
        dates = pd.date_range('2024-01-31', periods=4, freq='ME')
        index = pd.MultiIndex.from_product([['FR'], ['a', 'b'], dates], names=['country', 'sector', 'date'])
        return pd.DataFrame({'v': [float(k) for k in range(8)]}, index=index)

    def test_groups_are_deduced_from_every_entity_level(self, panel):
        """Without groups, the (country, sector) entities are split separately."""
        folds = collect_folds(PanelOutOfSampleSplit(n_splits=2), panel)
        # Valeur d'or : tests aux dates 2 puis 3 de chaque secteur
        assert as_lists(folds) == [([0, 1, 4, 5], [2, 6]), ([0, 1, 2, 4, 5, 6], [3, 7])]

    def test_per_entity_folds(self, panel):
        """PerEntity yields one fold per (country, sector) entity and per split."""
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(n_splits=2), panel)
        assert as_lists(folds) == [([0, 1], [2]), ([4, 5], [6]), ([0, 1, 2], [3]), ([4, 5, 6], [7])]

    @pytest.mark.parametrize(
        ('cls', 'expected'),
        [
            pytest.param(PanelInSampleSplit, [(list(range(8)), [2, 3, 6, 7])], id='global'),
            pytest.param(PanelInSampleSplitPerEntity, [([0, 1, 2, 3], [2, 3]), ([4, 5, 6, 7], [6, 7])], id='per-entity'),
        ],
    )
    def test_in_sample_folds(self, panel, cls, expected):
        """In sample, the last two dates of each (country, sector) entity are tested."""
        assert as_lists(collect_folds(cls(test_size=2), panel)) == expected

    @pytest.mark.parametrize(
        'item',
        [
            pytest.param((('FR', 'b'), '2024-03-31'), id='entity-tuple'),
            pytest.param(('FR', 'b', '2024-03-31'), id='level-by-level'),
        ],
    )
    def test_entity_date_tuple(self, panel, item):
        """An entity spread on two levels is named as one tuple or level by level."""
        # Valeur d'or : ('FR', 'b') mars = position 6, apprentissage janvier-février (4, 5)
        folds = collect_folds(PanelOutOfSampleSplit(test_indices=[item]), panel)
        assert as_lists(folds) == [([4, 5], [6])]

    @pytest.mark.parametrize('region_by_entity', [None, {'A': 'Nord', 'B': 'Nord', 'C': 'Sud'}],
                             ids=['constant-region', 'two-regions'])
    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_outer_level_does_not_change_the_folds(self, staggered_panel, cls, region_by_entity):
        """Adding a region level above the entity keeps every entity, hence every fold."""
        X = to_three_level_index(staggered_panel, region_by_entity=region_by_entity)
        expected = as_lists(collect_folds(cls(**AUGUST_TEST), staggered_panel))
        assert as_lists(collect_folds(cls(**AUGUST_TEST), X)) == expected


class TestLevelNames:
    """Levels are read by position, whatever their names."""

    @pytest.mark.parametrize(
        'names',
        [
            pytest.param([None, None], id='unnamed'),
            pytest.param(['pays', 'période'], id='other-names'),
            pytest.param(['date', 'entity'], id='misleading-names'),
        ],
    )
    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_same_folds(self, staggered_panel, cls, names):
        """Renaming the levels, even with swapped meanings, changes no fold."""
        expected = as_lists(collect_folds(cls(**AUGUST_TEST), staggered_panel))
        X = with_index_names(staggered_panel, names)
        assert as_lists(collect_folds(cls(**AUGUST_TEST), X)) == expected


# Perturbations de l'ordre des lignes : (fonction, tri annoncé par un avertissement)
ORDERS = [
    pytest.param(_interleave, True, id='interleaved'),
    pytest.param(_reverse_dates_within_entities, True, id='dates-reversed-within-entities'),
    pytest.param(lambda X: shuffle_rows(X, seed=3), True, id='shuffled'),
    pytest.param(reverse_entities, False, id='entity-blocks-reversed'),
]


class TestUnsortedPanels:
    """Unsorted panels give the folds of the sorted panel, on the same (entity, date) rows."""

    @pytest.mark.parametrize(('perturb', 'announced'), ORDERS)
    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_folds_select_the_same_rows(self, staggered_panel, cls, perturb, announced):
        """Each fold holds the same (entity, date) rows as on the sorted panel."""
        X = perturb(staggered_panel)
        expected = labelled_folds(staggered_panel, collect_folds(cls(**AUGUST_TEST), staggered_panel))
        # Plis comparés sans tenir compte de l'ordre des entités (ordre d'apparition dans X)
        actual = labelled_folds(X, collect_folds(cls(**AUGUST_TEST), X))
        assert sorted(actual, key=repr) == sorted(expected, key=repr)

    @pytest.mark.parametrize(('perturb', 'announced'), ORDERS)
    def test_sort_is_announced_only_when_needed(self, staggered_panel, perturb, announced, recwarn):
        """Sorting is announced unless entities are contiguous and dates increasing in each one."""
        collect_folds(PanelOutOfSampleSplit(**AUGUST_TEST), perturb(staggered_panel))
        messages = [str(w.message) for w in recwarn]
        assert ('Panel data is not sorted by group then by date. Sorting automatically.' in messages) is announced

    @pytest.mark.parametrize(('perturb', 'announced'), ORDERS)
    @pytest.mark.parametrize(
        ('cls', 'kwargs'),
        [
            pytest.param(PanelOutOfSampleSplit, {'test_indices': ['2024-09-30', '2024-10-31'], 'gap': 1}, id='global'),
            pytest.param(PanelOutOfSampleSplitPerEntity, {'test_indices': [('A', '2024-05-31'), ('B', '2024-09-30')],
                                                          'gap': 2}, id='per-entity'),
        ],
    )
    def test_no_leakage(self, staggered_panel, cls, kwargs, perturb, announced):
        """On an unsorted panel, training dates still precede test dates by gap dates."""
        X = perturb(staggered_panel)
        assert_no_leakage(X, collect_folds(cls(**kwargs), X), gap=kwargs['gap'])

    def test_interleaved_golden_folds(self, staggered_panel):
        """Positions refer to the rows of the interleaved panel as given."""
        X = _interleave(staggered_panel)
        # Valeur d'or : lignes triées par date puis entité ; août = 13 (A), 14 (B), 15 (C),
        # septembre = 16, 17, 18 ; apprentissage : A 0, 1, 2, 4, 6, 8, 10 ; B 3, 5, 7, 9, 11 ; C 12
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(**AUGUST_TEST), X)
        assert as_lists(folds) == [
            ([0, 1, 2, 4, 6, 8, 10], [13, 16]),
            ([3, 5, 7, 9, 11], [14, 17]),
            ([12], [15, 18]),
        ]


class TestDateInAColumn:
    """The splitters read dates from the index only: a date column is not a panel."""

    @pytest.mark.parametrize(
        'reset',
        [
            pytest.param(lambda X: X.reset_index('date'), id='entity-index-date-column'),
            pytest.param(lambda X: X.reset_index(), id='both-in-columns'),
        ],
    )
    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_is_refused(self, staggered_panel, cls, reset):
        """Entity and date must be index levels."""
        with pytest.raises(ValueError, match=r'Panel data requires MultiIndex \(entity, time\)'):
            collect_folds(cls(**AUGUST_TEST), reset(staggered_panel))

    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_set_index_restores_the_folds(self, staggered_panel, cls):
        """Moving the columns back to the index gives the folds of the panel."""
        X = staggered_panel.reset_index().set_index(['entity', 'date'])
        expected = as_lists(collect_folds(cls(**AUGUST_TEST), staggered_panel))
        assert as_lists(collect_folds(cls(**AUGUST_TEST), X)) == expected


# Clés d'entité de remplacement pour A, B, C
KEYS = [
    pytest.param({'A': 1, 'B': 2, 'C': 3}, id='integers'),
    pytest.param({'A': 'Côte d’Ivoire', 'B': 'New York', 'C': ' C '}, id='spaces-and-accents'),
    pytest.param({'A': 1, 'B': 'B', 'C': 3.5}, id='mixed-types'),
]


def _with_tuple_keys(X: pd.DataFrame) -> pd.DataFrame:
    """Replace the entity level by one level holding (country, sector) tuples."""
    keys = {'A': ('FR', 'a'), 'B': ('FR', 'b'), 'C': ('DE', 'a')}
    level = pd.Index([keys[e] for e in X.index.get_level_values(0)], tupleize_cols=False)
    out = X.copy()
    out.index = pd.MultiIndex.from_arrays([level, X.index.get_level_values(1)], names=['entity', 'date'])
    return out


class TestEntityKeys:
    """Entity keys of any hashable type give the same folds."""

    @pytest.mark.parametrize('mapping', KEYS)
    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_renamed_entities_give_the_same_folds(self, staggered_panel, cls, mapping):
        """Renaming the entities keeps every fold, rows staying in place."""
        X = staggered_panel.rename(index=mapping, level='entity')
        expected = as_lists(collect_folds(cls(**AUGUST_TEST), staggered_panel))
        assert as_lists(collect_folds(cls(**AUGUST_TEST), X)) == expected

    @pytest.mark.parametrize('mapping', KEYS)
    def test_renamed_entities_on_an_interleaved_panel(self, staggered_panel, mapping):
        """Keys that cannot be ordered alphabetically still group the rows of an interleaved panel."""
        X = _interleave(staggered_panel.rename(index=mapping, level='entity'))
        renamed = staggered_panel.rename(index=mapping, level='entity')
        expected = labelled_folds(renamed, collect_folds(PanelOutOfSampleSplit(**AUGUST_TEST), renamed))
        assert labelled_folds(X, collect_folds(PanelOutOfSampleSplit(**AUGUST_TEST), X)) == expected

    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_tuple_keys_in_one_level_give_the_same_folds(self, staggered_panel, cls):
        """A single entity level holding tuples keeps three entities."""
        expected = as_lists(collect_folds(cls(**AUGUST_TEST), staggered_panel))
        assert as_lists(collect_folds(cls(**AUGUST_TEST), _with_tuple_keys(staggered_panel))) == expected

    @pytest.mark.parametrize(
        ('mapping', 'entity'),
        [
            pytest.param({'A': 1, 'B': 2, 'C': 3}, 2, id='integer'),
            pytest.param({'A': 'Côte d’Ivoire', 'B': 'New York', 'C': ' C '}, 'New York', id='spaces'),
        ],
    )
    def test_entity_date_tuple_with_renamed_keys(self, staggered_panel, mapping, entity):
        """An (entity, date) tuple names the renamed entity."""
        # Valeur d'or : B juin = 15, apprentissage mars-mai (12-14)
        X = staggered_panel.rename(index=mapping, level='entity')
        folds = collect_folds(PanelOutOfSampleSplit(test_indices=[(entity, '2024-06-30')]), X)
        assert as_lists(folds) == [([12, 13, 14], [15])]

    def test_entity_date_tuple_with_tuple_keys(self, staggered_panel):
        """((country, sector), date) names the entity of a single level holding tuples."""
        X = _with_tuple_keys(staggered_panel)
        folds = collect_folds(PanelOutOfSampleSplit(test_indices=[(('FR', 'b'), '2024-06-30')]), X)
        assert as_lists(folds) == [([12, 13, 14], [15])]


class TestInputAndRepeatability:
    """split leaves its input intact and gives the same folds on every call."""

    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_input_is_left_intact(self, staggered_panel, cls):
        """Neither the values nor the index of an unsorted panel are changed."""
        X = _interleave(staggered_panel)
        before = copy.deepcopy(X)
        collect_folds(cls(**AUGUST_TEST), X)
        pd.testing.assert_frame_equal(X, before)

    @pytest.mark.parametrize('cls', ALL_SPLITTERS)
    def test_two_calls_give_the_same_folds(self, staggered_panel, cls):
        """Folds do not depend on a previous call of the same splitter."""
        splitter = cls(**AUGUST_TEST)
        assert as_lists(collect_folds(splitter, staggered_panel)) == as_lists(collect_folds(splitter, staggered_panel))
