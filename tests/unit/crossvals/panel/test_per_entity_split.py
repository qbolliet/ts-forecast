"""Tests of the per-entity panel splitters.

``PanelOutOfSampleSplitPerEntity`` and ``PanelInSampleSplitPerEntity`` cut each
entity on its own dates, one fold per entity and per split (entities with
different coverage are covered by ``test_contract.py``). Covered: hand-written folds
(default folds, ``gap`` and rolling window, explicit dates, several dates,
in-sample window, single entity), folds limited to one entity by positions or
``(entity, date)`` tuples, explicit ``groups`` in every container, unknown
dates, and the absence of leakage on the realistic heterogeneous-coverage
panel.
"""
# Importation des modules
import numpy as np
import pandas as pd
import pytest

from tests.support.splits import as_lists, assert_in_sample, assert_no_leakage, collect_folds
from tsforecast.crossvals.panel import PanelInSampleSplitPerEntity, PanelOutOfSampleSplitPerEntity

from .conftest import entities_of


def _upto(stop: int, start: int = 0) -> list[int]:
    """Return the positions ``start`` to ``stop`` (excluded) as a list."""
    return list(range(start, stop))


def _daily_panel(entities: list, periods: int) -> pd.DataFrame:
    """Balanced panel of daily dates from 2020-01-01, value equal to the row position."""
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    index = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
    return pd.DataFrame({'v': np.arange(len(index), dtype=float)}, index=index)


class TestOutOfSamplePerEntity:
    """Hand-written folds of PanelOutOfSampleSplitPerEntity."""

    def test_default_folds(self, panel_3x6):
        """Each default fold is yielded entity by entity, folds in chronological order."""
        # Valeur d'or : tests en avril, mai, juin ; une entité par pli, dans l'ordre A, B, C
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=1), panel_3x6)
        assert as_lists(folds) == [
            ([0, 1, 2], [3]), ([6, 7, 8], [9]), ([12, 13, 14], [15]),
            ([0, 1, 2, 3], [4]), ([6, 7, 8, 9], [10]), ([12, 13, 14, 15], [16]),
            ([0, 1, 2, 3, 4], [5]), ([6, 7, 8, 9, 10], [11]), ([12, 13, 14, 15, 16], [17]),
        ]

    def test_gap_and_rolling_window(self, panel_3x6):
        """gap and max_train_size apply inside each entity's fold."""
        # Valeur d'or : test en mai (4) puis juin (5) ; écart 1 ; fenêtre de 2 dates
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=1, gap=1, max_train_size=2)
        assert as_lists(collect_folds(splitter, panel_3x6)) == [
            ([1, 2], [4]), ([7, 8], [10]), ([13, 14], [16]),
            ([2, 3], [5]), ([8, 9], [11]), ([14, 15], [17]),
        ]

    def test_explicit_date(self):
        """A date gives one fold per entity, test_size dates from it."""
        X = _daily_panel(['A', 'B'], 20)
        # Valeur d'or : 2020-01-10 = position 9 de A, 29 de B
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=['2020-01-10'], test_size=3)
        assert as_lists(collect_folds(splitter, X)) == [(_upto(9), [9, 10, 11]), (_upto(29, 20), [29, 30, 31])]

    def test_entity_date_tuple_yields_its_entity_only(self, staggered_panel):
        """An (entity, date) tuple gives one fold, for its entity."""
        # Valeur d'or : B juin = 15, apprentissage mars-mai (12-14)
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=[('B', '2024-06-30')])
        assert as_lists(collect_folds(splitter, staggered_panel)) == [([12, 13, 14], [15])]

    def test_positions_yield_their_entity_only(self, staggered_panel):
        """Each position gives one fold, for the entity of that row; other entities are left out."""
        # Valeur d'or : 5 = A juin, 15 = B juin ; C n'est testée dans aucun pli
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=[5, 15], test_size=2)
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            ([0, 1, 2, 3, 4], [5, 6]),
            ([12, 13, 14], [15, 16]),
        ]

    def test_single_entity(self):
        """A panel of one entity gives the folds of its time series."""
        X = _daily_panel(['A'], 20)
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=2), X)
        assert as_lists(folds) == [(_upto(14), [14, 15]), (_upto(16), [16, 17]), (_upto(18), [18, 19])]

    def test_unknown_date_is_refused(self, panel_3x6):
        """A date absent from every entity is refused."""
        with pytest.raises(ValueError, match='Date 2025-01-31 not found in panel data'):
            collect_folds(PanelOutOfSampleSplitPerEntity(test_indices=['2025-01-31']), panel_3x6)


class TestInSamplePerEntity:
    """Hand-written folds of PanelInSampleSplitPerEntity."""

    def test_default_fold(self, panel_3x6):
        """The single fold is yielded entity by entity, each training on its whole history."""
        folds = collect_folds(PanelInSampleSplitPerEntity(test_size=2), panel_3x6)
        assert as_lists(folds) == [(_upto(6), [4, 5]), (_upto(12, 6), [10, 11]), (_upto(18, 12), [16, 17])]

    def test_explicit_date(self):
        """One date tests test_size dates of each entity; training stops at the last test row."""
        X = _daily_panel(['A', 'B', 'C'], 20)
        splitter = PanelInSampleSplitPerEntity(test_indices=['2020-01-10'], test_size=3)
        assert as_lists(collect_folds(splitter, X)) == [
            (_upto(12), [9, 10, 11]), (_upto(32, 20), [29, 30, 31]), (_upto(52, 40), [49, 50, 51]),
        ]

    def test_several_dates(self):
        """Several dates test one row per date in each entity; test_size is not used."""
        X = _daily_panel(['A', 'B'], 30)
        splitter = PanelInSampleSplitPerEntity(test_indices=['2020-01-10', '2020-01-20'], test_size=5)
        assert as_lists(collect_folds(splitter, X)) == [(_upto(20), [9, 19]), (_upto(50, 30), [39, 49])]

    def test_window(self):
        """max_train_size keeps exactly that many dates of each entity, the test included."""
        X = _daily_panel(['A', 'B'], 50)
        # Valeur d'or : test = 10 dernières dates ; fenêtre max(30, 10) = 30 dates par entité
        folds = collect_folds(PanelInSampleSplitPerEntity(test_size=10, max_train_size=30), X)
        assert as_lists(folds) == [(_upto(50, 20), _upto(50, 40)), (_upto(100, 70), _upto(100, 90))]

    def test_single_entity(self):
        """A panel of one entity gives the fold of its time series."""
        X = _daily_panel(['A'], 20)
        assert as_lists(collect_folds(PanelInSampleSplitPerEntity(test_size=5), X)) == [(_upto(20), _upto(20, 15))]

    def test_tuples_of_two_entities_leave_the_third_out(self, staggered_panel):
        """Tuples test their own entities; the entity without a tuple gets no fold."""
        # Valeur d'or : B août = 17, C décembre = 25 ; A absente
        splitter = PanelInSampleSplitPerEntity(test_indices=[('B', '2024-08-31'), ('C', '2024-12-31')])
        assert as_lists(collect_folds(splitter, staggered_panel)) == [
            (_upto(18, 12), [17]), (_upto(26, 20), [25]),
        ]


class TestExplicitGroups:
    """Explicit groups equal to the entity level give the folds of the deduced groups."""

    @pytest.mark.parametrize(
        'to_groups',
        [
            pytest.param(lambda labels: labels.to_numpy(), id='array'),
            pytest.param(lambda labels: labels.tolist(), id='list'),
            pytest.param(lambda labels: pd.Series(labels), id='series'),
            pytest.param(lambda labels: labels, id='index'),
        ],
    )
    @pytest.mark.parametrize(
        'splitter',
        [
            pytest.param(PanelOutOfSampleSplitPerEntity(test_indices=['2024-08-31'], test_size=2), id='out'),
            pytest.param(PanelInSampleSplitPerEntity(test_indices=['2024-08-31'], test_size=2), id='in'),
        ],
    )
    def test_same_folds(self, staggered_panel, splitter, to_groups):
        """Groups passed in any container change nothing."""
        groups = to_groups(staggered_panel.index.get_level_values('entity'))
        expected = as_lists(collect_folds(splitter, staggered_panel))
        assert as_lists(collect_folds(splitter, staggered_panel, groups=groups)) == expected

    @pytest.mark.parametrize('cls', [PanelOutOfSampleSplitPerEntity, PanelInSampleSplitPerEntity], ids=['out', 'in'])
    def test_groups_inconsistent_with_the_index_are_refused(self, staggered_panel, cls):
        """Groups that relabel the entities are refused before any fold."""
        groups = np.char.lower(staggered_panel.index.get_level_values('entity').to_numpy().astype(str))
        with pytest.raises(ValueError, match='groups must match the entity levels'):
            collect_folds(cls(test_indices=['2024-08-31']), staggered_panel, groups=groups)


class TestRealisticPanel:
    """Per-entity folds on the realistic panel whose entities cover different periods."""

    DATES = ['2022-01-01', '2023-01-01']

    def test_out_of_sample_no_leakage_and_one_entity_per_fold(self, heterogeneous_coverage_panel):
        """Two dates and three entities give six leak-free single-entity folds."""
        X = heterogeneous_coverage_panel
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(test_indices=self.DATES, test_size=3, gap=1), X)
        assert_no_leakage(X, folds, gap=1)
        assert [entities_of(X, test) for _, test in folds] == [['Allemagne'], ['France'], ['Italie']] * 2

    def test_in_sample_contract(self, heterogeneous_coverage_panel):
        """In sample, each entity's fold trains up to its last test date."""
        X = heterogeneous_coverage_panel
        folds = collect_folds(PanelInSampleSplitPerEntity(test_indices=self.DATES), X)
        assert_in_sample(X, folds)
        assert len(folds) == 3
