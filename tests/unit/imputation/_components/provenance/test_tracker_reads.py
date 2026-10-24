"""Reads, statistics and merges of ``ImputationProvenanceTracker``.

Scenarios: cell provenance, boolean masks, per-type statistics, matrix views (and their
dtypes), merging two trackers and the realistic panel.
"""

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.imputation._components.provenance import (
    PROVENANCE_DTYPE,
    ImputationProvenanceTracker,
    provenance_statistics,
)

# Outils de test partagés du paquet
from tests.unit.imputation._components.provenance.helpers import P, _tracker


class TestGetProvenance:
    """Reading the provenance of cells."""

    def test_scalar_returns_the_member(self):
        """A single timestamp returns the ProvenanceType itself."""
        tracker, dates = _tracker()
        assert tracker.get_provenance('a', dates[0]) is P.ORIGINAL

    def test_several_labels_return_a_series(self):
        """A DatetimeIndex returns a Series restricted to those labels."""
        tracker, dates = _tracker()
        result = tracker.get_provenance('a', dates[:2])
        assert isinstance(result, pd.Series)
        assert result.index.equals(dates[:2])

    def test_slice_returns_a_series(self):
        """A slice returns the matching rows."""
        tracker, dates = _tracker()
        assert len(tracker.get_provenance('a', slice(dates[1], dates[3]))) == 3

    def test_unfilled_cell_is_none(self):
        """A NaN input cell reads back as a null (NaN after ``initialize``)."""
        tracker, dates = _tracker()
        assert pd.isna(tracker.get_provenance('a', dates[1]))

    def test_unknown_column_raises(self):
        """An unknown column raises ValueError."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.get_provenance('zz', dates[0])

    def test_unknown_label_raises_key_error(self):
        """A label outside the matrix raises KeyError."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            tracker.get_provenance('a', pd.Timestamp('2030-01-01'))

    def test_uninitialized_raises(self):
        """Reading before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().get_provenance('a', pd.Timestamp('2023-01-01'))


# =============================================================================
# get_mask
# =============================================================================
class TestGetMask:
    """Boolean masks over the provenance matrix."""

    @staticmethod
    def _filled():
        tracker, dates = _tracker(values_b=(1.0, 2.0, np.nan, np.nan))
        tracker.mark_aggregated('a', dates[1])
        tracker.mark_interpolated('a', dates[3])
        tracker.mark_interpolated('b', dates[2])
        return tracker, dates

    def test_single_type_whole_frame(self):
        """A single type returns a boolean DataFrame over all columns."""
        tracker, _ = self._filled()
        mask = tracker.get_mask(P.INTERPOLATED)
        assert isinstance(mask, pd.DataFrame)
        assert mask.sum().to_dict() == {'a': 1, 'b': 1}

    def test_list_of_types_is_a_union(self):
        """A list selects cells matching ANY of the types."""
        tracker, _ = self._filled()
        mask = tracker.get_mask([P.INTERPOLATED, P.AGGREGATED], column='a')
        assert mask.tolist() == [False, True, False, True]

    def test_column_returns_a_series(self):
        """With ``column`` the mask is a Series."""
        tracker, dates = self._filled()
        mask = tracker.get_mask(P.ORIGINAL, column='a')
        assert isinstance(mask, pd.Series)
        assert mask.index.equals(dates)
        assert mask.tolist() == [True, False, True, False]

    def test_unfilled_cells_never_match(self):
        """None cells match no type."""
        tracker, _ = self._filled()
        # Colonne « a » entièrement renseignée, colonne « b » : dernière cellule vide
        assert tracker.get_mask(list(P), column='a').all()
        assert not tracker.get_mask(list(P), column='b').iloc[3]

    def test_empty_list_matches_nothing(self):
        """An empty type list gives an all-False mask."""
        tracker, _ = self._filled()
        assert not tracker.get_mask([]).to_numpy().any()

    def test_unknown_column_raises(self):
        """An unknown column raises ValueError."""
        tracker, _ = self._filled()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.get_mask(P.ORIGINAL, column='zz')

    def test_uninitialized_raises(self):
        """A mask before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().get_mask(P.ORIGINAL)


# =============================================================================
# compute_statistics
# =============================================================================
class TestComputeStatistics:
    """Counts and percentages per provenance type, overall and per column."""

    @staticmethod
    def _all_types_tracker():
        # Colonne « a » : une cellule par type (10) + 2 cellules vides ; colonne « b » vide
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'a': [np.nan] * 12, 'b': [np.nan] * 12}, index=dates)
        tracker = ImputationProvenanceTracker().initialize(data)
        for offset, prov_type in enumerate(P):
            tracker.mark_imputed('a', dates[offset], prov_type)
        return tracker

    def test_keys_cover_overall_columns_and_all_types(self):
        """One entry per column + 'overall', each with a count and a pct per type."""
        stats = self._all_types_tracker().compute_statistics()
        assert set(stats) == {'overall', 'a', 'b'}
        expected_keys = (
            {m.value for m in P} | {f'{m.value}_pct' for m in P} | {'not_imputed', 'not_imputed_pct'}
        )
        for scope in stats.values():
            assert set(scope) == expected_keys

    def test_golden_counts_and_percentages(self):
        """Golden values on 12 cells: 10 typed, 2 unfilled in ``a``; ``b`` empty."""
        stats = self._all_types_tracker().compute_statistics()
        # Colonne « a » : 1 cellule par type sur 12 => 100/12 % chacune
        for member in P:
            assert stats['a'][member.value] == 1
            assert stats['a'][f'{member.value}_pct'] == pytest.approx(100 / 12)
        assert stats['a']['not_imputed'] == 2
        # Colonne « b » : 12 cellules vides
        assert stats['b']['not_imputed'] == 12
        assert stats['b']['not_imputed_pct'] == pytest.approx(100.0)
        # Global : 24 cellules, 10 typées, 14 vides
        assert stats['overall']['model_on_true'] == 1
        assert stats['overall']['model_on_true_pct'] == pytest.approx(100 / 24)
        assert stats['overall']['not_imputed'] == 14

    def test_percentages_sum_to_one_hundred(self):
        """For each scope, type percentages + not_imputed_pct total 100."""
        stats = self._all_types_tracker().compute_statistics()
        for scope in stats.values():
            total = sum(v for k, v in scope.items() if k.endswith('_pct'))
            assert total == pytest.approx(100.0)

    def test_counts_sum_to_the_number_of_cells(self):
        """Counts of all types + not_imputed total the matrix size."""
        tracker = self._all_types_tracker()
        stats = tracker.compute_statistics()
        counts = sum(v for k, v in stats['overall'].items() if not k.endswith('_pct'))
        assert counts == tracker.provenance_matrix_.size

    def test_result_is_stored_in_statistics_attribute(self):
        """The returned dict is also kept in ``statistics_``."""
        tracker = self._all_types_tracker()
        assert tracker.statistics_ is None
        assert tracker.compute_statistics() is tracker.statistics_

    def test_recomputation_reflects_new_marks(self):
        """Statistics are recomputed from the current matrix at each call."""
        tracker, dates = _tracker()
        assert tracker.compute_statistics()['a']['interpolated'] == 0
        tracker.mark_interpolated('a', dates[1])
        assert tracker.compute_statistics()['a']['interpolated'] == 1

    def test_fully_unfilled_dataset(self):
        """A dataset with no value at all: everything is not_imputed, no ZeroDivision."""
        tracker, _ = _tracker(values_a=(np.nan,) * 4)
        stats = tracker.compute_statistics()
        assert stats['overall']['not_imputed_pct'] == pytest.approx(100.0)
        assert stats['overall']['original'] == 0

    def test_zero_row_matrix_yields_zero_percentages(self):
        """A matrix emptied of rows reports 0.0 percentages instead of dividing by zero."""
        tracker, _ = _tracker()
        tracker.provenance_matrix_ = tracker.provenance_matrix_.iloc[0:0]
        stats = tracker.compute_statistics()
        assert stats['overall']['original_pct'] == 0.0
        assert stats['a']['not_imputed_pct'] == 0.0

    def test_uninitialized_raises(self):
        """Statistics before ``initialize`` raise ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().compute_statistics()

    def test_module_function_matches_the_tracker(self):
        """``provenance_statistics`` on the tracker's matrix gives the very same census.

        The function serves the matrices ``HighFrequencyImputer.transform``
        assembles stage by stage, which no tracker holds.
        """
        tracker = self._all_types_tracker()
        expected = tracker.compute_statistics()

        actual = provenance_statistics(tracker.get_provenance_matrix())

        assert actual == expected

    def test_module_function_on_a_frame_without_a_tracker(self):
        """A plain frame of marks and missing values is counted like a tracker's matrix."""
        matrix = pd.DataFrame({
            'm1': [P.AGGREGATED, P.AGGREGATED, None],
            'a1': [P.MODEL_ON_TRUE, None, None],
        })

        stats = provenance_statistics(matrix)

        assert stats['m1']['aggregated'] == 2
        assert stats['a1']['not_imputed_pct'] == pytest.approx(200 / 3)
        assert stats['overall']['not_imputed'] == 3


# =============================================================================
# get_provenance_matrix / to_string_matrix / repr
# =============================================================================
class TestMatrixViews:
    """Copies and string rendering of the provenance matrix."""

    def test_get_provenance_matrix_is_an_independent_copy(self):
        """Mutating the returned frame does not alter the tracker."""
        tracker, dates = _tracker()
        copy = tracker.get_provenance_matrix()
        copy.loc[dates[0], 'a'] = P.MODEL_ON_TRUE
        assert tracker.get_provenance('a', dates[0]) is P.ORIGINAL

    def test_get_provenance_matrix_equals_the_attribute(self):
        """The copy has the same content as ``provenance_matrix_``."""
        tracker, _ = _tracker()
        pd.testing.assert_frame_equal(tracker.get_provenance_matrix(), tracker.provenance_matrix_)

    def test_get_provenance_matrix_uninitialized_raises(self):
        """Reading the matrix before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().get_provenance_matrix()

    def test_to_string_matrix_golden(self):
        """Members become their value, unfilled cells 'not_imputed'."""
        tracker, dates = _tracker()
        tracker.mark_model_imputed('a', dates[1], covariate_taint='imputed')
        result = tracker.to_string_matrix()
        assert result['a'].tolist() == ['original', 'model_on_imputed', 'original', 'not_imputed']
        assert result['b'].tolist() == ['not_imputed'] * 4

    def test_to_string_matrix_keeps_shape_and_labels(self):
        """Index and columns are those of the matrix."""
        tracker, _ = _tracker()
        result = tracker.to_string_matrix()
        assert result.index.equals(tracker.provenance_matrix_.index)
        assert list(result.columns) == ['a', 'b']

    def test_to_string_matrix_holds_plain_strings(self):
        """No ProvenanceType instance is left: every cell is a bare str value."""
        tracker, _ = _tracker()
        cells = tracker.to_string_matrix().to_numpy().ravel()
        assert all(type(cell) is str for cell in cells)

    def test_matrix_dtypes(self):
        """String view in ``str``; the matrix itself in ``PROVENANCE_DTYPE``, read back by value."""
        tracker, dates = _tracker()
        tracker.mark_interpolated('a', dates[1])

        assert (tracker.to_string_matrix().dtypes == 'str').all()
        assert (tracker.provenance_matrix_.dtypes == PROVENANCE_DTYPE).all()
        assert tracker.provenance_matrix_.loc[dates[1], 'a'] == P.INTERPOLATED

    def test_matrix_refuses_values_outside_the_vocabulary(self):
        """The categorical matrix refuses a value that is not a ``ProvenanceType`` value."""
        tracker, dates = _tracker()
        with pytest.raises(TypeError, match='new category'):
            tracker.provenance_matrix_.loc[dates[0], 'a'] = 'legacy_value'

    def test_to_string_matrix_uninitialized_raises(self):
        """Rendering before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().to_string_matrix()

    def test_repr(self):
        """The repr gives the matrix shape, or 'not initialized'."""
        assert repr(ImputationProvenanceTracker()) == 'ImputationProvenanceTracker(not initialized)'
        tracker, _ = _tracker()
        assert repr(tracker) == 'ImputationProvenanceTracker(rows=4, cols=2)'


# =============================================================================
# merge
# =============================================================================
class TestMerge:
    """Combination of the matrices of two trackers of the same shape."""

    @staticmethod
    def _pair():
        # self : a = [ORIGINAL, AGGREGATED, ORIGINAL, None]
        # other: a = [INTERPOLATED, INTERPOLATED, ORIGINAL, None]
        base, dates = _tracker()
        base.mark_aggregated('a', dates[1])
        other, _ = _tracker()
        other.mark_interpolated('a', dates[[0, 1]])
        return base, other, dates

    def test_update_other_overwrites_where_it_is_set(self):
        """``overwrite=True``: filled cells of ``other`` win, overlapping or not."""
        base, other, _ = self._pair()
        base.merge(other, overwrite=True)
        column = base.provenance_matrix_['a']
        assert column.iloc[:3].tolist() == [P.INTERPOLATED, P.INTERPOLATED, P.ORIGINAL]
        assert pd.isna(column.iloc[3])

    def test_update_never_erases_with_a_none(self):
        """``overwrite=True``: a null in ``other`` leaves ``self`` untouched."""
        base, other, dates = self._pair()
        other.clear_provenance('a', dates[2])
        base.merge(other, overwrite=True)
        assert base.get_provenance('a', dates[2]) is P.ORIGINAL

    def test_overwrite_is_the_default(self):
        """No ``overwrite`` argument means ``overwrite=True``."""
        base, other, _ = self._pair()
        base.merge(other)
        assert base.provenance_matrix_['a'].iloc[0] == P.INTERPOLATED

    def test_preserve_keeps_self_and_fills_none(self):
        """``overwrite=False``: conflicts resolve to ``self``, null cells are filled from ``other``."""
        base, other, dates = self._pair()
        other.mark_aggregated('a', dates[3])
        base.merge(other, overwrite=False)
        assert base.provenance_matrix_['a'].tolist() == [
            P.ORIGINAL, P.AGGREGATED, P.ORIGINAL, P.AGGREGATED,
        ]

    def test_merge_returns_self(self):
        """``merge`` is chainable and mutates the receiver."""
        base, other, _ = self._pair()
        assert base.merge(other) is base

    def test_merge_does_not_mutate_other(self):
        """The merged-in tracker is left untouched."""
        base, other, _ = self._pair()
        before = other.provenance_matrix_.copy()
        base.merge(other)
        pd.testing.assert_frame_equal(other.provenance_matrix_, before)

    def test_merge_invalidates_statistics(self):
        """Statistics computed before a merge are reset to None."""
        base, other, _ = self._pair()
        base.compute_statistics()
        base.merge(other)
        assert base.statistics_ is None

    def test_merge_handles_each_column_independently(self):
        """Overlap is handled per column: ``b`` is merged, ``a`` is not touched."""
        base, dates = _tracker(values_b=(1.0, np.nan, np.nan, np.nan))
        other, _ = _tracker(values_b=(1.0, np.nan, np.nan, np.nan))
        other.mark_interpolated('b', dates[2])
        base.merge(other)
        column_b = base.provenance_matrix_['b']
        assert column_b.iloc[[0, 2]].tolist() == [P.ORIGINAL, P.INTERPOLATED]
        assert pd.isna(column_b.iloc[1])
        assert base.provenance_matrix_['a'].iloc[0] == P.ORIGINAL

    def test_how_argument_is_gone(self):
        """The former ``how`` strategy is replaced by ``overwrite``."""
        base, other, _ = self._pair()
        with pytest.raises(TypeError):
            base.merge(other, how='update')

    def test_merge_never_adds_cells(self):
        """Only cells ``self`` already carries are updated: shape and labels are unchanged."""
        base, other, _ = self._pair()
        before_index, before_columns = base.provenance_matrix_.index, base.provenance_matrix_.columns
        base.merge(other)
        assert base.provenance_matrix_.index.equals(before_index)
        assert base.provenance_matrix_.columns.equals(before_columns)

    def test_column_order_may_differ(self):
        """The same columns in another order are compatible (label alignment)."""
        base, dates = _tracker()
        other, _ = _tracker()
        other.mark_interpolated('a', dates[1])
        other.provenance_matrix_ = other.provenance_matrix_[['b', 'a']]
        base.merge(other)
        assert base.get_provenance('a', dates[1]) is P.INTERPOLATED

    def test_unsorted_identical_index_merges(self):
        """Two unsorted matrices sharing the very same row order merge."""
        dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan]}, index=dates).iloc[[2, 0, 3, 1]]
        base = ImputationProvenanceTracker().initialize(data)
        other = ImputationProvenanceTracker().initialize(data)
        other.mark_interpolated('a', dates[0])
        base.merge(other)
        assert base.get_provenance('a', dates[0]) is P.INTERPOLATED

    def test_incompatible_shapes_raise(self):
        """Different shapes raise ValueError naming both."""
        base, _, _ = self._pair()
        narrow = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [1.0] * 4}, index=pd.date_range('2023-01-01', periods=4, freq='MS'))
        )
        with pytest.raises(ValueError, match='Incompatible shapes'):
            base.merge(narrow)

    def test_uninitialized_self_raises(self):
        """An uninitialized receiver raises ValueError."""
        _, other, _ = self._pair()
        with pytest.raises(ValueError, match="This tracker"):
            ImputationProvenanceTracker().merge(other)

    def test_uninitialized_other_raises(self):
        """An uninitialized argument raises ValueError."""
        base, _, _ = self._pair()
        with pytest.raises(ValueError, match="Other tracker"):
            base.merge(ImputationProvenanceTracker())

    def test_same_shape_but_other_columns_raises(self):
        """Same shape is not enough, unmatched columns raise."""
        base, dates = _tracker()
        other = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'x': [1.0] * 4, 'y': [2.0] * 4}, index=dates)
        )
        before = base.provenance_matrix_.copy()
        with pytest.raises(ValueError, match='Incompatible labels'):
            base.merge(other)
        pd.testing.assert_frame_equal(base.provenance_matrix_, before)

    def test_same_shape_but_other_index_raises(self):
        """Same shape but other row labels raises."""
        base, _ = _tracker()
        shifted = pd.date_range('2030-01-01', periods=4, freq='MS')
        other = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [1.0] * 4, 'b': [1.0] * 4}, index=shifted)
        )
        with pytest.raises(ValueError, match='Incompatible labels'):
            base.merge(other)

    def test_same_labels_in_another_row_order_raises(self):
        """Rows must be identical, order included, so rows are paired as expected."""
        base, _ = _tracker()
        other, _ = _tracker()
        other.provenance_matrix_ = other.provenance_matrix_.iloc[::-1]
        with pytest.raises(ValueError, match='Incompatible labels'):
            base.merge(other)


# =============================================================================
# Panel à couverture hétérogène
# =============================================================================
class TestHeterogeneousCoveragePanel:
    """Marking one entity of the realistic panel leaves the others untouched."""

    @staticmethod
    def _entity_index(panel, entity):
        return pd.MultiIndex.from_product(
            [[entity], panel.loc[entity].index], names=panel.index.names
        )

    def test_initialization_reflects_per_entity_coverage(self, heterogeneous_coverage_panel):
        """climat_affaires has no ORIGINAL cell for Italie, one per observation elsewhere."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        by_entity = tracker.get_mask(P.ORIGINAL, column='climat_affaires').groupby(level=0).sum()
        # Valeur d'or : le nombre de cellules ORIGINAL est le nombre d'observations
        # de l'entité ; Italie n'observe jamais la colonne (couverture structurelle)
        assert by_entity['Italie'] == 0
        assert by_entity.to_dict() == {
            e: int(panel.loc[e, 'climat_affaires'].notna().sum())
            for e in ['France', 'Allemagne', 'Italie']
        }

    def test_marking_italie_leaves_france_and_allemagne_untouched(self, heterogeneous_coverage_panel):
        """Interpolating Italie's climat_affaires modifies no other cell."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        before = tracker.provenance_matrix_.copy()

        tracker.mark_interpolated('climat_affaires', self._entity_index(panel, 'Italie'))
        after = tracker.provenance_matrix_

        # Aucune cellule France / Allemagne (toutes colonnes) n'a bougé
        others = after.index.get_level_values(0) != 'Italie'
        pd.testing.assert_frame_equal(after[others], before[others])
        # Italie : colonne ciblée entièrement INTERPOLATED, autres colonnes inchangées
        italie = after.loc['Italie']
        assert (italie['climat_affaires'] == P.INTERPOLATED).all()
        pd.testing.assert_frame_equal(
            italie.drop(columns='climat_affaires'),
            before.loc['Italie'].drop(columns='climat_affaires'),
        )

    def test_marking_with_a_multiindex_slice_stays_inside_the_entity(self, heterogeneous_coverage_panel):
        """A label slice from (Italie, first) to (Italie, last) stays in Italie."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        dates = panel.loc['Italie'].index
        tracker.mark_aggregated('climat_affaires', slice(('Italie', dates[0]), ('Italie', dates[-1])))

        mask = tracker.get_mask(P.AGGREGATED, column='climat_affaires')
        assert mask.groupby(level=0).sum().to_dict() == {
            'Allemagne': 0, 'France': 0, 'Italie': len(dates),
        }

    def test_statistics_over_the_marked_entity(self, heterogeneous_coverage_panel):
        """Counts add up to the cells marked for Italie, and percentages total 100."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        tracker.mark_interpolated('climat_affaires', self._entity_index(panel, 'Italie'))
        stats = tracker.compute_statistics()
        assert stats['climat_affaires']['interpolated'] == len(panel.loc['Italie'])
        assert sum(v for k, v in stats['overall'].items() if k.endswith('_pct')) == pytest.approx(100.0)
