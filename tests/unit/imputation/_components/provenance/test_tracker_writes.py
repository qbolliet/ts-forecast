"""Initialization and writes of ``ImputationProvenanceTracker``.

Scenarios: matrix initialization, extension to a densified grid, marking wrappers and
clearing cells.
"""

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.imputation._components.provenance import ImputationProvenanceTracker

# Outils de test partagés du paquet
from tests.unit.imputation._components.provenance.helpers import P, _tracker


# =============================================================================
# ImputationProvenanceTracker — initialize
# =============================================================================
class TestInitialize:
    """Initialization of the provenance matrix from the input data."""

    def test_non_null_cells_are_original_and_null_cells_unmarked(self):
        """Golden values: ORIGINAL where the value exists, NaN elsewhere."""
        tracker, _ = _tracker()
        matrix = tracker.provenance_matrix_
        assert matrix['a'].tolist()[0] == P.ORIGINAL
        assert matrix['a'].tolist()[2] == P.ORIGINAL
        assert matrix['a'].isna().tolist() == [False, True, False, True]
        # Colonne entièrement NaN : aucune provenance
        assert matrix['b'].isna().all()

    def test_returns_self_and_keeps_index_and_columns(self):
        """``initialize`` is chainable and the matrix mirrors the data's labels."""
        data = pd.DataFrame({'a': [1.0, np.nan]}, index=pd.date_range('2023-01-01', periods=2, freq='MS'))
        tracker = ImputationProvenanceTracker()
        assert tracker.initialize(data) is tracker
        assert tracker.provenance_matrix_.index.equals(data.index)
        assert list(tracker.provenance_matrix_.columns) == ['a']

    @pytest.mark.parametrize('bad', [[1.0, 2.0], np.array([1.0]), None], ids=['list', 'ndarray', 'none'])
    def test_non_dataframe_raises(self, bad):
        """Anything but a DataFrame raises ValueError."""
        with pytest.raises(ValueError, match='DataFrame'):
            ImputationProvenanceTracker().initialize(bad)

    def test_series_is_rejected(self):
        """A Series is not a DataFrame: rejected like any other input."""
        series = pd.Series([1.0], index=pd.date_range('2023-01-01', periods=1))
        with pytest.raises(ValueError, match='Series'):
            ImputationProvenanceTracker().initialize(series)

    @pytest.mark.parametrize(
        'empty',
        [pd.DataFrame(), pd.DataFrame({'a': []})],
        ids=['no-column-no-row', 'column-no-row'],
    )
    def test_empty_frame_raises(self, empty):
        """An empty dataset (no row or no column) raises ValueError."""
        with pytest.raises(ValueError, match='empty'):
            ImputationProvenanceTracker().initialize(empty)

    def test_single_observation(self):
        """A one-row frame is a valid dataset."""
        tracker = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [5.0]}, index=pd.date_range('2023-01-31', periods=1))
        )
        assert tracker.provenance_matrix_.shape == (1, 1)
        assert tracker.provenance_matrix_.iloc[0, 0] == P.ORIGINAL

    def test_unsorted_index_keeps_input_order_and_pairing(self):
        """Disorder is preserved, each label keeps the provenance of its own value."""
        dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan]}, index=dates).iloc[[2, 0, 3, 1]]
        matrix = ImputationProvenanceTracker().initialize(data).provenance_matrix_
        assert matrix.index.equals(data.index)
        # Ordre d'entrée : mars (valeur), janvier (valeur), avril (NaN), février (NaN)
        assert matrix['a'].isna().tolist() == [False, False, True, True]

    def test_duplicated_index_keeps_every_row(self):
        """Duplicated labels are kept one row each, provenance follows each value."""
        dates = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0]}, index=dates)
        matrix = ImputationProvenanceTracker().initialize(data).provenance_matrix_
        assert len(matrix) == 3
        assert matrix['a'].isna().tolist() == [False, True, False]

    def test_mark_on_duplicated_label_marks_every_duplicate(self):
        """A label-based mark reaches all rows sharing the label."""
        dates = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0]}, index=dates)
        tracker = ImputationProvenanceTracker().initialize(data)
        tracker.mark_interpolated('a', pd.Timestamp('2023-01-01'))
        assert tracker.provenance_matrix_['a'].tolist() == [
            P.INTERPOLATED, P.INTERPOLATED, P.ORIGINAL,
        ]

    @pytest.mark.parametrize(
        'index',
        [
            pd.period_range('2023-01', periods=4, freq='M'),
            pd.RangeIndex(4),
            pd.date_range('2023-01-31', periods=4, freq='ME'),
        ],
        ids=['period', 'range', 'month-end'],
    )
    def test_index_types(self, index):
        """Period, Range and month-end indexes initialize and mark by label."""
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data)
        tracker.mark_interpolated('a', index[1])
        column = tracker.provenance_matrix_['a']
        assert column.iloc[:3].tolist() == [P.ORIGINAL, P.INTERPOLATED, P.ORIGINAL]
        assert pd.isna(column.iloc[3])

    def test_special_column_names(self):
        """Spaces, accents and symbols in column names are carried unchanged."""
        names = ['pib trimestriel', 'taux_chômage (%)', 'a/b']
        data = pd.DataFrame(
            {name: [1.0, np.nan] for name in names},
            index=pd.date_range('2023-01-01', periods=2, freq='MS'),
        )
        tracker = ImputationProvenanceTracker().initialize(data)
        assert list(tracker.provenance_matrix_.columns) == names
        tracker.mark_aggregated('taux_chômage (%)', data.index[1])
        assert tracker.get_provenance('taux_chômage (%)', data.index[1]) is P.AGGREGATED

    def test_multiindex_panel_is_auto_detected(self):
        """A MultiIndex panel tracks every column and records the entity level."""
        index = pd.MultiIndex.from_product(
            [['FR', 'DE'], pd.date_range('2023-01-01', periods=2, freq='MS')],
            names=['country', 'date'],
        )
        data = pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data)
        assert tracker._panel_cols == ['country']
        assert list(tracker.provenance_matrix_.columns) == ['a']
        assert tracker.provenance_matrix_.index.equals(index)

    def test_three_level_panel_records_both_entity_levels(self):
        """On a 3-level index all levels but the last are entity levels."""
        index = pd.MultiIndex.from_tuples(
            [('EU', 'FR', pd.Timestamp('2023-01-01')), ('EU', 'FR', pd.Timestamp('2023-02-01'))],
            names=['region', 'country', 'date'],
        )
        data = pd.DataFrame({'a': [1.0, np.nan]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data)
        assert tracker._panel_cols == ['region', 'country']
        assert list(tracker.provenance_matrix_.columns) == ['a']

    def test_flat_panel_excludes_declared_entity_columns(self):
        """``panel_cols`` naming ordinary columns removes them from the tracked ones."""
        data = pd.DataFrame(
            {'country': ['x', 'x', 'y', 'y'], 'a': [1.0, np.nan, np.nan, 4.0]},
            index=pd.date_range('2023-01-01', periods=4, freq='MS'),
        )
        tracker = ImputationProvenanceTracker().initialize(data, panel_cols=['country'])
        assert list(tracker.provenance_matrix_.columns) == ['a']

    def test_no_tracked_column_keeps_the_label_dtype(self):
        """Every column declared as panel: an empty matrix whose columns keep the ``str`` labels dtype."""
        data = pd.DataFrame(
            {'country': ['x', 'y']}, index=pd.date_range('2023-01-01', periods=2, freq='MS'),
        )
        tracker = ImputationProvenanceTracker().initialize(data, panel_cols=['country'])
        assert tracker.provenance_matrix_.shape == (2, 0)
        assert tracker.provenance_matrix_.columns.dtype == 'str'

    def test_flat_frame_without_panel_cols_tracks_every_column(self):
        """Without ``panel_cols`` and without MultiIndex, nothing is excluded."""
        data = pd.DataFrame(
            {'country': ['x', 'y'], 'a': [1.0, np.nan]},
            index=pd.date_range('2023-01-01', periods=2, freq='MS'),
        )
        tracker = ImputationProvenanceTracker().initialize(data)
        assert list(tracker.provenance_matrix_.columns) == ['country', 'a']

    def test_declared_panel_cols_matching_index_levels(self):
        """``panel_cols`` naming index levels is taken as already in the index."""
        index = pd.MultiIndex.from_product(
            [['FR'], pd.date_range('2023-01-01', periods=2, freq='MS')], names=['country', 'date']
        )
        data = pd.DataFrame({'a': [1.0, np.nan]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data, panel_cols=['country'])
        assert list(tracker.provenance_matrix_.columns) == ['a']


# =============================================================================
# extend_index
# =============================================================================
class TestExtendIndex:
    """Extension of the matrix to a densified grid wider than the input."""

    @staticmethod
    def _tracker(index):
        data = pd.DataFrame({'a': [1.0] * len(index), 'b': [2.0] * len(index)}, index=index)
        return ImputationProvenanceTracker().initialize(data)

    def test_new_rows_are_unfilled_and_sorted_in(self):
        """Added dates are unmarked (NaN) and take their chronological place."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-04-01']))
        grid = pd.date_range('2015-01-01', '2015-04-01', freq='MS')
        tracker.extend_index(grid)

        matrix = tracker.provenance_matrix_
        assert list(matrix.index) == list(grid)
        assert matrix['a'].isna().tolist() == [False, True, True, False]
        assert matrix['a'].iloc[[0, 3]].tolist() == [P.ORIGINAL, P.ORIGINAL]

    def test_new_rows_are_unfilled_in_every_column(self):
        """The added rows are None in all columns, not only one."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-03-01']))
        tracker.extend_index(pd.to_datetime(['2015-02-01']))
        assert tracker.provenance_matrix_.loc['2015-02-01'].isna().all()

    def test_extension_makes_mark_possible(self):
        """Without extension, marking an absent set of dates raises KeyError."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-04-01']))
        grid = pd.date_range('2015-01-01', '2015-04-01', freq='MS')
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', grid)

        tracker.extend_index(grid)
        tracker.mark_interpolated('a', grid)
        assert (tracker.provenance_matrix_['a'] == P.INTERPOLATED).all()

    def test_known_labels_are_a_no_op(self):
        """A grid already covered leaves the matrix strictly intact."""
        index = pd.date_range('2015-01-01', periods=3, freq='MS')
        tracker = self._tracker(index)
        before = tracker.provenance_matrix_.copy()
        tracker.extend_index(index[:2])
        pd.testing.assert_frame_equal(tracker.provenance_matrix_, before)

    def test_empty_index_is_a_no_op(self):
        """An empty index adds nothing."""
        tracker = self._tracker(pd.date_range('2015-01-01', periods=3, freq='MS'))
        before = tracker.provenance_matrix_.copy()
        tracker.extend_index(pd.DatetimeIndex([]))
        pd.testing.assert_frame_equal(tracker.provenance_matrix_, before)

    def test_existing_provenance_is_preserved(self):
        """Marks already written survive an extension."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-03-01']))
        tracker.mark_aggregated('a', pd.Timestamp('2015-03-01'))
        tracker.extend_index(pd.to_datetime(['2015-02-01']))
        assert tracker.get_provenance('a', pd.Timestamp('2015-03-01')) is P.AGGREGATED
        assert tracker.get_provenance('b', pd.Timestamp('2015-01-01')) is P.ORIGINAL

    def test_unsorted_matrix_keeps_its_order(self):
        """An unsorted matrix is not reordered: the new rows go at the end."""
        tracker = self._tracker(pd.to_datetime(['2015-03-01', '2015-01-01']))
        tracker.extend_index(pd.to_datetime(['2015-02-01']))
        assert list(tracker.provenance_matrix_.index) == list(
            pd.to_datetime(['2015-03-01', '2015-01-01', '2015-02-01'])
        )

    def test_panel_rows_land_inside_their_entity(self):
        """On a sorted MultiIndex the added row stays in its entity's block."""
        index = pd.MultiIndex.from_tuples(
            [('FR', pd.Timestamp('2015-01-01')), ('FR', pd.Timestamp('2015-03-01')),
             ('IT', pd.Timestamp('2015-01-01'))],
            names=['country', 'date'],
        )
        tracker = self._tracker(index)
        tracker.extend_index(pd.MultiIndex.from_tuples(
            [('FR', pd.Timestamp('2015-02-01'))], names=['country', 'date']
        ))
        assert tracker.provenance_matrix_.index.get_level_values('country').tolist() == [
            'FR', 'FR', 'FR', 'IT'
        ]
        assert tracker.provenance_matrix_['a'].isna().sum() == 1

    def test_uninitialized_raises(self):
        """Extending an uninitialized tracker raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().extend_index(pd.date_range('2015-01-01', periods=2))


# =============================================================================
# mark_* / clear_provenance / get_provenance
# =============================================================================
class TestMarks:
    """Writing provenance: ``mark_imputed`` and its convenience wrappers."""

    @pytest.mark.parametrize('provenance', list(P), ids=lambda m: m.name)
    def test_mark_imputed_writes_every_provenance_type(self, provenance):
        """``mark_imputed`` accepts all ten members."""
        tracker, dates = _tracker()
        tracker.mark_imputed('a', dates[1], provenance)
        assert tracker.get_provenance('a', dates[1]) is provenance

    @pytest.mark.parametrize(
        'method, expected',
        [
            ('mark_aggregated', P.AGGREGATED),
            ('mark_disaggregated', P.DISAGGREGATED),
            ('mark_interpolated', P.INTERPOLATED),
        ],
    )
    def test_convenience_wrappers(self, method, expected):
        """Each wrapper writes its own provenance type."""
        tracker, dates = _tracker()
        getattr(tracker, method)('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is expected

    @pytest.mark.parametrize(
        'covariate_taint, target_taint, expected',
        [
            ('none', 'none', P.MODEL_ON_TRUE),
            ('interpolated', 'none', P.MODEL_ON_INTERPOLATED),
            ('imputed', 'none', P.MODEL_ON_IMPUTED),
            ('none', 'imputed', P.MODEL_ON_IMPUTED_TARGET),
            ('imputed', 'imputed', P.MODEL_ON_IMPUTED_BOTH),
        ],
        ids=['true', 'interpolated', 'imputed', 'target', 'both'],
    )
    def test_mark_model_imputed_reaches_the_five_labels(self, covariate_taint, target_taint, expected):
        """The five MODEL_* labels are reachable through the taint signature."""
        tracker, dates = _tracker()
        tracker.mark_model_imputed(
            'a', dates[1], covariate_taint=covariate_taint, target_taint=target_taint
        )
        assert tracker.get_provenance('a', dates[1]) is expected

    def test_mark_model_imputed_defaults_to_model_on_true(self):
        """Without taint arguments the cell is MODEL_ON_TRUE."""
        tracker, dates = _tracker()
        tracker.mark_model_imputed('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is P.MODEL_ON_TRUE

    def test_mark_model_imputed_has_no_trained_on_imputed_argument(self):
        """``mark_model_imputed`` takes no ``trained_on_imputed`` boolean."""
        tracker, dates = _tracker()
        with pytest.raises(TypeError):
            tracker.mark_model_imputed('a', dates[1], trained_on_imputed=True)

    def test_mark_with_datetimeindex(self):
        """A DatetimeIndex marks several cells at once, others untouched."""
        tracker, dates = _tracker()
        tracker.mark_interpolated('a', dates[[1, 3]])
        assert tracker.provenance_matrix_['a'].tolist() == [
            P.ORIGINAL, P.INTERPOLATED, P.ORIGINAL, P.INTERPOLATED,
        ]

    def test_mark_with_slice_is_label_inclusive(self):
        """A label slice includes both bounds."""
        tracker, dates = _tracker()
        tracker.mark_aggregated('a', slice(dates[1], dates[2]))
        column = tracker.provenance_matrix_['a']
        assert column.iloc[:3].tolist() == [P.ORIGINAL, P.AGGREGATED, P.AGGREGATED]
        assert pd.isna(column.iloc[3])

    def test_mark_leaves_other_columns_untouched(self):
        """Writing column ``a`` never changes column ``b``."""
        tracker, dates = _tracker(values_b=(1.0, np.nan, np.nan, np.nan))
        before = tracker.provenance_matrix_['b'].copy()
        tracker.mark_interpolated('a', dates)
        pd.testing.assert_series_equal(tracker.provenance_matrix_['b'], before)

    def test_partially_absent_index_raises_key_error(self):
        """A list-like with a label outside the matrix raises KeyError."""
        tracker, dates = _tracker()
        partial = pd.DatetimeIndex([dates[1], pd.Timestamp('2030-01-01')])
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', partial)

    def test_fully_absent_index_raises_key_error(self):
        """A list-like entirely outside the matrix raises KeyError."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', pd.DatetimeIndex(['2030-01-01', '2030-02-01']))
        assert len(tracker.provenance_matrix_) == 4

    def test_slice_outside_the_matrix_is_a_no_op(self):
        """A label slice entirely past the matrix marks nothing and adds no row."""
        tracker, _ = _tracker()
        before = tracker.provenance_matrix_.copy()
        tracker.mark_interpolated(
            'a', slice(pd.Timestamp('2030-01-01'), pd.Timestamp('2031-01-01'))
        )
        pd.testing.assert_frame_equal(tracker.provenance_matrix_, before)

    @pytest.mark.parametrize(
        'method', ['mark_aggregated', 'mark_disaggregated', 'mark_interpolated', 'mark_model_imputed'],
    )
    def test_scalar_timestamp_outside_the_matrix_raises_key_error(self, method):
        """An absent scalar raises like a list-like, adding no row."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            getattr(tracker, method)('a', pd.Timestamp('2030-01-01'))
        assert tracker.provenance_matrix_.shape == (4, 2)

    def test_scalar_timestamp_inside_the_matrix_still_marks(self):
        """A present scalar label is written as before."""
        tracker, dates = _tracker()
        tracker.mark_interpolated('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is P.INTERPOLATED

    def test_multiindex_key_outside_the_matrix_raises_key_error(self):
        """On a panel, an absent (entity, date) key raises; a present one marks."""
        index = pd.MultiIndex.from_product(
            [['FR', 'DE'], pd.date_range('2023-01-01', periods=2, freq='MS')],
            names=['country', 'date'],
        )
        tracker = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0]}, index=index)
        )
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', ('IT', pd.Timestamp('2023-01-01')))
        assert tracker.provenance_matrix_.shape == (4, 1)
        tracker.mark_interpolated('a', ('FR', pd.Timestamp('2023-02-01')))
        assert tracker.get_provenance('a', ('FR', pd.Timestamp('2023-02-01'))) is P.INTERPOLATED

    def test_unknown_column_raises(self):
        """Marking an unknown column raises ValueError naming it."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.mark_imputed('zz', dates[0], P.INTERPOLATED)

    @pytest.mark.parametrize('bad', ['original', None, 3], ids=['str', 'none', 'int'])
    def test_non_provenance_type_raises(self, bad):
        """Only ProvenanceType members are accepted, not even their string value."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match='ProvenanceType'):
            tracker.mark_imputed('a', dates[0], bad)

    def test_uninitialized_raises(self):
        """Marking before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().mark_imputed('a', pd.Timestamp('2023-01-01'), P.ORIGINAL)

    # Priorité d'écrasement : aucune. La dernière écriture l'emporte, y compris
    # sur ORIGINAL (une date-ancre ré-exprimée ne porte plus l'observation)
    # et y compris vers une provenance « plus propre » (pas de max de souillure :
    # la souillure se calcule dans le plan, jamais dans la matrice).
    @pytest.mark.parametrize(
        'first, second',
        [
            (P.ORIGINAL, P.MODEL_ON_TRUE),
            (P.MODEL_ON_IMPUTED_BOTH, P.INTERPOLATED),
            (P.INTERPOLATED, P.MODEL_ON_IMPUTED_BOTH),
            (P.MODEL_ON_TRUE, P.MODEL_UNANCHORED),
            (P.AGGREGATED, P.AGGREGATED),
        ],
        ids=['original-overwritten', 'dirty-to-clean', 'clean-to-dirty', 'to-unanchored', 'same'],
    )
    def test_last_write_wins(self, first, second):
        """Overwriting an existing provenance: the last write wins, no priority."""
        tracker, dates = _tracker()
        tracker.mark_imputed('a', dates[0], first)
        tracker.mark_imputed('a', dates[0], second)
        assert tracker.get_provenance('a', dates[0]) is second

    def test_marking_an_original_cell_overwrites_original(self):
        """An anchor date overwritten by a prediction is not ORIGINAL."""
        tracker, dates = _tracker()
        assert tracker.get_provenance('a', dates[0]) is P.ORIGINAL
        tracker.mark_model_imputed('a', dates[0])
        assert tracker.get_provenance('a', dates[0]) is P.MODEL_ON_TRUE


class TestClearProvenance:
    """Reset of cells to "not filled"."""

    def test_clear_resets_to_none(self):
        """A cleared cell goes back to the NaN-cell convention (None)."""
        tracker, dates = _tracker()
        tracker.clear_provenance('a', dates[0])
        assert tracker.get_provenance('a', dates[0]) is None

    def test_clear_then_mark(self):
        """A cleared cell can be marked again."""
        tracker, dates = _tracker()
        tracker.clear_provenance('a', dates[0])
        tracker.mark_aggregated('a', dates[0])
        assert tracker.get_provenance('a', dates[0]) is P.AGGREGATED

    def test_clear_several_cells_and_slice(self):
        """DatetimeIndex and slice clear several cells, others untouched."""
        tracker, dates = _tracker(values_a=(1.0, 2.0, 3.0, 4.0))
        tracker.clear_provenance('a', dates[[0, 1]])
        tracker.clear_provenance('a', slice(dates[3], dates[3]))
        assert tracker.provenance_matrix_['a'].isna().tolist() == [True, True, False, True]

    def test_clear_already_empty_cell_is_idempotent(self):
        """Clearing a None cell leaves it None."""
        tracker, dates = _tracker()
        tracker.clear_provenance('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is None

    def test_unknown_column_raises(self):
        """An unknown column raises ValueError."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.clear_provenance('zz', dates[0])

    def test_scalar_timestamp_outside_the_matrix_raises_key_error(self):
        """Clearing an absent scalar adds no row either."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            tracker.clear_provenance('a', pd.Timestamp('2030-01-01'))
        assert tracker.provenance_matrix_.shape == (4, 2)

    def test_uninitialized_raises(self):
        """Clearing before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().clear_provenance('a', pd.Timestamp('2023-01-01'))
