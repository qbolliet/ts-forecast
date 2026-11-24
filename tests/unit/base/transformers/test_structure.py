"""Tests of the structure handling of ``PanelTimeSeriesTransformer``.

Covers the detection of the data layout through ``fit`` (series or panel, time
and entities in columns or in the index, ``is_panel_``), the invalid layouts and
their messages, ``convert_cols_to_index`` and the restoration of the original
columns, the consistency check of the panel identifiers (missing values, missing
columns), and entities that are missing, unbalanced or unseen at ``fit``.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from tsforecast.panel.utils import is_panel_data
from tsforecast.utils.validation import restore_original_structure
from tests.support.perturbations import drop_entity, shuffle_rows, to_three_level_index

from .conftest import CenteredScaler, ConcreteTransformer, month_ends


def _panel_issue_messages(caught) -> list:
    """Messages of the recorded warnings that report a panel structure issue."""
    return [str(w.message) for w in caught if 'Panel structure issues' in str(w.message)]


# =============================================================================
# Détection de la structure
# =============================================================================
class TestStructureDetection:
    """``is_panel_`` and the output layout follow the configuration and the index."""

    def test_datetime_index_is_a_series(self, series_frame):
        """A ``DatetimeIndex`` without configuration is a plain time series."""
        transformer = ConcreteTransformer().fit(series_frame)
        assert transformer.is_panel_ is False

    def test_time_col_alone_is_a_series(self, series_frame):
        """A ``time_col`` without ``panel_cols`` is a time series stored in columns."""
        transformer = ConcreteTransformer(time_col='date').fit(series_frame.reset_index())
        assert transformer.is_panel_ is False

    def test_time_col_and_panel_cols_make_a_panel(self, column_panel):
        """``time_col`` plus ``panel_cols`` is a panel stored in columns."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country']).fit(column_panel)
        assert transformer.is_panel_ is True

    @pytest.mark.parametrize('levels', [2, 3], ids=['two-levels', 'three-levels'])
    def test_multiindex_is_a_panel(self, index_panel, levels):
        """A ``MultiIndex`` of at least two levels is a panel."""
        panel = index_panel if levels == 2 else to_three_level_index(index_panel)
        assert ConcreteTransformer().fit(panel).is_panel_ is True

    def test_time_col_absent_from_a_multiindex_reads_the_panel(self, index_panel):
        """A ``time_col`` that is not a column of a ``MultiIndex`` panel leaves the panel in the index.

        The entities (``FR`` then ``DE``) are not in increasing order: read as one
        series, the index would be refused as unsorted.
        """
        transformer = ConcreteTransformer(time_col='date').fit(index_panel)

        assert transformer.is_panel_ is True
        pd.testing.assert_frame_equal(transformer.fit_input_, index_panel)

    def test_single_entity_multiindex_is_a_panel_like_for_is_panel_data(self, index_panel):
        """A one-entity ``MultiIndex`` is a panel, as for ``tsforecast.panel.is_panel_data``.

        The panel flag describes the layout of the index, not the number of entities.
        """
        panel = drop_entity(index_panel, 'DE')
        assert ConcreteTransformer().fit(panel).is_panel_ is is_panel_data(panel) is True

    def test_single_entity_column_panel_is_a_panel(self, column_panel):
        """A column panel with a single entity is still flagged as a panel."""
        panel = column_panel[column_panel['country'] == 'FR']
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country']).fit(panel)
        assert transformer.is_panel_ is True

    def test_converted_column_panel_is_indexed_by_entity_then_date(self, column_panel, index_panel):
        """Under ``convert_cols_to_index=True``, the column panel reaches ``_fit`` indexed by ``(country, date)``."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country'], convert_cols_to_index=True)
        transformer.fit(column_panel)
        pd.testing.assert_frame_equal(transformer.fit_input_, index_panel, check_freq=False)

    def test_unconverted_column_panel_keeps_its_columns(self, column_panel):
        """By default, the validated column panel reaches ``_fit`` as it was given."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country']).fit(column_panel)
        pd.testing.assert_frame_equal(transformer.fit_input_, column_panel)


class TestInvalidStructures:
    """Layouts that cannot be read raise a ``ValueError`` with an explicit message."""

    def test_panel_cols_without_time_col(self, column_panel):
        """``panel_cols`` without ``time_col`` is an invalid configuration."""
        with pytest.raises(ValueError, match='Invalid configuration: panel_cols specified but time_col is None'):
            ConcreteTransformer(panel_cols=['country']).fit(column_panel)

    @pytest.mark.parametrize('validate_input', [True, False], ids=['validated', 'not-validated'])
    def test_single_level_multiindex(self, series_frame, validate_input):
        """A one-level ``MultiIndex`` has no entity level, with or without validation."""
        frame = series_frame.copy()
        frame.index = pd.MultiIndex.from_arrays([series_frame.index])

        with pytest.raises(ValueError, match='MultiIndex must have at least 2 levels for panel data'):
            ConcreteTransformer(validate_input=validate_input).fit(frame)

    @pytest.mark.parametrize('validate_input', [True, False], ids=['validated', 'not-validated'])
    @pytest.mark.parametrize(
        ('data', 'kind'), [([[1.0], [2.0]], 'list'), (np.array([[1.0], [2.0]]), 'ndarray')],
        ids=['list', 'ndarray'],
    )
    def test_non_pandas_input(self, data, kind, validate_input):
        """Only a ``DataFrame`` carries the time information, with or without validation."""
        with pytest.raises(ValueError, match=f'X must be a pandas DataFrame, got {kind}$'):
            ConcreteTransformer(validate_input=validate_input).fit(data)


# =============================================================================
# Structures en colonnes et en index : valeurs de sortie
# =============================================================================
class TestColumnBasedStructure:
    """Panels and series stored in columns are transformed on their value columns."""

    def test_column_panel_values(self, column_panel):
        """Only the value columns are scaled; entity and date stay columns."""
        output = CenteredScaler(time_col='date', panel_cols=['country']).fit_transform(column_panel)

        # Valeur d'or : moyenne poolée de a = 4.5, (a - 4.5) * 2
        assert list(output.columns) == ['country', 'date', 'a', 'b']
        assert output['a'].tolist() == [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]

    def test_column_series_values(self, series_frame):
        """A series in columns keeps its date column and is scaled."""
        frame = series_frame.reset_index()
        output = CenteredScaler(time_col='date').fit_transform(frame)

        # Valeur d'or : moyenne 2.5, (v - 2.5) * 2
        assert output['value'].tolist() == [-3.0, -1.0, 1.0, 3.0]
        pd.testing.assert_series_equal(output['date'], frame['date'])

    def test_converted_column_series_is_indexed_by_its_dates(self, series_frame):
        """Under ``convert_cols_to_index=True``, a series in columns is indexed by its dates."""
        output = CenteredScaler(time_col='date', convert_cols_to_index=True).fit_transform(series_frame.reset_index())

        pd.testing.assert_index_equal(output.index, month_ends(), exact=False)


class TestIndexBasedStructure:
    """Panels and series stored in the index keep their index through ``fit_transform``."""

    def test_datetime_index_series_values(self, series_frame):
        """A ``DatetimeIndex`` series keeps its index and is scaled."""
        output = CenteredScaler().fit_transform(series_frame)

        pd.testing.assert_index_equal(output.index, series_frame.index)
        assert output['value'].tolist() == [-3.0, -1.0, 1.0, 3.0]

    def test_two_level_panel_values(self, index_panel):
        """A ``(country, date)`` panel keeps its index and both columns are scaled."""
        output = CenteredScaler().fit_transform(index_panel)

        # Valeur d'or : moyennes poolées 4.5 et 45
        pd.testing.assert_index_equal(output.index, index_panel.index)
        assert output['b'].tolist() == [-70.0, -50.0, -30.0, -10.0, 10.0, 30.0, 50.0, 70.0]

    def test_three_level_panel_keeps_its_levels(self, index_panel):
        """A ``(region, country, date)`` panel keeps its three named levels."""
        panel = to_three_level_index(index_panel)

        output = CenteredScaler().fit_transform(panel)

        assert list(output.index.names) == ['region', 'country', 'date']
        assert output['a'].tolist() == [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]

    def test_three_level_panel_with_one_date_per_entity(self):
        """Entities defined by two levels may each hold a single date."""
        index = pd.MultiIndex.from_arrays(
            [['US', 'US', 'FR', 'FR'], ['Tech', 'Finance', 'Tech', 'Finance'], [pd.Timestamp('2023-01-31')] * 4],
            names=['country', 'sector', 'date'],
        )
        panel = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0]}, index=index)

        output = CenteredScaler().fit_transform(panel)

        pd.testing.assert_index_equal(output.index, panel.index)
        assert output['value'].tolist() == [-3.0, -1.0, 1.0, 3.0]

    def test_unsorted_entities_are_sorted_by_auto_sort(self):
        """``auto_sort=True`` puts entity blocks in index order, values following their rows."""
        index = pd.MultiIndex.from_arrays(
            [['B', 'B', 'A', 'A'], list(pd.date_range('2023-01-03', periods=2)) + list(pd.date_range('2023-01-01', periods=2))],
            names=['entity', 'date'],
        )
        panel = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0]}, index=index)

        output = ConcreteTransformer(auto_sort=True).fit_transform(panel)

        assert output.index.get_level_values('entity').tolist() == ['A', 'A', 'B', 'B']
        assert output['value'].tolist() == [3.0, 4.0, 1.0, 2.0]

    def test_convert_cols_to_index_leaves_a_multiindex_as_is(self, index_panel):
        """``convert_cols_to_index`` has nothing to convert on a ``MultiIndex`` panel."""
        transformer = CenteredScaler(convert_cols_to_index=True)
        output = transformer.fit_transform(index_panel)

        pd.testing.assert_index_equal(output.index, index_panel.index)
        assert not hasattr(transformer, 'conversion_metadata_')


# =============================================================================
# convert_cols_to_index et restitution de la structure
# =============================================================================
class TestConvertColsToIndex:
    """Columns moved into the index are given back by ``inverse_transform``."""

    def test_round_trip_restores_the_columns(self, column_panel):
        """``inverse_transform(transform(X))`` gives the column panel back."""
        transformer = CenteredScaler(time_col='date', panel_cols=['country'], convert_cols_to_index=True)

        restored = transformer.inverse_transform(transformer.fit_transform(column_panel))

        pd.testing.assert_frame_equal(restored, column_panel, check_exact=False, rtol=1e-12)

    def test_inverse_matches_restore_original_structure(self, column_panel):
        """The restoration is the one of ``restore_original_structure`` with the stored metadata."""
        transformer = CenteredScaler(time_col='date', panel_cols=['country'], convert_cols_to_index=True)
        output = transformer.fit_transform(column_panel)

        # Inversion des valeurs à la main, puis restitution par l'utilitaire de validation
        unscaled = output / 2.0 + transformer.center_
        expected = restore_original_structure(unscaled, transformer.conversion_metadata_)

        pd.testing.assert_frame_equal(transformer.inverse_transform(output), expected)

    def test_round_trip_of_shuffled_rows_gives_the_sorted_frame(self, column_panel):
        """Rows sorted by ``auto_sort`` stay sorted, each value keeping its entity and date."""
        shuffled = shuffle_rows(column_panel, seed=0)
        transformer = CenteredScaler(
            time_col='date', panel_cols=['country'], convert_cols_to_index=True, auto_sort=True
        )

        restored = transformer.inverse_transform(transformer.fit_transform(shuffled))

        # Valeur d'or : lignes triées par (pays, date), étiquettes 0..n-1 neuves
        expected = column_panel.sort_values(['country', 'date']).reset_index(drop=True)
        pd.testing.assert_frame_equal(restored, expected, check_exact=False, rtol=1e-12)

    def test_column_series_round_trip(self, series_frame):
        """A series in columns gets its date column back."""
        frame = series_frame.reset_index()
        transformer = CenteredScaler(time_col='date', convert_cols_to_index=True)

        restored = transformer.inverse_transform(transformer.fit_transform(frame))

        pd.testing.assert_frame_equal(restored, frame, check_exact=False, rtol=1e-12)

    def test_without_conversion_the_columns_stay_columns(self, column_panel):
        """``convert_cols_to_index=False`` leaves the layout as is: no index replaced, no metadata."""
        transformer = CenteredScaler(time_col='date', panel_cols=['country'])

        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Index replaced')
            output = transformer.fit_transform(column_panel)

        pd.testing.assert_index_equal(output.index, column_panel.index)
        assert not hasattr(transformer, 'conversion_metadata_')

    def test_without_conversion_the_round_trip_restores_the_input(self, column_panel):
        """Nothing having moved, ``inverse_transform(transform(X))`` is the column panel itself."""
        transformer = CenteredScaler(time_col='date', panel_cols=['country'])

        restored = transformer.inverse_transform(transformer.fit_transform(column_panel))

        pd.testing.assert_frame_equal(restored, column_panel, check_exact=False, rtol=1e-12)

    def test_without_conversion_sorted_rows_get_fresh_labels(self, column_panel):
        """Rows sorted by ``auto_sort`` keep their columns and get fresh labels ``0..n-1``."""
        shuffled = shuffle_rows(column_panel, seed=0)

        output = ConcreteTransformer(time_col='date', panel_cols=['country'], auto_sort=True).fit_transform(shuffled)

        # Valeur d'or : lignes triées par (pays, date) ; DE avant FR dans l'ordre alphabétique
        expected = column_panel.sort_values(['country', 'date']).reset_index(drop=True)
        pd.testing.assert_frame_equal(output, expected)


# =============================================================================
# Cohérence des identifiants de panel
# =============================================================================
class TestPanelIdentifierConsistency:
    """Missing panel identifiers raise ``Panel structure issues`` under strict validation, else warn."""

    @pytest.fixture
    def panel_with_missing_entity(self) -> pd.DataFrame:
        """Panel whose second row has no entity label."""
        index = pd.MultiIndex.from_arrays(
            [['A', pd.NA, 'B', 'B'], list(pd.date_range('2023-01-01', periods=2)) * 2],
            names=['entity', 'date'],
        )
        return pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0]}, index=index)

    def test_consistent_panel_emits_no_issue(self, index_panel):
        """A clean panel raises no structure warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ConcreteTransformer().fit(index_panel)

        assert _panel_issue_messages(caught) == []

    def test_missing_entity_level_value_warns_when_not_strict(self, panel_with_missing_entity):
        """A missing entity label on the index is named by level in a warning."""
        with pytest.warns(UserWarning, match='Missing values in panel identifier level: entity'):
            ConcreteTransformer(strict_validation=False).fit(panel_with_missing_entity)

    def test_missing_entity_level_value_raises_in_strict_mode(self, panel_with_missing_entity):
        """In strict mode, a missing entity label is an error naming the level."""
        with pytest.raises(ValueError, match='Panel structure issues: Missing values in panel identifier level: entity'):
            ConcreteTransformer(strict_validation=True).fit(panel_with_missing_entity)

    def test_missing_entity_level_value_warns_without_validation(self, panel_with_missing_entity):
        """With validation disabled, the issue is only reported, whatever ``strict_validation``."""
        with pytest.warns(UserWarning, match='Missing values in panel identifier level: entity'):
            ConcreteTransformer(validate_input=False, strict_validation=True).fit(panel_with_missing_entity)

    def test_unnamed_entity_level_is_reported_by_position(self, panel_with_missing_entity):
        """An unnamed entity level is reported as ``level_0``."""
        panel = panel_with_missing_entity.rename_axis([None, None])
        with pytest.raises(ValueError, match='Missing values in panel identifier level: level_0'):
            ConcreteTransformer().fit(panel)

    def test_missing_entity_column_value_warns(self, column_panel):
        """A missing value in a ``panel_cols`` column is named in a warning."""
        panel = column_panel.copy()
        panel.loc[4, 'country'] = None

        with pytest.warns(UserWarning, match='Missing values in panel identifier: country'):
            ConcreteTransformer(time_col='date', panel_cols=['country'], strict_validation=False).fit(panel)

    def test_missing_panel_column_without_validation_warns(self, column_panel):
        """Without validation, an absent ``panel_cols`` column is reported, not raised."""
        with pytest.warns(UserWarning, match=r"Missing panel columns: \{'sector'\}"):
            ConcreteTransformer(time_col='date', panel_cols=['sector'], validate_input=False).fit(column_panel)

    def test_panel_cols_moved_into_the_index_are_not_missing(self, column_panel):
        """Panel columns moved into the index by ``convert_cols_to_index`` are found there."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country'], convert_cols_to_index=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            transformer.fit(column_panel)

        assert _panel_issue_messages(caught) == []


@pytest.mark.internal
class TestCheckPanelConsistencyDirectly:
    """Branches of ``_check_panel_consistency`` that ``fit`` never reaches.

    ``fit`` only calls the check after detecting the structure itself, on the data it
    has just validated; a direct call may come before ``fit`` or with other data.
    """

    def test_unfitted_call_detects_the_structure(self, column_panel):
        """Before ``fit``, the structure is detected from the configuration."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country'])
        assert transformer._check_panel_consistency(column_panel) == (True, [])

    def test_flat_index_after_a_multiindex_fit(self, index_panel, series_frame):
        """Fitted on a ``MultiIndex``, a flat index is reported as not a panel."""
        transformer = ConcreteTransformer().fit(index_panel)
        assert transformer._check_panel_consistency(series_frame) == (False, ['Expected MultiIndex for panel data'])

    def test_single_level_multiindex_after_a_multiindex_fit(self, index_panel, series_frame):
        """Fitted on a ``MultiIndex``, a one-level ``MultiIndex`` lacks the entity level."""
        transformer = ConcreteTransformer().fit(index_panel)
        frame = series_frame.set_axis(pd.MultiIndex.from_arrays([series_frame.index]))

        assert transformer._check_panel_consistency(frame) == (False, ['MultiIndex must have at least 2 levels'])


# =============================================================================
# Entités manquantes, déséquilibrées ou inconnues
# =============================================================================
class TestEntities:
    """The base class keeps no per-entity state: any set of entities is transformed."""

    def test_unbalanced_panel(self, index_panel):
        """Entities covering different dates are fitted and transformed together."""
        # Construction : DE n'a que ses deux premières dates
        panel = index_panel.drop(index=index_panel.index[-2:])

        output = CenteredScaler().fit_transform(panel)

        # Valeur d'or : moyenne de a = (1+2+3+4+5+6) / 6 = 3.5, puis (a - 3.5) * 2
        assert output['a'].tolist() == [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]

    def test_entity_absent_at_transform(self, index_panel):
        """A panel missing an entity seen at ``fit`` is transformed with the fitted means."""
        transformer = CenteredScaler().fit(index_panel)

        output = transformer.transform(drop_entity(index_panel, 'DE'))

        assert output['a'].tolist() == [-7.0, -5.0, -3.0, -1.0]

    def test_entity_unseen_at_fit(self, index_panel):
        """An entity unseen at ``fit`` is transformed with the fitted means."""
        transformer = CenteredScaler().fit(index_panel)
        index = pd.MultiIndex.from_product([['IT'], month_ends(2)], names=['country', 'date'])
        new = pd.DataFrame({'a': [4.5, 5.5], 'b': [45.0, 55.0]}, index=index)

        output = transformer.transform(new)

        # Valeur d'or : (4.5 - 4.5) * 2 = 0, (5.5 - 4.5) * 2 = 2
        assert output['a'].tolist() == [0.0, 2.0]

    def test_heterogeneous_coverage_panel(self, heterogeneous_coverage_panel):
        """On a realistic panel, the index and the missing values are left in place."""
        output = CenteredScaler().fit_transform(heterogeneous_coverage_panel)

        pd.testing.assert_index_equal(output.index, heterogeneous_coverage_panel.index)
        pd.testing.assert_frame_equal(output.isna(), heterogeneous_coverage_panel.isna())

    def test_irregular_index_timeseries(self, irregular_index_timeseries):
        """On a realistic irregular series, the index and the missing values are left in place."""
        output = CenteredScaler().fit_transform(irregular_index_timeseries)

        pd.testing.assert_index_equal(output.index, irregular_index_timeseries.index)
        pd.testing.assert_frame_equal(output.isna(), irregular_index_timeseries.isna())
