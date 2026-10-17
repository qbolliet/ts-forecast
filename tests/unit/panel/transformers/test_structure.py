"""Tests of the structure kept by ``PanelwiseTransformer`` from its input.

Covers the non-mutation of the input (``fit``, ``transform``, ``inverse_transform``),
the row labels, row order (shuffled, interleaved or unsorted entities), column order,
column and index names and dtypes of the output, the transformers that change the
columns (``get_feature_names_out`` names, generic ``feature_i`` names) or the rows
(subset of the dates, new dates), and the default ``time_col`` on a ``MultiIndex``.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from tsforecast.panel.transformers import PanelwiseTransformer
from tests.support.perturbations import (
    interleave_entities,
    shuffle_rows,
    to_three_level_index,
    unsort_within_entities,
    with_index_names,
    with_special_column_names,
)

from .conftest import centring, month_ends


def _column_transformer(**kwargs) -> PanelwiseTransformer:
    """Per-entity centring of a column panel identified by ``country``."""
    return PanelwiseTransformer(centring(), panel_cols=['country'], **kwargs)


def _index_transformer(**kwargs) -> PanelwiseTransformer:
    """Per-entity centring of a ``MultiIndex`` panel."""
    return PanelwiseTransformer(centring(), time_col=None, **kwargs)


# =============================================================================
# Non-mutation de l'entrée
# =============================================================================
class TestInputIsNotModified:
    """``fit``, ``transform`` and ``inverse_transform`` leave the caller's data untouched."""

    @pytest.mark.parametrize('method', ['fit', 'fit_transform', 'inverse_transform'])
    @pytest.mark.parametrize('layout', ['index', 'columns'])
    def test_input_is_unchanged(self, index_panel, column_panel, method, layout):
        """A deep copy taken before the call equals the data after it."""
        data, transformer = (
            (index_panel, _index_transformer()) if layout == 'index' else (column_panel, _column_transformer())
        )
        before = data.copy(deep=True)

        if method == 'inverse_transform':
            transformer.fit(data).inverse_transform(data)
        else:
            getattr(transformer, method)(data)

        pd.testing.assert_frame_equal(data, before)


# =============================================================================
# Étiquettes, ordre, noms et types
# =============================================================================
class TestLabelsAndOrder:
    """Row labels and row order of the input are those of the output."""

    def test_custom_row_labels_are_kept(self, column_panel):
        """Row labels other than ``0..n-1`` come out unchanged."""
        panel = column_panel.set_axis(range(100, 106))
        output = _column_transformer().fit_transform(panel)
        pd.testing.assert_index_equal(output.index, panel.index)

    def test_repeated_row_labels_keep_their_rows(self, column_panel, centred_column_panel):
        """Row labels repeated across entities do not mix the rows."""
        panel = column_panel.set_axis([0, 1, 2, 0, 1, 2])
        output = _column_transformer().fit_transform(panel)
        pd.testing.assert_frame_equal(output, centred_column_panel.set_axis(panel.index))

    def test_shuffled_rows_keep_their_order_without_validation(self, column_panel, centred_column_panel):
        """Unvalidated shuffled rows come out in the same shuffled order, values on their rows."""
        shuffled = shuffle_rows(column_panel, seed=42)
        output = _column_transformer(validate_input=False).fit_transform(shuffled)
        pd.testing.assert_frame_equal(output, centred_column_panel.loc[shuffled.index])

    def test_interleaved_index_panel_keeps_its_order(self, index_panel, centred_index_panel):
        """Entities alternating row by row (unvalidated) are not regrouped."""
        panel = interleave_entities(index_panel)
        output = _index_transformer(validate_input=False).fit_transform(panel)
        pd.testing.assert_frame_equal(output, centred_index_panel.loc[panel.index])

    def test_dates_unsorted_within_entities_keep_their_order(self, index_panel, centred_index_panel):
        """Dates permuted inside each entity (unvalidated) come out in the same order."""
        panel = unsort_within_entities(index_panel, seed=0)
        output = _index_transformer(validate_input=False).fit_transform(panel)
        pd.testing.assert_frame_equal(output, centred_index_panel.loc[panel.index])

    def test_interleaved_entities_are_refused_by_strict_validation(self, index_panel):
        """Validated strictly, entities must be contiguous."""
        with pytest.raises(ValueError, match='Panel entities are not contiguous in MultiIndex'):
            _index_transformer().fit(interleave_entities(index_panel))

    def test_column_order_is_kept(self, column_panel):
        """Identifier columns may come after the values: the input column order is kept."""
        panel = column_panel[['a', 'date', 'b', 'country']]
        output = _column_transformer().fit_transform(panel)
        assert list(output.columns) == ['a', 'date', 'b', 'country']

    def test_special_column_names_are_kept(self, index_panel):
        """Column names with spaces, accents and symbols come out verbatim."""
        panel, mapping = with_special_column_names(index_panel)
        output = _index_transformer().fit_transform(panel)
        assert list(output.columns) == [mapping['a'], mapping['b']]

    def test_index_names_are_kept(self, index_panel):
        """Non-standard level names of a three-level panel are kept."""
        panel = with_index_names(to_three_level_index(index_panel), ['zone géo', 'pays', 'période'])
        output = _index_transformer().fit_transform(panel)
        assert list(output.index.names) == ['zone géo', 'pays', 'période']

    def test_dtypes(self, column_panel):
        """Integer values come out as floats; entity and date columns keep their dtypes."""
        panel = column_panel.astype({'a': 'int64', 'b': 'int64'})
        output = _column_transformer().fit_transform(panel)
        assert output.dtypes.astype(str).tolist() == ['object', 'datetime64[ns]', 'float64', 'float64']


# =============================================================================
# Transformateurs qui changent les colonnes ou les lignes
# =============================================================================
class TestChangedColumns:
    """A transformer that changes the columns names them in the output."""

    def test_names_from_get_feature_names_out(self, index_panel):
        """A one-component PCA output is named ``pca0``, as sklearn names it."""
        output = PanelwiseTransformer(PCA(n_components=1), time_col=None).fit_transform(index_panel)
        assert list(output.columns) == ['pca0']

    def test_generic_names_without_feature_names(self, index_panel):
        """Without usable ``get_feature_names_out``, columns are named ``feature_i``."""
        doubling = FunctionTransformer(lambda X: np.c_[X, X])
        output = PanelwiseTransformer(doubling, time_col=None).fit_transform(index_panel)
        assert list(output.columns) == ['feature_0', 'feature_1', 'feature_2', 'feature_3']

    def test_generic_names_when_feature_names_fail(self, index_panel):
        """A ``get_feature_names_out`` that raises falls back on ``feature_i`` names."""
        # Pipeline expose get_feature_names_out, que sa FunctionTransformer sans noms fait échouer
        doubling = Pipeline([('double', FunctionTransformer(lambda X: np.c_[X, X]))])
        output = PanelwiseTransformer(doubling, time_col=None).fit_transform(index_panel)
        assert list(output.columns) == ['feature_0', 'feature_1', 'feature_2', 'feature_3']

    def test_column_panel_keeps_its_identifiers_first(self, column_panel):
        """On a column panel, identifier columns keep their place and new columns follow."""
        output = PanelwiseTransformer(PCA(n_components=1), panel_cols=['country']).fit_transform(column_panel)
        assert list(output.columns) == ['country', 'date', 'pca0']


class TestChangedRows:
    """A transformer that drops or moves rows keeps each output row on its entity."""

    @staticmethod
    def _drop_first_date(X: pd.DataFrame) -> pd.DataFrame:
        """Keep every row but the first."""
        return X.iloc[1:]

    @staticmethod
    def _next_month(X: pd.DataFrame) -> pd.DataFrame:
        """Move every row to the next month end."""
        return X.set_axis(X.index + pd.offsets.MonthEnd(1))

    def test_dropped_rows_on_an_index_panel(self, index_panel):
        """Rows dropped by the entity transformer are absent; the others keep their index."""
        transformer = PanelwiseTransformer(FunctionTransformer(self._drop_first_date), time_col=None)
        output = transformer.fit_transform(index_panel)
        pd.testing.assert_frame_equal(output, index_panel.iloc[[1, 2, 4, 5]])

    def test_dropped_rows_on_a_column_panel(self, column_panel):
        """On a column panel, the remaining rows keep their labels and identifiers."""
        transformer = PanelwiseTransformer(FunctionTransformer(self._drop_first_date), panel_cols=['country'])
        output = transformer.fit_transform(column_panel)
        pd.testing.assert_frame_equal(output, column_panel.iloc[[1, 2, 4, 5]])

    def test_dropped_rows_of_an_entity_with_repeated_dates(self, index_panel):
        """Unvalidated repeated dates cannot be matched: the rows still keep their entity."""
        # Construction : la première date de FR répétée, lignes non validées
        index = index_panel.index.to_frame(index=False)
        index.loc[1, 'date'] = index.loc[0, 'date']
        panel = index_panel.set_axis(pd.MultiIndex.from_frame(index))
        transformer = PanelwiseTransformer(
            FunctionTransformer(self._drop_first_date), time_col=None, validate_input=False
        )

        output = transformer.fit_transform(panel)

        pd.testing.assert_frame_equal(output, panel.iloc[[1, 2, 4, 5]])

    def test_shorter_array_on_a_column_panel_without_dates(self, column_panel):
        """A shorter array has positions, not rows: its rows get the entity and fresh labels."""
        panel = column_panel.drop(columns='date')
        transformer = PanelwiseTransformer(
            FunctionTransformer(lambda X: X.to_numpy()[1:]), time_col='period', panel_cols=['country'],
            validate_input=False,
        )

        output = transformer.fit_transform(panel)

        # Valeur d'or : chaque entité perd sa première ligne, étiquettes neuves 0..3
        expected = pd.DataFrame({'country': ['FR', 'FR', 'DE', 'DE'], 'a': [2.0, 6.0, 20.0, 60.0],
                                 'b': [4.0, 8.0, 3.0, 3.0]})
        pd.testing.assert_frame_equal(output, expected)

    def test_new_dates_on_an_index_panel(self, index_panel):
        """Moved rows are indexed by their entity and their new dates."""
        transformer = PanelwiseTransformer(FunctionTransformer(self._next_month), time_col=None)

        output = transformer.fit_transform(index_panel)

        expected_index = pd.MultiIndex.from_product([['FR', 'DE'], month_ends(4)[1:]], names=['country', 'date'])
        pd.testing.assert_index_equal(output.index, expected_index, exact=False)

    def test_new_dates_on_a_column_panel(self, column_panel):
        """Moved rows of a column panel carry their new dates and fresh labels ``0..n-1``."""
        transformer = PanelwiseTransformer(FunctionTransformer(self._next_month), panel_cols=['country'])

        output = transformer.fit_transform(column_panel)

        # Valeur d'or : mêmes valeurs, dates avancées d'un mois, étiquettes neuves
        expected = column_panel.assign(date=list(month_ends(4)[1:]) * 2)
        pd.testing.assert_frame_equal(output, expected)


# =============================================================================
# Colonne temporelle par défaut
# =============================================================================
class TestDefaultTimeColumn:
    """The default ``time_col='date'`` must not change how a ``MultiIndex`` panel is read."""

    def test_explicit_none_reads_the_panel_from_the_index(self, index_panel, centred_index_panel):
        """With ``time_col=None``, a panel whose entities are not alphabetically sorted is accepted."""
        pd.testing.assert_frame_equal(_index_transformer().fit_transform(index_panel), centred_index_panel)

    def test_default_time_col_accepts_a_multiindex_panel(self, index_panel, centred_index_panel):
        """``PanelwiseTransformer(transformer)`` transforms a ``(country, date)`` panel as is."""
        output = PanelwiseTransformer(centring()).fit_transform(index_panel)
        pd.testing.assert_frame_equal(output, centred_index_panel)

    def test_default_time_col_reads_a_panel(self, index_panel):
        """With the default ``time_col``, a ``MultiIndex`` panel is flagged as a panel."""
        assert PanelwiseTransformer(centring()).fit(index_panel).is_panel_ is True

    def test_default_time_col_checks_the_contiguity_of_entities(self, index_panel):
        """The panel checks apply: interleaved entities are refused as a panel, not as a series."""
        with pytest.raises(ValueError, match='Panel entities are not contiguous in MultiIndex'):
            PanelwiseTransformer(centring()).fit(interleave_entities(index_panel))
