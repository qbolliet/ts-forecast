"""Tests of ``PanelwiseTransformer`` fit, transform and inverse on each data layout.

Covers ``fit``, ``transform``, ``fit_transform`` and ``inverse_transform`` with the
entities in a ``MultiIndex`` (two and three levels, one entity, an entity absent at
``transform``, other rows than those of the fit) and in columns (validated or not,
converted to the index or not, one or two panel columns, dates given as strings),
the target cut entity by entity, the dates of an array output, and the realistic
``heterogeneous_coverage_panel`` (index kept, missing values in place, no leak
between entities).
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from tsforecast.panel.transformers import PanelwiseTransformer
from tests.support.perturbations import drop_entity, to_three_level_index

from .conftest import RecordingScaler, centred_values, centring, month_ends


# =============================================================================
# Entités portées par l'index
# =============================================================================
class TestIndexedPanel:
    """Entities in the ``MultiIndex``: each entity is centred on its own means."""

    def test_fit_transform_values(self, index_panel, centred_index_panel):
        """Each entity is centred on its own means, the index being kept."""
        output = PanelwiseTransformer(centring(), time_col=None).fit_transform(index_panel)
        pd.testing.assert_frame_equal(output, centred_index_panel)

    def test_fit_then_transform_equals_fit_transform(self, index_panel, centred_index_panel):
        """``fit`` then ``transform`` on the same data gives the ``fit_transform`` output."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)
        pd.testing.assert_frame_equal(transformer.transform(index_panel), centred_index_panel)

    def test_three_level_panel_values(self, index_panel, centred_index_panel):
        """With two entity levels ``(region, country)``, entities and level names are kept."""
        panel = to_three_level_index(index_panel)

        output = PanelwiseTransformer(centring(), time_col=None).fit_transform(panel)

        pd.testing.assert_frame_equal(output, centred_index_panel.set_axis(panel.index))

    def test_three_level_entities_are_pairs_of_labels(self, index_panel):
        """Each entity of a three-level panel is keyed by its two labels."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(to_three_level_index(index_panel))
        assert list(transformer.transformers_) == [('Zone euro', 'FR'), ('Zone euro', 'DE')]

    def test_single_entity(self, index_panel):
        """A one-entity panel is centred on its own means."""
        panel = drop_entity(index_panel, 'DE')

        output = PanelwiseTransformer(centring(), time_col=None).fit_transform(panel)

        # Valeur d'or : a - 3, b - 4
        assert output['a'].tolist() == [-2.0, -1.0, 3.0]
        assert output['b'].tolist() == [-4.0, 0.0, 4.0]

    def test_entity_absent_at_transform(self, index_panel):
        """Transforming one entity seen at ``fit`` uses that entity's means only."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)

        output = transformer.transform(drop_entity(index_panel, 'FR'))

        # Valeur d'or : moyennes de DE, a - 30
        assert output['a'].tolist() == [-20.0, -10.0, 30.0]

    def test_other_rows_than_the_fit(self, index_panel):
        """New rows of a known entity are centred on the means learnt at ``fit``."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)
        index = pd.MultiIndex.from_product([['DE'], month_ends(5)[3:]], names=['country', 'date'])
        new = pd.DataFrame({'a': [30.0, 35.0], 'b': [3.0, 13.0]}, index=index)

        output = transformer.transform(new)

        # Valeur d'or : (30, 35) - 30 et (3, 13) - 3
        pd.testing.assert_frame_equal(output, pd.DataFrame({'a': [0.0, 5.0], 'b': [0.0, 10.0]}, index=index))

    @pytest.mark.parametrize('levels', [2, 3], ids=['two-levels', 'three-levels'])
    def test_inverse_transform_restores_the_input(self, index_panel, levels):
        """``inverse_transform(transform(X))`` gives the panel back, values and index."""
        panel = index_panel if levels == 2 else to_three_level_index(index_panel)
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(panel)

        restored = transformer.inverse_transform(transformer.transform(panel))

        pd.testing.assert_frame_equal(restored, panel)

    def test_each_entity_is_standardized(self, index_panel, scaled_rows):
        """``StandardScaler`` gives each entity a zero mean and a unit (population) deviation."""
        output = PanelwiseTransformer(StandardScaler(), time_col=None).fit_transform(index_panel)
        np.testing.assert_allclose(output['a'].to_numpy(), scaled_rows)

    def test_missing_entity_label_is_refused(self, index_panel):
        """Rows whose entity label is missing would be dropped silently: the fit refuses them."""
        index = index_panel.index.to_frame(index=False)
        index.loc[0, 'country'] = np.nan
        broken = index_panel.set_axis(pd.MultiIndex.from_frame(index))

        with pytest.raises(ValueError, match='must not be missing'):
            PanelwiseTransformer(centring(), time_col=None, validate_input=False).fit(broken)


# =============================================================================
# Entités portées par des colonnes
# =============================================================================
class TestColumnPanel:
    """Entities in ``panel_cols`` columns: centred per entity, the columns staying columns."""

    def test_fit_transform_with_validation(self, column_panel, centred_column_panel):
        """By default the panel is validated and comes out with its columns and row labels."""
        output = PanelwiseTransformer(centring(), panel_cols=['country']).fit_transform(column_panel)
        pd.testing.assert_frame_equal(output, centred_column_panel)

    def test_fit_transform_without_validation(self, column_panel, centred_column_panel):
        """Without validation the output is the same: time and panel columns are left untouched."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country'], validate_input=False)
        pd.testing.assert_frame_equal(transformer.fit_transform(column_panel), centred_column_panel)

    def test_fit_then_transform_equals_fit_transform(self, column_panel, centred_column_panel):
        """``fit`` learns one transformer per entity; ``transform`` then gives the centred panel."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country']).fit(column_panel)

        assert list(transformer.transformers_) == [('FR',), ('DE',)]
        pd.testing.assert_frame_equal(transformer.transform(column_panel), centred_column_panel)

    def test_transform_columns_exclude_time_and_panel_columns(self, column_panel):
        """Only the value columns are passed to the entity transformers."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country']).fit(column_panel)
        assert transformer.transform_columns_ == ['a', 'b']

    def test_each_entity_is_standardized(self, column_panel, scaled_rows):
        """``StandardScaler`` standardizes each country on its own rows."""
        output = PanelwiseTransformer(StandardScaler(), panel_cols=['country']).fit_transform(column_panel)
        np.testing.assert_allclose(output['a'].to_numpy(), scaled_rows)

    def test_inverse_transform_restores_the_input(self, column_panel):
        """``inverse_transform(transform(X))`` gives the column panel back."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country']).fit(column_panel)

        restored = transformer.inverse_transform(transformer.transform(column_panel))

        pd.testing.assert_frame_equal(restored, column_panel)

    def test_converted_panel_comes_out_indexed(self, column_panel, centred_index_panel):
        """Under ``convert_cols_to_index=True`` the output is indexed by ``(country, date)``."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country'], convert_cols_to_index=True)
        pd.testing.assert_frame_equal(transformer.fit_transform(column_panel), centred_index_panel)

    def test_converted_panel_inverse_gives_the_columns_back(self, column_panel):
        """The inverse of a converted panel restores its columns."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country'], convert_cols_to_index=True)

        restored = transformer.inverse_transform(transformer.fit_transform(column_panel))

        pd.testing.assert_frame_equal(restored, column_panel)

    def test_two_panel_columns(self):
        """Entities defined by two columns ``(country, sector)`` are centred separately."""
        dates = list(month_ends(2))
        panel = pd.DataFrame({
            'country': ['FR'] * 4 + ['DE'] * 4,
            'sector': ['tech', 'tech', 'finance', 'finance'] * 2,
            'date': dates * 4,
            'v': [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 5.0, 7.0],
        })
        transformer = PanelwiseTransformer(centring(), panel_cols=['country', 'sector'])

        output = transformer.fit_transform(panel)

        # Valeur d'or : chaque couple (pays, secteur) centré sur sa moyenne (2, 20, 200, 6)
        assert transformer.n_entities_ == 4
        assert output['v'].tolist() == [-1.0, 1.0, -10.0, 10.0, -100.0, 100.0, -1.0, 1.0]

    def test_date_strings_are_read_as_dates(self, column_panel, centred_column_panel):
        """Validated, dates given as strings come out as ``datetime64`` dates."""
        panel = column_panel.assign(date=column_panel['date'].dt.strftime('%Y-%m-%d'))

        output = PanelwiseTransformer(centring(), panel_cols=['country']).fit_transform(panel)

        pd.testing.assert_frame_equal(output, centred_column_panel)

    def test_panel_without_a_time_column(self, column_panel, centred_column_panel):
        """Unvalidated, a panel whose ``time_col`` is not a column is cut on its row labels."""
        panel = column_panel.drop(columns='date')
        transformer = PanelwiseTransformer(centring(), time_col='period', panel_cols=['country'], validate_input=False)

        output = transformer.fit_transform(panel)

        pd.testing.assert_frame_equal(output, centred_column_panel.drop(columns='date'))

    def test_target_is_matched_by_position(self, column_panel):
        """Each entity transformer receives the target of its own rows, indexed by date."""
        y = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        transformer = PanelwiseTransformer(RecordingScaler(), panel_cols=['country']).fit(column_panel, y)

        expected = pd.Series([3.0, 4.0, 5.0], index=month_ends())
        pd.testing.assert_series_equal(transformer.get_entity_transformer('DE').target_, expected, check_freq=False)

    def test_target_of_another_length_is_refused(self, column_panel):
        """A target that does not have one value per row cannot be cut by entity."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country'])
        with pytest.raises(ValueError, match='y has 5 rows, X has 6'):
            transformer.fit(column_panel, pd.Series(range(5)))

    def test_missing_entity_label_is_refused(self, column_panel):
        """Rows without entity label would be dropped silently: unvalidated, the fit refuses them."""
        panel = column_panel.copy()
        panel.loc[4, 'country'] = None

        transformer = PanelwiseTransformer(centring(), panel_cols=['country'], validate_input=False)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match="column 'country' holds missing values"):
                transformer.fit(panel)


class TestInverseSupport:
    """``inverse_transform`` requires entity transformers that can invert."""

    def test_base_transformer_without_inverse(self, index_panel):
        """A base transformer without ``inverse_transform`` is refused by name."""
        transformer = PanelwiseTransformer(PolynomialFeatures(), time_col=None).fit(index_panel)
        with pytest.raises(AttributeError, match='The base transformer PolynomialFeatures does not support inverse_transform'):
            transformer.inverse_transform(index_panel)

    def test_factory_transformers_without_inverse(self, index_panel):
        """In factory mode, the fitted entity transformers are checked."""
        transformer = PanelwiseTransformer(lambda entity_key: PolynomialFeatures(), time_col=None).fit(index_panel)
        with pytest.raises(AttributeError, match='The transformer PolynomialFeatures does not support inverse_transform'):
            transformer.inverse_transform(index_panel)

    def test_factory_transformers_with_inverse(self, index_panel):
        """In factory mode, invertible entity transformers give the input back."""
        transformer = PanelwiseTransformer(lambda entity_key: centring(), time_col=None).fit(index_panel)
        restored = transformer.inverse_transform(transformer.transform(index_panel))
        pd.testing.assert_frame_equal(restored, index_panel)


# =============================================================================
# Cible et sorties en tableau
# =============================================================================
class TestTargetOfIndexedPanel:
    """The target of an indexed panel is cut like the data."""

    def test_target_is_cut_like_the_data(self, index_panel):
        """Each entity transformer receives the target of its own rows, indexed by date."""
        y = pd.Series(np.arange(6.0), index=index_panel.index)

        transformer = PanelwiseTransformer(RecordingScaler(), time_col=None).fit(index_panel, y)

        pd.testing.assert_series_equal(transformer.get_entity_transformer('DE').target_, y.loc['DE'])

    def test_array_target_is_read_on_the_rows_of_x(self, index_panel):
        """A target given as an array is aligned on the rows of ``X``."""
        transformer = PanelwiseTransformer(RecordingScaler(), time_col=None).fit(index_panel, np.arange(6.0))
        assert transformer.get_entity_transformer('FR').target_.tolist() == [0.0, 1.0, 2.0]


class TestDatesOfAnArrayOutput:
    """A transformer returning an array keeps the dates of each entity in the output index."""

    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        """Two entities, A on three month ends and B on two, values chosen for exact scaling."""
        dates = month_ends()
        index = pd.MultiIndex.from_tuples(
            [('A', dates[0]), ('A', dates[1]), ('A', dates[2]), ('B', dates[1]), ('B', dates[2])],
            names=['entity', 'date'],
        )
        return pd.DataFrame({'v': [1.0, 2.0, 3.0, 10.0, 30.0]}, index=index)

    def test_transform_keeps_the_index(self, panel):
        """``StandardScaler`` returns an array: the output index is the input index."""
        out = PanelwiseTransformer(StandardScaler(), time_col=None).fit(panel).transform(panel)
        pd.testing.assert_index_equal(out.index, panel.index)

    def test_transform_values(self, panel):
        """Each entity is standardized with its own mean and deviation."""
        out = PanelwiseTransformer(StandardScaler(), time_col=None).fit(panel).transform(panel)
        # Valeur d'or : A moyenne 2, écart-type sqrt(2/3) ; B moyenne 20, écart-type 10
        a = 1.0 / np.sqrt(2.0 / 3.0)
        np.testing.assert_allclose(out['v'].to_numpy(), [-a, 0.0, a, -1.0, 1.0])

    def test_transform_of_other_rows_keeps_their_dates(self, panel):
        """Transforming a subset of rows gives back the dates of that subset."""
        transformer = PanelwiseTransformer(StandardScaler(), time_col=None).fit(panel)
        subset = panel.iloc[[2, 4]]
        pd.testing.assert_index_equal(transformer.transform(subset).index, subset.index)

    def test_inverse_transform_restores_the_input(self, panel):
        """``transform`` then ``inverse_transform`` gives back the values and the index."""
        transformer = PanelwiseTransformer(StandardScaler(), time_col=None).fit(panel)
        restored = transformer.inverse_transform(transformer.transform(panel))
        pd.testing.assert_frame_equal(restored, panel)


# =============================================================================
# Jeu réaliste
# =============================================================================
class TestHeterogeneousCoveragePanel:
    """On a realistic panel with per-entity coverage, entities stay independent."""

    @staticmethod
    def _standardize(panel: pd.DataFrame) -> pd.DataFrame:
        """Standardize each entity of ``panel``; empty columns are expected to warn."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return PanelwiseTransformer(StandardScaler(), time_col=None).fit_transform(panel)

    def test_index_and_missing_values_are_kept(self, heterogeneous_coverage_panel):
        """The output has the input index, and missing values exactly where they were."""
        output = self._standardize(heterogeneous_coverage_panel)

        pd.testing.assert_index_equal(output.index, heterogeneous_coverage_panel.index)
        pd.testing.assert_frame_equal(output.isna(), heterogeneous_coverage_panel.isna())

    def test_no_leak_between_entities(self, heterogeneous_coverage_panel):
        """Transforming France and Allemagne together equals transforming each one alone."""
        panel = drop_entity(heterogeneous_coverage_panel, 'Italie')

        together = self._standardize(panel)
        alone = pd.concat([self._standardize(drop_entity(panel, other)) for other in ['Allemagne', 'France']])

        pd.testing.assert_frame_equal(together, alone.loc[together.index])
