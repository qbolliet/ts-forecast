"""Tests of the error handling of ``PanelwiseTransformer``.

Covers ``error_handling`` (``'raise'``, ``'warn'``, ``'ignore'``) for an entity unseen
at ``fit`` (``transform`` and ``inverse_transform``), for an entity transformer that
fails at ``fit``, ``transform`` or ``inverse_transform`` (``FailingTransformer``
targeted at one entity), the content of ``failed_entities_`` and of the warnings,
the refit that forgets the former entities, invalid ``error_handling`` values,
missing ``panel_cols`` and empty data.
"""
import warnings

import pandas as pd
import pytest
from sklearn.decomposition import PCA

from tsforecast.panel.transformers import PanelwiseTransformer
from tests.support.perturbations import drop_entity, empty_like

from .conftest import FailingTransformer, centring, month_ends


def _failing_on_de(method: str, error_handling: str, error: type = RuntimeError) -> PanelwiseTransformer:
    """Transformer whose ``DE`` entity fails at ``method``, the other entities working."""
    return PanelwiseTransformer(
        FailingTransformer(error=error), time_col=None, error_handling=error_handling,
        entity_kwargs={('DE',): {'fail_on': method}},
    )


def _with_italy(index_panel: pd.DataFrame) -> pd.DataFrame:
    """The index panel plus an entity ``IT`` (``a`` = 7, 8, 9; ``b`` = 0)."""
    index = pd.MultiIndex.from_product([['IT'], month_ends()], names=['country', 'date'])
    italy = pd.DataFrame({'a': [7.0, 8.0, 9.0], 'b': [0.0, 0.0, 0.0]}, index=index)
    return pd.concat([index_panel, italy])


# =============================================================================
# Entité inconnue au transform
# =============================================================================
class TestUnknownEntity:
    """An entity unseen at ``fit`` is handled according to ``error_handling``."""

    @pytest.fixture
    def fitted_on_fr(self, index_panel):
        """Transformer fitted on ``FR`` only."""
        return lambda error_handling: PanelwiseTransformer(
            centring(), time_col=None, error_handling=error_handling
        ).fit(drop_entity(index_panel, 'DE'))

    def test_raise(self, fitted_on_fr, index_panel):
        """With ``'raise'``, the unseen entity is an invalid input: a ``ValueError`` names it."""
        with pytest.raises(ValueError, match=r"Entity \('DE',\) was not seen during fit"):
            fitted_on_fr('raise').transform(index_panel)

    def test_warn_names_the_entity(self, fitted_on_fr, index_panel):
        """With ``'warn'``, a warning names the unseen entity."""
        with pytest.warns(UserWarning, match=r"Entity \('DE',\) was not seen during fit. Using untransformed data"):
            fitted_on_fr('warn').transform(index_panel)

    @pytest.mark.parametrize('error_handling', ['warn', 'ignore'])
    def test_unseen_rows_are_left_untransformed(self, fitted_on_fr, index_panel, error_handling):
        """The unseen entity keeps its values, the known one is centred."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            output = fitted_on_fr(error_handling).transform(index_panel)

        # Valeur d'or : FR centré (a - 3), DE inchangé
        assert output['a'].tolist() == [-2.0, -1.0, 3.0, 10.0, 20.0, 60.0]

    def test_ignore_is_silent(self, fitted_on_fr, index_panel):
        """With ``'ignore'``, nothing is reported."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fitted_on_fr('ignore').transform(index_panel)

    def test_inverse_transform_raises_on_an_unseen_entity(self, fitted_on_fr, index_panel):
        """``inverse_transform`` applies the same rule as ``transform``."""
        with pytest.raises(ValueError, match='was not seen during fit'):
            fitted_on_fr('raise').inverse_transform(index_panel)

    def test_only_the_unseen_entity_is_reported(self, index_panel):
        """Fitted on ``FR`` and ``DE``, a panel with ``IT`` warns for ``IT`` alone."""
        transformer = PanelwiseTransformer(centring(), time_col=None, error_handling='warn').fit(index_panel)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            transformer.transform(_with_italy(index_panel))

        assert [str(w.message) for w in caught] == ["Entity ('IT',) was not seen during fit. Using untransformed data."]

    def test_fit_transform_fits_every_entity_of_its_data(self, index_panel):
        """``fit_transform`` fits the entities it transforms: none of them is unseen."""
        transformer = PanelwiseTransformer(centring(), time_col=None, error_handling='raise')
        transformer.fit_transform(_with_italy(index_panel))
        assert list(transformer.transformers_) == [('FR',), ('DE',), ('IT',)]

    def test_refit_forgets_the_former_entities(self, index_panel):
        """A new ``fit`` replaces the fitted entities instead of adding to them."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(drop_entity(index_panel, 'DE'))
        transformer.fit(drop_entity(index_panel, 'FR'))
        assert list(transformer.transformers_) == [('DE',)]


# =============================================================================
# Transformateur d'entité en échec
# =============================================================================
class TestFailingEntityTransformer:
    """An entity transformer that raises is handled according to ``error_handling``."""

    @pytest.mark.parametrize('method', ['fit', 'transform'])
    def test_raise_propagates_the_error(self, index_panel, method):
        """With ``'raise'``, the error of the entity transformer propagates unchanged."""
        with pytest.raises(RuntimeError, match='deliberate failure'):
            _failing_on_de(method, 'raise').fit_transform(index_panel)

    def test_failed_fit_warns_and_skips_the_entity(self, index_panel):
        """With ``'warn'``, a failed fit is reported and the entity has no transformer."""
        with pytest.warns(
            UserWarning, match=r"Error fitting entity \('DE',\): RuntimeError: deliberate failure. Skipping"
        ):
            transformer = _failing_on_de('fit', 'warn').fit(index_panel)

        assert list(transformer.transformers_) == [('FR',)]
        assert transformer.failed_entities_ == [('DE',)]

    def test_failed_fit_is_silent_with_ignore(self, index_panel):
        """With ``'ignore'``, a failed fit is only recorded in ``failed_entities_``."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            transformer = _failing_on_de('fit', 'ignore').fit(index_panel)
        assert transformer.failed_entities_ == [('DE',)]

    def test_entity_failed_at_fit_is_unseen_at_transform(self, index_panel):
        """The entity whose fit failed is left untransformed, the others are centred."""
        transformer = _failing_on_de('fit', 'ignore').fit(index_panel)

        output = transformer.transform(index_panel)

        # Valeur d'or : FR centré (a - 3), DE inchangé
        assert output['a'].tolist() == [-2.0, -1.0, 3.0, 10.0, 20.0, 60.0]

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_failed_transform_keeps_the_entity_rows(self, index_panel, method):
        """With ``'warn'``, an entity failing at ``method`` keeps its rows unchanged."""
        transformer = _failing_on_de(method, 'warn').fit(index_panel)

        with pytest.warns(UserWarning, match=r"entity \('DE',\): RuntimeError: deliberate failure"):
            output = getattr(transformer, method)(index_panel)

        # Valeur d'or : FR décalé de sa moyenne (-3 ou +3), DE inchangé
        assert output['a'].tolist()[3:] == [10.0, 20.0, 60.0]

    def test_no_failure_leaves_failed_entities_empty(self, index_panel):
        """Without any failure, ``failed_entities_`` is empty after ``fit``."""
        assert PanelwiseTransformer(centring(), time_col=None).fit(index_panel).failed_entities_ == []

    def test_warning_names_the_type_of_the_error(self, index_panel):
        """The warning names the type of the error: a ``KeyError`` does not read as a bare key."""
        with pytest.warns(UserWarning, match=r"Error fitting entity \('DE',\): KeyError: 'deliberate failure'"):
            _failing_on_de('fit', 'warn', error=KeyError).fit(index_panel)

    def test_transform_does_not_change_the_fitted_state(self, index_panel):
        """``failed_entities_`` describes the fit: transforming does not change it."""
        transformer = _failing_on_de('transform', 'ignore').fit(index_panel)

        transformer.transform(index_panel)
        transformer.transform(index_panel)

        assert transformer.failed_entities_ == []


# =============================================================================
# Paramètres invalides et données limites
# =============================================================================
class TestInvalidSettings:
    """Settings that cannot work are refused with an explicit message."""

    def test_unknown_error_handling_is_refused_at_fit(self, index_panel):
        """``error_handling`` accepts ``'raise'``, ``'warn'`` and ``'ignore'`` only."""
        transformer = PanelwiseTransformer(centring(), time_col=None, error_handling='oops')
        with pytest.raises(ValueError, match=r"error_handling must be one of \['raise', 'warn', 'ignore'\], got 'oops'"):
            transformer.fit(index_panel)

    def test_unknown_error_handling_set_after_fit_is_refused(self, index_panel):
        """A strategy changed by ``set_params`` after ``fit`` is checked at ``transform``."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)
        transformer.set_params(error_handling='warning')
        with pytest.raises(ValueError, match="got 'warning'"):
            transformer.transform(index_panel)

    def test_unvalidated_column_panel_without_panel_cols_is_refused(self, column_panel):
        """Entities held in columns cannot be found without ``panel_cols``."""
        transformer = PanelwiseTransformer(centring(), panel_cols=None, validate_input=False)
        with pytest.raises(ValueError, match='panel_cols must name the entity columns when X has a simple index'):
            transformer.fit(column_panel)

    def test_validated_column_panel_without_panel_cols_is_a_series_with_repeated_dates(self, column_panel):
        """Validated without ``panel_cols``, the panel is read as one series whose dates repeat."""
        transformer = PanelwiseTransformer(centring(), panel_cols=None)
        with pytest.raises(ValueError, match=r"Duplicate rows found for combination of columns: \['date'\]"):
            transformer.fit(column_panel)


class TestEmptyData:
    """Empty data holds no entity: the output is an empty copy of the input."""

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_empty_index_panel(self, index_panel, method):
        """Fitted on a non-empty panel, empty data gives its empty copy."""
        empty = empty_like(index_panel)
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)

        pd.testing.assert_frame_equal(getattr(transformer, method)(empty), empty)

    def test_fit_transform_of_an_empty_panel(self, index_panel):
        """An empty panel fits zero entities and comes out empty, with its columns and index levels."""
        empty = empty_like(index_panel)
        transformer = PanelwiseTransformer(centring(), time_col=None)

        output = transformer.fit_transform(empty)

        assert transformer.entities_ == 0
        pd.testing.assert_frame_equal(output, empty)

    def test_empty_column_panel(self, column_panel):
        """An empty column panel comes out empty, its identifier columns kept."""
        empty = column_panel.iloc[:0]
        output = PanelwiseTransformer(centring(), panel_cols=['country']).fit_transform(empty)
        assert list(output.columns) == ['country', 'date', 'a', 'b'] and len(output) == 0

    def test_columns_of_a_column_changing_transformer_are_the_input_ones(self, index_panel):
        """Without rows, the entity transformer cannot be called: the input columns are kept."""
        transformer = PanelwiseTransformer(PCA(n_components=1), time_col=None).fit(index_panel)
        assert list(transformer.transform(empty_like(index_panel)).columns) == ['a', 'b']
