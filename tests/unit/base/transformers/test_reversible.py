"""Tests of ``ReversibleTransformerMixin`` and ``InverseTransformer``.

Covers the round trip ``inverse_transform(transform(X)) == X`` on every
perturbation of ``tests/support/perturbations.py`` (series and panel), the
metadata stored by ``_store_transformation_info``, the structure restoration of
``_restore_structure_if_converted``, ``inverse_transform`` before ``fit``, the
abstract contract of the mixin, and ``InverseTransformer``: swap of ``transform``
and ``inverse_transform``, wrapped estimator not refitted, sklearn parameters,
``clone`` and export from ``tsforecast.base``.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

import tsforecast.base
from tsforecast.base.transformers import (
    InverseTransformer,
    PanelTimeSeriesTransformer,
    ReversibleTransformerMixin,
)
from tests.support.perturbations import (
    drop_entity,
    empty_like,
    reverse_entities,
    shuffle_rows,
    single_observation,
    to_period_end,
    to_period_index,
    to_period_start,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)

from .conftest import CenteredScaler, build_index_panel, build_series_frame


# =============================================================================
# Aller-retour transform → inverse_transform sur chaque perturbation
# =============================================================================
# Perturbations applicables à une série comme à un panel
_COMMON_PERTURBATIONS = [
    pytest.param(lambda df: df, id='identity'),
    pytest.param(lambda df: shuffle_rows(df, seed=0), id='shuffle_rows'),
    pytest.param(lambda df: with_special_column_names(df)[0], id='special_column_names'),
    pytest.param(to_period_start, id='to_period_start'),
    pytest.param(to_period_end, id='to_period_end'),
    pytest.param(single_observation, id='single_observation'),
    pytest.param(empty_like, id='empty_like'),
    pytest.param(to_period_index, id='to_period_index'),
]

# Perturbations propres aux panels
_PANEL_PERTURBATIONS = [
    pytest.param(reverse_entities, id='reverse_entities'),
    pytest.param(lambda df: with_index_names(df, ['pays', 'date de publication']), id='index_names'),
    pytest.param(to_three_level_index, id='to_three_level_index'),
    pytest.param(lambda df: drop_entity(df, 'DE'), id='drop_entity'),
]


def _assert_round_trip(data: pd.DataFrame) -> None:
    """Check ``inverse_transform(fit_transform(data))`` against ``data`` sorted by index.

    ``auto_sort=True`` lets the unsorted perturbations through; the rows come back
    sorted, each value on its own label.
    """
    transformer = CenteredScaler(auto_sort=True)
    restored = transformer.inverse_transform(transformer.fit_transform(data))
    pd.testing.assert_frame_equal(
        restored, data.sort_index(), check_exact=False, rtol=1e-12, check_freq=False
    )


class TestRoundTripOnPerturbations:
    """``inverse_transform`` undoes ``transform`` up to floating-point tolerance."""

    @pytest.mark.parametrize('perturb', _COMMON_PERTURBATIONS)
    def test_series(self, perturb):
        """Round trip on a perturbed monthly series."""
        _assert_round_trip(perturb(build_series_frame()))

    @pytest.mark.parametrize('perturb', _COMMON_PERTURBATIONS + _PANEL_PERTURBATIONS)
    def test_panel(self, perturb):
        """Round trip on a perturbed two-entity panel."""
        _assert_round_trip(perturb(build_index_panel()))

    def test_duplicated_rows_are_refused_in_strict_mode(self, index_panel):
        """Duplicated rows have no round trip: strict validation refuses them."""
        with pytest.raises(ValueError, match='MultiIndex contains duplicate combinations'):
            CenteredScaler().fit(with_duplicated_rows(index_panel, n=2))

    def test_duplicated_rows_round_trip_to_the_deduplicated_panel(self, index_panel):
        """In permissive mode the round trip gives the panel without its duplicates."""
        transformer = CenteredScaler(strict_validation=False)
        data = with_duplicated_rows(index_panel, n=2)

        with pytest.warns(UserWarning, match='Keeping first occurrence'):
            restored = transformer.inverse_transform(transformer.fit_transform(data))

        pd.testing.assert_frame_equal(restored, index_panel, check_exact=False, rtol=1e-12)

    def test_heterogeneous_coverage_panel(self, heterogeneous_coverage_panel):
        """Round trip on a realistic panel with missing values."""
        _assert_round_trip(heterogeneous_coverage_panel)


# =============================================================================
# Métadonnées de transformation et restitution de la structure
# =============================================================================
class _DropFirstRow(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
    """Reversible double whose ``_transform`` drops the first row (changes the index)."""

    def _fit(self, X, y=None):
        self.first_row_ = X.iloc[[0]]

    def _transform(self, X):
        out = X.iloc[1:]
        self._store_transformation_info(X, out)
        return out

    def inverse_transform(self, X):
        """Put the first row seen at ``fit`` back in front."""
        return self._restore_structure_if_converted(pd.concat([self.first_row_, X]))


class TestStoredTransformationInfo:
    """``_store_transformation_info`` records shapes, columns and, if changed, the index."""

    def test_shapes_and_columns_are_recorded(self, index_panel):
        """Shapes and column lists of both sides are stored after ``transform``."""
        transformer = CenteredScaler().fit(index_panel)
        transformer.transform(index_panel)

        assert (transformer.original_shape_, transformer.transformed_shape_) == ((8, 2), (8, 2))
        assert transformer.original_columns_ == transformer.transformed_columns_ == ['a', 'b']

    def test_unchanged_index_is_not_recorded(self, index_panel):
        """The index is stored only when the transformation changes it."""
        transformer = CenteredScaler().fit(index_panel)
        transformer.transform(index_panel)
        assert not hasattr(transformer, 'original_index_')

    def test_changed_index_is_recorded(self, series_frame):
        """A transformation dropping a row stores both indexes."""
        transformer = _DropFirstRow().fit(series_frame)

        output = transformer.transform(series_frame)

        pd.testing.assert_index_equal(transformer.original_index_, series_frame.index)
        pd.testing.assert_index_equal(transformer.transformed_index_, output.index)

    def test_changed_index_round_trip(self, series_frame):
        """The double's inverse puts the dropped row back."""
        transformer = _DropFirstRow().fit(series_frame)
        restored = transformer.inverse_transform(transformer.transform(series_frame))
        pd.testing.assert_frame_equal(restored, series_frame, check_freq=False)


class TestRestoreStructure:
    """``_restore_structure_if_converted`` only acts on data converted from columns."""

    @pytest.mark.internal
    def test_unfitted_transformer_returns_the_data_as_is(self, index_panel):
        """Before ``fit`` nothing is known about a conversion: the data is returned untouched."""
        assert CenteredScaler()._restore_structure_if_converted(index_panel) is index_panel

    @pytest.mark.parametrize(
        'kwargs', [{}, {'convert_cols_to_index': True}], ids=['plain', 'convert-flag']
    )
    def test_index_panel_is_returned_as_is(self, index_panel, kwargs):
        """A ``MultiIndex`` panel had no columns to restore: the inverse keeps its index."""
        transformer = CenteredScaler(**kwargs).fit(index_panel)
        restored = transformer.inverse_transform(transformer.transform(index_panel))
        pd.testing.assert_index_equal(restored.index, index_panel.index)

    @pytest.mark.parametrize('build', [build_series_frame, build_index_panel], ids=['series', 'panel'])
    def test_periods_left_unconverted_are_kept(self, build):
        """Without validation the periods are never converted, and the inverse keeps them."""
        data = to_period_index(build())
        transformer = CenteredScaler(validate_input=False).fit(data)

        restored = transformer.inverse_transform(transformer.transform(data))

        pd.testing.assert_frame_equal(restored, data, check_exact=False, rtol=1e-12)

    def test_series_without_conversion_is_returned_as_is(self, series_frame):
        """A ``DatetimeIndex`` series keeps its index through the inverse."""
        transformer = CenteredScaler().fit(series_frame)
        restored = transformer.inverse_transform(transformer.transform(series_frame))
        pd.testing.assert_index_equal(restored.index, series_frame.index)

    def test_inverse_does_not_depend_on_later_transforms(self):
        """Transforming other data in between does not change the inverse of an output."""
        first = build_index_panel().reset_index()
        second = first.set_axis(range(10, 18))
        transformer = CenteredScaler(time_col='date', panel_cols=['country'], convert_cols_to_index=True)
        output = transformer.fit(first).transform(first)

        transformer.transform(second)

        pd.testing.assert_frame_equal(transformer.inverse_transform(output), first)

    def test_transform_leaves_the_fitted_structure_untouched(self, column_panel):
        """``conversion_metadata_`` is a fitted attribute: ``transform`` does not replace it."""
        transformer = CenteredScaler(time_col='date', panel_cols=['country'], convert_cols_to_index=True)
        metadata = transformer.fit(column_panel).conversion_metadata_

        transformer.transform(column_panel.iloc[:4])

        assert transformer.conversion_metadata_ is metadata

    def test_restored_columns_get_a_fresh_range_index(self, column_panel):
        """Row labels of the input are not kept: the restored rows are numbered from 0."""
        data = column_panel.set_axis(range(100, 108))
        transformer = CenteredScaler(time_col='date', panel_cols=['country'], convert_cols_to_index=True)

        restored = transformer.inverse_transform(transformer.fit_transform(data))

        pd.testing.assert_frame_equal(restored, column_panel, check_exact=False, rtol=1e-12)

    def test_refit_without_conversion_drops_the_structure(self, column_panel, index_panel):
        """A new ``fit`` on data with nothing to convert forgets the previous structure."""
        transformer = CenteredScaler(time_col='date', panel_cols=['country'], convert_cols_to_index=True)
        transformer.fit(column_panel)

        transformer.set_params(time_col=None, panel_cols=None).fit(index_panel)

        assert not hasattr(transformer, 'conversion_metadata_')


# =============================================================================
# Contrat du mixin et inversion avant fit
# =============================================================================
class TestReversibleContract:
    """``inverse_transform`` is the contract of the mixin."""

    def test_inverse_before_fit_raises(self, series_frame):
        """An inverse without learned parameters refuses to run."""
        with pytest.raises(NotFittedError, match='is not fitted yet'):
            CenteredScaler().inverse_transform(series_frame)

    def test_missing_inverse_transform_is_refused(self):
        """A reversible subclass must implement ``inverse_transform``."""
        class NoInverse(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
            def _fit(self, X, y=None):
                pass

            def _transform(self, X):
                return X

        with pytest.raises(TypeError, match='inverse_transform'):
            NoInverse()


# =============================================================================
# InverseTransformer
# =============================================================================
class TestInverseTransformer:
    """``InverseTransformer`` swaps ``transform`` and ``inverse_transform`` of a fitted transformer."""

    @pytest.fixture
    def fitted(self, series_frame) -> CenteredScaler:
        """``CenteredScaler`` fitted on the monthly series (mean 2.5, factor 2)."""
        return CenteredScaler().fit(series_frame)

    def test_transform_is_the_wrapped_inverse(self, fitted, series_frame):
        """``transform`` applies the wrapped ``inverse_transform``."""
        # Valeur d'or : v / 2 + 2.5
        output = InverseTransformer(fitted).transform(series_frame)
        assert output['value'].tolist() == [3.0, 3.5, 4.0, 4.5]

    def test_inverse_transform_is_the_wrapped_transform(self, fitted, series_frame):
        """``inverse_transform`` applies the wrapped ``transform``."""
        output = InverseTransformer(fitted).inverse_transform(series_frame)
        assert output['value'].tolist() == [-3.0, -1.0, 1.0, 3.0]

    def test_round_trip(self, fitted, series_frame):
        """``inverse_transform(transform(X))`` is the identity."""
        inverse = InverseTransformer(fitted)
        restored = inverse.inverse_transform(inverse.transform(series_frame))
        pd.testing.assert_frame_equal(restored, series_frame, check_exact=False, rtol=1e-12)

    def test_fit_returns_self_without_refitting(self, fitted, series_frame):
        """``fit`` returns the wrapper and leaves the wrapped state untouched."""
        center = fitted.center_.copy()
        inverse = InverseTransformer(fitted)

        assert inverse.fit(series_frame * 100) is inverse
        pd.testing.assert_series_equal(fitted.center_, center)

    def test_sklearn_transformer_on_arrays(self):
        """With ``StandardScaler`` on an array, ``transform`` is the scaler's inverse."""
        X = np.array([[1.0], [3.0], [5.0]])
        scaler = StandardScaler().fit(X)

        output = InverseTransformer(scaler).fit_transform(scaler.transform(X))

        np.testing.assert_allclose(output, X)

    def test_unfitted_wrapped_transformer_raises(self, series_frame):
        """Wrapping an unfitted transformer fails at the first ``transform``."""
        with pytest.raises(NotFittedError, match='is not fitted yet'):
            InverseTransformer(CenteredScaler()).fit(series_frame).transform(series_frame)

    def test_parameters(self, fitted):
        """The wrapped transformer is the only parameter, stored as given."""
        inverse = InverseTransformer(fitted)

        assert inverse.get_params(deep=False) == {'transformer': fitted}
        assert inverse.get_params(deep=True)['transformer__factor'] == 2.0

    def test_clone_keeps_a_usable_transformer(self, fitted, series_frame):
        """A clone fitted like any sklearn step transforms like the original."""
        expected = InverseTransformer(fitted).transform(series_frame)

        cloned = clone(InverseTransformer(fitted)).fit(series_frame)

        pd.testing.assert_frame_equal(cloned.transform(series_frame), expected)

    def test_clone_is_the_wrapper_itself(self, fitted):
        """Like a frozen estimator, the pre-fitted wrapper is its own clone."""
        inverse = InverseTransformer(fitted)
        assert clone(inverse) is inverse

    def test_fitted_state_follows_the_wrapped_transformer(self, fitted):
        """The wrapper is fitted exactly when the wrapped transformer is."""
        check_is_fitted(InverseTransformer(fitted))
        with pytest.raises(NotFittedError):
            check_is_fitted(InverseTransformer(CenteredScaler()))

    def test_cross_validate_with_a_pipeline(self):
        """Cloned by ``cross_validate``, the step still undoes the fitted scaling."""
        # Construction : la cible est une fonction linéaire des données d'origine ;
        # le pipeline reçoit les données mises à l'échelle et les ramène à l'origine
        X = np.arange(1.0, 21.0).reshape(-1, 1)
        y = 3.0 * X.ravel()
        scaler = StandardScaler().fit(X)
        pipeline = Pipeline([('undo', InverseTransformer(scaler)), ('model', LinearRegression())])

        scores = cross_validate(pipeline, scaler.transform(X), y, cv=KFold(3))['test_score']

        np.testing.assert_allclose(scores, 1.0)

    def test_exported_by_the_package(self):
        """The public class is listed in ``tsforecast.base.__all__``."""
        assert 'InverseTransformer' in tsforecast.base.__all__
