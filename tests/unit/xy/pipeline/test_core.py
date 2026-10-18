"""Core behaviour of ``XYPipeline``.

Covers the pipeline as a drop-in ``sklearn.pipeline.Pipeline`` on golden
values: ``fit``, ``predict``, ``score``, ``transform`` and ``fit_transform``
(exactly linear target, scaled features); the passing of ``(X, y)`` through
the XY transformers ``LogTransformXY`` and ``ScaleXY``; the preservation of the
pandas structure (columns, index, ``Series`` target, ``preserve_dataframe``),
including the index of a ``transform`` on new data and the names of the new
features; ``'passthrough'`` /
``None`` steps, single-step pipelines, slicing and parameter access.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tsforecast.xy.pipeline import XYPipeline
from tests.support.estimators import LogTransformXY, ScaleXY

from .conftest import month_ends


def _scaled(X: pd.DataFrame) -> np.ndarray:
    """Independent reference: ``StandardScaler`` fitted and applied outside any pipeline."""
    return StandardScaler().fit_transform(X)


# ==============================================================================
# fit, predict, score, transform
# ==============================================================================


class TestXYPipelineBasic:
    """Without XY step, the pipeline fits, predicts, scores and transforms like sklearn."""

    @pytest.fixture
    def regression(self):
        """``StandardScaler -> LinearRegression``."""
        return XYPipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])

    def test_pipeline_keeps_its_steps(self, regression):
        """The steps are kept in order, under their names."""
        assert [name for name, _ in regression.steps] == ['scaler', 'reg']

    def test_fit_returns_self(self, regression, linear_xy):
        """``fit`` follows the sklearn convention."""
        assert regression.fit(*linear_xy) is regression

    def test_predict_recovers_an_exactly_linear_target(self, regression, linear_xy):
        """Valeur d'or : y = 2a + 3 est retrouvé exactement (à l'arrondi près)."""
        X, y = linear_xy

        np.testing.assert_allclose(regression.fit(X, y).predict(X), y.to_numpy())

    def test_score_of_an_exact_fit_is_one(self, regression, linear_xy):
        """Valeur d'or : R² = 1 sur une cible exactement linéaire."""
        X, y = linear_xy

        assert regression.fit(X, y).score(X, y) == pytest.approx(1.0)

    def test_fit_transform_without_xy_step_returns_x_alone(self, linear_xy):
        """No XY step: ``fit_transform`` returns ``X'``, not a pair."""
        result = XYPipeline([('scaler', StandardScaler())]).fit_transform(*linear_xy)

        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_without_xy_step_scales(self, linear_xy):
        """``X'`` is the output of ``StandardScaler``."""
        X, y = linear_xy

        result = XYPipeline([('scaler', StandardScaler())]).fit_transform(X, y)

        np.testing.assert_allclose(result.to_numpy(), _scaled(X))

    def test_transform_scales_with_the_fitted_parameters(self, linear_xy):
        """``transform`` after ``fit`` applies the parameters learnt on the training set."""
        X, y = linear_xy
        pipeline = XYPipeline([('scaler', StandardScaler())]).fit(X, y)

        np.testing.assert_allclose(pipeline.transform(X).to_numpy(), _scaled(X))


# ==============================================================================
# Transformateurs XY
# ==============================================================================


class TestXYTransformers:
    """XY steps transform ``y`` along with ``X``."""

    @pytest.fixture
    def positive_xy(self, linear_xy):
        """``linear_xy`` with ``X`` in absolute value: every value is positive."""
        X, y = linear_xy
        return X.abs(), y

    def test_fit_transform_transforms_x(self, positive_xy):
        """Valeur d'or : X' = log1p(X)."""
        X, y = positive_xy

        X_t, _y_t = XYPipeline([('log_xy', LogTransformXY())]).fit_transform(X, y)

        np.testing.assert_allclose(X_t.to_numpy(), np.log1p(X.to_numpy()))

    def test_fit_transform_transforms_y(self, positive_xy):
        """Valeur d'or : y' = log1p(y)."""
        X, y = positive_xy

        _X_t, y_t = XYPipeline([('log_xy', LogTransformXY())]).fit_transform(X, y)

        np.testing.assert_allclose(y_t.to_numpy(), np.log1p(y.to_numpy()))

    def test_transform_with_target_transforms_y(self, positive_xy):
        """``transform(X, y)`` after ``fit`` gives the same ``y'``."""
        X, y = positive_xy
        pipeline = XYPipeline([('scaler', MinMaxScaler()), ('log_xy', LogTransformXY())]).fit(X, y)

        _X_t, y_t = pipeline.transform(X, y)

        np.testing.assert_allclose(y_t.to_numpy(), np.log1p(y.to_numpy()))

    def test_transform_without_target_returns_x_alone(self, positive_xy):
        """``transform(X)`` returns ``X'``: log1p of the min-max scaled features."""
        X, y = positive_xy
        pipeline = XYPipeline([('scaler', MinMaxScaler()), ('log_xy', LogTransformXY())]).fit(X, y)

        np.testing.assert_allclose(pipeline.transform(X).to_numpy(), np.log1p(MinMaxScaler().fit_transform(X)))

    def test_inverse_transform_round_trip(self, positive_xy):
        """``inverse_transform`` gives back the positive ``X`` and ``y``."""
        X, y = positive_xy
        pipeline = XYPipeline([('log_xy', LogTransformXY())])
        X_t, y_t = pipeline.fit_transform(X, y)

        _X_inv, y_inv = pipeline.inverse_transform(X_t, y_t)

        np.testing.assert_allclose(y_inv.to_numpy(), y.to_numpy())

    def test_xy_steps_compose(self, positive_xy):
        """Valeur d'or : ScaleXY(2) puis log → y' = log1p(2y)."""
        X, y = positive_xy

        _X_t, y_t = XYPipeline([('scale_xy', ScaleXY(scale_factor=2.0)), ('log_xy', LogTransformXY())]).fit_transform(X, y)

        np.testing.assert_allclose(y_t.to_numpy(), np.log1p(2.0 * y.to_numpy()))


# ==============================================================================
# Préservation pandas
# ==============================================================================


class TestPandasPreservation:
    """Pandas inputs come out as pandas objects, with their labels."""

    def test_columns_are_kept(self, linear_xy):
        """A scaler output gets the input columns back."""
        X, y = linear_xy

        X_t = XYPipeline([('scaler', StandardScaler())], preserve_dataframe=True).fit_transform(X, y)

        assert list(X_t.columns) == ['a', 'b']

    def test_training_index_is_kept(self, linear_xy):
        """``fit_transform`` gives back the training dates."""
        X, y = linear_xy

        X_t = XYPipeline([('scaler', StandardScaler())]).fit_transform(X, y)

        assert X_t.index.equals(X.index)

    def test_series_target_stays_a_series_through_an_xy_step(self, linear_xy):
        """A ``Series`` target goes through an XY step as a ``Series`` with its name."""
        X, y = linear_xy

        _X_t, y_t = XYPipeline([('log_xy', LogTransformXY())]).fit_transform(X, y)

        assert y_t.name == 't'

    def test_array_inputs_stay_arrays(self, linear_xy):
        """Array inputs have no structure to restore: the outputs are arrays."""
        X, y = linear_xy

        _X_t, y_t = XYPipeline([('log_xy', LogTransformXY())]).fit_transform(X.to_numpy(), y.to_numpy())

        assert isinstance(y_t, np.ndarray)

    def test_restoration_can_be_disabled(self, linear_xy):
        """``preserve_dataframe=False`` returns the scaler array."""
        X_t = XYPipeline([('scaler', StandardScaler())], preserve_dataframe=False).fit_transform(*linear_xy)

        assert isinstance(X_t, np.ndarray)

    @pytest.mark.parametrize('periods', [6, 4], ids=['same_length', 'shorter'])
    def test_transform_of_new_data_keeps_its_index(self, periods, linear_xy):
        """``transform`` of later dates comes out with those dates."""
        X, y = linear_xy
        pipeline = XYPipeline([('scaler', StandardScaler())]).fit(X, y)
        # Construction de dates postérieures à l'apprentissage
        new_index = pd.date_range(month_ends()[-1] + pd.offsets.MonthEnd(1), periods=periods, freq='ME', name='date')
        X_new = pd.DataFrame(X.to_numpy()[:periods], columns=X.columns, index=new_index)

        assert pipeline.transform(X_new).index.equals(new_index)

    def test_new_features_are_named_by_the_steps(self, linear_xy):
        """A PCA output is named ``pca0``, ``pca1``, not after the input columns."""
        X, y = linear_xy

        X_t = XYPipeline([('pca', PCA(n_components=2, svd_solver='full'))]).fit_transform(X, y)

        assert list(X_t.columns) == ['pca0', 'pca1']

    def test_step_without_feature_names_keeps_the_input_columns(self, linear_xy):
        """Same width and a step without ``get_feature_names_out``: input columns kept."""
        X, y = linear_xy

        X_t = XYPipeline([('scaler', StandardScaler()), ('log_xy', LogTransformXY())]).fit_transform(X)

        assert list(X_t.columns) == ['a', 'b']


# ==============================================================================
# Cas limites de composition
# ==============================================================================


class TestEdgeCases:
    """``'passthrough'`` / ``None`` steps, single step, slicing, parameters."""

    @pytest.mark.parametrize('skipped', ['passthrough', None], ids=['passthrough', 'none'])
    def test_skipped_step_changes_nothing(self, skipped, linear_xy):
        """A skipped step gives the predictions of the pipeline without it."""
        X, y = linear_xy
        with_skip = XYPipeline([('skip', skipped), ('scaler', StandardScaler()), ('reg', Ridge())])
        without = XYPipeline([('scaler', StandardScaler()), ('reg', Ridge())])

        np.testing.assert_array_equal(with_skip.fit(X, y).predict(X), without.fit(X, y).predict(X))

    def test_single_transformer_without_target(self, linear_xy):
        """A one-step pipeline transforms without ``y``."""
        X, _y = linear_xy

        X_t = XYPipeline([('scaler', StandardScaler())]).fit_transform(X)

        np.testing.assert_allclose(X_t.to_numpy(), _scaled(X))

    def test_slice_keeps_the_first_steps(self, linear_xy):
        """``pipe[:2]`` keeps the first two steps, in order."""
        pipeline = XYPipeline([('scaler', StandardScaler()), ('minmax', MinMaxScaler()), ('reg', LinearRegression())])
        pipeline.fit(*linear_xy)

        assert [name for name, _ in pipeline[:2].steps] == ['scaler', 'minmax']

    def test_slice_is_an_xy_pipeline(self, linear_xy):
        """The slice is itself an ``XYPipeline``, sharing the fitted steps."""
        pipeline = XYPipeline([('scaler', StandardScaler()), ('minmax', MinMaxScaler()), ('reg', LinearRegression())])
        pipeline.fit(*linear_xy)

        assert isinstance(pipeline[:2], XYPipeline)

    def test_get_params_exposes_the_nested_parameters(self):
        """``get_params(deep=True)`` lists the steps and their parameters."""
        params = XYPipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=0.5))]).get_params(deep=True)

        assert (('scaler' in params), ('reg' in params), params['reg__alpha']) == (True, True, 0.5)

    def test_set_params_reaches_the_step(self):
        """``set_params(reg__alpha=...)`` changes the step parameter."""
        pipeline = XYPipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=0.5))])

        pipeline.set_params(reg__alpha=1.0)

        assert pipeline.named_steps['reg'].alpha == 1.0
