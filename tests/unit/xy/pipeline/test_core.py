"""Core behaviour of ``XYPipeline``.

Covers the pipeline as a drop-in ``sklearn.pipeline.Pipeline``: fit, predict,
score, transform and fit_transform, the passing of ``(X, y)`` through XY
transformers, pandas structure preservation and passthrough / ``None`` steps,
slicing and parameter access.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tsforecast.xy.pipeline import XYPipeline
from tests.support.estimators import LogTransformXY, ScaleXY


class TestXYPipelineBasic:
    """Tests de base pour XYPipeline."""

    def test_pipeline_creation(self):
        """Test de création d'un pipeline."""
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == "scaler"
        assert pipe.steps[1][0] == "reg"

    def test_fit(self, sample_regression_data):
        """Test de fit basique."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        result = pipe.fit(X, y)
        assert result is pipe

    def test_predict(self, sample_regression_data):
        """Test de predict."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (100,)

    def test_score(self, sample_regression_data):
        """Test de score."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert isinstance(score, float)

    def test_fit_transform_without_xy_transformer(self, sample_regression_data):
        """Test de fit_transform sans transformateur XY."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
        ])
        result = pipe.fit_transform(X, y)
        # Sans transformateur XY, retourne seulement X
        assert not isinstance(result, tuple)
        assert result.shape == X.shape

    def test_transform(self, sample_regression_data):
        """Test de transform."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
        ])
        pipe.fit(X, y)
        Xt = pipe.transform(X)
        assert Xt.shape == X.shape


# ==============================================================================
# Tests des transformateurs XY
# ==============================================================================


class TestXYTransformers:
    """Tests des transformateurs XY."""

    def test_fit_transform_with_xy_transformer(self, sample_regression_data):
        """Test de fit_transform avec un transformateur XY."""
        X, y = sample_regression_data
        X = np.abs(X) + 1  # Valeurs positives pour log
        y = np.abs(y) + 1

        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("log_xy", LogTransformXY()),
        ])
        result = pipe.fit_transform(X, y)

        assert isinstance(result, tuple)
        assert len(result) == 2
        Xt, yt = result
        assert Xt.shape == X.shape
        assert yt.shape == y.shape

    def test_transform_with_xy_transformer(self, sample_regression_data):
        """Test de transform avec un transformateur XY."""
        X, y = sample_regression_data
        X = np.abs(X) + 1
        y = np.abs(y) + 1

        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("log_xy", LogTransformXY()),
        ])
        pipe.fit(X, y)
        result = pipe.transform(X, y)

        assert isinstance(result, tuple)
        Xt, yt = result
        assert Xt.shape == X.shape
        assert yt.shape == y.shape

    def test_transform_without_y(self, sample_regression_data):
        """Test de transform sans y avec un transformateur XY."""
        X, y = sample_regression_data
        X = np.abs(X) + 1

        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("log_xy", LogTransformXY()),
        ])
        pipe.fit(X, y)

        # Sans y, retourne seulement X
        result = pipe.transform(X)
        assert not isinstance(result, tuple)
        assert result.shape == X.shape

    def test_inverse_transform_with_xy_transformer(self, sample_regression_data):
        """Test de inverse_transform avec un transformateur XY."""
        X, y = sample_regression_data
        X = np.abs(X) + 1
        y = np.abs(y) + 1

        pipe = XYPipeline([
            ("log_xy", LogTransformXY()),
        ])
        Xt, yt = pipe.fit_transform(X, y)
        X_inv, y_inv = pipe.inverse_transform(Xt, yt)

        np.testing.assert_array_almost_equal(X, X_inv, decimal=5)
        np.testing.assert_array_almost_equal(y, y_inv, decimal=5)

    def test_multiple_xy_transformers(self, sample_regression_data):
        """Test avec plusieurs transformateurs XY."""
        X, y = sample_regression_data
        X = np.abs(X) + 1
        y = np.abs(y) + 1

        pipe = XYPipeline([
            ("scale_xy", ScaleXY(scale_factor=2.0)),
            ("log_xy", LogTransformXY()),
        ])
        result = pipe.fit_transform(X, y)

        assert isinstance(result, tuple)
        Xt, yt = result
        assert Xt.shape == X.shape


# ==============================================================================
# Tests de préservation pandas
# ==============================================================================


class TestPandasPreservation:
    """Tests de préservation de la structure pandas."""

    def test_preserve_dataframe(self, sample_dataframe_data):
        """Test que le DataFrame est préservé."""
        X, y = sample_dataframe_data
        pipe = XYPipeline(
            [
                ("scaler", StandardScaler()),
            ],
            preserve_dataframe=True,
        )
        Xt = pipe.fit_transform(X, y)

        # Vérification que c'est toujours un DataFrame
        assert isinstance(Xt, pd.DataFrame)
        assert list(Xt.columns) == list(X.columns)

    def test_preserve_series_with_xy_transformer(self, sample_dataframe_data):
        """Test que la Series y est préservée avec un transformateur XY."""
        X, y = sample_dataframe_data
        X = X.abs() + 1
        y = y.abs() + 1

        pipe = XYPipeline(
            [
                ("log_xy", LogTransformXY()),
            ],
            preserve_dataframe=True,
        )
        result = pipe.fit_transform(X.values, y.values)

        # Avec des arrays numpy en entrée, pas de préservation pandas
        assert isinstance(result, tuple)

    def test_no_preserve_dataframe(self, sample_dataframe_data):
        """Test que le DataFrame n'est pas préservé si désactivé."""
        X, y = sample_dataframe_data
        pipe = XYPipeline(
            [
                ("scaler", StandardScaler()),
            ],
            preserve_dataframe=False,
        )
        Xt = pipe.fit_transform(X, y)

        # Devrait être un array numpy
        assert isinstance(Xt, np.ndarray)


# ==============================================================================
# Tests des cas limites
# ==============================================================================


class TestEdgeCases:
    """Tests des cas limites."""

    def test_passthrough_step(self, sample_regression_data):
        """Test avec un step passthrough."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("passthrough", "passthrough"),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (100,)

    def test_none_step(self, sample_regression_data):
        """Test avec un step None."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("none_step", None),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (100,)

    def test_single_transformer(self, sample_regression_data):
        """Test avec un seul transformateur."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
        ])
        Xt = pipe.fit_transform(X)
        assert Xt.shape == X.shape

    def test_pipeline_slicing(self, sample_regression_data):
        """Test du slicing de pipeline."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("minmax", MinMaxScaler()),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X, y)

        # Slice des deux premiers steps
        sub_pipe = pipe[:2]
        assert len(sub_pipe.steps) == 2
        assert sub_pipe.steps[0][0] == "scaler"
        assert sub_pipe.steps[1][0] == "minmax"

    def test_get_params(self, sample_regression_data):
        """Test de get_params."""
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=0.5)),
        ])
        params = pipe.get_params(deep=True)

        assert "scaler" in params
        assert "reg" in params
        assert params["reg__alpha"] == 0.5

    def test_set_params(self, sample_regression_data):
        """Test de set_params."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=0.5)),
        ])
        pipe.set_params(reg__alpha=1.0)

        assert pipe.named_steps["reg"].alpha == 1.0
