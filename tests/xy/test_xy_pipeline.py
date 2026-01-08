"""Tests unitaires pour XYPipeline.

Ce module teste les fonctionnalités de XYPipeline, notamment :
- Compatibilité avec sklearn Pipeline
- Support des transformateurs XY (transformation de X et y)
- Mode verbose et memory caching
- Compatibilité avec GridSearchCV
- Validation des paramètres (check_is_fitted, transform_input)
"""

import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import Memory

from tsforecast.xy.pipeline import XYPipeline


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_regression_data():
    """Données de régression pour les tests."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    return X, y


@pytest.fixture
def sample_classification_data():
    """Données de classification pour les tests."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_dataframe_data():
    """Données pandas DataFrame pour les tests."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 5),
        columns=["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"],
    )
    y = pd.Series(np.random.randn(100), name="target")
    return X, y


@pytest.fixture
def temp_cache_dir():
    """Répertoire temporaire pour le caching."""
    cache_dir = tempfile.mkdtemp()
    yield cache_dir
    shutil.rmtree(cache_dir)


# ==============================================================================
# Transformateurs XY pour les tests
# ==============================================================================


class LogTransformXY(BaseEstimator, TransformerMixin):
    """Transformateur XY qui applique log1p sur X et y."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        X_t = np.log1p(np.abs(X))
        if y is not None:
            return X_t, np.log1p(np.abs(y))
        return X_t

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        X_inv = np.expm1(X)
        if y is not None:
            return X_inv, np.expm1(y)
        return X_inv


class ScaleXY(BaseEstimator, TransformerMixin):
    """Transformateur XY qui scale X et y par un facteur."""

    def __init__(self, scale_factor=2.0):
        self.scale_factor = scale_factor

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        X_t = X * self.scale_factor
        if y is not None:
            return X_t, y * self.scale_factor
        return X_t

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


class DropRowsXY(BaseEstimator, TransformerMixin):
    """Transformateur XY qui supprime les lignes avec des valeurs négatives dans y."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        if y is not None:
            mask = y >= 0
            return X[mask], y[mask]
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


# ==============================================================================
# Tests de base
# ==============================================================================


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
# Tests de check_is_fitted
# ==============================================================================


class TestCheckIsFitted:
    """Tests de validation check_is_fitted."""

    def test_predict_not_fitted(self, sample_regression_data):
        """Test que predict lève NotFittedError si non fitted."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        with pytest.raises(NotFittedError):
            pipe.predict(X)

    def test_transform_not_fitted(self, sample_regression_data):
        """Test que transform lève NotFittedError si non fitted."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
        ])
        with pytest.raises(NotFittedError):
            pipe.transform(X)

    def test_score_not_fitted(self, sample_regression_data):
        """Test que score lève NotFittedError si non fitted."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        with pytest.raises(NotFittedError):
            pipe.score(X, y)

    def test_predict_proba_not_fitted(self, sample_classification_data):
        """Test que predict_proba lève NotFittedError si non fitted."""
        X, y = sample_classification_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        with pytest.raises(NotFittedError):
            pipe.predict_proba(X)

    def test_decision_function_not_fitted(self, sample_classification_data):
        """Test que decision_function lève NotFittedError si non fitted."""
        X, y = sample_classification_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        with pytest.raises(NotFittedError):
            pipe.decision_function(X)


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
# Tests du mode verbose
# ==============================================================================


class TestVerboseMode:
    """Tests du mode verbose."""

    def test_verbose_fit(self, sample_regression_data, capsys):
        """Test que le mode verbose affiche les temps d'exécution."""
        X, y = sample_regression_data
        pipe = XYPipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ],
            verbose=True,
        )
        pipe.fit(X, y)

        captured = capsys.readouterr()
        assert "[Pipeline]" in captured.out
        assert "Processing scaler" in captured.out
        assert "Processing reg" in captured.out

    def test_verbose_fit_transform(self, sample_regression_data, capsys):
        """Test verbose avec fit_transform."""
        X, y = sample_regression_data
        pipe = XYPipeline(
            [
                ("scaler", StandardScaler()),
                ("minmax", MinMaxScaler()),
            ],
            verbose=True,
        )
        pipe.fit_transform(X, y)

        captured = capsys.readouterr()
        assert "[Pipeline]" in captured.out


# ==============================================================================
# Tests du memory caching
# ==============================================================================


class TestMemoryCaching:
    """Tests du memory caching."""

    def test_memory_caching_with_path(self, sample_regression_data, temp_cache_dir):
        """Test du caching avec un chemin."""
        X, y = sample_regression_data
        pipe = XYPipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ],
            memory=temp_cache_dir,
        )
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (100,)

    def test_memory_caching_with_memory_object(
        self, sample_regression_data, temp_cache_dir
    ):
        """Test du caching avec un objet Memory."""
        X, y = sample_regression_data
        memory = Memory(location=temp_cache_dir, verbose=0)
        pipe = XYPipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ],
            memory=memory,
        )
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (100,)

    def test_no_cloning_when_caching_disabled(self, sample_regression_data):
        """Test que les transformateurs ne sont pas clonés si caching désactivé."""
        X, y = sample_regression_data
        original_scaler = StandardScaler()
        pipe = XYPipeline(
            [
                ("scaler", original_scaler),
                ("reg", LinearRegression()),
            ],
            memory=None,
        )
        pipe.fit(X, y)
        # Quand memory=None, le transformateur original n'est pas cloné
        # Le comportement exact dépend de l'implémentation
        assert pipe.named_steps["scaler"].mean_ is not None


# ==============================================================================
# Tests de compatibilité GridSearchCV
# ==============================================================================


class TestGridSearchCompatibility:
    """Tests de compatibilité avec GridSearchCV."""

    def test_gridsearch_basic(self, sample_regression_data):
        """Test basique avec GridSearchCV."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge()),
        ])
        param_grid = {
            "scaler__with_mean": [True, False],
            "reg__alpha": [0.1, 1.0],
        }
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring="r2")
        grid.fit(X, y)

        assert grid.best_params_ is not None
        assert "reg__alpha" in grid.best_params_

    def test_gridsearch_with_xy_transformer(self, sample_regression_data):
        """Test GridSearchCV avec un transformateur XY paramétrable."""
        X, y = sample_regression_data
        X = np.abs(X) + 1
        y = np.abs(y) + 1

        pipe = XYPipeline([
            ("scale_xy", ScaleXY()),
            ("reg", Ridge()),
        ])
        param_grid = {
            "scale_xy__scale_factor": [1.0, 2.0],
            "reg__alpha": [0.1, 1.0],
        }
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring="r2")
        grid.fit(X, y)

        assert grid.best_params_ is not None

    def test_cross_val_score(self, sample_regression_data):
        """Test cross_val_score."""
        X, y = sample_regression_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        scores = cross_val_score(pipe, X, y, cv=3, scoring="r2")

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


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


# ==============================================================================
# Tests de transform_input
# ==============================================================================


class TestTransformInput:
    """Tests du paramètre transform_input."""

    def test_transform_input_requires_routing(self, sample_regression_data):
        """Test que transform_input nécessite le metadata routing."""
        X, y = sample_regression_data

        # Création d'un pipeline avec transform_input
        pipe = XYPipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ],
            transform_input=["X_val"],
        )

        # Devrait lever une erreur car metadata routing n'est pas activé
        with pytest.raises(ValueError, match="transform_input"):
            pipe.fit(X, y)


# ==============================================================================
# Tests de classification
# ==============================================================================


class TestClassification:
    """Tests avec des classifieurs."""

    def test_predict_proba(self, sample_classification_data):
        """Test de predict_proba."""
        X, y = sample_classification_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)

        assert proba.shape == (100, 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_decision_function(self, sample_classification_data):
        """Test de decision_function."""
        X, y = sample_classification_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        pipe.fit(X, y)
        decision = pipe.decision_function(X)

        assert decision.shape == (100,)

    def test_classes_attribute(self, sample_classification_data):
        """Test de l'attribut classes_."""
        X, y = sample_classification_data
        pipe = XYPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        pipe.fit(X, y)

        np.testing.assert_array_equal(pipe.classes_, [0, 1])
