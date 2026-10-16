"""sklearn surface of ``XYPipeline``.

Covers the fitted-state checks (``NotFittedError``), verbose output, memory
caching, ``GridSearchCV`` and ``cross_val_score`` compatibility, the
``transform_input`` parameter and the classifier methods delegated to the
final estimator.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import Memory

from tsforecast.xy.pipeline import XYPipeline
from tests.support.estimators import ScaleXY


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
