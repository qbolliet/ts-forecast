"""Unit tests of ``tsforecast.xy.transformers.DropNaNTarget``.

Covers the rows dropped by the training pass (``fit_transform``) and by
``transform(X, y)`` — ``Series``, ``DataFrame`` (multi-output) and array
targets —, the prediction pass without ``y`` (no row lost), the edge cases
(no NaN, all NaN, empty input, length mismatch), the sklearn contract of a
stateless step (fitted without ``fit``, ``get_params``, ``clone``) and its
place in an ``XYPipeline`` (rows dropped at ``fit`` and ``score``, kept at
``predict``).
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

# Objets testés
from tsforecast.xy import DropNaNTarget, XYPipeline


@pytest.fixture
def frame_xy():
    """``(X, y)``: 5 dated rows, target NaN on the 2nd and the 5th."""
    dates = pd.date_range('2024-01-01', periods=5, freq='MS', name='date')
    X = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'z': [0.0, 1.0, 0.0, 1.0, 0.0]}, index=dates)
    y = pd.Series([10.0, np.nan, 30.0, 40.0, np.nan], index=dates, name='y')
    return X, y


# =============================================================================
# Lignes retirées
# =============================================================================
class TestRowsDropped:
    """The rows whose target is missing leave ``X`` and ``y`` together."""

    def test_series_target_drops_its_nan_rows(self, frame_xy):
        """Valeur d'or : lignes 1, 3, 4 conservées (cible 10, 30, 40)."""
        X_kept, y_kept = DropNaNTarget().fit_transform(*frame_xy)

        assert (X_kept['x'].tolist(), y_kept.tolist()) == ([1.0, 3.0, 4.0], [10.0, 30.0, 40.0])

    def test_kept_rows_keep_their_index(self, frame_xy):
        """The index of the kept rows is the input index minus the dropped dates."""
        X, y = frame_xy

        X_kept, y_kept = DropNaNTarget().fit_transform(X, y)

        assert X_kept.index.equals(X.index[[0, 2, 3]]) and y_kept.index.equals(X_kept.index)

    def test_transform_with_target_drops_too(self, frame_xy):
        """``transform(X, y)`` — the ``score`` path — drops like ``fit_transform``."""
        X, y = frame_xy
        step = DropNaNTarget().fit(X, y)

        X_kept, _y_kept = step.transform(X, y)

        assert len(X_kept) == 3

    def test_multi_output_target_drops_a_row_with_any_nan(self, frame_xy):
        """A row goes as soon as one of its targets is missing."""
        X, y = frame_xy
        targets = pd.DataFrame({'a': y, 'b': [1.0, 2.0, np.nan, 4.0, 5.0]}, index=X.index)

        X_kept, y_kept = DropNaNTarget().fit_transform(X, targets)

        assert X_kept['x'].tolist() == [1.0, 4.0] and isinstance(y_kept, pd.DataFrame)

    def test_arrays_are_supported(self, frame_xy):
        """NumPy inputs come out as arrays, filtered by position."""
        X, y = frame_xy

        X_kept, y_kept = DropNaNTarget().fit_transform(X.to_numpy(), y.to_numpy())

        assert isinstance(X_kept, np.ndarray) and y_kept.tolist() == [10.0, 30.0, 40.0]

    def test_rows_are_matched_by_position(self, frame_xy):
        """``X`` and ``y`` with different index labels are still filtered row by row."""
        X, y = frame_xy

        _X_kept, y_kept = DropNaNTarget().fit_transform(X, y.reset_index(drop=True))

        assert y_kept.index.tolist() == [0, 2, 3]


# =============================================================================
# Passe de prédiction et cas limites
# =============================================================================
class TestEdgeCases:
    """No ``y``, nothing to drop, everything to drop, inconsistent lengths."""

    def test_without_target_x_goes_through(self, frame_xy):
        """``transform(X)`` — the ``predict`` path — returns ``X`` untouched."""
        X, y = frame_xy
        step = DropNaNTarget().fit(X, y)

        assert step.transform(X) is X

    def test_complete_target_drops_nothing(self, frame_xy):
        """A target without NaN leaves every row."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().fit_transform(X, y.fillna(0.0))

        assert X_kept.equals(X)

    def test_all_nan_target_drops_everything(self, frame_xy):
        """A target entirely NaN leaves two empty outputs."""
        X, y = frame_xy

        X_kept, y_kept = DropNaNTarget().fit_transform(X, y * np.nan)

        assert (len(X_kept), len(y_kept)) == (0, 0)

    def test_empty_input_stays_empty(self, frame_xy):
        """Zero rows in, zero rows out."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().fit_transform(X.iloc[:0], y.iloc[:0])

        assert X_kept.empty

    def test_length_mismatch_is_refused(self, frame_xy):
        """``X`` and ``y`` of different lengths raise, naming both lengths."""
        X, y = frame_xy

        with pytest.raises(ValueError, match='5 and 4'):
            DropNaNTarget().fit_transform(X, y.iloc[:4])


# =============================================================================
# Contrat sklearn d'une étape sans état
# =============================================================================
class TestSklearnContract:
    """Stateless step: fitted from the start, no parameter, clonable."""

    def test_reported_fitted_without_fit(self):
        """``check_is_fitted`` passes on a fresh instance: there is nothing to learn."""
        check_is_fitted(DropNaNTarget())

    def test_has_no_parameter(self):
        """``get_params`` is empty."""
        assert DropNaNTarget().get_params() == {}

    def test_clone_gives_a_new_step(self):
        """``clone`` returns another instance of the class."""
        step = DropNaNTarget()

        assert type(clone(step)) is DropNaNTarget and clone(step) is not step


# =============================================================================
# Dans un XYPipeline
# =============================================================================
class TestInXYPipeline:
    """Placed before the estimator: rows dropped at ``fit`` and ``score``, kept at ``predict``."""

    @pytest.fixture
    def pipeline(self):
        """``DropNaNTarget -> LinearRegression``, which refuses a NaN target."""
        return XYPipeline([('drop', DropNaNTarget()), ('model', LinearRegression())])

    def test_fit_succeeds_despite_nan_targets(self, pipeline, frame_xy):
        """The estimator is fitted on the 3 rows with a target."""
        pipeline.fit(*frame_xy)

        assert pipeline.named_steps['model'].n_features_in_ == 2

    def test_predict_keeps_every_row(self, pipeline, frame_xy):
        """``predict`` returns one value per input row, NaN targets included."""
        X, y = frame_xy

        predictions = pipeline.fit(X, y).predict(X)

        assert len(predictions) == len(X)

    def test_score_ignores_the_nan_targets(self, pipeline, frame_xy):
        """``score`` drops the rows without target before scoring: a finite R²."""
        X, y = frame_xy

        assert np.isfinite(pipeline.fit(X, y).score(X, y))
