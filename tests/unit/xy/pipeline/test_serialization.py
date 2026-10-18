"""Serialization of ``XYPipeline`` with ``joblib`` and ``pickle``.

Covers the round trip of a fitted pipeline (``joblib.dump`` / ``load``,
compressed or not, and ``pickle``): class and parameters kept, identical
predictions, scores and transforms (rows dropped by ``DropNaNTarget``, pandas
structure restored), the fitted state of a minimal ``XYTransformerMixin``
subclass, an unfitted pipeline that stays unfitted, and a loaded copy that is
independent from the original.
"""
import pickle
from pathlib import Path
from typing import Any, Callable, Dict

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from tsforecast.xy import DropNaNTarget, XYPipeline
from tests.support.estimators import MixinScaleXY

from .conftest import GAPPED_ROWS


def _joblib(pipeline: Any, folder: Path, compress: int = 0) -> Any:
    """Dump ``pipeline`` to a file of ``folder`` with joblib and load it back."""
    path = folder / 'pipeline.joblib'
    joblib.dump(pipeline, path, compress=compress)
    return joblib.load(path)


ROUND_TRIPS: Dict[str, Callable[[Any, Path], Any]] = {
    'joblib': lambda pipeline, folder: _joblib(pipeline, folder),
    'joblib_compressed': lambda pipeline, folder: _joblib(pipeline, folder, compress=3),
    'pickle': lambda pipeline, folder: pickle.loads(pickle.dumps(pipeline)),
}

round_trips = pytest.mark.parametrize('round_trip', list(ROUND_TRIPS), ids=list(ROUND_TRIPS))


def _steps():
    """Fresh steps: scale, drop the rows without target, ridge."""
    return [('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', Ridge(alpha=1.0))]


@pytest.fixture
def fitted(gapped_linear_xy):
    """``[scale, drop, ridge]`` fitted on ``gapped_linear_xy``."""
    return XYPipeline(_steps()).fit(*gapped_linear_xy)


@round_trips
class TestFittedRoundTrip:
    """A fitted pipeline comes back with the same class, parameters and fitted state."""

    def test_class_is_kept(self, round_trip, fitted, tmp_path):
        """The loaded object is an ``XYPipeline``."""
        assert type(ROUND_TRIPS[round_trip](fitted, tmp_path)) is XYPipeline

    def test_parameters_are_kept(self, round_trip, gapped_linear_xy, tmp_path):
        """``preserve_dataframe=False`` and the step parameters survive the round trip."""
        pipeline = XYPipeline(_steps(), preserve_dataframe=False).fit(*gapped_linear_xy)

        loaded = ROUND_TRIPS[round_trip](pipeline, tmp_path)

        assert (loaded.preserve_dataframe, loaded['model'].alpha) == (False, 1.0)

    def test_predictions_are_identical(self, round_trip, fitted, gapped_linear_xy, tmp_path):
        """The loaded pipeline gives exactly the same predictions."""
        X, _y = gapped_linear_xy

        np.testing.assert_array_equal(ROUND_TRIPS[round_trip](fitted, tmp_path).predict(X), fitted.predict(X))

    def test_score_is_identical(self, round_trip, fitted, gapped_linear_xy, tmp_path):
        """The loaded pipeline gives exactly the same score, rows without target dropped."""
        X, y = gapped_linear_xy

        assert ROUND_TRIPS[round_trip](fitted, tmp_path).score(X, y) == fitted.score(X, y)

    def test_transform_with_target_drops_the_same_rows(self, round_trip, fitted, gapped_linear_xy, tmp_path):
        """``transform(X, y)`` of the loaded pipeline drops the rows without target."""
        X, y = gapped_linear_xy

        _X_t, y_t = ROUND_TRIPS[round_trip](fitted, tmp_path)[:-1].transform(X, y)

        assert y_t.index.equals(X.index.delete(list(GAPPED_ROWS)))

    def test_transform_restores_the_pandas_structure(self, round_trip, fitted, gapped_linear_xy, tmp_path):
        """``transform(X)`` of the loaded slice keeps the index and columns of ``X``."""
        X, _y = gapped_linear_xy

        X_t = ROUND_TRIPS[round_trip](fitted, tmp_path)[:-1].transform(X)

        assert (X_t.index.equals(X.index), list(X_t.columns)) == (True, ['a', 'b'])

    def test_mixin_subclass_stays_fitted(self, round_trip, linear_xy, tmp_path):
        """A minimal ``XYTransformerMixin`` subclass is still fitted once loaded."""
        pipeline = XYPipeline([('xy', MixinScaleXY(k=2.0)), ('model', LinearRegression())]).fit(*linear_xy)

        check_is_fitted(ROUND_TRIPS[round_trip](pipeline, tmp_path)['xy'])

    def test_mixin_subclass_predicts_identically(self, round_trip, linear_xy, tmp_path):
        """A pipeline built on a minimal ``XYTransformerMixin`` subclass predicts the same once loaded."""
        X, y = linear_xy
        pipeline = XYPipeline([('xy', MixinScaleXY(k=2.0)), ('model', LinearRegression())]).fit(X, y)

        np.testing.assert_array_equal(ROUND_TRIPS[round_trip](pipeline, tmp_path).predict(X), pipeline.predict(X))


@round_trips
class TestUnfittedAndIndependence:
    """An unfitted pipeline stays unfitted; a loaded copy does not share state with the original."""

    def test_unfitted_pipeline_stays_unfitted(self, round_trip, gapped_linear_xy, tmp_path):
        """The loaded copy of an unfitted pipeline refuses to predict."""
        X, _y = gapped_linear_xy
        loaded = ROUND_TRIPS[round_trip](XYPipeline(_steps()), tmp_path)

        with pytest.raises(NotFittedError):
            loaded.predict(X)

    def test_refitting_the_copy_leaves_the_original(self, round_trip, fitted, gapped_linear_xy, tmp_path):
        """Refitting the loaded copy on other data leaves the predictions of the original unchanged."""
        X, y = gapped_linear_xy
        before = fitted.predict(X)

        ROUND_TRIPS[round_trip](fitted, tmp_path).fit(X * 10.0, y)

        np.testing.assert_array_equal(fitted.predict(X), before)

    def test_loaded_copy_refits_to_the_same_predictions(self, round_trip, fitted, gapped_linear_xy, tmp_path):
        """Refitted on the same data, the loaded copy predicts like the original."""
        X, y = gapped_linear_xy

        np.testing.assert_allclose(ROUND_TRIPS[round_trip](fitted, tmp_path).fit(X, y).predict(X), fitted.predict(X))
