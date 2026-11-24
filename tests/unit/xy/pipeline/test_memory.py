"""Caching of the fitted steps (``memory=``) of ``XYPipeline``, compared with ``Pipeline``.

Covers the reuse of a cached step by a second fit on the same data (ordinary
and XY steps, cache given as a path or as a ``joblib.Memory``, array and dated
pandas inputs), the refit on new data or another index, the results of a cached pipeline (predictions, target given to the
final estimator), the cloning of the given steps when a cache is set, the fit
in place with ``Memory(location=None)`` and the refusal of an invalid
``memory``.
"""
import numpy as np
import pandas as pd
import pytest
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tsforecast.xy import DropNaNTarget, XYPipeline, XYTransformerMixin
from tests.support.estimators import SpyEstimator


class CountingTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer counting, in a shared class counter, the fits really computed.

    A fit loaded from a cache does not run ``fit``, so it leaves the counter
    unchanged.
    """

    n_fits = 0

    def fit(self, X, y=None):
        """Count the fit."""
        CountingTransformer.n_fits += 1
        self.fitted_ = True
        return self

    def transform(self, X):
        """Return ``X``."""
        return X


class CountingXY(BaseEstimator, XYTransformerMixin):
    """XY identity step (``(X, y) -> (X, y)``) counting its fits in the counter of ``CountingTransformer``."""

    def fit(self, X, y=None):
        """Count the fit."""
        CountingTransformer.n_fits += 1
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        """Return ``X``, or ``(X, y)`` when a target is given."""
        return X if y is None else (X, y)


@pytest.fixture(autouse=True)
def reset_counter():
    """Start each test with a zero fit counter."""
    CountingTransformer.n_fits = 0


# Formes du cache : chemin de répertoire ou objet joblib.Memory
MEMORIES = {
    'path': lambda tmp_path: str(tmp_path),
    'joblib_memory': lambda tmp_path: Memory(location=str(tmp_path), verbose=0),
}


def _fit_twice(pipeline_class, memory, X, y, step=CountingTransformer):
    """Fit two fresh pipelines ``[step, linear]`` sharing ``memory`` on ``(X, y)``; return the last one."""
    for _ in range(2):
        pipeline = pipeline_class([('count', step()), ('model', LinearRegression())], memory=memory)
        pipeline.fit(X, y)
    return pipeline


def _arrays(data):
    """``(X, y)`` of ``data`` as numpy arrays."""
    return tuple(np.asarray(item) for item in data)


# ==============================================================================
# Réutilisation du cache
# ==============================================================================


class TestCaching:
    """A second fit on the same data loads the cached steps instead of fitting them again."""

    def test_pipeline_fits_a_cached_step_once(self, tmp_path, linear_xy):
        """Reference: under ``Pipeline``, two identical fits on arrays sharing a cache fit the step once.

        This checks the counting double itself: a step loaded from the cache
        leaves the counter unchanged.
        """
        _fit_twice(Pipeline, str(tmp_path), *_arrays(linear_xy))

        assert CountingTransformer.n_fits == 1

    @pytest.mark.parametrize('memory', list(MEMORIES), ids=list(MEMORIES))
    def test_second_fit_reuses_the_cached_step(self, memory, tmp_path, linear_xy):
        """Two identical fits on arrays sharing a cache fit the ordinary step once."""
        _fit_twice(XYPipeline, MEMORIES[memory](tmp_path), *_arrays(linear_xy))

        assert CountingTransformer.n_fits == 1

    def test_second_fit_reuses_the_cached_xy_step(self, tmp_path, linear_xy):
        """Two identical fits sharing a cache fit the XY step once."""
        _fit_twice(XYPipeline, str(tmp_path), *_arrays(linear_xy), step=CountingXY)

        assert CountingTransformer.n_fits == 1

    @pytest.mark.parametrize('step', [CountingTransformer, CountingXY], ids=['ordinary', 'xy'])
    def test_pandas_inputs_are_cached(self, step, tmp_path, linear_xy):
        """On dated pandas inputs too, the second fit loads the step from the cache.

        The first fit fills internal caches of the date index, which enter the
        pickle of the frame: the cache key reads the content of the data instead.
        """
        _fit_twice(XYPipeline, str(tmp_path), *linear_xy, step=step)

        assert CountingTransformer.n_fits == 1

    def test_other_index_is_another_cache_entry(self, tmp_path, linear_xy):
        """Same values under other dates: the step is fitted again."""
        X, y = linear_xy
        later = X.index + pd.offsets.MonthEnd(12)
        for X_fit, y_fit in ((X, y), (X.set_axis(later), y.set_axis(later))):
            XYPipeline([('count', CountingTransformer()), ('model', LinearRegression())], memory=str(tmp_path)).fit(
                X_fit, y_fit
            )

        assert CountingTransformer.n_fits == 2

    def test_new_data_fits_the_step_again(self, tmp_path, linear_xy):
        """A fit on other data does not reuse the cache: the step is fitted twice."""
        X, y = _arrays(linear_xy)
        for factor in (1.0, 2.0):
            XYPipeline([('count', CountingTransformer()), ('model', LinearRegression())], memory=str(tmp_path)).fit(
                X * factor, y
            )

        assert CountingTransformer.n_fits == 2

    def test_cached_pipeline_predicts_like_the_uncached(self, tmp_path, gapped_linear_xy):
        """After two fits sharing a cache, the predictions are those of a pipeline without cache."""
        X, y = gapped_linear_xy
        steps = lambda: [('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', LinearRegression())]
        for _ in range(2):
            cached = XYPipeline(steps(), memory=str(tmp_path)).fit(X, y)

        np.testing.assert_allclose(cached.predict(X), XYPipeline(steps()).fit(X, y).predict(X))

    def test_cached_xy_step_still_filters_the_target(self, tmp_path, frame_xy):
        """On the second fit sharing a cache, the estimator still receives the 5 observed targets."""
        X, y = frame_xy
        for _ in range(2):
            pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', SpyEstimator())], memory=str(tmp_path))
            pipeline.fit(X, y)

        assert pipeline['model'].fit_y_.tolist() == [1.0, 3.0, 4.0, 5.0, 6.0]


# ==============================================================================
# Clonage des étapes
# ==============================================================================


class TestCloning:
    """With a cache, the given steps are cloned and stay unfitted, as in ``Pipeline``."""

    def test_given_step_stays_unfitted(self, tmp_path, linear_xy):
        """The scaler given to the pipeline learns nothing: its clone is fitted."""
        scaler = StandardScaler()

        XYPipeline([('scale', scaler), ('model', LinearRegression())], memory=str(tmp_path)).fit(*linear_xy)

        assert not hasattr(scaler, 'mean_')

    def test_fitted_step_is_a_copy(self, tmp_path, linear_xy):
        """The fitted step of the pipeline is not the given object."""
        scaler = StandardScaler()

        pipeline = XYPipeline([('scale', scaler), ('model', LinearRegression())], memory=str(tmp_path))

        assert pipeline.fit(*linear_xy)['scale'] is not scaler

    def test_same_cloning_as_pipeline(self, tmp_path, linear_xy):
        """``Pipeline`` leaves its given step unfitted as well."""
        scaler = StandardScaler()

        Pipeline([('scale', scaler), ('model', LinearRegression())], memory=str(tmp_path)).fit(*linear_xy)

        assert not hasattr(scaler, 'mean_')

    def test_memory_without_location_fits_in_place(self, linear_xy):
        """``Memory(location=None)`` disables caching: the given step is the fitted one."""
        scaler = StandardScaler()

        pipeline = XYPipeline([('scale', scaler), ('model', LinearRegression())], memory=Memory(location=None))

        assert pipeline.fit(*linear_xy)['scale'] is scaler

    def test_invalid_memory_is_refused(self, linear_xy):
        """A ``memory`` that is neither None, a path nor a cache object is refused at fit."""
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())], memory=3)

        with pytest.raises(
            ValueError,
            match="The 'memory' parameter of XYPipeline must be None, an instance of 'str' or an object "
            "implementing 'cache'",
        ):
            pipeline.fit(*linear_xy)


class TestCachingWithoutTarget:
    """An unsupervised fit (no ``y``) is cached as well."""

    def test_second_fit_without_target_reuses_the_cache(self, tmp_path, linear_xy):
        """Two identical fits without target sharing a cache fit the step once."""
        X, _y = linear_xy
        for _ in range(2):
            XYPipeline([('count', CountingTransformer()), ('scale', StandardScaler())], memory=str(tmp_path)).fit(X)

        assert CountingTransformer.n_fits == 1
