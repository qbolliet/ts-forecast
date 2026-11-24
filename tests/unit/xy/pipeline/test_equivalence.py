"""``XYPipeline`` as a drop-in replacement of ``sklearn.pipeline.Pipeline``.

Without any XY step, both pipelines are built from the same steps and must
give the same results: ``predict``, ``score``, ``transform``, ``fit_transform``,
``inverse_transform`` and the fitted attributes of the final estimator, for
several sklearn transformers (scalers, polynomial features, PCA, imputer,
feature selection using ``y``, function transformer), alone, chained or next
to a ``'passthrough'`` step. Also covers the calls each step receives (same
journal under both pipelines) and the one deliberate difference, the
``DataFrame`` output kept by ``preserve_dataframe``.
"""
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from tsforecast.xy import XYPipeline
from tests.support.estimators import SpyTransformer


# ==============================================================================
# Données et étapes
# ==============================================================================


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.Series]:
    """``(X, y)``: 30 dated rows, 4 strictly positive features, a noisy linear target.

    One missing feature value (row 3, column ``c``) exercises the imputer; the
    other transformers are given ``X`` with that value filled.
    """
    rng = np.random.default_rng(7)
    index = pd.date_range('2022-01-31', periods=30, freq='ME', name='date')
    X = pd.DataFrame(rng.uniform(1.0, 10.0, size=(30, 4)), columns=['a', 'b', 'c', 'd'], index=index)
    y = pd.Series(X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(0.0, 0.1, 30), index=index, name='t')
    return X, y


# Étapes intermédiaires à comparer : (fabrique des étapes, X avec NaN ?)
STEPS = {
    'standard_scaler': (lambda: [('t', StandardScaler())], False),
    'minmax_scaler': (lambda: [('t', MinMaxScaler())], False),
    'robust_scaler': (lambda: [('t', RobustScaler())], False),
    'polynomial': (lambda: [('t', PolynomialFeatures(degree=2))], False),
    'pca': (lambda: [('t', PCA(n_components=2, svd_solver='full'))], False),
    'imputer': (lambda: [('t', SimpleImputer(strategy='mean'))], True),
    'select_k_best_uses_y': (lambda: [('t', SelectKBest(f_regression, k=2))], False),
    'function_log1p': (lambda: [('t', FunctionTransformer(np.log1p, inverse_func=np.expm1))], False),
    'chain_impute_scale': (lambda: [('i', SimpleImputer()), ('s', StandardScaler())], True),
    'chain_with_passthrough': (lambda: [('s', StandardScaler()), ('p', 'passthrough'), ('m', MinMaxScaler())], False),
}

# Étapes inversibles (inverse_transform défini partout dans la chaîne)
INVERTIBLE = ['standard_scaler', 'minmax_scaler', 'robust_scaler', 'pca', 'function_log1p', 'chain_with_passthrough']


def _inputs(data, with_nan: bool):
    """``(X, y)`` of ``data``, with one missing feature value when ``with_nan``."""
    X, y = data
    if with_nan:
        X = X.copy()
        X.iloc[3, 2] = np.nan
    return X, y


def _pair(make: Callable[[], List], final=None, **xy_kwargs):
    """Two pipelines built from fresh, identical steps: ``(Pipeline, XYPipeline)``."""
    tail = [] if final is None else [('model', final())]
    return Pipeline(make() + tail), XYPipeline(make() + tail, **xy_kwargs)


# ==============================================================================
# Résultats identiques
# ==============================================================================


class TestSameResults:
    """Same steps, same data: same numbers, bit for bit."""

    @pytest.mark.parametrize(('make', 'with_nan'), STEPS.values(), ids=STEPS.keys())
    def test_predict(self, make, with_nan, data):
        """Predictions of a ``LinearRegression`` behind the steps."""
        X, y = _inputs(data, with_nan)
        sk, xy = _pair(make, LinearRegression)

        np.testing.assert_array_equal(xy.fit(X, y).predict(X), sk.fit(X, y).predict(X))

    @pytest.mark.parametrize(('make', 'with_nan'), STEPS.values(), ids=STEPS.keys())
    def test_score(self, make, with_nan, data):
        """R² of a ``Ridge`` behind the steps."""
        X, y = _inputs(data, with_nan)
        sk, xy = _pair(make, Ridge)

        assert xy.fit(X, y).score(X, y) == sk.fit(X, y).score(X, y)

    @pytest.mark.parametrize(('make', 'with_nan'), STEPS.values(), ids=STEPS.keys())
    def test_final_coefficients(self, make, with_nan, data):
        """The final estimator learns the same coefficients."""
        X, y = _inputs(data, with_nan)
        sk, xy = _pair(make, LinearRegression)

        np.testing.assert_array_equal(xy.fit(X, y)[-1].coef_, sk.fit(X, y)[-1].coef_)

    @pytest.mark.parametrize(('make', 'with_nan'), STEPS.values(), ids=STEPS.keys())
    def test_fit_transform(self, make, with_nan, data):
        """``fit_transform`` without final estimator; ``preserve_dataframe=False`` gives arrays."""
        X, y = _inputs(data, with_nan)
        sk, xy = _pair(make, preserve_dataframe=False)

        np.testing.assert_array_equal(xy.fit_transform(X, y), sk.fit_transform(X, y))

    @pytest.mark.parametrize(('make', 'with_nan'), STEPS.values(), ids=STEPS.keys())
    def test_transform(self, make, with_nan, data):
        """``transform`` of the training data after ``fit``."""
        X, y = _inputs(data, with_nan)
        sk, xy = _pair(make, preserve_dataframe=False)

        np.testing.assert_array_equal(xy.fit(X, y).transform(X), sk.fit(X, y).transform(X))

    @pytest.mark.parametrize('name', INVERTIBLE)
    def test_inverse_transform(self, name, data):
        """``inverse_transform`` of the transformed data."""
        make, with_nan = STEPS[name]
        X, y = _inputs(data, with_nan)
        sk, xy = _pair(make, preserve_dataframe=False)
        X_t = sk.fit(X, y).transform(X)

        np.testing.assert_array_equal(xy.fit(X, y).inverse_transform(X_t), sk.inverse_transform(X_t))

    @pytest.mark.parametrize(('make', 'with_nan'), STEPS.values(), ids=STEPS.keys())
    def test_array_input_gives_the_same_type(self, make, with_nan, data):
        """With array inputs, both pipelines return arrays, whatever ``preserve_dataframe``."""
        X, y = _inputs(data, with_nan)
        sk, xy = _pair(make)

        assert type(xy.fit_transform(X.to_numpy(), y.to_numpy())) is type(sk.fit_transform(X.to_numpy(), y.to_numpy()))


# ==============================================================================
# Appels reçus par les étapes
# ==============================================================================

# Scénarios d'appel, sur deux pipelines construits de la même façon
SCENARIOS = {
    'fit': lambda pipe, X, y: pipe.fit(X, y),
    'fit_then_predict': lambda pipe, X, y: pipe.fit(X, y).predict(X),
    'fit_then_score': lambda pipe, X, y: pipe.fit(X, y).score(X, y),
}


def _journal(spy: SpyTransformer) -> List:
    """Comparable form of a spy journal: method, row count and target values."""
    return [
        (call.method, call.X_shape, None if call.y is None else np.asarray(call.y).tolist())
        for call in spy.calls_
    ]


class TestSameCalls:
    """Each step receives the same calls, with the same data, under both pipelines."""

    @pytest.mark.parametrize('scenario', SCENARIOS.values(), ids=SCENARIOS.keys())
    def test_same_journal_with_estimator(self, scenario, data):
        """``SpyTransformer -> StandardScaler -> SpyTransformer -> LinearRegression``."""
        X, y = data

        def make():
            return [('spy1', SpyTransformer()), ('scale', StandardScaler()), ('spy2', SpyTransformer())]

        sk, xy = _pair(make, LinearRegression)
        scenario(sk, X, y)
        scenario(xy, X, y)

        assert [_journal(xy[name]) for name in ('spy1', 'spy2')] == [_journal(sk[name]) for name in ('spy1', 'spy2')]

    def test_same_journal_at_transform(self, data):
        """``fit`` then ``transform`` without final estimator."""
        X, y = data
        sk, xy = _pair(lambda: [('spy', SpyTransformer()), ('scale', StandardScaler())])

        sk.fit(X, y).transform(X)
        xy.fit(X, y).transform(X)

        assert _journal(xy['spy']) == _journal(sk['spy'])

    def test_steps_are_fitted_in_place_without_memory(self, data):
        """Like ``Pipeline`` with ``memory=None``, the given step objects are the fitted ones."""
        X, y = data
        scaler = StandardScaler()

        pipeline = XYPipeline([('scale', scaler), ('model', LinearRegression())]).fit(X, y)

        assert pipeline.named_steps['scale'] is scaler


# ==============================================================================
# Écart voulu : sortie DataFrame
# ==============================================================================


class TestPreservedDataFrame:
    """With ``preserve_dataframe=True`` (default) and pandas input, the output stays a ``DataFrame``."""

    def test_values_are_those_of_pipeline(self, data):
        """Same values as the ``Pipeline`` array."""
        X, y = data
        sk, xy = _pair(lambda: [('t', StandardScaler())])

        np.testing.assert_array_equal(xy.fit_transform(X, y).to_numpy(), sk.fit_transform(X, y))

    def test_output_is_a_dataframe(self, data):
        """Where ``Pipeline`` gives an array, ``XYPipeline`` gives a ``DataFrame``.

        Deliberate difference, controlled by ``preserve_dataframe``; with
        ``preserve_dataframe=False`` the two pipelines return the same type.
        """
        X, y = data
        _sk, xy = _pair(lambda: [('t', StandardScaler())])

        assert isinstance(xy.fit_transform(X, y), pd.DataFrame)

    def test_dataframe_keeps_the_input_labels(self, data):
        """The ``DataFrame`` has the columns and the dates of the training ``X``."""
        X, y = data
        _sk, xy = _pair(lambda: [('t', StandardScaler())])

        X_t = xy.fit_transform(X, y)

        assert (list(X_t.columns), X_t.index.equals(X.index)) == (['a', 'b', 'c', 'd'], True)
