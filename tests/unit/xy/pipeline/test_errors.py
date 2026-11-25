"""Invalid ``XYPipeline`` definitions and the recognition of XY steps.

Covers the errors raised at fit for invalid steps (duplicate names, names with
``__``, names conflicting with a constructor argument, ``preserve_dataframe``
included, intermediate step that is not a transformer, step without ``fit``,
last step without ``fit``, steps that are not ``(name, estimator)`` pairs):
messages checked and identical to those of ``Pipeline``, raised at fit and not
at construction. Also covers how a step is recognised as an XY step (it
inherits ``XYTransformerMixin`` or ``XYPanelTimeSeriesTransformer``): a declared
step transforms the target and must return a pair when given one; a pair
returned by an ordinary step goes on as its features, as in ``Pipeline``; an
undeclared step with an XY-like signature never receives the target.
"""
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from tsforecast.xy import XYPipeline, XYTransformerMixin
from tests.support.estimators import SpyEstimator


class TransformOnly:
    """Object with a ``transform`` but no ``fit``."""

    def __repr__(self) -> str:
        return 'TransformOnly()'

    def transform(self, X):
        """Return ``X``."""
        return X


# Définitions invalides : (fabrique des étapes, classe d'exception, motif du message)
INVALID: Dict[str, Tuple[Callable, type, str]] = {
    'duplicate_names': (
        lambda: [('a', StandardScaler()), ('a', Ridge())],
        ValueError,
        r"Names provided are not unique: \['a', 'a'\]",
    ),
    'double_underscore': (
        lambda: [('a__b', StandardScaler()), ('model', Ridge())],
        ValueError,
        r"Estimator names must not contain __: got \['a__b'\]",
    ),
    'name_is_a_constructor_argument': (
        lambda: [('memory', StandardScaler()), ('model', Ridge())],
        ValueError,
        r"Estimator names conflict with constructor arguments: \['memory'\]",
    ),
    'intermediate_estimator': (
        lambda: [('first', Ridge()), ('model', Ridge())],
        TypeError,
        r"All intermediate steps should be transformers and implement fit and transform or be the string "
        r"'passthrough' 'Ridge\(\)'",
    ),
    'intermediate_without_fit': (
        lambda: [('first', TransformOnly()), ('model', Ridge())],
        TypeError,
        r"All intermediate steps should be transformers .* 'TransformOnly\(\)'",
    ),
    'last_without_fit': (
        lambda: [('scale', StandardScaler()), ('last', TransformOnly())],
        TypeError,
        r"Last step of Pipeline should implement fit or be the string 'passthrough'. 'TransformOnly\(\)'",
    ),
    'not_pairs': (
        lambda: [StandardScaler(), Ridge()],
        TypeError,
        r'cannot unpack non-iterable StandardScaler object',
    ),
}


# ==============================================================================
# Définitions invalides
# ==============================================================================


class TestInvalidSteps:
    """Invalid steps are refused at fit with the messages of ``Pipeline``."""

    @pytest.mark.parametrize('case', list(INVALID), ids=list(INVALID))
    def test_message(self, case, linear_xy):
        """The error class and message name the faulty step."""
        make, error, message = INVALID[case]

        with pytest.raises(error, match=message):
            XYPipeline(make()).fit(*linear_xy)

    @pytest.mark.parametrize('case', list(INVALID), ids=list(INVALID))
    def test_same_error_as_pipeline(self, case, linear_xy):
        """``Pipeline`` built on the same step objects raises the same class with the same message."""
        steps = INVALID[case][0]()
        with pytest.raises(Exception) as sk_error:
            Pipeline(steps).fit(*linear_xy)

        with pytest.raises(Exception) as xy_error:
            XYPipeline(steps).fit(*linear_xy)

        assert (type(xy_error.value), str(xy_error.value)) == (type(sk_error.value), str(sk_error.value))

    def test_preserve_dataframe_is_a_reserved_name(self, linear_xy):
        """``preserve_dataframe``, a constructor argument of ``XYPipeline`` only, cannot name a step."""
        pipeline = XYPipeline([('preserve_dataframe', StandardScaler()), ('model', Ridge())])

        with pytest.raises(ValueError, match=r"Estimator names conflict with constructor arguments: \['preserve_dataframe'\]"):
            pipeline.fit(*linear_xy)

    @pytest.mark.parametrize('case', list(INVALID), ids=list(INVALID))
    def test_construction_does_not_validate(self, case):
        """As in sklearn, the steps are only checked at fit: the constructor accepts them."""
        assert isinstance(XYPipeline(INVALID[case][0]()), XYPipeline)

    @pytest.mark.parametrize(
        'method, final',
        [('fit_transform', MinMaxScaler), ('fit_predict', lambda: KMeans(n_clusters=2, n_init=1))],
        ids=['fit_transform', 'fit_predict'],
    )
    def test_every_fitting_method_validates(self, method, final, linear_xy):
        """``fit_transform`` and ``fit_predict`` check the names too."""
        pipeline = XYPipeline([('a', StandardScaler()), ('a', final())])

        with pytest.raises(ValueError, match='Names provided are not unique'):
            getattr(pipeline, method)(*linear_xy)


# ==============================================================================
# Reconnaissance des étapes XY
# ==============================================================================


class PairOutput(BaseEstimator, TransformerMixin):
    """Ordinary transformer (``transform(X)``) whose output is the pair ``(X, zeros)``."""

    def fit(self, X, y=None):
        """Learn nothing."""
        self.fitted_ = True
        return self

    def transform(self, X):
        """Return ``X`` and a column of zeros."""
        return X, np.zeros(len(X))


def _first_item(pair):
    """First item of a pair."""
    return pair[0]


class UndeclaredXY(BaseEstimator, TransformerMixin):
    """Transformer with an XY-like signature that does not inherit an XY base class.

    ``transform(X, y)`` would return ``(X, y + 100)``; it records the target of
    each call to ``transform``.
    """

    def fit(self, X, y=None):
        """Learn nothing."""
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        """Record ``y``; return ``X``, or ``(X, y + 100)`` when a target is given."""
        self.transform_targets_ = getattr(self, 'transform_targets_', []) + [y]
        return X if y is None else (X, y + 100.0)


class DeclaredXY(BaseEstimator, XYTransformerMixin):
    """XY step declared by ``XYTransformerMixin``: ``(X, y) -> (X, y + 100)``."""

    def _fit(self, X, y):
        return self

    def _transform(self, X, y=None):
        return X if y is None else (X, y + 100.0)


class XOnlyXY(BaseEstimator, XYTransformerMixin):
    """Declared XY step that wrongly returns ``X`` alone when given a target."""

    def _fit(self, X, y):
        return self

    def _transform(self, X, y=None):
        return X


class TestStepRecognition:
    """A step is XY when it inherits an XY base class; every other step is handled as in ``Pipeline``."""

    def test_declared_step_transforms_the_target(self, linear_xy):
        """The estimator behind a declared XY step receives ``y + 100``."""
        X, y = linear_xy

        pipeline = XYPipeline([('xy', DeclaredXY()), ('model', SpyEstimator())]).fit(X, y)

        assert pipeline['model'].fit_y_.tolist() == (y + 100.0).tolist()

    def test_pair_output_of_an_ordinary_step_is_its_features(self, linear_xy):
        """The pair returned by an ordinary step goes on as ``X``: the target is untouched."""
        X, y = linear_xy
        steps = [('pair', PairOutput()), ('first', FunctionTransformer(_first_item)), ('model', SpyEstimator())]

        pipeline = XYPipeline(steps).fit(X, y)

        assert pipeline['model'].fit_y_.tolist() == y.tolist()

    def test_pair_output_of_an_ordinary_step_like_pipeline(self, linear_xy):
        """Behind an ordinary step returning a pair, the estimator sees what it sees under ``Pipeline``."""
        X, y = linear_xy
        steps = lambda: [('pair', PairOutput()), ('first', FunctionTransformer(_first_item)), ('model', SpyEstimator())]

        xy = XYPipeline(steps()).fit(X, y)
        sk = Pipeline(steps()).fit(X, y)

        assert (xy['model'].fit_X_.equals(sk['model'].fit_X_), xy['model'].fit_y_.equals(sk['model'].fit_y_)) == (
            True,
            True,
        )

    def test_undeclared_step_is_never_given_the_target(self, linear_xy):
        """``transform(X, y)`` of the pipeline gives an undeclared step ``X`` only."""
        X, y = linear_xy
        pipeline = XYPipeline([('undeclared', UndeclaredXY())]).fit(X, y)

        pipeline.transform(X, y)

        assert pipeline['undeclared'].transform_targets_ == [None]

    def test_undeclared_step_returns_x_alone(self, linear_xy):
        """``transform(X, y)`` without declared XY step returns ``X`` alone."""
        X, y = linear_xy
        pipeline = XYPipeline([('undeclared', UndeclaredXY())]).fit(X)

        assert isinstance(pipeline.transform(X, y), pd.DataFrame)

    @pytest.mark.parametrize('method', ['fit', 'transform', 'score'])
    def test_declared_step_must_return_a_pair(self, method, linear_xy):
        """A declared XY step given a target and returning ``X`` alone is refused with an explicit error."""
        X, y = linear_xy
        pipeline = XYPipeline([('bad', XOnlyXY()), ('density', KernelDensity())])
        if method != 'fit':
            pipeline.fit(X)
        target = pipeline[:-1] if method == 'transform' else pipeline

        with pytest.raises(
            TypeError, match=r"XY step 'bad' must return the pair \(X, y\) when it is given a target, got DataFrame"
        ):
            getattr(target, method)(X, y)

    def test_declared_step_without_target_returns_x(self, linear_xy):
        """Without target, a declared XY step returns ``X`` alone, which the pipeline accepts."""
        X, _y = linear_xy
        pipeline = XYPipeline([('bad', XOnlyXY()), ('density', KernelDensity())]).fit(X)

        assert pipeline.score_samples(X).shape == (6,)


class FitAndTransformOnly(BaseEstimator):
    """Ordinary step with ``fit`` and ``transform`` but no ``fit_transform``: adds 1 to ``X``."""

    def fit(self, X, y=None):
        """Learn nothing."""
        self.fitted_ = True
        return self

    def transform(self, X):
        """Return ``X + 1``."""
        return X + 1.0


class TestOrdinaryStepWithoutFitTransform:
    """An ordinary step without ``fit_transform`` is fitted, then applied, as in ``Pipeline``."""

    def test_intermediate_step(self, linear_xy):
        """In the middle, the estimator receives ``X + 1``, as under ``Pipeline``."""
        X, y = linear_xy
        steps = lambda: [('plus_one', FitAndTransformOnly()), ('model', SpyEstimator())]

        xy = XYPipeline(steps()).fit(X, y)
        sk = Pipeline(steps()).fit(X, y)

        assert xy['model'].fit_X_.equals(sk['model'].fit_X_)

    def test_last_step(self, linear_xy):
        """Last, ``fit_transform`` returns ``X + 1``."""
        X, y = linear_xy

        X_t = XYPipeline([('scale', 'passthrough'), ('plus_one', FitAndTransformOnly())]).fit_transform(X, y)

        assert X_t.equals(X + 1.0)
