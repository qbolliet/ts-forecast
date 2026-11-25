"""Methods and attributes ``XYPipeline`` delegates to its steps, compared with ``Pipeline``.

Covers the conditional availability of the methods (``available_if``:
``predict``, ``predict_proba``, ``predict_log_proba``, ``decision_function``,
``score_samples``, ``transform``, ``inverse_transform``, ``fit_transform``,
``fit_predict``, ``score``) for several final estimators, the
``NotFittedError`` raised before ``fit``, the values of the classifier methods,
``score_samples`` and ``fit_predict`` (equal to ``Pipeline`` without XY step,
equal to an estimator fitted on the observed rows behind ``DropNaNTarget``), and
the fitted attributes ``classes_``, ``n_features_in_``, ``feature_names_in_``
and ``get_feature_names_out``.
"""
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from tsforecast.xy import DropNaNTarget, XYPipeline
from tests.support.estimators import ScaleXY, SpyTransformer


# Estimateurs finals aux surfaces différentes (classifieur, régresseur, transformateur,
# partitionnement, détection d'anomalies)
FINALS = {
    'logistic': lambda: LogisticRegression(),
    'linear': lambda: LinearRegression(),
    'scaler': lambda: StandardScaler(),
    'kmeans': lambda: KMeans(n_clusters=2, n_init=1, random_state=0),
    'isolation_forest': lambda: IsolationForest(n_estimators=10, random_state=0),
    'passthrough': lambda: 'passthrough',
}

METHODS = [
    'predict', 'predict_proba', 'predict_log_proba', 'decision_function', 'score_samples',
    'transform', 'inverse_transform', 'fit_transform', 'fit_predict', 'score',
]


def _binary(y: pd.Series) -> pd.Series:
    """Labels ``'down'`` / ``'up'`` as floats 0 / 1."""
    return (y == 'up').astype(float)


# ==============================================================================
# Disponibilité conditionnelle
# ==============================================================================


class TestAvailability:
    """A method exists on the pipeline exactly when it exists on ``Pipeline`` built from the same steps."""

    @pytest.mark.parametrize('method', METHODS)
    @pytest.mark.parametrize('final', list(FINALS), ids=list(FINALS))
    def test_same_methods_as_pipeline(self, final, method):
        """``hasattr(XYPipeline, m) == hasattr(Pipeline, m)`` for every final estimator."""
        xy = XYPipeline([('scale', StandardScaler()), ('final', FINALS[final]())])
        sk = Pipeline([('scale', StandardScaler()), ('final', FINALS[final]())])

        assert hasattr(xy, method) == hasattr(sk, method)

    def test_regressor_has_no_predict_proba(self):
        """A regression pipeline has no ``predict_proba``."""
        assert not hasattr(XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())]), 'predict_proba')

    def test_missing_method_raises_attribute_error(self):
        """Calling a missing method raises ``AttributeError`` naming the method."""
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())])

        with pytest.raises(AttributeError, match="has no attribute 'decision_function'"):
            pipeline.decision_function

    def test_xy_step_in_front_keeps_the_classifier_methods(self):
        """An XY step before the classifier leaves ``predict_proba`` available."""
        assert hasattr(XYPipeline([('drop', DropNaNTarget()), ('model', LogisticRegression())]), 'predict_proba')

    def test_availability_follows_set_params(self):
        """Replacing the final estimator by ``set_params`` changes the available methods."""
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())])

        pipeline.set_params(model=LogisticRegression())

        assert hasattr(pipeline, 'predict_proba')


# ==============================================================================
# NotFittedError avant fit
# ==============================================================================


class TestNotFitted:
    """Before ``fit``, every method reading fitted steps raises ``NotFittedError`` before running any step."""

    @pytest.mark.parametrize(
        'final, method, with_y',
        [
            ('logistic', 'predict', False),
            ('logistic', 'predict_proba', False),
            ('logistic', 'predict_log_proba', False),
            ('logistic', 'decision_function', False),
            ('logistic', 'score', True),
            ('isolation_forest', 'score_samples', False),
            ('scaler', 'transform', False),
            ('scaler', 'inverse_transform', False),
        ],
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_raises_not_fitted_error(self, final, method, with_y, classification_xy):
        """The error names the pipeline: ``This XYPipeline instance is not fitted yet``."""
        X, y = classification_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('final', FINALS[final]())])
        args = (X, y) if with_y else (X,)

        with pytest.raises(NotFittedError, match='This XYPipeline instance is not fitted yet'):
            getattr(pipeline, method)(*args)

    def test_no_step_runs_before_the_error(self, classification_xy):
        """The check comes first: the first step receives no call."""
        X, _y = classification_xy
        spy = SpyTransformer()

        with pytest.raises(NotFittedError):
            XYPipeline([('spy', spy), ('model', LogisticRegression())]).predict(X)

        assert not hasattr(spy, 'calls_')

    @pytest.mark.parametrize('attribute', ['classes_', 'n_features_in_', 'feature_names_in_'])
    def test_fitted_attributes_are_missing(self, attribute):
        """The delegated fitted attributes do not exist before ``fit``."""
        assert not hasattr(XYPipeline([('scale', StandardScaler()), ('model', LogisticRegression())]), attribute)


# ==============================================================================
# Méthodes de classification
# ==============================================================================


CLASSIFIER_METHODS = ['predict', 'predict_proba', 'predict_log_proba', 'decision_function']


class TestClassifierMethods:
    """Classifier methods: values of ``Pipeline`` without XY step, of the observed rows behind ``DropNaNTarget``."""

    @pytest.mark.parametrize('method', CLASSIFIER_METHODS)
    def test_values_equal_pipeline(self, method, classification_xy):
        """Same output as ``Pipeline([scale, logistic])``."""
        X, y = classification_xy
        xy = XYPipeline([('scale', StandardScaler()), ('model', LogisticRegression())]).fit(X, y)
        sk = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression())]).fit(X, y)

        expected = getattr(sk, method)(X)
        actual = getattr(xy, method)(X)
        if method == 'predict':
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected)

    def test_probabilities_sum_to_one(self, classification_xy):
        """Each row of ``predict_proba`` sums to 1."""
        X, y = classification_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LogisticRegression())]).fit(X, y)

        np.testing.assert_allclose(pipeline.predict_proba(X).sum(axis=1), 1.0)

    def test_classes_are_those_of_the_final_estimator(self, classification_xy):
        """``classes_`` lists the sorted labels, as the final estimator does."""
        X, y = classification_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LogisticRegression())]).fit(X, y)

        np.testing.assert_array_equal(pipeline.classes_, ['down', 'up'])

    @pytest.mark.parametrize('method', CLASSIFIER_METHODS)
    def test_behind_drop_values_of_the_observed_rows(self, method, classification_xy):
        """Behind ``DropNaNTarget``: outputs of a classifier fitted on the labelled rows, for every row."""
        X, y = classification_xy
        y = _binary(y)
        y.iloc[[0, 5, 17]] = np.nan
        labelled = y.notna()
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', LogisticRegression())]).fit(X, y)

        # Référence indépendante : classifieur ajusté sur les seules lignes étiquetées
        reference = LogisticRegression().fit(X[labelled], y[labelled])
        np.testing.assert_allclose(getattr(pipeline, method)(X), getattr(reference, method)(X))

    def test_behind_drop_missing_label_is_not_a_class(self, classification_xy):
        """The missing label is not learnt as a class: ``classes_`` is ``[0, 1]``."""
        X, y = classification_xy
        y = _binary(y)
        y.iloc[[0, 5, 17]] = np.nan

        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', LogisticRegression())]).fit(X, y)

        np.testing.assert_array_equal(pipeline.classes_, [0.0, 1.0])


# ==============================================================================
# score_samples, fit_predict
# ==============================================================================


class TestUnsupervisedMethods:
    """``score_samples`` and ``fit_predict`` of unsupervised final estimators."""

    def test_score_samples_equal_pipeline(self, classification_xy):
        """``score_samples`` of an isolation forest equals the one of ``Pipeline``."""
        X, _y = classification_xy
        xy = XYPipeline([('scale', StandardScaler()), ('model', FINALS['isolation_forest']())]).fit(X)
        sk = Pipeline([('scale', StandardScaler()), ('model', FINALS['isolation_forest']())]).fit(X)

        np.testing.assert_allclose(xy.score_samples(X), sk.score_samples(X))

    def test_score_samples_behind_drop_of_the_observed_rows(self, classification_xy):
        """Behind ``DropNaNTarget``, the forest is grown on the labelled rows and scores every row."""
        X, y = classification_xy
        y = _binary(y)
        y.iloc[[2, 3]] = np.nan
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', FINALS['isolation_forest']())]).fit(X, y)

        # Référence indépendante : forêt ajustée sur les lignes étiquetées
        reference = FINALS['isolation_forest']().fit(X[y.notna()])
        np.testing.assert_allclose(pipeline.score_samples(X), reference.score_samples(X))

    def test_fit_predict_equal_pipeline(self, classification_xy):
        """``fit_predict`` of an isolation forest gives the inlier / outlier labels of ``Pipeline``."""
        X, _y = classification_xy
        xy = XYPipeline([('scale', StandardScaler()), ('model', FINALS['isolation_forest']())])
        sk = Pipeline([('scale', StandardScaler()), ('model', FINALS['isolation_forest']())])

        np.testing.assert_array_equal(xy.fit_predict(X), sk.fit_predict(X))


# ==============================================================================
# Attributs ajustés délégués
# ==============================================================================


def _fit_both(make: Callable, X, y):
    """``(XYPipeline, Pipeline)`` built from ``make()`` and fitted on ``(X, y)``."""
    return XYPipeline(make()).fit(X, y), Pipeline(make()).fit(X, y)


class TestFittedAttributes:
    """``n_features_in_`` and ``feature_names_in_`` come from the first step, as in ``Pipeline``."""

    @staticmethod
    def _steps():
        return [('scale', StandardScaler()), ('model', Ridge())]

    def test_n_features_in(self, linear_xy):
        """``n_features_in_`` is 2, as in ``Pipeline``."""
        xy, sk = _fit_both(self._steps, *linear_xy)

        assert xy.n_features_in_ == sk.n_features_in_ == 2

    def test_feature_names_in(self, linear_xy):
        """``feature_names_in_`` lists the columns of ``X``, as in ``Pipeline``."""
        xy, sk = _fit_both(self._steps, *linear_xy)

        np.testing.assert_array_equal(xy.feature_names_in_, sk.feature_names_in_)

    def test_feature_names_in_golden(self, linear_xy):
        """``feature_names_in_`` is ``['a', 'b']``."""
        xy, _sk = _fit_both(self._steps, *linear_xy)

        assert list(xy.feature_names_in_) == ['a', 'b']

    def test_array_input_has_no_feature_names(self, linear_xy):
        """Fitted on arrays, the pipeline has no ``feature_names_in_``."""
        X, y = linear_xy

        assert not hasattr(XYPipeline(self._steps()).fit(X.to_numpy(), y.to_numpy()), 'feature_names_in_')

    def test_first_step_without_the_attribute(self, linear_xy):
        """A first step without ``n_features_in_`` (``ScaleXY``) leaves the pipeline without it."""
        assert not hasattr(XYPipeline([('xy', ScaleXY()), ('model', Ridge())]).fit(*linear_xy), 'n_features_in_')

    def test_first_step_drop_exposes_the_feature_names(self, frame_xy):
        """A pipeline starting with ``DropNaNTarget`` exposes the columns it was fitted on."""
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', Ridge())]).fit(*frame_xy)

        assert list(pipeline.feature_names_in_) == ['a', 'b']


class TestFeatureNamesOut:
    """``get_feature_names_out`` chains the names given by each step, as in ``Pipeline``."""

    @staticmethod
    def _steps():
        return [('scale', StandardScaler()), ('poly', PolynomialFeatures(degree=2)), ('model', Ridge())]

    def test_slice_gives_the_names_of_pipeline(self, linear_xy):
        """``pipe[:-1].get_feature_names_out()`` equals the one of ``Pipeline``."""
        xy, sk = _fit_both(self._steps, *linear_xy)

        np.testing.assert_array_equal(xy[:-1].get_feature_names_out(), sk[:-1].get_feature_names_out())

    def test_slice_names_golden(self, linear_xy):
        """The degree-2 features of ``a`` and ``b`` are named by ``PolynomialFeatures``."""
        xy, _sk = _fit_both(self._steps, *linear_xy)

        assert list(xy[:-1].get_feature_names_out()) == ['1', 'a', 'b', 'a^2', 'a b', 'b^2']

    def test_input_features_are_used_for_arrays(self, linear_xy):
        """Fitted on arrays, the names derive from ``input_features``."""
        X, y = linear_xy
        xy = XYPipeline(self._steps()).fit(X.to_numpy(), y.to_numpy())

        assert list(xy[:-1].get_feature_names_out(['u', 'v'])) == ['1', 'u', 'v', 'u^2', 'u v', 'v^2']

    def test_final_regressor_has_no_names(self, linear_xy):
        """On the whole pipeline, the regressor stops the chain with the message of ``Pipeline``."""
        xy, _sk = _fit_both(self._steps, *linear_xy)

        with pytest.raises(AttributeError, match='Estimator model does not provide get_feature_names_out'):
            xy.get_feature_names_out()

    def test_names_go_through_drop(self, frame_xy):
        """``pipe[:-1]`` of ``[scale, drop, model]`` names the features ``a`` and ``b``."""
        pipeline = XYPipeline([('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', Ridge())])
        pipeline.fit(*frame_xy)

        assert list(pipeline[:-1].get_feature_names_out()) == ['a', 'b']

    def test_new_features_keep_their_names_through_drop(self, frame_xy):
        """``fit_transform`` of ``[poly, drop]`` names the new columns as ``PolynomialFeatures`` does."""
        X_t, _y_t = XYPipeline([('poly', PolynomialFeatures(degree=2)), ('drop', DropNaNTarget())]).fit_transform(
            *frame_xy
        )

        assert list(X_t.columns) == ['1', 'a', 'b', 'a^2', 'a b', 'b^2']
