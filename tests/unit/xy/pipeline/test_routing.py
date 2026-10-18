"""Fit parameters and metadata routing of ``XYPipeline``, compared with ``Pipeline``.

Covers the ``step__param`` fit parameters without routing (to an ordinary
step, an XY step, the final estimator; ``fit``, ``fit_transform``,
``fit_predict``; parameters without step name), the parameters of the
prediction and scoring methods without routing, metadata routing
(``set_fit_request`` towards a sklearn step, an XY step or the final estimator,
aliases, unset requests, prediction / score / transform parameters,
``cross_validate(params=...)``), ``transform_input``, and the row-aligned
parameters (sample weights) that follow the rows an XY step keeps. The ``routing`` fixture enables
routing for one test and restores the configuration afterwards.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tsforecast.crossvals import TSOutOfSampleSplit
from tsforecast.xy import DropNaNTarget, XYPipeline, XYTransformerMixin

from .conftest import month_ends


# ==============================================================================
# Doublures et données
# ==============================================================================


class WeightedXY(BaseEstimator, XYTransformerMixin):
    """XY step that records the weights given to its fit and returns ``(X, y)`` unchanged."""

    def fit(self, X, y=None, sample_weight=None):
        """Record ``sample_weight`` (None when not given)."""
        self.weights_ = None if sample_weight is None else np.asarray(sample_weight)
        return self

    def transform(self, X, y=None):
        """Return ``X``, or ``(X, y)`` when a target is given."""
        return X if y is None else (X, y)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit with the weights, then transform ``X`` and ``y`` together."""
        return self.fit(X, y, sample_weight=sample_weight).transform(X, y)


class WeightedEstimator(BaseEstimator, RegressorMixin):
    """Regressor that records its fit weights and row count, and predicts ``factor`` everywhere."""

    def fit(self, X, y, sample_weight=None):
        """Record ``sample_weight`` and the number of training rows."""
        self.weights_ = None if sample_weight is None else np.asarray(sample_weight)
        self.n_rows_ = len(y)
        return self

    def predict(self, X, factor=1.0):
        """Return ``factor`` for every row of ``X``."""
        return np.full(len(X), factor)


class ValidationSetEstimator(BaseEstimator, RegressorMixin):
    """Regressor that records the validation set ``X_val`` given to its fit."""

    def fit(self, X, y, X_val=None):
        """Record ``X_val`` as an array (None when not given)."""
        self.X_val_ = None if X_val is None else np.asarray(X_val)
        return self

    def predict(self, X):
        """Return zeros."""
        return np.zeros(len(X))


@pytest.fixture
def noisy_xy():
    """``(X, y, w)``: 20 month ends, 2 uniform features, a noisy linear target, positive weights."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.uniform(0.0, 10.0, size=(20, 2)), columns=['a', 'b'], index=month_ends(20))
    y = pd.Series(X['a'] - 2.0 * X['b'] + rng.normal(0.0, 1.0, 20), name='t')
    w = rng.uniform(0.5, 2.0, 20)
    return X, y, w


# ==============================================================================
# Paramètres d'ajustement par étape, sans routage
# ==============================================================================


class TestStepParams:
    """``fit(X, y, step__param=value)`` gives ``param`` to the fit of ``step`` only."""

    def test_param_reaches_an_ordinary_step(self, noisy_xy):
        """``scale__sample_weight`` gives the weighted means of ``StandardScaler``, as in ``Pipeline``."""
        X, y, w = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', LinearRegression())]
        xy = XYPipeline(steps()).fit(X, y, scale__sample_weight=w)
        sk = Pipeline(steps()).fit(X, y, scale__sample_weight=w)

        np.testing.assert_allclose(xy['scale'].mean_, sk['scale'].mean_)

    def test_weighted_means_golden(self, noisy_xy):
        """The learnt means are the weighted means of the columns."""
        X, y, w = noisy_xy

        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())])
        pipeline.fit(X, y, scale__sample_weight=w)

        # Valeur d'or : moyenne pondérée de chaque colonne
        np.testing.assert_allclose(pipeline['scale'].mean_, np.average(X.to_numpy(), axis=0, weights=w))

    def test_param_reaches_the_final_estimator(self, noisy_xy):
        """``model__sample_weight`` gives the coefficients of a weighted regression on the scaled features."""
        X, y, w = noisy_xy

        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())])
        pipeline.fit(X, y, model__sample_weight=w)

        # Référence indépendante : régression pondérée sur les colonnes centrées réduites
        reference = LinearRegression().fit(StandardScaler().fit_transform(X), y, sample_weight=w)
        np.testing.assert_allclose(pipeline['model'].coef_, reference.coef_)

    def test_param_reaches_an_xy_step(self, noisy_xy):
        """``xy__sample_weight`` reaches the fit of the XY step."""
        X, y, w = noisy_xy

        pipeline = XYPipeline([('xy', WeightedXY()), ('model', WeightedEstimator())])
        pipeline.fit(X, y, xy__sample_weight=w)

        np.testing.assert_array_equal(pipeline['xy'].weights_, w)

    def test_param_of_a_step_does_not_reach_the_others(self, noisy_xy):
        """The final estimator does not receive the weights of the XY step."""
        X, y, w = noisy_xy

        pipeline = XYPipeline([('xy', WeightedXY()), ('model', WeightedEstimator())])
        pipeline.fit(X, y, xy__sample_weight=w)

        assert pipeline['model'].weights_ is None

    def test_fit_transform_routes_step_params(self, noisy_xy):
        """``fit_transform(X, y, xy__sample_weight=w)`` reaches the XY step too."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('xy', WeightedXY())])

        pipeline.fit_transform(X, y, xy__sample_weight=w)

        np.testing.assert_array_equal(pipeline['xy'].weights_, w)

    def test_fit_predict_routes_step_params(self, noisy_xy):
        """``fit_predict(X, model__sample_weight=w)`` gives the labels of ``Pipeline`` with the same weights."""
        X, _y, w = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', IsolationForest(n_estimators=10, random_state=0))]

        labels = XYPipeline(steps()).fit_predict(X, model__sample_weight=w)

        np.testing.assert_array_equal(labels, Pipeline(steps()).fit_predict(X, model__sample_weight=w))

    def test_param_without_step_name_is_refused(self, noisy_xy):
        """A parameter without ``step__`` prefix is refused with the message of ``Pipeline``."""
        X, y, w = noisy_xy

        with pytest.raises(ValueError, match='Pipeline.fit does not accept the sample_weight parameter'):
            XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())]).fit(X, y, sample_weight=w)

    def test_unknown_step_name_is_refused(self, noisy_xy):
        """A prefix naming no step raises ``KeyError``, as in ``Pipeline``."""
        X, y, w = noisy_xy

        with pytest.raises(KeyError, match='nope'):
            XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())]).fit(X, y, nope__sample_weight=w)


# ==============================================================================
# Paramètres des méthodes de prédiction et de score, sans routage
# ==============================================================================


class TestPredictionParamsWithoutRouting:
    """Without routing, ``Pipeline`` gives the extra parameters of ``predict`` / ``score`` to the final estimator."""

    def test_predict_return_std(self, noisy_xy):
        """``predict(X, return_std=True)`` returns the means and deviations of ``Pipeline``."""
        X, y, _w = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', BayesianRidge())]
        expected = Pipeline(steps()).fit(X, y).predict(X, return_std=True)

        mean, std = XYPipeline(steps()).fit(X, y).predict(X, return_std=True)

        np.testing.assert_allclose(np.stack([mean, std]), np.stack(expected))

    def test_predict_param_reaches_the_final_estimator(self, noisy_xy):
        """``predict(X, factor=3.0)`` returns 3 for every row."""
        X, y, _w = noisy_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', WeightedEstimator())]).fit(X, y)

        np.testing.assert_array_equal(pipeline.predict(X, factor=3.0), np.full(len(X), 3.0))

    def test_score_sample_weight(self, noisy_xy):
        """``score(X, y, sample_weight=w)`` is the weighted R² of the predictions."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', Ridge(alpha=10.0))]).fit(X, y)

        # Valeur d'or : R² pondéré des prédictions
        assert pipeline.score(X, y, sample_weight=w) == pytest.approx(r2_score(y, pipeline.predict(X), sample_weight=w))

    @pytest.mark.parametrize('method', ['transform', 'decision_function'])
    def test_extra_params_ask_for_routing(self, method, classification_xy):
        """``transform`` / ``decision_function`` with extra parameters ask to enable routing, like ``Pipeline``."""
        X, y = classification_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('model', LogisticRegression())]).fit(X, y)
        target = pipeline[:-1] if method == 'transform' else pipeline

        with pytest.raises(ValueError, match='only supported if enable_metadata_routing=True'):
            getattr(target, method)(X, copy=True)

    def test_score_without_weights_is_unweighted(self, noisy_xy):
        """``score(X, y)`` without weights is the plain R², as in ``Pipeline``."""
        X, y, _w = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', Ridge(alpha=10.0))]

        assert XYPipeline(steps()).fit(X, y).score(X, y) == pytest.approx(Pipeline(steps()).fit(X, y).score(X, y))


# ==============================================================================
# Metadata routing
# ==============================================================================


@pytest.mark.usefixtures('routing')
class TestMetadataRouting:
    """With ``enable_metadata_routing=True``, each step receives the metadata it requests."""

    def test_weights_routed_to_the_final_estimator(self, noisy_xy):
        """Requested by the regression only, the weights give the coefficients of ``Pipeline``."""
        X, y, w = noisy_xy
        steps = lambda: [
            ('scale', StandardScaler().set_fit_request(sample_weight=False)),
            ('model', LinearRegression().set_fit_request(sample_weight=True)),
        ]
        xy = XYPipeline(steps()).fit(X, y, sample_weight=w)
        sk = Pipeline(steps()).fit(X, y, sample_weight=w)

        np.testing.assert_allclose(xy['model'].coef_, sk['model'].coef_)

    def test_weights_routed_to_the_final_estimator_golden(self, noisy_xy):
        """The coefficients are those of a weighted regression on the (unweighted) scaled features."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([
            ('scale', StandardScaler().set_fit_request(sample_weight=False)),
            ('model', LinearRegression().set_fit_request(sample_weight=True)),
        ])

        pipeline.fit(X, y, sample_weight=w)

        reference = LinearRegression().fit(StandardScaler().fit_transform(X), y, sample_weight=w)
        np.testing.assert_allclose(pipeline['model'].coef_, reference.coef_)

    def test_weights_routed_to_an_xy_step(self, noisy_xy):
        """Requested by the XY step, the weights reach its fit."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([
            ('xy', WeightedXY().set_fit_request(sample_weight=True)),
            ('model', WeightedEstimator().set_fit_request(sample_weight=False)),
        ])

        pipeline.fit(X, y, sample_weight=w)

        np.testing.assert_array_equal(pipeline['xy'].weights_, w)

    def test_declined_request_receives_nothing(self, noisy_xy):
        """A step that declines the weights (``set_fit_request(sample_weight=False)``) receives None."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([
            ('xy', WeightedXY().set_fit_request(sample_weight=True)),
            ('model', WeightedEstimator().set_fit_request(sample_weight=False)),
        ])

        pipeline.fit(X, y, sample_weight=w)

        assert pipeline['model'].weights_ is None

    @pytest.mark.parametrize('step, expected_factor', [('xy', 10.0), ('model', 1.0)], ids=['alias', 'own_name'])
    def test_alias(self, step, expected_factor, noisy_xy):
        """An aliased request reads ``xy_weight``; the other step still reads ``sample_weight``."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([
            ('xy', WeightedXY().set_fit_request(sample_weight='xy_weight')),
            ('model', WeightedEstimator().set_fit_request(sample_weight=True)),
        ])

        pipeline.fit(X, y, sample_weight=w, xy_weight=10.0 * w)

        np.testing.assert_array_equal(pipeline[step].weights_, expected_factor * w)

    def test_unset_request_is_refused(self, noisy_xy):
        """Metadata passed to a step that set no request raises ``UnsetMetadataPassedError``."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([('xy', WeightedXY()), ('model', WeightedEstimator())])

        with pytest.raises(
            UnsetMetadataPassedError,
            match=r'\[sample_weight\] are passed but are not explicitly set as requested or not requested '
            r'for WeightedXY.fit_transform, which is used within XYPipeline.fit',
        ):
            pipeline.fit(X, y, sample_weight=w)

    def test_unset_request_raises_like_pipeline(self, noisy_xy):
        """On ordinary steps, ``Pipeline`` raises the same error class."""
        X, y, w = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', LinearRegression())]

        with pytest.raises(UnsetMetadataPassedError):
            Pipeline(steps()).fit(X, y, sample_weight=w)
        with pytest.raises(UnsetMetadataPassedError):
            XYPipeline(steps()).fit(X, y, sample_weight=w)

    def test_predict_param_routed(self, noisy_xy):
        """``set_predict_request(factor=True)``: ``predict(X, factor=4.0)`` returns 4 for every row."""
        X, y, _w = noisy_xy
        pipeline = XYPipeline([
            ('scale', StandardScaler()),
            ('model', WeightedEstimator().set_predict_request(factor=True)),
        ]).fit(X, y)

        np.testing.assert_array_equal(pipeline.predict(X, factor=4.0), np.full(len(X), 4.0))

    def test_score_weights_routed(self, noisy_xy):
        """``set_score_request(sample_weight=True)``: the score is the weighted R² of the predictions."""
        X, y, w = noisy_xy
        pipeline = XYPipeline([
            ('scale', StandardScaler().set_fit_request(sample_weight=False)),
            ('model', Ridge(alpha=10.0).set_fit_request(sample_weight=False).set_score_request(sample_weight=True)),
        ]).fit(X, y)

        assert pipeline.score(X, y, sample_weight=w) == pytest.approx(r2_score(y, pipeline.predict(X), sample_weight=w))

    def test_transform_param_routed(self, noisy_xy):
        """``set_transform_request(copy=True)``: ``transform(X, copy=True)`` scales as ``Pipeline`` does."""
        X, y, _w = noisy_xy
        steps = lambda: [('scale', StandardScaler().set_transform_request(copy=True))]
        xy = XYPipeline(steps()).fit(X, y)
        sk = Pipeline(steps()).fit(X, y)

        np.testing.assert_allclose(xy.transform(X, copy=True).to_numpy(), sk.transform(X, copy=True))

    def test_routing_map_is_that_of_pipeline(self):
        """``get_metadata_routing`` describes the same mappings as ``Pipeline``."""
        steps = lambda: [
            ('scale', StandardScaler().set_fit_request(sample_weight=False)),
            ('model', LinearRegression().set_fit_request(sample_weight=True)),
        ]

        assert repr(XYPipeline(steps()).get_metadata_routing()) == repr(Pipeline(steps()).get_metadata_routing())

    def test_cross_validate_routes_the_weights(self, noisy_xy):
        """``cross_validate(params={'sample_weight': w})`` gives the fold scores of ``Pipeline``.

        The weights go to the fit of the regression and to a weighted R² scorer.
        """
        X, y, w = noisy_xy
        steps = lambda: [
            ('scale', StandardScaler().set_fit_request(sample_weight=False)),
            ('model', Ridge(alpha=1.0).set_fit_request(sample_weight=True)),
        ]
        cv = TSOutOfSampleSplit(n_splits=3, test_size=4)
        scoring = make_scorer(r2_score).set_score_request(sample_weight=True)

        xy = cross_validate(XYPipeline(steps()), X, y, cv=cv, scoring=scoring, params={'sample_weight': w})
        sk = cross_validate(Pipeline(steps()), X, y, cv=cv, scoring=scoring, params={'sample_weight': w})

        np.testing.assert_allclose(xy['test_score'], sk['test_score'])


# ==============================================================================
# Poids après un retrait de lignes
# ==============================================================================


class _ReverseXY(BaseEstimator, XYTransformerMixin):
    """XY step returning the rows of ``X`` and ``y`` in reverse order, labels kept."""

    def _fit(self, X, y):
        return self

    def _transform(self, X, y=None):
        return X.iloc[::-1] if y is None else (X.iloc[::-1], y.iloc[::-1])


# Poids de frame_xy : 1..6 ; la 2ᵉ cible manque, d'où les poids gardés 1, 3, 4, 5, 6
WEIGHTS = np.arange(1.0, 7.0)
KEPT_WEIGHTS = [1.0, 3.0, 4.0, 5.0, 6.0]


@pytest.fixture
def gapped_noisy_xy(noisy_xy):
    """``noisy_xy`` with the targets of rows 3 and 11 missing."""
    X, y, w = noisy_xy
    y = y.copy()
    y.iloc[[3, 11]] = np.nan
    return X, y, w


class TestWeightsFollowTheKeptRows:
    """A parameter with one value per input row follows the rows an XY step keeps."""

    @pytest.mark.parametrize('routed', [False, True], ids=['step_param', 'metadata_routing'])
    def test_estimator_receives_the_weights_of_the_kept_rows(self, routed, frame_xy, request):
        """Behind ``DropNaNTarget``, the estimator gets the 5 weights of its 5 training rows."""
        X, y = frame_xy
        if routed:
            request.getfixturevalue('routing')
            model, kwargs = WeightedEstimator().set_fit_request(sample_weight=True), {'sample_weight': WEIGHTS}
        else:
            model, kwargs = WeightedEstimator(), {'model__sample_weight': WEIGHTS}

        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', model)]).fit(X, y, **kwargs)

        assert pipeline['model'].weights_.tolist() == KEPT_WEIGHTS

    def test_weighted_regression_of_the_observed_rows(self, gapped_noisy_xy):
        """The coefficients are those of a weighted regression on the observed rows and their weights."""
        X, y, w = gapped_noisy_xy
        observed = y.notna().to_numpy()

        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', LinearRegression())])
        pipeline.fit(X, y, model__sample_weight=w)

        # Référence indépendante : régression pondérée sur les lignes observées
        reference = LinearRegression().fit(X[observed], y[observed], sample_weight=w[observed])
        np.testing.assert_allclose(pipeline['model'].coef_, reference.coef_)

    def test_positions_read_on_the_target_behind_an_array_step(self, frame_xy):
        """Behind a scaler returning arrays, the kept rows are located on the index of ``y``."""
        X, y = frame_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', WeightedEstimator())])

        pipeline.fit(X, y, model__sample_weight=WEIGHTS)

        assert pipeline['model'].weights_.tolist() == KEPT_WEIGHTS

    def test_step_before_the_drop_receives_every_weight(self, frame_xy):
        """A step placed before ``DropNaNTarget`` gets the 6 weights of its 6 rows."""
        X, y = frame_xy
        pipeline = XYPipeline([('xy', WeightedXY()), ('drop', DropNaNTarget()), ('model', WeightedEstimator())])

        pipeline.fit(X, y, xy__sample_weight=WEIGHTS, model__sample_weight=WEIGHTS)

        assert pipeline['xy'].weights_.tolist() == WEIGHTS.tolist()

    def test_reordered_rows_reorder_the_weights(self, frame_xy):
        """Behind a step reversing the rows, the weights are reversed too."""
        X, y = frame_xy
        pipeline = XYPipeline([('reverse', _ReverseXY()), ('model', WeightedEstimator())])

        pipeline.fit(X, y, model__sample_weight=WEIGHTS)

        assert pipeline['model'].weights_.tolist() == WEIGHTS[::-1].tolist()

    def test_fit_transform_aligns_the_last_step_weights(self, frame_xy):
        """``fit_transform``: an XY last step placed after the drop gets the weights of the kept rows."""
        X, y = frame_xy
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('xy', WeightedXY())])

        pipeline.fit_transform(X, y, xy__sample_weight=WEIGHTS)

        assert pipeline['xy'].weights_.tolist() == KEPT_WEIGHTS

    @pytest.mark.parametrize('routed', [False, True], ids=['plain', 'metadata_routing'])
    def test_score_weights_follow_the_kept_rows(self, routed, gapped_noisy_xy, request):
        """``score(X, y, sample_weight=w)`` is the weighted R² of the observed rows."""
        X, y, w = gapped_noisy_xy
        model = Ridge(alpha=1.0)
        if routed:
            request.getfixturevalue('routing')
            model = model.set_fit_request(sample_weight=False).set_score_request(sample_weight=True)
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', model)]).fit(X, y)
        observed = y.notna().to_numpy()

        # Valeur d'or : R² pondéré des lignes observées
        expected = r2_score(y[observed], pipeline.predict(X[observed]), sample_weight=w[observed])
        assert pipeline.score(X, y, sample_weight=w) == pytest.approx(expected)

    def test_arrays_without_index_are_refused(self, frame_xy):
        """On arrays, the kept rows cannot be located: the weights are refused with an explicit error."""
        X, y = (np.asarray(data) for data in frame_xy)
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', WeightedEstimator())])

        with pytest.raises(
            ValueError, match="rows kept by step 'drop' cannot be located: give X or y a unique pandas index"
        ):
            pipeline.fit(X, y, model__sample_weight=WEIGHTS)

    def test_arrays_without_weights_fit(self, frame_xy):
        """On arrays, a fit without row-aligned parameter drops the rows as usual."""
        X, y = (np.asarray(data) for data in frame_xy)

        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', WeightedEstimator())]).fit(X, y)

        assert pipeline['model'].n_rows_ == 5

    def test_duplicated_labels_are_refused(self, frame_xy):
        """With duplicated index labels, the kept rows cannot be located either."""
        X, y = frame_xy
        labels = pd.Index(['p', 'p', 'q', 'q', 'r', 'r'])
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', WeightedEstimator())])

        with pytest.raises(ValueError, match='cannot be located'):
            pipeline.fit(X.set_axis(labels), y.set_axis(labels), model__sample_weight=WEIGHTS)


# ==============================================================================
# transform_input
# ==============================================================================


class TestTransformInput:
    """``transform_input``: metadata transformed by the preceding steps before reaching a step."""

    @pytest.mark.parametrize(
        'method, final',
        [
            ('fit', lambda: Ridge()),
            ('fit_transform', lambda: StandardScaler()),
            ('fit_predict', lambda: KMeans(n_clusters=2, n_init=1, random_state=0)),
        ],
        ids=['fit', 'fit_transform', 'fit_predict'],
    )
    def test_requires_routing(self, method, final, linear_xy):
        """Without routing, the three fitting methods refuse ``transform_input`` with the message of ``Pipeline``."""
        X, y = linear_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('final', final())], transform_input=['X_val'])

        with pytest.raises(ValueError, match='The `transform_input` parameter can only be set if metadata routing'):
            getattr(pipeline, method)(X, y)

    def test_validation_set_is_scaled_by_the_fitted_scaler(self, routing, noisy_xy):
        """``X_val`` reaches the estimator scaled with the means and deviations learnt on ``X``."""
        X, y, _w = noisy_xy
        X_val = X.iloc[:3] + 1.0
        pipeline = XYPipeline(
            [('scale', StandardScaler()), ('model', ValidationSetEstimator().set_fit_request(X_val=True))],
            transform_input=['X_val'],
        )

        pipeline.fit(X, y, X_val=X_val)

        # Référence indépendante : StandardScaler ajusté sur X, appliqué à X_val
        np.testing.assert_allclose(pipeline['model'].X_val_, StandardScaler().fit(X).transform(X_val))

    def test_validation_set_is_that_of_pipeline(self, routing, noisy_xy):
        """``X_val`` reaches the estimator with the values given by ``Pipeline``."""
        X, y, _w = noisy_xy
        X_val = X.iloc[:3] + 1.0
        steps = lambda: [('scale', StandardScaler()), ('model', ValidationSetEstimator().set_fit_request(X_val=True))]

        xy = XYPipeline(steps(), transform_input=['X_val']).fit(X, y, X_val=X_val)
        sk = Pipeline(steps(), transform_input=['X_val']).fit(X, y, X_val=X_val)

        np.testing.assert_allclose(xy['model'].X_val_, sk['model'].X_val_)

    def test_validation_set_keeps_every_row_behind_drop(self, routing, frame_xy):
        """``X_val`` goes through ``DropNaNTarget`` without target: none of its rows is dropped."""
        X, y = frame_xy
        pipeline = XYPipeline(
            [('drop', DropNaNTarget()), ('model', ValidationSetEstimator().set_fit_request(X_val=True))],
            transform_input=['X_val'],
        )

        pipeline.fit(X, y, X_val=X)

        assert pipeline['model'].X_val_.shape == (6, 2)


# ==============================================================================
# Paramètres non alignés sur les lignes
# ==============================================================================


class ParamsEstimator(BaseEstimator, RegressorMixin):
    """Regressor that records the extra fit parameters it receives."""

    def fit(self, X, y, **params):
        """Record ``params``."""
        self.params_ = params
        return self

    def predict(self, X):
        """Return zeros."""
        return np.zeros(len(X))


class TestParamsNotAlignedOnRows:
    """Behind a dropping step, a parameter without one value per input row reaches the step unchanged."""

    @pytest.mark.parametrize(
        'value',
        [0.5, np.ones(3), XYPipeline([('scale', StandardScaler())])],
        ids=['scalar', 'other_length', 'estimator_object'],
    )
    def test_parameter_is_unchanged(self, value, frame_xy):
        """A scalar, an array of another length or an estimator object is passed as given."""
        X, y = frame_xy

        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', ParamsEstimator())])
        pipeline.fit(X, y, model__extra=value)

        assert pipeline['model'].params_['extra'] is value

    def test_second_unlocatable_drop_names_the_first(self, frame_xy):
        """After two drops on arrays, the error names the first step whose kept rows are unknown."""
        X, y = (np.asarray(data) for data in frame_xy)
        pipeline = XYPipeline([('drop1', DropNaNTarget()), ('drop2', DropNaNTarget()), ('model', WeightedEstimator())])

        with pytest.raises(ValueError, match="rows kept by step 'drop1' cannot be located"):
            pipeline.fit(X, y, model__sample_weight=WEIGHTS)


# ==============================================================================
# Méthodes de classification avec routage
# ==============================================================================


@pytest.mark.usefixtures('routing')
class TestRoutedClassifierMethods:
    """With routing enabled and no metadata, the classifier methods give the values of ``Pipeline``."""

    @pytest.mark.parametrize('method', ['predict_proba', 'predict_log_proba'])
    def test_values_equal_pipeline(self, method, classification_xy):
        """Same probabilities as ``Pipeline([scale, logistic])`` under routing."""
        X, y = classification_xy
        xy = XYPipeline([('scale', StandardScaler()), ('model', LogisticRegression())]).fit(X, y)
        sk = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression())]).fit(X, y)

        np.testing.assert_allclose(getattr(xy, method)(X), getattr(sk, method)(X))
