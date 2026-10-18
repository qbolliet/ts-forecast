"""``XYPipeline`` in sklearn model selection, with ``TSOutOfSampleSplit`` and an XY step dropping rows.

Covers ``cross_validate`` (scores of ``Pipeline`` without XY step; golden
scores and per-fold training sets behind ``DropNaNTarget``; predicting scorers
facing missing test targets), ``cross_val_predict`` (refusal of a time split,
one prediction per row with a partition), ``GridSearchCV`` and
``RandomizedSearchCV`` (candidates and scores of ``Pipeline``, best parameters,
a step replaced by the grid, refitted best estimator) and ``n_jobs=2``.

On ``gapped_linear_xy`` the target is exactly ``2a - b + 1`` on the observed
rows: a linear model fitted behind ``DropNaNTarget`` scores 1 on every fold and
predicts ``2a - b + 1`` on every row; a misalignment of rows and targets would
break both.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tsforecast.crossvals import TSOutOfSampleSplit
from tsforecast.xy import DropNaNTarget, XYPipeline
from tests.support.estimators import SpyEstimator

from .conftest import GAPPED_ROWS, month_ends


def _time_split() -> TSOutOfSampleSplit:
    """3 folds of 3 test rows on 15 rows: tests 6-8, 9-11, 12-14, expanding training before each."""
    return TSOutOfSampleSplit(n_splits=3, test_size=3)


def _truth(X: pd.DataFrame) -> np.ndarray:
    """Exact target ``2a - b + 1`` of ``gapped_linear_xy``, observed or not."""
    return (2.0 * X['a'] - X['b'] + 1.0).to_numpy()


def _drop_then_linear() -> XYPipeline:
    """``[DropNaNTarget, LinearRegression]``."""
    return XYPipeline([('drop', DropNaNTarget()), ('model', LinearRegression())])


@pytest.fixture
def noisy_xy():
    """``(X, y)``: 15 month ends, 2 Gaussian features, a noisy linear target without missing value."""
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(15, 2)), columns=['a', 'b'], index=month_ends(15))
    y = pd.Series(X['a'] - X['b'] + rng.normal(0.0, 0.3, 15), name='t')
    return X, y


# ==============================================================================
# cross_validate
# ==============================================================================


class TestCrossValidate:
    """``cross_validate`` with a time split: fold scores computed on the observed rows of each fold."""

    def test_scores_equal_pipeline_without_xy_step(self, noisy_xy):
        """Without XY step, the fold scores are those of ``Pipeline``."""
        X, y = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', Ridge(alpha=1.0))]

        xy = cross_validate(XYPipeline(steps()), X, y, cv=_time_split())['test_score']
        sk = cross_validate(Pipeline(steps()), X, y, cv=_time_split())['test_score']

        np.testing.assert_allclose(xy, sk)

    def test_exact_model_scores_one_on_every_fold(self, gapped_linear_xy):
        """Behind ``DropNaNTarget``, the exact linear model scores R² = 1 on the three folds."""
        X, y = gapped_linear_xy

        scores = cross_validate(_drop_then_linear(), X, y, cv=_time_split())['test_score']

        np.testing.assert_allclose(scores, [1.0, 1.0, 1.0])

    def test_scores_match_folds_computed_by_hand(self, gapped_linear_xy):
        """``[scale, drop, ridge]``: each fold score equals the one computed outside any pipeline.

        By hand: the scaler is fitted on every training row (it comes before the
        drop), the ridge on the observed training rows, and the R² is computed
        on the observed test rows.
        """
        X, y = gapped_linear_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', Ridge(alpha=1.0))])

        # Valeurs d'or recalculées pli par pli, hors de tout pipeline
        expected = []
        for train, test in _time_split().split(X):
            scaler = StandardScaler().fit(X.iloc[train])
            seen, unseen = y.iloc[train].notna().to_numpy(), y.iloc[test].notna().to_numpy()
            model = Ridge(alpha=1.0).fit(scaler.transform(X.iloc[train])[seen], y.iloc[train][seen])
            prediction = model.predict(scaler.transform(X.iloc[test])[unseen])
            expected.append(r2_score(y.iloc[test][unseen], prediction))

        np.testing.assert_allclose(cross_validate(pipeline, X, y, cv=_time_split())['test_score'], expected)

    def test_each_fold_trains_on_its_observed_rows(self, gapped_linear_xy):
        """The final estimators of the folds see 5, 8 and 10 targets: training rows minus the missing ones.

        Training sets of 6, 9 and 12 rows; row 4 is missing in all three, row 10
        in the last one.
        """
        X, y = gapped_linear_xy
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', SpyEstimator())])

        estimators = cross_validate(pipeline, X, y, cv=_time_split(), return_estimator=True)['estimator']

        assert [len(estimator['model'].fit_y_) for estimator in estimators] == [5, 8, 10]

    def test_predicting_scorer_meets_the_missing_test_targets(self, gapped_linear_xy):
        """Pinned: ``scoring='r2'`` compares one prediction per row with the raw target, NaN included.

        ``predict`` never drops a row, so the scorer receives the missing target
        of row 10 (second test fold); only ``scoring=None`` (``pipeline.score``)
        drops it.
        """
        X, y = gapped_linear_xy

        with pytest.raises(ValueError, match='Input contains NaN'):
            cross_validate(_drop_then_linear(), X, y, cv=_time_split(), scoring='r2', error_score='raise')


# ==============================================================================
# cross_val_predict
# ==============================================================================


class TestCrossValPredict:
    """``cross_val_predict``: one prediction per row of ``X``, in the row order."""

    @pytest.mark.parametrize('pipeline_class', [Pipeline, XYPipeline], ids=['pipeline', 'xy_pipeline'])
    def test_time_split_is_refused(self, pipeline_class, gapped_linear_xy):
        """A time split does not test every row: refused before any fit, by both pipelines."""
        X, y = gapped_linear_xy
        pipeline = pipeline_class([('model', LinearRegression())])

        with pytest.raises(ValueError, match='cross_val_predict only works for partitions'):
            cross_val_predict(pipeline, X, y, cv=_time_split())

    def test_partition_gives_the_exact_target_on_every_row(self, gapped_linear_xy):
        """With ``KFold(3)``, the out-of-fold predictions are ``2a - b + 1``, row by row."""
        X, y = gapped_linear_xy

        predictions = cross_val_predict(_drop_then_linear(), X, y, cv=KFold(n_splits=3))

        np.testing.assert_allclose(predictions, _truth(X))

    def test_rows_without_target_are_predicted(self, gapped_linear_xy):
        """The rows whose target is missing get a prediction too."""
        X, y = gapped_linear_xy

        predictions = cross_val_predict(_drop_then_linear(), X, y, cv=KFold(n_splits=3))

        np.testing.assert_allclose(predictions[list(GAPPED_ROWS)], _truth(X)[list(GAPPED_ROWS)])


# ==============================================================================
# GridSearchCV
# ==============================================================================


class TestGridSearch:
    """``GridSearchCV`` over the parameters and the steps of the pipeline."""

    @pytest.fixture
    def intercept_search(self, gapped_linear_xy):
        """``GridSearchCV`` over ``model__fit_intercept``, fitted on ``gapped_linear_xy``."""
        search = GridSearchCV(_drop_then_linear(), {'model__fit_intercept': [False, True]}, cv=_time_split())
        return search.fit(*gapped_linear_xy)

    def test_best_params(self, intercept_search):
        """The target has an intercept: the search keeps ``fit_intercept=True``."""
        assert intercept_search.best_params_ == {'model__fit_intercept': True}

    def test_best_score_is_one(self, intercept_search):
        """The kept candidate scores R² = 1 on average over the folds."""
        assert intercept_search.best_score_ == pytest.approx(1.0)

    def test_refitted_best_estimator_predicts_every_row(self, intercept_search, gapped_linear_xy):
        """Refitted on all the data, the best pipeline predicts ``2a - b + 1`` on the 15 rows."""
        X, _y = gapped_linear_xy

        np.testing.assert_allclose(intercept_search.best_estimator_.predict(X), _truth(X))

    def test_cv_results_equal_pipeline_without_xy_step(self, noisy_xy):
        """Without XY step, the mean scores of every candidate are those of ``Pipeline``."""
        X, y = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', Ridge())]
        grid = {'model__alpha': [0.1, 10.0], 'scale': [StandardScaler(), 'passthrough']}

        xy = GridSearchCV(XYPipeline(steps()), grid, cv=_time_split()).fit(X, y)
        sk = GridSearchCV(Pipeline(steps()), grid, cv=_time_split()).fit(X, y)

        np.testing.assert_allclose(xy.cv_results_['mean_test_score'], sk.cv_results_['mean_test_score'])

    @pytest.mark.filterwarnings('ignore::sklearn.exceptions.FitFailedWarning')
    @pytest.mark.filterwarnings('ignore:One or more of the test scores are non-finite:UserWarning')
    def test_grid_can_skip_the_drop_step(self, gapped_linear_xy):
        """Grid over the step itself: without ``DropNaNTarget`` the fit fails, so the search keeps the step."""
        X, y = gapped_linear_xy
        search = GridSearchCV(
            _drop_then_linear(), {'drop': [DropNaNTarget(), 'passthrough']}, cv=_time_split(), error_score=np.nan
        )

        search.fit(X, y)

        assert isinstance(search.best_estimator_['drop'], DropNaNTarget)

    def test_skipped_drop_step_fails_every_fold(self, gapped_linear_xy):
        """Alone in the grid, the ``'passthrough'`` candidate fails on the missing targets of every fold."""
        X, y = gapped_linear_xy
        search = GridSearchCV(_drop_then_linear(), {'drop': ['passthrough']}, cv=_time_split(), error_score=np.nan)

        with pytest.raises(ValueError, match='All the 3 fits failed'):
            search.fit(X, y)


# ==============================================================================
# RandomizedSearchCV
# ==============================================================================


class TestRandomizedSearch:
    """``RandomizedSearchCV``: same draws as ``Pipeline`` for the same seed."""

    def test_same_candidates_as_pipeline(self, noisy_xy):
        """With ``random_state=0``, both searches draw the same three candidates."""
        X, y = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', Ridge())]
        space = {'model__alpha': [0.01, 0.1, 1.0, 10.0]}

        xy = RandomizedSearchCV(XYPipeline(steps()), space, n_iter=3, cv=_time_split(), random_state=0).fit(X, y)
        sk = RandomizedSearchCV(Pipeline(steps()), space, n_iter=3, cv=_time_split(), random_state=0).fit(X, y)

        assert xy.cv_results_['params'] == sk.cv_results_['params']

    def test_same_scores_as_pipeline(self, noisy_xy):
        """With ``random_state=0``, the mean scores of the candidates are those of ``Pipeline``."""
        X, y = noisy_xy
        steps = lambda: [('scale', StandardScaler()), ('model', Ridge())]
        space = {'model__alpha': [0.01, 0.1, 1.0, 10.0]}

        xy = RandomizedSearchCV(XYPipeline(steps()), space, n_iter=3, cv=_time_split(), random_state=0).fit(X, y)
        sk = RandomizedSearchCV(Pipeline(steps()), space, n_iter=3, cv=_time_split(), random_state=0).fit(X, y)

        np.testing.assert_allclose(xy.cv_results_['mean_test_score'], sk.cv_results_['mean_test_score'])

    def test_best_params_behind_drop(self, gapped_linear_xy):
        """Behind ``DropNaNTarget``, the search keeps ``fit_intercept=True``."""
        search = RandomizedSearchCV(
            _drop_then_linear(), {'model__fit_intercept': [False, True]}, n_iter=2, cv=_time_split(), random_state=0
        )

        assert search.fit(*gapped_linear_xy).best_params_ == {'model__fit_intercept': True}


# ==============================================================================
# Parallélisme
# ==============================================================================


@pytest.mark.slow
class TestParallel:
    """``n_jobs=2``: the pipelines are pickled to worker processes and give the results of ``n_jobs=1``."""

    def test_cross_validate(self, gapped_linear_xy):
        """``cross_validate(n_jobs=2)`` gives the fold scores of ``n_jobs=1``."""
        X, y = gapped_linear_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', Ridge(alpha=1.0))])

        parallel = cross_validate(pipeline, X, y, cv=_time_split(), n_jobs=2)['test_score']
        sequential = cross_validate(pipeline, X, y, cv=_time_split(), n_jobs=1)['test_score']

        np.testing.assert_allclose(parallel, sequential)

    def test_grid_search(self, gapped_linear_xy):
        """``GridSearchCV(n_jobs=2)`` keeps ``fit_intercept=True`` with a mean score of 1."""
        search = GridSearchCV(
            _drop_then_linear(), {'model__fit_intercept': [False, True]}, cv=_time_split(), n_jobs=2
        ).fit(*gapped_linear_xy)

        assert (search.best_params_, search.best_score_) == ({'model__fit_intercept': True}, pytest.approx(1.0))

    def test_cross_val_predict(self, gapped_linear_xy):
        """``cross_val_predict(n_jobs=2)`` predicts ``2a - b + 1`` on every row."""
        X, y = gapped_linear_xy

        predictions = cross_val_predict(_drop_then_linear(), X, y, cv=KFold(n_splits=3), n_jobs=2)

        np.testing.assert_allclose(predictions, _truth(X))
