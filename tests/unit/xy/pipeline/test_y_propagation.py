"""Propagation of ``y`` through the steps of ``XYPipeline``.

Covers, with spy steps that log every call (``SpyTransformer``,
``SpyXYTransformer``, ``SpyEstimator``), what each step really receives:
along the compositions (XY steps only, XY mixed with ordinary sklearn
steps, ``'passthrough'`` / ``None``, with or without a final estimator, XY step
in last position, single step, empty pipeline, step without ``fit_transform``);
along the kinds of target (``Series``, multi-target ``DataFrame``, 1-D and
2-D arrays, ``None``, index different from ``X``, rows dropped, all NaN);
along the methods (``fit``, ``fit_transform``, ``transform`` with and without
``y``, ``predict``, ``fit_predict``, ``score``), including the refusal of a
target by the prediction methods. Also covers the alignment of
``X`` and ``y`` by index once rows are dropped, the restoration of the pandas
structure and the non-mutation of the inputs.
"""
from typing import Callable, List

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from tsforecast.xy import DropNaNTarget, XYPipeline, XYTransformerMixin
from tests.support.estimators import (
    DropRowsXY,
    SpyEstimator,
    SpyTransformer,
    SpyXYTransformer,
)


# ==============================================================================
# Doublures locales
# ==============================================================================


class _FitThenTransformXY(BaseEstimator, XYTransformerMixin):
    """XY step without its own ``fit_transform``: the one of the mixin calls ``fit`` then ``transform``.

    ``transform(X, y)`` returns ``(X, y + y_offset)``; ``fit`` logs the target it got.
    """

    y_offset = 100.0

    def fit(self, X, y=None):
        self.fit_y_ = None if y is None else y.copy()
        return self

    def transform(self, X, y=None):
        if y is None:
            return X
        return X, y + self.y_offset


class _FitPredictSpy(BaseEstimator):
    """Final estimator exposing ``fit_predict``; records the ``(X, y)`` it receives."""

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None):
        self.fit_X_ = X.copy()
        self.fit_y_ = None if y is None else y.copy()
        return np.arange(len(X))


class _ArrayXY(BaseEstimator, XYTransformerMixin):
    """XY identity step that returns arrays: ``(np.asarray(X), np.asarray(y))``."""

    def _fit(self, X, y):
        return self

    def _transform(self, X, y=None):
        if y is None:
            return np.asarray(X)
        return np.asarray(X), np.asarray(y)


class _ArrayDropXY(BaseEstimator, XYTransformerMixin):
    """XY step that drops the rows without target and returns arrays."""

    def _fit(self, X, y):
        return self

    def _transform(self, X, y=None):
        if y is None:
            return np.asarray(X)
        keep = ~np.isnan(np.asarray(y, dtype=float))
        return np.asarray(X)[keep], np.asarray(y)[keep]


class _DuplicateColumns(BaseEstimator, TransformerMixin):
    """Ordinary step returning ``X`` twice side by side, as an array, without feature names."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X):
        return np.hstack([np.asarray(X), np.asarray(X)])


def _calls(step, method: str) -> List:
    """Targets received by ``step`` at each call of ``method``, in order."""
    return [call.y for call in step.calls_ if call.method == method]


def _offset(step) -> float:
    """Constant added to ``y`` by a step of these tests (0 for the others)."""
    return float(getattr(step, 'y_offset', 0.0))


# ==============================================================================
# Composition des étapes
# ==============================================================================

# Compositions terminées par l'estimateur espion, et décalage total attendu sur y
WITH_ESTIMATOR = {
    'all_xy': (lambda: [('xy1', SpyXYTransformer(1.0)), ('xy2', SpyXYTransformer(10.0)), ('model', SpyEstimator())], 11.0),
    'xy_then_sklearn': (lambda: [('xy', SpyXYTransformer(1.0)), ('sk', SpyTransformer()), ('model', SpyEstimator())], 1.0),
    'sklearn_then_xy': (lambda: [('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('model', SpyEstimator())], 1.0),
    'passthrough_step': (lambda: [('xy', SpyXYTransformer(1.0)), ('skip', 'passthrough'), ('model', SpyEstimator())], 1.0),
    'none_step': (lambda: [('skip', None), ('xy', SpyXYTransformer(1.0)), ('model', SpyEstimator())], 1.0),
    'single_estimator': (lambda: [('model', SpyEstimator())], 0.0),
    'step_without_fit_transform': (lambda: [('xy', _FitThenTransformXY()), ('model', SpyEstimator())], 100.0),
}

# Compositions sans estimateur final : fit_transform rend (X, y décalé)
WITHOUT_ESTIMATOR = {
    'final_passthrough': (lambda: [('xy', SpyXYTransformer(1.0)), ('sk', SpyTransformer()), ('end', 'passthrough')], 1.0),
    'final_none': (lambda: [('xy', SpyXYTransformer(1.0)), ('end', None)], 1.0),
    'xy_last': (lambda: [('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0))], 1.0),
    'single_xy_step': (lambda: [('xy', SpyXYTransformer(1.0))], 1.0),
    'two_xy_steps': (lambda: [('xy1', SpyXYTransformer(1.0)), ('xy2', SpyXYTransformer(10.0))], 11.0),
    'last_without_fit_transform': (lambda: [('sk', SpyTransformer()), ('xy', _FitThenTransformXY())], 100.0),
}


class TestComposition:
    """Whatever the composition, each step receives ``y`` as transformed by the steps before it."""

    @pytest.mark.parametrize(('make', 'offset'), WITH_ESTIMATOR.values(), ids=WITH_ESTIMATOR.keys())
    def test_estimator_receives_the_shifted_target(self, make: Callable, offset, linear_xy):
        """Valeur d'or : y + somme des décalages des étapes XY en amont."""
        X, y = linear_xy
        pipeline = XYPipeline(make()).fit(X, y)

        pd.testing.assert_series_equal(pipeline.named_steps['model'].fit_y_, y + offset)

    @pytest.mark.parametrize(('make', 'offset'), WITH_ESTIMATOR.values(), ids=WITH_ESTIMATOR.keys())
    def test_estimator_receives_x_unchanged(self, make: Callable, offset, linear_xy):
        """Every step here keeps ``X``: the estimator sees the input frame."""
        X, y = linear_xy
        pipeline = XYPipeline(make()).fit(X, y)

        pd.testing.assert_frame_equal(pipeline.named_steps['model'].fit_X_, X)

    @pytest.mark.parametrize(
        'make', [make for make, _ in WITH_ESTIMATOR.values()] + [make for make, _ in WITHOUT_ESTIMATOR.values()],
        ids=list(WITH_ESTIMATOR) + list(WITHOUT_ESTIMATOR),
    )
    def test_each_step_is_fitted_on_the_target_of_its_rank(self, make: Callable, linear_xy):
        """At ``fit``, step k receives ``y`` plus the offsets of steps 1..k-1."""
        X, y = linear_xy
        pipeline = XYPipeline(make()).fit(X, y)

        # Reconstitution des cibles attendues, étape par étape
        received, expected, shift = [], [], 0.0
        for _name, step in pipeline.steps:
            if isinstance(step, (SpyTransformer, SpyXYTransformer)):
                received.append(_calls(step, 'fit')[0].tolist())
                expected.append((y + shift).tolist())
            elif isinstance(step, _FitThenTransformXY):
                received.append(step.fit_y_.tolist())
                expected.append((y + shift).tolist())
            if step not in (None, 'passthrough'):
                shift += _offset(step)

        assert received == expected

    @pytest.mark.parametrize(('make', 'offset'), WITHOUT_ESTIMATOR.values(), ids=WITHOUT_ESTIMATOR.keys())
    def test_fit_transform_returns_the_shifted_target(self, make: Callable, offset, linear_xy):
        """Without final estimator, ``fit_transform`` returns ``(X, y + offsets)``."""
        X, y = linear_xy

        _X_t, y_t = XYPipeline(make()).fit_transform(X, y)

        pd.testing.assert_series_equal(y_t, y + offset)

    @pytest.mark.parametrize(('make', 'offset'), WITHOUT_ESTIMATOR.values(), ids=WITHOUT_ESTIMATOR.keys())
    def test_fit_transform_returns_x_unchanged(self, make: Callable, offset, linear_xy):
        """The XY steps of these compositions leave ``X`` as it is."""
        X, y = linear_xy

        X_t, _y_t = XYPipeline(make()).fit_transform(X, y)

        pd.testing.assert_frame_equal(X_t, X)

    @pytest.mark.parametrize(('make', 'offset'), WITHOUT_ESTIMATOR.values(), ids=WITHOUT_ESTIMATOR.keys())
    def test_transform_with_target_returns_the_shifted_target(self, make: Callable, offset, linear_xy):
        """After ``fit``, ``transform(X, y)`` shifts ``y`` like ``fit_transform``."""
        X, y = linear_xy
        pipeline = XYPipeline(make()).fit(X, y)

        _X_t, y_t = pipeline.transform(X, y)

        pd.testing.assert_series_equal(y_t, y + offset)

    @pytest.mark.parametrize(('make', 'offset'), WITHOUT_ESTIMATOR.values(), ids=WITHOUT_ESTIMATOR.keys())
    def test_transform_without_target_returns_x_alone(self, make: Callable, offset, linear_xy):
        """``transform(X)`` returns ``X'`` only, even with XY steps."""
        X, y = linear_xy
        pipeline = XYPipeline(make()).fit(X, y)

        assert isinstance(pipeline.transform(X), pd.DataFrame)

    @pytest.mark.parametrize('position', ['intermediate', 'last'])
    def test_step_without_fit_transform_without_target(self, position, linear_xy):
        """Without ``y``, the ``fit`` then ``transform`` fallback fits with ``y=None``."""
        X, _y = linear_xy
        step = _FitThenTransformXY()
        steps = [('xy', step), ('sk', SpyTransformer())] if position == 'intermediate' else [('sk', SpyTransformer()), ('xy', step)]

        XYPipeline(steps).fit_transform(X)

        assert step.fit_y_ is None

    def test_empty_pipeline_is_refused(self, linear_xy):
        """A pipeline without step raises at ``fit``, like ``sklearn.pipeline.Pipeline``."""
        with pytest.raises(ValueError, match='The pipeline is empty'):
            XYPipeline([]).fit(*linear_xy)


# ==============================================================================
# Natures de cible
# ==============================================================================


def _supervised_pipeline() -> XYPipeline:
    """``SpyTransformer -> SpyXYTransformer(+1) -> DropNaNTarget -> SpyEstimator``."""
    return XYPipeline([
        ('sk', SpyTransformer()),
        ('xy', SpyXYTransformer(1.0)),
        ('drop', DropNaNTarget()),
        ('model', SpyEstimator()),
    ])


def _two_targets(y: pd.Series) -> pd.DataFrame:
    """Targets ``u`` = ``y`` (NaN on the 2nd row) and ``v`` = 10..15, on the index of ``y``."""
    return pd.DataFrame({'u': y, 'v': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}, index=y.index)


# Valeurs d'or : y = 1, NaN, 3, 4, 5, 6 → la ligne 2 disparaît, puis + 1
TARGET_KINDS = {
    'series': (lambda y: y, pd.Series, [2.0, 4.0, 5.0, 6.0, 7.0]),
    'dataframe_2_targets': (_two_targets, pd.DataFrame, [[2.0, 11.0], [4.0, 13.0], [5.0, 14.0], [6.0, 15.0], [7.0, 16.0]]),
    'ndarray_1d': (lambda y: y.to_numpy(), np.ndarray, [2.0, 4.0, 5.0, 6.0, 7.0]),
    'ndarray_2d': (lambda y: _two_targets(y).to_numpy(), np.ndarray, [[2.0, 11.0], [4.0, 13.0], [5.0, 14.0], [6.0, 15.0], [7.0, 16.0]]),
    'index_differs_from_x': (lambda y: y.reset_index(drop=True), pd.Series, [2.0, 4.0, 5.0, 6.0, 7.0]),
}


class TestTargetKinds:
    """The kind of ``y`` survives the steps; its values arrive filtered and shifted."""

    @pytest.mark.parametrize(('make_y', 'kind', 'expected'), TARGET_KINDS.values(), ids=TARGET_KINDS.keys())
    def test_estimator_receives_the_same_kind(self, make_y, kind, expected, frame_xy):
        """A ``Series`` stays a ``Series``, an array an array, a ``DataFrame`` a ``DataFrame``."""
        X, y = frame_xy
        pipeline = _supervised_pipeline().fit(X, make_y(y))

        assert isinstance(pipeline.named_steps['model'].fit_y_, kind)

    @pytest.mark.parametrize(('make_y', 'kind', 'expected'), TARGET_KINDS.values(), ids=TARGET_KINDS.keys())
    def test_estimator_receives_the_golden_values(self, make_y, kind, expected, frame_xy):
        """Rows without target dropped, then ``+ 1`` from the XY step."""
        X, y = frame_xy
        pipeline = _supervised_pipeline().fit(X, make_y(y))

        np.testing.assert_array_equal(np.asarray(pipeline.named_steps['model'].fit_y_), expected)

    @pytest.mark.parametrize(('make_y', 'kind', 'expected'), TARGET_KINDS.values(), ids=TARGET_KINDS.keys())
    def test_estimator_receives_the_rows_of_its_target(self, make_y, kind, expected, frame_xy):
        """The rows of ``X`` kept are those of the kept targets: the 2nd date is gone."""
        X, y = frame_xy
        pipeline = _supervised_pipeline().fit(X, make_y(y))

        assert pipeline.named_steps['model'].fit_X_.index.equals(X.index.delete(1))

    def test_index_different_from_x_is_matched_by_position(self, frame_xy):
        """``y`` keeps its own labels: the rows are paired by position, not by label.

        Behaviour pinned, the one of ``sklearn.pipeline.Pipeline``: no alignment on
        the index labels is attempted.
        """
        X, y = frame_xy
        pipeline = _supervised_pipeline().fit(X, y.reset_index(drop=True))

        assert pipeline.named_steps['model'].fit_y_.index.tolist() == [0, 2, 3, 4, 5]

    def test_all_nan_target_leaves_no_row(self, frame_xy):
        """A target entirely NaN reaches the estimator as zero rows.

        Behaviour pinned: the pipeline does not refuse an empty training set; the
        NaN-tolerant spy fits on nothing.
        """
        X, y = frame_xy
        pipeline = _supervised_pipeline().fit(X, y * np.nan)

        assert pipeline.named_steps['model'].fit_X_.shape == (0, 2)

    def test_all_nan_target_is_refused_by_a_real_estimator(self, frame_xy):
        """With ``LinearRegression``, the empty training set raises from the estimator."""
        X, y = frame_xy
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', LinearRegression())])

        with pytest.raises(ValueError, match='0 sample'):
            pipeline.fit(X, y * np.nan)

    def test_no_target_gives_x_alone(self, frame_xy):
        """Unsupervised pipeline: ``fit_transform(X)`` returns ``X``, no row dropped."""
        X, _y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget())])

        pd.testing.assert_frame_equal(pipeline.fit_transform(X), X)

    def test_no_target_reaches_no_step(self, frame_xy):
        """Without ``y``, every step receives ``y=None`` at every call."""
        X, _y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget())])

        pipeline.fit_transform(X)

        assert [call.y for name in ('sk', 'xy') for call in pipeline.named_steps[name].calls_] == [None] * 4

    def test_unsupervised_final_estimator_is_fitted_without_target(self, frame_xy):
        """``KernelDensity`` last, no ``y``: the upstream spy is fitted with ``y=None``."""
        X, _y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('density', KernelDensity())])

        pipeline.fit(X)

        assert _calls(pipeline.named_steps['sk'], 'fit') == [None]


# ==============================================================================
# Alignement de X et y par l'index
# ==============================================================================

ALIGNMENT = [
    pytest.param(lambda: [('drop', DropNaNTarget())], id='drop_nan_on_dataframe'),
    pytest.param(lambda: [('sk', SpyTransformer()), ('drop', DropNaNTarget())], id='pandas_step_then_drop'),
    pytest.param(lambda: [('scale', StandardScaler()), ('drop', DropNaNTarget())], id='array_step_then_drop'),
]


class TestRowAlignment:
    """Once rows are dropped, ``X`` and ``y`` come out with the same index."""

    @pytest.mark.parametrize('make', ALIGNMENT)
    def test_fit_transform_aligns_x_and_y(self, make, frame_xy):
        """``fit_transform``: the index of ``X'`` is the index of ``y'``."""
        X_t, y_t = XYPipeline(make()).fit_transform(*frame_xy)

        assert X_t.index.equals(y_t.index)

    @pytest.mark.parametrize('make', ALIGNMENT)
    def test_transform_aligns_x_and_y(self, make, frame_xy):
        """``transform(X, y)``: the index of ``X'`` is the index of ``y'``."""
        X, y = frame_xy
        pipeline = XYPipeline(make()).fit(X, y)

        X_t, y_t = pipeline.transform(X, y)

        assert X_t.index.equals(y_t.index)

    def test_array_step_then_drop_keeps_the_dates_of_the_kept_rows(self, frame_xy):
        """Behind an array step, ``X'`` gets the dates of the rows whose target is observed."""
        X, y = frame_xy

        X_t, _y_t = XYPipeline([('scale', StandardScaler()), ('drop', DropNaNTarget())]).fit_transform(X, y)

        assert X_t.index.equals(X.index.delete(1))

    def test_unaligned_inputs_give_positions(self, frame_xy):
        """``X`` and ``y`` given with different indexes: ``X'`` cannot borrow labels from ``y'``.

        The labels of ``y`` are not those of ``X``, so the rows of ``X'`` are numbered
        ``0..n-1`` once rows are dropped.
        """
        X, y = frame_xy

        X_t, _y_t = XYPipeline([('scale', StandardScaler()), ('drop', DropNaNTarget())]).fit_transform(
            X, y.reset_index(drop=True)
        )

        assert X_t.index.tolist() == [0, 1, 2, 3, 4]

    def test_negative_targets_dropped_with_their_rows(self, frame_xy):
        """``DropRowsXY`` keeps the dates of the non-negative targets in ``X`` and ``y``."""
        X, y = frame_xy
        signed = pd.Series([1.0, -1.0, 3.0, -4.0, 5.0, 6.0], index=X.index)

        X_t, y_t = XYPipeline([('drop', DropRowsXY())]).fit_transform(X, signed)

        assert X_t.index.equals(y_t.index)

    def test_estimator_rows_stay_paired_after_an_array_step(self, frame_xy):
        """Behind an array step, the estimator still gets each row with its own target.

        The arrays carry no index, but the rows kept in ``X`` are the scaled rows
        whose target is observed, in the order of ``y``.
        """
        X, y = frame_xy
        pipeline = XYPipeline([('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', SpyEstimator())])

        pipeline.fit(X, y)

        # Référence indépendante : mise à l'échelle de tout X, puis lignes de cible observée
        expected = StandardScaler().fit_transform(X)[[0, 2, 3, 4, 5]]
        np.testing.assert_allclose(pipeline.named_steps['model'].fit_X_, expected)


# ==============================================================================
# Méthodes : fit, transform, predict, fit_predict, score
# ==============================================================================


class TestPasses:
    """Training passes give ``y`` to every step; prediction passes give it to none."""

    @pytest.fixture
    def fitted(self, frame_xy):
        """The supervised pipeline fitted on ``frame_xy``."""
        return _supervised_pipeline().fit(*frame_xy)

    def test_fit_gives_y_to_the_sklearn_step_at_fit_only(self, fitted):
        """The ordinary step gets ``y`` at ``fit`` (as in sklearn), not at its ``transform``."""
        calls = fitted.named_steps['sk'].calls_

        assert [(call.method, call.y is None) for call in calls] == [('fit', False), ('transform', True)]

    def test_fit_gives_y_to_the_xy_step_at_both_calls(self, fitted):
        """The XY step gets ``y`` at ``fit`` and at ``transform``."""
        calls = fitted.named_steps['xy'].calls_

        assert [(call.method, call.y is None) for call in calls] == [('fit', False), ('transform', False)]

    def test_fit_gives_the_raw_target_to_the_xy_step(self, fitted, frame_xy):
        """Upstream of the XY step, ``y`` is the input target, NaN included."""
        _X, y = frame_xy

        pd.testing.assert_series_equal(_calls(fitted.named_steps['xy'], 'fit')[0], y)

    def test_fit_gives_the_filtered_target_to_the_estimator(self, fitted, frame_xy):
        """Valeur d'or : (1, 3, 4, 5, 6) + 1 sur les dates à cible observée."""
        X, _y = frame_xy

        expected = pd.Series([2.0, 4.0, 5.0, 6.0, 7.0], index=X.index.delete(1), name='t')
        pd.testing.assert_series_equal(fitted.named_steps['model'].fit_y_, expected)

    def test_final_passthrough_fits_the_steps(self, frame_xy):
        """A final ``'passthrough'`` still fits the steps before it."""
        pipeline = XYPipeline([('xy', SpyXYTransformer(1.0)), ('end', 'passthrough')]).fit(*frame_xy)

        assert [call.method for call in pipeline.named_steps['xy'].calls_] == ['fit', 'transform']

    def test_transform_without_target_gives_none_to_every_step(self, frame_xy):
        """``transform(X)``: no step receives a target."""
        X, y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget())])
        pipeline.fit(X, y)

        pipeline.transform(X)

        assert [_calls(pipeline.named_steps[name], 'transform')[-1] for name in ('sk', 'xy')] == [None, None]

    def test_transform_without_target_drops_no_row(self, frame_xy):
        """``transform(X)`` keeps every row: ``DropNaNTarget`` has no target to read."""
        X, y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget())])

        pd.testing.assert_frame_equal(pipeline.fit(X, y).transform(X), X)

    def test_transform_with_target_and_no_xy_step_returns_x_alone(self, frame_xy):
        """Without XY step, ``transform(X, y)`` returns ``X'`` only: ``y`` is not given back."""
        X, y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer())]).fit(X, y)

        assert isinstance(pipeline.transform(X, y), pd.DataFrame)

    def test_transform_with_target_gives_y_to_the_xy_step_only(self, frame_xy):
        """``transform(X, y)``: the XY step reads ``y``, the ordinary step does not."""
        X, y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget())])
        pipeline.fit(X, y)

        pipeline.transform(X, y)

        received = [_calls(pipeline.named_steps[name], 'transform')[-1] is None for name in ('sk', 'xy')]
        assert received == [True, False]

    def test_transform_with_target_drops_and_shifts(self, frame_xy):
        """``transform(X, y)`` returns ``y`` filtered and shifted, like the training pass."""
        X, y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget())])
        pipeline.fit(X, y)

        _X_t, y_t = pipeline.transform(X, y)

        assert y_t.tolist() == [2.0, 4.0, 5.0, 6.0, 7.0]

    def test_predict_gives_none_to_every_step(self, fitted, frame_xy):
        """``predict(X)``: no step receives a target."""
        X, _y = frame_xy

        fitted.predict(X)

        assert [_calls(fitted.named_steps[name], 'transform')[-1] for name in ('sk', 'xy')] == [None, None]

    def test_predict_keeps_every_row(self, fitted, frame_xy):
        """``predict(X)``: one prediction per row of ``X``, NaN targets included."""
        X, _y = frame_xy

        assert len(fitted.predict(X)) == len(X)

    def test_predict_gives_every_row_to_the_estimator(self, fitted, frame_xy):
        """The estimator predicts on the 6 input dates."""
        X, _y = frame_xy

        fitted.predict(X)

        assert fitted.named_steps['model'].predict_X_[-1].index.equals(X.index)

    def test_predict_returns_the_learnt_mean(self, fitted, frame_xy):
        """Valeur d'or : moyenne de (2, 4, 5, 6, 7) = 4.8, sur chaque ligne."""
        X, _y = frame_xy

        np.testing.assert_allclose(fitted.predict(X), [4.8] * 6)

    def test_predict_refuses_a_positional_target(self, fitted, frame_xy):
        """``predict(X, y)``: no target slot, like ``Pipeline.predict``."""
        X, y = frame_xy

        with pytest.raises(TypeError, match='positional argument'):
            fitted.predict(X, y)

    @pytest.mark.parametrize('method', ['predict', 'predict_proba', 'predict_log_proba', 'decision_function'])
    def test_prediction_methods_refuse_a_keyword_target(self, method, frame_xy):
        """``<method>(X, y=y)`` raises, naming the method; ``score`` is the way to use ``y``."""
        X, y = frame_xy
        labels = pd.Series([0, 1, 0, 1, 0, 1], index=X.index)
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('clf', LogisticRegression())]).fit(X, labels)

        with pytest.raises(TypeError, match=f'XYPipeline.{method} does not take y'):
            getattr(pipeline, method)(X, y=labels)

    def test_score_samples_refuses_a_target(self, frame_xy):
        """``score_samples`` takes ``X`` only."""
        X, y = frame_xy
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('density', KernelDensity())]).fit(X)

        with pytest.raises(TypeError, match="unexpected keyword argument 'y'"):
            pipeline.score_samples(X, y=y)

    @pytest.mark.parametrize('method', ['predict_proba', 'predict_log_proba', 'decision_function'])
    def test_prediction_methods_keep_every_row(self, method, frame_xy):
        """Behind ``DropNaNTarget``, each prediction method answers for every row of ``X``."""
        X, _y = frame_xy
        labels = pd.Series([0.0, np.nan, 0.0, 1.0, 0.0, 1.0], index=X.index)
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('clf', LogisticRegression())]).fit(X, labels)

        assert len(getattr(pipeline, method)(X)) == len(X)

    def test_score_samples_keeps_every_row(self, frame_xy):
        """``score_samples`` behind ``DropNaNTarget`` gives one value per row of ``X``."""
        X, y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('density', KernelDensity())]).fit(X, y)

        assert len(pipeline.score_samples(X)) == len(X)

    def test_fit_predict_gives_the_filtered_target_to_the_estimator(self, frame_xy):
        """``fit_predict`` is a training pass: rows dropped and target shifted."""
        X, y = frame_xy
        pipeline = XYPipeline([
            ('sk', SpyTransformer()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget()), ('model', _FitPredictSpy()),
        ])

        pipeline.fit_predict(X, y)

        assert pipeline.named_steps['model'].fit_y_.tolist() == [2.0, 4.0, 5.0, 6.0, 7.0]

    def test_fit_predict_returns_the_estimator_output(self, frame_xy):
        """The output is the one of the final ``fit_predict``, on the 5 kept rows."""
        X, y = frame_xy
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', _FitPredictSpy())])

        np.testing.assert_array_equal(pipeline.fit_predict(X, y), np.arange(5))

    def test_fit_predict_without_target(self, frame_xy):
        """Without ``y``, the final ``fit_predict`` receives ``y=None``."""
        X, _y = frame_xy
        pipeline = XYPipeline([('sk', SpyTransformer()), ('model', _FitPredictSpy())])

        pipeline.fit_predict(X)

        assert pipeline.named_steps['model'].fit_y_ is None

    def test_score_drops_the_nan_targets(self, fitted, frame_xy):
        """Valeur d'or : R² = 0 — la moyenne apprise est notée sur ses propres lignes.

        The spy predicts the mean of the kept, shifted targets; scored against those
        same targets, the residual and total sums of squares are equal. A NaN left in
        the target would make sklearn's ``r2_score`` raise.
        """
        X, y = frame_xy

        assert fitted.score(X, y) == pytest.approx(0.0, abs=1e-12)

    def test_score_predicts_on_the_kept_rows(self, fitted, frame_xy):
        """At ``score``, the estimator predicts on the 5 dates with a target."""
        X, y = frame_xy

        fitted.score(X, y)

        assert fitted.named_steps['model'].predict_X_[-1].index.equals(X.index.delete(1))

    def test_score_gives_y_to_the_xy_step(self, fitted, frame_xy):
        """``score`` hands ``y`` to the XY step, not to the ordinary one."""
        X, y = frame_xy

        fitted.score(X, y)

        received = [_calls(fitted.named_steps[name], 'transform')[-1] is None for name in ('sk', 'xy')]
        assert received == [True, False]


# ==============================================================================
# Restauration de la structure pandas
# ==============================================================================


class TestPandasRestore:
    """A step returning arrays: the pandas structure of the input is put back."""

    @pytest.fixture
    def output(self, linear_xy):
        """``fit_transform`` of ``_ArrayXY`` on the pandas ``linear_xy``."""
        return XYPipeline([('array', _ArrayXY())]).fit_transform(*linear_xy)

    def test_x_gets_its_columns_back(self, output):
        """``X'`` is a ``DataFrame`` with the input columns."""
        assert list(output[0].columns) == ['a', 'b']

    def test_x_gets_its_index_back(self, output, linear_xy):
        """``X'`` has the input dates."""
        assert output[0].index.equals(linear_xy[0].index)

    def test_y_gets_its_name_back(self, output):
        """``y'`` is a ``Series`` named like the input target."""
        assert output[1].name == 't'

    def test_y_gets_its_index_back(self, output, linear_xy):
        """``y'`` has the input dates."""
        assert output[1].index.equals(linear_xy[1].index)

    def test_wider_output_is_named_by_the_steps(self, linear_xy):
        """Valeur d'or : noms de ``PolynomialFeatures(2)`` sur ``a``, ``b``."""
        X, y = linear_xy

        X_t = XYPipeline([('poly', PolynomialFeatures(degree=2))]).fit_transform(X, y)

        assert list(X_t.columns) == ['1', 'a', 'b', 'a^2', 'a b', 'b^2']

    def test_wider_output_without_names_gets_positions(self, linear_xy):
        """A wider output from a step without ``get_feature_names_out``: columns 0..n-1."""
        X, y = linear_xy

        X_t = XYPipeline([('twice', _DuplicateColumns())]).fit_transform(X, y)

        assert list(X_t.columns) == [0, 1, 2, 3]

    def test_rows_dropped_as_arrays_get_a_positional_index(self, frame_xy):
        """Fewer rows than at the input: ``X'`` and ``y'`` both get the positions 0..n-1.

        The labels of the kept rows are lost, but ``X'`` and ``y'`` still share their
        index. The target keeps its name.
        """
        X_t, y_t = XYPipeline([('drop', _ArrayDropXY())]).fit_transform(*frame_xy)

        assert (X_t.index.tolist(), y_t.index.tolist(), y_t.name) == ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 't')

    def test_restoration_can_be_disabled(self, linear_xy):
        """``preserve_dataframe=False`` leaves the arrays as they are."""
        _X_t, y_t = XYPipeline([('array', _ArrayXY())], preserve_dataframe=False).fit_transform(*linear_xy)

        assert isinstance(y_t, np.ndarray)


# ==============================================================================
# Non-mutation des entrées
# ==============================================================================


def _full_pipeline() -> XYPipeline:
    """``StandardScaler -> SpyXYTransformer(+1) -> DropNaNTarget -> SpyEstimator``."""
    return XYPipeline([
        ('scale', StandardScaler()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget()), ('model', SpyEstimator()),
    ])


def _transformer_pipeline() -> XYPipeline:
    """``StandardScaler -> SpyXYTransformer(+1) -> DropNaNTarget``, no estimator."""
    return XYPipeline([('scale', StandardScaler()), ('xy', SpyXYTransformer(1.0)), ('drop', DropNaNTarget())])


CALLS = {
    'fit': lambda X, y: _full_pipeline().fit(X, y),
    'fit_transform': lambda X, y: _transformer_pipeline().fit_transform(X, y),
    'transform': lambda X, y: _transformer_pipeline().fit(X.copy(), y.copy()).transform(X),
    'transform_with_target': lambda X, y: _transformer_pipeline().fit(X.copy(), y.copy()).transform(X, y),
    'predict': lambda X, y: _full_pipeline().fit(X.copy(), y.copy()).predict(X),
    'score': lambda X, y: _full_pipeline().fit(X.copy(), y.copy()).score(X, y),
}


class TestInputNotMutated:
    """No method changes the ``X`` or the ``y`` it is given."""

    @pytest.mark.parametrize('call', CALLS.values(), ids=CALLS.keys())
    def test_x_is_unchanged(self, call, frame_xy):
        """``X`` is equal to its deep copy taken before the call."""
        X, y = frame_xy
        X_before = X.copy(deep=True)

        call(X, y)

        pd.testing.assert_frame_equal(X, X_before)

    @pytest.mark.parametrize('call', CALLS.values(), ids=CALLS.keys())
    def test_y_is_unchanged(self, call, frame_xy):
        """``y`` is equal to its deep copy taken before the call."""
        X, y = frame_xy
        y_before = y.copy(deep=True)

        call(X, y)

        pd.testing.assert_series_equal(y, y_before)
