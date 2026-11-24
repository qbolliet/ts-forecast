"""Tests of the transformer doubles of ``tests.support.estimators``.

Covers the call journal of ``SpyTransformer`` and ``SpyXYTransformer``
(method, index, shape and target of each call, fresh journal after
``clone``), the golden outputs of ``MixinScaleXY``, ``LogTransformXY``,
``ScaleXY`` and ``DropRowsXY``, and the targeted failures of
``FailingTransformer``.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from tests.support.estimators import (
    DropRowsXY,
    FailingTransformer,
    LogTransformXY,
    MixinScaleXY,
    ScaleXY,
    SpyTransformer,
    SpyXYTransformer,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Three dated rows, one column ``x`` = 1, 2, 3."""
    return pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=pd.date_range('2024-01-31', periods=3, freq='ME'))


@pytest.fixture
def target(frame) -> pd.Series:
    """Target 10, 20, 30 on the dates of ``frame``."""
    return pd.Series([10.0, 20.0, 30.0], index=frame.index, name='t')


# =============================================================================
# Journal des espions
# =============================================================================
class TestSpyJournal:
    """Each call is logged with its method, the index and shape of ``X`` and the target."""

    def test_sklearn_spy_logs_fit_then_transform(self, frame, target):
        """The inherited ``fit_transform`` gives a ``fit`` with ``y`` then a ``transform`` without."""
        spy = SpyTransformer()

        spy.fit_transform(frame, target)

        assert [(call.method, call.y is None) for call in spy.calls_] == [('fit', False), ('transform', True)]

    def test_index_and_shape_are_recorded(self, frame, target):
        """The call keeps a copy of the index and the shape of ``X``."""
        spy = SpyTransformer().fit(frame, target)

        assert (spy.calls_[0].X_index.equals(frame.index), spy.calls_[0].X_shape) == (True, (3, 1))

    def test_array_input_records_no_index(self, frame):
        """An array has no index: ``X_index`` is None."""
        spy = SpyTransformer().fit(frame.to_numpy())

        assert spy.calls_[0].X_index is None

    def test_recorded_target_is_a_copy(self, frame, target):
        """Changing the target after the call leaves the journal intact."""
        spy = SpyTransformer().fit(frame, target)
        target.iloc[0] = -1.0

        assert spy.calls_[0].y.tolist() == [10.0, 20.0, 30.0]

    def test_sklearn_spy_transform_returns_a_copy(self, frame):
        """``transform`` is the identity, on a copy of ``X``."""
        spy = SpyTransformer().fit(frame)

        X_t = spy.transform(frame)

        assert X_t.equals(frame) and X_t is not frame

    def test_xy_spy_shifts_the_target(self, frame, target):
        """Valeur d'or : y + 5 = 15, 25, 35 ; X inchangé."""
        X_t, y_t = SpyXYTransformer(y_offset=5.0).fit_transform(frame, target)

        assert y_t.tolist() == [15.0, 25.0, 35.0]

    def test_xy_spy_logs_the_target_at_transform(self, frame, target):
        """The XY spy receives ``y`` at ``fit`` and at ``transform``."""
        spy = SpyXYTransformer()

        spy.fit_transform(frame, target)

        assert [(call.method, call.y is None) for call in spy.calls_] == [('fit', False), ('transform', False)]

    def test_xy_spy_without_target_returns_x(self, frame, target):
        """Without ``y``, ``transform`` returns ``X`` alone."""
        spy = SpyXYTransformer().fit(frame, target)

        assert isinstance(spy.transform(frame), pd.DataFrame)

    @pytest.mark.parametrize('spy', [SpyTransformer(), SpyXYTransformer(y_offset=1.0)], ids=['sklearn', 'xy'])
    def test_clone_starts_an_empty_journal(self, spy, frame, target):
        """A clone has no journal: the calls of the original are not copied."""
        spy.fit(frame, target)

        assert not hasattr(clone(spy), 'calls_')


# =============================================================================
# Transformateurs XY
# =============================================================================
class TestXYDoubles:
    """Golden outputs of the XY doubles."""

    def test_mixin_scale_multiplies_both(self, frame, target):
        """Valeur d'or : k = 3 → X = 3, 6, 9 et y = 30, 60, 90."""
        X_t, y_t = MixinScaleXY(k=3.0).fit_transform(frame, target)

        assert (X_t['x'].tolist(), y_t.tolist()) == ([3.0, 6.0, 9.0], [30.0, 60.0, 90.0])

    def test_scale_xy_multiplies_both(self, frame, target):
        """Valeur d'or : facteur 2 → y = 20, 40, 60."""
        _X_t, y_t = ScaleXY(scale_factor=2.0).fit_transform(frame, target)

        assert y_t.tolist() == [20.0, 40.0, 60.0]

    def test_log_round_trip_on_positive_values(self, frame, target):
        """``inverse_transform`` gives back non-negative inputs."""
        step = LogTransformXY()
        X_t, y_t = step.fit_transform(frame, target)

        X_inv, y_inv = step.inverse_transform(X_t, y_t)

        np.testing.assert_allclose(y_inv.to_numpy(), target.to_numpy())

    def test_log_inverse_loses_the_sign(self):
        """A negative input comes back as its absolute value."""
        step = LogTransformXY()

        X_inv = step.inverse_transform(step.fit_transform(np.array([[-2.0]])))

        np.testing.assert_allclose(X_inv, [[2.0]])

    def test_drop_rows_keeps_non_negative_targets(self, frame):
        """Valeur d'or : la cible -1 (2ᵉ ligne) disparaît, avec sa ligne de X."""
        X_kept, _y_kept = DropRowsXY().fit_transform(frame, pd.Series([1.0, -1.0, 2.0], index=frame.index))

        assert X_kept.index.equals(frame.index[[0, 2]])


# =============================================================================
# Échecs ciblés
# =============================================================================
class TestFailingTransformer:
    """The failure happens at the method named by ``fail_on``, and only there."""

    @pytest.mark.parametrize('method', ['fit', 'transform', 'inverse_transform'])
    def test_fails_at_the_named_method(self, method, frame):
        """``fail_on=method`` raises ``'deliberate failure'`` at that method."""
        step = FailingTransformer(fail_on=method)
        if method != 'fit':
            step.mean_ = frame.mean()

        with pytest.raises(RuntimeError, match='deliberate failure'):
            getattr(step, method)(frame)

    def test_error_class_is_configurable(self, frame):
        """``error`` sets the class of the exception raised."""
        with pytest.raises(KeyError, match='deliberate failure'):
            FailingTransformer(fail_on='fit', error=KeyError).fit(frame)

    def test_round_trip_without_failure(self, frame):
        """Without ``fail_on``, centring then de-centring gives back ``X``."""
        step = FailingTransformer().fit(frame)

        assert step.inverse_transform(step.transform(frame)).equals(frame)
