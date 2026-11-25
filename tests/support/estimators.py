"""Test doubles shared by the ``tsforecast`` tests: estimators and transformers.

Estimators: ``SpyEstimator``, ``FailingEstimator``, ``ConstantEstimator`` and
``RecordingEstimator``. Transformers: the spies ``SpyTransformer`` (ordinary
sklearn step) and ``SpyXYTransformer`` (XY step), which log every call they
receive (``SpyCall``); the XY transformers ``LogTransformXY``, ``ScaleXY``,
``DropRowsXY`` and ``MixinScaleXY`` (minimal ``XYTransformerMixin`` subclass);
and ``FailingTransformer``, which raises at a chosen method. Reference adapters
built on ``tsforecast.adapters._base``: ``RowRegressorAdapter`` (row-wise
regressor) and ``LastValueAdapter`` (local per-entity model), on which the
adapter contract battery is validated.
"""
# Modules de base
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from tsforecast.adapters._base import (
    check_adapter_input,
    import_optional,
    restore_row_order,
    steps_between,
)
from tsforecast.xy import XYTransformerMixin


class SpyEstimator(BaseEstimator, RegressorMixin):
    """Spy estimator, NaN-tolerant, predicting a constant.

    It records ``fit_X_``, ``fit_y_`` and the list ``predict_X_`` of
    prediction frames, which makes the central invariant (never a
    fit -> predict degradation) and the scale of the data passed
    measurable. The class counter ``n_fits`` measures the "a single fit
    per (stage, variable)" rule.
    """

    n_fits = 0

    def __init__(self, constant: float = 1.0):
        self.constant = constant

    def fit(self, X, y):
        """Record the training set and the target mean."""
        type(self)._record_fit()
        self.fit_X_ = X.copy()
        self.fit_y_ = y.copy()
        self.predict_X_ = []
        values = np.asarray(y, dtype=float)
        finite = values[~np.isnan(values)]
        self.mean_ = float(finite.mean()) if finite.size else 0.0
        return self

    def predict(self, X):
        """Record the prediction frame and return the learnt mean."""
        if not hasattr(self, 'predict_X_'):
            self.predict_X_ = []
        self.predict_X_.append(X.copy())
        return np.full(len(X), self.mean_)

    @classmethod
    def _record_fit(cls):
        """Increment the class fit counter."""
        SpyEstimator.n_fits += 1


class FailingEstimator(BaseEstimator, RegressorMixin):
    """Estimator whose fit always fails."""

    def fit(self, X, y):
        """Always raise, to exercise the interpolation fallback."""
        raise RuntimeError('deliberate fit failure')

    def predict(self, X):
        """Never reached: the fit has already failed."""
        raise RuntimeError('deliberate predict failure')


class ConstantEstimator(BaseEstimator, RegressorMixin):
    """Estimator that always predicts the same constant, whatever ``X``.

    It fully ignores ``X`` and ``y`` at ``fit``: unlike
    :class:`SpyEstimator` (mean of ``y``), the constant is set at
    ``__init__``, which makes the output a trivial golden value known in
    advance - useful to isolate the behaviour of an upstream component
    (windowing, scaling) from that of the estimator itself.
    """

    def __init__(self, constant: float = 0.0):
        self.constant = constant

    def fit(self, X, y):
        """Learn nothing: only the presence of ``fit`` satisfies the sklearn contract."""
        return self

    def predict(self, X):
        """Return ``constant``, repeated for each row of ``X``."""
        return np.full(len(X), self.constant)


class RecordingEstimator(BaseEstimator, RegressorMixin):
    """Wrapper delegating to a real estimator while recording what it sees.

    Where :class:`SpyEstimator` predicts a constant, this double keeps the
    predictions of a real model - needed when a later stage reads them - and
    still exposes ``fit_X_`` / ``fit_y_`` and the list ``predict_X_``, so a
    test reads the exact training set through the public plan
    (``step.model``) rather than through a private hook.

    Args:
        estimator: Estimator the calls are delegated to. Cloned at ``fit``,
            the parameter itself is never fitted.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> import pandas as pd
        >>> X = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        >>> model = RecordingEstimator(LinearRegression()).fit(X, X['x'] * 2)
        >>> model.fit_y_.tolist()
        [2.0, 4.0, 6.0]
        >>> [round(float(value), 6) for value in model.predict(X)]
        [2.0, 4.0, 6.0]
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        """Record the training set, then fit a clone of the wrapped estimator."""
        self.fit_X_ = X.copy()
        self.fit_y_ = y.copy()
        self.predict_X_ = []
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        """Record the prediction frame and delegate the prediction."""
        if not hasattr(self, 'predict_X_'):
            self.predict_X_ = []
        self.predict_X_.append(X.copy())
        return self.estimator_.predict(X)


# ==============================================================================
# Espions de transformation
# ==============================================================================


@dataclass(frozen=True)
class SpyCall:
    """One call received by a spy transformer.

    Attributes:
        method: Name of the method called (``'fit'``, ``'transform'``...).
        X_index: Copy of the row index of ``X`` (``None`` for an array).
        X_shape: Shape of ``X``.
        y: Copy of the target received, ``None`` when no target was passed.
    """

    method: str
    X_index: Any
    X_shape: Tuple[int, ...]
    y: Any


def _record(spy: Any, method: str, X: Any, y: Any = None) -> None:
    """Append a :class:`SpyCall` describing ``(X, y)`` to ``spy.calls_``.

    Args:
        spy: Spy transformer whose journal ``calls_`` receives the call.
        method: Name of the method called.
        X: Features received.
        y: Target received, or None.
    """
    # Journal créé au premier appel : un clone neuf part d'un journal vide
    if not hasattr(spy, 'calls_'):
        spy.calls_ = []
    index = X.index.copy() if hasattr(X, 'index') else None
    target = None if y is None else (y.copy() if hasattr(y, 'copy') else np.array(y))
    spy.calls_.append(SpyCall(method, index, np.shape(X), target))


class SpyTransformer(BaseEstimator, TransformerMixin):
    """Ordinary sklearn transformer (identity) that logs every call it receives.

    It behaves like any sklearn step: ``transform(X)`` takes no target, and the
    ``fit_transform`` inherited from ``TransformerMixin`` calls ``fit(X, y)``
    then ``transform(X)``. Each call is appended to ``calls_`` as a
    :class:`SpyCall`; ``transform`` returns a copy of ``X``.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'x': [1.0, 2.0]})
        >>> spy = SpyTransformer()
        >>> spy.fit_transform(X, pd.Series([5.0, 6.0]))['x'].tolist()
        [1.0, 2.0]
        >>> [(call.method, call.y is None) for call in spy.calls_]
        [('fit', False), ('transform', True)]
    """

    def fit(self, X, y=None):
        """Log the call and learn the number of features."""
        _record(self, 'fit', X, y)
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X):
        """Log the call and return a copy of ``X``."""
        check_is_fitted(self)
        _record(self, 'transform', X)
        return X.copy()


class SpyXYTransformer(BaseEstimator, XYTransformerMixin):
    """XY transformer that logs its calls, keeps ``X`` and shifts ``y``.

    ``transform(X, y)`` returns ``(X, y + y_offset)`` (copies), ``transform(X)``
    returns ``X``: a downstream step that receives ``y + y_offset`` proves
    that the target went through this step. Each call is appended to
    ``calls_`` as a :class:`SpyCall`.

    Args:
        y_offset: Constant added to the target.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'x': [1.0, 2.0]})
        >>> X_t, y_t = SpyXYTransformer(y_offset=10.0).fit_transform(X, pd.Series([1.0, 2.0]))
        >>> y_t.tolist()
        [11.0, 12.0]
    """

    def __init__(self, y_offset: float = 0.0):
        self.y_offset = y_offset

    def fit(self, X, y=None):
        """Log the call, then fit through ``XYTransformerMixin.fit``."""
        _record(self, 'fit', X, y)
        return super().fit(X, y)

    def _fit(self, X, y):
        """Learn the number of features."""
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X, y=None):
        """Log the call, then transform through ``XYTransformerMixin.transform``."""
        _record(self, 'transform', X, y)
        return super().transform(X, y)

    def _transform(self, X, y=None):
        """Return ``X``, and ``y + y_offset`` when a target is given."""
        X_t = X.copy()
        if y is None:
            return X_t
        return X_t, y + self.y_offset


# ==============================================================================
# Transformateurs XY : déclarés par XYTransformerMixin, méthodes publiques surchargées
# ==============================================================================


class LogTransformXY(BaseEstimator, XYTransformerMixin):
    """XY transformer applying ``log1p(|.|)`` to ``X`` and ``y``.

    The absolute value is taken first: the transform accepts negative values,
    but ``inverse_transform`` (``expm1``) gives back ``|X|`` and ``|y|``, so it
    is an exact inverse on non-negative data only.

    Examples:
        >>> import numpy as np
        >>> X_t, y_t = LogTransformXY().fit_transform(np.array([[0.0], [-1.0]]), np.array([0.0, 1.0]))
        >>> [round(float(value), 6) for value in X_t.ravel()]
        [0.0, 0.693147]
        >>> [round(float(value), 6) for value in LogTransformXY().inverse_transform(X_t).ravel()]
        [0.0, 1.0]
    """

    def fit(self, X, y=None):
        """Learn nothing; mark the step as fitted."""
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """Return ``log1p(|X|)``, and ``log1p(|y|)`` when a target is given."""
        X_t = np.log1p(np.abs(X))
        if y is not None:
            return X_t, np.log1p(np.abs(y))
        return X_t

    def fit_transform(self, X, y=None):
        """Fit, then transform ``X`` and ``y`` together."""
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        """Return ``expm1(X)``, and ``expm1(y)`` when a target is given."""
        X_inv = np.expm1(X)
        if y is not None:
            return X_inv, np.expm1(y)
        return X_inv


class ScaleXY(BaseEstimator, XYTransformerMixin):
    """XY transformer multiplying ``X`` and ``y`` by ``scale_factor``.

    Args:
        scale_factor: Factor applied to both ``X`` and ``y``.

    Examples:
        >>> import numpy as np
        >>> X_t, y_t = ScaleXY(scale_factor=3.0).fit_transform(np.array([[1.0]]), np.array([2.0]))
        >>> float(X_t[0, 0]), float(y_t[0])
        (3.0, 6.0)
    """

    def __init__(self, scale_factor=2.0):
        self.scale_factor = scale_factor

    def fit(self, X, y=None):
        """Learn nothing; mark the step as fitted."""
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """Return ``scale_factor * X``, and ``scale_factor * y`` when a target is given."""
        X_t = X * self.scale_factor
        if y is not None:
            return X_t, y * self.scale_factor
        return X_t

    def fit_transform(self, X, y=None):
        """Fit, then transform ``X`` and ``y`` together."""
        return self.fit(X, y).transform(X, y)


class DropRowsXY(BaseEstimator, XYTransformerMixin):
    """XY transformer dropping the rows whose target is negative.

    Rows are selected with the boolean mask ``y >= 0``: on pandas inputs,
    ``X`` and ``y`` keep their own index labels. Without ``y``, ``X`` goes
    through untouched.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        >>> X_kept, y_kept = DropRowsXY().fit_transform(X, pd.Series([1.0, -1.0, 2.0]))
        >>> X_kept.index.tolist(), y_kept.tolist()
        ([0, 2], [1.0, 2.0])
    """

    def fit(self, X, y=None):
        """Learn nothing; mark the step as fitted."""
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """Drop the rows with ``y < 0`` from ``X`` and ``y``; return ``X`` without ``y``."""
        if y is not None:
            mask = y >= 0
            return X[mask], y[mask]
        return X

    def fit_transform(self, X, y=None):
        """Fit, then transform ``X`` and ``y`` together."""
        return self.fit(X, y).transform(X, y)


class MixinScaleXY(BaseEstimator, XYTransformerMixin):
    """Minimal ``XYTransformerMixin`` subclass: ``(X, y) -> (k * X, k * y)``.

    Only ``_fit`` and ``_transform`` are written; ``fit``, ``transform`` and
    ``fit_transform`` come from the mixin, and ``_inverse_transform`` is left
    unimplemented. ``_fit`` learns nothing and sets no learnt attribute: the
    double is the bare contract of the mixin, whose ``fit`` alone marks the
    instance as fitted.

    Args:
        k: Factor applied to both ``X`` and ``y``.

    Examples:
        >>> import numpy as np
        >>> X_t, y_t = MixinScaleXY(k=3.0).fit_transform(np.array([[1.0]]), np.array([2.0]))
        >>> float(X_t[0, 0]), float(y_t[0])
        (3.0, 6.0)
    """

    def __init__(self, k: float = 2.0):
        self.k = k

    def _fit(self, X, y):
        """Learn nothing."""
        return self

    def _transform(self, X, y=None):
        """Return ``k * X``, and ``k * y`` when a target is given."""
        if y is None:
            return X * self.k
        return X * self.k, y * self.k


# ==============================================================================
# Transformateur en échec ciblé
# ==============================================================================


class FailingTransformer(BaseEstimator, TransformerMixin):
    """Centring transformer (``X - mean``) that raises at the method named by ``fail_on``.

    It returns a ``DataFrame`` that keeps the index of its input. Combined with
    ``PanelwiseTransformer(entity_kwargs={('DE',): {'fail_on': 'fit'}})``, the
    failure hits one entity only.

    Args:
        fail_on: ``'fit'``, ``'transform'``, ``'inverse_transform'`` or None (never fails).
        error: Exception class raised, with the message ``'deliberate failure'``.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1.0, 3.0]})
        >>> FailingTransformer().fit_transform(X)['a'].tolist()
        [-1.0, 1.0]
        >>> FailingTransformer(fail_on='fit').fit(X)
        Traceback (most recent call last):
            ...
        RuntimeError: deliberate failure
    """

    def __init__(self, fail_on: Optional[str] = None, error: type = RuntimeError):
        self.fail_on = fail_on
        self.error = error

    def _maybe_fail(self, method: str) -> None:
        # Échec délibéré à la méthode désignée
        if self.fail_on == method:
            raise self.error('deliberate failure')

    def fit(self, X, y=None):
        """Learn the column means, unless ``fail_on='fit'``."""
        self._maybe_fail('fit')
        self.mean_ = X.mean()
        return self

    def transform(self, X):
        """Subtract the learnt means, unless ``fail_on='transform'``."""
        check_is_fitted(self)
        self._maybe_fail('transform')
        return X - self.mean_

    def inverse_transform(self, X):
        """Add the learnt means back, unless ``fail_on='inverse_transform'``."""
        check_is_fitted(self)
        self._maybe_fail('inverse_transform')
        return X + self.mean_


# Adapter de référence : régresseur ligne à ligne bâti sur le socle des adapters
class RowRegressorAdapter(RegressorMixin, BaseEstimator):
    """Reference adapter: a row-wise regressor wrapped like a third-party model.

    The regressor is cloned into ``regressor_`` (the wrapped object is never
    fitted) and sees the rows sorted by index, as a third-party package
    reordering its input would; the predictions are realigned on the rows of
    ``X``. The default regressor is imported lazily from
    ``sklearn.linear_model`` through ``import_optional`` (extra
    ``adapters-reference``), which makes the missing-package path testable.
    """

    def __init__(self, regressor: Any = None):
        self.regressor = regressor

    def fit(self, X, y):
        """Fit a clone of the regressor on the rows sorted by index."""
        linear_model = import_optional('sklearn.linear_model', 'adapters-reference')
        X, y, layout = check_adapter_input(self, X, y, reset=True)
        template = self.regressor if self.regressor is not None else linear_model.LinearRegression()
        self.regressor_ = clone(template)
        # Lignes triées par entité puis par date, comme un paquet tiers qui réordonne
        order = np.lexsort((layout.dates.asi8, layout.codes))
        self.regressor_.fit(X.to_numpy()[order], y.to_numpy()[order])
        self.freq_ = layout.freq
        return self

    def predict(self, X):
        """Predict the rows sorted by index, then restore the order of ``X``."""
        check_is_fitted(self)
        X, _, _ = check_adapter_input(self, X, reset=False, freq=self.freq_)
        X_sorted = X.sort_index()
        values = self.regressor_.predict(X_sorted.to_numpy())
        return restore_row_order(values, X_sorted.index, X.index)


# Adapter de référence : modèle local par entité, prévision à travers les pas de grille
class LastValueAdapter(RegressorMixin, BaseEstimator):
    """Reference adapter: a local model extrapolating the last value of each entity.

    The prediction of a row is the last training target of its entity plus
    ``drift`` per grid step between that entity's last training date and the
    row's date (``steps_between``). An entity unknown at fit cannot be
    predicted.

    Args:
        drift: Increment per grid step; must be a real number.
    """

    def __init__(self, drift: float = 0.0):
        self.drift = drift

    def fit(self, X, y):
        """Record the last target and the last date of each entity."""
        if isinstance(self.drift, bool) or not isinstance(self.drift, (int, float)):
            raise ValueError(f'drift must be a real number, got {self.drift!r}')
        X, y, layout = check_adapter_input(self, X, y, reset=True)
        if layout.freq is None:
            raise ValueError('The frequency of the training dates cannot be read')
        # Dernière ligne de chaque entité : tri par code puis par date
        order = np.lexsort((layout.dates.asi8, layout.codes))
        sorted_codes = layout.codes[order]
        last_rows = order[np.r_[np.flatnonzero(np.diff(sorted_codes)), len(order) - 1]]
        self.layout_ = layout
        self.last_values_ = y.to_numpy()[last_rows]
        return self

    def predict(self, X):
        """Extrapolate each row from the last training value of its entity."""
        check_is_fitted(self)
        X, _, layout = check_adapter_input(self, X, reset=False, freq=self.layout_.freq)
        codes = self.layout_.match_entities(layout)
        if (codes < 0).any():
            unknown = sorted({layout.entities[code] for code in np.unique(layout.codes[codes < 0])})
            raise ValueError(f'Entities unknown at fit: {unknown}')
        steps = steps_between(self.layout_.last_dates()[codes], layout.dates, self.layout_.freq)
        return self.last_values_[codes] + self.drift * steps
