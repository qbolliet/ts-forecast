"""Fake sklearn estimators shared by the ``tsforecast`` tests.

Extracted from ``tests/unit/frequency/test_high_frequency_imputer.py``
(formerly ``_SpyEstimator`` / ``_FailingEstimator``, private to that
module), renamed without prefix for cross-cutting use.
"""
# Modules de base
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


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
