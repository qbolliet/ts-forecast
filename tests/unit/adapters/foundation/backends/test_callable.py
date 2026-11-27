"""Unit tests of ``tsforecast.adapters.foundation.backends.callable.CallableBackend``.

Capabilities assembled from the parameters, outputs returned as arrays or as
``ForecastResult``, missing function, invalid output, and the backend searched by
``GridSearchCV`` inside ``FoundationModelAdapter`` (function and capability
parameters).
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from tests.support.adapter_contract import default_splitter
from tests.support.datasets import build_adapter_series
from tsforecast.adapters import CallableBackend, FoundationModelAdapter
from tsforecast.adapters.foundation import ForecastRequest, ForecastResult

# Requête d'une série d'une cible, deux pas, deux niveaux
REQUEST = ForecastRequest(context=(np.array([[1.0, 3.0]]),), prediction_length=2, quantile_levels=(0.1, 0.9))


def last_value(request):
    """Last context value of each series, at every step and level."""
    shape = (request.prediction_length, len(request.quantile_levels))
    return np.stack([np.broadcast_to(series[:, -1][:, None, None], (request.n_targets,) + shape) for series in request.context])


def zero(request):
    """Zero at every step and level."""
    return np.zeros((request.n_series, request.n_targets, request.prediction_length, len(request.quantile_levels)))


class TestCallableBackend:
    """Function plugged in as a backend."""

    def test_capabilities(self):
        assert CallableBackend(last_value).capabilities.known_covariates
        capabilities = CallableBackend(
            last_value, known_covariates=False, max_context=5, quantile_levels=[0.1, 0.9], interpolates_quantiles=False
        ).capabilities
        assert not capabilities.known_covariates and capabilities.max_context == 5
        assert capabilities.quantile_levels == (0.1, 0.9) and not capabilities.interpolates_quantiles
        with pytest.raises(ValueError):
            CallableBackend(last_value, max_context=0).capabilities

    def test_outputs(self):
        np.testing.assert_array_equal(CallableBackend(last_value).predict(REQUEST).quantiles, np.full((1, 1, 2, 2), 3.0))
        result = ForecastResult(np.ones((1, 1, 2, 2)))
        assert CallableBackend(lambda request: result).predict(REQUEST) is result
        with pytest.raises(TypeError, match='no function'):
            CallableBackend(None).predict(REQUEST)
        with pytest.raises(ValueError, match='shape'):
            CallableBackend(lambda request: np.zeros(3)).predict(REQUEST)

    def test_searched_inside_the_adapter(self):
        X, y = build_adapter_series()
        adapter = FoundationModelAdapter(backend=CallableBackend(zero))
        assert clone(adapter).backend.function is zero
        search = GridSearchCV(
            adapter,
            {'backend__function': [zero, last_value], 'backend__known_covariates': [True, False]},
            cv=default_splitter('series'),
            error_score='raise',
        ).fit(X, y)
        assert set(search.best_params_) == {'backend__function', 'backend__known_covariates'}
        assert search.best_estimator_.covariates_ in ('known', 'past')
