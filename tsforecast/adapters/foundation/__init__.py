"""Forecasting foundation models as scikit-learn regressors: ``FoundationModelAdapter`` and its backends.

Re-exports ``FoundationModelAdapter``, the backend protocol (``ForecastBackend``,
``BackendCapabilities``, ``ForecastRequest``, ``ForecastResult``), ``CallableBackend``
(any model plugged in through a function), and the registry functions
``register_backend``, ``available_backends`` and ``clear_model_cache``. No
third-party package is imported until a model is loaded.
"""
from tsforecast.adapters.foundation.protocol import (
    BackendCapabilities,
    ForecastBackend,
    ForecastRequest,
    ForecastResult,
)
from tsforecast.adapters.foundation.backends import CallableBackend
from tsforecast.adapters.foundation.registry import available_backends, clear_model_cache, register_backend
from tsforecast.adapters.foundation.adapter import FoundationModelAdapter

__all__ = [
    "BackendCapabilities",
    "CallableBackend",
    "FoundationModelAdapter",
    "ForecastBackend",
    "ForecastRequest",
    "ForecastResult",
    "available_backends",
    "clear_model_cache",
    "register_backend",
]
