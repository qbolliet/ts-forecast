"""Fake forecasting backends of the ``FoundationModelAdapter`` tests, without network nor third-party package.

``DriftBackend`` forecasts the last observed value plus a drift per step, plus a
multiple of the sum of every known covariate value it receives after the context
(a model that would read the future if the adapter gave it), with tunable
capabilities; ``RecordingBackend`` records the requests it receives;
``counting_loader`` builds a loading function that counts its calls.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator

from tsforecast.adapters.foundation.protocol import BackendCapabilities, ForecastRequest, ForecastResult


def _last_observed(context: np.ndarray) -> np.ndarray:
    """Last finite value of each target row of a ``(n_targets, n_steps)`` context."""
    last = np.empty(context.shape[0])
    for target, values in enumerate(context):
        last[target] = values[np.isfinite(values)][-1]
    return last


class DriftBackend(BaseEstimator):
    """Deterministic fake model, its capabilities set by its parameters.

    At step ``j`` of a forecast of a series, for the level ``q``:
    ``last + drift·j + coef·S_known + past_coef·S_past + spread·(q − 0.5)``,
    ``last`` being the last observed value of the target, ``S_known`` the sum of
    the non-missing known covariates over every forecast step of the request
    and ``S_past`` the sum of the non-missing past covariates on the last step of
    the context.

    Args:
        drift: Increment per step.
        coef: Weight of the known covariates after the context.
        past_coef: Weight of the past covariates of the last context step.
        spread: Distance between the levels.
        past_covariates: Declared capability.
        known_covariates: Declared capability.
        missing_values: Declared capability.
        multivariate: Declared capability.
        max_context: Declared capability.
        max_prediction_length: Declared capability.
        quantile_levels: Declared capability.
        interpolates_quantiles: Declared capability; when False and
            ``quantile_levels`` is set, a level that is not native raises
            ``ValueError``.
    """

    def __init__(
        self,
        drift: float = 0.5,
        coef: float = 0.0,
        past_coef: float = 0.0,
        spread: float = 1.0,
        *,
        past_covariates: bool = True,
        known_covariates: bool = True,
        missing_values: bool = True,
        multivariate: bool = True,
        max_context: Optional[int] = None,
        max_prediction_length: Optional[int] = None,
        quantile_levels: Optional[Sequence[float]] = None,
        interpolates_quantiles: bool = True,
    ):
        self.drift = drift
        self.coef = coef
        self.past_coef = past_coef
        self.spread = spread
        self.past_covariates = past_covariates
        self.known_covariates = known_covariates
        self.missing_values = missing_values
        self.multivariate = multivariate
        self.max_context = max_context
        self.max_prediction_length = max_prediction_length
        self.quantile_levels = quantile_levels
        self.interpolates_quantiles = interpolates_quantiles

    @property
    def capabilities(self) -> BackendCapabilities:
        """Capabilities declared by the parameters."""
        return BackendCapabilities(
            past_covariates=self.past_covariates,
            known_covariates=self.known_covariates,
            missing_values=self.missing_values,
            multivariate=self.multivariate,
            max_context=self.max_context,
            max_prediction_length=self.max_prediction_length,
            quantile_levels=None if self.quantile_levels is None else tuple(self.quantile_levels),
            interpolates_quantiles=self.interpolates_quantiles,
        )

    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Forecast every series of the request with the drift formula."""
        levels = np.array(request.quantile_levels)
        if self.quantile_levels is not None and not self.interpolates_quantiles:
            native = np.array(self.quantile_levels)
            if not all(np.isclose(native, level).any() for level in levels):
                raise ValueError(f'Level not native: {levels.tolist()} not in {native.tolist()}')
        steps = np.arange(1, request.prediction_length + 1)
        forecasts = []
        for position, context in enumerate(request.context):
            shift = 0.0
            if request.known_covariates is not None:
                shift += self.coef * np.nansum(request.known_covariates[position][:, context.shape[1]:])
            if request.past_covariates is not None:
                shift += self.past_coef * np.nansum(request.past_covariates[position][:, -1])
            paths = _last_observed(context)[:, None] + self.drift * steps[None, :] + shift
            forecasts.append(paths[:, :, None] + self.spread * (levels - 0.5))
        return ForecastResult(np.stack(forecasts))


class RecordingBackend:
    """Backend delegating to another one and recording every request it receives.

    Args:
        inner: Backend that computes the forecasts.

    Attributes:
        requests: Requests received, in order.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.requests: List[ForecastRequest] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        """Capabilities of the inner backend."""
        return self.inner.capabilities

    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Record the request, then forecast it with the inner backend."""
        self.requests.append(request)
        return self.inner.predict(request)


def counting_loader(calls: List[tuple], make: Callable[[], Any] = DriftBackend) -> Callable[..., Any]:
    """Return a loading function that appends ``(model_id, device, kwargs)`` to ``calls`` at each load.

    Args:
        calls: List receiving one entry per load.
        make: Builds the loaded backend.

    Returns:
        Function ``(model_id, *, device, **kwargs) -> backend``.
    """

    def load(model_id: str, *, device: str, **kwargs: Any) -> Any:
        calls.append((model_id, device, kwargs))
        return make()

    return load
