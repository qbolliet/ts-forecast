"""Backend built from a user function: ``CallableBackend``.

Plugs any forecasting model into ``FoundationModelAdapter`` through a function
``function(request) -> forecasts``; the capabilities of the model are constructor
parameters, so that they can be set and searched like any scikit-learn parameter.
"""
# Importation des modules
# Modules de base
from typing import Any, Callable, Optional, Sequence

# Modules de manipulation de données
import numpy as np
from sklearn.base import BaseEstimator

# tsforecast
from tsforecast.adapters.foundation.protocol import BackendCapabilities, ForecastRequest, ForecastResult


# Backend fondé sur une fonction de prévision
class CallableBackend(BaseEstimator):
    """Forecasting backend calling a user function on each request.

    The function receives a :class:`ForecastRequest` and returns a
    :class:`ForecastResult` or an array of shape ``(n_series, n_targets,
    prediction_length, n_levels)``. It must not keep state between calls. The
    capability parameters describe what the function handles: by default it
    receives past and known covariates, missing values and several targets, and
    must return any requested level.

    Args:
        function: Forecasting function, or None (the backend then cannot
            predict; useful as a placeholder).
        past_covariates: Whether the function reads covariates known over the
            context only.
        known_covariates: Whether it reads covariates known over the context and
            the horizon.
        missing_values: Whether it accepts ``NaN`` in the context and the covariates.
        multivariate: Whether it forecasts several targets of a series jointly.
        max_context: Longest context it reads, or None.
        max_prediction_length: Longest forecast it produces, or None.
        quantile_levels: Levels it was built for, or None for any level.
        interpolates_quantiles: Whether it returns any level within the range
            of ``quantile_levels``.

    Examples:
        >>> import numpy as np
        >>> def last_value(request):
        ...     last = [series[:, ~np.isnan(series[0])][:, -1] for series in request.context]
        ...     shape = (request.prediction_length, len(request.quantile_levels))
        ...     return np.stack([np.broadcast_to(value[:, None, None], (len(value),) + shape) for value in last])
        >>> backend = CallableBackend(last_value)
        >>> request = ForecastRequest(context=(np.array([[1.0, 2.0, np.nan]]),), prediction_length=2,
        ...                           quantile_levels=(0.5,))
        >>> backend.predict(request).quantiles[0, 0, :, 0].tolist()
        [2.0, 2.0]
    """

    # Initialisation
    def __init__(
        self,
        function: Optional[Callable[[ForecastRequest], Any]],
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
        self.function = function
        self.past_covariates = past_covariates
        self.known_covariates = known_covariates
        self.missing_values = missing_values
        self.multivariate = multivariate
        self.max_context = max_context
        self.max_prediction_length = max_prediction_length
        self.quantile_levels = quantile_levels
        self.interpolates_quantiles = interpolates_quantiles

    # Capacités assemblées depuis les paramètres
    @property
    def capabilities(self) -> BackendCapabilities:
        """Capabilities declared by the parameters.

        Raises:
            ValueError: If a length or the levels are invalid.
        """
        return BackendCapabilities(
            past_covariates=bool(self.past_covariates),
            known_covariates=bool(self.known_covariates),
            missing_values=bool(self.missing_values),
            multivariate=bool(self.multivariate),
            max_context=self.max_context,
            max_prediction_length=self.max_prediction_length,
            quantile_levels=None if self.quantile_levels is None else tuple(self.quantile_levels),
            interpolates_quantiles=bool(self.interpolates_quantiles),
        )

    # Prévision d'un lot de séries
    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Call the function on ``request``.

        Args:
            request: Series, number of steps and levels to forecast.

        Returns:
            The forecasts returned by the function, as a :class:`ForecastResult`.

        Raises:
            TypeError: If no function is set.
            ValueError: If the function returns an array without four dimensions.
        """
        # Fonction absente : aucun modèle à appeler
        if self.function is None:
            raise TypeError("CallableBackend has no function to call")
        output = self.function(request)
        # Résultat déjà au format du protocole
        if isinstance(output, ForecastResult):
            return output
        return ForecastResult(np.asarray(output, dtype=np.float64))
