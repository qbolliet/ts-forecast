"""Backend of the t0 foundation models of the tfc-t0 package: ``T0Backend``.

``load_t0`` loads ``theforecastingcompany/t0-*`` models with ``T0Forecaster``
(several targets, past and known covariates, any quantile level) and
``T0Backend.predict`` translates a request into calls of its ``predict`` method on
a batch of ``TimeSeries``, in which past covariates are further variates of the
context. tfc-t0 and torch are imported only when a model is loaded or called. Code
and weights: Apache-2.0.
"""
# Importation des modules
# Modules de base
from types import ModuleType
from typing import Any, Tuple

# Modules de manipulation de données
import numpy as np

# tsforecast
from tsforecast.adapters._base import import_optional
from tsforecast.adapters.foundation.backends._tensors import to_numpy, to_tensor
from tsforecast.adapters.foundation.protocol import BackendCapabilities, ForecastRequest, ForecastResult

# Extra installant tfc-t0 et torch
_EXTRA = "adapters-t0"
# Préfixes des identifiants Hugging Face servis par le backend
T0_PREFIXES: Tuple[str, ...] = ("theforecastingcompany/t0-",)


# Backend des modèles t0
class T0Backend:
    """Loaded t0 model following the backend protocol.

    Each series is a ``TimeSeries`` of the package: its targets, followed by its
    past covariates as further variates of the context (the model forecasts them
    jointly and the backend drops their forecasts), and its known covariates as
    ``future_covariates`` (context and forecast steps, ``NaN`` for an unknown
    value). The series of a request are batched with ``TimeSeries.batch``,
    right-aligned, the shorter ones padded: a forecast is that of the series
    alone, up to rounding. Beyond one forward pass (``max_horizon``, 1 024
    steps) the model continues auto-regressively over a window as wide as the
    batch, so series of different context lengths are then forecast in separate
    calls. There, the forecast of a level also depends on the other levels
    requested (by a few percent of the 0.1-0.9 interval for these two levels,
    by rounding only for the median), so ``predict_quantiles`` may differ from
    ``predict``.

    The options of the request are passed to ``predict``. Its defaults apply:
    ``symmetric_averaging``, ``clip_nonnegative``, ``snap_integers`` and
    ``trim_leading_missing`` are off (the package recommends enabling all four),
    and ``drop_constant_covariates=True`` ignores a known covariate that is
    constant over the context of a series. The package replaces a non-finite
    forecast with ``0`` and only logs a warning. Inference has no randomness.

    Licence: the t0 weights and the tfc-t0 package are Apache-2.0.

    Args:
        forecaster: Loaded ``T0Forecaster``, in evaluation mode.

    Attributes:
        capabilities: What the model accepts: covariates, missing values, several
            targets, any context length, horizon and quantile level (levels
            beyond the trained ones follow exponential tails).

    Examples:
        >>> capabilities = T0Backend(forecaster=None).capabilities
        >>> capabilities.known_covariates, capabilities.multivariate, capabilities.quantile_levels
        (True, True, None)
    """

    # Initialisation depuis un modèle chargé
    def __init__(self, forecaster: Any):
        self.forecaster = forecaster
        # Ni contexte, ni horizon, ni niveaux bornés
        self.capabilities = BackendCapabilities(
            past_covariates=True,
            known_covariates=True,
            missing_values=True,
            multivariate=True,
        )

    # Entrée du modèle pour une série
    @staticmethod
    def _time_series(request: ForecastRequest, position: int, t0: ModuleType, torch: ModuleType) -> Any:
        """Build the ``TimeSeries`` of one series of a request.

        Args:
            request: Series to forecast.
            position: Position of the series in the request.
            t0: The imported ``t0`` module.
            torch: The imported ``torch`` module.

        Returns:
            A ``TimeSeries`` holding the targets, then the past covariates, and
            the known covariates when the request has them.
        """
        variates = request.context[position]
        # Covariables passées : variables supplémentaires du contexte
        if request.past_covariates is not None:
            variates = np.concatenate([variates, request.past_covariates[position]])
        # Covariables connues, avec un axe d'échantillon
        known = None
        if request.known_covariates is not None:
            known = to_tensor(torch, request.known_covariates[position])[None]
        return t0.TimeSeries.from_array(to_tensor(torch, variates)[None], known)

    # Prévision d'un lot de séries
    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Forecast the series of ``request`` with the t0 forecaster.

        Args:
            request: Series, number of steps and levels to forecast.

        Returns:
            Quantile forecasts of shape ``(n_series, n_targets,
            prediction_length, n_levels)``.

        Raises:
            ImportError: If tfc-t0 or torch is not installed.
            ValueError: If the forecaster refuses its input or its options.
        """
        t0 = import_optional("t0", _EXTRA)
        torch = import_optional("torch", _EXTRA)
        lengths = np.array([context.shape[1] for context in request.context])
        # Au-delà d'une passe, un appel par longueur de contexte ; sinon un seul appel
        if request.prediction_length > self.forecaster.max_horizon:
            groups = [np.flatnonzero(lengths == length) for length in np.unique(lengths)]
        else:
            groups = [np.arange(request.n_series)]
        shape = (request.prediction_length, len(request.quantile_levels))
        result = np.empty((request.n_series, request.n_targets) + shape)
        # Parcours des groupes de séries : un lot par appel du modèle
        for group in groups:
            batch = t0.TimeSeries.batch(
                [self._time_series(request, position, t0, torch) for position in group.tolist()]
            )
            forecast = self.forecaster.predict(
                batch,
                request.prediction_length,
                list(request.quantile_levels),
                context_length=int(lengths[group].max()),
                **request.options,
            )
            # Sortie par variable, série après série : cibles gardées, covariables passées écartées
            quantiles = to_numpy(forecast.quantiles).reshape((len(group), -1) + shape)
            result[group] = quantiles[:, : request.n_targets]
        return ForecastResult(result)


# Fonction de chargement d'un modèle t0
def load_t0(model_id: str, *, device: str, **kwargs: Any) -> T0Backend:
    """Load a t0 model with the torch runtime of tfc-t0. Code and weights: Apache-2.0.

    Args:
        model_id: Hugging Face repository id (``'theforecastingcompany/t0-gamma'``,
            ``t0-beta``, ``t0-alpha``) or local directory.
        device: Device of the model.
        **kwargs: Other arguments of ``T0Forecaster.from_pretrained``
            (``revision`` to pin the weights, ``dtype``, ``local_files_only``...);
            the package silently ignores an argument it does not know.

    Returns:
        The loaded backend, in evaluation mode.

    Raises:
        ImportError: If tfc-t0 or torch is not installed.

    Examples:
        >>> 'theforecastingcompany/t0-gamma'.startswith(T0_PREFIXES)
        True
    """
    t0 = import_optional("t0", _EXTRA)
    forecaster = t0.T0Forecaster.from_pretrained(model_id, **kwargs)
    return T0Backend(forecaster.to(device).eval())
