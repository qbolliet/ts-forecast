"""Backend of the Timestral foundation models of the timestral package: ``TimestralBackend``.

``load_timestral`` loads ``mistralai/Timestral-*`` models with ``TimestralForecaster``
(several targets, past and known covariates, 21 quantile levels) and
``TimestralBackend.predict`` translates a request into one call of its ``forecast``
method, series of unequal lengths being padded on the left with missing values.
timestral is imported only when a model is loaded. The code of the package is
Apache-2.0; the weights are under a Modified MIT licence that grants no right to a
user whose company (or employer) had a global consolidated monthly revenue above
USD 20 million in the preceding month.
"""
# Importation des modules
# Modules de base
from typing import Any, List, Sequence, Tuple

# Modules de manipulation de données
import numpy as np

# tsforecast
from tsforecast.adapters._base import import_optional
from tsforecast.adapters.foundation.backends._tensors import to_numpy
from tsforecast.adapters.foundation.protocol import (
    BackendCapabilities,
    ForecastRequest,
    ForecastResult,
    same_level,
)

# Extra installant timestral
_EXTRA = "adapters-timestral"
# Préfixes des identifiants Hugging Face servis par le backend
TIMESTRAL_PREFIXES: Tuple[str, ...] = ("mistralai/timestral-",)


# Fonction auxiliaire d'empilement de séries de longueurs inégales
def _pad_and_stack(arrays: Sequence[np.ndarray], width: int) -> np.ndarray:
    """Stack ``(n_rows, n_steps)`` arrays, padding each on the left with ``NaN``.

    Args:
        arrays: One array per series, with the same number of rows and at most
            ``width`` steps.
        width: Number of steps of the result.

    Returns:
        A new ``(n_series, n_rows, width)`` array, the last step of every series
        in the last column.
    """
    stacked = np.full((len(arrays), arrays[0].shape[0], width), np.nan)
    # Parcours des séries : valeurs alignées sur la dernière colonne
    for position, values in enumerate(arrays):
        stacked[position, :, width - values.shape[1] :] = values
    return stacked


# Backend des modèles Timestral
class TimestralBackend:
    """Loaded Timestral model following the backend protocol.

    A request is one call of ``TimestralForecaster.forecast``: targets
    ``(n_series, n_targets, n_steps)``, ``past_covariates``
    ``(n_series, n_past, n_steps)`` and ``known_covariates``
    ``(n_series, n_known, n_steps + prediction_length)``. A series shorter than
    the longest one is padded on the left with ``NaN``, which the model reads as
    not started: its forecast is that of the series alone, up to rounding.
    Horizons longer than one decoding block (1 024 steps by default) are
    forecast block by block, each block reading the median of the previous ones.

    The options of the request are passed to ``forecast``. Its defaults apply:
    ``symmetric_averaging=True`` averages the forecast with the mirrored forecast
    of the negated series, and ``clip_nonnegative=True`` clips at zero the
    forecast of a target whose observed context is non-negative. Inference has no
    randomness.

    Licence: the weights are under a Modified MIT licence that grants no right to
    a user whose company (or employer) had a global consolidated monthly revenue
    above USD 20 million (or its equivalent) in the preceding month, derivatives
    of the model included; Mistral AI offers a commercial licence and hosted
    services. The code of the timestral package is Apache-2.0.

    Args:
        forecaster: Loaded ``TimestralForecaster``.

    Attributes:
        capabilities: What the model accepts, read from the forecaster: the
            covariates of its checkpoint, its context length and its quantile
            levels, the only levels it returns.

    Examples:
        >>> import types
        >>> import numpy as np
        >>> def forecast(context, prediction_length, past_covariates=None, known_covariates=None):
        ...     shape = context.shape[:2] + (prediction_length, 3)
        ...     return types.SimpleNamespace(quantiles=np.broadcast_to([1.0, 2.0, 3.0], shape))
        >>> forecaster = types.SimpleNamespace(
        ...     forecast=forecast, supports_covariates=True, supports_known_covariates=True,
        ...     max_context=64, quantile_levels=(0.1, 0.5, 0.9))
        >>> backend = TimestralBackend(forecaster)
        >>> backend.capabilities.max_context, backend.capabilities.interpolates_quantiles
        (64, False)
        >>> request = ForecastRequest(context=(np.array([[1.0, 2.0]]),), prediction_length=2,
        ...                           quantile_levels=(0.1, 0.9))
        >>> backend.predict(request).quantiles[0, 0].tolist()
        [[1.0, 3.0], [1.0, 3.0]]
    """

    # Initialisation depuis un modèle chargé
    def __init__(self, forecaster: Any):
        self.forecaster = forecaster
        # Covariables acceptées par le modèle chargé
        covariates = bool(forecaster.supports_covariates)
        self.capabilities = BackendCapabilities(
            past_covariates=covariates,
            known_covariates=covariates and bool(forecaster.supports_known_covariates),
            missing_values=True,
            multivariate=True,
            max_context=int(forecaster.max_context),
            quantile_levels=tuple(float(level) for level in forecaster.quantile_levels),
            interpolates_quantiles=False,
        )

    # Colonnes de la sortie correspondant aux niveaux demandés
    def _level_columns(self, levels: Sequence[float]) -> List[int]:
        """Find the position of each requested level among the levels of the model.

        Args:
            levels: Requested levels.

        Returns:
            One column index per level.

        Raises:
            ValueError: If a level is not one of the levels of the model.
        """
        native = self.capabilities.quantile_levels
        columns = []
        # Parcours des niveaux demandés : colonne du niveau appris égal
        for level in levels:
            matches = [index for index, value in enumerate(native) if same_level(value, level)]
            if not matches:
                raise ValueError(f"Timestral only returns its own quantile levels {native}, got {level}")
            columns.append(matches[0])
        return columns

    # Prévision d'un lot de séries
    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Forecast the series of ``request`` in one call of the Timestral forecaster.

        Args:
            request: Series, number of steps and levels to forecast; the levels
                are among those of the model.

        Returns:
            Quantile forecasts of shape ``(n_series, n_targets,
            prediction_length, n_levels)``.

        Raises:
            ValueError: If a level is not one of the levels of the model, or if
                the forecaster refuses its input.
        """
        columns = self._level_columns(request.quantile_levels)
        # Longueur commune : celle du plus long contexte
        width = max(context.shape[1] for context in request.context)
        # Covariables passées sur le contexte, covariables connues prolongées sur l'horizon
        past = known = None
        if request.past_covariates is not None:
            past = _pad_and_stack(request.past_covariates, width)
        if request.known_covariates is not None:
            known = _pad_and_stack(request.known_covariates, width + request.prediction_length)
        forecast = self.forecaster.forecast(
            _pad_and_stack(request.context, width),
            request.prediction_length,
            past_covariates=past,
            known_covariates=known,
            **request.options,
        )
        return ForecastResult(to_numpy(forecast.quantiles)[..., columns])


# Fonction de chargement d'un modèle Timestral
def load_timestral(model_id: str, *, device: str, **kwargs: Any) -> TimestralBackend:
    """Load a Timestral model with the timestral package.

    The weights are under a Modified MIT licence that grants no right to a user
    whose company (or employer) had a global consolidated monthly revenue above
    USD 20 million in the preceding month (see :class:`TimestralBackend`).

    Args:
        model_id: Hugging Face repository id (``'mistralai/Timestral-5M'``,
            ``-28M``, ``-123M``, ``-209M``) or local directory.
        device: Device of the model.
        **kwargs: Other arguments of ``TimestralForecaster.from_pretrained``
            (``revision`` to pin the weights, ``max_context``,
            ``per_shot_horizon_patches``, ``local_files_only``...).

    Returns:
        The loaded backend.

    Raises:
        ImportError: If timestral is not installed.

    Examples:
        >>> 'mistralai/Timestral-5M'.lower().startswith(TIMESTRAL_PREFIXES)
        True
    """
    timestral = import_optional("timestral", _EXTRA)
    return TimestralBackend(timestral.TimestralForecaster.from_pretrained(model_id, device=device, **kwargs))
