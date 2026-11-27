"""Backend of the Chronos foundation models of the chronos-forecasting package: ``ChronosBackend``.

``load_chronos`` loads ``Chronos2Pipeline`` models (``amazon/chronos-2``,
``autogluon/chronos-2-small``: past and known covariates, several targets) and
``ChronosBoltPipeline`` models (``amazon/chronos-bolt-*``: one target, no
covariates); ``ChronosBackend.predict`` translates a request into a call to the
pipeline's ``predict_quantiles``. The sample-based Chronos-T5 models are refused.
chronos is imported only when a model is loaded. Code and weights: Apache-2.0.
"""
# Importation des modules
# Modules de base
from typing import Any, Dict, List, Tuple

# Modules de manipulation de données
import numpy as np

# tsforecast
from tsforecast.adapters._base import import_optional
from tsforecast.adapters.foundation.backends._tensors import to_numpy, to_tensor
from tsforecast.adapters.foundation.protocol import BackendCapabilities, ForecastRequest, ForecastResult

# Extra installant chronos-forecasting
_EXTRA = "adapters-chronos"
# Préfixes des identifiants Hugging Face servis par le backend
CHRONOS_PREFIXES: Tuple[str, ...] = ("amazon/chronos", "autogluon/chronos")
# Préfixes des modèles Chronos-T5, dont la prévision tire des trajectoires au hasard
_SAMPLING_PREFIXES: Tuple[str, ...] = ("amazon/chronos-t5", "autogluon/chronos-t5")
# Familles de pipelines prises en charge
_CHRONOS_2, _BOLT = "chronos-2", "chronos-bolt"


# Backend des modèles Chronos
class ChronosBackend:
    """Loaded Chronos model (Chronos-2 or Chronos-Bolt) following the backend protocol.

    Chronos-2 receives one dictionary per series: the target (1-D, or 2-D for
    several targets), the past values of every covariate in
    ``past_covariates`` and the forecast-step values of the known covariates in
    ``future_covariates`` (``NaN`` for an unknown step). Chronos-Bolt receives
    one 1-D tensor per series. The options of the request are passed to
    ``predict_quantiles`` (``batch_size``, ``cross_learning``,
    ``limit_prediction_length``...). Inference has no randomness.

    Args:
        pipeline: Loaded ``Chronos2Pipeline`` or ``ChronosBoltPipeline``.
        family: ``'chronos-2'`` or ``'chronos-bolt'``.

    Attributes:
        capabilities: What the model accepts, read from the pipeline: the
            context length and the quantile levels it was trained on; Chronos-2
            interpolates levels between them.

    Examples:
        >>> 'amazon/chronos-2'.startswith(CHRONOS_PREFIXES)
        True
    """

    # Initialisation depuis un pipeline chargé
    def __init__(self, pipeline: Any, family: str):
        self.pipeline = pipeline
        self.family = family
        # Niveaux appris et contexte maximal lus sur le pipeline
        levels = tuple(float(level) for level in pipeline.quantiles)
        context = int(pipeline.model_context_length)
        # Chronos-2 : covariables, valeurs manquantes, plusieurs cibles
        if family == _CHRONOS_2:
            self.capabilities = BackendCapabilities(
                past_covariates=True,
                known_covariates=True,
                missing_values=True,
                multivariate=True,
                max_context=context,
                quantile_levels=levels,
                interpolates_quantiles=True,
            )
        # Chronos-Bolt : une cible, sans covariable
        else:
            self.capabilities = BackendCapabilities(
                missing_values=True,
                max_context=context,
                quantile_levels=levels,
                interpolates_quantiles=True,
            )

    # Entrées de Chronos-2 : un dictionnaire par série
    @staticmethod
    def _chronos2_inputs(request: ForecastRequest) -> List[Dict[str, Any]]:
        """Build the dictionaries of ``Chronos2Pipeline.predict_quantiles``.

        Args:
            request: Series to forecast.

        Returns:
            One dictionary per series, with the keys ``target`` and, when the
            request has covariates, ``past_covariates`` and ``future_covariates``.
        """
        # Dictionnaires construits série par série
        inputs = []
        # Parcours des séries de la requête
        for position, context in enumerate(request.context):
            length = context.shape[1]
            # Cible à une dimension pour une seule cible
            item: Dict[str, Any] = {"target": context[0] if context.shape[0] == 1 else context}
            past: Dict[str, np.ndarray] = {}
            future: Dict[str, np.ndarray] = {}
            # Covariables passées : valeurs du contexte seulement
            if request.past_covariates is not None:
                for name, values in zip(request.past_covariate_names, request.past_covariates[position]):
                    past[name] = values
            # Covariables connues : passé dans past_covariates, pas prévus dans future_covariates
            if request.known_covariates is not None:
                for name, values in zip(request.known_covariate_names, request.known_covariates[position]):
                    past[name] = values[:length]
                    future[name] = values[length:]
            if past:
                item["past_covariates"] = past
            if future:
                item["future_covariates"] = future
            inputs.append(item)
        return inputs

    # Prévision d'un lot de séries
    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Forecast the series of ``request`` with the Chronos pipeline.

        Args:
            request: Series, number of steps and levels to forecast.

        Returns:
            Quantile forecasts of shape ``(n_series, n_targets,
            prediction_length, n_levels)``.
        """
        # Arguments communs aux deux familles
        kwargs = dict(request.options)
        levels = list(request.quantile_levels)
        # Chronos-2 : dictionnaires, une sortie (n_cibles, H, Q) par série
        if self.family == _CHRONOS_2:
            quantiles, _ = self.pipeline.predict_quantiles(
                self._chronos2_inputs(request),
                prediction_length=request.prediction_length,
                quantile_levels=levels,
                **kwargs,
            )
            return ForecastResult(np.stack([to_numpy(values) for values in quantiles]))
        # Chronos-Bolt : tenseurs à une dimension, sortie (n_séries, H, Q)
        torch = import_optional("torch", _EXTRA)
        inputs = [to_tensor(torch, context[0]) for context in request.context]
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs, prediction_length=request.prediction_length, quantile_levels=levels, **kwargs
        )
        return ForecastResult(to_numpy(quantiles)[:, None])


# Fonction de chargement d'un modèle Chronos
def load_chronos(model_id: str, *, device: str, **kwargs: Any) -> ChronosBackend:
    """Load a Chronos-2 or Chronos-Bolt model with chronos-forecasting.

    Args:
        model_id: Hugging Face model id (``'autogluon/chronos-2-small'``,
            ``'amazon/chronos-bolt-tiny'``...) or local path.
        device: Device of the model, passed as ``device_map``.
        **kwargs: Other arguments of ``BaseChronosPipeline.from_pretrained``
            (``revision`` to pin the weights, ``dtype``...).

    Returns:
        The loaded backend.

    Raises:
        ImportError: If chronos-forecasting is not installed.
        ValueError: If the model is a sample-based Chronos-T5 model, or any
            pipeline other than Chronos-2 and Chronos-Bolt.

    Examples:
        >>> try:
        ...     load_chronos('amazon/chronos-t5-tiny', device='cpu')
        ... except ValueError as error:
        ...     print(str(error)[:46])
        Chronos-T5 models draw random sample paths: th
    """
    # Modèles Chronos-T5 refusés avant tout téléchargement
    if str(model_id).lower().startswith(_SAMPLING_PREFIXES):
        raise ValueError(
            "Chronos-T5 models draw random sample paths: their forecasts are not reproducible. "
            "Use a Chronos-2 or Chronos-Bolt model"
        )
    chronos = import_optional("chronos", _EXTRA)
    pipeline = chronos.BaseChronosPipeline.from_pretrained(model_id, device_map=device, **kwargs)
    # Famille du pipeline chargé
    if isinstance(pipeline, chronos.Chronos2Pipeline):
        return ChronosBackend(pipeline, _CHRONOS_2)
    if isinstance(pipeline, chronos.ChronosBoltPipeline):
        return ChronosBackend(pipeline, _BOLT)
    raise ValueError(
        f"{type(pipeline).__name__} is not supported: only Chronos-2 and Chronos-Bolt pipelines forecast "
        "quantiles without sampling"
    )
