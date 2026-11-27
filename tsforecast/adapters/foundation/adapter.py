"""Sklearn adapter for forecasting foundation models: ``FoundationModelAdapter``.

A foundation model learns nothing at fit: the adapter stores the target history
of each entity (its context) and, at predict, forecasts each row of ``X`` from the
last training date of its entity, the number of steps being read from the index.
The model is reached through a backend (``chronos``, ``timestral``, ``t0``, a
``CallableBackend``, or any registered backend), loaded once per process; quantile
forecasts are available through ``predict_quantiles``.
"""
# Importation des modules
# Modules de base
from typing import Any, List, Mapping, Optional, Sequence, Tuple

# Modules de manipulation de données
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# tsforecast
from tsforecast.adapters._base import _preview, check_adapter_input, steps_between
from tsforecast.adapters.foundation._series import (
    bracketing_levels,
    check_quantile_levels,
    entity_grid_positions,
    entity_rank,
    future_windows,
    interpolate_quantiles,
    place_on_grids,
)
from tsforecast.adapters.foundation.protocol import (
    BackendCapabilities,
    ForecastBackend,
    ForecastRequest,
    ForecastResult,
)
from tsforecast.adapters.foundation.registry import load_backend, resolve_backend

# Modes d'usage des colonnes de X
_AUTO, _KNOWN, _PAST = "auto", "known", "past"
_COVARIATE_MODES = (_AUTO, _KNOWN, _PAST, None)
# Nom de la cible quand la cible n'a pas de nom
_DEFAULT_TARGET_NAME = "target"
# Niveaux de quantiles par défaut de predict_quantiles
_DEFAULT_LEVELS = (0.1, 0.5, 0.9)
# Attributs posés par fit, retirés avant un nouvel ajustement
_FITTED_ATTRIBUTES = (
    "layout_",
    "backend_name_",
    "capabilities_",
    "covariates_",
    "context_length_",
    "context_",
    "context_covariates_",
    "target_names_",
    "_backend_object",
    "_backend_source",
    "_covariate_names",
    "_target_ndim",
)


# Fonction auxiliaire de contrôle d'un entier positif facultatif
def _is_count(value: Any) -> bool:
    """Whether a parameter value is None or an integer of at least 1 (booleans excluded).

    Args:
        value: Parameter value.

    Returns:
        True for None or a positive integer.
    """
    return value is None or (
        not isinstance(value, bool) and isinstance(value, (int, np.integer)) and value >= 1
    )


# Fonction auxiliaire de découpage d'une suite en lots
def _batches(items: List[Any], size: Optional[int]) -> List[List[Any]]:
    """Split a list into consecutive batches of at most ``size`` items.

    Args:
        items: Items to split.
        size: Largest batch, or None for a single batch.

    Returns:
        The batches, in order.
    """
    # Lot unique sans taille maximale
    if size is None:
        return [items]
    return [items[start : start + size] for start in range(0, len(items), size)]


# Wrapper des modèles de fondation dans un régresseur sklearn
class FoundationModelAdapter(RegressorMixin, BaseEstimator):
    """Sklearn regressor forecasting the rows of ``X`` with a pretrained foundation model.

    Rows are indexed by forecast origin: a ``DatetimeIndex`` for a time series,
    or a ``MultiIndex`` whose last level holds the dates and whose other levels
    identify the entity for a panel; dates lie on a regular grid, holes allowed.
    Nothing is shifted and the model is never trained:

    - ``fit`` stores, for each entity, the target on its grid from its first
      to its last training date (``NaN`` on the dates without a row), keeping
      the ``context_length`` most recent steps; with covariates, the columns of
      ``X`` on the same grid.
    - ``predict`` gives a row ``k`` grid steps after the last training date of
      its entity the last step of a ``k``-step forecast of that context. With
      known covariates, that forecast also receives the rows of ``X`` of the
      entity 1 to ``k`` steps after the end of training, a step without a row
      (the gap) being declared unknown: a row reads the features of its own
      origin and of the earlier rows of its entity in ``X``, never those of a
      later origin. Without known covariates, the prediction of a row does
      not depend on the other rows of ``X``. Rows sharing a number of steps
      are forecast together, in batches of ``batch_size`` series; a forecast
      does not depend on the series of its batch, up to rounding.

    The returned value is the ``quantile`` level of the forecast (the median by
    default); ``predict_quantiles`` returns several levels. Only rows after the
    training data of a known entity can be predicted.

    The model is chosen by ``model_id`` (Hugging Face id, served by the backend
    registered for its prefix, or a local path with ``backend=<name>``), or
    passed as a backend object (``backend=CallableBackend(function)``). A
    registered model is loaded on the first ``fit`` and kept for the whole
    process (``clear_model_cache`` releases it): clones, cross-validation folds
    and grid-search candidates share it, and the weights are never copied by
    ``clone`` or ``pickle``. With ``n_jobs > 1``, each worker process loads its
    own copy: keep ``n_jobs=1``, the model already uses several threads.

    Licences: Chronos models and chronos-forecasting are Apache-2.0, as are t0
    models and tfc-t0. The timestral package is Apache-2.0, but the Timestral
    weights are under a Modified MIT licence that grants no right to a user whose
    company (or employer) had a global consolidated monthly revenue above USD 20
    million in the preceding month.

    Args:
        model_id: Model id or local path of a registered backend; None with a
            backend object.
        backend: None to deduce the backend from ``model_id``, the name of a
            registered backend, or a backend object (``capabilities`` and
            ``predict``), used as is.
        context_length: Number of most recent grid steps of each entity
            passed to the model, or None for the whole history (within the
            longest context of the model).
        quantile: Level returned by ``predict``, in ``]0, 1[``.
        covariates: Use of the columns of ``X``: ``'known'`` (known over the
            context and at the predicted rows), ``'past'`` (known over the
            context only), None (ignored), or ``'auto'``: ``'known'`` when the
            model accepts it, else ``'past'``, else None.
        device: Device of the model (``'cpu'``, ``'cuda'``...), for a
            registered backend.
        batch_size: Largest number of series per call of the model, or None.
        backend_kwargs: Loading options of a registered backend (``revision``
            for all three; ``dtype`` for Chronos and t0, ``max_context`` for
            Timestral); hashable values.
        predict_kwargs: Options passed to the backend at each call (Chronos:
            ``batch_size``, ``cross_learning``; with ``cross_learning=True``,
            a forecast depends on the other series of its batch. Timestral:
            ``symmetric_averaging``, ``clip_nonnegative``, both on by default.
            t0: ``symmetric_averaging``, ``clip_nonnegative``, ``snap_integers``,
            ``trim_leading_missing``, off by default, and
            ``drop_constant_covariates``, on by default).

    Attributes:
        backend_name_: Name of the registered backend, or class name of the
            backend object.
        capabilities_: Capabilities of the model.
        covariates_: Use of the columns of ``X`` retained: ``'known'``,
            ``'past'`` or None.
        context_length_: Largest number of steps stored per entity, or None.
        context_: Target context of each entity, one read-only
            ``(n_targets, n_steps)`` array per entity of ``layout_.entities``.
        context_covariates_: Columns of ``X`` on the same grids, or None
            without covariates.
        layout_: Temporal layout of the training rows (entities, dates, grid
            frequency).
        target_names_: Names of the target columns (``'target'`` for an
            unnamed ``Series``).
        n_features_in_: Number of columns of ``X`` seen at fit.
        feature_names_in_: Column names of ``X`` seen at fit.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tsforecast.adapters import CallableBackend, FoundationModelAdapter
        >>> def drift(request):
        ...     # Last value plus the mean change of the context, per step and level
        ...     steps = np.arange(1, request.prediction_length + 1)
        ...     paths = []
        ...     for series in request.context:
        ...         slope = (series[:, -1] - series[:, 0]) / (series.shape[1] - 1)
        ...         path = series[:, -1:] + slope[:, None] * steps
        ...         paths.append(np.repeat(path[:, :, None], len(request.quantile_levels), axis=2))
        ...     return np.stack(paths)
        >>> dates = pd.date_range('2024-01-01', periods=6, freq='MS', name='date')
        >>> X = pd.DataFrame({'x': np.zeros(6)}, index=dates)
        >>> y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=dates)
        >>> model = FoundationModelAdapter(backend=CallableBackend(drift), covariates=None)
        >>> model = model.fit(X.iloc[:3], y.iloc[:3])
        >>> # Rows 3 and 1 grid steps after the last training month
        >>> model.predict(X.iloc[[5, 3]]).tolist()
        [6.0, 4.0]
    """

    # Initialisation
    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        backend: Any = None,
        context_length: Optional[int] = None,
        quantile: float = 0.5,
        covariates: Optional[str] = _AUTO,
        device: str = "cpu",
        batch_size: Optional[int] = None,
        backend_kwargs: Optional[Mapping[str, Any]] = None,
        predict_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.model_id = model_id
        self.backend = backend
        self.context_length = context_length
        self.quantile = quantile
        self.covariates = covariates
        self.device = device
        self.batch_size = batch_size
        self.backend_kwargs = backend_kwargs
        self.predict_kwargs = predict_kwargs

    # Étiquettes sklearn : covariables manquantes admises
    def __sklearn_tags__(self) -> Any:
        """Declare that ``X`` may hold missing values (unknown covariates).

        Returns:
            The scikit-learn tags of the adapter.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    # Vérification des paramètres
    def _check_params(self) -> None:
        """Validate the constructor parameters.

        Raises:
            ValueError: If a parameter has an invalid value or type.
        """
        # Longueurs facultatives : entiers d'au moins 1
        for name in ("context_length", "batch_size"):
            if not _is_count(getattr(self, name)):
                raise ValueError(f"{name} must be None or an integer of at least 1, got {getattr(self, name)!r}")
        # Niveau renvoyé par predict
        if isinstance(self.quantile, bool) or not isinstance(self.quantile, (int, float, np.floating)) or not (
            0.0 < float(self.quantile) < 1.0
        ):
            raise ValueError(f"quantile must be a number strictly between 0 and 1, got {self.quantile!r}")
        # Usage des covariables
        if self.covariates not in _COVARIATE_MODES:
            raise ValueError(f"covariates must be one of {_COVARIATE_MODES}, got {self.covariates!r}")
        # Périphérique et options
        if not isinstance(self.device, str):
            raise ValueError(f"device must be a string, got {self.device!r}")
        for name in ("backend_kwargs", "predict_kwargs"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be None or a mapping, got {getattr(self, name)!r}")
        # Identifiant de modèle : chaîne, ou absent avec un objet backend
        if self.model_id is not None and not isinstance(self.model_id, str):
            raise ValueError(f"model_id must be None or a string, got {self.model_id!r}")

    # Résolution et chargement du backend
    def _resolve_backend(self) -> Tuple[ForecastBackend, str, Optional[Tuple[Any, ...]]]:
        """Return the backend of the parameters, loading a registered model through the cache.

        Returns:
            ``(backend, name, source)``: the loaded model, the name of its
            backend, and the arguments that load it again
            (``(name, model_id, device, backend_kwargs)``), or None for a backend
            object.

        Raises:
            ValueError: If neither ``model_id`` nor a backend is given, if a
                backend object comes with a ``model_id``, or if no registered
                backend serves ``model_id``.
            TypeError: If ``backend`` is neither None, a string nor a backend
                object.
            ImportError: If the package of the backend is not installed.
        """
        # Objet backend utilisé tel quel, sans chargement
        if self.backend is not None and not isinstance(self.backend, str):
            if not isinstance(self.backend, ForecastBackend):
                raise TypeError(
                    "backend must be None, the name of a registered backend, or an object with "
                    f"capabilities and a predict method, got {type(self.backend).__name__}"
                )
            if self.model_id is not None:
                raise ValueError("model_id must be None when backend is a backend object")
            return self.backend, type(self.backend).__name__, None
        # Backend enregistré : nom donné ou déduit de l'identifiant
        if self.model_id is None:
            raise ValueError("Give a model_id (for a registered backend) or a backend object")
        spec = resolve_backend(self.model_id, self.backend)
        kwargs = dict(self.backend_kwargs or {})
        loaded = load_backend(spec, self.model_id, device=self.device, backend_kwargs=kwargs)
        return loaded, spec.name, (spec.name, self.model_id, self.device, kwargs)

    # Modèle chargé de l'ajustement
    def _backend(self) -> ForecastBackend:
        """Return the model of the fit, from the process cache for a registered backend.

        Returns:
            The loaded model (loaded again in a process that does not hold it).
        """
        # Objet backend mémorisé au fit
        if self._backend_source is None:
            return self._backend_object
        name, model_id, device, kwargs = self._backend_source
        return load_backend(resolve_backend(model_id, name), model_id, device=device, backend_kwargs=kwargs)

    # Choix de l'usage des covariables
    def _covariate_mode(self, capabilities: BackendCapabilities, n_features: int) -> Optional[str]:
        """Resolve ``covariates`` against the capabilities of the model.

        Args:
            capabilities: Capabilities of the model.
            n_features: Number of columns of ``X``.

        Returns:
            ``'known'``, ``'past'`` or None.

        Raises:
            ValueError: If an explicit use is not supported by the model.
        """
        # Aucune colonne ou covariables ignorées
        if self.covariates is None or n_features == 0:
            return None
        # Choix automatique : covariables connues, sinon passées, sinon aucune
        if self.covariates == _AUTO:
            if capabilities.known_covariates:
                return _KNOWN
            return _PAST if capabilities.past_covariates else None
        # Usage explicite : refusé si le modèle ne le prend pas en charge
        supported = capabilities.known_covariates if self.covariates == _KNOWN else capabilities.past_covariates
        if not supported:
            raise ValueError(
                f"The model does not accept {self.covariates} covariates; use covariates='auto' or None"
            )
        return self.covariates

    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Any) -> "FoundationModelAdapter":
        """Load the model and store the context of each entity; nothing is trained.

        Args:
            X: Features indexed by a ``DatetimeIndex`` or an
                ``(entity..., date)`` ``MultiIndex``, rows in any order; used
                as covariates according to ``covariates``.
            y: Target aligned with ``X``: ``Series`` (one target) or
                ``DataFrame`` (several targets); ``NaN`` marks a missing value.

        Returns:
            The fitted adapter.

        Raises:
            ImportError: If the package of the backend is not installed.
            TypeError: If ``X`` is not a ``DataFrame`` or its dates are not
                ``datetime64``.
            ValueError: If a parameter is invalid, if the dates are not on a
                regular grid, if an entity has no observed target in its
                context, if a value is missing or the covariates are not
                numeric while the model refuses it, or if the model does not
                accept the requested covariates or level.
        """
        # Attributs d'un ajustement précédent retirés : aucun état n'en survit
        for name in _FITTED_ATTRIBUTES:
            self.__dict__.pop(name, None)
        # Vérification des paramètres et des entrées
        self._check_params()
        X, y, layout = check_adapter_input(self, X, y, reset=True)
        # Grille irrégulière : nombre de pas illisible
        if layout.freq is None:
            raise ValueError(
                "The training dates are not on a regular grid: the number of steps to each predicted "
                "row cannot be read from the index"
            )

        # Modèle et capacités ; niveau renvoyé par predict dans la plage du modèle
        backend, name, source = self._resolve_backend()
        capabilities = backend.capabilities
        if not isinstance(capabilities, BackendCapabilities):
            raise TypeError(f"The capabilities of backend {name!r} must be BackendCapabilities")
        check_quantile_levels([self.quantile], capabilities, "quantile")
        mode = self._covariate_mode(capabilities, X.shape[1])

        # Cible : tableau à deux dimensions, noms mémorisés pour la sortie
        target_ndim = y.ndim
        names = [_DEFAULT_TARGET_NAME if y.name is None else y.name] if target_ndim == 1 else list(y.columns)
        targets = y.to_numpy(dtype=np.float64, na_value=np.nan).reshape(len(y), -1)
        # Valeurs infinies refusées
        if np.isinf(targets).any():
            raise ValueError("y contains infinite values")

        # Nombre de pas gardés : context_length dans la limite du modèle
        limits = [value for value in (self.context_length, capabilities.max_context) if value is not None]
        keep = min(limits) if limits else None
        positions, lengths = entity_grid_positions(layout, layout.freq)
        contexts = place_on_grids(targets, layout.codes, positions, lengths, keep)

        # Entité sans cible observée dans son contexte
        empty = [
            layout.entities[code] for code, context in enumerate(contexts) if not np.isfinite(context).any(axis=1).all()
        ]
        if empty:
            where = f" of the entities {_preview(empty)}" if layout.is_panel else ""
            raise ValueError(f"The context{where} has no observed value of a target")
        # Valeurs manquantes (trous, cibles NaN) refusées par le modèle
        if not capabilities.missing_values and any(np.isnan(context).any() for context in contexts):
            raise ValueError(
                f"The backend {name!r} does not accept missing values, but the training target has missing "
                "values or the training rows leave grid dates uncovered"
            )

        # Covariables sur les mêmes grilles, nommées par leurs colonnes (uniques, contrôlées par sklearn)
        covariate_contexts = None
        covariate_names: Tuple[str, ...] = ()
        if mode is not None:
            features = self._features(X)
            covariate_contexts = place_on_grids(features, layout.codes, positions, lengths, keep)
            if not capabilities.missing_values and any(np.isnan(values).any() for values in covariate_contexts):
                raise ValueError(
                    f"The backend {name!r} does not accept missing values, but the covariates have missing "
                    "values or the training rows leave grid dates uncovered"
                )
            covariate_names = tuple(str(column) for column in X.columns)

        # Attributs ajustés, posés une fois l'ajustement réussi
        self.backend_name_ = name
        self.capabilities_ = capabilities
        self.covariates_ = mode
        self.context_length_ = keep
        self.context_ = contexts
        self.context_covariates_ = covariate_contexts
        self.layout_ = layout
        self.target_names_ = names
        self._backend_object = backend if source is None else None
        self._backend_source = source
        self._covariate_names = covariate_names
        self._target_ndim = target_ndim
        return self

    # Covariables numériques de X
    @staticmethod
    def _features(X: pd.DataFrame) -> np.ndarray:
        """Convert the columns of ``X`` to a ``float64`` array, ``NaN`` for a missing value.

        Args:
            X: Features.

        Returns:
            Array of shape ``(n_rows, n_columns)``.

        Raises:
            ValueError: If a column is not numeric or a value is infinite.
        """
        # Conversion numérique : colonnes non numériques refusées
        try:
            features = X.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError):
            raise ValueError("The covariates (columns of X) must be numeric") from None
        # Valeurs infinies refusées
        if np.isinf(features).any():
            raise ValueError("X contains infinite values")
        return features

    # Prévision des lignes de X à des niveaux donnés
    def _forecast_rows(self, X: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
        """Forecast the rows of ``X`` at the given levels.

        Args:
            X: Features of the rows to predict.
            levels: Levels to return, validated, in the wanted order.

        Returns:
            Forecasts of shape ``(n_rows, n_targets, n_levels)``, in the row
            order of ``X``.

        Raises:
            ValueError: If an entity is unknown at fit, if a row is not after
                the training data of its entity or too far for the model, or if
                the columns differ from those seen at fit.
        """
        # Vérification des entrées ; fréquence de l'ajustement pour un test à une ligne par entité
        X, _, layout = check_adapter_input(self, X, reset=False, freq=self.layout_.freq)
        n_targets = len(self.target_names_)
        # Aucune ligne à prévoir
        if len(X) == 0:
            return np.empty((0, n_targets, len(levels)))

        # Entités inconnues : aucun historique à prolonger
        codes = self.layout_.match_entities(layout)
        if (codes < 0).any():
            unknown = [layout.entities[code] for code in np.unique(layout.codes[codes < 0])]
            raise ValueError(f"Entities unknown at fit: {_preview(unknown)}")
        # Nombre de pas de chaque ligne après la dernière date d'entraînement de son entité
        steps = steps_between(self.layout_.last_dates()[codes], layout.dates, self.layout_.freq)
        if (steps <= 0).any():
            raise ValueError(
                "A foundation model forecasts after the training data: it cannot predict rows at or before "
                f"the last training date of their entity "
                f"({_preview(layout.dates[steps <= 0].unique().strftime('%Y-%m-%d %H:%M:%S'))})"
            )
        # Ligne au-delà de la prévision la plus longue du modèle
        longest = self.capabilities_.max_prediction_length
        if longest is not None and steps.max() > longest:
            raise ValueError(
                f"The model forecasts at most {longest} steps, but a row is {int(steps.max())} steps after "
                "the training data of its entity"
            )
        # Covariables connues : lignes de X placées sur les pas suivant l'entraînement
        windows = None
        if self.covariates_ == _KNOWN:
            windows = future_windows(self._features(X), codes, steps, self.layout_.n_entities)
        return self._forecast(codes, steps, windows, levels)

    # Prévision groupée par nombre de pas
    def _forecast(
        self,
        codes: np.ndarray,
        steps: np.ndarray,
        windows: Optional[Tuple[Optional[np.ndarray], ...]],
        levels: np.ndarray,
    ) -> np.ndarray:
        """Forecast each row as the last step of a forecast as long as its number of steps.

        Args:
            codes: Training entity code of each row.
            steps: Number of steps of each row after the training data of its entity.
            windows: Known covariates after the training data of each entity
                (see :func:`future_windows`), or None.
            levels: Levels to return, in the wanted order.

        Returns:
            Forecasts of shape ``(n_rows, n_targets, n_levels)``.
        """
        # Niveaux demandés au modèle : triés, ou niveaux appris qui les encadrent
        capabilities = self.capabilities_
        sorted_levels = np.unique(levels)
        if capabilities.quantile_levels is not None and not capabilities.interpolates_quantiles:
            request_levels = bracketing_levels(sorted_levels, capabilities.quantile_levels)
        else:
            request_levels = tuple(sorted_levels.tolist())

        # Une série par ligne, ou par (ligne, cible) pour un modèle à une cible
        n_targets = len(self.target_names_)
        split = n_targets > 1 and not capabilities.multivariate
        targets = list(range(n_targets)) if split else [None]
        rank = entity_rank(self.layout_.entities)
        backend = self._backend()
        result = np.empty((len(steps), n_targets, len(request_levels)))

        # Parcours des nombres de pas distincts : une prévision de longueur k par groupe
        for k in np.unique(steps).tolist():
            rows = np.flatnonzero(steps == k)
            # Ordre canonique des entités : lots indépendants de l'ordre des lignes
            rows = rows[np.argsort(rank[codes[rows]], kind="stable")]
            series = [(row, target) for row in rows.tolist() for target in targets]
            # Parcours des lots de séries
            for batch in _batches(series, self.batch_size):
                request = self._request(batch, codes, k, windows, request_levels)
                quantiles = self._call(backend, request)
                # Dernier pas de chaque prévision, rangé à sa ligne (et sa cible)
                last = quantiles[:, :, k - 1, :]
                batch_rows = np.array([row for row, _ in batch])
                if split:
                    batch_targets = np.array([target for _, target in batch])
                    result[batch_rows, batch_targets] = last[:, 0]
                else:
                    result[batch_rows] = last

        # Interpolation entre niveaux appris, puis ordre demandé
        if request_levels != tuple(sorted_levels.tolist()):
            result = interpolate_quantiles(result, request_levels, sorted_levels.tolist())
        return result[..., np.searchsorted(sorted_levels, levels)]

    # Construction d'une requête
    def _request(
        self,
        batch: List[Tuple[int, Optional[int]]],
        codes: np.ndarray,
        k: int,
        windows: Optional[Tuple[Optional[np.ndarray], ...]],
        levels: Tuple[float, ...],
    ) -> ForecastRequest:
        """Build the request of a batch of series forecast over ``k`` steps.

        Args:
            batch: ``(row, target)`` pairs, ``target`` None for every target.
            codes: Training entity code of each row.
            k: Number of steps to forecast.
            windows: Known covariates after the training data of each entity, or None.
            levels: Levels to request.

        Returns:
            The request of the batch.

        Raises:
            ValueError: If the model refuses missing values and a known
                covariate is missing on one of the ``k`` steps.
        """
        # Contexte de chaque série : toutes les cibles de l'entité, ou une seule
        entity_codes = [int(codes[row]) for row, _ in batch]
        context = tuple(
            self.context_[code] if target is None else self.context_[code][target : target + 1]
            for code, (_, target) in zip(entity_codes, batch)
        )
        kwargs: dict = {}
        # Covariables passées : grilles de l'entraînement
        if self.covariates_ == _PAST:
            kwargs["past_covariates"] = tuple(self.context_covariates_[code] for code in entity_codes)
            kwargs["past_covariate_names"] = self._covariate_names
        # Covariables connues : grilles de l'entraînement suivies des k premiers pas
        elif self.covariates_ == _KNOWN:
            known = tuple(
                np.concatenate([self.context_covariates_[code], windows[code][:, :k]], axis=1)
                for code in entity_codes
            )
            if not self.capabilities_.missing_values and any(np.isnan(values).any() for values in known):
                raise ValueError(
                    f"The backend {self.backend_name_!r} does not accept missing values, but X has no row (or "
                    "a missing value) on a step between the training data and a predicted row; no value is "
                    "invented for them: use no gap or covariates=None"
                )
            kwargs["known_covariates"] = known
            kwargs["known_covariate_names"] = self._covariate_names
        return ForecastRequest(
            context=context,
            prediction_length=k,
            quantile_levels=levels,
            freq=self.layout_.freq,
            options=dict(self.predict_kwargs or {}),
            **kwargs,
        )

    # Appel du modèle et contrôle de sa sortie
    def _call(self, backend: ForecastBackend, request: ForecastRequest) -> np.ndarray:
        """Call the model on a request and check its output.

        Args:
            backend: Loaded model.
            request: Request to forecast.

        Returns:
            Forecasts of shape ``(n_series, n_targets, prediction_length, n_levels)``.

        Raises:
            TypeError: If the model does not return a :class:`ForecastResult`.
            ValueError: If the forecasts have another shape or non-finite values.
        """
        # Appel du modèle, puis contrôle du type, de la forme et des valeurs de sa sortie
        result = backend.predict(request)
        if not isinstance(result, ForecastResult):
            raise TypeError(f"The backend {self.backend_name_!r} must return a ForecastResult, got {type(result).__name__}")
        expected = (request.n_series, request.n_targets, request.prediction_length, len(request.quantile_levels))
        if result.quantiles.shape != expected:
            raise ValueError(
                f"The backend {self.backend_name_!r} returned forecasts of shape {result.quantiles.shape}, "
                f"expected {expected}"
            )
        if not np.isfinite(result.quantiles).all():
            raise ValueError(f"The backend {self.backend_name_!r} returned non-finite forecasts")
        return result.quantiles

    # Méthode de prédiction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the rows of ``X`` at the level ``quantile``.

        Args:
            X: Features of the rows to predict, with the columns seen at fit;
                rows in any order, of entities known at fit, after their last
                training date.

        Returns:
            Predictions in the row order of ``X``: shape ``(n_samples,)`` for a
            ``Series`` target, ``(n_samples, n_targets)`` for a ``DataFrame``.

        Raises:
            ImportError: If the package of the backend is not installed.
            NotFittedError: If the adapter is not fitted.
            ValueError: If the columns differ from those seen at fit, if a date
                is off the training grid, if an entity is unknown at fit, if a
                row is not after the training data of its entity or further
                than the model forecasts, or if a known covariate is missing
                for a model refusing missing values.
        """
        # Vérification que l'estimateur est entraîné
        check_is_fitted(self)
        values = self._forecast_rows(X, np.array([float(self.quantile)]))[:, :, 0]
        return values[:, 0] if self._target_ndim == 1 else values

    # Méthode de prévision des quantiles
    def predict_quantiles(
        self, X: pd.DataFrame, quantile_levels: Sequence[float] = _DEFAULT_LEVELS
    ) -> pd.DataFrame:
        """Predict several quantile levels of the rows of ``X``.

        Levels the model was trained on are returned as such; a level between
        them is interpolated by the model, or linearly by the adapter for a
        model that only returns its own levels. Quantile crossings of the model
        are left as they are.

        Args:
            X: Features of the rows to predict (see :meth:`predict`).
            quantile_levels: Distinct levels in ``]0, 1[``, within the range of
                the levels the model was trained on; columns follow their order.

        Returns:
            ``DataFrame`` indexed like ``X`` (same rows, same order, dates in
            the unit of ``X``), one ``float64`` column per level (columns named
            by the levels, index name ``quantile``), or ``(target, quantile)``
            columns for several targets.

        Raises:
            NotFittedError: If the adapter is not fitted.
            ValueError: If a level is invalid or outside the range of the
                model, or for the reasons of :meth:`predict`.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from tsforecast.adapters import CallableBackend, FoundationModelAdapter
            >>> def spread(request):
            ...     # Last value shifted by (level - 0.5) * 10
            ...     shape = (request.n_series, request.n_targets, request.prediction_length)
            ...     last = np.array([series[:, -1] for series in request.context])[:, :, None, None]
            ...     offsets = (np.array(request.quantile_levels) - 0.5) * 10
            ...     return np.broadcast_to(last, shape + (1,)) + offsets
            >>> dates = pd.date_range('2024-01-01', periods=4, freq='QS', name='date')
            >>> X = pd.DataFrame({'x': np.zeros(4)}, index=dates)
            >>> y = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
            >>> model = FoundationModelAdapter(backend=CallableBackend(spread)).fit(X.iloc[:3], y.iloc[:3])
            >>> model.predict_quantiles(X.iloc[3:], quantile_levels=[0.9, 0.1]).round(6)
            quantile    0.9  0.1
            date
            2024-10-01  7.0 -1.0
        """
        # Vérification que l'estimateur est entraîné et des niveaux
        check_is_fitted(self)
        levels = check_quantile_levels(quantile_levels, self.capabilities_, "quantile_levels")
        values = self._forecast_rows(X, levels)
        # Colonnes : niveaux, ou couples (cible, niveau)
        level_index = pd.Index(levels, dtype="float64", name="quantile")
        if self._target_ndim == 1:
            return pd.DataFrame(values[:, 0, :], index=X.index, columns=level_index)
        columns = pd.MultiIndex.from_product([self.target_names_, level_index], names=["target", "quantile"])
        return pd.DataFrame(values.reshape(len(values), -1), index=X.index, columns=columns)
