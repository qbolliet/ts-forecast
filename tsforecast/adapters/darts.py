"""Sklearn adapter for darts forecasting models: ``DartsAdapter``.

The wrapped darts model predicts the rows of ``X``. Its own configuration
decides how: a global model reading only the covariates of the predicted date
(no target lag, ``lags_future_covariates=[0]``) predicts each row from its
features, whatever the gap after the training data; any other model
extrapolates the target history of each entity up to the dates of the rows.
Panels are passed as one ``TimeSeries`` per entity, and the predictions are
returned as an array in the row order of ``X``.
"""
# Importation des modules
# Modules de base
import copy
import inspect
from typing import Any, Dict, List, Tuple

# Modules de manipulation de données
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_classifier
from sklearn.utils.validation import check_is_fitted

# tsforecast
from tsforecast.adapters._base import (
    TemporalLayout,
    _preview,
    check_adapter_input,
    import_optional,
    steps_between,
)
from tsforecast.utils.frequency import to_dateoffset

# Extra installant darts
_EXTRA = "adapters-darts"
# Nom de la cible transmise quand la cible n'a pas de nom
_DEFAULT_TARGET_NAME = "target"
# Décalages extrêmes (extreme_lags de darts) d'un modèle qui ne lit que les covariables
# de la date prévue : aucun lag de cible ni de covariable passée, covariable future au
# lag 0 seul, un pas prévu par passe, sans décalage de sortie
_ROW_EXTREME_LAGS = (None, 0, None, None, 0, 0, 0)
# Modes de prévision
_ROW, _GLOBAL, _LOCAL = "row", "global", "local"
# Attributs ajustés propres à un mode
_MODE_ATTRIBUTES = ("model_", "entity_models_", "series_", "covariates_")


# Fonction auxiliaire de comparaison de deux valeurs de paramètre
def _same_value(left: Any, right: Any) -> bool:
    """Whether two parameter values are the same object or compare equal."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        # Comparaison ambiguë (tableaux) : valeurs considérées comme différentes
        return False


# Fonction auxiliaire de détermination du mode de prévision d'un modèle darts
def _forecasting_mode(model: Any) -> str:
    """Return ``'row'``, ``'global'`` or ``'local'`` for a darts model, refusing other objects.

    Raises:
        TypeError: If ``model`` is not a darts forecasting model, or is a classifier.
        ValueError: If a row-wise model uses position encoders.
    """
    # Importation du package
    base = import_optional("darts.models.forecasting.forecasting_model", _EXTRA)

    # Vérification du type du modèle
    if not isinstance(model, base.ForecastingModel):
        raise TypeError(f"model must be a darts forecasting model, got {type(model).__name__}")
    
    # Classifieur darts : l'estimateur sklearn sous-jacent est un classifieur
    inner = getattr(model, "model", None)
    if isinstance(inner, BaseEstimator) and is_classifier(inner):
        raise TypeError(
            f"{type(model).__name__} is a classifier; DartsAdapter is a regressor"
        )
    if isinstance(model, base.LocalForecastingModel):
        return _LOCAL
    if not isinstance(model, base.GlobalForecastingModel):
        raise TypeError(
            f"{type(model).__name__} is neither a local nor a global darts forecasting model"
        )
    if tuple(model.extreme_lags) != _ROW_EXTREME_LAGS:
        return _GLOBAL
    
    # Position relative à la fin de la série : sans signification pour une ligne isolée
    encoders = model.model_params.get("add_encoders") or {}
    if "position" in encoders:
        raise ValueError(
            "Position encoders make the prediction of a row depend on its distance to the end "
            "of a series; a model predicting each row from its own features cannot use them"
        )
    return _ROW


# Fonction auxiliaire de découpage des lignes en suites contiguës par entité
def _contiguous_runs(layout: TemporalLayout, freq: str) -> List[np.ndarray]:
    """Split the rows into runs of consecutive grid dates of one entity.

    Returns:
        Row positions of each run, sorted by date; runs ordered by entity
        label, then by date. The order does not depend on the order of the
        rows, so that a model sensitive to the order of its training samples
        (random forests) fits the same way.
    """
    # Rang de chaque entité dans l'ordre de ses étiquettes
    rank = np.zeros(layout.n_entities, dtype=np.intp)
    if layout.n_entities > 1:
        rank[pd.MultiIndex.from_tuples(layout.entities).argsort()] = np.arange(layout.n_entities)
    # Position de chaque date sur une grille commune ancrée sur la première date
    first = layout.dates.min()
    positions = steps_between(pd.DatetimeIndex([first] * len(layout.dates)), layout.dates, freq)
    order = np.lexsort((positions, rank[layout.codes]))
    codes, positions = layout.codes[order], positions[order]
    # Rupture : changement d'entité ou date non consécutive
    breaks = np.flatnonzero((np.diff(codes) != 0) | (np.diff(positions) != 1)) + 1
    return np.split(order, breaks)


# Wrapper des modèles darts dans un régresseur sklearn
class DartsAdapter(RegressorMixin, BaseEstimator):
    """Sklearn regressor predicting the rows of ``X`` with a darts forecasting model.

    Rows are indexed by forecast origin: a ``DatetimeIndex`` for a time series,
    or a ``MultiIndex`` whose last level holds the dates and whose other levels
    identify the entity for a panel. Nothing is shifted. The configuration of
    the darts model selects one of three modes, recorded in ``mode_``:

    - ``'row'``: a global model reading no target lag and only the covariates
      of the predicted date (``lags_future_covariates=[0]``, one step per
      pass, no output shift), such as
      ``LinearRegressionModel(lags_future_covariates=[0])``. It is fitted on
      one ``TimeSeries`` per run of consecutive rows of each entity, with
      ``X`` as future covariates, and predicts each row from its own
      features: the prediction does not depend on the gap after the training
      data, and training rows or entities unknown at fit can be predicted.
      Training rows with holes are used as they are.
    - ``'global'``: any other global model (darts regression models with
      lags, ensembles), fitted on one ``TimeSeries`` per entity. ``X`` is
      passed as past and/or future covariates when the lags of the model
      read them; models configured otherwise (torch models) do not receive
      ``X``. A row ``k`` grid steps after the last training date of its
      entity is the ``k``-th step of a forecast from that date.
    - ``'local'``: a darts local model (``NaiveDrift``, ``ExponentialSmoothing``,
      ``ARIMA``, ``Theta``...), fitted once per entity on its target history,
      with ``X`` as future covariates when the model supports them; rows are
      extracted from the forecast like in ``'global'`` mode.

    In the two extrapolation modes (``'global'`` and ``'local'``), each
    explicit ``ValueError`` marks a limit: the training rows of an entity must
    be contiguous on the grid; only rows after the last training date of a
    known entity can be predicted; a model reading covariates needs ``X`` on
    every grid date between its training data and the predicted rows, and a
    gap absent from ``X`` is never filled with invented values. The
    ``'global'`` mode is the generic path of torch models, which are not
    tested with this adapter.

    The model passed to the constructor is never modified: ``fit`` builds
    untrained copies (``untrained_model()``). Its parameters are tunable
    through ``model__<param>`` (keys of ``model_params``) and the parameters
    of a nested scikit-learn estimator through ``model__model__<param>``;
    ``set_params`` rebuilds the darts model when a value changes, so a value
    that darts refuses raises at ``set_params``. Dates must lie on a regular
    grid, holes allowed.

    Args:
        model: Unfitted darts forecasting model.

    Attributes:
        mode_: Forecasting mode: ``'row'``, ``'global'`` or ``'local'``.
        model_: Fitted copy of ``model`` (``'row'`` and ``'global'`` modes).
        entity_models_: One fitted copy of ``model`` per entity, in the order
            of ``layout_.entities`` (``'local'`` mode).
        series_: Training target of each entity as a darts ``TimeSeries``
            (``'global'`` mode).
        covariates_: Training features, kept when an extrapolating model reads
            covariates.
        layout_: Temporal layout of the training rows (entities, dates, grid
            frequency).
        target_names_: Names of the target columns (``'target'`` for an
            unnamed ``Series``).
        n_features_in_: Number of columns of ``X`` seen at fit.
        feature_names_in_: Column names of ``X`` seen at fit.

    Examples:
        >>> import pandas as pd
        >>> from darts.models import LinearRegressionModel, NaiveDrift
        >>> from tsforecast.adapters import DartsAdapter
        >>> dates = pd.date_range('2024-01-01', periods=6, freq='MS', name='date')
        >>> X = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        >>> y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0, 11.0], index=dates)
        >>> # y = 1 + 2 x learnt row by row; rows 3 and 2 months after training
        >>> row = DartsAdapter(LinearRegressionModel(lags_future_covariates=[0]))
        >>> row = row.fit(X.iloc[:3], y.iloc[:3])
        >>> row.mode_, row.predict(X.iloc[[5, 4]]).round(6).tolist()
        ('row', [11.0, 9.0])
        >>> # Drift of the target history, extrapolated through the gap
        >>> drift = DartsAdapter(NaiveDrift()).fit(X.iloc[:3], y.iloc[:3])
        >>> drift.mode_, drift.predict(X.iloc[[5, 4]]).round(6).tolist()
        ('local', [11.0, 9.0])
    """

    # Initialisation
    def __init__(self, model: Any):
        self.model = model

    # Méthode d'extraction des paramètres
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get the parameters of the adapter and, with ``deep``, of the darts model.

        darts models have no ``get_params``: their parameters are read from
        ``model_params`` and exposed as ``model__<param>``; a parameter value
        with its own ``get_params`` (scikit-learn estimator of a
        ``SKLearnModel``) also exposes ``model__<param>__<sub_param>``.

        Args:
            deep: Whether to include the parameters of the darts model.

        Returns:
            Parameter names mapped to their values.

        Examples:
            >>> from darts.models import LinearRegressionModel
            >>> adapter = DartsAdapter(LinearRegressionModel(lags_future_covariates=[0]))
            >>> adapter.get_params()['model__lags_future_covariates']
            [0]
        """
        # Extraction du modèle
        params = {"model": self.model}
        # Extraction des paramètres du modèle
        if not deep or not hasattr(self.model, "model_params"):
            return params
        # Copie indépendante des paramètres du modèle
        for name, value in self.model.model_params.items():
            params[f"model__{name}"] = value
            # Estimateur imbriqué : paramètres exposés avec un préfixe supplémentaire
            if hasattr(value, "get_params") and not isinstance(value, type):
                for sub_name, sub_value in value.get_params(deep=True).items():
                    params[f"model__{name}__{sub_name}"] = sub_value
        return params

    # Méthode d'initialisation des paramètres
    def set_params(self, **params: Any) -> "DartsAdapter":
        """Set the parameters of the adapter and of the darts model.

        ``model`` replaces the darts model; ``model__<param>`` and
        ``model__<param>__<sub_param>`` change its parameters. darts models
        cannot be changed in place: a new model is built from ``model_params``
        when at least one value differs, and the previous model is left
        untouched. Unchanged values keep the current model.

        Args:
            **params: Parameters, named as by :meth:`get_params`.

        Returns:
            The adapter.

        Raises:
            ValueError: If a parameter is unknown, or if darts refuses a value.

        Examples:
            >>> from darts.models import LinearRegressionModel
            >>> model = LinearRegressionModel(lags_future_covariates=[0])
            >>> adapter = DartsAdapter(model).set_params(model__fit_intercept=False)
            >>> adapter.model is model, adapter.get_params()['model__fit_intercept']
            (False, False)
        """
        # Cas où aucun paramètre n'est spécifié
        if not params:
            return self
        # Extraction du modèle des paramètres
        model = params.pop("model", self.model)
        # Initialisation des dictionnaires de paramètres
        changes: Dict[str, Any] = {}
        nested: Dict[str, Dict[str, Any]] = {}
        # Répartition des clés : paramètre du modèle ou d'un estimateur imbriqué
        for key, value in params.items():
            prefix, _, name = key.partition("__")
            if prefix != "model" or not name:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. Valid parameters are: ['model']."
                )
            name, _, sub_name = name.partition("__")
            if sub_name:
                nested.setdefault(name, {})[sub_name] = value
            else:
                changes[name] = value
        if not changes and not nested:
            self.model = model
            return self

        current = dict(getattr(model, "model_params", {}))
        # Paramètres admis : ceux de model_params et du constructeur ; un constructeur à
        # **kwargs (transmis à l'estimateur sous-jacent) admet tout nom
        signature = inspect.signature(type(model).__init__).parameters.values()
        accepts_any = any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature)
        valid = set(current) | {parameter.name for parameter in signature if parameter.name != "self"}
        unknown = sorted(set(changes) - valid) if not accepts_any else []
        unknown += sorted(set(nested) - set(current) - set(changes))
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) {unknown} for the darts model {type(model).__name__}; "
                f"valid parameters are {sorted(valid)}"
            )
        # Estimateurs imbriqués : clone réglé seulement si une valeur change
        for name, sub_params in nested.items():
            inner = changes.get(name, current[name])
            inner_params = inner.get_params(deep=True)
            if not all(_same_value(inner_params.get(key), value) for key, value in sub_params.items()):
                changes[name] = clone(inner).set_params(**sub_params)
        changed = {
            name: value
            for name, value in changes.items()
            if name not in current or not _same_value(current[name], value)
        }
        # Modèle darts non modifiable en place : nouveau modèle construit au changement
        if changed:
            model = type(model)(**{**copy.deepcopy(current), **changed})
        self.model = model
        return self

    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Any) -> "DartsAdapter":
        """Fit untrained copies of the darts model on the training rows.

        Args:
            X: Features indexed by a ``DatetimeIndex`` or an
                ``(entity..., date)`` ``MultiIndex``, rows in any order.
            y: Target aligned with ``X``: ``Series`` (one target) or
                ``DataFrame`` (several targets).

        Returns:
            The fitted adapter.

        Raises:
            ImportError: If darts is not installed.
            TypeError: If ``X`` is not a ``DataFrame``, if its dates are not
                ``datetime64``, or if ``model`` is not a darts regression or
                forecasting model.
            ValueError: If the dates are not on a regular grid, if a row-wise
                model uses position encoders, if the training rows of an
                extrapolating model have holes, or if darts refuses the data.
        """
        # Attributs d'un ajustement précédent retirés : aucun état n'en survit
        for name in _MODE_ATTRIBUTES:
            self.__dict__.pop(name, None)
        # Vérification des entrées et du modèle
        X, y, layout = check_adapter_input(self, X, y, reset=True)
        mode = _forecasting_mode(self.model)
        if layout.freq is None:
            raise ValueError(
                "The training dates are not on a regular grid: darts needs a frequency to build "
                "its time series"
            )

        # Cible : tableau à deux dimensions, noms mémorisés pour la sortie
        target_ndim = y.ndim
        names = [_DEFAULT_TARGET_NAME if y.name is None else y.name] if target_ndim == 1 else list(y.columns)
        targets = y.to_numpy(dtype=float).reshape(len(y), -1)
        features = X.to_numpy(dtype=float)
        runs = _contiguous_runs(layout, layout.freq)

        # Classification suivant le modèle d'entraînement
        if mode == _ROW:
            self._fit_row(layout, runs, targets, features)
        else:
            # Extrapolation : historique contigu de chaque entité
            if len(runs) > layout.n_entities:
                broken = np.unique([layout.codes[run[0]] for run in runs], return_counts=True)
                entities = [layout.entities[code] for code, count in zip(*broken) if count > 1]
                rows = f"of the entities {_preview(entities)} " if layout.is_panel else ""
                raise ValueError(
                    f"The training rows {rows}do not cover every date of the '{layout.freq}' grid; "
                    f"{type(self.model).__name__} extrapolates the target history and needs "
                    "contiguous training rows"
                )
            # Mode global
            if mode == _GLOBAL:
                self._fit_global(layout, runs, targets, features, X)
            # Mode local
            else:
                self._fit_local(layout, runs, targets, features, X)

        # Attributs ajustés communs, posés une fois l'ajustement réussi
        self.mode_ = mode
        self.layout_ = layout
        self.target_names_ = names
        self._target_ndim = target_ndim
        return self

    # Nom du modèle ajusté, pour les messages d'erreur
    def _model_name(self) -> str:
        """Class name of the fitted darts model."""
        fitted = self.model_ if hasattr(self, "model_") else self.entity_models_[0]
        return type(fitted).__name__

    # Construction d'une série darts
    @staticmethod
    def _series(dates: pd.DatetimeIndex, values: np.ndarray, freq: str) -> Any:
        """Build a darts ``TimeSeries`` from dates on the grid and a 2-D array."""
        TimeSeries = import_optional("darts", _EXTRA).TimeSeries
        return TimeSeries.from_times_and_values(pd.DatetimeIndex(dates), values, freq=freq)

    # Rôles des covariables lus dans les décalages d'un modèle global
    @staticmethod
    def _covariate_roles(model: Any) -> Tuple[bool, bool]:
        """Whether the lags of a global model read past and future covariates."""
        lags = tuple(model.extreme_lags)
        return lags[2] is not None, lags[4] is not None

    # Entraînement ligne à ligne
    def _fit_row(
        self, layout: TemporalLayout, runs: List[np.ndarray], targets: np.ndarray, features: np.ndarray
    ) -> None:
        """Fit the row-wise model on one series per run of consecutive rows."""
        # Extraction de la fréquence
        freq = layout.freq
        # Trous dans les lignes : une série par suite contiguë, aucun échantillon à cheval
        series = [self._series(layout.dates[run], targets[run], freq) for run in runs]
        # Construction des covariables
        covariates = [self._series(layout.dates[run], features[run], freq) for run in runs]
        # Initialisation du modèle vierge
        model = self.model.untrained_model()
        # Entrainement du modèle
        model.fit(series, future_covariates=covariates)
        self.model_ = model

    # Entraînement d'un modèle global extrapolant
    def _fit_global(
        self,
        layout: TemporalLayout,
        runs: List[np.ndarray],
        targets: np.ndarray,
        features: np.ndarray,
        X: pd.DataFrame,
    ) -> None:
        """Fit the global model on one series per entity, ``X`` in the roles of its lags."""
        # Extraction de la fréquence
        freq = layout.freq
        # Construction de la série pour chaque run
        series = [self._series(layout.dates[run], targets[run], freq) for run in runs]
        # Séries indexées par code d'entité pour la prévision
        by_code = [None] * layout.n_entities
        for run, entity_series in zip(runs, series):
            by_code[layout.codes[run[0]]] = entity_series
        uses_past, uses_future = self._covariate_roles(self.model)

        # Initialisation des kwargs
        kwargs = {}
        if uses_past or uses_future:
            covariates = [self._series(layout.dates[run], features[run], freq) for run in runs]
            if uses_past:
                kwargs["past_covariates"] = covariates
            if uses_future:
                kwargs["future_covariates"] = covariates

        # Initialisation 
        model = self.model.untrained_model()
        # Entraînement
        model.fit(series, **kwargs)
        self.model_ = model
        # Sauvegarde de la série et des covariables
        self.series_ = tuple(by_code)
        if kwargs:
            self.covariates_ = X.copy()

    # Entraînement d'un modèle local par entité
    def _fit_local(
        self,
        layout: TemporalLayout,
        runs: List[np.ndarray],
        targets: np.ndarray,
        features: np.ndarray,
        X: pd.DataFrame,
    ) -> None:
        """Fit one copy of the local model per entity, ``X`` as future covariates when supported."""
        # Extraction de la fréquence
        freq = layout.freq
        # Vérification que la cible est univariée ou supportée par le modèle
        if targets.shape[1] > 1 and not self.model.supports_multivariate:
            raise ValueError(f"{type(self.model).__name__} does not support several targets")
        uses_future = bool(self.model.supports_future_covariates) and features.shape[1] > 0
        # Initialisation de la liste des modèles pour chaque entité
        models = [None] * layout.n_entities
        # Une suite contiguë par entité, modèle rangé au code de l'entité
        for run in runs:
            # Initialisation du modèle vierge
            model = self.model.untrained_model()
            series = self._series(layout.dates[run], targets[run], freq)
            # Entraînement
            if uses_future:
                model.fit(series, future_covariates=self._series(layout.dates[run], features[run], freq))
            else:
                model.fit(series)
            models[layout.codes[run[0]]] = model
        self.entity_models_ = tuple(models)
        if uses_future:
            self.covariates_ = X.copy()

    # Prévision ligne à ligne
    def _predict_row(self, layout: TemporalLayout, features: np.ndarray, n_targets: int) -> np.ndarray:
        """Predict each run of consecutive rows from a one-step placeholder series ending just before it.

        With no target lag, the model never reads the values of the
        placeholder series (they are missing values): only its end date,
        which sets where the forecast starts, and the covariates matter.
        """
        # Extraction de la fréquence
        freq = self.layout_.freq
        # Déduction du pas de temps entre deux index
        step = to_dateoffset(freq)
        runs = _contiguous_runs(layout, freq)

        # Liste de la longueur de chaque run
        lengths = np.array([len(run) for run in runs])
        # Initialisation du résultat
        result = np.empty((len(features), n_targets))

        # Un appel darts par longueur de suite : darts prévoit le même nombre de pas par série
        for length in np.unique(lengths):
            group = [run for run, size in zip(runs, lengths) if size == length]
            placeholders = [
                self._series(pd.DatetimeIndex([layout.dates[run[0]] - step]), np.full((1, n_targets), np.nan), freq)
                for run in group
            ]
            # Initialisation des covariables
            covariates = [self._series(layout.dates[run], features[run], freq) for run in group]
            # Construction des prédiction
            forecasts = self.model_.predict(n=int(length), series=placeholders, future_covariates=covariates)
            # Imputation des prédictions dans le résultat
            for run, forecast in zip(group, forecasts):
                result[run] = forecast.values()
        return result

    # Covariables d'une entité sur l'entraînement et les lignes à prévoir
    def _entity_covariates(
        self, code: int, train_features: np.ndarray, dates: pd.DatetimeIndex, features: np.ndarray
    ) -> Any:
        """Join the training and predicted features of an entity; refuse missing grid dates."""
        # Extraction de la fréquence
        freq = self.layout_.freq
        # Observations d'entrainement
        train_rows = np.flatnonzero(self.layout_.codes == code)
        # Complétion des dates
        all_dates = self.layout_.dates[train_rows].append(dates)
        # Création des valeurs
        values = np.concatenate([train_features[train_rows], features])
        # Tri des valeurs par ordre chronologique
        order = np.argsort(all_dates)
        all_dates, values = all_dates[order], values[order]
        # Dates de grille sans ligne de X : jamais de valeur inventée
        expected = pd.date_range(all_dates[0], all_dates[-1], freq=freq)
        # Vérification qu'aucune date ne manque
        missing = expected.difference(all_dates)
        if len(missing):
            where = f" for entity {self.layout_.entities[code]!r}" if self.layout_.is_panel else ""
            raise ValueError(
                f"{self._model_name()} reads covariates on every date between the end of "
                f"training and the forecast date, but X has no row{where} on "
                f"{_preview(missing.strftime('%Y-%m-%d %H:%M:%S'))}; no value is invented for them: "
                "use a model without covariates or no gap"
            )
        return self._series(all_dates, values, freq)

    # Prévision par extrapolation
    def _predict_extrapolated(
        self, layout: TemporalLayout, codes: np.ndarray, features: np.ndarray, n_targets: int
    ) -> np.ndarray:
        """Forecast each entity from its last training date and extract the rows of ``X``."""
        # Extraction de la fréquence
        freq = self.layout_.freq
        # Nombre de pas de chaque ligne après la dernière date d'entraînement de son entité
        steps = steps_between(self.layout_.last_dates()[codes], layout.dates, freq)
        if (steps <= 0).any():
            raise ValueError(
                f"{self._model_name()} extrapolates the target history: it cannot predict "
                f"rows at or before the last training date of their entity "
                f"({_preview(layout.dates[steps <= 0].unique().strftime('%Y-%m-%d %H:%M:%S'))})"
            )
        # Utilisation des covariables si disponibles
        uses_covariates = hasattr(self, "covariates_")
        fitted_codes = np.unique(codes)
        rows_of = {code: np.flatnonzero(codes == code) for code in fitted_codes}
        horizons = {code: int(steps[rows].max()) for code, rows in rows_of.items()}
        # Spécification des covariables
        covariates = {}
        if uses_covariates:
            train_features = self.covariates_.to_numpy(dtype=float)
            covariates = {
                code: self._entity_covariates(code, train_features, layout.dates[rows], features[rows])
                for code, rows in rows_of.items()
            }
        # Initialisation du résultat
        result = np.empty((len(features), n_targets))
        # Prédiction en mode local
        if self.mode_ == _LOCAL:
            for code, rows in rows_of.items():
                kwargs = {"future_covariates": covariates[code]} if uses_covariates else {}
                forecast = self.entity_models_[code].predict(n=horizons[code], **kwargs)
                result[rows] = forecast.values()[steps[rows] - 1]
            return result

        # Modèle global : un appel par horizon, darts prévoyant le même nombre de pas par série
        uses_past, uses_future = self._covariate_roles(self.model_)
        for horizon in sorted(set(horizons.values())):
            group = [code for code in fitted_codes if horizons[code] == horizon]
            kwargs = {}
            if uses_past:
                kwargs["past_covariates"] = [covariates[code] for code in group]
            if uses_future:
                kwargs["future_covariates"] = [covariates[code] for code in group]
            forecasts = self.model_.predict(n=horizon, series=[self.series_[code] for code in group], **kwargs)
            for code, forecast in zip(group, forecasts):
                rows = rows_of[code]
                result[rows] = forecast.values()[steps[rows] - 1]
        return result

    # Méthode de prédiction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the rows of ``X``.

        Args:
            X: Features of the rows to predict, with the columns seen at fit;
                rows in any order. In the extrapolation modes, the entities
                must be known at fit and the rows after their last training
                date.

        Returns:
            Predictions in the row order of ``X``: shape ``(n_samples,)`` for a
            ``Series`` target, ``(n_samples, n_targets)`` for a ``DataFrame``.

        Raises:
            ImportError: If darts is not installed.
            NotFittedError: If the adapter is not fitted.
            ValueError: If the columns differ from those seen at fit, if a date
                is off the training grid, or, for an extrapolating model, if an
                entity is unknown at fit, if a row is not after the training
                data of its entity, or if the model reads covariates on dates
                absent from ``X``.
        """
        # Vérification que l'estimateur est entrainé
        check_is_fitted(self)
        # Vérification de l'entrée X
        X, _, layout = check_adapter_input(self, X, reset=False, freq=self.layout_.freq)
        # Nombre de variables à prédire
        n_targets = len(self.target_names_)
        # Vérification que le jeu de données de prédictions est non vide
        if len(X) == 0:
            return np.empty((0,) if self._target_ndim == 1 else (0, n_targets))
        # Conversion des features en numpy array numérique
        features = X.to_numpy(dtype=float)

        # Distinction de la méthode de prédiction suivant le mode
        if self.mode_ == _ROW:
            values = self._predict_row(layout, features, n_targets)
        else:
            # Entités inconnues : aucun historique à extrapoler
            codes = self.layout_.match_entities(layout)
            if (codes < 0).any():
                unknown = [layout.entities[code] for code in np.unique(layout.codes[codes < 0])]
                raise ValueError(f"Entities unknown at fit: {_preview(unknown)}")
            values = self._predict_extrapolated(layout, codes, features, n_targets)
        return values.reshape(len(X)) if self._target_ndim == 1 else values
