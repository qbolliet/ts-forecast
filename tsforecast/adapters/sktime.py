"""Sklearn adapter for sktime forecasters: ``SktimeAdapter``.

The wrapped forecaster predicts the rows of ``X``: the forecasting horizon is
made of the dates of these rows, as absolute dates, never of a user parameter.
Time series and panels (entity levels, then a date level) are passed to sktime
in its native formats, and the predictions are returned as an array in the row
order of ``X``.
"""
# Importation des modules
# Modules de base
from typing import Any, List, Tuple, Union

# Modules de manipulation de données
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

# tsforecast
from tsforecast.adapters._base import (
    TemporalLayout,
    _preview,
    check_adapter_input,
    import_optional,
    restore_row_order,
    steps_between,
)
from tsforecast.utils.frequency import to_pandas_freq

# Extra installant sktime
_EXTRA = "adapters-sktime"
# Nom de la colonne de cible transmise à sktime quand la cible n'a pas de nom
_DEFAULT_TARGET_NAME = "target"


# Fonction auxiliaire de lecture d'un tag sktime
def _tag(forecaster: Any, name: str, default: Any) -> Any:
    """Read a tag of a sktime forecaster, with a default when the tag is unknown.

    Args:
        forecaster: Sktime forecaster.
        name: Name of the tag (``'capability:exogenous'``, ...).
        default: Value returned when the forecaster does not define the tag.

    Returns:
        The value of the tag, or ``default``.
    """
    return forecaster.get_tag(name, tag_value_default=default, raise_error=False)


# Fonction auxiliaire : bornes temporelles et nombre de lignes de chaque entité
def _entity_bounds(layout: TemporalLayout) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray]:
    """Return the first date, the last date and the row count of each entity, in code order.

    Args:
        layout: Temporal and entity structure of the rows.

    Returns:
        Triple ``(first, last, counts)``: first and last date of each entity
        and its number of rows, indexed by entity code.
    """
    first = pd.Series(layout.dates).groupby(layout.codes, sort=True).min()
    return (
        pd.DatetimeIndex(first),
        layout.last_dates(),
        np.bincount(layout.codes, minlength=layout.n_entities),
    )


# Fonction auxiliaire : index d'origine des entités, un libellé par code
def _entity_labels(index: pd.Index, layout: TemporalLayout) -> pd.Index:
    """Return the entity part of ``index`` for each entity code (dtypes of the input kept).

    Args:
        index: Panel ``MultiIndex`` of the rows described by ``layout``.
        layout: Temporal and entity structure of ``index``.

    Returns:
        Index of the entity levels, one entry per entity code, in code order.
    """
    _, first_rows = np.unique(layout.codes, return_index=True)
    return index[first_rows].droplevel(-1)


# Fonction auxiliaire : index complet de la grille de chaque entité
def _complete_index(index: pd.Index, layout: TemporalLayout) -> pd.Index:
    """Index covering every grid date between the first and the last date of each entity.

    Args:
        index: Index of the rows described by ``layout``.
        layout: Temporal and entity structure of ``index``; its frequency is known.

    Returns:
        A ``DatetimeIndex`` for a time series, or a panel ``MultiIndex`` with
        the names of ``index``, dates in the unit of the input.
    """
    first, last, _ = _entity_bounds(layout)
    grid = pd.date_range(first.min(), last.max(), freq=to_pandas_freq(layout.freq))
    start, stop = grid.get_indexer(first), grid.get_indexer(last)
    # Positions de grille de chaque entité, entité par entité
    positions = np.concatenate([np.arange(a, b + 1) for a, b in zip(start, stop)])
    dates = pd.DatetimeIndex(grid[positions], name=layout.time_name).as_unit(layout.unit)
    if not layout.is_panel:
        return dates
    codes = np.repeat(np.arange(layout.n_entities), stop - start + 1)
    labels = _entity_labels(index, layout).take(codes)
    arrays = [labels.get_level_values(level) for level in range(labels.nlevels)] + [dates]
    return pd.MultiIndex.from_arrays(arrays, names=index.names)


# Fonction auxiliaire : fréquence posée sur un index de dates contigu
def _with_freq(obj: Union[pd.Series, pd.DataFrame], freq: str) -> Union[pd.Series, pd.DataFrame]:
    """Set ``freq`` on the ``DatetimeIndex`` of ``obj`` when its dates are contiguous on the grid.

    Args:
        obj: Series or frame indexed by dates.
        freq: Pandas alias of the grid frequency.

    Returns:
        ``obj`` with ``freq`` set on its index, or ``obj`` unchanged when its
        index is a ``MultiIndex``, has fewer than two dates or skips grid dates.
    """
    dates = obj.index
    if isinstance(dates, pd.MultiIndex) or len(dates) < 2:
        return obj
    if steps_between(dates[:1], dates[-1:], freq)[0] + 1 != len(dates):
        return obj
    return obj.set_axis(pd.DatetimeIndex(dates, freq=to_pandas_freq(freq)), axis=0)


# Wrapper des forecasters sktime dans un régresseur sklearn
class SktimeAdapter(RegressorMixin, BaseEstimator):
    """Sklearn regressor predicting the rows of ``X`` with a sktime forecaster.

    Rows are indexed by forecast origin: a ``DatetimeIndex`` for a time series,
    or a ``MultiIndex`` whose last level holds the dates and whose other levels
    identify the entity for a panel. ``fit`` gives the forecaster the training
    target (and ``X`` as exogenous data when the forecaster uses it);
    ``predict`` forecasts the dates of the rows of ``X``, entity by entity, as
    an absolute horizon: a row ``k`` grid steps after the last training date of
    its entity is a ``k``-step-ahead forecast. Nothing is shifted.

    The forecaster passed to the constructor is never modified: ``fit`` fits a
    clone stored in ``forecaster_``. Its parameters are tunable through
    ``forecaster__<param>``.

    On a panel, sktime fits a forecaster without native panel support once per
    entity (``NaiveForecaster``, ``YfromX``, ``SARIMAX``...): the model is local
    and an entity unknown at fit cannot be predicted. For a regression of ``y``
    on ``X`` pooled over the entities, use the scikit-learn regressor directly.

    Constraints, each raising an explicit ``ValueError``:

    - the dates must lie on a regular grid (holes are allowed);
    - forecasters needing the horizon at fit (``requires-fh-in-fit``, direct
      reductions) are refused;
    - training rows with holes (rows dropped by ``DropNaNTarget``) require a
      forecaster that handles missing values; a forecaster ignoring ``X`` sees
      the holes as missing target values on the complete grid;
    - a forecaster that needs ``X`` on every date between the training data
      and the forecast (``capability:non_contiguous_X`` false, e.g.
      ``SARIMAX``) cannot skip a gap absent from ``X``: no value is invented.

    On a panel, sktime infers the frequency of each entity from its training
    dates: a forecaster that needs it (``NaiveForecaster``) requires at least
    three training dates per entity.

    Args:
        forecaster: Unfitted sktime forecaster.

    Attributes:
        forecaster_: Fitted clone of ``forecaster``.
        target_names_: Names of the target columns given to sktime
            (``'target'`` for an unnamed ``Series``).
        layout_: Temporal layout of the training rows (entities, dates, grid
            frequency).
        n_features_in_: Number of columns of ``X`` seen at fit.
        feature_names_in_: Column names of ``X`` seen at fit.

    Examples:
        >>> import pandas as pd
        >>> from sktime.forecasting.naive import NaiveForecaster
        >>> from tsforecast.adapters import SktimeAdapter
        >>> dates = pd.date_range('2024-01-01', periods=6, freq='MS', name='date')
        >>> X = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        >>> y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=dates)
        >>> model = SktimeAdapter(NaiveForecaster(strategy='drift')).fit(X.iloc[:3], y.iloc[:3])
        >>> # Rows 3 and 2 months after the last training date, a one-month gap left out
        >>> model.predict(X.iloc[[5, 4]]).round(6)
        array([6., 5.])
    """

    # Initialisation
    def __init__(self, forecaster: Any):
        self.forecaster = forecaster

    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Any) -> "SktimeAdapter":
        """Fit a clone of the forecaster on the training rows.

        Args:
            X: Features indexed by a ``DatetimeIndex`` or an
                ``(entity..., date)`` ``MultiIndex``, rows in any order.
            y: Target aligned with ``X``: ``Series`` (one target) or
                ``DataFrame`` (several targets).

        Returns:
            The fitted adapter.

        Raises:
            ImportError: If sktime is not installed.
            TypeError: If ``X`` is not a ``DataFrame``, if its dates are not
                ``datetime64``, or if ``forecaster`` is not a sktime forecaster.
            ValueError: If the dates are not on a regular grid, if the
                forecaster needs the horizon at fit, or if the training rows
                have holes that the forecaster cannot handle.
        """
        # Importation de sktime
        base = import_optional("sktime.forecasting.base", _EXTRA)
        # Vérification des entrées X et y
        X, y, layout = check_adapter_input(self, X, y, reset=True)
        # Vérification de la fréquence
        if layout.freq is None:
            raise ValueError(
                "The training dates are not on a regular grid: sktime needs a frequency "
                "to forecast the dates of the rows of X"
            )
        # Vérification que le forecaster
        if not isinstance(self.forecaster, base.BaseForecaster):
            raise TypeError(
                f"forecaster must be a sktime forecaster, got {type(self.forecaster).__name__}"
            )
        forecaster = clone(self.forecaster)
        # Vérification de l'horizon
        if _tag(forecaster, "requires-fh-in-fit", False):
            raise ValueError(
                f"{type(forecaster).__name__} needs the forecasting horizon at fit; "
                "the horizon of SktimeAdapter is only known at predict (dates of the rows of X)"
            )
        # Forecaster qui ignore X : X jamais transmis
        uses_X = bool(_tag(forecaster, "capability:exogenous", True))

        # Cible au format sktime : DataFrame, colonnes mémorisées pour la sortie
        target_ndim = y.ndim
        if isinstance(y, pd.Series):
            y = y.to_frame(_DEFAULT_TARGET_NAME if y.name is None else y.name)

        # Lignes triées par entité puis par date, X et y de la même façon
        y_fit = y.sort_index()
        X_fit = X.sort_index() if uses_X else None

        # Trous dans la grille d'une entité : dates manquantes
        first, last, n_rows = _entity_bounds(layout)
        holes = steps_between(first, last, layout.freq) + 1 > n_rows
        if holes.any():
            if not _tag(forecaster, "capability:missing_values", False):
                entities = [layout.entities[code] for code in np.flatnonzero(holes)]
                rows = f"of the entities {_preview(entities)} " if layout.is_panel else ""
                raise ValueError(
                    f"The training rows {rows}do not cover every date of the "
                    f"'{layout.freq}' grid, and {type(forecaster).__name__} does not handle missing "
                    "values: train on contiguous rows"
                )
            # Forecaster sans X : cible manquante déclarée sur la grille complète
            if not uses_X:
                y_fit = y_fit.reindex(_complete_index(y.index, layout)).sort_index()

        if not layout.is_panel:
            y_fit = _with_freq(y_fit, layout.freq)
            X_fit = None if X_fit is None else _with_freq(X_fit, layout.freq)
        forecaster.fit(y=y_fit, X=X_fit)

        # Attributs ajustés, posés une fois l'ajustement réussi
        self.forecaster_ = forecaster
        self.layout_ = layout
        self.target_names_ = list(y.columns)
        self._target_ndim = target_ndim
        return self

    # Contrôle des dates de X exigées par un forecaster à X contigu
    def _check_contiguous_X(self, layout: TemporalLayout, codes: np.ndarray) -> None:
        """Refuse rows of ``X`` that leave grid dates without ``X`` after the training data.

        Args:
            layout: Temporal and entity structure of the rows to predict.
            codes: Training entity code of each row to predict.

        Raises:
            ValueError: If an entity has no row of ``X`` on a grid date between
                the end of its training data and its last date to predict.
        """
        # Extraction de la fréquence
        freq = self.layout_.freq
        # Extraction de la dernière date d'entrainement
        last_train = self.layout_.last_dates()[codes]
        # Extraction des dates postérieures à la dernière date d'entrainement
        after = np.asarray(layout.dates > last_train)
        # Par entité de test : nombre de dates postérieures à l'entraînement et dernière date
        n_after = np.bincount(layout.codes[after], minlength=layout.n_entities)
        last_test = pd.Series(layout.dates).groupby(layout.codes, sort=True).max()
        _, first_rows = np.unique(layout.codes, return_index=True)
        # Fin de la période d'entrainement
        train_end = last_train[first_rows]
        # Calcul du nombre de pas entre la fin du train et la fin du test
        needed = np.maximum(steps_between(train_end, pd.DatetimeIndex(last_test), freq), 0)
        incomplete = np.flatnonzero(n_after < needed)
        # Complétion de l'index si nécessaire
        if incomplete.size:
            code = incomplete[0]
            # Crétion de l'index
            expected = pd.date_range(train_end[code], last_test.iloc[code], freq=to_pandas_freq(freq))[1:]
            missing = expected.difference(layout.dates[layout.codes == code])
            entity = layout.entities[code]
            where = f" for entity {entity!r}" if layout.is_panel else ""
            raise ValueError(
                f"{type(self.forecaster_).__name__} needs X on every date between the end of "
                f"training and the forecast date, but X has no row{where} on "
                f"{_preview(missing.strftime('%Y-%m-%d %H:%M:%S'))}; no value is invented for "
                "them: use a forecaster that accepts non-contiguous X (YfromX, ...) or no gap"
            )

    # Prévision entité par entité par les forecasters vectorisés de sktime
    def _predict_per_entity(
        self, X: pd.DataFrame, layout: TemporalLayout, codes: np.ndarray, columns: List[Any]
    ) -> np.ndarray:
        """Forecast each entity of ``X`` with its own dates and features.

        Args:
            X: Features of the rows to predict.
            layout: Temporal and entity structure of ``X``.
            codes: Training entity code of each row of ``X``.
            columns: Target columns to extract from the sktime forecasts.

        Returns:
            Predictions of shape ``(len(X), len(columns))``, in the order of
            the rows of ``X``.
        """
        # Importation de l'horizon de prévision de sktime
        ForecastingHorizon = import_optional("sktime.forecasting.base", _EXTRA).ForecastingHorizon
        # Extraction des prédicteurs
        table = self.forecaster_.forecasters_
        # Inialisation du jeu de données résultat
        result = np.empty((len(X), len(columns)))
        for code in np.unique(layout.codes):
            rows = np.flatnonzero(layout.codes == code)
            rows = rows[np.argsort(layout.dates[rows])]
            dates = pd.DatetimeIndex(layout.dates[rows], name=layout.time_name)
            X_entity = _with_freq(X.iloc[rows].set_axis(dates, axis=0), self.layout_.freq)
            # Création de l'horizon de prédiction
            fh = ForecastingHorizon(dates, is_relative=False, freq=self.layout_.freq)
            # Ligne de la table des forecasters : clé simple ou tuple selon les niveaux d'entité
            entity = self.layout_.entities[codes[rows[0]]]
            cells = table.loc[entity[0] if len(entity) == 1 else entity]
            # Concaténation des prédictions
            forecasts = pd.concat([cell.predict(fh=fh, X=X_entity) for cell in cells], axis=1)
            result[rows] = restore_row_order(forecasts[columns].to_numpy(), forecasts.index, dates)
        return result

    # Méthode de prédiction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast the dates of the rows of ``X``.

        Args:
            X: Features of the rows to predict, with the columns seen at fit;
                rows in any order, entities known at fit.

        Returns:
            Predictions in the row order of ``X``: shape ``(n_samples,)`` for a
            ``Series`` target, ``(n_samples, n_targets)`` for a ``DataFrame``.

        Raises:
            ImportError: If sktime is not installed.
            NotFittedError: If the adapter is not fitted.
            ValueError: If an entity is unknown at fit, if the columns differ
                from those seen at fit, or if the forecaster needs ``X`` on
                dates absent from ``X``.
        """
        # Vérification que l'estimateur est entraîné
        check_is_fitted(self)
        # Importation de l'horizon de prédiction
        ForecastingHorizon = import_optional("sktime.forecasting.base", _EXTRA).ForecastingHorizon
        # Vérification de l'entrée X
        X, _, layout = check_adapter_input(self, X, reset=False, freq=self.layout_.freq)
        # Nombre de variables à prévoir
        n_targets = len(self.target_names_)
        # Vérification que le jeu de données de prédictions n'est pas vide
        if len(X) == 0:
            return np.empty((0,) if self._target_ndim == 1 else (0, n_targets))

        # Entités inconnues : aucun modèle ajusté pour elles
        codes = self.layout_.match_entities(layout)
        if (codes < 0).any():
            unknown = [layout.entities[code] for code in np.unique(layout.codes[codes < 0])]
            raise ValueError(f"Entities unknown at fit: {_preview(unknown)}")

        # Utilisation de variables exogènes par l'estimateur
        uses_X = bool(_tag(self.forecaster_, "capability:exogenous", True))
        # Vérification de la contiguité des observations de X
        if uses_X and not _tag(self.forecaster_, "capability:non_contiguous_X", True):
            self._check_contiguous_X(layout, codes)

        columns = self.target_names_
        dates = layout.dates.unique().sort_values()

        # Lignes de test couvrant toutes les entités ajustées à toutes les dates : un appel sktime
        rectangular = len(X) == self.layout_.n_entities * len(dates)
        if not uses_X or rectangular:
            # Construction du jeu de données de variables exogènes attendues par sktime
            X_sk = None
            if uses_X:
                X_sk = X.sort_index()
                X_sk = X_sk if layout.is_panel else _with_freq(X_sk, self.layout_.freq)
            # Construction de l'horizon de prédiction
            fh = ForecastingHorizon(dates, is_relative=False, freq=self.layout_.freq)
            # Le predict de sktime mémorise l'horizon dans le forecaster, sans effet sur ses prévisions
            forecasts = self.forecaster_.predict(fh=fh, X=X_sk)
            values = restore_row_order(forecasts[columns].to_numpy(), forecasts.index, X.index)
        # Forecaster vectorisé par sktime : un forecaster ajusté par entité, chacune à ses dates
        elif hasattr(self.forecaster_, "forecasters_"):
            values = self._predict_per_entity(X, layout, codes, columns)
        else:
            raise ValueError(
                f"{type(self.forecaster_).__name__} forecasts every entity on the same dates and "
                f"needs X on each of them; X lacks {self.layout_.n_entities * len(dates) - len(X)} "
                "(entity, date) row(s)"
            )
        values = np.asarray(values, dtype=float)
        return values.reshape(len(X)) if self._target_ndim == 1 else values.reshape(len(X), n_targets)
