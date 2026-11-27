"""Online aggregation of expert forecasts (BOA, MLpol) as a scikit-learn regressor.

``OnlineAggregator`` combines the forecasts of several experts with weights learnt
sequentially, in date order, by Bernstein Online Aggregation or by the
polynomially weighted average with multiple learning rates. As a meta-estimator it
fits its experts and aggregates their out-of-sample forecasts; in ``'passthrough'``
mode the columns of ``X`` are already expert forecasts. Time series and panels
(one aggregation for the whole panel or one per entity) are supported.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from tsforecast.adapters._base import TemporalLayout, check_adapter_input
from tsforecast.ensemble._algorithms import (
    ALGORITHMS,
    BOAState,
    Loss,
    MLpolState,
    make_loss,
    run,
)

# Valeur du paramètre experts désignant des prévisions d'experts déjà calculées
PASSTHROUGH = "passthrough"
# Préfixe des paramètres imbriqués des experts (experts__<nom>__<paramètre>)
_EXPERTS_PREFIX = "experts__"
# Modes de mutualisation de l'agrégation sur un panel
POOLINGS: Tuple[str, ...] = ("global", "per_entity")
# Dictionnaire classe d'état -> nom d'algorithme
_STATE_ALGORITHMS: Dict[type, str] = {BOAState: "BOA", MLpolState: "MLpol"}


# Fonction de mise en forme des prévisions d'un expert
def _as_column(values: Any, n_rows: int, name: str) -> np.ndarray:
    """Return the forecasts of an expert as a float array of shape ``(n_rows,)``.

    Args:
        values: Output of the ``predict`` method of the expert.
        n_rows: Expected number of rows.
        name: Name of the expert, for the error message.

    Returns:
        The forecasts, one per row.

    Raises:
        ValueError: If the output does not hold one value per row.
    """
    array = np.asarray(values, dtype=float)
    # Sortie en colonne unique (n, 1) acceptée et aplatie
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != (n_rows,):
        raise ValueError(f"Expert {name!r} returned shape {array.shape}, expected ({n_rows},)")
    return array


# Fonction d'ajustement d'un clone d'expert puis de prévision de lignes de test
def _fit_predict(expert: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, name: str) -> np.ndarray:
    """Fit ``expert`` on the training rows and forecast the test rows.

    Args:
        expert: Unfitted clone of the expert.
        X_train: Training features.
        y_train: Training target.
        X_test: Rows to forecast.
        name: Name of the expert.

    Returns:
        Forecasts of the test rows, shape ``(len(X_test),)``.
    """
    return _as_column(expert.fit(X_train, y_train).predict(X_test), len(X_test), name)


# Fonction d'ajustement d'un clone d'expert sur tout l'entraînement
def _fit(expert: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """Fit ``expert`` on every training row and return it."""
    return expert.fit(X, y)


# Fonction de construction de l'index des entités d'un panel
def _entity_index(entities: List[tuple], names: Tuple[Any, ...]) -> pd.Index:
    """Index labelling the entities: an ``Index`` for one level, a ``MultiIndex`` otherwise.

    Args:
        entities: Entity keys (tuples).
        names: Names of the entity levels.

    Returns:
        One label per entity, the levels keeping the dtype of the data.
    """
    # Cas d'un seul niveau d'entité
    if len(names) == 1:
        return pd.Index([key[0] for key in entities], name=names[0])
    return pd.MultiIndex.from_tuples(entities, names=list(names))


# Classe d'agrégation d'experts en ligne
class OnlineAggregator(RegressorMixin, MetaEstimatorMixin, BaseEstimator):
    """Online aggregation of expert forecasts with BOA or MLpol weights.

    The forecast of a row is ``sum_k w_k * x_k``, where ``x_k`` is the forecast
    of expert ``k`` and the weights ``w`` lie on the simplex. The weights are
    learnt by playing the training rows in date order: before each date, the
    current weights forecast its rows; once the target is revealed, every
    expert's excess loss over the mixture updates the weights. ``predict``
    uses the final weights, which no row to predict ever updates.

    Algorithms (``aggregation``):

    - ``'BOA'``: Bernstein Online Aggregation with one adaptive learning rate
      per expert. With ``l`` the excess losses and ``eta`` the learning rates
      of the previous round, the weights are
      ``w_k ∝ eta_k * exp(-eta_k * L_k) * pi_k``, where ``pi`` is the prior
      (``init_weights``), ``L += (l * (1 + eta * l) + E * 1{eta * l > 1/2}) / 2``,
      ``E`` is a power of two at least twice the largest ``|l|`` seen, and
      ``eta = min(1 / E, sqrt(log(1 / pi) / sum of l**2))``.
    - ``'MLpol'``: polynomially weighted average with multiple learning rates,
      without tuning parameter: ``w_k ∝ eta_k * max(R_k, 0)``, with ``R`` the
      cumulative regrets ``-l`` and ``1 / eta += r**2 + B_t**2 - B_{t-1}**2``
      (``B_t`` the largest regret seen). The weights do not depend on the
      scale of ``y``. While no regret is positive the weights are
      ``init_weights``.

    With ``loss_gradient=True`` the excess loss of expert ``k`` is linearised
    at the forecast ``p`` of the mixture, ``loss'(p, y) * (x_k - p)``, which
    compares the mixture with the best convex combination of the experts;
    with ``False`` it is ``loss(x_k, y) - loss(p, y)``, which compares it with
    the best single expert.

    Modes (``experts``):

    - a list of ``(name, estimator)``: meta-estimator. ``fit`` forecasts the
      training rows **out of sample** with the internal splitter ``cv`` (each
      expert is fitted on the training part of a fold and forecasts its test
      part, never a row it was fitted on: in-sample forecasts would favour the
      expert that overfits), aggregates these forecasts, then refits every
      expert on all the training rows (``experts_``), which forecast at
      ``predict``. Rows of the first training window of ``cv`` have no
      out-of-sample forecast and are not aggregated.
    - ``'passthrough'``: the columns of ``X`` are expert forecasts (computed
      by other estimators, for example with
      ``cross_validate(..., return_estimator=True, return_indices=True)``);
      every row is aggregated.

    Panels (``pooling``): ``'global'`` runs one aggregation for the whole panel;
    the rows of one date form a single round (same weights, mean excess
    loss), since in real time the targets of the other entities at that date
    are not known yet. ``'per_entity'`` runs one aggregation per entity, plus
    the global one, whose weights forecast an entity unknown at fit. On a time
    series both poolings are the same.

    Cost: in meta mode, ``fit`` fits ``n_experts * (n_folds + 1)`` experts; a
    cross-validation multiplies this by its number of folds and a grid search
    also by its number of candidates (3 experts, 5 internal folds, 5 outer
    folds, 10 candidates: 900 fits). To bound it: fewer internal folds (a
    smaller integer ``cv`` tests larger blocks), a splitter with a rolling
    window (``max_train_size``), ``n_jobs``, and, to tune only ``aggregation``, ``loss`` or
    ``pooling``, the passthrough mode on out-of-sample forecasts computed once
    (``oos_predictions_``): no expert is fitted again.

    Leakage: the final weights only read training targets. With a horizon
    ``h > 1`` (rows indexed by forecast origin), the target of a row is only
    known ``h`` steps later in real time: the weights used *during* the
    sequence (``weights_history_``) are those of an idealised online run, while
    the final weights are legitimate as soon as the outer splitter leaves a gap
    of ``h + d - 1`` rows. The internal splitter built from an integer ``cv``
    has no gap; pass a splitter built with the gap of the forecasting task to
    reproduce the staleness of real-time expert forecasts.

    Args:
        experts: List of ``(name, estimator)`` pairs (meta-estimator), or
            ``'passthrough'`` (the columns of ``X`` are expert forecasts).
        aggregation: ``'BOA'`` or ``'MLpol'``.
        loss: ``'mse'``, ``'mae'``, ``'pinball'`` or a
            :class:`~tsforecast.ensemble.Loss`.
        loss_gradient: Whether to aggregate on the loss linearised at the
            forecast of the mixture (gradient trick).
        quantile: Level of the pinball loss, in ``(0, 1)``.
        init_weights: ``'uniform'`` or strictly positive weights, one per
            expert (normalised): prior of BOA, initial weights of MLpol.
        pooling: ``'global'`` or ``'per_entity'``.
        cv: Internal splitter of the meta-estimator: a splitter whose test
            folds never overlap (a ``tsforecast`` splitter, typically with the
            gap of the task), or an integer number of folds: the last ``cv``
            blocks of ``n_dates // (cv + 1)`` consecutive dates, common to the
            entities, each forecast by experts fitted on all earlier dates,
            without gap. Unused in passthrough mode.
        n_jobs: Number of jobs fitting the experts (joblib).

    Attributes:
        n_features_in_: Number of columns of ``X``.
        feature_names_in_: Columns of ``X`` (when they are strings).
        expert_names_: Names of the experts (columns of ``X`` in passthrough
            mode).
        experts_: Experts fitted on every training row (meta mode).
        oos_predictions_: Out-of-sample forecasts of the experts on the
            aggregated rows, in the row order of ``X`` (meta mode).
        weights_: Final weights of the global aggregation, indexed by expert.
        entity_weights_: Final weights of each entity, one row per entity and
            one column per expert (``pooling='per_entity'`` on a panel).
        weights_history_: Weights used to forecast each aggregated row, before
            its target was seen, in date order.
        prior_: Normalised initial weights.
        state_: State of the global aggregation.
        entity_states_: State of each entity's aggregation, by entity key
            (``pooling='per_entity'`` on a panel).
        entity_names_: Names of the entity levels (empty for a series).
        last_date_: Last aggregated date, in the unit of the input.

    References:
        Wintenberger, O. (2017). Optimal learning with Bernstein Online
        Aggregation. Machine Learning, 106(1), 119-141.

        Gaillard, P., Stoltz, G. and van Erven, T. (2014). A second-order bound
        with excess losses. Conference on Learning Theory, 176-196.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=6, freq='MS')
        >>> y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=dates)
        >>> X = pd.DataFrame({'close': y + 0.1, 'far': y + 2.0}, index=dates)
        >>> model = OnlineAggregator('passthrough', aggregation='MLpol').fit(X, y)
        >>> model.weights_.to_dict()
        {'close': 1.0, 'far': 0.0}
        >>> model.predict(X.iloc[-1:]).round(2).tolist()
        [6.1]
    """

    # Initialisation : stockage des paramètres, sans validation
    def __init__(
        self,
        experts: Union[List[Tuple[str, Any]], str],
        *,
        aggregation: str = "BOA",
        loss: Union[str, Loss] = "mse",
        loss_gradient: bool = True,
        quantile: float = 0.5,
        init_weights: Union[str, Any] = "uniform",
        pooling: str = "global",
        cv: Any = 5,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Store the parameters unchanged; they are validated at ``fit``.

        Args:
            experts: ``(name, estimator)`` pairs or ``'passthrough'``.
            aggregation: ``'BOA'`` or ``'MLpol'``.
            loss: Loss name or :class:`~tsforecast.ensemble.Loss`.
            loss_gradient: Whether to use the linearised loss.
            quantile: Level of the pinball loss.
            init_weights: ``'uniform'`` or initial weights.
            pooling: ``'global'`` or ``'per_entity'``.
            cv: Internal splitter or number of folds.
            n_jobs: Number of jobs fitting the experts.
        """
        self.experts = experts
        self.aggregation = aggregation
        self.loss = loss
        self.loss_gradient = loss_gradient
        self.quantile = quantile
        self.init_weights = init_weights
        self.pooling = pooling
        self.cv = cv
        self.n_jobs = n_jobs

    # Méthode de lecture des couples (nom, expert) bien formés
    def _named_experts(self) -> List[Tuple[str, Any]]:
        """Return the ``(name, estimator)`` pairs of ``experts``, or ``[]`` when it is not such a list."""
        # Cas passthrough ou liste mal formée : aucun paramètre imbriqué
        if isinstance(self.experts, str) or not isinstance(self.experts, (list, tuple)):
            return []
        if not all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in self.experts):
            return []
        return list(self.experts)

    # Paramètres, dont ceux des experts sous la forme experts__<nom>__<paramètre>
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the parameters, with the experts and their parameters when ``deep``.

        The experts are named ``experts__<name>`` and their parameters
        ``experts__<name>__<param>``, the names ``GridSearchCV`` searches.

        Args:
            deep: Whether to include the experts and their parameters.

        Returns:
            Parameter names mapped to their values.

        Examples:
            >>> from sklearn.linear_model import Ridge
            >>> params = OnlineAggregator([('ridge', Ridge(alpha=2.0))]).get_params()
            >>> params['experts__ridge__alpha']
            2.0
        """
        params = super().get_params(deep=False)
        # Experts et paramètres des experts, préfixés par experts__<nom>
        if deep:
            for name, expert in self._named_experts():
                params[f"{_EXPERTS_PREFIX}{name}"] = expert
                if hasattr(expert, "get_params"):
                    for key, value in expert.get_params(deep=True).items():
                        params[f"{_EXPERTS_PREFIX}{name}__{key}"] = value
        return params

    # Modification des paramètres, dont ceux des experts
    def set_params(self, **params: Any) -> "OnlineAggregator":
        """Set parameters: the experts list, an expert by name, or a parameter of an expert.

        Order: ``experts`` first, then ``experts__<name>`` (the expert is
        replaced in a new list), then ``experts__<name>__<param>`` (set on the
        expert), then the other parameters.

        Args:
            **params: Parameter names mapped to their new values.

        Returns:
            The aggregator itself.

        Raises:
            ValueError: If a name designates no expert or no parameter.

        Examples:
            >>> from sklearn.linear_model import Ridge
            >>> model = OnlineAggregator([('ridge', Ridge())]).set_params(experts__ridge__alpha=5.0)
            >>> model.experts[0][1].alpha
            5.0
        """
        # Liste des experts remplacée en premier
        if "experts" in params:
            self.experts = params.pop("experts")
        # Paramètres imbriqués des experts, répartis en remplacements par nom et paramètres par expert
        replacements: Dict[str, Any] = {}
        nested: Dict[str, Dict[str, Any]] = {}
        for key in [key for key in params if key.startswith(_EXPERTS_PREFIX)]:
            name, _, sub_key = key[len(_EXPERTS_PREFIX):].partition("__")
            value = params.pop(key)
            # Cas d'un paramètre d'expert, puis cas d'un expert entier
            if sub_key:
                nested.setdefault(name, {})[sub_key] = value
            else:
                replacements[name] = value
        if replacements or nested:
            # Dictionnaire nom -> position de l'expert dans la liste
            positions = {name: k for k, (name, _) in enumerate(self._named_experts())}
            unknown = sorted((set(replacements) | set(nested)) - set(positions))
            if unknown:
                raise ValueError(f"Invalid expert name(s) {unknown} for {type(self).__name__}")
            # Nouvelle liste : la liste reçue au constructeur n'est jamais modifiée
            experts = list(self.experts)
            for name, expert in replacements.items():
                experts[positions[name]] = (name, expert)
            self.experts = experts
            # Paramètres posés sur l'expert, comme pour les étapes d'une Pipeline
            for name, sub_params in nested.items():
                experts[positions[name]][1].set_params(**sub_params)
        # Autres paramètres
        super().set_params(**params)
        return self

    # Propriété indiquant le mode passthrough
    @property
    def _passthrough(self) -> bool:
        """Whether the columns of ``X`` are expert forecasts."""
        return isinstance(self.experts, str) and self.experts == PASSTHROUGH

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    # Méthode de validation de la liste des experts
    def _expert_names(self, X: pd.DataFrame) -> List[Any]:
        """Validate ``experts`` and return the names of the experts.

        Args:
            X: Features (their columns name the experts in passthrough mode).

        Returns:
            The names of the experts, in order.

        Raises:
            ValueError: If ``experts`` is neither ``'passthrough'`` nor a
                non-empty list of ``(name, estimator)`` pairs with unique
                names.
        """
        # Cas passthrough : une colonne de X par expert
        if self._passthrough:
            if X.shape[1] == 0:
                raise ValueError("In passthrough mode X must have one column per expert")
            return list(X.columns)
        # Cas méta-estimateur : liste non vide de couples (nom, estimateur)
        if isinstance(self.experts, str) or not isinstance(self.experts, (list, tuple)) or not self.experts:
            raise ValueError(
                f"experts must be {PASSTHROUGH!r} or a non-empty list of (name, estimator) pairs, "
                f"got {self.experts!r}"
            )
        # Vérification de la forme de chaque couple
        for item in self.experts:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise ValueError(f"Each expert must be a (name, estimator) pair, got {item!r}")
            if not (hasattr(item[1], "fit") and hasattr(item[1], "predict")):
                raise ValueError(f"Expert {item[0]!r} must implement fit and predict")
        names = [name for name, _ in self.experts]
        # Noms uniques et sans « __ » (séparateur des paramètres imbriqués)
        if len(set(names)) != len(names):
            raise ValueError(f"Expert names must be unique, got {names}")
        if any("__" in name for name in names):
            raise ValueError(f"Expert names must not contain '__', got {names}")
        return names

    # Méthode de validation des paramètres d'agrégation
    def _aggregation_config(self, n_experts: int) -> Tuple[str, Loss, bool, np.ndarray]:
        """Validate the aggregation parameters.

        Args:
            n_experts: Number of experts.

        Returns:
            ``(algorithm, loss, gradient, prior)``: key of the algorithm, loss,
            whether to linearise it, and normalised initial weights.

        Raises:
            ValueError: If a parameter is invalid.
        """
        # Algorithme
        if self.aggregation not in ALGORITHMS:
            raise ValueError(f"aggregation must be one of {tuple(ALGORITHMS)}, got {self.aggregation!r}")
        # Perte et niveau de quantile
        loss = make_loss(self.loss, self.quantile)
        # Choix de la perte linéarisée
        if not isinstance(self.loss_gradient, (bool, np.bool_)):
            raise ValueError(f"loss_gradient must be a boolean, got {self.loss_gradient!r}")
        # Mutualisation
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        # Poids initiaux : uniformes, ou strictement positifs et normalisés
        if isinstance(self.init_weights, str):
            if self.init_weights != "uniform":
                raise ValueError(f"init_weights must be 'uniform' or an array, got {self.init_weights!r}")
            prior = np.full(n_experts, 1.0 / n_experts)
        else:
            prior = np.array(self.init_weights, dtype=float)
            if prior.shape != (n_experts,) or not np.all(np.isfinite(prior)) or not np.all(prior > 0):
                raise ValueError(
                    f"init_weights must hold {n_experts} strictly positive finite weights, got {self.init_weights!r}"
                )
            prior = prior / prior.sum()
        return self.aggregation, loss, bool(self.loss_gradient), prior

    # Méthode de validation de la cible
    @staticmethod
    def _target_values(y: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """Return the target as a finite float array of shape ``(n,)``.

        Raises:
            ValueError: If ``y`` is missing, has several columns or holds a
                non-finite value.
        """
        # Cible absente
        if y is None:
            raise ValueError("OnlineAggregator requires y")
        # Cible à une seule colonne
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"y must have a single target, got {y.shape[1]} columns")
            y = y.iloc[:, 0]
        values = y.to_numpy(dtype=float)
        # Valeurs finies (lignes sans cible à retirer en amont, par DropNaNTarget)
        if not np.all(np.isfinite(values)):
            raise ValueError("y contains non-finite values; drop the rows without target first")
        return values

    # Méthode de contrôle des niveaux d'entité au predict et au partial_fit
    def _check_entity_levels(self, layout: TemporalLayout) -> None:
        """Raise ``ValueError`` when the entity levels differ from those seen at fit."""
        if len(layout.entity_names) != len(self.entity_names_):
            raise ValueError(
                f"Expected {len(self.entity_names_)} entity level(s) as at fit, got {len(layout.entity_names)}"
            )

    # ------------------------------------------------------------------
    # Prévisions des experts
    # ------------------------------------------------------------------

    # Méthode de construction des plis du découpage interne
    def _folds(self, X: pd.DataFrame, y: pd.Series, layout: TemporalLayout) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return the ``(train, test)`` row positions of the internal folds.

        With an integer ``cv``, the last ``cv`` blocks of
        ``n_dates // (cv + 1)`` consecutive dates are tested in turn, each by
        experts fitted on every earlier date (expanding window, no gap); the
        blocks are common to the entities and never overlap, whatever the
        holes of the data. A splitter receives the rows sorted by entity then
        date, as ``tsforecast`` splitters expect them.

        Args:
            X: Training features.
            y: Training target.
            layout: Layout of ``X.index``.

        Returns:
            One ``(train, test)`` pair of positions in ``X`` per fold.

        Raises:
            ValueError: If ``cv`` is neither a splitter nor a positive integer,
                or the data has too few dates for the requested folds.
        """
        # Cas d'un splitter fourni (une chaîne a aussi une méthode split)
        if not isinstance(self.cv, str) and hasattr(self.cv, "split"):
            # Permutation triant les lignes par entité puis date
            permutation = np.lexsort((layout.dates.to_numpy(), layout.codes))
            return [
                (permutation[train], permutation[test])
                for train, test in self.cv.split(X.iloc[permutation], y.iloc[permutation])
            ]
        # Cas d'un nombre de plis
        if not isinstance(self.cv, Integral) or isinstance(self.cv, bool) or self.cv < 1:
            raise ValueError(f"cv must be a splitter or a positive integer, got {self.cv!r}")
        n_folds = int(self.cv)
        # Calendrier trié des dates distinctes et taille des blocs testés
        dates = layout.dates.to_numpy()
        calendar = np.unique(dates)
        block = len(calendar) // (n_folds + 1)
        if block < 1:
            raise ValueError(f"The data has {len(calendar)} dates, too few for cv={self.cv} folds")
        # Rang de chaque ligne dans le calendrier
        ranks = np.searchsorted(calendar, dates)
        # Un pli par bloc : bloc testé, dates antérieures à l'entraînement
        first = len(calendar) - n_folds * block
        return [
            (
                np.flatnonzero(ranks < first + j * block),
                np.flatnonzero((ranks >= first + j * block) & (ranks < first + (j + 1) * block)),
            )
            for j in range(n_folds)
        ]

    # Méthode de calcul des prévisions hors échantillon des experts
    def _out_of_sample(
        self, X: pd.DataFrame, y: pd.Series, layout: TemporalLayout, names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast the training rows out of sample, expert by expert and fold by fold.

        Args:
            X: Training features.
            y: Training target.
            layout: Layout of ``X.index``.
            names: Names of the experts.

        Returns:
            ``(positions, forecasts)``: positions of the rows forecast by a
            fold, increasing, and the forecasts of every expert on them, shape
            ``(len(positions), n_experts)``.

        Raises:
            ValueError: If a row is tested by several folds, or no row is
                tested.
        """
        # Plis du découpage interne, en positions des lignes de X
        folds = self._folds(X, y, layout)
        # Nombre de plis testant chaque ligne
        coverage = np.zeros(len(X), dtype=np.intp)
        for _, test in folds:
            coverage[test] += 1
        if np.any(coverage > 1):
            raise ValueError(
                "The internal splitter tests some rows in several folds, so their forecast is ambiguous "
                "(a tsforecast panel splitter counts test_size in rows per entity: with holes, use an integer cv)"
            )
        if not np.any(coverage):
            raise ValueError("The internal splitter tests no row: no out-of-sample forecast to aggregate")
        # Liste des tâches (colonne de l'expert, nom, expert, entraînement, test), expert puis pli
        tasks = [
            (k, name, expert, train, test)
            for k, (name, expert) in enumerate(self.experts)
            for train, test in folds
        ]
        # Ajustement d'un clone par tâche, en parallèle
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict)(clone(expert), X.iloc[train], y.iloc[train], X.iloc[test], name)
            for _, name, expert, train, test in tasks
        )
        # Matrice des prévisions sur toutes les lignes, remplie tâche par tâche
        forecasts = np.full((len(X), len(names)), np.nan)
        for (k, _, _, _, test), values in zip(tasks, results):
            forecasts[test, k] = values
        positions = np.flatnonzero(coverage)
        return positions, forecasts[positions]

    # Méthode de calcul des prévisions des experts réajustés
    def _expert_forecasts(self, X: pd.DataFrame) -> np.ndarray:
        """Forecasts of the experts on the rows of ``X``, shape ``(len(X), n_experts)``.

        Raises:
            ValueError: If a forecast is not finite.
        """
        # Cas passthrough : colonnes de X
        if self._passthrough:
            forecasts = X.to_numpy(dtype=float)
        else:
            # Cas méta-estimateur : une colonne par expert réajusté
            forecasts = np.column_stack([
                _as_column(expert.predict(X), len(X), name)
                for name, expert in zip(self.expert_names_, self.experts_)
            ])
        if not np.all(np.isfinite(forecasts)):
            raise ValueError("The expert forecasts contain non-finite values")
        return forecasts

    # ------------------------------------------------------------------
    # Agrégation séquentielle
    # ------------------------------------------------------------------

    # Méthode de poursuite des agrégations sur de nouvelles lignes
    def _advance(
        self,
        forecasts: np.ndarray,
        y: np.ndarray,
        layout: TemporalLayout,
        positions: np.ndarray,
        config: Tuple[str, Loss, bool, np.ndarray],
        global_state: Any,
        entity_states: Optional[Dict[tuple, Any]],
    ) -> Tuple[np.ndarray, Any, np.ndarray]:
        """Play the given rows, in date order, from the current states.

        Args:
            forecasts: Expert forecasts of the rows, shape ``(n, K)``.
            y: Targets of the rows, shape ``(n,)``.
            layout: Layout of the index the rows come from.
            positions: Position of each row in ``layout``.
            config: Validated ``(algorithm, loss, gradient, prior)``.
            global_state: State of the global aggregation.
            entity_states: States by entity key (updated in place), or None
                for a single aggregation.

        Returns:
            ``(order, global_state, history)``: order of the rows in the
            sequence (indices into ``positions``), new global state, and
            weights used for each row in that order, shape ``(n, K)``.
        """
        algorithm, loss, gradient, prior = config
        # Ordre chronologique des lignes (tri stable sur les dates)
        dates = layout.dates.to_numpy()[positions]
        order = np.argsort(dates, kind="stable")
        sorted_forecasts, sorted_y = forecasts[order], y[order]
        # Un tour par date : code croissant à chaque changement de date
        sorted_dates = dates[order]
        rounds = np.concatenate(([0], np.cumsum(sorted_dates[1:] != sorted_dates[:-1])))
        # Agrégation globale, toujours tenue (repli des entités inconnues)
        global_state, history, _ = run(algorithm, global_state, sorted_forecasts, sorted_y, rounds, loss, gradient)
        # Cas d'une agrégation par entité
        if entity_states is not None:
            history = np.empty_like(history)
            # Code d'entité de chaque ligne, dans l'ordre chronologique
            codes = layout.codes[positions][order]
            # Parcours des entités présentes : une séquence par entité, une ligne par date
            for code in np.unique(codes):
                rows = codes == code
                key = layout.entities[code]
                state = entity_states.get(key)
                # Entité nouvelle : état initial
                if state is None:
                    state = ALGORITHMS[algorithm].init(prior)
                state, weights, _ = run(
                    algorithm, state, sorted_forecasts[rows], sorted_y[rows], np.arange(rows.sum()), loss, gradient
                )
                entity_states[key] = state
                history[rows] = weights
        return order, global_state, history

    # Méthode de publication des attributs de poids
    def _publish_weights(self) -> None:
        """Set ``weights_`` and, per entity, ``entity_weights_`` from the states."""
        algorithm = ALGORITHMS[_STATE_ALGORITHMS[type(self.state_)]]
        # Colonnes nommées par les experts (dtype déduit des noms)
        columns = pd.Index(self.expert_names_)
        self.weights_ = pd.Series(algorithm.weights(self.state_), index=columns, name="weight", dtype="float64")
        # Cas d'une agrégation par entité : une ligne de poids par entité
        if hasattr(self, "entity_states_"):
            keys = list(self.entity_states_)
            values = np.array([algorithm.weights(self.entity_states_[key]) for key in keys], dtype=float)
            self.entity_weights_ = pd.DataFrame(
                values.reshape(len(keys), len(columns)),
                index=_entity_index(keys, self.entity_names_),
                columns=columns,
            )

    # ------------------------------------------------------------------
    # API sklearn
    # ------------------------------------------------------------------

    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame, Any]) -> "OnlineAggregator":
        """Learn the aggregation weights on the training rows, in date order.

        Args:
            X: Features (meta mode) or expert forecasts (passthrough mode),
                indexed by a ``DatetimeIndex`` or an ``(entity..., date)``
                ``MultiIndex``; rows in any order, unbalanced panels accepted.
            y: Target, indexed like ``X`` (or one value per row).

        Returns:
            The fitted aggregator.

        Raises:
            TypeError: If ``X`` is not a ``DataFrame`` or its dates are not
                ``datetime64``.
            ValueError: If a parameter is invalid, ``y`` is not a finite
                single target, an expert forecast is not finite, or the
                internal splitter tests no row or a row twice.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from sklearn.dummy import DummyRegressor
            >>> dates = pd.date_range('2024-01-01', periods=12, freq='MS')
            >>> X = pd.DataFrame({'x': np.arange(12.0)}, index=dates)
            >>> y = 2.0 * X['x']
            >>> experts = [('linear', LinearRegression()), ('mean', DummyRegressor())]
            >>> model = OnlineAggregator(experts, cv=3).fit(X, y)
            >>> len(model.weights_history_), bool(model.weights_['linear'] > 0.9)
            (9, True)
        """
        # Validation des entrées : attributs de colonnes, cible alignée, analyse de l'index
        X, y, layout = check_adapter_input(self, X, y, reset=True, detect_freq=False)
        y_values = self._target_values(y)
        names = self._expert_names(X)
        config = self._aggregation_config(len(names))
        y_series = pd.Series(y_values, index=X.index)

        # Prévisions agrégées : colonnes de X, ou prévisions hors échantillon des experts
        if self._passthrough:
            positions = np.arange(len(X))
            forecasts = X.to_numpy(dtype=float)
            if not np.all(np.isfinite(forecasts)):
                raise ValueError("The expert forecasts contain non-finite values")
        else:
            positions, forecasts = self._out_of_sample(X, y_series, layout, names)
            if not np.all(np.isfinite(forecasts)):
                raise ValueError("The out-of-sample expert forecasts contain non-finite values")

        # États initiaux : agrégation globale, et une par entité sur un panel en per_entity
        algorithm, _, _, prior = config
        initial = ALGORITHMS[algorithm].init(prior)
        entity_states = {} if (self.pooling == "per_entity" and layout.is_panel) else None
        order, state, history = self._advance(
            forecasts, y_values[positions], layout, positions, config, initial, entity_states
        )

        # Suppression des attributs propres à un mode ou à un panel, laissés par un ajustement précédent
        for attribute in ("experts_", "oos_predictions_", "entity_states_", "entity_weights_"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        # Réajustement des experts sur tout l'entraînement (mode méta)
        if not self._passthrough:
            self.experts_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit)(clone(expert), X, y_series) for _, expert in self.experts
            )
            self.oos_predictions_ = pd.DataFrame(forecasts, index=X.index[positions], columns=pd.Index(names))

        # Attributs ajustés
        # Noms des experts (object : libellés hétérogènes possibles, comme feature_names_in_)
        self.expert_names_ = np.array(names, dtype=object)
        self.prior_ = prior
        self.entity_names_ = layout.entity_names
        self.state_ = state
        # États par entité (per_entity sur un panel)
        if entity_states is not None:
            self.entity_states_ = entity_states
        self.weights_history_ = pd.DataFrame(
            history, index=X.index[positions[order]], columns=pd.Index(names)
        )
        self.last_date_ = layout.dates[positions].max()
        self._publish_weights()
        return self

    # Méthode d'entraînement incrémental
    def partial_fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame, Any]) -> "OnlineAggregator":
        """Continue the sequence with new rows, later than every aggregated row.

        Before any ``fit``, this is ``fit``. Afterwards, the rows are forecast
        by the experts fitted at ``fit`` (meta mode, the experts are not
        refitted) or read from ``X`` (passthrough mode), and played in date
        order from the current weights: a ``fit`` followed by a
        ``partial_fit`` in passthrough mode gives the weights of a single
        ``fit`` on all the rows. A new entity starts from the initial weights.

        Args:
            X: New rows, all dated after ``last_date_``.
            y: Their target.

        Returns:
            The updated aggregator.

        Raises:
            ValueError: If a row is not later than ``last_date_``, the entity
                levels or columns differ from those seen at fit, the
                aggregation algorithm changed since ``fit``, or the target or a
                forecast is not finite.

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2024-01-01', periods=6, freq='MS')
            >>> y = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], index=dates)
            >>> X = pd.DataFrame({'a': y + 0.5, 'b': y - 1.0}, index=dates)
            >>> online = OnlineAggregator('passthrough').fit(X.iloc[:3], y.iloc[:3]).partial_fit(X.iloc[3:], y.iloc[3:])
            >>> batch = OnlineAggregator('passthrough').fit(X, y)
            >>> bool((online.weights_ == batch.weights_).all())
            True
        """
        # Cas d'un estimateur jamais ajusté : ajustement complet
        if not hasattr(self, "state_"):
            return self.fit(X, y)

        # Validation des entrées et des paramètres
        X, y, layout = check_adapter_input(self, X, y, reset=False, detect_freq=False)
        self._check_entity_levels(layout)
        y_values = self._target_values(y)
        config = self._aggregation_config(len(self.expert_names_))
        if _STATE_ALGORITHMS[type(self.state_)] != config[0]:
            raise ValueError(
                f"aggregation changed from {_STATE_ALGORITHMS[type(self.state_)]!r} to {config[0]!r} "
                "since fit; call fit to restart the sequence"
            )
        # Lignes postérieures à toutes les lignes déjà agrégées
        if len(X) and layout.dates.min() <= self.last_date_:
            raise ValueError(
                f"partial_fit needs rows dated after the last aggregated date {self.last_date_}, "
                f"got {layout.dates.min()}"
            )
        if not len(X):
            return self

        # Prévisions des experts et poursuite des séquences
        forecasts = self._expert_forecasts(X)
        positions = np.arange(len(X))
        entity_states = getattr(self, "entity_states_", None)
        # Une série ajustée en per_entity n'a qu'une agrégation ; un panel ajouté plus tard non plus
        order, self.state_, history = self._advance(
            forecasts, y_values, layout, positions, config, self.state_, entity_states
        )

        # Mise à jour des attributs ajustés
        columns = pd.Index(self.expert_names_)
        if not self._passthrough:
            new = pd.DataFrame(forecasts, index=X.index, columns=columns)
            self.oos_predictions_ = pd.concat([self.oos_predictions_, new])
        self.weights_history_ = pd.concat(
            [self.weights_history_, pd.DataFrame(history, index=X.index[order], columns=columns)]
        )
        self.last_date_ = layout.dates.max()
        self._publish_weights()
        return self

    # Méthode de prévision
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Combine the expert forecasts of the rows of ``X`` with the final weights.

        No row updates the weights (their target is unknown). With
        ``pooling='per_entity'``, a row takes the weights of its entity, or
        the global weights for an entity unknown at fit.

        Args:
            X: Rows to forecast, indexed like the training rows.

        Returns:
            One forecast per row, in the row order of ``X``.

        Raises:
            NotFittedError: If the aggregator is not fitted.
            ValueError: If the columns or entity levels differ from those seen
                at fit, or an expert forecast is not finite.

        Examples:
            >>> import pandas as pd
            >>> index = pd.MultiIndex.from_product(
            ...     [['A', 'B'], pd.date_range('2024-01-01', periods=4, freq='MS')], names=['id', 'date'])
            >>> y = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], index=index)
            >>> close, far = [0.1] * 4, [3.0] * 4
            >>> X = pd.DataFrame({'p': y + (close + far), 'q': y + (far + close)}, index=index)
            >>> model = OnlineAggregator('passthrough', aggregation='MLpol', pooling='per_entity').fit(X, y)
            >>> model.entity_weights_.round(2).to_dict('index')
            {'A': {'p': 1.0, 'q': 0.0}, 'B': {'p': 0.0, 'q': 1.0}}
        """
        check_is_fitted(self, "state_")
        # Validation des entrées : colonnes vues au fit, analyse de l'index
        X, _, layout = check_adapter_input(self, X, reset=False, detect_freq=False)
        self._check_entity_levels(layout)
        forecasts = self._expert_forecasts(X)
        # Poids de chaque ligne : globaux par défaut
        weights = np.broadcast_to(self.weights_.to_numpy(), forecasts.shape).copy()
        # Cas d'une agrégation par entité : poids de l'entité connue de chaque ligne
        entity_states = getattr(self, "entity_states_", None)
        if entity_states is not None:
            algorithm = ALGORITHMS[_STATE_ALGORITHMS[type(self.state_)]]
            # Poids calculés une fois par entité distincte, puis diffusés par les codes des lignes
            for code, key in enumerate(layout.entities):
                if key in entity_states:
                    weights[layout.codes == code] = algorithm.weights(entity_states[key])
        return np.sum(forecasts * weights, axis=1)
