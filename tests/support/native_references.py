"""Native references of the adapter contract: the same models used through their package alone.

Each factory returns a function ``(X_train, y_train, X_test, fh) -> ndarray``
(``AdapterCase.native_reference``) that fits and predicts a model with the API of
sktime, darts, scikit-learn or hierarchicalforecast, without ``tsforecast``:
relative horizon ``fh`` for sktime, ``predict(n)`` for darts, a pooled
regression for the row-wise darts models, ``reconcile`` per copy for
hierarchicalforecast. ``temporal_reference`` and ``cross_temporal_reference``
reconcile complete calendar years with ``aggregate_temporal`` and
``reconcile(temporal=True)``, the cross-temporal one in two chained passes.
``drift_reference`` computes the fake foundation backend by hand;
``chronos_reference``, ``timestral_reference`` and ``t0_reference`` call
``predict_quantiles`` of a Chronos pipeline, ``TimestralForecaster.forecast`` and
``T0Forecaster.predict`` directly, one ``k``-step forecast per test row
(``foundation_quantiles``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone

from tests.support.adapter_contract import NativeReference
from tests.support.datasets import HIERARCHY_AGGREGATE_LABEL, HIERARCHY_LEVELS, TEMPORAL_HIERARCHY

# Combinaison des prévisions de base : une colonne, 'mean', ou {nom de tag : colonne}
Combination = Union[str, Mapping[str, str]]

# Arbre des jeux hiérarchiques (a = b + c, c = d + e) écrit à la main : agrégats puis feuilles,
# colonnes = feuilles ; aggregate de hierarchicalforecast ne sait pas construire un arbre dont
# les feuilles sont à des profondeurs différentes sans dupliquer la feuille b
_TREE_S_DF = pd.DataFrame(
    {
        'unique_id': ['a', 'c', 'b', 'd', 'e'],
        'b': [1.0, 0.0, 1.0, 0.0, 0.0],
        'd': [1.0, 1.0, 0.0, 1.0, 0.0],
        'e': [1.0, 1.0, 0.0, 0.0, 1.0],
    }
)
# Nœuds de l'arbre par profondeur
_TREE_TAGS = {'depth_0': np.array(['a']), 'depth_1': np.array(['b', 'c']), 'depth_2': np.array(['d', 'e'])}


# ---------------------------------------------------------------------------
# Outils communs
# ---------------------------------------------------------------------------


# Fonction auxiliaire de lecture de la fréquence d'une série de dates
def _frequency(dates: pd.Index) -> str:
    """Return the pandas frequency of sorted, regular dates (``pd.infer_freq``)."""
    return pd.infer_freq(pd.DatetimeIndex(dates.unique()).sort_values())


# Fonction auxiliaire de découpage des lignes par entité
def _entity_rows(frame: Union[pd.DataFrame, pd.Series]) -> Dict[Any, np.ndarray]:
    """Return the positions of the rows of each entity, sorted by date.

    Args:
        frame: Series or panel indexed by dates or by ``(entity..., date)``.

    Returns:
        Dictionary entity key (None for a series) -> row positions in date
        order, entities in order of first appearance.
    """
    dates = pd.DatetimeIndex(frame.index.get_level_values(-1))
    # Série temporelle : une seule entité
    if not isinstance(frame.index, pd.MultiIndex):
        return {None: np.argsort(dates.to_numpy(), kind='stable')}
    # Code de l'entité de chaque ligne et étiquettes des entités
    codes, keys = frame.index.droplevel(-1).factorize()
    # Dictionnaire entité -> positions de ses lignes, triées par date
    positions = {}
    for code, key in enumerate(keys):
        rows = np.flatnonzero(codes == code)
        positions[key] = rows[np.argsort(dates.to_numpy()[rows], kind='stable')]
    return positions


# Fonction auxiliaire d'extraction d'une entité indexée par ses seules dates
def _entity_frame(frame: Union[pd.DataFrame, pd.Series], rows: np.ndarray) -> Union[pd.DataFrame, pd.Series]:
    """Return the rows of one entity, indexed by their dates."""
    part = frame.iloc[rows]
    return part.set_axis(pd.DatetimeIndex(part.index.get_level_values(-1), name=None), axis=0)


# ---------------------------------------------------------------------------
# sktime
# ---------------------------------------------------------------------------


# Fabrique de la référence d'un forecaster sktime
def sktime_reference(forecaster: Any, uses_X: bool = False) -> NativeReference:
    """Reference of a sktime forecaster: ``fit`` then ``predict`` on a relative horizon.

    Args:
        forecaster: Unfitted sktime forecaster, cloned at each call.
        uses_X: Whether ``X`` is passed to ``fit`` and ``predict``.

    Returns:
        Native reference; a panel is passed in sktime's ``pd-multiindex``
        format, which sktime vectorizes over the entities.
    """

    # Référence native du forecaster
    def reference(X_train, y_train, X_test, fh):
        from sktime.forecasting.base import ForecastingHorizon

        freq = _frequency(y_train.index.get_level_values(-1))
        # Cible au format sktime : DataFrame trié par entité puis date
        y_fit = y_train.to_frame().sort_index()
        X_fit = X_train.sort_index() if uses_X else None
        X_pred = X_test.sort_index() if uses_X else None
        # Série temporelle : fréquence posée sur l'index, exigée pour un horizon relatif
        if not isinstance(y_fit.index, pd.MultiIndex):
            y_fit = y_fit.asfreq(freq)
            X_fit = None if X_fit is None else X_fit.asfreq(freq)
            X_pred = None if X_pred is None else X_pred.asfreq(freq)
        fitted = forecaster.clone().fit(y_fit, X=X_fit)
        forecasts = fitted.predict(fh=ForecastingHorizon(list(fh), is_relative=True, freq=freq), X=X_pred)
        # Prévisions réalignées sur les lignes de X_test
        return forecasts.reindex(X_test.index).iloc[:, 0].to_numpy()

    return reference


# ---------------------------------------------------------------------------
# darts
# ---------------------------------------------------------------------------


# Fonction auxiliaire de construction d'une série darts
def _darts_series(frame: Union[pd.DataFrame, pd.Series], freq: str) -> Any:
    """Build a darts ``TimeSeries`` from a frame indexed by consecutive dates."""
    from darts import TimeSeries

    frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    return TimeSeries.from_dataframe(frame.astype(float), freq=freq)


# Fonction auxiliaire de placement des pas prévus sur les lignes de test d'une entité
def _place(result: np.ndarray, rows: np.ndarray, forecast: Any, fh: Tuple[int, ...]) -> None:
    """Write the steps ``fh`` of a darts forecast on the test rows of one entity, in date order."""
    result[rows] = forecast.values()[np.asarray(fh) - 1, 0]


# Fabrique de la référence d'un modèle local darts
def darts_local_reference(model: Any) -> NativeReference:
    """Reference of a darts local model: one copy per entity, ``predict(n)``, steps ``fh`` extracted.

    Args:
        model: Unfitted darts local forecasting model.

    Returns:
        Native reference.
    """

    # Référence native du modèle local
    def reference(X_train, y_train, X_test, fh):
        freq = _frequency(y_train.index.get_level_values(-1))
        train_rows, test_rows = _entity_rows(y_train), _entity_rows(X_test)
        # Prévisions alignées sur les lignes de X_test
        result = np.full(len(X_test), np.nan)
        # Une copie non ajustée du modèle par entité
        for entity, rows in train_rows.items():
            fitted = model.untrained_model()
            fitted.fit(_darts_series(_entity_frame(y_train, rows), freq))
            _place(result, test_rows[entity], fitted.predict(n=max(fh)), fh)
        return result

    return reference


# Fabrique de la référence d'un modèle global darts
def darts_global_reference(model: Any, past: bool = False, future: bool = False) -> NativeReference:
    """Reference of a darts global model: one series per entity, ``predict(n)``, steps ``fh`` extracted.

    Args:
        model: Unfitted darts global forecasting model.
        past: Whether ``X`` is passed as past covariates.
        future: Whether ``X`` is passed as future covariates.

    Returns:
        Native reference; the covariates of an entity are its training then
        test rows of ``X`` (contiguous only without gap).
    """

    # Référence native du modèle global
    def reference(X_train, y_train, X_test, fh):
        freq = _frequency(y_train.index.get_level_values(-1))
        train_rows, test_rows = _entity_rows(y_train), _entity_rows(X_test)
        entities = list(train_rows)
        # Série de la cible de chaque entité, dans l'ordre de entities
        series = [_darts_series(_entity_frame(y_train, train_rows[entity]), freq) for entity in entities]
        # Covariables de chaque entité : lignes d'entraînement puis de test de X
        kwargs = {}
        if past or future:
            covariates = [
                _darts_series(
                    pd.concat(
                        [_entity_frame(X_train, train_rows[entity]), _entity_frame(X_test, test_rows[entity])]
                    ),
                    freq,
                )
                for entity in entities
            ]
            if past:
                kwargs['past_covariates'] = covariates
            if future:
                kwargs['future_covariates'] = covariates
        fitted = model.untrained_model()
        fitted.fit(series, **kwargs)
        forecasts = fitted.predict(n=max(fh), series=series, **kwargs)
        # Prévisions alignées sur les lignes de X_test
        result = np.full(len(X_test), np.nan)
        for entity, forecast in zip(entities, forecasts):
            _place(result, test_rows[entity], forecast, fh)
        return result

    return reference


# Fabrique de la référence d'un régresseur scikit-learn mutualisé
def pooled_sklearn_reference(regressor: Any) -> NativeReference:
    """Reference of a row-wise model: a scikit-learn regressor pooled over every training row.

    The training rows are sorted by entity then date, the order in which darts
    builds the samples of its list of series (a random forest depends on it).

    Args:
        regressor: Unfitted scikit-learn regressor, cloned at each call.

    Returns:
        Native reference.
    """

    # Référence native du régresseur ligne à ligne
    def reference(X_train, y_train, X_test, fh):
        X_sorted = X_train.sort_index()
        fitted = clone(regressor).fit(X_sorted.to_numpy(), y_train.reindex(X_sorted.index).to_numpy())
        return fitted.predict(X_test.to_numpy())

    return reference


# ---------------------------------------------------------------------------
# hierarchicalforecast
# ---------------------------------------------------------------------------


# Description d'une hiérarchie cross-section pour hierarchicalforecast
@dataclass(frozen=True)
class _NativeHierarchy:
    """Cross-sectional hierarchy as hierarchicalforecast reads it."""

    # Nombre de niveaux d'index portant le nœud (juste avant la date, ou avant le niveau temporel)
    n_levels: int
    # Identifiant hierarchicalforecast du nœud de chaque ligne, lu sur ces niveaux
    node_ids: Callable[[pd.DataFrame], np.ndarray]
    # Matrice de sommation et tags, construits depuis les niveaux de nœud des lignes
    structure: Callable[[pd.DataFrame], Tuple[pd.DataFrame, Dict[str, np.ndarray]]]


# Fonction auxiliaire : identifiant de nœud d'un arbre
def _tree_ids(nodes: pd.DataFrame) -> np.ndarray:
    """Node labels of a tree (one node level)."""
    return nodes.iloc[:, 0].to_numpy(dtype=object)


# Fonction auxiliaire : identifiants de nœud de la hiérarchie pays / régions, nommés comme aggregate
def _levels_ids(nodes: pd.DataFrame) -> np.ndarray:
    """``'<country>'`` for a country row, ``'<country>/<region>'`` for a region row."""
    country, region = nodes['country'].to_numpy(dtype=object), nodes['region'].to_numpy(dtype=object)
    return np.where(region == HIERARCHY_AGGREGATE_LABEL, country, country + '/' + region)


# Fonction auxiliaire : structure de la hiérarchie pays / régions par aggregate
def _levels_structure(nodes: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Summing matrix and tags of the country / region hierarchy, from its region rows."""
    from hierarchicalforecast.utils import aggregate

    leaves = nodes[nodes['region'] != HIERARCHY_AGGREGATE_LABEL].drop_duplicates()
    # Une date factice : aggregate agrège des séries, seule la structure est lue
    bottom = leaves.assign(ds=pd.Timestamp('2020-01-01'), y=0.0)
    _, S_df, tags = aggregate(bottom, HIERARCHY_LEVELS)
    return S_df, tags


# Hiérarchies cross-section des jeux : arbre de nœuds, niveaux d'index pays / régions
TREE = _NativeHierarchy(1, _tree_ids, lambda nodes: (_TREE_S_DF, _TREE_TAGS))
LEVELS = _NativeHierarchy(2, _levels_ids, _levels_structure)


# Fonction auxiliaire de calcul de la prévision de base à réconcilier
def _base_forecast(X: pd.DataFrame, node_ids: np.ndarray, tags: Dict[str, np.ndarray], combination: Combination):
    """Column, row-wise mean, or one column per tag of the node of each row."""
    # Moyenne ligne à ligne de toutes les colonnes
    if combination == 'mean':
        return X.mean(axis=1).to_numpy()
    # Une colonne pour toutes les lignes
    if isinstance(combination, str):
        return X[combination].to_numpy()
    # Une colonne par niveau : colonne du tag de chaque nœud
    column_of = {node: combination[tag] for tag, nodes in tags.items() for node in nodes}
    columns = np.array([column_of[node] for node in node_ids], dtype=object)
    return np.array([X[column].iloc[row] for row, column in enumerate(columns)], dtype=float)


# Fonction auxiliaire de réconciliation native d'un jeu de prévisions de base
def _reconcile(reconciler: Any, Y_hat_df: pd.DataFrame, tags, S_df, Y_df=None, temporal: bool = False) -> np.ndarray:
    """Call ``HierarchicalReconciliation.reconcile`` and return the reconciled ``model`` column.

    The output is realigned on the rows of ``Y_hat_df`` by ``(unique_id, ds)``
    (``(unique_id, temporal_id)`` in temporal mode).
    """
    import copy

    from hierarchicalforecast.core import HierarchicalReconciliation

    reconciled = HierarchicalReconciliation([copy.deepcopy(reconciler)]).reconcile(
        Y_hat_df, tags, S_df=S_df, Y_df=Y_df, temporal=temporal
    )
    column = next(name for name in reconciled.columns if name.startswith('model/'))
    keys = ['unique_id', 'temporal_id' if temporal else 'ds']
    target = pd.MultiIndex.from_frame(Y_hat_df[keys])
    return reconciled.set_index(keys)[column].reindex(target).to_numpy()


# Fonction auxiliaire de découpage des lignes par copie de la hiérarchie
def _copy_rows(index: pd.MultiIndex, n_copy_levels: int) -> Dict[Any, np.ndarray]:
    """Return the positions of the rows of each copy (its first ``n_copy_levels`` index levels)."""
    if n_copy_levels == 0:
        return {(): np.arange(len(index))}
    codes, keys = index.droplevel(list(range(n_copy_levels, index.nlevels))).factorize()
    return {keys[code]: np.flatnonzero(codes == code) for code in range(len(keys))}


# Fonction auxiliaire : niveaux de nœud d'un jeu, en colonnes
def _nodes(X: pd.DataFrame, first: int, hierarchy: _NativeHierarchy) -> pd.DataFrame:
    """Index levels carrying the node, from position ``first``, as columns."""
    names = list(X.index.names[first : first + hierarchy.n_levels])
    return pd.DataFrame({name: X.index.get_level_values(name) for name in names})


# Fabrique de la référence d'une réconciliation cross-section
def reconcile_reference(reconciler: Any, hierarchy: _NativeHierarchy, combination: Combination) -> NativeReference:
    """Reference of a cross-sectional reconciliation: ``reconcile`` of each copy, date by date.

    The index levels before the node levels are the copies, each one
    reconciled by its own call (with its own in-sample rows for an in-sample
    reconciler, built from the training rows).

    Args:
        reconciler: hierarchicalforecast reconciler, deep-copied at each call.
        hierarchy: ``TREE`` or ``LEVELS``.
        combination: Base forecast: a column, ``'mean'`` or ``{tag: column}``.

    Returns:
        Native reference.
    """

    # Référence native de la réconciliation
    def reference(X_train, y_train, X_test, fh):
        n_copy = X_test.index.nlevels - 1 - hierarchy.n_levels
        train_copies = _copy_rows(X_train.index, n_copy)
        # Prévisions réconciliées alignées sur les lignes de X_test
        result = np.full(len(X_test), np.nan)
        # Un appel à reconcile par copie
        for key, rows in _copy_rows(X_test.index, n_copy).items():
            test = X_test.iloc[rows]
            nodes = _nodes(test, n_copy, hierarchy)
            ids = hierarchy.node_ids(nodes)
            S_df, tags = hierarchy.structure(nodes)
            dates = test.index.get_level_values(-1)
            Y_hat_df = pd.DataFrame({'unique_id': ids, 'ds': dates, 'model': _base_forecast(test, ids, tags, combination)})
            # Lignes insample de la copie : réalisations et prévisions de base hors échantillon
            Y_df = None
            if reconciler.insample:
                train = X_train.iloc[train_copies[key]]
                train_ids = hierarchy.node_ids(_nodes(train, n_copy, hierarchy))
                Y_df = pd.DataFrame(
                    {
                        'unique_id': train_ids,
                        'ds': train.index.get_level_values(-1),
                        'y': y_train.reindex(train.index).to_numpy(),
                        'model': _base_forecast(train, train_ids, tags, combination),
                    }
                )
            result[rows] = _reconcile(reconciler, Y_hat_df, tags, S_df, Y_df)
        return result

    return reference


# ---------------------------------------------------------------------------
# hierarchicalforecast : temporel et cross-temporel
# ---------------------------------------------------------------------------


# Fonction auxiliaire : identifiant temporel de hierarchicalforecast de chaque ligne
def _temporal_ids(levels: np.ndarray, dates: pd.DatetimeIndex, temporal_hierarchy: Mapping[str, int]) -> np.ndarray:
    """``'<level>-<k>'``, ``k`` counting the blocks of fine steps from the first fine date.

    The aggregates of the datasets are dated by their first fine step: the
    position of a date among the sorted fine dates gives its block.
    """
    fine_level = next(level for level, size in temporal_hierarchy.items() if size == 1)
    fine_dates = pd.DatetimeIndex(dates[levels == fine_level]).unique().sort_values()
    positions = fine_dates.get_indexer(dates)
    sizes = np.array([temporal_hierarchy[level] for level in levels])
    return np.array([f'{level}-{block}' for level, block in zip(levels, positions // sizes + 1)], dtype=object)


# Fonction auxiliaire de la passe temporelle native
def _temporal_pass(
    reconciler: Any, ids: np.ndarray, levels: np.ndarray, dates: pd.DatetimeIndex, values: np.ndarray,
    temporal_hierarchy: Mapping[str, int],
) -> np.ndarray:
    """Reconcile every series of ``ids`` over the temporal hierarchy with ``reconcile(temporal=True)``."""
    from hierarchicalforecast.utils import aggregate_temporal

    fine_level = next(level for level, size in temporal_hierarchy.items() if size == 1)
    fine = levels == fine_level
    # Structure temporelle des pas fins de la fenêtre (périodes comptées depuis son premier pas)
    bottom = pd.DataFrame({'unique_id': ids[fine], 'ds': dates[fine], 'y': 0.0})
    _, S_df, tags = aggregate_temporal(bottom, dict(temporal_hierarchy))
    Y_hat_df = pd.DataFrame(
        {
            'unique_id': ids,
            'temporal_id': _temporal_ids(levels, dates, temporal_hierarchy),
            'ds': dates,
            'model': values,
        }
    )
    return _reconcile(reconciler, Y_hat_df, tags, S_df, temporal=True)


# Référence d'une réconciliation temporelle
def temporal_reference(
    X_test: pd.DataFrame, reconciler: Any, column: str, temporal_hierarchy: Mapping[str, int] = TEMPORAL_HIERARCHY
) -> np.ndarray:
    """Reconcile complete periods with ``aggregate_temporal`` and ``reconcile(temporal=True)``.

    Args:
        X_test: Base forecasts indexed by ``(copy..., level, date)``, complete
            periods starting on the first fine step of the window.
        reconciler: Non in-sample hierarchicalforecast reconciler.
        column: Base forecast column.
        temporal_hierarchy: ``{level: number of fine steps}``.

    Returns:
        Reconciled forecasts in the row order of ``X_test``, every copy being a
        series of one call.
    """
    n_copy = X_test.index.nlevels - 2
    # Identifiant de série : étiquettes de copie jointes
    if n_copy == 0:
        ids = np.full(len(X_test), 'series', dtype=object)
    else:
        copies = X_test.index.droplevel([-2, -1])
        ids = np.array(['|'.join(map(str, key if isinstance(key, tuple) else (key,))) for key in copies], dtype=object)
    levels = X_test.index.get_level_values(-2).to_numpy(dtype=object)
    dates = pd.DatetimeIndex(X_test.index.get_level_values(-1))
    return _temporal_pass(reconciler, ids, levels, dates, X_test[column].to_numpy(), temporal_hierarchy)


# Référence d'une réconciliation cross-temporelle par deux passes natives
def cross_temporal_reference(
    X_test: pd.DataFrame,
    reconciler: Any,
    column: str,
    hierarchy: _NativeHierarchy,
    temporal_hierarchy: Mapping[str, int] = TEMPORAL_HIERARCHY,
) -> np.ndarray:
    """Reconcile cross-sectionally each temporal level, then temporally each node of each copy.

    hierarchicalforecast has no cross-temporal reconciliation: two chained
    native passes give the cross-temporal result for the reconcilers whose
    projection is the Kronecker product of the cross-sectional and temporal
    projections (``BottomUp``, ``MinTrace`` ``'ols'``).

    Args:
        X_test: Base forecasts indexed by ``(copy..., node..., level, date)``,
            complete periods.
        reconciler: ``BottomUp()`` or ``MinTrace(method='ols')``.
        column: Base forecast column.
        hierarchy: ``TREE`` or ``LEVELS``.
        temporal_hierarchy: ``{level: number of fine steps}``.

    Returns:
        Reconciled forecasts in the row order of ``X_test``.
    """
    n_copy = X_test.index.nlevels - 2 - hierarchy.n_levels
    levels = X_test.index.get_level_values(-2).to_numpy(dtype=object)
    dates = pd.DatetimeIndex(X_test.index.get_level_values(-1))
    # Prévisions réconciliées alignées sur les lignes de X_test
    result = np.full(len(X_test), np.nan)
    # Deux passes par copie
    for rows in _copy_rows(X_test.index, n_copy).values():
        test = X_test.iloc[rows]
        nodes = _nodes(test, n_copy, hierarchy)
        ids = hierarchy.node_ids(nodes)
        S_df, tags = hierarchy.structure(nodes)
        # Passe cross-section, niveau temporel par niveau temporel (dates communes entre niveaux)
        cross = np.full(len(rows), np.nan)
        for level in temporal_hierarchy:
            mask = levels[rows] == level
            Y_hat_df = pd.DataFrame({'unique_id': ids[mask], 'ds': dates[rows][mask], 'model': test[column].to_numpy()[mask]})
            cross[mask] = _reconcile(reconciler, Y_hat_df, tags, S_df)
        # Passe temporelle sur toutes les séries cross-section de la copie
        result[rows] = _temporal_pass(reconciler, ids, levels[rows], dates[rows], cross, temporal_hierarchy)
    return result


# ---------------------------------------------------------------------------
# Modèles de fondation
# ---------------------------------------------------------------------------


# Fonction auxiliaire de parcours des pas de test d'une entité
def _test_steps(X_test: pd.DataFrame, rows: np.ndarray, fh: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Row positions of one entity of ``X_test`` in date order and their steps after training.

    The ``i``-th date of the test window is ``fh[i]`` steps after the last
    training date (every entity of a balanced panel shares the window).

    Args:
        X_test: Test features.
        rows: Positions of the rows of the entity, sorted by date.
        fh: Relative horizon of the test dates.

    Returns:
        ``(rows, steps)``.
    """
    calendar = pd.DatetimeIndex(X_test.index.get_level_values(-1)).unique().sort_values()
    dates = pd.DatetimeIndex(X_test.index.get_level_values(-1))[rows]
    steps = np.asarray(fh)[calendar.get_indexer(dates)]
    return rows, steps


# Fabrique de la référence du backend factice DriftBackend
def drift_reference(drift: float, coef: float = 0.0, quantile: float = 0.5, spread: float = 1.0) -> NativeReference:
    """Reference of ``DriftBackend`` computed by hand from the training and test frames.

    A test row ``k`` steps after training is ``last + drift·k + coef·S +
    spread·(quantile − 0.5)``, where ``last`` is the last training target of its
    entity and ``S`` the sum of the features of the test rows of its entity at
    most ``k`` steps after training (known covariates of a forecast of ``k``
    steps; the gap has no row).

    Args:
        drift: Increment per step.
        coef: Weight of the known covariates.
        quantile: Level returned.
        spread: Distance between the levels.

    Returns:
        Native reference.
    """

    # Référence du modèle factice
    def reference(X_train, y_train, X_test, fh):
        result = np.full(len(X_test), np.nan)
        train_rows = _entity_rows(y_train)
        # Parcours des entités du test
        for key, rows in _entity_rows(X_test).items():
            last = y_train.to_numpy()[train_rows[key][-1]]
            rows, steps = _test_steps(X_test, rows, fh)
            # Somme des covariables des lignes de test jusqu'au pas de chaque ligne
            sums = np.cumsum(X_test.to_numpy()[rows].sum(axis=1))
            result[rows] = last + drift * steps + coef * sums + spread * (quantile - 0.5)
        return result

    return reference


# Chargement mis en cache d'un pipeline Chronos, sans passer par tsforecast
_CHRONOS_PIPELINES: Dict[str, Any] = {}


def chronos_pipeline(model_id: str) -> Any:
    """Load a Chronos pipeline on CPU with chronos-forecasting alone, once per model id."""
    if model_id not in _CHRONOS_PIPELINES:
        from chronos import BaseChronosPipeline

        _CHRONOS_PIPELINES[model_id] = BaseChronosPipeline.from_pretrained(model_id, device_map='cpu')
    return _CHRONOS_PIPELINES[model_id]


# Chargement mis en cache d'un modèle Timestral, sans passer par tsforecast
_TIMESTRAL_FORECASTERS: Dict[str, Any] = {}


def timestral_forecaster(model_id: str) -> Any:
    """Load a ``TimestralForecaster`` on CPU with timestral alone, once per model id."""
    if model_id not in _TIMESTRAL_FORECASTERS:
        from timestral import TimestralForecaster

        _TIMESTRAL_FORECASTERS[model_id] = TimestralForecaster.from_pretrained(model_id, device='cpu')
    return _TIMESTRAL_FORECASTERS[model_id]


# Chargement mis en cache d'un modèle t0, sans passer par tsforecast
_T0_FORECASTERS: Dict[str, Any] = {}


def t0_forecaster(model_id: str) -> Any:
    """Load a ``T0Forecaster`` on CPU with tfc-t0 alone, once per model id."""
    if model_id not in _T0_FORECASTERS:
        from t0 import T0Forecaster

        _T0_FORECASTERS[model_id] = T0Forecaster.from_pretrained(model_id).eval()
    return _T0_FORECASTERS[model_id]


# Fonction auxiliaire des covariables connues des k pas d'une prévision
def _future_values(X_future: Dict[int, np.ndarray], k: int, n_columns: int) -> np.ndarray:
    """Lay the features of the test rows on the steps ``1..k``, ``NaN`` on the other steps."""
    future = np.full((k, n_columns), np.nan)
    for step, values in X_future.items():
        if step <= k:
            future[step - 1] = values
    return future


# Prévision native des quantiles d'une ligne par Chronos
def chronos_row_quantiles(
    pipeline: Any,
    y_context: np.ndarray,
    X_context: Optional[np.ndarray],
    X_future: Optional[Dict[int, np.ndarray]],
    k: int,
    levels: Sequence[float],
    columns: Sequence[str] = (),
) -> np.ndarray:
    """Quantiles of the step ``k`` of a ``k``-step Chronos forecast of one series.

    Args:
        pipeline: ``Chronos2Pipeline`` or ``ChronosBoltPipeline``.
        y_context: Training target, oldest first.
        X_context: Training features ``(n_steps, n_columns)`` (Chronos-2 with
            covariates), or None.
        X_future: Step (1-based) -> features of the test row at that step,
            for the steps at most ``k``; the other steps are unknown (``NaN``).
        k: Number of steps of the forecast.
        levels: Quantile levels.
        columns: Names of the feature columns.

    Returns:
        The quantiles of the last step, one per level.
    """
    import torch
    from chronos import Chronos2Pipeline

    # Chronos-Bolt : un tenseur par série
    if not isinstance(pipeline, Chronos2Pipeline):
        quantiles, _ = pipeline.predict_quantiles(
            [torch.tensor(y_context, dtype=torch.float32)], prediction_length=k, quantile_levels=list(levels)
        )
        return np.asarray(quantiles[0, k - 1, :], dtype=float)
    item: Dict[str, Any] = {'target': y_context}
    # Covariables connues : passé, puis lignes de test des pas 1..k (NaN ailleurs)
    if X_context is not None:
        future = _future_values(X_future, k, X_context.shape[1])
        item['past_covariates'] = {name: X_context[:, j] for j, name in enumerate(columns)}
        item['future_covariates'] = {name: future[:, j] for j, name in enumerate(columns)}
    quantiles, _ = pipeline.predict_quantiles([item], prediction_length=k, quantile_levels=list(levels))
    return quantiles[0][0, k - 1, :].numpy().astype(float)


# Prévision native des quantiles d'une ligne par Timestral
def timestral_row_quantiles(
    forecaster: Any,
    y_context: np.ndarray,
    X_context: Optional[np.ndarray],
    X_future: Optional[Dict[int, np.ndarray]],
    k: int,
    levels: Sequence[float],
    columns: Sequence[str] = (),
) -> np.ndarray:
    """Quantiles of the step ``k`` of a ``k``-step ``TimestralForecaster.forecast`` of one series.

    Args:
        forecaster: ``TimestralForecaster``.
        y_context: Training target, oldest first.
        X_context: Training features ``(n_steps, n_columns)``, or None.
        X_future: Step (1-based) -> features of the test row at that step, for
            known covariates; None when the features are known over the context
            only.
        k: Number of steps of the forecast.
        levels: Quantile levels of the model.
        columns: Unused (the model reads covariates by position).

    Returns:
        The quantiles of the last step, one per level.
    """
    covariates: Dict[str, np.ndarray] = {}
    if X_context is not None:
        # Covariables passées : contexte seul ; connues : contexte puis pas 1..k
        if X_future is None:
            covariates['past_covariates'] = np.array(X_context.T[None])
        else:
            future = _future_values(X_future, k, X_context.shape[1])
            covariates['known_covariates'] = np.concatenate([X_context, future]).T[None]
    forecast = forecaster.forecast(np.array(y_context)[None, None], k, **covariates)
    return np.array([float(forecast.quantile(level)[0, 0, k - 1]) for level in levels])


# Prévision native des quantiles d'une ligne par t0
def t0_row_quantiles(
    forecaster: Any,
    y_context: np.ndarray,
    X_context: Optional[np.ndarray],
    X_future: Optional[Dict[int, np.ndarray]],
    k: int,
    levels: Sequence[float],
    columns: Sequence[str] = (),
) -> np.ndarray:
    """Quantiles of the step ``k`` of a ``k``-step ``T0Forecaster.predict`` of one series.

    Args:
        forecaster: ``T0Forecaster``.
        y_context: Training target, oldest first.
        X_context: Training features ``(n_steps, n_columns)``, or None.
        X_future: Step (1-based) -> features of the test row at that step, for
            ``future_covariates``; None when the features are known over the
            context only, passed as further variates of the context.
        k: Number of steps of the forecast.
        levels: Quantile levels, increasing.
        columns: Unused (the model reads covariates by position).

    Returns:
        The quantiles of the last step, one per level.
    """
    context = np.array(y_context, dtype=np.float32)[None]
    future_covariates = None
    if X_context is not None:
        # Covariables passées : variables du contexte, la cible en premier
        if X_future is None:
            context = np.concatenate([context, X_context.T.astype(np.float32)])[None]
        else:
            future = _future_values(X_future, k, X_context.shape[1])
            future_covariates = np.concatenate([X_context, future]).T[None].astype(np.float32)
    quantiles = forecaster.predict(
        context, k, quantile_levels=list(levels), future_covariates=future_covariates
    ).quantiles.numpy()
    # Contexte à plusieurs variables : prévision de la cible seule
    if context.ndim == 3:
        quantiles = quantiles[:, 0]
    return quantiles[0, k - 1].astype(float)


# Quantiles natifs de toutes les lignes de test d'un modèle de fondation
def foundation_quantiles(
    row_quantiles: Callable[..., np.ndarray],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    fh: Sequence[int],
    levels: Sequence[float],
    covariates: Optional[str] = 'known',
) -> np.ndarray:
    """Quantiles of every test row from one native ``k``-step forecast per row and entity.

    A test row ``k`` steps after training is the last step of a ``k``-step
    forecast of its entity's training target. With ``covariates='known'``, the
    features of the training rows are past values and those of the test rows of
    its entity at most ``k`` steps after training are known future values; with
    ``'past'``, only the former are passed.

    Args:
        row_quantiles: Function ``(y_context, X_context, X_future, k, levels,
            columns)`` calling the package on one series.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        fh: Relative horizon of the sorted test dates.
        levels: Quantile levels.
        covariates: ``'known'``, ``'past'`` or None.

    Returns:
        Array ``(n_test_rows, n_levels)`` in the row order of ``X_test``.
    """
    result = np.full((len(X_test), len(levels)), np.nan)
    train_rows = _entity_rows(y_train)
    columns = [str(column) for column in X_train.columns]
    # Parcours des entités du test
    for key, rows in _entity_rows(X_test).items():
        y_context = y_train.to_numpy()[train_rows[key]]
        X_context = None if covariates is None else X_train.to_numpy()[train_rows[key]]
        rows, steps = _test_steps(X_test, rows, fh)
        X_future = dict(zip(steps.tolist(), X_test.to_numpy()[rows])) if covariates == 'known' else None
        # Une prévision de k pas par ligne de test
        for row, k in zip(rows, steps.tolist()):
            result[row] = row_quantiles(y_context, X_context, X_future, k, levels, columns)
    return result


# Fabrique de la référence d'un modèle de fondation appelé directement
def _foundation_reference(
    load: Callable[[str], Any],
    row_quantiles: Callable[..., np.ndarray],
    model_id: str,
    quantile: float,
    covariates: bool,
) -> NativeReference:
    """Reference of a foundation model: one native forecast per test row and entity."""

    # Référence du modèle chargé au premier appel
    def reference(X_train, y_train, X_test, fh):
        rows = partial(row_quantiles, load(model_id))
        mode = 'known' if covariates else None
        return foundation_quantiles(rows, X_train, y_train, X_test, fh, [quantile], mode)[:, 0]

    return reference


# Fabrique de la référence d'un modèle Chronos appelé directement
def chronos_reference(model_id: str, quantile: float = 0.5, covariates: bool = True) -> NativeReference:
    """Reference of a Chronos model: one ``predict_quantiles`` call per test row and entity.

    See :func:`foundation_quantiles`; the covariates are those of Chronos-2.

    Args:
        model_id: Hugging Face id of the model.
        quantile: Level returned.
        covariates: Whether the features are passed (Chronos-2 only).

    Returns:
        Native reference.
    """
    return _foundation_reference(chronos_pipeline, chronos_row_quantiles, model_id, quantile, covariates)


# Fabrique de la référence d'un modèle Timestral appelé directement
def timestral_reference(model_id: str, quantile: float = 0.5, covariates: bool = True) -> NativeReference:
    """Reference of a Timestral model: one ``forecast`` call per test row and entity.

    See :func:`foundation_quantiles`.

    Args:
        model_id: Hugging Face id of the model.
        quantile: Level returned, one of the model.
        covariates: Whether the features are passed as known covariates.

    Returns:
        Native reference.
    """
    return _foundation_reference(timestral_forecaster, timestral_row_quantiles, model_id, quantile, covariates)


# Fabrique de la référence d'un modèle t0 appelé directement
def t0_reference(model_id: str, quantile: float = 0.5, covariates: bool = True) -> NativeReference:
    """Reference of a t0 model: one ``predict`` call per test row and entity.

    See :func:`foundation_quantiles`.

    Args:
        model_id: Hugging Face id of the model.
        quantile: Level returned.
        covariates: Whether the features are passed as known covariates.

    Returns:
        Native reference.
    """
    return _foundation_reference(t0_forecaster, t0_row_quantiles, model_id, quantile, covariates)
