"""Adapter configurations checked by the contract battery of ``tests.support.adapter_contract``.

``sktime_naive_case`` (``SktimeAdapter`` around ``NaiveForecaster``, a local time
series model), ``sktime_yfromx_case`` (around ``YfromX(LinearRegression())``, a
row-wise regressor) and the ``DartsAdapter`` cases (row-wise linear and random
forest models, ``NaiveSeasonal`` local model, lagged global linear model) and the
``HierarchicalForecastAdapter`` cases (``BottomUp``, ``MinTrace`` ``'ols'`` with the
mean of two base models, ``MinTrace`` ``'mint_shrink'`` with one base model per
level, on a node tree; ``MinTrace`` ``'ols'`` on a hierarchy of index levels; the
temporal and cross-temporal cases on years, quarters and months) are shared by the
unit and integration tests. Every case on data that its package handles alone
carries the native reference of its model; ``sktime_native_cases`` and
``darts_native_cases`` add models checked only against their package used
natively. A third-party package is imported only when an estimator, a parameter
grid or a native reference is built. ``ensemble_meta_case`` and
``ensemble_passthrough_case`` check ``tsforecast.ensemble.OnlineAggregator`` with the
same battery (no third-party package, no native reference); ``foundation_drift_case``
(fake backend, offline) and ``foundation_chronos_case`` (``autogluon/chronos-2-small``)
check ``FoundationModelAdapter``.
"""
from functools import partial
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from tests.support import native_references as native
from tests.support.adapter_contract import AdapterCase, NativeReference, default_data, default_splitter
from tests.support.datasets import (
    HIERARCHY_LEVELS,
    HIERARCHY_TREE,
    TEMPORAL_HIERARCHY,
    build_cross_temporal_hierarchy,
    build_hierarchy_levels,
    build_hierarchy_nodes,
    build_temporal_hierarchy,
)
from tests.support.forecast_backends import DriftBackend
from tsforecast.adapters import DartsAdapter, FoundationModelAdapter, HierarchicalForecastAdapter, SktimeAdapter
from tsforecast.ensemble import OnlineAggregator


# Fabrique d'une référence native construite à son premier appel
def _lazy(build: Callable[[], NativeReference]) -> NativeReference:
    """Native reference returned by ``build()`` at call time, the third-party package imported then."""

    # Référence construite à l'appel : le paquet tiers n'est importé que par le test
    def reference(X_train, y_train, X_test, fh):
        return build()(X_train, y_train, X_test, fh)

    return reference


def _naive_adapter() -> SktimeAdapter:
    """``SktimeAdapter`` around a default ``NaiveForecaster``."""
    from sktime.forecasting.naive import NaiveForecaster

    return SktimeAdapter(NaiveForecaster())


def _yfromx_adapter() -> SktimeAdapter:
    """``SktimeAdapter`` around ``YfromX(LinearRegression())``."""
    from sktime.forecasting.compose import YfromX

    return SktimeAdapter(YfromX(LinearRegression()))


def sktime_naive_case() -> AdapterCase:
    """Contract case of a sktime forecaster ignoring ``X`` and extrapolating the target.

    Returns:
        Local per-entity model whose prediction depends on the gap.
    """
    return AdapterCase(
        make_estimator=_naive_adapter,
        wrapped_param='forecaster',
        param_grid={'forecaster__strategy': ['last', 'mean']},
        package='sktime',
        extra='adapters-sktime',
        local_model=True,
        row_regressor=False,
        invalid_params=({'forecaster__strategy': 'unknown'},),
        native_reference=_lazy(lambda: native.sktime_reference(_naive_adapter().forecaster)),
    )


def sktime_yfromx_case() -> AdapterCase:
    """Contract case of a sktime forecaster predicting each row from its features.

    Returns:
        Local per-entity row-wise regressor, independent of the gap.
    """
    return AdapterCase(
        make_estimator=_yfromx_adapter,
        wrapped_param='forecaster',
        param_grid={'forecaster__estimator__fit_intercept': [True, False]},
        package='sktime',
        extra='adapters-sktime',
        local_model=True,
        row_regressor=True,
        invalid_params=({'forecaster__estimator__fit_intercept': 'yes'},),
        native_reference=_lazy(lambda: native.sktime_reference(_yfromx_adapter().forecaster, uses_X=True)),
    )


def _darts_linear_adapter() -> DartsAdapter:
    """``DartsAdapter`` around a row-wise ``LinearRegressionModel``."""
    from darts.models import LinearRegressionModel

    return DartsAdapter(LinearRegressionModel(lags_future_covariates=[0]))


def _darts_forest_adapter() -> DartsAdapter:
    """``DartsAdapter`` around a small, seeded, row-wise ``RandomForestModel``."""
    from darts.models import RandomForestModel

    return DartsAdapter(RandomForestModel(lags_future_covariates=[0], n_estimators=10, random_state=0))


def _darts_seasonal_adapter() -> DartsAdapter:
    """``DartsAdapter`` around the local ``NaiveSeasonal`` model repeating the last value."""
    from darts.models import NaiveSeasonal

    return DartsAdapter(NaiveSeasonal(K=1))


def _darts_lagged_adapter() -> DartsAdapter:
    """``DartsAdapter`` around a global ``LinearRegressionModel`` on three target lags."""
    from darts.models import LinearRegressionModel

    return DartsAdapter(LinearRegressionModel(lags=3))


# Référence native de la forêt aléatoire ligne à ligne : forêt scikit-learn de mêmes paramètres
def _forest_reference() -> NativeReference:
    """Pooled ``RandomForestRegressor`` with the parameters of the darts forest of the case."""
    from sklearn.ensemble import RandomForestRegressor

    return native.pooled_sklearn_reference(RandomForestRegressor(n_estimators=10, random_state=0))


def darts_linear_case() -> AdapterCase:
    """Contract case of a darts linear model predicting each row from its features.

    Returns:
        Global row-wise regressor, independent of the gap.
    """
    return AdapterCase(
        make_estimator=_darts_linear_adapter,
        wrapped_param='model',
        param_grid={'model__fit_intercept': [True, False]},
        package='darts',
        extra='adapters-darts',
        local_model=False,
        row_regressor=True,
        invalid_params=({'model__add_encoders': {'position': {'future': ['relative']}}},),
        native_reference=native.pooled_sklearn_reference(LinearRegression()),
    )


def darts_forest_case() -> AdapterCase:
    """Contract case of a darts random forest predicting each row from its features.

    Returns:
        Global row-wise tree model, independent of the gap.
    """
    return AdapterCase(
        make_estimator=_darts_forest_adapter,
        wrapped_param='model',
        param_grid={'model__max_depth': [2, None]},
        package='darts',
        extra='adapters-darts',
        local_model=False,
        row_regressor=True,
        invalid_params=({'model__n_estimators': 0},),
        native_reference=_lazy(_forest_reference),
    )


def darts_seasonal_case() -> AdapterCase:
    """Contract case of a darts local model repeating the last season of each entity.

    Returns:
        Local per-entity model, depending on the gap, refusing training holes.
    """
    return AdapterCase(
        make_estimator=_darts_seasonal_adapter,
        wrapped_param='model',
        param_grid={'model__K': [1, 3]},
        package='darts',
        extra='adapters-darts',
        local_model=True,
        row_regressor=False,
        handles_training_holes=False,
        native_reference=_lazy(lambda: native.darts_local_reference(_darts_seasonal_adapter().model)),
    )


def darts_lagged_case() -> AdapterCase:
    """Contract case of a darts global model extrapolating three target lags.

    Returns:
        Global model fitted on all entities, extrapolating the history of each
        one: an entity unknown at fit cannot be predicted.
    """
    return AdapterCase(
        make_estimator=_darts_lagged_adapter,
        wrapped_param='model',
        param_grid={'model__lags': [2, 3]},
        package='darts',
        extra='adapters-darts',
        local_model=True,
        row_regressor=False,
        handles_training_holes=False,
        invalid_params=({'model__lags': 40},),
        native_reference=_lazy(lambda: native.darts_global_reference(_darts_lagged_adapter().model)),
    )


# Modèle de base par niveau de l'arbre des jeux hiérarchiques
HIERARCHY_MODEL_BY_LEVEL = {'depth_0': 'm1', 'depth_1': 'm2', 'depth_2': 'm1'}
# Paramètres invalides d'un adapter hiérarchique (arbre)
_HIERARCHY_INVALID_PARAMS = (
    {'model_combination': 'median'},
    {'hierarchy': {'a': ['b'], 'b': ['a']}},
    {'hierarchy': {'a': ['b', 'c'], 'd': ['c', 'e']}},
)


def _hierarchy_splitter(kind: str):
    """Panel splitter for every kind: hierarchical data always carry a node level."""
    return default_splitter('panel')


def _reconciler(name: str):
    """Build a hierarchicalforecast reconciler: ``'bottomup'``, ``'ols'`` or ``'mint_shrink'``."""
    from hierarchicalforecast.methods import BottomUp, MinTrace

    return BottomUp() if name == 'bottomup' else MinTrace(method=name)


def _hierarchy_adapter(reconciler: str, **params) -> HierarchicalForecastAdapter:
    """``HierarchicalForecastAdapter`` around a new reconciler."""
    params.setdefault('hierarchy', HIERARCHY_TREE)
    return HierarchicalForecastAdapter(reconciler=_reconciler(reconciler), **params)


def _hierarchy_case(make_estimator, make_data, invalid_params, native_reference=None) -> AdapterCase:
    """Contract case of a reconciliation: a row-wise, date-wise operation on every copy."""
    from hierarchicalforecast.methods import BottomUp, MinTrace

    return AdapterCase(
        make_estimator=make_estimator,
        wrapped_param='reconciler',
        param_grid={'reconciler': [BottomUp(), MinTrace(method='ols')]},
        package='hierarchicalforecast',
        extra='adapters-hierarchical',
        local_model=False,
        row_regressor=True,
        invalid_params=invalid_params,
        make_data=make_data,
        make_splitter=_hierarchy_splitter,
        frame_method='reconcile',
        native_reference=native_reference,
    )


def hierarchical_bottomup_case() -> AdapterCase:
    """Contract case of ``BottomUp`` on the node tree, base model ``m1``.

    Returns:
        Reconciliation summing the bottom forecasts, independent of the gap.
    """
    return _hierarchy_case(
        partial(_hierarchy_adapter, 'bottomup', forecast_columns=['m1']),
        build_hierarchy_nodes,
        _HIERARCHY_INVALID_PARAMS,
        _lazy(lambda: native.reconcile_reference(_reconciler('bottomup'), native.TREE, 'm1')),
    )


def hierarchical_ols_case() -> AdapterCase:
    """Contract case of ``MinTrace(method='ols')`` on the node tree, mean of ``m1`` and ``m2``.

    Returns:
        Orthogonal projection on the coherent forecasts, independent of the gap.
    """
    return _hierarchy_case(
        partial(_hierarchy_adapter, 'ols', model_combination='mean'),
        build_hierarchy_nodes,
        _HIERARCHY_INVALID_PARAMS,
        _lazy(lambda: native.reconcile_reference(_reconciler('ols'), native.TREE, 'mean')),
    )


def hierarchical_mint_shrink_case() -> AdapterCase:
    """Contract case of ``MinTrace(method='mint_shrink')`` on the node tree, one base model per level.

    Returns:
        Reconciliation weighted by the in-sample errors of each copy, fixed at
        fit, independent of the gap.
    """
    return _hierarchy_case(
        partial(_hierarchy_adapter, 'mint_shrink', model_combination=HIERARCHY_MODEL_BY_LEVEL),
        build_hierarchy_nodes,
        _HIERARCHY_INVALID_PARAMS,
        _lazy(
            lambda: native.reconcile_reference(_reconciler('mint_shrink'), native.TREE, HIERARCHY_MODEL_BY_LEVEL)
        ),
    )


def hierarchical_levels_case() -> AdapterCase:
    """Contract case of ``MinTrace(method='ols')`` on a hierarchy of index levels, base model ``m2``.

    Returns:
        Reconciliation of countries and regions, the copies being markets.
    """
    return _hierarchy_case(
        partial(_hierarchy_adapter, 'ols', hierarchy=HIERARCHY_LEVELS, forecast_columns=['m2']),
        build_hierarchy_levels,
        (
            {'node_level': 'region'},
            {'hierarchy': [['region'], ['country']]},
            {'hierarchy': [['country'], ['country', 'city']]},
        ),
        _lazy(lambda: native.reconcile_reference(_reconciler('ols'), native.LEVELS, 'm2')),
    )


# Paramètres invalides d'un adapter temporel
_TEMPORAL_INVALID_PARAMS = (
    {'temporal_hierarchy': {'year': 12, 'quarter': 3}},
    {'temporal_hierarchy': {'year': 12, 'five_months': 5, 'month': 1}},
    {'temporal_origin': 'not a date'},
    {'temporal_level': 'region'},
)


def hierarchical_temporal_case() -> AdapterCase:
    """Contract case of ``MinTrace(method='ols')`` on years, quarters and months, base model ``m1``.

    Only the checks that do not cut test windows on arbitrary dates apply: a
    temporal reconciliation needs complete years (see ``period_splits``).

    Returns:
        Temporal reconciliation of each series, one calendar year at a time.
    """
    return _hierarchy_case(
        partial(
            _hierarchy_adapter, 'ols', hierarchy=None, forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        ),
        build_temporal_hierarchy,
        _TEMPORAL_INVALID_PARAMS + ({'reconciler': _reconciler('mint_shrink')},),
    )


def hierarchical_cross_temporal_case() -> AdapterCase:
    """Contract case of ``MinTrace(method='ols')`` on the node tree crossed with years, quarters and months.

    Returns:
        Cross-temporal reconciliation of each copy, one calendar year at a time.
    """
    return _hierarchy_case(
        partial(_hierarchy_adapter, 'ols', model_combination='mean', temporal_hierarchy=TEMPORAL_HIERARCHY),
        build_cross_temporal_hierarchy,
        _TEMPORAL_INVALID_PARAMS + ({'hierarchy': {'a': ['b'], 'b': ['a']}},),
    )


# ---------------------------------------------------------------------------
# Modèles comparés au seul paquet natif
# ---------------------------------------------------------------------------


# Cas d'un modèle soumis à la seule règle d'équivalence au paquet natif
def _native_case(make_estimator, wrapped_param: str, package: str, extra: str, reference, gaps) -> AdapterCase:
    """Case of a model checked only against its package used natively."""
    return AdapterCase(
        make_estimator=make_estimator,
        wrapped_param=wrapped_param,
        param_grid={},
        package=package,
        extra=extra,
        local_model=True,
        row_regressor=False,
        native_reference=reference,
        native_gaps=gaps,
    )


def sktime_native_cases() -> Dict[str, AdapterCase]:
    """sktime forecasters checked against sktime used natively, beyond the contract cases.

    Returns:
        Dictionary name -> case: drift, additive-trend exponential smoothing
        and Theta (without deseasonalization: the target of the adapter
        datasets takes negative values, which a multiplicative decomposition
        refuses) through any gap; ``SARIMAX`` reading ``X`` without gap only,
        sktime needing ``X`` on every date up to the forecast.
    """
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.sarimax import SARIMAX
    from sktime.forecasting.theta import ThetaForecaster

    # Dictionnaire nom -> (forecaster, lecture de X, gaps traités nativement)
    forecasters = {
        'drift': (NaiveForecaster(strategy='drift'), False, (0, 2)),
        'exponential_smoothing': (ExponentialSmoothing(trend='add'), False, (0, 2)),
        'theta': (ThetaForecaster(deseasonalize=False), False, (0, 2)),
        'sarimax': (SARIMAX(order=(1, 0, 0)), True, (0,)),
    }
    # Un cas par forecaster, l'adapter et la référence partageant le forecaster non ajusté
    return {
        name: _native_case(
            partial(SktimeAdapter, forecaster),
            'forecaster',
            'sktime',
            'adapters-sktime',
            native.sktime_reference(forecaster, uses_X=uses_X),
            gaps,
        )
        for name, (forecaster, uses_X, gaps) in forecasters.items()
    }


def darts_native_cases() -> Dict[str, AdapterCase]:
    """darts models checked against darts used natively, beyond the contract cases.

    Returns:
        Dictionary name -> case: local ``ExponentialSmoothing`` through any
        gap; global ``LinearRegressionModel`` on target lags, past and future
        covariates without gap only, darts needing the covariates on every
        date up to the forecast.
    """
    from darts.models import ExponentialSmoothing, LinearRegressionModel

    # Modèles non ajustés, partagés par l'adapter et la référence
    smoothing = ExponentialSmoothing()
    lagged = LinearRegressionModel(lags=3, lags_past_covariates=2, lags_future_covariates=[0])
    return {
        'exponential_smoothing': _native_case(
            partial(DartsAdapter, smoothing),
            'model',
            'darts',
            'adapters-darts',
            native.darts_local_reference(smoothing),
            (0, 2),
        ),
        'lagged_covariates': _native_case(
            partial(DartsAdapter, lagged),
            'model',
            'darts',
            'adapters-darts',
            native.darts_global_reference(lagged, past=True, future=True),
            (0,),
        ),
    }


# ---------------------------------------------------------------------------
# Agrégation d'experts en ligne (tsforecast.ensemble)
# ---------------------------------------------------------------------------

# Paramètres refusés au fit par l'agrégateur, quel que soit son mode
_AGGREGATOR_INVALID_PARAMS = (
    {'aggregation': 'EWA'},
    {'loss': 'huber'},
    {'loss': 'pinball', 'quantile': 1.5},
    {'pooling': 'local'},
    {'init_weights': [1.0]},
    {'loss_gradient': 'yes'},
)


# Fabrique de l'agrégateur méta-estimateur des cas de contrat
def _meta_aggregator() -> OnlineAggregator:
    """Meta-estimator over a ridge and a shallow tree, three internal folds."""
    return OnlineAggregator(
        [('ridge', Ridge(alpha=1.0)), ('tree', DecisionTreeRegressor(max_depth=2, random_state=0))],
        cv=3,
    )


# Fonction de construction de prévisions d'experts sur les jeux de la batterie
def expert_forecast_data(kind: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return expert forecasts of the target of a dataset kind of the battery.

    The rows and target are those of :func:`default_data`; the columns are
    three experts: ``sharp`` (target plus a small noise), ``rough`` (larger
    noise) and ``biased`` (target plus a constant).

    Args:
        kind: ``'series'``, ``'panel'`` or ``'unbalanced_panel'``.

    Returns:
        ``(X, y)``: expert forecasts and target, on the same index.
    """
    X, y = default_data(kind)
    # Bruits des experts, tirés une fois pour toutes les lignes
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((len(y), 2))
    forecasts = pd.DataFrame(
        {'sharp': y + 0.1 * noise[:, 0], 'rough': y + noise[:, 1], 'biased': y + 2.0},
        index=X.index,
    )
    return forecasts, y


# Cas de contrat de l'agrégateur en mode méta-estimateur
def ensemble_meta_case() -> AdapterCase:
    """``OnlineAggregator`` as a meta-estimator: ridge and tree experts, nested grid."""
    return AdapterCase(
        make_estimator=_meta_aggregator,
        wrapped_param='experts',
        param_grid={'aggregation': ['BOA', 'MLpol'], 'experts__ridge__alpha': [0.1, 10.0]},
        invalid_params=_AGGREGATOR_INVALID_PARAMS + ({'cv': 0},),
    )


# Cas de contrat de l'agrégateur en mode passthrough
def ensemble_passthrough_case() -> AdapterCase:
    """``OnlineAggregator`` on precomputed expert forecasts (columns of ``X``)."""
    return AdapterCase(
        make_estimator=lambda: OnlineAggregator('passthrough'),
        wrapped_param='experts',
        param_grid={'aggregation': ['BOA', 'MLpol'], 'loss': ['mse', 'mae']},
        invalid_params=_AGGREGATOR_INVALID_PARAMS,
        make_data=expert_forecast_data,
    )


# ---------------------------------------------------------------------------
# Modèles de fondation (FoundationModelAdapter)
# ---------------------------------------------------------------------------

# Paramètres du backend factice des cas de contrat : dérive et poids des covariables connues
DRIFT, DRIFT_COEF = 0.5, 0.25
# Plus petit modèle Chronos-2 et plus petit modèle Chronos-Bolt
CHRONOS_2_SMALL = 'autogluon/chronos-2-small'
CHRONOS_BOLT_TINY = 'amazon/chronos-bolt-tiny'
# Paramètres refusés au fit, quel que soit le backend
_FOUNDATION_INVALID_PARAMS = (
    {'quantile': 1.5},
    {'context_length': 0},
    {'covariates': 'future'},
    {'batch_size': 0},
)


# Fabrique de l'adapter du backend factice
def _drift_adapter() -> FoundationModelAdapter:
    """Adapter around ``DriftBackend`` reading the known covariates."""
    return FoundationModelAdapter(backend=DriftBackend(drift=DRIFT, coef=DRIFT_COEF))


# Cas de contrat du backend factice
def foundation_drift_case() -> AdapterCase:
    """``FoundationModelAdapter`` around the fake ``DriftBackend``, with known covariates.

    Returns:
        Local per-entity model whose prediction depends on the gap; native
        reference computed by hand.
    """
    return AdapterCase(
        make_estimator=_drift_adapter,
        wrapped_param='backend',
        param_grid={'backend__drift': [0.0, DRIFT], 'context_length': [6, None]},
        local_model=True,
        row_regressor=False,
        invalid_params=_FOUNDATION_INVALID_PARAMS + ({'backend__max_context': 0},),
        frame_method='predict_quantiles',
        native_reference=native.drift_reference(DRIFT, DRIFT_COEF),
    )


# Cas de contrat de Chronos-2 (téléchargement du modèle)
def foundation_chronos_case() -> AdapterCase:
    """``FoundationModelAdapter`` around ``autogluon/chronos-2-small``, with known covariates.

    Returns:
        Case of the real model; native reference calling ``Chronos2Pipeline``
        directly; float32 tolerances.
    """
    return AdapterCase(
        make_estimator=lambda: FoundationModelAdapter(CHRONOS_2_SMALL),
        wrapped_param='model_id',
        param_grid={'context_length': [12, None]},
        package='chronos',
        extra='adapters-chronos',
        local_model=True,
        row_regressor=False,
        invalid_params=_FOUNDATION_INVALID_PARAMS
        + ({'quantile': 0.001}, {'model_id': 'amazon/chronos-t5-tiny'}),
        frame_method='predict_quantiles',
        native_reference=native.chronos_reference(CHRONOS_2_SMALL),
        rtol=1e-5,
        atol=1e-6,
    )
