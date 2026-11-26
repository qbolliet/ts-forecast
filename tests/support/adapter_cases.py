"""Adapter configurations checked by the contract battery of ``tests.support.adapter_contract``.

``sktime_naive_case`` (``SktimeAdapter`` around ``NaiveForecaster``, a local time
series model), ``sktime_yfromx_case`` (around ``YfromX(LinearRegression())``, a
row-wise regressor) and the ``DartsAdapter`` cases (row-wise linear and random
forest models, ``NaiveSeasonal`` local model, lagged global linear model) and the
``HierarchicalForecastAdapter`` cases (``BottomUp``, ``MinTrace`` ``'ols'`` with the
mean of two base models, ``MinTrace`` ``'mint_shrink'`` with one base model per
level, on a node tree; ``MinTrace`` ``'ols'`` on a hierarchy of index levels) are
shared by the unit and integration tests. A third-party package is imported only
when an estimator or a parameter grid is built.
"""
from functools import partial

from sklearn.linear_model import LinearRegression

from tests.support.adapter_contract import AdapterCase, default_splitter
from tests.support.datasets import (
    HIERARCHY_LEVELS,
    HIERARCHY_TREE,
    build_hierarchy_levels,
    build_hierarchy_nodes,
)
from tsforecast.adapters import DartsAdapter, HierarchicalForecastAdapter, SktimeAdapter


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


def _hierarchy_case(make_estimator, make_data, invalid_params) -> AdapterCase:
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
    )
