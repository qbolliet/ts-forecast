"""Adapter configurations checked by the contract battery of ``tests.support.adapter_contract``.

``sktime_naive_case`` (``SktimeAdapter`` around ``NaiveForecaster``, a local time
series model), ``sktime_yfromx_case`` (around ``YfromX(LinearRegression())``, a
row-wise regressor) and the ``DartsAdapter`` cases (row-wise linear and random
forest models, ``NaiveSeasonal`` local model, lagged global linear model) are shared
by the unit and integration tests. A third-party package is imported only when an
estimator is built.
"""
from sklearn.linear_model import LinearRegression

from tests.support.adapter_contract import AdapterCase
from tsforecast.adapters import DartsAdapter, SktimeAdapter


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
