"""Adapter configurations checked by the contract battery of ``tests.support.adapter_contract``.

``sktime_naive_case`` (``SktimeAdapter`` around ``NaiveForecaster``, a local time
series model) and ``sktime_yfromx_case`` (around ``YfromX(LinearRegression())``,
a row-wise regressor) are shared by the unit and integration tests. A third-party
package is imported only when an estimator is built.
"""
from sklearn.linear_model import LinearRegression

from tests.support.adapter_contract import AdapterCase
from tsforecast.adapters import SktimeAdapter


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
