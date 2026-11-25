"""Workflow tests of ``DartsAdapter``: the integration rules of the adapter contract.

``cross_val_score`` with ``tsforecast`` splitters and a gap (series, balanced and
unbalanced panels), ``GridSearchCV`` over a nested model parameter, ``XYPipeline``
with ``DropNaNTarget`` and ``sklearn.pipeline.Pipeline``, for row-wise models
(``LinearRegressionModel``, ``RandomForestModel``) and extrapolating models (local
``NaiveSeasonal``, global ``LinearRegressionModel`` on target lags).
"""
import pytest

pytest.importorskip('darts')

from tests.support.adapter_cases import (  # noqa: E402
    darts_forest_case,
    darts_lagged_case,
    darts_linear_case,
    darts_seasonal_case,
)
from tests.support.adapter_contract import AdapterWorkflowContractTests  # noqa: E402


class TestDartsLinearWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of a row-wise ``LinearRegressionModel``."""

    case = darts_linear_case()


class TestDartsForestWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of a row-wise ``RandomForestModel``."""

    case = darts_forest_case()


class TestDartsSeasonalWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of the local ``NaiveSeasonal`` model."""

    case = darts_seasonal_case()


class TestDartsLaggedWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of a global ``LinearRegressionModel`` on target lags."""

    case = darts_lagged_case()
