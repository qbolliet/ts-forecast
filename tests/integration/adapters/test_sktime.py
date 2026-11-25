"""Workflow tests of ``SktimeAdapter``: the integration rules of the adapter contract.

``cross_val_score`` with ``tsforecast`` splitters and a gap (series, balanced and
unbalanced panels), ``GridSearchCV`` over a nested forecaster parameter,
``XYPipeline`` with ``DropNaNTarget`` and ``sklearn.pipeline.Pipeline``, for a
forecaster ignoring ``X`` (``NaiveForecaster``) and a row-wise one (``YfromX``).
"""
import pytest

pytest.importorskip('sktime')

from tests.support.adapter_cases import sktime_naive_case, sktime_yfromx_case  # noqa: E402
from tests.support.adapter_contract import AdapterWorkflowContractTests  # noqa: E402


class TestSktimeNaiveWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of ``SktimeAdapter(NaiveForecaster())``."""

    case = sktime_naive_case()


class TestSktimeYfromXWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of ``SktimeAdapter(YfromX(LinearRegression()))``."""

    case = sktime_yfromx_case()
