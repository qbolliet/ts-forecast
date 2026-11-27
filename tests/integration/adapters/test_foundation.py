"""Workflow tests of ``FoundationModelAdapter``: the integration rules of the adapter contract.

``cross_val_score`` with ``tsforecast`` splitters and a gap (series, balanced and
unbalanced panels), ``GridSearchCV`` over a nested backend parameter and the context
length, ``XYPipeline`` with ``DropNaNTarget`` and ``sklearn.pipeline.Pipeline``, on the
fake ``DriftBackend`` (offline) and on ``autogluon/chronos-2-small`` (marked ``slow``
and ``network``, skipped without chronos-forecasting or network).
"""
import pytest

from tests.support.adapter_cases import CHRONOS_2_SMALL, foundation_chronos_case, foundation_drift_case
from tests.support.adapter_contract import AdapterWorkflowContractTests
from tsforecast.adapters.foundation import registry


@pytest.fixture(scope='module')
def chronos_small():
    """``autogluon/chronos-2-small`` loaded through the registry, or the test skipped."""
    pytest.importorskip('chronos')
    try:
        return registry.load_backend(registry.resolve_backend(CHRONOS_2_SMALL), CHRONOS_2_SMALL, device='cpu')
    except OSError as error:
        pytest.skip(f'{CHRONOS_2_SMALL} cannot be downloaded: {error}')


class TestDriftWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of the adapter around the fake backend with known covariates."""

    case = foundation_drift_case()


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('chronos_small')
class TestChronosWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of the adapter around Chronos-2 small, with known covariates."""

    case = foundation_chronos_case()
