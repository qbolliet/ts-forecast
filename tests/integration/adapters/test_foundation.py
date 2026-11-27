"""Workflow tests of ``FoundationModelAdapter``: the integration rules of the adapter contract.

``cross_val_score`` with ``tsforecast`` splitters and a gap (series, balanced and
unbalanced panels), ``GridSearchCV`` over a nested backend parameter and the context
length, ``XYPipeline`` with ``DropNaNTarget`` and ``sklearn.pipeline.Pipeline``, on the
fake ``DriftBackend`` (offline) and on ``autogluon/chronos-2-small``,
``mistralai/Timestral-5M`` and ``theforecastingcompany/t0-gamma`` (marked ``slow``
and ``network``, skipped without their package or network).
"""
import pytest

from tests.support.adapter_cases import (
    CHRONOS_2_SMALL,
    T0_GAMMA,
    TIMESTRAL_5M,
    foundation_chronos_case,
    foundation_drift_case,
    foundation_t0_case,
    foundation_timestral_case,
)
from tests.support.adapter_contract import AdapterWorkflowContractTests
from tsforecast.adapters.foundation import registry


# Chargement d'un modèle réel par le registre, test sauté sans paquet ou sans réseau
def _load_or_skip(package: str, model_id: str):
    """The backend of ``model_id`` loaded through the registry, or the test skipped."""
    pytest.importorskip(package)
    try:
        return registry.load_backend(registry.resolve_backend(model_id), model_id, device='cpu')
    except OSError as error:
        pytest.skip(f'{model_id} cannot be downloaded: {error}')


@pytest.fixture(scope='module')
def chronos_small():
    """``autogluon/chronos-2-small`` loaded through the registry, or the test skipped."""
    return _load_or_skip('chronos', CHRONOS_2_SMALL)


@pytest.fixture(scope='module')
def timestral_5m():
    """``mistralai/Timestral-5M`` loaded through the registry, or the test skipped."""
    return _load_or_skip('timestral', TIMESTRAL_5M)


@pytest.fixture(scope='module')
def t0_gamma():
    """``theforecastingcompany/t0-gamma`` loaded through the registry, or the test skipped."""
    return _load_or_skip('t0', T0_GAMMA)


class TestDriftWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of the adapter around the fake backend with known covariates."""

    case = foundation_drift_case()


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('chronos_small')
class TestChronosWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of the adapter around Chronos-2 small, with known covariates."""

    case = foundation_chronos_case()


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('timestral_5m')
class TestTimestralWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of the adapter around Timestral-5M, with known covariates."""

    case = foundation_timestral_case()


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('t0_gamma')
class TestT0Workflow(AdapterWorkflowContractTests):
    """Workflow contract of the adapter around t0-gamma, with known covariates."""

    case = foundation_t0_case()
