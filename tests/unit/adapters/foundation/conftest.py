"""Shared fixtures of the ``tsforecast.adapters.foundation`` unit tests.

``isolated_registry`` gives a test its own copy of the backend registry and an
empty model cache; ``chronos_small``, ``timestral_5m`` and ``t0_gamma`` load
``autogluon/chronos-2-small``, ``mistralai/Timestral-5M`` and
``theforecastingcompany/t0-gamma`` through the registry, skipping the test without
their package or without network.
"""
import pytest

from tests.support.adapter_cases import CHRONOS_2_SMALL, T0_GAMMA, TIMESTRAL_5M
from tsforecast.adapters.foundation import registry


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    """Registry copy and empty model cache for one test; the module returned for registrations."""
    monkeypatch.setattr(registry, '_REGISTRY', dict(registry._REGISTRY))
    monkeypatch.setattr(registry, '_MODEL_CACHE', {})
    return registry


# Chargement d'un modèle réel par le registre, test sauté sans paquet ou sans réseau
def _load_or_skip(package: str, model_id: str):
    """The backend of ``model_id`` loaded through the registry, or the test skipped."""
    pytest.importorskip(package)
    try:
        return registry.load_backend(registry.resolve_backend(model_id), model_id, device='cpu')
    except OSError as error:
        pytest.skip(f'{model_id} cannot be downloaded: {error}')


@pytest.fixture(scope='session')
def chronos_small():
    """``autogluon/chronos-2-small`` loaded through the registry, or the test skipped."""
    return _load_or_skip('chronos', CHRONOS_2_SMALL)


@pytest.fixture(scope='session')
def timestral_5m():
    """``mistralai/Timestral-5M`` loaded through the registry, or the test skipped."""
    return _load_or_skip('timestral', TIMESTRAL_5M)


@pytest.fixture(scope='session')
def t0_gamma():
    """``theforecastingcompany/t0-gamma`` loaded through the registry, or the test skipped."""
    return _load_or_skip('t0', T0_GAMMA)
