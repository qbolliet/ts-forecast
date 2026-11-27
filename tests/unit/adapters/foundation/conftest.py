"""Shared fixtures of the ``tsforecast.adapters.foundation`` unit tests.

``isolated_registry`` gives a test its own copy of the backend registry and an
empty model cache; ``chronos_small`` loads ``autogluon/chronos-2-small`` once per
session, skipping the test without chronos-forecasting or without network.
"""
import pytest

from tests.support.adapter_cases import CHRONOS_2_SMALL
from tsforecast.adapters.foundation import registry


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    """Registry copy and empty model cache for one test; the module returned for registrations."""
    monkeypatch.setattr(registry, '_REGISTRY', dict(registry._REGISTRY))
    monkeypatch.setattr(registry, '_MODEL_CACHE', {})
    return registry


@pytest.fixture(scope='session')
def chronos_small():
    """``autogluon/chronos-2-small`` loaded through the registry, or the test skipped."""
    pytest.importorskip('chronos')
    try:
        return registry.load_backend(registry.resolve_backend(CHRONOS_2_SMALL), CHRONOS_2_SMALL, device='cpu')
    except OSError as error:
        pytest.skip(f'{CHRONOS_2_SMALL} cannot be downloaded: {error}')
