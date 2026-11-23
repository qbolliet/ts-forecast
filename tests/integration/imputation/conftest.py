"""Fixtures of the ``frequency`` integration tests: cached fits of the scenario matrix."""
# Modules de base
from typing import Callable, Dict, Tuple

import pytest

from tests.integration.imputation.realistic_runs import DATASET_FIXTURES, build_record


@pytest.fixture(scope='session')
def realistic_run(request) -> Callable[[str, str, bool], dict]:
    """Fit each cell of the matrix of ``realistic_runs.py`` once per session.

    Returns:
        Callable ``(dataset, strategy, modality) -> record`` (see
        :func:`~tests.integration.imputation.realistic_runs.build_record`). The
        records are shared between modules: a test must never mutate them.
    """
    cache: Dict[Tuple[str, str, str], dict] = {}

    def _get(dataset: str, strategy: str, modality: bool) -> dict:
        # Clé de cache : la modalité est encodée en str
        key = (dataset, strategy, str(modality))
        if key not in cache:
            data = request.getfixturevalue(DATASET_FIXTURES[dataset]).copy()
            cache[key] = build_record(data, strategy, modality)
        return cache[key]

    return _get


@pytest.fixture
def record(realistic_run, dataset: str, strategy: str, modality: bool) -> dict:
    """Record of the matrix cell the test is parametrized on.

    Moves the (cached) fit into the fixture setup: the cost of a cell is paid
    once per session, by the setup of its first test, never by a test body.
    """
    return realistic_run(dataset, strategy, modality)
