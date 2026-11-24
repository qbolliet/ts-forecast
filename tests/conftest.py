"""Root configuration of the ``tests/`` suite.

Registers the shared fixtures of ``tests/support/fixtures.py``,
transiently ignores the modules whose import is broken by a renaming
not yet handled by the test campaign, and automatically marks each
test ``unit`` or ``integration`` according to its path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["tests.support.fixtures"]


# =============================================================================
# Modules dont l'import est cassé — TRANSITOIRE, retiré au prompt qui répare
# =============================================================================
collect_ignore: list[str] = []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark each test ``unit`` or ``integration``.

    The marker is derived from the test path relative to ``tests/``:
    ``tests/integration/...`` gets ``integration``, everything else
    (``tests/unit/...`` and ``tests/support/...``) gets ``unit``.

    Args:
        config: Pytest configuration of the session.
        items: Collected test items, modified in place.
    """
    root = Path(config.rootpath) / "tests"

    for item in items:
        relative_path = Path(item.fspath).relative_to(root)

        if relative_path.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

