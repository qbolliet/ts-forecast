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


# =============================================================================
# Échecs hérités — TRANSITOIRE (voir tests/legacy_failures.txt)
# =============================================================================
def _load_legacy_failures() -> dict[str, str]:
    """Read ``tests/legacy_failures.txt`` and map each node id to its reason.

    File format: lines grouped by block, each block preceded by a
    ``# prompt <id>`` or ``# hors campagne`` comment; empty lines and
    header comments (before the first group) are ignored.

    Returns:
        Dictionary ``{node_id: xfail_reason}``.
    """
    path = Path(__file__).parent / "legacy_failures.txt"
    if not path.exists():
        return {}

    failures: dict[str, str] = {}
    current_prompt = "?"
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# prompt "):
            current_prompt = line.removeprefix("# prompt ").strip()
            continue
        if line.startswith("# hors campagne"):
            current_prompt = "hors campagne"
            continue
        if line.startswith("#"):
            continue
        failures[line] = f"legacy: à trier au prompt {current_prompt}"

    return failures


_LEGACY_FAILURES = _load_legacy_failures()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark each test ``unit`` / ``integration`` and apply the inherited xfails.

    The marker is derived from the test path relative to ``tests/``:
    ``tests/integration/...`` gets ``integration``, everything else
    (``tests/unit/...`` and ``tests/support/...``) gets ``unit``. Node
    ids listed in ``tests/legacy_failures.txt`` additionally get a
    transient ``xfail(strict=False)``, removed as the inherited failures
    are triaged (parts D, U, F of the campaign plan).

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

        reason = _LEGACY_FAILURES.get(item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
