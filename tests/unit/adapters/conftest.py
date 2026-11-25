"""Shared fixtures of the ``tsforecast.adapters`` unit tests.

The origin-indexed datasets of the adapter contract (``adapter_series``,
``adapter_panel``, ``adapter_unbalanced_panel``) and ``block_package``, which makes
the import of a package and of its submodules fail for one test.
"""
import sys
from typing import Callable

import pytest

from tests.support.datasets import build_adapter_panel, build_adapter_series


@pytest.fixture
def adapter_series():
    """``(X, y)``: 36 month starts from 2020-01, columns ``x1``, ``x2``, ``y = 2·x1 − x2 + noise``."""
    return build_adapter_series()


@pytest.fixture
def adapter_panel():
    """``(X, y)``: entities ``A``, ``B``, ``C`` over 36 month starts, same relation."""
    return build_adapter_panel(balanced=True)


@pytest.fixture
def adapter_unbalanced_panel():
    """``(X, y)``: ``B`` starts six months and ``C`` three months after ``A``, common end."""
    return build_adapter_panel(balanced=False)


@pytest.fixture
def block_package(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Return a function making the import of a package (and its submodules) fail during the test."""

    def block(package: str) -> None:
        # Entrée None dans sys.modules : l'import lève ModuleNotFoundError
        for name in [name for name in sys.modules if name.startswith(package + '.')]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, package, None)

    return block
