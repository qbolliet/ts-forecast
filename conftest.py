"""Repository-level pytest configuration, loaded by every run, doctests of ``tsforecast`` included.

Keeps out of the collection the adapter modules whose doctests cannot run because
their third-party package is not installed.
"""
from __future__ import annotations

from importlib.util import find_spec

# Module d'adapter -> paquet tiers requis par ses doctests ; un module d'adapter
# n'important aucun paquet tiers au chargement, seule l'exécution de ses doctests
# dépend du paquet
_ADAPTER_PACKAGES: dict[str, str] = {
    "tsforecast/adapters/darts.py": "darts",
    "tsforecast/adapters/hierarchicalforecast/adapter.py": "hierarchicalforecast",
    "tsforecast/adapters/sktime.py": "sktime",
}

collect_ignore: list[str] = sorted(
    path for path, package in _ADAPTER_PACKAGES.items() if find_spec(package) is None
)
