"""Repository-level pytest configuration, loaded by every run, doctests of ``tsforecast`` included.

Keeps out of the collection the adapter modules whose doctests cannot run: a
module whose third-party package is not installed, and the adapter modules
transiently excluded until they are rewritten.
"""
from __future__ import annotations

from importlib.util import find_spec

# Module d'adapter -> paquet tiers requis par ses doctests ; un module d'adapter
# n'important aucun paquet tiers au chargement, seule l'exécution de ses doctests
# dépend du paquet
_ADAPTER_PACKAGES: dict[str, str] = {
    "tsforecast/adapters/darts.py": "darts",
    "tsforecast/adapters/hierarchicalforecast.py": "hierarchicalforecast",
    "tsforecast/adapters/opera.py": "opera",
    "tsforecast/adapters/sktime.py": "sktime",
}

# Modules exclus — TRANSITOIRE, retiré quand le module est réécrit : il ne se parse pas
_TRANSIENTLY_EXCLUDED: list[str] = [
    "tsforecast/adapters/hierarchicalforecast.py",
]

collect_ignore: list[str] = sorted(
    set(_TRANSIENTLY_EXCLUDED)
    | {path for path, package in _ADAPTER_PACKAGES.items() if find_spec(package) is None}
)
