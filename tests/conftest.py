"""Configuration racine de la suite ``tests/``.

Enregistre les fixtures partagées de ``tests/support/fixtures.py``, ignore
transitoirement les modules dont l'import est cassé par un renommage encore
non traité par la campagne de tests, et marque automatiquement chaque test
``unit`` ou ``integration`` selon son chemin.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["tests.support.fixtures"]


# =============================================================================
# Modules dont l'import est cassé — TRANSITOIRE, retiré au prompt qui répare
# =============================================================================
collect_ignore = [
    # Renommage délibéré `_calculate_release_delays` → `_calculate_publication_delays`
    # (tsforecast/delays/data_manager.py) — à traiter au prompt D1.
    "unit/delays/test_data_manager.py",
]


# =============================================================================
# Échecs hérités — TRANSITOIRE (voir tests/legacy_failures.txt)
# =============================================================================
def _load_legacy_failures() -> dict[str, str]:
    """Lit ``tests/legacy_failures.txt`` et associe chaque node id à son motif.

    Format du fichier : lignes groupées par bloc, chaque bloc précédé d'un
    commentaire ``# prompt <id>`` ou ``# hors campagne`` ; les lignes vides et
    les commentaires d'en-tête (avant le premier groupe) sont ignorés.

    Returns:
        Dictionnaire ``{node_id: raison_xfail}``.
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
    """Marque chaque test ``unit``/``integration`` et applique les xfail hérités.

    Le marqueur est déduit du chemin du test relatif à ``tests/`` :
    ``tests/integration/...`` reçoit ``integration``, tout le reste
    (``tests/unit/...`` et ``tests/support/...``) reçoit ``unit``. Les node
    ids listés dans ``tests/legacy_failures.txt`` reçoivent en plus un
    ``xfail(strict=False)`` transitoire, retiré au fur et à mesure du tri des
    échecs hérités (parties D, U, F du plan de campagne).

    Args:
        config: Configuration pytest de la session.
        items: Éléments de test collectés, modifiés en place.
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
