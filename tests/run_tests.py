"""Lanceur de tests pour l'ensemble du package ``tsforecast``.

Ce script fournit une manière pratique de lancer la suite complète avec
différentes configurations : tous les tests, seulement les tests unitaires
ou d'intégration, en excluant les tests lents, ou avec un rapport de
couverture.
"""
# Importation des modules
import argparse
import json
import os
import sys

import pytest

# Ajout de la racine du projet au path Python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_all_tests() -> int:
    """Run the full test suite (unit + integration + support).

    Returns:
        Pytest exit code (0 = success).
    """
    return pytest.main([
        "tests/",
        "-v",
        "--tb=short",
    ])


def run_unit_tests() -> int:
    """Run unit tests only (``tests/unit/`` and ``tests/support/``).

    Returns:
        Pytest exit code (0 = success).
    """
    return pytest.main([
        "tests/",
        "-v",
        "-m", "unit",
        "--tb=short",
    ])


def run_integration_tests() -> int:
    """Run integration tests only (``tests/integration/``).

    Returns:
        Pytest exit code (0 = success).
    """
    return pytest.main([
        "tests/",
        "-v",
        "-m", "integration",
        "--tb=short",
    ])


def run_fast_tests() -> int:
    """Run all tests except those marked ``slow``.

    Returns:
        Pytest exit code (0 = success).
    """
    return pytest.main([
        "tests/",
        "-v",
        "-m", "not slow",
        "--tb=short",
    ])


# Paquets soumis au seuil de couverture, et seuil par module (lignes + branches)
COVERED_PACKAGES = ("utils", "delays", "frequency", "base", "crossvals", "panel", "xy")
COVERAGE_THRESHOLD = 90.0


def find_uncovered_modules(
    coverage_json: str, threshold: float = COVERAGE_THRESHOLD
) -> list[tuple[str, float]]:
    """List the modules of the monitored packages below the coverage threshold.

    Args:
        coverage_json: Path to a ``coverage.json`` report produced with branch coverage.
        threshold: Minimum percentage (lines and branches) required per module.

    Returns:
        ``(module path, percentage)`` pairs below the threshold, sorted by path.
    """
    with open(coverage_json, encoding="utf-8") as handle:
        files = json.load(handle)["files"]
    prefixes = tuple(f"tsforecast/{package}/" for package in COVERED_PACKAGES)
    failing = []
    for path, data in files.items():
        normalized = path.replace("\\", "/")
        if not normalized.startswith(prefixes):
            continue
        # Les fichiers sans instruction exécutable (ex. __init__ vides) sont ignorés
        if data["summary"]["num_statements"] == 0:
            continue
        percent = data["summary"]["percent_covered"]
        if percent < threshold:
            failing.append((normalized, percent))
    return sorted(failing)


def run_with_coverage() -> int:
    """Run the full suite with a branch coverage report and enforce the threshold.

    Produces terminal (``term-missing``), HTML (``htmlcov/``), XML
    (``coverage.xml``) and JSON (``coverage.json``) reports, then fails if a
    module of the monitored packages is below ``COVERAGE_THRESHOLD`` percent.

    Returns:
        0 if tests pass and every monitored module reaches the threshold,
        the pytest exit code or 1 otherwise.
    """
    exit_code = pytest.main([
        "tests/",
        "--cov=tsforecast",
        "--cov-branch",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-report=json",
        "-v",
        "--tb=short",
    ])
    if exit_code != 0:
        return int(exit_code)

    failing = find_uncovered_modules(os.path.join(PROJECT_ROOT, "coverage.json"))
    if failing:
        print(f"\nModules under {COVERAGE_THRESHOLD:.0f}% coverage:")
        for path, percent in failing:
            print(f"  {percent:6.2f}%  {path}")
        return 1
    print(f"\nAll monitored modules reach {COVERAGE_THRESHOLD:.0f}% coverage.")
    return 0


def run_path(path: str) -> int:
    """Run tests under an arbitrary file or directory target.

    Args:
        path: Path to a test file, a test class/function node id, or a
            directory, relative to the project root (e.g.
            ``tests/unit/crossvals/test_time_series.py``).

    Returns:
        Pytest exit code (0 = success).
    """
    return pytest.main([
        path,
        "-v",
        "--tb=short",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lance la suite de tests de tsforecast")
    parser.add_argument(
        "--mode",
        choices=["all", "unit", "integration", "fast", "coverage"],
        default="fast",
        help="Mode d'exécution des tests",
    )
    parser.add_argument(
        "--path",
        help="Cible libre (fichier, dossier ou node id) à passer directement à pytest",
    )

    args = parser.parse_args()

    if args.path:
        exit_code = run_path(args.path)
    elif args.mode == "all":
        exit_code = run_all_tests()
    elif args.mode == "unit":
        exit_code = run_unit_tests()
    elif args.mode == "integration":
        exit_code = run_integration_tests()
    elif args.mode == "fast":
        exit_code = run_fast_tests()
    elif args.mode == "coverage":
        exit_code = run_with_coverage()

    sys.exit(exit_code)
