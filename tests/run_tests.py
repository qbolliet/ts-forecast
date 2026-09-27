"""Lanceur de tests pour l'ensemble du package ``tsforecast``.

Ce script fournit une manière pratique de lancer la suite complète avec
différentes configurations : tous les tests, seulement les tests unitaires
ou d'intégration, en excluant les tests lents, ou avec un rapport de
couverture.
"""
# Importation des modules
import argparse
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


def run_with_coverage() -> int:
    """Run the full suite with a branch coverage report.

    Produces terminal (``term-missing``), HTML (``htmlcov/``) and XML
    (``coverage.xml``) reports.

    Returns:
        Pytest exit code (0 = success).
    """
    return pytest.main([
        "tests/",
        "--cov=tsforecast",
        "--cov-branch",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "-v",
        "--tb=short",
    ])


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
