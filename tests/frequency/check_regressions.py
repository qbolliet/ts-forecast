"""Regression checker for tests/frequency/ against a frozen baseline.

Relance la suite pytest de ``tests/frequency/``, compare l'ensemble des échecs
au fichier ``BASELINE_FAILURES.txt`` et affiche les régressions (nouveaux
échecs) ainsi que les échecs résolus depuis la référence.
"""
import re
import subprocess
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).parent / "BASELINE_FAILURES.txt"
FREQUENCY_DIR = Path(__file__).parent

# Une ligne d'identifiant de test brut (pas une ligne de tableau markdown ou
# de puce qui, elle, cite un identifiant entre backticks au milieu de texte)
_TEST_ID_LINE = re.compile(r"^tests/frequency/\S+\.py::\S+$")


def load_baseline_failures(baseline_path: Path) -> set[str]:
    """Load the set of baseline-failing test ids from BASELINE_FAILURES.txt.

    Seules les lignes constituées uniquement d'un identifiant de test brut
    (``tests/frequency/fichier.py::classe::test``) sont prises en compte :
    le tableau markdown de triage et le résumé qui suivent citent aussi des
    identifiants (entre backticks, dans des lignes de tableau ou des puces)
    mais ne doivent pas être comptés comme des échecs de référence.

    Args:
        baseline_path: Path to BASELINE_FAILURES.txt.

    Returns:
        Set of test node ids (``fichier::classe::test``) that were failing
        at the time the baseline was captured.

    Raises:
        FileNotFoundError: If baseline_path does not exist.
    """
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline introuvable : {baseline_path}")

    failures: set[str] = set()
    for line in baseline_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if _TEST_ID_LINE.match(stripped):
            failures.add(stripped)
    return failures


def run_pytest_and_collect_failures(target_dir: Path) -> set[str]:
    """Run pytest on target_dir and collect the set of failing test ids.

    Args:
        target_dir: Directory passed to pytest.

    Returns:
        Set of failing test node ids, parsed from the ``FAILED`` lines of
        the short test summary.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(target_dir), "-q"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    failures: set[str] = set()
    for line in result.stdout.splitlines():
        if line.startswith("FAILED "):
            test_id = line[len("FAILED "):].split(" - ")[0].strip()
            failures.add(test_id)
    return failures


def main() -> int:
    """Compare current pytest failures to the frozen baseline.

    Returns:
        Exit code: 1 if at least one regression (new failure) is found,
        0 otherwise.
    """
    baseline_failures = load_baseline_failures(BASELINE_PATH)
    current_failures = run_pytest_and_collect_failures(FREQUENCY_DIR)

    new_failures = sorted(current_failures - baseline_failures)
    resolved_failures = sorted(baseline_failures - current_failures)

    print("\n" + "=" * 70)
    print("NOUVEAUX ÉCHECS (régressions)")
    print("=" * 70)
    if new_failures:
        for test_id in new_failures:
            print(f"  {test_id}")
    else:
        print("  Aucun")

    print("\n" + "=" * 70)
    print("ÉCHECS RÉSOLUS")
    print("=" * 70)
    if resolved_failures:
        for test_id in resolved_failures:
            print(f"  {test_id}")
    else:
        print("  Aucun")

    print()
    return 1 if new_failures else 0


if __name__ == "__main__":
    sys.exit(main())
