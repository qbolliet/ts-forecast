"""Generation script of the ``HighFrequencyImputer`` golden master, run by hand.

``python tests/support/golden/generate.py`` runs every case of
:func:`tests.support.golden.scenarios.build_cases` and writes one gzip pickle
per case under ``cases/`` plus ``manifest.json`` (configurations, datasets,
environment). ``--output DIR`` writes elsewhere; ``--compare DIR_A DIR_B``
checks that two generations are identical (determinism). Never collected nor
run by pytest.
"""
# Modules de base
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Racine du dépôt sur le chemin d'import, pour une exécution comme script
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tests.support.golden.compare import assert_same  # noqa: E402
from tests.support.golden.scenarios import (  # noqa: E402
    BARE_ESTIMATOR,
    COMMON_PARAMS,
    CV_SEED,
    CV_SPLITS,
    DATASETS,
    NAN_TOLERANT_ESTIMATOR,
    build_cases,
    run_case,
)
from tests.support.golden.store import (  # noqa: E402
    CASE_SUFFIX,
    CASES_DIRNAME,
    GOLDEN_DIR,
    case_path,
    current_environment,
    read_manifest,
    read_record,
    write_manifest,
    write_record,
)


def build_manifest(case_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe a generation: environment, fixed settings and cases.

    Args:
        case_entries: One entry per case (id, dataset, target flag,
            estimator, parameters, file).

    Returns:
        JSON-serializable manifest.
    """
    return {
        'generator': 'tests/support/golden/generate.py',
        'environment': current_environment(),
        'common_params': COMMON_PARAMS,
        'estimators': {
            NAN_TOLERANT_ESTIMATOR: 'make_pipeline(SimpleImputer(), LinearRegression())',
            BARE_ESTIMATOR: 'LinearRegression()',
        },
        'cv': f'KFold(n_splits={CV_SPLITS}, shuffle=True, random_state={CV_SEED})',
        'datasets': {
            name: {'cutoff': dataset.cutoff, 'target': dataset.target}
            for name, dataset in DATASETS.items()
        },
        'cases': case_entries,
    }


def generate(output: Path, only: Optional[str] = None) -> None:
    """Run the cases and write the golden master into a directory.

    Golden files of cases that are no longer in the matrix are removed, so
    that the directory always mirrors the manifest. With ``only``, the run is
    a partial preview: neither the manifest nor the stale files are touched.

    Args:
        output: Root of the golden master to write.
        only: Optional substring; when given, only the cases whose id
            contains it are run.
    """
    cases = build_cases()
    if only is not None:
        cases = [case for case in cases if only in case.case_id]

    entries: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for position, case in enumerate(cases, start=1):
        case_start = time.perf_counter()
        path = case_path(output, case.case_id)
        write_record(path, run_case(case))
        entries.append({
            'id': case.case_id,
            'dataset': case.dataset,
            'with_target': case.with_target,
            'estimator': case.estimator,
            'params': case.imputer_params(),
            'file': path.relative_to(output).as_posix(),
        })
        print(
            f'[{position}/{len(cases)}] {case.case_id} '
            f'{time.perf_counter() - case_start:.2f} s, {path.stat().st_size / 1024:.0f} Ko'
        )
    if only is not None:
        return

    # Suppression des fichiers de cas sortis de la matrice
    produced = {case_path(output, case.case_id) for case in cases}
    for stale in (output / CASES_DIRNAME).glob(f'*{CASE_SUFFIX}'):
        if stale not in produced:
            stale.unlink()
    write_manifest(output, build_manifest(entries))

    total_size = sum(path.stat().st_size for path in produced)
    print(
        f'{len(cases)} cases in {time.perf_counter() - start:.1f} s, '
        f'{total_size / 1024 / 1024:.2f} Mo'
    )


def compare(first: Path, second: Path) -> None:
    """Check that two generated golden masters are identical.

    Args:
        first: Root of the first generation.
        second: Root of the second generation.

    Raises:
        AssertionError: On the first difference: case list, configuration,
            or any recorded object (see
            :func:`tests.support.golden.compare.assert_same`).
    """
    manifest_first, manifest_second = read_manifest(first), read_manifest(second)
    assert_same(manifest_first['cases'], manifest_second['cases'], 'manifest.cases')
    for entry in manifest_first['cases']:
        assert_same(
            read_record(first / entry['file']),
            read_record(second / entry['file']),
            entry['id'],
        )
        # Octets identiques : pickle et gzip sans horodatage sont reproductibles
        if (first / entry['file']).read_bytes() != (second / entry['file']).read_bytes():
            print(f'{entry["id"]}: same records, different bytes')
    print(f'{len(manifest_first["cases"])} cases identical')


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the command line and run the generation or the comparison.

    Args:
        argv: Arguments, ``sys.argv[1:]`` when None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=GOLDEN_DIR,
                        help='root of the golden master to write (default: this folder)')
    parser.add_argument('--only', help='run only the cases whose id contains this text')
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('DIR_A', 'DIR_B'),
                        help='compare two generations instead of generating')
    arguments = parser.parse_args(argv)
    if arguments.compare:
        compare(*arguments.compare)
    else:
        generate(arguments.output, arguments.only)


if __name__ == '__main__':
    main()
