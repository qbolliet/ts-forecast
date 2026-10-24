"""Storage of the ``HighFrequencyImputer`` golden master.

Locations (:data:`GOLDEN_DIR`, :data:`MANIFEST_NAME`, :data:`CASES_DIRNAME`),
the environment the golden files depend on (:func:`current_environment`,
:func:`environment_mismatches`), and the reading and writing of the case
files (``pd.to_pickle``, then gzip without timestamp nor file name, so that
two generations of identical records give identical bytes) and of the JSON
manifest.
"""
# Modules de base
import gzip
import io
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Dossier du golden master : manifest et fichiers de cas
GOLDEN_DIR = Path(__file__).resolve().parent
MANIFEST_NAME = 'manifest.json'
CASES_DIRNAME = 'cases'
CASE_SUFFIX = '.pkl.gz'


# Distributions dont dépendent les valeurs figées (scipy : moindres carrés de
# LinearRegression)
PINNED_DISTRIBUTIONS = ('pandas', 'numpy', 'scikit-learn', 'scipy')


def current_environment() -> Dict[str, str]:
    """Describe the environment the golden values depend on.

    Returns:
        ``{'python': 'X.Y', 'platform': '<system>-<machine>', '<distribution>':
        '<version>', ...}`` for every distribution of
        :data:`PINNED_DISTRIBUTIONS`.

    Examples:
        >>> sorted(current_environment())
        ['numpy', 'pandas', 'platform', 'python', 'scikit-learn', 'scipy']
    """
    environment = {
        'python': f'{sys.version_info.major}.{sys.version_info.minor}',
        'platform': f'{platform.system()}-{platform.machine()}',
    }
    environment.update({name: version(name) for name in PINNED_DISTRIBUTIONS})
    return environment


def environment_mismatches(recorded: Dict[str, str]) -> List[str]:
    """List the differences between a recorded environment and the current one.

    Floating-point results of a linear regression may differ in their last
    bits across versions of numpy / scipy or across platforms (BLAS kernels):
    a bit-exact comparison is only meaningful in the environment of the
    generation.

    Args:
        recorded: Environment read from the manifest.

    Returns:
        One ``'<key>: <recorded> != <current>'`` entry per difference, empty
        when the environments match.

    Examples:
        >>> environment_mismatches(current_environment())
        []
    """
    current = current_environment()
    return [
        f'{key}: {recorded.get(key)} != {value}'
        for key, value in current.items()
        if recorded.get(key) != value
    ]


def case_path(golden_dir: Path, case_id: str) -> Path:
    """Return the path of the golden file of a case.

    Args:
        golden_dir: Root of a golden master (manifest and ``cases/``).
        case_id: Identifier of the case.

    Returns:
        ``<golden_dir>/cases/<case_id>.pkl.gz``.
    """
    return golden_dir / CASES_DIRNAME / f'{case_id}{CASE_SUFFIX}'


def write_record(path: Path, record: Dict[str, Any]) -> None:
    """Write a canonical record, refusing any object of the package.

    Args:
        path: Destination file (``.pkl.gz``).
        record: Canonical record of a case.

    Raises:
        TypeError: If the pickled record references a ``tsforecast`` class:
            it would no longer load once the package is reorganized.
    """
    buffer = io.BytesIO()
    pd.to_pickle(record, buffer, compression=None)
    payload = buffer.getvalue()
    # Aucune référence à un module du paquet dans le flux sérialisé
    if b'tsforecast' in payload:
        raise TypeError(f'{path.name}: the record references a tsforecast object')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compression reproductible : en-tête gzip sans horodatage ni nom de fichier
    path.write_bytes(gzip.compress(payload, mtime=0))


def read_record(path: Path) -> Dict[str, Any]:
    """Read the canonical record of a case.

    Args:
        path: Golden file (``.pkl.gz``).

    Returns:
        The record as written by :func:`write_record`.
    """
    return pd.read_pickle(path, compression='gzip')


def write_manifest(golden_dir: Path, manifest: Dict[str, Any]) -> None:
    """Write the manifest of a golden master, keys in a stable order.

    Args:
        golden_dir: Root of the golden master.
        manifest: JSON-serializable description of the generation.
    """
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    (golden_dir / MANIFEST_NAME).write_text(text + '\n', encoding='utf-8')


def read_manifest(golden_dir: Path = GOLDEN_DIR) -> Dict[str, Any]:
    """Read the manifest of a golden master.

    Args:
        golden_dir: Root of the golden master.

    Returns:
        The manifest written by the generation.

    Raises:
        FileNotFoundError: If the golden master was never generated there.
    """
    return json.loads((golden_dir / MANIFEST_NAME).read_text(encoding='utf-8'))
