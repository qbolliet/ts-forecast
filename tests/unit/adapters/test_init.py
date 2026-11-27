"""Tests of the ``tsforecast.adapters`` package import and exports.

The exported adapters, conversion functions and foundation-model registry
functions; in a fresh interpreter, importing
the package loads no third-party forecasting package, and once every optional
package is made unimportable each adapter can still be built while fitting raises
an ``ImportError`` naming its extra.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import tsforecast.adapters

# Racine du dépôt, pour importer tsforecast depuis un interpréteur neuf
REPO_ROOT = Path(__file__).resolve().parents[3]
# Paquets tiers optionnels des adapters
OPTIONAL_PACKAGES = (
    'sktime', 'darts', 'hierarchicalforecast', 'utilsforecast', 'chronos', 'timestral', 't0', 'torch',
    'transformers',
)

# Chercheur d'imports rendant les paquets optionnels introuvables, comme s'ils n'étaient pas installés
BLOCKER = textwrap.dedent(f'''
    import sys
    from importlib.abc import MetaPathFinder

    BLOCKED = {OPTIONAL_PACKAGES!r}

    class _OptionalPackageBlocker(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in BLOCKED:
                raise ModuleNotFoundError(f'No module named {{name!r}}', name=name)
            return None

    sys.meta_path.insert(0, _OptionalPackageBlocker())
''')


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter from the repository root."""
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_exported_names():
    assert sorted(tsforecast.adapters.__all__) == [
        'CallableBackend',
        'DartsAdapter',
        'FoundationModelAdapter',
        'HierarchicalForecastAdapter',
        'SktimeAdapter',
        'clear_model_cache',
        'period_splits',
        'register_backend',
        'stack_nodes',
        'unstack_nodes',
    ]
    for name in tsforecast.adapters.__all__:
        assert getattr(tsforecast.adapters, name).__name__ == name


@pytest.mark.slow
def test_import_without_loading_or_requiring_optional_packages():
    """One fresh interpreter (about two seconds): no package loaded, then none available."""
    code = textwrap.dedent(f'''
        import sys
        import pandas as pd
        import tsforecast.adapters as adapters

        # Aucun paquet tiers chargé par l'import du paquet, même installé
        loaded = sorted({{name.split('.')[0] for name in sys.modules}} & set({OPTIONAL_PACKAGES!r}))
        assert not loaded, loaded
    ''') + BLOCKER + textwrap.dedent('''
        # Paquets optionnels désormais introuvables : construction, puis fit refusé avec l'extra
        adapters.SktimeAdapter(forecaster=None)
        darts = adapters.DartsAdapter(model=None)
        hierarchical = adapters.HierarchicalForecastAdapter({'a': ['b']})
        chronos = adapters.FoundationModelAdapter('amazon/chronos-2')
        timestral = adapters.FoundationModelAdapter('mistralai/Timestral-5M')
        t0 = adapters.FoundationModelAdapter('theforecastingcompany/t0-gamma')

        dates = pd.date_range('2024-01-01', periods=3, freq='MS')
        X = pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=dates)
        y = pd.Series([1.0, 2.0, 3.0], index=dates)
        # Conversion large -> long : pandas seul
        assert adapters.stack_nodes(X).index.names == ['node', None]
        for estimator, extra in [
            (darts, 'adapters-darts'),
            (hierarchical, 'adapters-hierarchical'),
            (chronos, 'adapters-chronos'),
            (timestral, 'adapters-timestral'),
            (t0, 'adapters-t0'),
        ]:
            try:
                estimator.fit(X, y)
            except ImportError as error:
                assert extra in str(error), str(error)
            else:
                raise AssertionError(f'{type(estimator).__name__}.fit succeeded without its package')
    ''')
    result = run_python(code)
    assert result.returncode == 0, result.stderr
