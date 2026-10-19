"""Execution of the ``HighFrequencyImputer`` step-by-step notebook (notebook 5).

Runs ``notebooks/5 - QB - HighFrequencyImputer pas a pas.ipynb`` from top to
bottom with ``nbclient``, in memory (nothing is written back to disk), with the
repository root as the kernel's working directory - the notebook imports
``tsforecast`` and ``tests.support.datasets.build_panel_reference`` from there
- and the non-interactive ``Agg`` matplotlib backend. The notebook is a
user-facing walkthrough of the imputer: this test guards it against drifting
away from the public API.

Marked ``slow`` (a kernel start plus about forty cells).
"""
# Modules de base
from pathlib import Path

import pytest

# Racine du dépôt : répertoire de travail du noyau
REPO_ROOT = Path(__file__).resolve().parents[3]

# Notebook pas à pas de HighFrequencyImputer
NOTEBOOK = REPO_ROOT / 'notebooks' / '5 - QB - HighFrequencyImputer pas a pas.ipynb'


@pytest.mark.slow
class TestStepByStepNotebook:
    """Notebook 5 executes end to end without error."""

    def test_notebook_runs_without_error(self, monkeypatch):
        """Every cell executes and none produces an ``error`` output.

        The notebook deliberately reads private methods (it audits the
        internals): a deliberate change of their signature breaks it.
        """
        nbformat = pytest.importorskip('nbformat')
        nbclient = pytest.importorskip('nbclient')

        # Backend non interactif, hérité par le noyau via l'environnement
        monkeypatch.setenv('MPLBACKEND', 'Agg')
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        client = nbclient.NotebookClient(
            notebook,
            timeout=600,
            kernel_name='python3',
            resources={'metadata': {'path': str(REPO_ROOT)}},
        )

        # Une cellule en erreur lève CellExecutionError, qui fait échouer le test
        client.execute()

        errors = [
            index for index, cell in enumerate(notebook.cells)
            if cell.cell_type == 'code'
            and any(output.get('output_type') == 'error' for output in cell.get('outputs', []))
        ]
        assert errors == []
