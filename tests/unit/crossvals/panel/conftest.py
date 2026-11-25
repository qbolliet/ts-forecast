"""Fixtures and fold readers shared by the panel splitter tests.

Provides the splitter classes as parametrize-ready lists, a three-entity
balanced panel of month-ends, and ``labelled_folds``, which turns folds into
the ``(entity, date)`` labels of their rows so that folds computed on
differently ordered copies of a panel can be compared.
"""
# Importation des modules
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from tsforecast.crossvals.panel import (
    PanelInSampleSplit,
    PanelInSampleSplitPerEntity,
    PanelOutOfSampleSplit,
    PanelOutOfSampleSplitPerEntity,
)

# Les quatre splitters de panel, avec un identifiant lisible pour parametrize
ALL_SPLITTERS = [
    pytest.param(PanelOutOfSampleSplit, id='global-out'),
    pytest.param(PanelInSampleSplit, id='global-in'),
    pytest.param(PanelOutOfSampleSplitPerEntity, id='per-entity-out'),
    pytest.param(PanelInSampleSplitPerEntity, id='per-entity-in'),
]

# Configuration commune aux quatre splitters sur le panel décalé : la date d'août
# est portée par les trois entités, et chacune a de l'historique avant elle
AUGUST_TEST = {'test_indices': ['2024-08-31'], 'test_size': 2}


def labelled_folds(X: pd.DataFrame, folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[tuple[list, list]]:
    """Return each fold as the sorted index labels of its training and test rows.

    Args:
        X: Panel the folds index into.
        folds: ``(train_positions, test_positions)`` pairs.

    Returns:
        One ``(train_labels, test_labels)`` pair per fold, labels sorted by
        their representation (entity keys of mixed types cannot be compared).
    """
    # Étiquettes (entité, date) des lignes : indépendantes de l'ordre des lignes de X
    return [(sorted(X.index[train], key=repr), sorted(X.index[test], key=repr)) for train, test in folds]


def entities_of(X: pd.DataFrame, positions: Any) -> list:
    """Return the distinct entities (all index levels but the last) of the given rows, in order."""
    return list(dict.fromkeys(X.index.droplevel(-1)[np.asarray(positions, dtype=int)]))


@pytest.fixture
def panel_3x6() -> pd.DataFrame:
    """Balanced panel of three entities on the six month-ends of January to June 2024.

    ``MultiIndex`` (``entity``, ``date``): ``A`` on positions 0-5, ``B`` on
    6-11, ``C`` on 12-17; value equal to the row position.
    """
    dates = pd.date_range('2024-01-31', periods=6, freq='ME')
    index = pd.MultiIndex.from_product([['A', 'B', 'C'], dates], names=['entity', 'date'])
    return pd.DataFrame({'v': np.arange(18.0)}, index=index)
