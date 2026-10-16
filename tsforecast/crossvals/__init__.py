"""Time-aware cross-validators for time series and panel data.

Re-exports the base splitters ``OutOfSampleSplit`` and ``InSampleSplit``, the
time series splitters ``TSOutOfSampleSplit`` and ``TSInSampleSplit``, and the
panel splitters ``PanelOutOfSampleSplit``, ``PanelInSampleSplit``,
``PanelOutOfSampleSplitPerEntity`` and ``PanelInSampleSplitPerEntity``. All
follow the sklearn splitter contract (``split``, ``get_n_splits``).

Examples:
    >>> import numpy as np
    >>> from tsforecast.crossvals import TSOutOfSampleSplit
    >>> splitter = TSOutOfSampleSplit(n_splits=2, test_size=3, gap=1)
    >>> [(train.tolist(), test.tolist()) for train, test in splitter.split(np.arange(10))]
    [([0, 1, 2], [4, 5, 6]), ([0, 1, 2, 3, 4, 5], [7, 8, 9])]
"""

# Importation des classes de base
from .base import OutOfSampleSplit, InSampleSplit

# Importation des classes pour les séries temporelles
from .time_series import TSOutOfSampleSplit, TSInSampleSplit

# Import des classes pour les données de panel
from .panel import PanelOutOfSampleSplit, PanelInSampleSplit, PanelOutOfSampleSplitPerEntity, PanelInSampleSplitPerEntity

__all__ = [
    # Base
    "OutOfSampleSplit",
    "InSampleSplit",
    # Séries temporelles
    "TSOutOfSampleSplit", 
    "TSInSampleSplit",
    # Panel - global
    "PanelOutOfSampleSplit",
    "PanelInSampleSplit",
    # Panel - Par entité
    "PanelOutOfSampleSplitPerEntity",
    "PanelInSampleSplitPerEntity"
]