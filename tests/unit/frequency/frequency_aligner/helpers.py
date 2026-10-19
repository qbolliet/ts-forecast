"""Shared builders of the ``FrequencyAligner`` tests (not a test module).

The aligner fixture-free constructors of frames, panels and series used by several test
files of the package.
"""

# Modules de base
from typing import Dict, List

import numpy as np
import pandas as pd


def _frame(values: List[float], start: str, freq: str, col: str = 'x') -> pd.DataFrame:
    """Build a one-column float frame on a regular grid starting at ``start``."""
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({col: np.asarray(values, dtype=float)}, index=index)


def _panel(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack one frame per entity into an (entity, date) panel."""
    panel = pd.concat(frames)
    panel.index.names = ['entity', 'date']
    return panel


def _series(mapping: Dict, name: str = 'x') -> pd.Series:
    """Build a date-indexed float series of golden values from a {date: value} mapping."""
    return pd.Series(
        list(mapping.values()), index=pd.DatetimeIndex(list(mapping)), dtype=float, name=name
    )
