"""Shared builders of the ``FrequencyDetector`` tests (not a test module)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

# Formats de sortie acceptés par toutes les fonctions de détection
RETURN_FORMATS = ('base', 'with_position', 'full', 'components')


def make_series(index: pd.Index) -> pd.Series:
    """Build a float series of consecutive values on ``index``.

    Args:
        index: Index of the series (any type).

    Returns:
        ``pd.Series`` holding ``0.0, 1.0, ...`` on ``index``.
    """
    return pd.Series(np.arange(len(index), dtype=float), index=index)


def make_panel_series(entities: dict) -> pd.Series:
    """Build a panel series with an ``(entity, date)`` ``MultiIndex``.

    Args:
        entities: Mapping ``{entity: dates}``, each entity keeping its own dates.

    Returns:
        ``pd.Series`` of consecutive floats, index levels named ``entity`` and ``date``.
    """
    tuples = [(entity, pd.Timestamp(date)) for entity, dates in entities.items() for date in dates]
    index = pd.MultiIndex.from_tuples(tuples, names=['entity', 'date'])
    return make_series(index)


def is_on_grid(index: pd.DatetimeIndex, frequency: str) -> bool:
    """Tell whether every date of ``index`` lies on the grid of ``frequency``.

    Args:
        index: Dates to check.
        frequency: Pandas offset alias (e.g. ``'W-MON'``, ``'QS'``).

    Returns:
        ``True`` if ``to_offset(frequency).is_on_offset`` holds for every date.
    """
    # Alias nus dépréciés ('Q', 'Y') : l'avertissement pandas n'est pas le sujet
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        offset = to_offset(frequency)
    return all(offset.is_on_offset(date) for date in index)

