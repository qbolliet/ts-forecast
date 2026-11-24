"""Shared builders of the temporal-validation tests (not a test module).

Month, panel and flat-frame constructors used by several test files of the package.
"""

import pandas as pd


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _months(periods: int = 4, start: str = '2023-01-01') -> pd.DatetimeIndex:
    """Build a month-start ``DatetimeIndex`` (``MS``) of ``periods`` dates."""
    return pd.date_range(start, periods=periods, freq='MS')


def _panel(entities=('A', 'B'), periods: int = 3, names=('entity', 'date')) -> pd.DataFrame:
    """Build a small sorted two-level panel with values ``0..n-1`` in column ``v``.

    Args:
        entities: Entity labels, in the order of the blocks.
        periods: Number of monthly dates per entity.
        names: Names of the two index levels.

    Returns:
        DataFrame indexed by ``(entity, date)``, sorted by entity then date.
    """
    index = pd.MultiIndex.from_product([list(entities), _months(periods)], names=list(names))
    return pd.DataFrame({'v': range(len(index))}, index=index)


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """Move the index levels of a frame into columns (column-based input)."""
    return df.reset_index()


def _validate_kwargs(df: pd.DataFrame, path: str) -> tuple:
    """Return ``(input frame, kwargs)`` to validate ``df`` on the requested path.

    Args:
        df: Frame with a datetime last index level, indexed by entity levels first.
        path: ``'index'`` (keep the index) or ``'columns'`` (index moved to columns).

    Returns:
        The frame to validate and the ``time_col`` / ``panel_cols`` keyword arguments.
    """
    if path == 'index':
        return df, {}
    names = list(df.index.names)
    kwargs = {'time_col': names[-1]}
    if len(names) > 1:
        kwargs['panel_cols'] = names[:-1]
    return df.reset_index(), kwargs
