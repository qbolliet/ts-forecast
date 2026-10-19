"""Shared builders of the ``compare_and_detect_delays`` tests (not a test module).

Month, series, panel and record constructors and the output column vocabulary used by
several test files of the package.
"""

import pandas as pd


TS = pd.Timestamp

# Colonnes de sortie, dans l'ordre (`has_changes` marque les observations détectées).
OUTPUT_COLUMNS = [
    'observation_date', 'has_changes', 'download_date', 'frequency',
    'period_start', 'period_end', 'reference_point', 'delay', 'unit',
]


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _months(periods: int = 4, start: str = '2023-01-01') -> pd.DatetimeIndex:
    """Build a month-start ``DatetimeIndex`` (``MS``) of ``periods`` dates."""
    return pd.date_range(start, periods=periods, freq='MS')


def _series(values, column: str = 'PIB', start: str = '2023-01-01') -> pd.DataFrame:
    """Build a one-column monthly frame from a list of values (``NaN`` allowed)."""
    return pd.DataFrame({column: values}, index=_months(len(values), start))


def _panel(entities: dict, entity_name: str = 'country') -> pd.DataFrame:
    """Build a monthly panel from per-entity values.

    Args:
        entities: ``{entity: {column: values}}``, every entity starting on
            ``2023-01-01`` (``MS``) with as many months as values.
        entity_name: Name of the entity level.

    Returns:
        A sorted two-level ``MultiIndex`` (entity, ``date``) frame.
    """
    frames = []
    for entity, cols in entities.items():
        n_rows = len(next(iter(cols.values())))
        frame = pd.DataFrame(cols, index=_months(n_rows))
        frame.index = pd.MultiIndex.from_product([[entity], frame.index], names=[entity_name, 'date'])
        frames.append(frame)
    return pd.concat(frames).sort_index()


def _records(result: pd.DataFrame, *fields: str) -> list:
    """Flatten a result into sorted ``(*index_key, *fields)`` tuples for exact comparison."""
    records = []
    for key, row in zip(result.index, result[list(fields)].itertuples(index=False)):
        key = key if isinstance(key, tuple) else (key,)
        records.append((*key, *row))
    return sorted(records, key=repr)


def _expected(*rows: tuple) -> list:
    """Sort hand-written expected rows the same way as :func:`_records`."""
    return sorted(rows, key=repr)
