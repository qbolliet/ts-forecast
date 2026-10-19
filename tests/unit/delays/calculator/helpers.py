"""Shared builders of the ``calculate_applicable_delay`` tests (not a test module).

Row, frame and detection constructors, column vocabulary and the unit table used by
several test files of the package.
"""

# Modules de base
import warnings

import pandas as pd

# Fonction à tester
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays


TS = pd.Timestamp

# Colonnes requises en entrée (ordre du message d'erreur)
REQUIRED_COLUMNS = [
    'observation_date', 'download_date', 'frequency', 'period_start',
    'period_end', 'reference_point', 'delay', 'unit',
]

# Code pandas de chaque unité de délai produite par compare_and_detect_delays
_PANDAS_UNITS = {'day': 'D', 'second': 's', 'microsecond': 'us'}


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _row(observation, start, end, delay, *, reference_point='end', unit='day', frequency='monthly') -> dict:
    """Build one publication-delay row, in the format of ``compare_and_detect_delays``.

    Args:
        observation: Date of the observation.
        start: Start of its period.
        end: Exclusive end of its period (first instant of the next period).
        delay: Delay counted from ``reference_point``, in ``unit``.
        reference_point: Reference point of the delay (``'start'`` or ``'end'``).
        unit: Unit of the delay.
        frequency: Frequency of the (entity, indicator) couple, as a literal.

    Returns:
        A dictionary holding the required columns; the download date is rebuilt as
        ``reference date + delay`` (``NaT`` for a ``NaN`` delay).
    """
    reference = TS(end if reference_point == 'end' else start)
    download = reference + pd.Timedelta(delay, unit=_PANDAS_UNITS.get(unit, unit))
    return {
        'observation_date': TS(observation), 'download_date': download, 'frequency': frequency,
        'period_start': TS(start), 'period_end': TS(end), 'reference_point': reference_point,
        'delay': delay, 'unit': unit,
    }


def _monthly(month: str = '2023-12', delay=14, **kwargs) -> dict:
    """Build the row of a monthly observation made on the 15th of ``month`` (``'YYYY-MM'``)."""
    period = pd.Period(month, freq='M')
    return _row(period.start_time + pd.Timedelta(days=14), period.start_time, (period + 1).start_time, delay,
                frequency=kwargs.pop('frequency', 'monthly'), **kwargs)


def _frame(rows: list, index=None, names=('indicator',)) -> pd.DataFrame:
    """Build a publication-delays frame from rows.

    Args:
        rows: Dictionaries returned by :func:`_row` (an empty list gives an empty frame
            with the required columns).
        index: One key per row (a tuple for a ``MultiIndex``); defaults to ``'PIB'``.
        names: Names of the index levels, the last one being the indicator.

    Returns:
        The frame, indexed by ``index``.
    """
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS) if not rows else pd.DataFrame(rows)
    keys = list(index) if index is not None else ['PIB'] * len(rows)
    if len(names) == 1:
        frame.index = pd.Index(keys, name=names[0])
    else:
        frame.index = pd.MultiIndex.from_tuples(keys, names=names)
    return frame


def _detect(*args, **kwargs) -> pd.DataFrame:
    """Call ``compare_and_detect_delays`` silencing the index-replacement and undetectable-frequency warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compare_and_detect_delays(*args, **kwargs)


def _delay(frame: pd.DataFrame, reference_point: str, frequency, **kwargs) -> list:
    """Return the ``delay`` column of ``calculate_applicable_delay`` as a list."""
    return calculate_applicable_delay(frame, reference_point, frequency, **kwargs)['delay'].tolist()
