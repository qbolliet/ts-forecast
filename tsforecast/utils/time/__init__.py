"""Time manipulation utilities for time series processing.

Re-exports the date helpers of ``utils.py`` (``resolve_date``, string
conversions, period bounds ``get_period_start`` / ``get_period_end`` /
``get_period_boundaries``) and ``as_exact_unit`` (``resolution.py``), which
changes the storage unit of dates without rounding.
"""
# Import des éléments du module
from .utils import (
    resolve_date,
    timeseries_to_string,
    string_to_timeseries,
    get_period_start,
    get_period_end,
    get_period_boundaries
)
from .resolution import as_exact_unit
# Réexport des éléments d'intérêt
__all__ = [
    'resolve_date',
    'timeseries_to_string',
    'string_to_timeseries',
    'get_period_start',
    'get_period_end',
    'get_period_boundaries',
    'as_exact_unit'
]
