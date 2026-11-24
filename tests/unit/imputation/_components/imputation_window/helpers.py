"""Shared builders of the ``ImputationWindowCalculator`` tests (not a test module).

Frame and panel constructors, quiet fit and mask-attribute vocabulary used by several
test files of the package.
"""

# Modules de base
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

# Objet testé
from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator


# Les trois masques exposés par le calculateur
_KINDS = ('strict', 'imputation', 'training')

# Attributs de masque correspondant à chaque kind
_MASK_ATTRIBUTES = (
    'imputation_strict_window_mask_',
    'imputation_window_mask_',
    'training_window_mask_',
)

# Attributs de bornes : des dicts indexés par tuple d'entité sur un panel
_BOUND_ATTRIBUTES = (
    'imputation_window_start_',
    'imputation_window_end_',
    'imputation_strict_window_start_',
    'imputation_strict_window_end_',
    'training_window_start_',
    'training_window_end_',
)


# =============================================================================
# Constructeurs de données et utilitaires
# =============================================================================
def _frame(columns: Dict[str, List[float]], start: str = '2020-01-01', freq: str = 'MS') -> pd.DataFrame:
    """Build a float frame on a regular grid from ``{column: values}``."""
    n_rows = len(next(iter(columns.values())))
    index = pd.date_range(start, periods=n_rows, freq=freq, name='date')
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in columns.items()}, index=index)


def _make_panel(entities: List[tuple], n_periods: int = 36) -> pd.DataFrame:
    """Build a fully observed two-column monthly panel indexed by the given entity tuples."""
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='MS')
    n_levels = len(entities[0])
    # Construction du MultiIndex (niveaux d'entité + date)
    tuples = [(*entity, date) for entity in entities for date in dates]
    names = [f'level_{i}' for i in range(n_levels)] + ['date']
    index = pd.MultiIndex.from_tuples(tuples, names=names)
    n_rows = len(index)
    return pd.DataFrame({
        'a': np.arange(n_rows, dtype=float),
        'b': np.arange(n_rows, dtype=float) * 2,
    }, index=index)


def _make_df_with_shoulders():
    """Build a monthly frame with a strict window flanked by two half-coverage shoulders.

    Returns:
        Tuple ``(dates, df)``: both columns observed on ``dates[10:15]``
        (strict window), ``a`` alone on ``dates[8:10]`` and ``dates[15:17]``
        (coverage 0.5, kept by a 0.5 threshold, dropped by 0.9).
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='MS')
    df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)
    # Fenêtre stricte : couverture totale sur [10, 15)
    df.loc[dates[10:15], ['a', 'b']] = 1.0
    # Épaules à couverture == 0.5, de part et d'autre de la fenêtre stricte
    df.loc[dates[8:10], 'a'] = 1.0
    df.loc[dates[15:17], 'a'] = 1.0
    return dates, df


def _fit_quietly(calc: ImputationWindowCalculator, data: pd.DataFrame) -> ImputationWindowCalculator:
    """Fit ``calc`` on ``data``, silencing the UserWarnings that are not the test subject."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return calc.fit(data)


def _entities_of(series: pd.Series) -> set:
    """Return the set of entity key tuples present in a MultiIndex mask Series."""
    entity_index = series.index.droplevel(-1)
    return {key if isinstance(key, tuple) else (key,) for key in entity_index}


def _warning_messages(records) -> List[str]:
    """Extract the UserWarning messages of a ``warnings.catch_warnings(record=True)`` list."""
    return [str(r.message) for r in records if issubclass(r.category, UserWarning)]
