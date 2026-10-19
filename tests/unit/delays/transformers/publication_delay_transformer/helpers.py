"""Shared builders of the ``PublicationDelayTransformer`` tests (not a test module).

Monthly and daily series, fit wrapper and delays-frame constructor used by several test
files of the package.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd

# Classe à tester et producteurs réels de tableaux de délais
from tsforecast.delays.transformers import PublicationDelayTransformer


TS = pd.Timestamp

# Date de prédiction des valeurs d'or : 14 jours écoulés depuis le début de décembre 2023
PREDICTION = '2023-12-15'


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _monthly(columns=('GDP', 'CPI'), periods: int = 12) -> pd.DataFrame:
    """Build a monthly (``MS``) frame of 2023 whose column ``i`` holds ``(i + 1) * [0, 1, ...]``.

    Args:
        columns: Column names.
        periods: Number of months from January 2023.

    Returns:
        Deterministic float frame indexed by month starts.
    """
    index = pd.date_range('2023-01-01', periods=periods, freq='MS')
    return pd.DataFrame({col: np.arange(periods, dtype=float) * (i + 1) for i, col in enumerate(columns)}, index=index)


def _fit(X: pd.DataFrame, **kwargs) -> PublicationDelayTransformer:
    """Fit a transformer at the gold prediction date, the warnings of ``fit`` being silenced.

    Args:
        X: Data to fit on.
        **kwargs: Constructor arguments (``prediction_date`` defaults to 2023-12-15).

    Returns:
        The fitted transformer.
    """
    kwargs.setdefault('prediction_date', PREDICTION)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PublicationDelayTransformer(**kwargs).fit(X)


def _delays_frame(columns=('GDP', 'CPI'), delays=(45.0, 20.0), unit='D', reference_point='start',
                  frequency='Q') -> pd.DataFrame:
    """Build a flat delays table with the columns expected by the transformer.

    Args:
        columns: Variable names (column ``column``).
        delays: Delay of each variable.
        unit: Unit of every delay.
        reference_point: Reference point of every delay.
        frequency: Target frequency of every delay.

    Returns:
        Table with the columns ``column``, ``delay``, ``unit``, ``reference_point``, ``frequency``.
    """
    n = len(columns)
    return pd.DataFrame({'column': list(columns), 'delay': list(delays), 'unit': [unit] * n,
                         'reference_point': [reference_point] * n, 'frequency': [frequency] * n})
