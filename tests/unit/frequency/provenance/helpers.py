"""Shared builders of the provenance tests (not a test module).

Tracker factory and the model-provenance scale used by several test files of the
package.
"""

import numpy as np
import pandas as pd

# Objet testé
from tsforecast.frequency.provenance import ImputationProvenanceTracker, ProvenanceType


P = ProvenanceType


# Fabrique de tracker : série mensuelle de quatre lignes, deux colonnes
def _tracker(values_a=(1.0, np.nan, 3.0, np.nan), values_b=(np.nan,) * 4):
    """Build an initialized tracker on a 4-row monthly frame.

    Args:
        values_a: Values of column ``a``.
        values_b: Values of column ``b``.

    Returns:
        Tuple ``(tracker, dates)``.
    """
    dates = pd.date_range('2023-01-01', periods=len(values_a), freq='MS')
    data = pd.DataFrame({'a': list(values_a), 'b': list(values_b)}, index=dates)
    return ImputationProvenanceTracker().initialize(data), dates
