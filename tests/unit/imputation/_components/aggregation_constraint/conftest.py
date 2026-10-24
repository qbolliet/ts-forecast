"""Fixtures of the ``AggregationConstraint`` test package."""

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Outils de test partagés du paquet
from tests.unit.frequency.aggregation_constraint.helpers import _RAW_2021


# =============================================================================
# Protocole sklearn
# =============================================================================
@pytest.fixture
def two_column_stage(reference_timeseries):
    """Stage frame of a1 / a2 on the 2021 months and their annual observations."""
    grid = reference_timeseries.loc['2021'].index
    X = pd.DataFrame(
        {'a1': _RAW_2021, 'a2': [value / 2 for value in _RAW_2021]}, index=grid
    )
    observations = reference_timeseries.loc['2021', ['a1', 'a2']]
    return X, observations
