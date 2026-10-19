"""Fixtures of the delay-detection test package."""

import pandas as pd
import pytest

# Outils de test partagés du paquet
from tests.unit.delays.data_manager.helpers import _panel, _series


@pytest.fixture
def monthly_series() -> pd.DataFrame:
    """Reference monthly series ``PIB`` = 1..4 on ``2023-01-01 .. 2023-04-01``."""
    return _series([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """Two-entity monthly panel: ``A`` observed Jan-Apr, ``B`` observed Jan-Mar."""
    return _panel({'A': {'PIB': [1.0, 2.0, 3.0, 4.0]}, 'B': {'PIB': [10.0, 20.0, 30.0]}})
