"""Shared fixtures of the ``XYPipeline`` tests.

Regression, classification and pandas datasets drawn from a fixed seed, a
temporary cache directory for the ``memory`` parameter, and small dated frames
with golden values (``frame_xy`` with one missing target, ``linear_xy`` with an
exactly linear target).
"""

import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_regression_data():
    """Données de régression pour les tests."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    return X, y


@pytest.fixture
def sample_classification_data():
    """Données de classification pour les tests."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_dataframe_data():
    """Données pandas DataFrame pour les tests."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 5),
        columns=["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"],
    )
    y = pd.Series(np.random.randn(100), name="target")
    return X, y


@pytest.fixture
def temp_cache_dir():
    """Répertoire temporaire pour le caching."""
    cache_dir = tempfile.mkdtemp()
    yield cache_dir
    shutil.rmtree(cache_dir)


# ==============================================================================
# Petits jeux à valeurs d'or
# ==============================================================================


def month_ends(periods: int = 6) -> pd.DatetimeIndex:
    """Month ends from 2024-01-31, named ``date``."""
    return pd.date_range('2024-01-31', periods=periods, freq='ME', name='date')


@pytest.fixture
def frame_xy():
    """``(X, y)``: 6 month ends, ``a`` = 1..6, ``b`` = 6..1, ``y`` = a with NaN on the 2nd row.

    ``y`` named ``t``: 1, NaN, 3, 4, 5, 6. The 5 observed targets have mean 3.8.
    """
    X = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]}, index=month_ends())
    y = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0], index=X.index, name='t')
    return X, y


@pytest.fixture
def linear_xy():
    """``(X, y)``: 6 month ends, ``a`` = 1..6, ``b`` = (-1)^i, ``y = 2a + 3`` exactly."""
    X = pd.DataFrame(
        {'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]}, index=month_ends()
    )
    y = pd.Series(2.0 * X['a'] + 3.0, name='t')
    return X, y
