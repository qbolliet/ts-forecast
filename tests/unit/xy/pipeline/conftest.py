"""Shared fixtures of the ``XYPipeline`` tests.

Regression, classification and pandas datasets drawn from a fixed seed,
and a temporary cache directory for the ``memory`` parameter.
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
