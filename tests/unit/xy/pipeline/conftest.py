"""Shared fixtures of the ``XYPipeline`` tests.

Small dated frames with golden values (``frame_xy`` with one missing target,
``linear_xy`` with an exactly linear target, ``gapped_linear_xy`` with an exactly
linear target and missing targets spread over the folds of a time split), a
seeded classification set (``classification_xy``) and ``routing``, which enables
sklearn metadata routing for one test and restores the configuration afterwards.
"""

import numpy as np
import pandas as pd
import pytest
import sklearn


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


# Lignes sans cible de gapped_linear_xy : la 5ᵉ figure dans tous les entraînements
# d'un découpage temporel à 3 plis de 3 lignes, la 11ᵉ dans le 2ᵉ test et le 3ᵉ entraînement
GAPPED_ROWS = (4, 10)


@pytest.fixture
def gapped_linear_xy():
    """``(X, y)``: 15 month ends, ``a`` = 1..15, ``b`` = i mod 3, ``y = 2a - b + 1`` with 2 NaN.

    ``y`` is missing on the rows of ``GAPPED_ROWS``. A linear model fitted on the
    observed rows recovers ``2a - b + 1`` exactly, so every score on observed
    rows is 1 and every prediction equals ``2a - b + 1``, missing target or not.
    """
    index = month_ends(15)
    a = np.arange(1.0, 16.0)
    X = pd.DataFrame({'a': a, 'b': np.arange(15) % 3 * 1.0}, index=index)
    y = pd.Series(2.0 * X['a'] - X['b'] + 1.0, name='t')
    y.iloc[list(GAPPED_ROWS)] = np.nan
    return X, y


@pytest.fixture
def classification_xy():
    """``(X, y)``: 40 dated rows, 2 Gaussian features, labels ``'down'`` / ``'up'`` of a noisy linear rule."""
    rng = np.random.default_rng(3)
    index = month_ends(40)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=['a', 'b'], index=index)
    score = X['a'] + 0.5 * X['b'] + rng.normal(0.0, 0.5, 40)
    y = pd.Series(np.where(score > 0, 'up', 'down'), index=index, name='t')
    return X, y


# ==============================================================================
# Configuration sklearn
# ==============================================================================


@pytest.fixture
def routing():
    """Enable sklearn metadata routing for the test, then restore the previous configuration."""
    with sklearn.config_context(enable_metadata_routing=True):
        yield
