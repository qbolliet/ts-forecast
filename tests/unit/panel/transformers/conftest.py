"""Shared panels and doubles of the ``PanelwiseTransformer`` tests.

Small hand-built panels whose per-entity centring is computable by hand (two
entities ``FR`` then ``DE`` on three month ends, in a ``MultiIndex`` or in
columns), the expected centred values, and test doubles: ``FailingTransformer``
(re-exported from ``tests.support.estimators``: a centring transformer that
raises at a chosen method, targeted at one entity through ``entity_kwargs``)
and ``RecordingScaler`` (remembers the target it was fitted with).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# Doublure partagée, réexportée pour les modules de tests du paquet
from tests.support.estimators import FailingTransformer  # noqa: F401


# =============================================================================
# Doublures
# =============================================================================
class RecordingScaler(StandardScaler):
    """``StandardScaler`` remembering the target it was fitted with."""

    def fit(self, X, y=None, sample_weight=None):
        """Fit like ``StandardScaler`` and keep a copy of the target."""
        self.target_ = None if y is None else y.copy()
        return super().fit(X, y, sample_weight)


def centring() -> StandardScaler:
    """Entity transformer subtracting the entity mean of each column, returning an array."""
    return StandardScaler(with_std=False)


# =============================================================================
# Jeux de données
# =============================================================================
def month_ends(periods: int = 3) -> pd.DatetimeIndex:
    """Month ends from 2024-01-31, named ``date``."""
    return pd.date_range('2024-01-31', periods=periods, freq='ME', name='date')


def build_index_panel() -> pd.DataFrame:
    """Panel ``(country, date)``: ``FR`` then ``DE``, three month ends, columns ``a`` and ``b``.

    ``a``: FR 1, 2, 6 (mean 3), DE 10, 20, 60 (mean 30); ``b``: FR 0, 4, 8 (mean 4),
    DE constant 3. The pooled mean of ``a`` (16.5) differs from both entity means,
    so a centring that mixes the entities is visible.
    """
    index = pd.MultiIndex.from_product([['FR', 'DE'], month_ends()], names=['country', 'date'])
    return pd.DataFrame({'a': [1.0, 2.0, 6.0, 10.0, 20.0, 60.0], 'b': [0.0, 4.0, 8.0, 3.0, 3.0, 3.0]}, index=index)


def centred_values() -> dict:
    """Golden centred columns of :func:`build_index_panel`, entity by entity, rows in order."""
    # Valeur d'or : a - 3 pour FR, a - 30 pour DE ; b - 4 pour FR, b - 3 pour DE
    return {'a': [-2.0, -1.0, 3.0, -20.0, -10.0, 30.0], 'b': [-4.0, 0.0, 4.0, 0.0, 0.0, 0.0]}


@pytest.fixture
def index_panel() -> pd.DataFrame:
    """Two-entity ``MultiIndex`` panel, see :func:`build_index_panel`."""
    return build_index_panel()


@pytest.fixture
def column_panel() -> pd.DataFrame:
    """Column counterpart of :func:`build_index_panel`: ``country``, ``date``, ``a``, ``b``, labels 0..5."""
    return build_index_panel().reset_index()


@pytest.fixture
def centred_index_panel(index_panel) -> pd.DataFrame:
    """Expected output of the per-entity centring of ``index_panel``."""
    return pd.DataFrame(centred_values(), index=index_panel.index)


@pytest.fixture
def centred_column_panel(column_panel) -> pd.DataFrame:
    """Expected output of the per-entity centring of ``column_panel``, columns kept."""
    return column_panel.assign(**centred_values())


@pytest.fixture
def scaled_rows() -> np.ndarray:
    """Golden ``StandardScaler`` output of column ``a`` of ``index_panel``, rows in order."""
    # Valeur d'or : FR (1, 2, 6) de moyenne 3 et d'écart-type sqrt(14/3) ; DE = 10 × FR
    return np.array([-2.0, -1.0, 3.0] * 2) / np.sqrt(14.0 / 3.0)
