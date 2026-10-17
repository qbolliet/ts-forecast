"""Shared panels and doubles of the ``PanelwiseTransformer`` tests.

Small hand-built panels whose per-entity centring is computable by hand (two
entities ``FR`` then ``DE`` on three month ends, in a ``MultiIndex`` or in
columns), the expected centred values, and test doubles: ``FailingTransformer``
(a centring transformer that raises at a chosen method, targeted at one entity
through ``entity_kwargs``) and ``RecordingScaler`` (remembers the target it was
fitted with).
"""
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


# =============================================================================
# Doublures
# =============================================================================
class FailingTransformer(BaseEstimator, TransformerMixin):
    """Centring transformer (``X - mean``) that raises at the method named by ``fail_on``.

    It returns a ``DataFrame`` that keeps the index of its input. Combined with
    ``entity_kwargs={('DE',): {'fail_on': 'fit'}}``, the failure hits one entity only.

    Args:
        fail_on: ``'fit'``, ``'transform'``, ``'inverse_transform'`` or None (never fails).
        error: Exception class raised, with the message ``'deliberate failure'``.
    """

    def __init__(self, fail_on: Optional[str] = None, error: type = RuntimeError):
        self.fail_on = fail_on
        self.error = error

    def _maybe_fail(self, method: str) -> None:
        # Échec délibéré à la méthode désignée
        if self.fail_on == method:
            raise self.error('deliberate failure')

    def fit(self, X, y=None):
        """Learn the column means, unless ``fail_on='fit'``."""
        self._maybe_fail('fit')
        self.mean_ = X.mean()
        return self

    def transform(self, X):
        """Subtract the learnt means, unless ``fail_on='transform'``."""
        check_is_fitted(self)
        self._maybe_fail('transform')
        return X - self.mean_

    def inverse_transform(self, X):
        """Add the learnt means back, unless ``fail_on='inverse_transform'``."""
        check_is_fitted(self)
        self._maybe_fail('inverse_transform')
        return X + self.mean_


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
