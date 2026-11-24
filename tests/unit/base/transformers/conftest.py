"""Shared doubles and fixtures of the ``tsforecast.base.transformers`` tests.

Minimal concrete subclasses of the abstract bases (``ConcreteTransformer``, an
identity transformer that records what ``_fit`` receives; ``CenteredScaler``, a
reversible centring-and-scaling transformer with hand-computable outputs;
``FixedSortTransformer``, which does not expose ``auto_sort``) and small
deterministic datasets: a monthly series, a two-entity ``MultiIndex`` panel and
its column-based counterpart.
"""
from typing import Optional

import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from tsforecast.base.transformers import PanelTimeSeriesTransformer, ReversibleTransformerMixin


# =============================================================================
# Doublures : sous-classes concrètes minimales
# =============================================================================
class ConcreteTransformer(PanelTimeSeriesTransformer):
    """Identity transformer recording the data and target received by ``_fit``.

    Attributes:
        fit_input_: Copy of the (validated) data passed to ``_fit``.
        fit_y_: Target passed to ``_fit``, as received.
    """

    def _fit(self, X, y=None) -> None:
        # Enregistrement de ce que la classe de base transmet à la sous-classe
        self.fit_input_ = X.copy()
        self.fit_y_ = y

    def _transform(self, X):
        # Identité : la sortie est exactement la donnée validée
        return X


class CenteredScaler(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
    """Reversible transformer: ``(X - mean) * factor`` on the numeric columns seen at ``fit``.

    The mean is pooled over every row (all entities together), so outputs are
    computable by hand. ``inverse_transform`` divides by ``factor``, adds the mean
    back and restores the column structure when ``convert_cols_to_index=True``.

    Args:
        factor: Multiplicative factor applied after centring.
        time_col: See ``PanelTimeSeriesTransformer``.
        panel_cols: See ``PanelTimeSeriesTransformer``.
        validate_input: See ``PanelTimeSeriesTransformer``.
        strict_validation: See ``PanelTimeSeriesTransformer``.
        auto_sort: See ``PanelTimeSeriesTransformer``.
        convert_cols_to_index: See ``PanelTimeSeriesTransformer``.
    """

    def __init__(self, factor: float = 2.0, time_col: Optional[str] = None, panel_cols=None,
                 validate_input: bool = True, strict_validation: bool = True,
                 auto_sort: bool = False, convert_cols_to_index: bool = False):
        super().__init__(time_col=time_col, panel_cols=panel_cols, validate_input=validate_input,
                         strict_validation=strict_validation, auto_sort=auto_sort,
                         convert_cols_to_index=convert_cols_to_index)
        self.factor = factor

    def _fit(self, X, y=None) -> None:
        # Moyenne poolée des colonnes numériques : seul état appris
        self.center_ = X.select_dtypes(include='number').mean()

    def _transform(self, X):
        # Centrage puis mise à l'échelle des seules colonnes vues au fit
        out = X.copy()
        columns = list(self.center_.index)
        out[columns] = (X[columns] - self.center_) * self.factor
        self._store_transformation_info(X, out)
        return out

    def inverse_transform(self, X):
        """Undo the scaling and centring, then restore the original structure."""
        # Refus explicite avant fit : sans moyenne apprise, aucune inversion possible
        check_is_fitted(self)
        out = X.copy()
        columns = list(self.center_.index)
        out[columns] = X[columns] / self.factor + self.center_
        return self._restore_structure_if_converted(out)


class FixedSortTransformer(ConcreteTransformer):
    """Concrete transformer that never sorts and does not expose ``auto_sort``."""

    def __init__(self, time_col: Optional[str] = None, panel_cols=None):
        super().__init__(time_col=time_col, panel_cols=panel_cols, auto_sort=False)


# =============================================================================
# Constructeurs de données déterministes
# =============================================================================
def month_ends(periods: int = 4, start: str = '2023-01-31') -> pd.DatetimeIndex:
    """Build a month-end ``DatetimeIndex`` (``ME``) named ``date``."""
    return pd.date_range(start, periods=periods, freq='ME', name='date')


def build_series_frame() -> pd.DataFrame:
    """Monthly frame ``value = [1, 2, 3, 4]`` on four month ends.

    Golden values under ``CenteredScaler(factor=2)``: mean 2.5, output ``[-3, -1, 1, 3]``.
    """
    return pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0]}, index=month_ends())


def build_index_panel() -> pd.DataFrame:
    """Two-entity panel (``FR`` then ``DE``) on a ``(country, date)`` ``MultiIndex``.

    Columns ``a`` (FR 1..4, DE 5..8) and ``b`` (ten times ``a``). Golden values under
    ``CenteredScaler(factor=2)``: pooled means 4.5 and 45, so ``a`` becomes
    FR ``[-7, -5, -3, -1]`` and DE ``[1, 3, 5, 7]``.
    """
    index = pd.MultiIndex.from_product([['FR', 'DE'], month_ends()], names=['country', 'date'])
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    return pd.DataFrame({'a': a, 'b': [10.0 * v for v in a]}, index=index)


def build_column_panel() -> pd.DataFrame:
    """Column-based counterpart of :func:`build_index_panel` (``country``, ``date``, ``a``, ``b``)."""
    return build_index_panel().reset_index()


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def series_frame() -> pd.DataFrame:
    """Monthly single-column frame, see :func:`build_series_frame`."""
    return build_series_frame()


@pytest.fixture
def index_panel() -> pd.DataFrame:
    """Sorted two-entity ``MultiIndex`` panel, see :func:`build_index_panel`."""
    return build_index_panel()


@pytest.fixture
def column_panel() -> pd.DataFrame:
    """Sorted two-entity column-based panel, see :func:`build_column_panel`."""
    return build_column_panel()
