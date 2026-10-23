"""Cross-validators for a single time series: ``TSOutOfSampleSplit`` and ``TSInSampleSplit``.

Both treat the data as one temporal sequence: ``groups`` are ignored with a
warning, and panel data (a ``MultiIndex``) is refused in favour of the
``Panel*Split`` classes of :mod:`tsforecast.crossvals.panel`.
"""
# Importation des modules
# Modules de base
from collections.abc import Iterator
from typing import Any
import warnings

import numpy as np
import pandas as pd

# Importations locales
from ..panel.utils import is_panel_data
from .base import InSampleSplit, OutOfSampleSplit


# Fonction de refus des données de panel par un splitter de série temporelle
def _check_single_series(splitter, X, panel_class: str) -> None:
    """Refuse panel data: a time series splitter needs a single sequence.

    Raises:
        ValueError: If ``X`` has a MultiIndex.
    """
    # Panel (MultiIndex entité..., date) : découpé par position, il mêlerait les entités
    if isinstance(X, (pd.Series, pd.DataFrame)) and is_panel_data(X):
        raise ValueError(
            f"{splitter.__class__.__name__} splits a single time series, but X has a MultiIndex "
            f"(panel data); use {panel_class} instead"
        )


# Fonction d'avertissement pour des groupes ignorés
def _warn_ignored_groups(splitter, groups) -> None:
    """Warn that ``groups`` are ignored by a time series splitter."""
    # Groupes fournis : sans effet sur une série unique, signalés à l'utilisateur
    if groups is not None:
        warnings.warn(
            f"The groups parameter is ignored by {splitter.__class__.__name__}. "
            f"Time series splits treat data as a single temporal sequence.",
            UserWarning,
        )


# Classe de validation croisée out-of-sample sur séries temporelles
class TSOutOfSampleSplit(OutOfSampleSplit):
    """Out-of-sample cross-validator for a single time series.

    Same folds as :class:`~tsforecast.crossvals.base.OutOfSampleSplit` without
    groups: training strictly before each test set, ``gap`` rows apart, in an
    expanding or rolling (``max_train_size``) window.

    Args:
        n_splits: Number of folds when ``test_indices`` is None.
        test_indices: Test starts: integer positions or dates.
        max_train_size: Maximum number of training rows, or None.
        test_size: Number of rows of each test set.
        gap: Number of rows between training and test.

    Examples:
        >>> import numpy as np
        >>> X = np.arange(10).reshape(-1, 1)
        >>> [(train.tolist(), test.tolist()) for train, test in TSOutOfSampleSplit(n_splits=2, test_size=2, gap=1).split(X)]
        [([0, 1, 2, 3, 4], [6, 7]), ([0, 1, 2, 3, 4, 5, 6], [8, 9])]
    """

    # Méthode principale de séparation des données de séries temporelles
    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate out-of-sample folds of a single time series.

        Args:
            X: Array-like, Series or DataFrame without MultiIndex.
            y: Target with the same index as ``X``.
            groups: Ignored, with a warning.

        Returns:
            Generator of ``(train_positions, test_positions)`` integer arrays.

        Raises:
            ValueError: If ``X`` has a MultiIndex (panel data), or for any
                error of ``OutOfSampleSplit.split``.

        Warns:
            UserWarning: If ``groups`` is given.

        Examples:
            >>> import pandas as pd
            >>> X = pd.Series(range(100), index=pd.date_range('2020-01-01', periods=100))
            >>> len(list(TSOutOfSampleSplit(n_splits=3, test_size=10).split(X)))
            3
        """
        # Refus d'un panel, puis avertissement si des groupes sont fournis
        _check_single_series(self, X, 'PanelOutOfSampleSplit')
        _warn_ignored_groups(self, groups)
        # Découpage de la classe parente sans groupes : comportement de série temporelle
        return super().split(X, y, groups=None)


# Classe de validation croisée in-sample sur séries temporelles
class TSInSampleSplit(InSampleSplit):
    """In-sample cross-validator for a single time series.

    Same fold as :class:`~tsforecast.crossvals.base.InSampleSplit` without
    groups: one fold whose training set contains the test period.

    Args:
        n_splits: Sets the default ``test_size`` (``n_samples // (n_splits + 1)``).
        test_indices: Test rows: integer positions or dates.
        max_train_size: Maximum number of training rows, or None.
        test_size: Number of rows of the test period.

    Examples:
        >>> import numpy as np
        >>> [(train.tolist(), test.tolist()) for train, test in TSInSampleSplit(test_size=2).split(np.arange(5))]
        [([0, 1, 2, 3, 4], [3, 4])]
    """

    # Méthode principale de séparation des données de séries temporelles (in-sample)
    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate the in-sample fold of a single time series.

        Args:
            X: Array-like, Series or DataFrame without MultiIndex.
            y: Target with the same index as ``X``.
            groups: Ignored, with a warning.

        Returns:
            Generator of one ``(train_positions, test_positions)`` pair.

        Raises:
            ValueError: If ``X`` has a MultiIndex (panel data), or for any
                error of ``InSampleSplit.split``.

        Warns:
            UserWarning: If ``groups`` is given.

        Examples:
            >>> import pandas as pd
            >>> X = pd.Series(range(100), index=pd.date_range('2020-01-01', periods=100))
            >>> train, test = next(TSInSampleSplit(test_size=10).split(X))
            >>> bool(set(test) <= set(train))
            True
        """
        # Refus d'un panel, puis avertissement si des groupes sont fournis
        _check_single_series(self, X, 'PanelInSampleSplit')
        _warn_ignored_groups(self, groups)
        # Découpage de la classe parente sans groupes : comportement de série temporelle
        return super().split(X, y, groups=None)
