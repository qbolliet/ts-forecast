"""Cross-validation utilities for time series and panel data.

This module provides cross-validation classes that respect temporal ordering
and support both time series and panel data structures.

Classes:
    OutOfSampleSplit: Base class for out-of-sample cross-validation
    InSampleSplit: Base class for in-sample cross-validation  
    TSOutOfSampleSplit: Time series specific out-of-sample validation
    TSInSampleSplit: Time series specific in-sample validation
    PanelOutOfSampleSplit: Panel data specific out-of-sample validation
    PanelInSampleSplit: Panel data specific in-sample validation

Examples:
    >>> # Time series validation
    >>> from tsforecast.crossvals import TSOutOfSampleSplit
    >>> splitter = TSOutOfSampleSplit(n_splits=5, test_size=10, gap=1)
    >>> for train_idx, test_idx in splitter.split(X):
    ...     # Perform validation
    
    >>> # Panel data validation
    >>> from tsforecast.crossvals import PanelOutOfSampleSplit
    >>> splitter = PanelOutOfSampleSplit(test_size=5)
    >>> for train_idx, test_idx in splitter.split(X, groups=groups):
    ...     # Perform validation across entities
"""

# Importtation des classes de base
from .base import OutOfSampleSplit, InSampleSplit

# Importation des classes pour les séries temporelles
from .time_series import TSOutOfSampleSplit, TSInSampleSplit

# Import des classes pour les données de panel
from .panel import PanelOutOfSampleSplit, PanelInSampleSplit, PanelOutOfSampleSplitPerEntity, PanelInSampleSplitPerEntity

__all__ = [
    # Base
    "OutOfSampleSplit",
    "InSampleSplit",
    # Séries temporelles
    "TSOutOfSampleSplit", 
    "TSInSampleSplit",
    # Panel - global
    "PanelOutOfSampleSplit",
    "PanelInSampleSplit",
    # Panel - Par entité
    "PanelOutOfSampleSplitPerEntity",
    "PanelInSampleSplitPerEntity"
]