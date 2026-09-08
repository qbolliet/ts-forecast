"""TS-Forecast: time series and panel data processing for pseudo-real-time forecasting.

This package provides sklearn-compatible tools for:

- Time-aware cross-validation (in-sample / out-of-sample, time series and panel);
- Publication-delay handling (inference, application, inversion);
- Mixed-frequency imputation with provenance tracking;
- Robust temporal / panel data validation and manipulation.

Modules:
    crossvals: Cross-validation splitters that respect temporal ordering.
    delays: Publication-delay inference and transformers.
    frequency: Frequency detection, conversion and mixed-frequency imputation.
    xy: Pipeline and mixin for transformers that transform X and y jointly.
    panel: Per-entity transformer wrapper and panel utilities.
    base: Base classes and mixins for temporal transformers.
    utils: Cross-cutting temporal utilities (frequency, duration, position, ...).
    tracking: Flat metric builders for experiment trackers (MLflow, ...).

See the documentation (``docs/`` or the mkdocs site) for concepts, tutorials and
the full API reference.
"""

__version__ = "0.1.0"
__author__ = "Quentin Bolliet"

# Importation des classes principales pour en favoriser l'accès
from .crossvals import (
    OutOfSampleSplit,
    InSampleSplit,
    TSOutOfSampleSplit,
    TSInSampleSplit,
    PanelOutOfSampleSplit,
    PanelInSampleSplit,
    PanelOutOfSampleSplitPerEntity,
    PanelInSampleSplitPerEntity,
)
from .delays import PublicationDelayTransformer
from .frequency import HighFrequencyImputer, HighFrequencyImputer2
from .panel import PanelwiseTransformer
from .xy import XYPipeline, XYTransformerMixin
from .tracking import imputation_metrics, delay_metrics, split_summary

__all__ = [
    # Cross-validation
    "OutOfSampleSplit",
    "InSampleSplit",
    "TSOutOfSampleSplit",
    "TSInSampleSplit",
    "PanelOutOfSampleSplit",
    "PanelInSampleSplit",
    "PanelOutOfSampleSplitPerEntity",
    "PanelInSampleSplitPerEntity",
    # Délais de publication
    "PublicationDelayTransformer",
    # Imputation multi-fréquences
    "HighFrequencyImputer",
    "HighFrequencyImputer2",
    # Transformateurs
    "PanelwiseTransformer",
    "XYPipeline",
    "XYTransformerMixin",
    # Tracking
    "imputation_metrics",
    "delay_metrics",
    "split_summary",
]
