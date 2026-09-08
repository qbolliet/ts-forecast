"""Tracking helpers turning fitted tsforecast components into flat metric dicts.

The package is a thin, dependency-free layer meant for experiment trackers such
as MLflow: every function returns a plain ``dict[str, float]`` (or a mapping of
scalars) that can be handed straight to ``mlflow.log_metrics`` or written to a
CSV. Nothing here imports MLflow, and nothing here fits or mutates a component —
the inputs are already-fitted estimators.

Functions:
    imputation_metrics: Provenance breakdown and cascade summary of a fitted
        ``HighFrequencyImputer``.
    delay_metrics: Applied-delay summary of a fitted
        ``PublicationDelayTransformer``.
    split_summary: Fold-count and train/test size summary of any tsforecast
        cross-validation splitter.
"""
# Réexport des helpers de tracking
from .metrics import imputation_metrics, delay_metrics, split_summary

__all__ = [
    "imputation_metrics",
    "delay_metrics",
    "split_summary",
]
