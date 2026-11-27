"""Combination of forecasts: online expert aggregation.

Re-exports ``OnlineAggregator`` (scikit-learn regressor aggregating expert
forecasts online with BOA or MLpol weights, as a meta-estimator or on
precomputed forecasts, for time series and panels) and ``Loss`` (a custom loss
and its gradient for the aggregation). Depends on numpy, pandas and
scikit-learn only.
"""
from tsforecast.ensemble._algorithms import Loss
from tsforecast.ensemble.online import OnlineAggregator

__all__ = [
    "Loss",
    "OnlineAggregator",
]
