"""Hierarchical reconciliation adapter for hierarchicalforecast.

Re-exports ``HierarchicalForecastAdapter`` (sklearn regressor reconciling the base
forecasts of a cross-sectional, temporal or cross-temporal hierarchy given in a
long layout), the conversion functions ``stack_nodes`` / ``unstack_nodes`` between
wide data (one column per node) and that layout, and ``period_splits``
(cross-validation folds of complete periods of a temporal hierarchy). Importing
this package never imports hierarchicalforecast.
"""
# Importation des composants du sous-paquet
from tsforecast.adapters.hierarchicalforecast.adapter import HierarchicalForecastAdapter
from tsforecast.adapters.hierarchicalforecast.reshape import stack_nodes, unstack_nodes
from tsforecast.adapters.hierarchicalforecast.splits import period_splits

# Symboles exportés
__all__ = [
    "HierarchicalForecastAdapter",
    "period_splits",
    "stack_nodes",
    "unstack_nodes",
]
