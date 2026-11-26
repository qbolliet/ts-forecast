"""Hierarchical reconciliation adapter for hierarchicalforecast.

Re-exports ``HierarchicalForecastAdapter`` (sklearn regressor reconciling the base
forecasts of a hierarchy given in a long layout) and the conversion functions
``stack_nodes`` / ``unstack_nodes`` between wide data (one column per node) and
that layout. Importing this package never imports hierarchicalforecast.
"""
# Importation des composants du sous-paquet
from tsforecast.adapters.hierarchicalforecast.adapter import HierarchicalForecastAdapter
from tsforecast.adapters.hierarchicalforecast.reshape import stack_nodes, unstack_nodes

# Symboles exportés
__all__ = [
    "HierarchicalForecastAdapter",
    "stack_nodes",
    "unstack_nodes",
]
