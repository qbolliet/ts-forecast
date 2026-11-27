"""Sklearn-compatible adapters for third-party forecasting packages.

Re-exports ``SktimeAdapter`` (sktime forecasters), ``DartsAdapter`` (darts
forecasting models), ``HierarchicalForecastAdapter`` (hierarchicalforecast
reconciliation, with the conversion functions ``stack_nodes`` and
``unstack_nodes`` between wide and long hierarchical data and the
cross-validation folds ``period_splits`` of temporal hierarchies). Importing this
package never imports a third-party package: each adapter imports its package
inside its methods and, when it is missing, raises an ``ImportError`` naming the
extra to install
(``adapters-sktime``, ``adapters-darts``, ``adapters-hierarchical`` or
``adapters-all``).
"""
from tsforecast.adapters.darts import DartsAdapter
from tsforecast.adapters.hierarchicalforecast import (
    HierarchicalForecastAdapter,
    period_splits,
    stack_nodes,
    unstack_nodes,
)
from tsforecast.adapters.sktime import SktimeAdapter

__all__ = [
    "DartsAdapter",
    "HierarchicalForecastAdapter",
    "SktimeAdapter",
    "period_splits",
    "stack_nodes",
    "unstack_nodes",
]
