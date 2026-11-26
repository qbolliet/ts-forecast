"""Sklearn-compatible adapters for third-party forecasting packages.

Re-exports ``SktimeAdapter`` (sktime forecasters), ``DartsAdapter`` (darts
forecasting models), ``HierarchicalForecastAdapter`` (hierarchicalforecast
reconciliation, with the conversion functions ``stack_nodes`` and
``unstack_nodes`` between wide and long hierarchical data) and ``OperaAdapter``
(opera online expert aggregation). Importing this package never imports a
third-party package: each adapter imports its package inside its methods and,
when it is missing, raises an ``ImportError`` naming the extra to install
(``adapters-sktime``, ``adapters-darts``, ``adapters-hierarchical``,
``adapters-opera`` or ``adapters-all``).
"""
from tsforecast.adapters.darts import DartsAdapter
from tsforecast.adapters.hierarchicalforecast import HierarchicalForecastAdapter, stack_nodes, unstack_nodes
from tsforecast.adapters.opera import OperaAdapter
from tsforecast.adapters.sktime import SktimeAdapter

__all__ = [
    "DartsAdapter",
    "HierarchicalForecastAdapter",
    "OperaAdapter",
    "SktimeAdapter",
    "stack_nodes",
    "unstack_nodes",
]
