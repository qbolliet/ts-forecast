"""Sklearn-compatible adapters for third-party forecasting packages.

Re-exports ``SktimeAdapter`` (sktime forecasters), ``DartsAdapter`` (darts
global forecasting models) and ``OperaAdapter`` (opera online expert
aggregation). Importing this package never imports a third-party package: each
adapter imports its package inside its methods and, when it is missing, raises
an ``ImportError`` naming the extra to install (``adapters-sktime``,
``adapters-darts``, ``adapters-hierarchical``, ``adapters-opera`` or
``adapters-all``).
"""
from tsforecast.adapters.darts import DartsAdapter
from tsforecast.adapters.opera import OperaAdapter
from tsforecast.adapters.sktime import SktimeAdapter

# HierarchicalForecastAdapter n'est pas exposé : son module ne s'importe pas encore
__all__ = [
    "DartsAdapter",
    "OperaAdapter",
    "SktimeAdapter",
]
