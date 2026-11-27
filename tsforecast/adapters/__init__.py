"""Sklearn-compatible adapters for third-party forecasting packages.

Re-exports ``SktimeAdapter`` (sktime forecasters), ``DartsAdapter`` (darts
forecasting models), ``HierarchicalForecastAdapter`` (hierarchicalforecast
reconciliation, with the conversion functions ``stack_nodes`` and
``unstack_nodes`` between wide and long hierarchical data and the
cross-validation folds ``period_splits`` of temporal hierarchies) and
``FoundationModelAdapter`` (pretrained foundation models such as Chronos, with
``CallableBackend``, ``register_backend`` and ``clear_model_cache``). Importing this
package never imports a third-party package: each adapter imports its package
inside its methods and, when it is missing, raises an ``ImportError`` naming the
extra to install
(``adapters-sktime``, ``adapters-darts``, ``adapters-hierarchical``,
``adapters-chronos``, ``adapters-light`` for every adapter without foundation
model, or ``adapters-all``).
"""
from tsforecast.adapters.darts import DartsAdapter
from tsforecast.adapters.foundation import (
    CallableBackend,
    FoundationModelAdapter,
    clear_model_cache,
    register_backend,
)
from tsforecast.adapters.hierarchicalforecast import (
    HierarchicalForecastAdapter,
    period_splits,
    stack_nodes,
    unstack_nodes,
)
from tsforecast.adapters.sktime import SktimeAdapter

__all__ = [
    "CallableBackend",
    "DartsAdapter",
    "FoundationModelAdapter",
    "HierarchicalForecastAdapter",
    "SktimeAdapter",
    "clear_model_cache",
    "period_splits",
    "register_backend",
    "stack_nodes",
    "unstack_nodes",
]
