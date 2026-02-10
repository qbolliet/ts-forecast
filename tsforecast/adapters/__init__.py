"""Adapters module for integrating third-party forecasting libraries.

This module provides sklearn-compatible wrappers for popular time series
forecasting libraries. Each adapter requires its respective library to be
installed separately as an optional dependency.

Available adapters:
    - SktimeAdapter: Wrapper for sktime forecasters
    - DartsAdapter: Wrapper for darts GlobalForecastingModel
    - HierarchicalForecastAdapter: Wrapper for hierarchicalforecast reconciliation
    - OperaAdapter: Wrapper for opera ensemble methods

Installation:
    pip install ts-forecast[adapters-sktime]
    pip install ts-forecast[adapters-darts]
    pip install ts-forecast[adapters-hierarchical]
    pip install ts-forecast[adapters-opera]
    pip install ts-forecast[adapters-all]  # Install all adapters
"""

# Lazy import pour ne pas forcer l'installation de l'ensemble des packages sktime, darts, opera-python et hierarchicalforecast
def __getattr__(name):
    """Lazy import of adapters with helpful error messages.

    This function enables lazy loading of adapters, ensuring that the module
    can be imported even if optional dependencies are not installed. Errors
    are only raised when trying to actually use an adapter whose dependency
    is missing.

    Args:
        name: The name of the adapter class to import.

    Returns:
        The requested adapter class.

    Raises:
        ImportError: If the required package for the adapter is not installed.
        AttributeError: If the requested attribute does not exist.
    """
    if name == "SktimeAdapter":
        try:
            from .sktime import SktimeAdapter
            return SktimeAdapter
        except ImportError:
            raise ImportError(
                "SktimeAdapter requires the 'sktime' package. "
                "Install it with: pip install ts-forecast[adapters-sktime] "
                "or: pip install sktime"
            )
    elif name == "DartsAdapter":
        try:
            from .darts import DartsAdapter
            return DartsAdapter
        except ImportError:
            raise ImportError(
                "DartsAdapter requires the 'darts' package. "
                "Install it with: pip install ts-forecast[adapters-darts] "
                "or: pip install darts"
            )

    elif name == "HierarchicalForecastAdapter":
        try:
            from .hierarchicalforecast import HierarchicalForecastAdapter
            return HierarchicalForecastAdapter
        except ImportError:
            raise ImportError(
                "HierarchicalForecastAdapter requires the 'hierarchicalforecast' package. "
                "Install it with: pip install ts-forecast[adapters-hierarchical] "
                "or: pip install hierarchicalforecast"
            )

    elif name == "OperaAdapter":
        try:
            from .opera import OperaAdapter
            return OperaAdapter
        except ImportError:
            raise ImportError(
                "OperaAdapter requires the 'opera' package. "
                "Install it with: pip install ts-forecast[adapters-opera] "
                "or: pip install opera"
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Export des modules du package
__all__ = [
    "SktimeAdapter",
    "DartsAdapter",
    "HierarchicalForecastAdapter",
    "OperaAdapter",
]
