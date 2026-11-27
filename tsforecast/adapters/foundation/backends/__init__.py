"""Backends of ``FoundationModelAdapter``: one module per model family.

Re-exports ``CallableBackend`` (any model plugged in through a function),
``ChronosBackend`` with its loader ``load_chronos`` (Chronos-2 and Chronos-Bolt),
``TimestralBackend`` with ``load_timestral`` and ``T0Backend`` with ``load_t0``.
No third-party package is imported until a model is loaded.
"""
from tsforecast.adapters.foundation.backends.callable import CallableBackend
from tsforecast.adapters.foundation.backends.chronos import ChronosBackend, load_chronos
from tsforecast.adapters.foundation.backends.t0 import T0Backend, load_t0
from tsforecast.adapters.foundation.backends.timestral import TimestralBackend, load_timestral

__all__ = [
    "CallableBackend",
    "ChronosBackend",
    "T0Backend",
    "TimestralBackend",
    "load_chronos",
    "load_t0",
    "load_timestral",
]
