"""Backends of ``FoundationModelAdapter``: one module per model family.

Re-exports ``CallableBackend`` (any model plugged in through a function) and
``ChronosBackend`` with its loader ``load_chronos`` (Chronos-2 and Chronos-Bolt).
No third-party package is imported until a model is loaded.
"""
from tsforecast.adapters.foundation.backends.callable import CallableBackend
from tsforecast.adapters.foundation.backends.chronos import ChronosBackend, load_chronos

__all__ = ["CallableBackend", "ChronosBackend", "load_chronos"]
