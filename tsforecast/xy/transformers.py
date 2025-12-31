"""Pipeline extension that supports transformers returning (X_transformed, y_transformed).

This module provides XYPipeline, a drop-in replacement for sklearn.pipeline.Pipeline
that can handle transformers with the following signatures:
- Standard: transform(X) -> X_t
- XY transformers: transform(X, y) -> (X_t, y_t)

The pipeline is fully compatible with sklearn's GridSearchCV, metadata routing (1.4+),
and pandas DataFrames.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

import numpy as np
from sklearn import get_config
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

# Import conditionnel pour pandas
try:
    import pandas as pd
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False
    pd = None

# Import conditionnel pour le metadata routing (sklearn >= 1.4)
try:
    from sklearn.utils.metadata_routing import (
        MetadataRouter,
        MethodMapping,
        process_routing,
        _routing_enabled,
    )
    METADATA_ROUTING_AVAILABLE = True
except ImportError:
    METADATA_ROUTING_AVAILABLE = False
    MetadataRouter = None
    MethodMapping = None
    process_routing = None
    _routing_enabled = lambda: False







class XYTransformerMixin(TransformerMixin):
    """Mixin class for transformers that transform both X and y.

    This mixin provides a standard interface for XY transformers and ensures
    proper compatibility with XYPipeline.

    Subclasses should implement:
    - _fit(X, y): Learn transformation parameters from X and y.
    - _transform_X(X): Transform features.
    - _transform_y(y): Transform targets.
    - _inverse_transform_X(X): Inverse transform features (optional).
    - _inverse_transform_y(y): Inverse transform targets (optional).

    Example:
        >>> class LogTransformXY(BaseEstimator, XYTransformerMixin):
        ...     def _fit(self, X, y):
        ...         self.X_offset_ = X.min(axis=0)
        ...         self.y_offset_ = y.min() if y is not None else 0
        ...         return self
        ...
        ...     def _transform_X(self, X):
        ...         return np.log1p(X - self.X_offset_)
        ...
        ...     def _transform_y(self, y):
        ...         return np.log1p(y - self.y_offset_)
        ...
        ...     def _inverse_transform_X(self, X):
        ...         return np.expm1(X) + self.X_offset_
        ...
        ...     def _inverse_transform_y(self, y):
        ...         return np.expm1(y) + self.y_offset_
    """

    def fit(self, X, y=None):
        """Fit the transformer.

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            self: The fitted transformer.
        """
        self._fit(X, y)
        self._is_fitted = True
        return self

    def _fit(self, X, y):
        """Learn transformation parameters. Override in subclasses.

        Args:
            X: Features.
            y: Targets.

        Returns:
            self
        """
        raise NotImplementedError("Subclasses must implement _fit")

    def transform(self, X, y=None):
        """Transform X and optionally y.

        Args:
            X: Features to transform.
            y: Targets to transform, optional.

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        X_t = self._transform_X(X)

        if y is not None:
            y_t = self._transform_y(y)
            return X_t, y_t

        return X_t

    def _transform_X(self, X):
        """Transform features. Override in subclasses.

        Args:
            X: Features to transform.

        Returns:
            Transformed features.
        """
        raise NotImplementedError("Subclasses must implement _transform_X")

    def _transform_y(self, y):
        """Transform targets. Override in subclasses.

        Args:
            y: Targets to transform.

        Returns:
            Transformed targets.
        """
        raise NotImplementedError("Subclasses must implement _transform_y")

    def inverse_transform(self, X, y=None):
        """Inverse transform X and optionally y.

        Args:
            X: Transformed features.
            y: Transformed targets, optional.

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.
        """
        X_inv = self._inverse_transform_X(X)

        if y is not None:
            y_inv = self._inverse_transform_y(y)
            return X_inv, y_inv

        return X_inv

    def _inverse_transform_X(self, X):
        """Inverse transform features. Override in subclasses.

        Args:
            X: Transformed features.

        Returns:
            Original features.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _inverse_transform_X"
        )

    def _inverse_transform_y(self, y):
        """Inverse transform targets. Override in subclasses.

        Args:
            y: Transformed targets.

        Returns:
            Original targets.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _inverse_transform_y"
        )

    def fit_transform(self, X, y=None):
        """Fit and transform.

        This method MUST be implemented to properly pass y to transform.

        Args:
            X: Features.
            y: Targets, optional.

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        return self.fit(X, y).transform(X, y)