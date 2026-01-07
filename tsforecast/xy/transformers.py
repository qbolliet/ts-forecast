"""Mixin pour transformateurs XY qui transforment à la fois X et y.

Ce module fournit XYTransformerMixin, une classe mixin qui standardise
l'interface pour les transformateurs capables de transformer simultanément
les features (X) et les targets (y).
"""

from __future__ import annotations

from sklearn.base import TransformerMixin


# =============================================================================
# Classe mixin pour transformateurs XY
# =============================================================================
# Cette classe fournit une interface standardisée pour les transformateurs
# qui transforment à la fois X et y. Elle est conçue pour être utilisée
# avec XYPipeline qui gère correctement le passage de y à travers le pipeline.

class XYTransformerMixin(TransformerMixin):
    """Mixin class for transformers that transform both X and y.

    This mixin provides a standard interface for XY transformers and ensures
    proper compatibility with XYPipeline.

    Subclasses should implement:
    - _fit(X, y): Learn transformation parameters from X and y.
    - _transform_X(X, y): Transform features (receives both X and y).
    - _transform_y(X, y): Transform targets (receives both X and y).
    - _inverse_transform_X(X, y): Inverse transform features (optional).
    - _inverse_transform_y(X, y): Inverse transform targets (optional).

    The dual-argument signature (_transform_X(X, y) instead of _transform_X(X))
    allows transformations where X depends on y or vice versa.

    Example:
        >>> class LogTransformXY(BaseEstimator, XYTransformerMixin):
        ...     def _fit(self, X, y):
        ...         self.X_offset_ = X.min(axis=0)
        ...         self.y_offset_ = y.min() if y is not None else 0
        ...         return self
        ...
        ...     def _transform_X(self, X, y=None):
        ...         return np.log1p(X - self.X_offset_)
        ...
        ...     def _transform_y(self, X, y=None):
        ...         if y is None:
        ...             return None
        ...         return np.log1p(y - self.y_offset_)
        ...
        ...     def _inverse_transform_X(self, X, y=None):
        ...         return np.expm1(X) + self.X_offset_
        ...
        ...     def _inverse_transform_y(self, X, y=None):
        ...         if y is None:
        ...             return None
        ...         return np.expm1(y) + self.y_offset_
    """

    # -------------------------------------------------------------------------
    # Méthode fit : apprentissage des paramètres de transformation
    # -------------------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the transformer.

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            self: The fitted transformer.
        """
        # Délégation à la méthode abstraite _fit
        self._fit(X, y)

        # Marquage de l'état fitted
        self._is_fitted = True

        return self

    # -------------------------------------------------------------------------
    # Méthode abstraite _fit : apprentissage des paramètres
    # -------------------------------------------------------------------------
    # À implémenter par les sous-classes pour définir la logique d'apprentissage
    def _fit(self, X, y):
        """Learn transformation parameters. Override in subclasses.

        Args:
            X: Features.
            y: Targets.

        Returns:
            self
        """
        raise NotImplementedError("Subclasses must implement _fit")

    # -------------------------------------------------------------------------
    # Méthode transform : transformation des données X et y
    # -------------------------------------------------------------------------
    def transform(self, X, y=None):
        """Transform X and optionally y.

        Args:
            X: Features to transform.
            y: Targets to transform, optional.

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        # Transformation des features (X et y passés pour permettre
        # des transformations conditionnelles)
        X_t = self._transform_X(X, y)

        # Transformation des targets si y est fourni
        if y is not None:
            y_t = self._transform_y(X, y)
            return X_t, y_t

        return X_t

    # -------------------------------------------------------------------------
    # Méthode abstraite _transform_X : transformation des features
    # -------------------------------------------------------------------------
    # À implémenter par les sous-classes
    # Reçoit X et y pour permettre des transformations où X dépend de y
    def _transform_X(self, X, y=None):
        """Transform features. Override in subclasses.

        Args:
            X: Features to transform.
            y: Targets (optional, for conditional transformations).

        Returns:
            Transformed features.
        """
        raise NotImplementedError("Subclasses must implement _transform_X")

    # -------------------------------------------------------------------------
    # Méthode abstraite _transform_y : transformation des targets
    # -------------------------------------------------------------------------
    # À implémenter par les sous-classes
    # Reçoit X et y pour permettre des transformations où y dépend de X
    def _transform_y(self, X, y=None):
        """Transform targets. Override in subclasses.

        Args:
            X: Features (for conditional transformations).
            y: Targets to transform.

        Returns:
            Transformed targets.
        """
        raise NotImplementedError("Subclasses must implement _transform_y")

    # -------------------------------------------------------------------------
    # Méthode inverse_transform : transformation inverse des données
    # -------------------------------------------------------------------------
    def inverse_transform(self, X, y=None):
        """Inverse transform X and optionally y.

        Args:
            X: Transformed features.
            y: Transformed targets, optional.

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.
        """
        # Transformation inverse des features
        X_inv = self._inverse_transform_X(X, y)

        # Transformation inverse des targets si y est fourni
        if y is not None:
            y_inv = self._inverse_transform_y(X, y)
            return X_inv, y_inv

        return X_inv

    # -------------------------------------------------------------------------
    # Méthode _inverse_transform_X : transformation inverse des features
    # -------------------------------------------------------------------------
    # Optionnelle - implémentation par défaut lève une erreur
    def _inverse_transform_X(self, X, y=None):
        """Inverse transform features. Override in subclasses.

        Args:
            X: Transformed features.
            y: Transformed targets (optional, for conditional transformations).

        Returns:
            Original features.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _inverse_transform_X"
        )

    # -------------------------------------------------------------------------
    # Méthode _inverse_transform_y : transformation inverse des targets
    # -------------------------------------------------------------------------
    # Optionnelle - implémentation par défaut lève une erreur
    def _inverse_transform_y(self, X, y=None):
        """Inverse transform targets. Override in subclasses.

        Args:
            X: Transformed features (for conditional transformations).
            y: Transformed targets.

        Returns:
            Original targets.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _inverse_transform_y"
        )

    # -------------------------------------------------------------------------
    # Méthode fit_transform : apprentissage et transformation combinés
    # -------------------------------------------------------------------------
    # Cette méthode DOIT être implémentée pour propager correctement y
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
