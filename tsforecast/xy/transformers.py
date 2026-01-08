"""Mixin pour transformateurs XY qui transforment à la fois X et y.

Ce module fournit XYTransformerMixin, une classe mixin qui standardise
l'interface pour les transformateurs capables de transformer simultanément
les features (X) et les targets (y).
"""
# Importation des modules
# Sklearn
from sklearn.base import TransformerMixin


# Classe mixin pour transformateurs XY, conçue pour être utilisée avec
# XYPipeline qui gère correctement le passage de y à travers le pipeline.
class XYTransformerMixin(TransformerMixin):
    """Mixin class for transformers that transform both X and y.

    This mixin provides a standard interface for XY transformers and ensures
    proper compatibility with XYPipeline.

    Subclasses should implement:
    - _fit(X, y): Learn transformation parameters from X and y.
    - _transform(X, y): Transform features and targets. Returns (X_t, y_t)
        if y is provided, X_t otherwise.
    - _inverse_transform(X, y): Inverse transform (optional). Same return
        convention as _transform.

    The dual-argument signature allows transformations where X depends on y
    or vice versa.

    Example:
        >>> class LogTransformXY(BaseEstimator, XYTransformerMixin):
        ...     def _fit(self, X, y):
        ...         self.X_offset_ = X.min(axis=0)
        ...         self.y_offset_ = y.min() if y is not None else 0
        ...         return self
        ...
        ...     def _transform(self, X, y=None):
        ...         X_t = np.log1p(X - self.X_offset_)
        ...         if y is None:
        ...             return X_t
        ...         y_t = np.log1p(y - self.y_offset_)
        ...         return X_t, y_t
        ...
        ...     def _inverse_transform(self, X, y=None):
        ...         X_inv = np.expm1(X) + self.X_offset_
        ...         if y is None:
        ...             return X_inv
        ...         y_inv = np.expm1(y) + self.y_offset_
        ...         return X_inv, y_inv
    """

    # Méthode d'apprentissage des paramètres de transformation
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

    # Méthode abstraite d'entrainement à implémenter par les sous-classes
    def _fit(self, X, y):
        """Learn transformation parameters. Override in subclasses.

        Args:
            X: Features.
            y: Targets.

        Returns:
            self
        """
        raise NotImplementedError("Subclasses must implement _fit")

    # Méthode de transformation des données X et y
    def transform(self, X, y=None):
        """Transform X and optionally y.

        Args:
            X: Features to transform.
            y: Targets to transform, optional.

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        return self._transform(X, y)

    # Méthode abstraite de transformation à implémenter par les sous-classes
    def _transform(self, X, y=None):
        """Transform features and targets. Override in subclasses.

        Args:
            X: Features to transform.
            y: Targets to transform (optional).

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        raise NotImplementedError("Subclasses must implement _transform")

    # Méthode de transformation inverse des données
    def inverse_transform(self, X, y=None):
        """Inverse transform X and optionally y.

        Args:
            X: Transformed features.
            y: Transformed targets, optional.

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.
        """
        return self._inverse_transform(X, y)

    # Méthode optionnelle de transformation inverse des données, l'implémentation par défaut lève une erreur
    def _inverse_transform(self, X, y=None):
        """Inverse transform features and targets. Override in subclasses.

        Args:
            X: Transformed features.
            y: Transformed targets (optional).

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _inverse_transform"
        )

    # Apprentissage et transformation combinés
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