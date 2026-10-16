"""Mixin pour transformateurs XY qui transforment à la fois X et y.

Ce module fournit XYTransformerMixin, une classe mixin qui standardise
l'interface pour les transformateurs capables de transformer simultanément
les features (X) et les targets (y).
"""
# Importation des modules
# Modules de base
from abc import ABC, abstractmethod
# Manipulation de données
import numpy as np
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
# Module du package
from ..base.transformers import PanelTimeSeriesTransformer


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


# Classe de transformation XYSur des données de panel et des séries temporelles
class XYPanelTimeSeriesTransformer(PanelTimeSeriesTransformer, ABC):
    """Panel transformer with XY transformation support.
    
    Combines PanelTimeSeriesTransformer validation with the ability to
    transform both X and y simultaneously.
    """
    
    # Surcharge de transform pour accepter y tout en gardant les validations
    def transform(self, X, y=None):
        """Transform X and optionally y with full validation.
        
        Args:
            X: Features to transform.
            y: Targets to transform (optional).
            
        Returns:
            X_transformed if y is None, (X_transformed, y_transformed) otherwise.
        """
        # Vérification de l'entraînement
        check_is_fitted(self)
        # Validation de X
        if self.validate_input:
            X = self._validate_input(X)

        return self._transform(X, y)
    
    @abstractmethod
    def _transform(self, X, y=None):
        """Transform features and targets. Override in subclasses.

        Args:
            X: Features to transform.
            y: Targets to transform (optional).

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        pass
    
    # fit_transform explicite pour propager y
    def fit_transform(self, X, y=None):
        """Fit and transform.

        Args:
            X: Features.
            y: Targets (optional).

        Returns:
            X_transformed if y is None, (X_transformed, y_transformed) otherwise.
        """
        return self.fit(X, y).transform(X, y)

    # Méthode de transformation inverse des données
    def inverse_transform(self, X, y=None):
        """Inverse transform X and optionally y.

        Args:
            X: Transformed features.
            y: Transformed targets (optional).

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.
        """
        # Vérification de l'entraînement
        check_is_fitted(self)
        return self._inverse_transform(X, y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform features and targets. Override in subclasses.

        Default implementation raises NotImplementedError.

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

# Étape XY de retrait des lignes sans cible, placée juste avant l'estimateur
class DropNaNTarget(BaseEstimator, XYTransformerMixin):
    """Drop the rows whose target is missing, right before an estimator.

    A training set augmented upstream (``HighFrequencyImputer.fit_transform``,
    lagged features towards several horizons...) keeps rows without target on
    purpose: a later step may still read them. No estimator learns from them,
    though, and most refuse a NaN target. This step drops them **at the last
    moment**, in the training pass (``fit_transform``) and when ``score``
    passes ``y`` (``transform(X, y)``); without ``y`` — ``predict`` — ``X``
    goes through untouched: a prediction never loses a row.

    Stateless: ``fit`` learns nothing.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> X = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        >>> y = pd.Series([10.0, np.nan, 30.0])
        >>> X_kept, y_kept = DropNaNTarget().fit_transform(X, y)
        >>> X_kept['x'].tolist(), y_kept.tolist()
        ([1.0, 3.0], [10.0, 30.0])
        >>> DropNaNTarget().fit(X, y).transform(X).shape
        (3, 1)
    """

    # Étape sans état : toujours ajustée au sens de sklearn
    def __sklearn_is_fitted__(self) -> bool:
        """Report the step as fitted: it learns nothing."""
        return True

    # Apprentissage : rien à apprendre
    def _fit(self, X, y):
        """Learn nothing.

        Args:
            X: Features.
            y: Target.

        Returns:
            self
        """
        return self

    # Transformation : retrait des lignes dont la cible manque
    def _transform(self, X, y=None):
        """Drop the rows of ``X`` and ``y`` whose target is NaN.

        A multi-output target drops a row as soon as one of its targets is
        missing. The rows are matched by position: ``X`` and ``y`` must have
        the same length.

        Args:
            X: Features (DataFrame, Series or array).
            y: Target (Series, DataFrame or array), optional.

        Returns:
            ``X`` unchanged when ``y`` is None, the pair ``(X_kept, y_kept)``
            otherwise, in the input types.

        Raises:
            ValueError: If ``X`` and ``y`` differ in length.
        """
        # Passe de prédiction : aucune ligne n'est retirée
        if y is None:
            return X
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length, got {len(X)} and {len(y)}"
            )

        # Masque des lignes dont toute cible est renseignée
        if hasattr(y, 'notna'):
            observed = y.notna()
            keep = (observed.all(axis=1) if observed.ndim == 2 else observed).to_numpy()
        else:
            values = np.asarray(y, dtype=float).reshape(len(y), -1)
            keep = ~np.isnan(values).any(axis=1)

        return self._take(X, keep), self._take(y, keep)

    # Méthode auxiliaire de sélection positionnelle, quel que soit le type
    @staticmethod
    def _take(data, keep: np.ndarray):
        """Select the rows of ``data`` flagged by ``keep``, by position."""
        return data.iloc[keep] if hasattr(data, 'iloc') else np.asarray(data)[keep]
