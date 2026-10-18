"""Mixin pour transformateurs XY qui transforment à la fois X et y.

Ce module fournit XYTransformerMixin, une classe mixin qui standardise
l'interface pour les transformateurs capables de transformer simultanément
les features (X) et les targets (y).
"""
# Importation des modules
# Modules de base
from abc import ABC, abstractmethod
from typing import Any
# Manipulation de données
import numpy as np
# Sklearn
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
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

    ``fit`` marks the instance as fitted, so that ``check_is_fitted`` (and an
    ``XYPipeline`` ending with the step) recognises a fitted subclass even
    when its ``_fit`` learns nothing.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.base import BaseEstimator
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
        >>> X = pd.DataFrame({'x': [1.0, 2.0, 4.0]})
        >>> y = pd.Series([10.0, 11.0, 13.0])
        >>> step = LogTransformXY()
        >>> X_t, y_t = step.fit_transform(X, y)
        >>> [round(value, 4) for value in y_t]
        [0.0, 0.6931, 1.3863]
        >>> X_inv, y_inv = step.inverse_transform(X_t, y_t)
        >>> [round(value, 6) for value in y_inv]
        [10.0, 11.0, 13.0]
    """

    # État ajusté au sens de sklearn : marqueur posé par fit, ou attribut
    # appris (suffixe _) d'une sous-classe qui surcharge fit
    def __sklearn_is_fitted__(self) -> bool:
        """Report whether the transformer is fitted.

        Returns:
            True once ``fit`` has run, or when the instance carries a learnt
            attribute (name ending with ``_``), as ``check_is_fitted`` expects.

        Examples:
            >>> from sklearn.base import BaseEstimator
            >>> class Identity(BaseEstimator, XYTransformerMixin):
            ...     def _fit(self, X, y):
            ...         return self
            ...     def _transform(self, X, y=None):
            ...         return X if y is None else (X, y)
            >>> step = Identity()
            >>> step.__sklearn_is_fitted__()
            False
            >>> step.fit([[1.0]], [1.0]).__sklearn_is_fitted__()
            True
        """
        # Marquage de l'attribut
        if getattr(self, '_is_fitted', False):
            return True
        return any(name.endswith('_') and not name.startswith('__') for name in vars(self))

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


# Classe de transformation XY sur des données de panel et des séries temporelles
class XYPanelTimeSeriesTransformer(PanelTimeSeriesTransformer, ABC):
    """Panel transformer with XY transformation support.

    Combines PanelTimeSeriesTransformer validation with the ability to
    transform both X and y simultaneously. Subclasses implement ``_fit(X, y)``
    and ``_transform(X, y=None)``, which returns ``(X_t, y_t)`` when ``y`` is
    given and ``X_t`` otherwise; ``_inverse_transform`` is optional.

    Examples:
        >>> import pandas as pd
        >>> class CentreXY(XYPanelTimeSeriesTransformer):
        ...     def _fit(self, X, y=None):
        ...         self.mean_ = X.mean()
        ...     def _transform(self, X, y=None):
        ...         return X - self.mean_ if y is None else (X - self.mean_, y * 2)
        >>> dates = pd.date_range('2024-01-31', periods=3, freq='ME')
        >>> X = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=dates)
        >>> X_t, y_t = CentreXY().fit_transform(X, pd.Series([1.0, 2.0, 3.0], index=dates))
        >>> X_t['a'].tolist(), y_t.tolist()
        ([-1.0, 0.0, 1.0], [2.0, 4.0, 6.0])
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
        # Données à deux dimensions seulement (convention sklearn)
        self._check_input_type(X)
        # Validation de X
        if self.validate_input:
            X = self._validate_input(X)
        # Vérification des colonnes vues au fit
        self._check_feature_names(X)

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

# Étape XY de retrait des lignes sans cible, placée juste avant l'estimateur ;
# sorties rendues dans les types d'entrée, d'où l'absence d'enveloppe set_output
class DropNaNTarget(OneToOneFeatureMixin, BaseEstimator, XYTransformerMixin, auto_wrap_output_keys=None):
    """Drop the rows whose target is missing, right before an estimator.

    A training set augmented upstream (``HighFrequencyImputer.fit_transform``,
    lagged features towards several horizons...) keeps rows without target on
    purpose: a later step may still read them. No estimator learns from them,
    though, and most refuse a NaN target. This step drops them **at the last
    moment**, in the training pass (``fit_transform``) and when ``score``
    passes ``y`` (``transform(X, y)``); without ``y`` — ``predict`` — ``X``
    goes through untouched: a prediction never loses a row.

    The columns are kept: ``fit`` only records ``n_features_in_`` (and
    ``feature_names_in_`` for a ``DataFrame``), so that
    ``get_feature_names_out`` gives back the input names. The outputs keep the
    types of the inputs: ``set_output`` is accepted and has no effect.

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
        >>> DropNaNTarget().fit(X, y).get_feature_names_out().tolist()
        ['x']
    """

    # Étape sans paramètre appris : toujours ajustée au sens de sklearn
    def __sklearn_is_fitted__(self) -> bool:
        """Report the step as fitted: it learns no parameter."""
        return True

    # Apprentissage : colonnes d'entrée seulement, pour les noms de sortie
    def _fit(self, X, y):
        """Record the number (and names) of the input columns.

        A one-dimensional ``X`` (``Series``) has no columns to record.

        Args:
            X: Features.
            y: Target.

        Returns:
            self
        """
        if np.ndim(X) == 2:
            validate_data(self, X, reset=True, skip_check_array=True)
        return self

    # Sorties dans les types d'entrée : configuration de sortie sans effet
    def set_output(self, *, transform: Any = None) -> 'DropNaNTarget':
        """Accept an output configuration and ignore it.

        The step returns its inputs, rows dropped, in their own types; it
        accepts ``set_output`` so that ``XYPipeline.set_output`` can configure
        the other steps.

        Args:
            transform: Output container requested (``'pandas'``,
                ``'default'``...), ignored.

        Returns:
            self

        Examples:
            >>> step = DropNaNTarget()
            >>> step.set_output(transform='pandas') is step
            True
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


# Reconnaissance d'une étape XY par déclaration : héritage d'une des deux classes de base XY
def is_xy_transformer(estimator: Any) -> bool:
    """Tell whether ``estimator`` is declared as an XY transformer.

    A step is XY when it inherits :class:`XYTransformerMixin` or
    :class:`XYPanelTimeSeriesTransformer`. ``XYPipeline`` gives such a step the
    target (``fit_transform(X, y)``, ``transform(X, y)``,
    ``inverse_transform(X, y)``) and reads its output as the pair ``(X, y)``;
    any other step is handled as by ``sklearn.pipeline.Pipeline``, whatever
    its signature or output.

    Args:
        estimator: Step to classify.

    Returns:
        True for an instance of one of the two XY base classes.

    Examples:
        >>> from sklearn.preprocessing import StandardScaler
        >>> is_xy_transformer(DropNaNTarget()), is_xy_transformer(StandardScaler())
        (True, False)
    """
    return isinstance(estimator, (XYTransformerMixin, XYPanelTimeSeriesTransformer))
