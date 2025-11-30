"""Base classes for time series transformers.

This module provides abstract base classes and mixins for creating
sklearn-compatible time series transformers.
"""
# Importation des modules
# Modules de base
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import warnings
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Importation des fonctions de validation
from .validation import (
    restore_original_structure,
    validate_temporal_data,
    validate_entities_grouped,
    validate_sorted_within_groups
)


# Mixin pour les séries temporelles
class TimeSeriesTransformerMixin:
    """Mixin class for time series transformers.

    Provides common functionality for time series data handling, including
    time index validation and extraction from various formats.

    Examples:
        >>> import pandas as pd
        >>> class MyTransformer(TimeSeriesTransformerMixin):
        ...     def transform(self, X):
        ...         time_idx = self._validate_time_index(X)
        ...         return X
        >>>
        >>> dates = pd.date_range('2023-01-01', periods=5)
        >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        >>> transformer = MyTransformer()
        >>> time_idx = transformer._validate_time_index(df)
        >>> isinstance(time_idx, pd.DatetimeIndex)
        True
    """
    
    # Méthode auxiliaire de validation de l'index
    def _validate_time_index(self, X: pd.DataFrame, time_col: Optional[str] = None) -> pd.DatetimeIndex:
        """Validate and extract time index from data.

        Extracts or converts time index from various formats:
        - From a specified time column
        - From an existing DatetimeIndex
        - By converting a non-datetime index

        Args:
            X: Input data with time information
            time_col: Name of time column. If None, uses the DataFrame index.

        Returns:
            Validated DatetimeIndex extracted from the data

        Raises:
            ValueError: If time index cannot be determined or converted

        Examples:
            >>> import pandas as pd
            >>> # Example 1: Using DatetimeIndex
            >>> dates = pd.date_range('2023-01-01', periods=3)
            >>> df = pd.DataFrame({'value': [1, 2, 3]}, index=dates)
            >>> mixin = TimeSeriesTransformerMixin()
            >>> time_idx = mixin._validate_time_index(df)
            >>> len(time_idx)
            3
            >>>
            >>> # Example 2: Using time column
            >>> df = pd.DataFrame({
            ...     'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            ...     'value': [1, 2, 3]
            ... })
            >>> time_idx = mixin._validate_time_index(df, time_col='date')
            >>> isinstance(time_idx, pd.DatetimeIndex)
            True
        """
        if time_col and time_col in X.columns:
            # Utilisation de la colonne temporelle spécifiée
            time_index = pd.to_datetime(X[time_col])
        elif isinstance(X.index, pd.DatetimeIndex):
            # L'index est déjà un DatetimeIndex
            time_index = X.index
        else:
            # Tentative de conversion de l'index
            try:
                time_index = pd.to_datetime(X.index)
            except:
                raise ValueError(
                    "Cannot determine time index. Please provide a datetime index "
                    "or specify time_col parameter."
                )
        
        return time_index
    

# Transformer pour les données de panel
class PanelTimeSeriesTransformer(BaseEstimator, TransformerMixin, TimeSeriesTransformerMixin, ABC):
    """Abstract base class for panel time series transformers.

    This class provides a template for creating sklearn-compatible transformers
    that handle both univariate time series and panel data (multiple entities
    observed over time). It includes comprehensive validation, sorting, and
    structure conversion capabilities.

    Parameters:
        time_col (str): Name of the time column. Default is 'date'.
        panel_cols (Optional[List[str]]): Columns identifying panel dimensions
            (e.g., ['country', 'sector']). If None, treats data as univariate
            time series.
        validate_input (bool): Whether to validate input data structure and
            temporal ordering. Default is True.
        strict_validation (bool): If True, raises errors on validation failures;
            if False, emits warnings. Default is True.
        auto_sort (bool): If True, automatically sorts unsorted data by panel
            and time columns. Default is False.
        convert_cols_to_index (bool): If True, converts time_col and panel_cols
            to index and stores metadata for restoration. Default is False.

    Examples:
        >>> import pandas as pd
        >>> from abc import abstractmethod
        >>>
        >>> # Example 1: Simple time series transformer
        >>> class SimpleScaler(PanelTimeSeriesTransformer):
        ...     def _fit(self, X, y=None):
        ...         self.mean_ = X.select_dtypes(include='number').mean()
        ...
        ...     def _transform(self, X):
        ...         numeric_cols = X.select_dtypes(include='number').columns
        ...         X_transformed = X.copy()
        ...         X_transformed[numeric_cols] = X[numeric_cols] - self.mean_
        ...         return X_transformed
        >>>
        >>> dates = pd.date_range('2023-01-01', periods=5)
        >>> df = pd.DataFrame({'date': dates, 'value': [1, 2, 3, 4, 5]})
        >>>
        >>> scaler = SimpleScaler(time_col='date')
        >>> scaler.fit(df)
        SimpleScaler(...)
        >>>
        >>> # Example 2: Panel data transformer
        >>> panel_df = pd.DataFrame({
        ...     'entity': ['A', 'A', 'A', 'B', 'B', 'B'],
        ...     'date': pd.date_range('2023-01-01', periods=3).tolist() * 2,
        ...     'value': [1, 2, 3, 4, 5, 6]
        ... })
        >>>
        >>> panel_scaler = SimpleScaler(
        ...     time_col='date',
        ...     panel_cols=['entity'],
        ...     validate_input=True
        ... )
        >>> panel_scaler.fit(panel_df)
        SimpleScaler(...)
    """

    # Initialisation
    def __init__(self,
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 validate_input: bool = True,
                 strict_validation: bool = True,
                 auto_sort: bool = False,
                 convert_cols_to_index: bool = False):
        """Initialize the transformer.

        Args:
            time_col: Name of the temporal column
            panel_cols: Columns identifying panel dimensions (entity identifiers)
            validate_input: Whether to activate input data validation
            strict_validation: If True, raises errors; if False, emits warnings
            auto_sort: If True, automatically sorts improperly ordered data
            convert_cols_to_index: If True, converts time_col and panel_cols to index
                and stores metadata for later restoration

        Examples:
            >>> # Example 1: Basic time series transformer
            >>> transformer = PanelTimeSeriesTransformer(
            ...     time_col='date',
            ...     validate_input=True
            ... )
            >>>
            >>> # Example 2: Panel transformer with auto-sorting
            >>> panel_transformer = PanelTimeSeriesTransformer(
            ...     time_col='date',
            ...     panel_cols=['country', 'sector'],
            ...     auto_sort=True,
            ...     strict_validation=False
            ... )
            >>>
            >>> # Example 3: Transformer with index conversion
            >>> index_transformer = PanelTimeSeriesTransformer(
            ...     time_col='date',
            ...     panel_cols=['entity_id'],
            ...     convert_cols_to_index=True
            ... )
        """
        # Initialisation des paramètres en attributs
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.validate_input = validate_input
        self.strict_validation = strict_validation
        self.auto_sort = auto_sort
        self.convert_cols_to_index = convert_cols_to_index

        # Attribut pour stocker les métadonnées de conversion
        self.conversion_metadata_ = None
    
    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PanelTimeSeriesTransformer':
        """Fit the transformer.

        Validates input data, determines panel structure, and calls the
        subclass-specific _fit() method for learning transformation parameters.

        Args:
            X: Input features as DataFrame with time and optional panel columns
            y: Target variable (optional, not used in most transformers)

        Returns:
            Self for method chaining (sklearn convention)

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=10)
            >>> df = pd.DataFrame({
            ...     'date': dates,
            ...     'value': range(10),
            ...     'feature': range(10, 20)
            ... })
            >>>
            >>> # Assuming MyTransformer inherits from PanelTimeSeriesTransformer
            >>> transformer = MyTransformer(time_col='date')
            >>> transformer.fit(df)
            MyTransformer(...)
            >>> transformer.is_panel_
            False
            >>> transformer.n_features_
            3
        """
        # Validation des données si demandée
        if self.validate_input:
            X = self._validate_input(X)
        
        # Détermination du type de données
        self.is_panel_ = self.panel_cols is not None and len(self.panel_cols) > 0
        
        # Stockage des métadonnées
        self.n_features_ = X.shape[1]
        self.feature_names_ = X.columns.tolist()
        
        # Vérification de la cohérence des individus du panel
        if self.is_panel_:
            is_consistent, issues = self._check_panel_consistency(X, self.panel_cols)
            if not is_consistent:
                warnings.warn(f"Panel structure issues: {'; '.join(issues)}")
        
        # Appel de la méthode spécifique à implémenter
        self._fit(X, y)
        
        return self
    
    # Méthode auxiliaire d'entraînement
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit implementation to be provided by subclasses.

        This method should implement the transformer-specific learning logic,
        such as computing statistics, learning parameters, or storing reference
        values needed for transformation.

        Args:
            X: Validated input features
            y: Target variable (optional)

        Examples:
            >>> # Example implementation in a subclass
            >>> class MeanNormalizer(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None):
            ...         # Learn mean values for normalization
            ...         numeric_cols = X.select_dtypes(include='number').columns
            ...         self.means_ = X[numeric_cols].mean()
            ...
            ...     def _transform(self, X):
            ...         X_transformed = X.copy()
            ...         numeric_cols = X.select_dtypes(include='number').columns
            ...         X_transformed[numeric_cols] = X[numeric_cols] - self.means_
            ...         return X_transformed
        """
        pass
    
    # Méthode de transformation
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Validates the transformer is fitted, validates input data if enabled,
        and applies the transformation defined in the subclass's _transform() method.

        Args:
            X: Input data to transform

        Returns:
            Transformed data with same structure as input

        Raises:
            NotFittedError: If transformer has not been fitted yet

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5)
            >>> df_train = pd.DataFrame({
            ...     'date': dates,
            ...     'value': [1, 2, 3, 4, 5]
            ... })
            >>> df_test = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-06', periods=3),
            ...     'value': [6, 7, 8]
            ... })
            >>>
            >>> transformer = MyTransformer(time_col='date')
            >>> transformer.fit(df_train)
            >>> transformed = transformer.transform(df_test)
        """
        # Vérification que le transformer est ajusté
        check_is_fitted(self)
        
        # Validation des données
        if self.validate_input:
            X = self._validate_input(X)
        
        # Appel de la méthode spécifique
        return self._transform(X)
    
    # Méthode auxiliaire de transformation
    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform implementation to be provided by subclasses.

        This method should implement the core transformation logic using
        parameters learned during fit().

        Args:
            X: Validated input data

        Returns:
            Transformed data maintaining original structure

        Examples:
            >>> # Example implementation in a subclass
            >>> class LogTransformer(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None):
            ...         # Store which columns are numeric
            ...         self.numeric_cols_ = X.select_dtypes(include='number').columns.tolist()
            ...
            ...     def _transform(self, X):
            ...         import numpy as np
            ...         X_transformed = X.copy()
            ...         X_transformed[self.numeric_cols_] = np.log1p(X[self.numeric_cols_])
            ...         return X_transformed
        """
        pass
    
    # Méthode auxiliaire de validation des données
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data with comprehensive temporal checks.

        Performs comprehensive validation including:
        - Data type and column presence
        - Temporal validation (index or column)
        - Uniqueness of (entity, date) combinations
        - Global and intra-group temporal sorting (panel)
        - Contiguous grouping of entities (panel)

        Args:
            X: Input data to validate

        Returns:
            Validated DataFrame (potentially sorted if auto_sort=True)

        Raises:
            ValueError: If validation fails and strict_validation=True

        Examples:
            >>> import pandas as pd
            >>> # Example 1: Valid time series
            >>> dates = pd.date_range('2023-01-01', periods=5)
            >>> df = pd.DataFrame({'date': dates, 'value': [1, 2, 3, 4, 5]})
            >>> transformer = PanelTimeSeriesTransformer(time_col='date')
            >>> validated = transformer._validate_input(df)
            >>>
            >>> # Example 2: Panel data with auto-sorting
            >>> panel_df = pd.DataFrame({
            ...     'entity': ['B', 'A', 'B', 'A'],
            ...     'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
            ...     'value': [1, 2, 3, 4]
            ... })
            >>> transformer = PanelTimeSeriesTransformer(
            ...     time_col='date',
            ...     panel_cols=['entity'],
            ...     auto_sort=True
            ... )
            >>> validated = transformer._validate_input(panel_df)
            >>> # Data is now sorted by entity then date
        """
        # 1. Vérification type de base
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # 2. Vérification colonnes panel
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found: {missing_cols}")

        # 3. Vérification colonne/index temporel (méthode existante)
        try:
            self._validate_time_index(X, self.time_col)
        except ValueError as e:
            raise ValueError(f"Time validation failed: {str(e)}")

        # 4. Validation complète via validate_temporal_data
        # Si convert_cols_to_index=True, demander les métadonnées pour restauration
        try:
            if self.convert_cols_to_index:
                # Demande des métadonnées pour restauration ultérieure
                X_validated, metadata = validate_temporal_data(
                    data=X,
                    time_col=self.time_col if self.time_col in X.columns else None,
                    panel_cols=self.panel_cols,
                    strict=self.strict_validation,
                    sort_data=self.auto_sort,
                    return_metadata=True
                )
                # Stockage des métadonnées pour inverse_transform
                self.conversion_metadata_ = metadata
            else:
                # Validation sans métadonnées
                X_validated = validate_temporal_data(
                    data=X,
                    time_col=self.time_col if self.time_col in X.columns else None,
                    panel_cols=self.panel_cols,
                    strict=self.strict_validation,
                    sort_data=self.auto_sort,
                    return_metadata=False
                )
        except ValueError as e:
            if self.strict_validation:
                raise ValueError(f"Temporal data validation failed: {str(e)}")
            else:
                warnings.warn(f"Temporal data validation warning: {str(e)}")
                X_validated = X

        # 5. Validations spécifiques panel
        if self.panel_cols:
            # Vérification groupement des entités
            if not validate_entities_grouped(X_validated, panel_cols=self.panel_cols):
                msg = (
                    "Panel entities are not contiguous. "
                    "Consider sorting by panel_cols or enabling auto_sort=True"
                )
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

            # Vérification tri intra-groupe
            if not validate_sorted_within_groups(
                X_validated,
                panel_cols=self.panel_cols,
                time_col=self.time_col if self.time_col in X.columns else None
            ):
                msg = (
                    "Time series not sorted within panel groups. "
                    "Consider enabling auto_sort=True"
                )
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

        # 6. Validation série temporelle simple (non-panel)
        elif isinstance(X_validated.index, pd.DatetimeIndex):
            if not X_validated.index.is_monotonic_increasing:
                msg = "Time series index is not sorted"
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

        return X_validated
    
    # Fonction auxiliaire de vérification de la consistence des éléments du panel
    def _check_panel_consistency(self,
                               X: pd.DataFrame,
                               panel_cols: List[str]) -> Tuple[bool, List[str]]:
        """Check consistency of panel structure.

        Verifies that panel identifier columns are present and contain no
        missing values, which would compromise panel integrity.

        Args:
            X: Input data with panel structure
            panel_cols: Panel identifier columns to check

        Returns:
            Tuple of (is_consistent, list_of_issues) where is_consistent is
            True if no issues found, and list_of_issues contains error messages

        Examples:
            >>> import pandas as pd
            >>> # Example 1: Consistent panel
            >>> df = pd.DataFrame({
            ...     'entity': ['A', 'A', 'B', 'B'],
            ...     'date': pd.date_range('2023-01-01', periods=2).tolist() * 2,
            ...     'value': [1, 2, 3, 4]
            ... })
            >>> transformer = PanelTimeSeriesTransformer(panel_cols=['entity'])
            >>> is_consistent, issues = transformer._check_panel_consistency(df, ['entity'])
            >>> is_consistent
            True
            >>>
            >>> # Example 2: Inconsistent panel with missing values
            >>> df_bad = pd.DataFrame({
            ...     'entity': ['A', None, 'B', 'B'],
            ...     'value': [1, 2, 3, 4]
            ... })
            >>> is_consistent, issues = transformer._check_panel_consistency(df_bad, ['entity'])
            >>> is_consistent
            False
            >>> 'Missing values in panel identifier: entity' in issues
            True
        """
        # Initialisation de la liste des problèmes
        issues = []
        
        # Vérification de la présence des colonnes panel
        missing_cols = set(panel_cols) - set(X.columns)
        if missing_cols:
            issues.append(f"Missing panel columns: {missing_cols}")
            return False, issues
        
        # Vérification des valeurs manquantes dans les identifiants
        for col in panel_cols:
            if X[col].isnull().any():
                issues.append(f"Missing values in panel identifier: {col}")
        
        return len(issues) == 0, issues

# Mixin d'inversion des transformations
class ReversibleTransformerMixin:
    """Mixin for transformers that support inverse transformation.

    Provides template and utilities for implementing reversible transformations,
    allowing transformed data to be converted back to its original form.
    This is useful for transformations like scaling, differencing, or
    log transforms that need to be reversed for interpretation.

    Examples:
        >>> import pandas as pd
        >>> class ReversibleScaler(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
        ...     def _fit(self, X, y=None):
        ...         self.mean_ = X.select_dtypes(include='number').mean()
        ...
        ...     def _transform(self, X):
        ...         X_transformed = X.copy()
        ...         numeric_cols = X.select_dtypes(include='number').columns
        ...         X_transformed[numeric_cols] = X[numeric_cols] - self.mean_
        ...         self._store_transformation_info(X, X_transformed)
        ...         return X_transformed
        ...
        ...     def inverse_transform(self, X):
        ...         X_original = X.copy()
        ...         numeric_cols = X.select_dtypes(include='number').columns
        ...         X_original[numeric_cols] = X[numeric_cols] + self.mean_
        ...         return self._restore_structure_if_converted(X_original)
        >>>
        >>> dates = pd.date_range('2023-01-01', periods=3)
        >>> df = pd.DataFrame({'date': dates, 'value': [10, 20, 30]})
        >>> scaler = ReversibleScaler(time_col='date')
        >>> scaler.fit(df)
        >>> transformed = scaler.transform(df)
        >>> restored = scaler.inverse_transform(transformed)
    """
    
    # Méthode d'inversion de la transformation
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation.

        Converts transformed data back to its original form using stored
        transformation parameters. This method must be implemented by subclasses.

        Args:
            X: Transformed data to reverse

        Returns:
            Data in original format before transformation

        Examples:
            >>> # Example implementation in a subclass
            >>> class DifferenceTransformer(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
            ...     def _fit(self, X, y=None):
            ...         pass  # No fitting needed
            ...
            ...     def _transform(self, X):
            ...         X_diff = X.copy()
            ...         self.first_values_ = X.iloc[0].copy()
            ...         X_diff.iloc[1:] = X.diff().iloc[1:]
            ...         return X_diff
            ...
            ...     def inverse_transform(self, X):
            ...         X_original = X.copy()
            ...         X_original.iloc[0] = self.first_values_
            ...         X_original = X_original.cumsum()
            ...         return self._restore_structure_if_converted(X_original)
        """
        pass
    
    # Méthode de stockage des méta-données (index, colonnes, dimensions) des jeux de données initial et transformé
    def _store_transformation_info(self, X: pd.DataFrame, X_transformed: pd.DataFrame) -> None:
        """Store information needed for inverse transformation.

        Saves metadata about the original and transformed data structures,
        including shapes, column names, and index information. This metadata
        is used by inverse_transform to restore the original structure.

        Args:
            X: Original data before transformation
            X_transformed: Data after transformation

        Examples:
            >>> import pandas as pd
            >>> # Inside a transformer's _transform method
            >>> class MyReversibleTransformer(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
            ...     def _transform(self, X):
            ...         X_transformed = X * 2  # Simple transformation
            ...         # Store metadata for later reversal
            ...         self._store_transformation_info(X, X_transformed)
            ...         return X_transformed
            >>>
            >>> dates = pd.date_range('2023-01-01', periods=3)
            >>> df = pd.DataFrame({'value': [1, 2, 3]}, index=dates)
            >>> transformer = MyReversibleTransformer()
            >>> # After calling transform, metadata is stored
            >>> # transformer.original_shape_, transformer.transformed_shape_, etc.
        """
        # Stockage des informations de forme
        self.original_shape_ = X.shape
        self.transformed_shape_ = X_transformed.shape

        # Stockage des colonnes
        self.original_columns_ = X.columns.tolist()
        self.transformed_columns_ = X_transformed.columns.tolist()

        # Stockage de l'index si différent
        if not X.index.equals(X_transformed.index):
            self.original_index_ = X.index
            self.transformed_index_ = X_transformed.index

    # Fonction de restauration de la structure originale des données
    def _restore_structure_if_converted(self, X: pd.DataFrame) -> pd.DataFrame:
        """Restore original structure if conversion was applied.

        Restores the original data structure (columns instead of index)
        if conversion was performed via convert_cols_to_index=True during
        validation. This ensures inverse_transform returns data in the
        same format as the original input.

        Args:
            X: Transformed data with modified index

        Returns:
            Data with original structure restored

        Examples:
            >>> import pandas as pd
            >>> # Example with index conversion
            >>> df = pd.DataFrame({
            ...     'entity': ['A', 'A', 'B', 'B'],
            ...     'date': pd.date_range('2023-01-01', periods=2).tolist() * 2,
            ...     'value': [1, 2, 3, 4]
            ... })
            >>>
            >>> transformer = MyReversibleTransformer(
            ...     time_col='date',
            ...     panel_cols=['entity'],
            ...     convert_cols_to_index=True
            ... )
            >>> transformer.fit(df)
            >>>
            >>> # After transform, index = (entity, date)
            >>> X_transformed = transformer.transform(df)
            >>>
            >>> # In inverse_transform, structure is restored
            >>> X_restored = transformer.inverse_transform(X_transformed)
            >>> # X_restored now has 'entity' and 'date' as columns again
            >>> 'entity' in X_restored.columns and 'date' in X_restored.columns
            True
        """
        # Vérification que les métadonnées existent
        if hasattr(self, 'conversion_metadata_') and self.conversion_metadata_ is not None:
            # Restauration de la structure originale
            return restore_original_structure(X, self.conversion_metadata_)
        # Pas de conversion appliquée : retour inchangé
        return X
