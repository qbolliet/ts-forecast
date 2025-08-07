"""Base classes for time series transformers.

This module provides abstract base classes and mixins for creating
sklearn-compatible time series transformers.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Tuple, List, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings


class TimeSeriesTransformerMixin:
    """Mixin class for time series transformers.
    
    Provides common functionality for time series data handling.
    """
    
    def _validate_time_index(self, X: pd.DataFrame, time_col: Optional[str] = None) -> pd.DatetimeIndex:
        """Validate and extract time index from data.
        
        Args:
            X: Input data
            time_col: Name of time column
            
        Returns:
            DatetimeIndex
            
        Raises:
            ValueError: If time index cannot be determined
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
    
    def _get_time_features(self, X: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
        """Extract time-based features from data.
        
        Args:
            X: Input data
            time_col: Name of time column
            
        Returns:
            DataFrame with time features
        """
        time_index = self._validate_time_index(X, time_col)
        
        features = pd.DataFrame(index=X.index)
        features['year'] = time_index.year
        features['month'] = time_index.month
        features['day'] = time_index.day
        features['dayofweek'] = time_index.dayofweek
        features['quarter'] = time_index.quarter
        
        return features
    
    def _check_panel_consistency(self, 
                               X: pd.DataFrame, 
                               panel_cols: List[str]) -> Tuple[bool, List[str]]:
        """Check consistency of panel structure.
        
        Args:
            X: Input data
            panel_cols: Panel identifier columns
            
        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        issues = []
        
        # Vérification de la présence des colonnes panel
        missing_cols = set(panel_cols) - set(X.columns)
        if missing_cols:
            issues.append(f"Missing panel columns: {missing_cols}")
            return False, issues
        
        # Vérification de l'équilibre du panel
        panel_counts = X.groupby(panel_cols).size()
        if panel_counts.std() > panel_counts.mean() * 0.1:  # Plus de 10% de variation
            issues.append("Unbalanced panel detected")
        
        # Vérification des valeurs manquantes dans les identifiants
        for col in panel_cols:
            if X[col].isnull().any():
                issues.append(f"Missing values in panel identifier: {col}")
        
        return len(issues) == 0, issues


class PanelTimeSeriesTransformer(BaseEstimator, TransformerMixin, TimeSeriesTransformerMixin, ABC):
    """Abstract base class for panel time series transformers.
    
    This class provides a template for creating transformers that handle
    both univariate time series and panel data.
    
    Parameters:
        time_col (str): Name of the time column
        panel_cols (Optional[List[str]]): Columns identifying panel dimensions
        validate_input (bool): Whether to validate input data
    """
    
    def __init__(self, 
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 validate_input: bool = True):
        """Initialize the transformer."""
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.validate_input = validate_input
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PanelTimeSeriesTransformer':
        """Fit the transformer.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        # Validation des données si demandée
        if self.validate_input:
            X = self._validate_input(X)
        
        # Détermination du type de données
        self.is_panel_ = self.panel_cols is not None and len(self.panel_cols) > 0
        
        # Stockage des métadonnées
        self.n_features_ = X.shape[1]
        self.feature_names_ = X.columns.tolist()
        
        # Vérification de la cohérence du panel
        if self.is_panel_:
            is_consistent, issues = self._check_panel_consistency(X, self.panel_cols)
            if not is_consistent:
                warnings.warn(f"Panel structure issues: {'; '.join(issues)}")
        
        # Appel de la méthode spécifique à implémenter
        self._fit(X, y)
        
        return self
    
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit implementation to be provided by subclasses.
        
        Args:
            X: Input features
            y: Target variable (optional)
        """
        pass
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        # Vérification que le transformer est ajusté
        check_is_fitted(self)
        
        # Validation des données
        if self.validate_input:
            X = self._validate_input(X)
        
        # Appel de la méthode spécifique
        return self._transform(X)
    
    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform implementation to be provided by subclasses.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        pass
    
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated data
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Vérification des colonnes panel si spécifiées
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found: {missing_cols}")
        
        # Vérification de la colonne temporelle ou index
        try:
            self._validate_time_index(X, self.time_col)
        except ValueError as e:
            raise ValueError(f"Time validation failed: {str(e)}")
        
        return X


class StatefulTransformerMixin:
    """Mixin for transformers that maintain state between fit and transform.
    
    Provides utilities for saving and loading transformer state.
    """
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the transformer.
        
        Returns:
            Dictionary containing transformer state
        """
        state = {}
        
        # Sauvegarde des attributs qui se terminent par '_'
        for attr_name in dir(self):
            if attr_name.endswith('_') and not attr_name.startswith('_'):
                attr_value = getattr(self, attr_name)
                
                # Sérialisation basique pour les types courants
                if isinstance(attr_value, (dict, list, tuple, str, int, float, bool, type(None))):
                    state[attr_name] = attr_value
                elif isinstance(attr_value, pd.DataFrame):
                    state[attr_name] = attr_value.to_dict('records')
                elif isinstance(attr_value, pd.Series):
                    state[attr_name] = attr_value.to_dict()
                elif isinstance(attr_value, np.ndarray):
                    state[attr_name] = attr_value.tolist()
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the transformer state.
        
        Args:
            state: Dictionary containing transformer state
        """
        for attr_name, attr_value in state.items():
            setattr(self, attr_name, attr_value)


class ReversibleTransformerMixin:
    """Mixin for transformers that support inverse transformation.
    
    Provides template for implementing reversible transformations.
    """
    
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation.
        
        Args:
            X: Transformed data
            
        Returns:
            Original data format
        """
        pass
    
    def _store_transformation_info(self, X: pd.DataFrame, X_transformed: pd.DataFrame) -> None:
        """Store information needed for inverse transformation.
        
        Args:
            X: Original data
            X_transformed: Transformed data
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


class WindowedTransformerMixin:
    """Mixin for transformers that use sliding windows.
    
    Provides utilities for window-based operations.
    """
    
    def _create_windows(self, 
                       X: pd.DataFrame, 
                       window_size: int,
                       step_size: int = 1,
                       min_periods: Optional[int] = None) -> List[pd.DataFrame]:
        """Create sliding windows from data.
        
        Args:
            X: Input data
            window_size: Size of each window
            step_size: Step between windows
            min_periods: Minimum observations in window
            
        Returns:
            List of DataFrame windows
        """
        if min_periods is None:
            min_periods = window_size
        
        windows = []
        
        for i in range(0, len(X) - window_size + 1, step_size):
            window = X.iloc[i:i + window_size]
            
            # Vérification du nombre minimal d'observations non-nulles
            if window.notna().sum().sum() >= min_periods:
                windows.append(window)
        
        return windows
    
    def _apply_to_windows(self,
                         X: pd.DataFrame,
                         func: callable,
                         window_size: int,
                         **kwargs) -> pd.DataFrame:
        """Apply a function to sliding windows.
        
        Args:
            X: Input data
            func: Function to apply to each window
            window_size: Size of window
            **kwargs: Additional arguments for window creation
            
        Returns:
            DataFrame with results
        """
        windows = self._create_windows(X, window_size, **kwargs)
        
        results = []
        for window in windows:
            result = func(window)
            results.append(result)
        
        # Concaténation des résultats
        if results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, axis=0)
        elif results and isinstance(results[0], pd.Series):
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(results, index=range(len(results)))


class CachedTransformerMixin:
    """Mixin for transformers with caching capabilities.
    
    Provides caching functionality to avoid redundant computations.
    """
    
    def __init__(self, cache_size: int = 100):
        """Initialize the cache.
        
        Args:
            cache_size: Maximum number of cached results
        """
        self.cache_size = cache_size
        self._cache = {}
        self._cache_keys = []
    
    def _get_cache_key(self, X: pd.DataFrame) -> str:
        """Generate cache key for data.
        
        Args:
            X: Input data
            
        Returns:
            Cache key
        """
        # Utilisation d'un hash simple basé sur la forme et quelques valeurs
        shape_str = f"{X.shape}"
        sample_values = X.iloc[:5, :5].values.flatten() if X.shape[0] >= 5 else X.values.flatten()
        sample_str = str(sample_values[:10])
        
        return f"{shape_str}_{sample_str}_{X.columns.tolist()}"
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get result from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None
        """
        return self._cache.get(key)
    
    def _add_to_cache(self, key: str, result: Any) -> None:
        """Add result to cache.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        # Gestion de la taille du cache
        if key not in self._cache and len(self._cache) >= self.cache_size:
            # Suppression du plus ancien élément (FIFO)
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = result
        if key not in self._cache_keys:
            self._cache_keys.append(key)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_keys.clear()