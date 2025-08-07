# Importation des modules
# Modules de base
import warnings

# Importations locales
from .base import OutOfSampleSplit, InSampleSplit

# Classe de base pour la validation croisée out-of-sample sur séries temporelles
class TSOutOfSampleSplit(OutOfSampleSplit):
    """Time series out-of-sample cross-validation split.
    
    This class extends OutOfSampleSplit for time series data where temporal ordering
    is crucial and no grouping by entities is needed. The split respects chronological
    order and prevents data leakage by ensuring training data comes before test data.
    
    Key characteristics:
    - Maintains temporal ordering in splits
    - Applies gap between training and test sets to avoid leakage
    - Supports rolling window validation with configurable train size
    - Groups parameter is ignored (time series = single entity)
    
    Args:
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        test_indices (list, optional): Specific test periods to use. Can be:
            - Dates/timestamps as strings or pd.Timestamp objects
            - Integer positions in the time series
            Defaults to None (uses equally spaced splits).
        max_train_size (int, optional): Maximum size of training window.
            If None, uses expanding window. Defaults to None.
        test_size (int, optional): Number of time periods per test set.
            Defaults to None (calculated automatically).
        gap (int, optional): Number of periods between training and test sets
            to prevent data leakage. Defaults to 0.
    
    Examples:
        >>> # Create time series data
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> X = pd.DataFrame({'value': np.random.randn(100)}, index=dates)
        
        >>> # Rolling window out-of-sample validation
        >>> splitter = TSOutOfSampleSplit(n_splits=5, test_size=10, gap=1)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        ...     # Train on past, test on future with 1-day gap
        
        >>> # Specific test dates
        >>> test_dates = ['2020-02-01', '2020-03-01', '2020-04-01']
        >>> splitter = TSOutOfSampleSplit(test_indices=test_dates, test_size=7)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     # Test on specific weeks
        ...     pass
    """
    
    # Méthode principale de séparation des données de séries temporelles
    def split(self, X, y=None, groups=None):
        """Generate indices to split time series data into training and test sets.
        
        Args:
            X (array-like): Time series data. Can be pandas DataFrame/Series with
                DatetimeIndex for better date handling.
            y (array-like, optional): Target values with same index as X. Defaults to None.
            groups (array-like, optional): Ignored for time series. Will issue warning
                if provided. Defaults to None.
                
        Yields:
            tuple: (train_indices, test_indices) where train_indices come before
                test_indices chronologically, respecting the gap parameter.
                
        Warnings:
            UserWarning: If groups parameter is provided (will be ignored)
            
        Examples:
            >>> # Simple time series split
            >>> X = pd.Series(range(100), index=pd.date_range('2020-01-01', periods=100))
            >>> splitter = TSOutOfSampleSplit(n_splits=3, test_size=10)
            >>> splits = list(splitter.split(X))
            >>> len(splits)  # 3 splits
            3
        """
        # Avertissement si des groupes sont spécifiés car ils sont ignorés
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}. "
                f"Time series splits treat data as a single temporal sequence.",
                UserWarning,
            )
        
        # Appel de la méthode parente avec groups=None pour le comportement série temporelle
        return super().split(X, y, groups=None)


# Classe de base pour la validation croisée in-sample sur séries temporelles
class TSInSampleSplit(InSampleSplit):
    """Time series in-sample cross-validation split.
    
    This class extends InSampleSplit for time series data where the training set
    includes the test period, allowing models to learn from future information.
    This is useful for historical backtesting and performance evaluation.
    
    Key characteristics:
    - Training data includes the test period (in-sample validation)
    - Maintains temporal structure of the data
    - Useful for historical model evaluation and calibration
    - Groups parameter is ignored (time series = single temporal sequence)
    
    Args:
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        test_indices (list, optional): Specific test periods to use. Can be:
            - Dates/timestamps as strings or pd.Timestamp objects
            - Integer positions in the time series
            Defaults to None (uses last portion of data).
        max_train_size (int, optional): Maximum size of training window.
            Training extends from start of data up to and including test period.
            Defaults to None (uses all available data).
        test_size (int, optional): Number of time periods per test set.
            Defaults to None (calculated automatically).
    
    Examples:
        >>> # Create time series data
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> X = pd.DataFrame({'value': np.random.randn(100)}, index=dates)
        
        >>> # In-sample validation on last portion
        >>> splitter = TSInSampleSplit(test_size=10)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        ...     # Training includes test period for historical analysis
        
        >>> # Specific test period with controlled training window
        >>> test_dates = ['2020-03-01']
        >>> splitter = TSInSampleSplit(test_indices=test_dates, test_size=7, max_train_size=50)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     # Test on specific week, train on preceding 50 days + test week
        ...     pass
    """
    
    # Méthode principale de séparation des données de séries temporelles (in-sample)
    def split(self, X, y=None, groups=None):
        """Generate indices to split time series data with in-sample validation.
        
        Args:
            X (array-like): Time series data. Can be pandas DataFrame/Series with
                DatetimeIndex for better date handling.
            y (array-like, optional): Target values with same index as X. Defaults to None.
            groups (array-like, optional): Ignored for time series. Will issue warning
                if provided. Defaults to None.
                
        Yields:
            tuple: (train_indices, test_indices) where train_indices include the
                test period, enabling in-sample validation.
                
        Warnings:
            UserWarning: If groups parameter is provided (will be ignored)
            
        Examples:
            >>> # In-sample time series validation
            >>> X = pd.Series(range(100), index=pd.date_range('2020-01-01', periods=100))
            >>> splitter = TSInSampleSplit(test_size=10)
            >>> for train_idx, test_idx in splitter.split(X):
            ...     # Training data includes test period
            ...     assert max(test_idx) < len(train_idx)  # Test period is in training
        """
        # Avertissement si des groupes sont spécifiés car ils sont ignorés
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}. "
                f"Time series splits treat data as a single temporal sequence.",
                UserWarning,
            )
        
        # Appel de la méthode parente avec groups=None pour le comportement série temporelle
        return super().split(X, y, groups=None)