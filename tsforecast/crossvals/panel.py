# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
import warnings

# Importations locales
from ..panel.utils import is_panel_data
from .base import InSampleSplit, OutOfSampleSplit, _entity_labels, _labels_array, _precompute_group_mappings

# Sklearn
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


# Fonction auxiliaire d'extraction des groupes depuis les données de panel
def _extract_groups_from_panel(X):
    """Extract the entity of each row of panel data.

    The entity is read from every index level but the last (the date): the
    value of the entity level when there is one, the tuple of the entity
    levels when there are several.

    Args:
        X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity..., time)

    Returns:
        np.ndarray: One-dimensional object array of the entity of each row

    Raises:
        ValueError: If X is not a pandas object with a MultiIndex of at least
            two levels

    Examples:
        >>> idx = pd.MultiIndex.from_product([['A', 'B'], pd.date_range('2020-01-01', periods=2, freq='D')])
        >>> _extract_groups_from_panel(pd.DataFrame({'value': range(4)}, index=idx)).tolist()
        ['A', 'A', 'B', 'B']
        >>> idx = pd.MultiIndex.from_product([['FR'], ['a', 'b'], [2020]])
        >>> _extract_groups_from_panel(pd.Series(range(2), index=idx)).tolist()
        [('FR', 'a'), ('FR', 'b')]
    """
    # Vérification qu'il s'agit d'un objet pandas
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        raise ValueError("Panel data requires pandas DataFrame/Series with MultiIndex")

    # Vérification de la structure de panel : MultiIndex (entité..., date)
    if not is_panel_data(X):
        raise ValueError("Panel data requires MultiIndex (entity, time)")

    # Entité de chaque ligne : niveaux d'entité (tous sauf le dernier)
    return _entity_labels(X)

# Classe de base pour la validation croisée out-of-sample sur données de panel
class PanelOutOfSampleSplit(OutOfSampleSplit):
    """Panel data out-of-sample cross-validation split.
    
    This class extends OutOfSampleSplit to handle panel data (multi-entity time series)
    where observations are grouped by entities. The gap and training window constraints
    are applied within each entity separately.
    
    Panel data is expected to have a MultiIndex with (entity, time) structure where:
    - First level: entity identifiers (e.g., company IDs, country codes)
    - Second level: time periods (e.g., dates, timestamps)
    
    Args:
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        test_indices (list, optional): Specific test periods to use. Can be:
            - Dates/timestamps for all entities
            - Tuples of (entity, date) for specific entity-date combinations
            Defaults to None.
        max_train_size (int, optional): Maximum training window size per entity.
            Defaults to None (no limit).
        test_size (int, optional): Number of periods per test set. Defaults to None.
        gap (int, optional): Number of periods between training and test sets
            to avoid data leakage. Defaults to 0.
    
    Examples:
        >>> # Create panel data
        >>> entities = ['AAPL', 'GOOGL', 'MSFT']
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'price': np.random.randn(300)}, index=idx)
        
        >>> # Out-of-sample split with specific test dates
        >>> test_dates = ['2020-03-01', '2020-03-15', '2020-04-01']
        >>> splitter = PanelOutOfSampleSplit(test_indices=test_dates, test_size=5, gap=1)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """
    
    # Méthode principale de séparation des données de panel
    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test sets.
        
        Args:
            X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity, time)
            y (array-like, optional): Target values with same index as X. Defaults to None.
            groups (array-like, optional): Group labels. If None, extracts from panel structure.
                Defaults to None.
                
        Yields:
            tuple: (train_indices, test_indices) for each split
            
        Raises:
            ValueError: If X doesn't have required MultiIndex structure
        """
        # Extraction des groupes depuis la structure du panel si non fournis
        if groups is None:
            groups = _extract_groups_from_panel(X)
        
        # Appel de la méthode parente avec les groupes pour le comportement panel
        return super().split(X, y, groups)

# Classe de base pour la validation croisée in-sample sur données de panel
class PanelInSampleSplit(InSampleSplit):
    """Panel data in-sample cross-validation split.
    
    This class extends InSampleSplit to handle panel data (multi-entity time series)
    where the training set includes the test period, allowing models to learn from
    future information. This is useful for evaluating historical model performance.
    
    Panel data is expected to have a MultiIndex with (entity, time) structure where:
    - First level: entity identifiers (e.g., company IDs, country codes)
    - Second level: time periods (e.g., dates, timestamps)
    
    Args:
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        test_indices (list, optional): Specific test periods to use. Can be:
            - Dates/timestamps for all entities
            - Tuples of (entity, date) for specific entity-date combinations
            Defaults to None.
        max_train_size (int, optional): Maximum training window size per entity.
            Defaults to None (no limit).
        test_size (int, optional): Number of periods per test set. Defaults to None.
    
    Examples:
        >>> # Create panel data
        >>> entities = ['AAPL', 'GOOGL', 'MSFT']
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'price': np.random.randn(300)}, index=idx)
        
        >>> # In-sample split - training includes test period
        >>> test_dates = ['2020-03-01']
        >>> splitter = PanelInSampleSplit(test_indices=test_dates, test_size=5)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     # Training data includes test period
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """
    
    # Méthode principale de séparation des données de panel
    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test sets (in-sample).
        
        Args:
            X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity, time)
            y (array-like, optional): Target values with same index as X. Defaults to None.
            groups (array-like, optional): Group labels. If None, extracts from panel structure.
                Defaults to None.
                
        Yields:
            tuple: (train_indices, test_indices) where training includes test period
            
        Raises:
            ValueError: If X doesn't have required MultiIndex structure
        """
        # Extraction des groupes depuis la structure du panel si non fournis
        if groups is None:
            groups = _extract_groups_from_panel(X)
        
        # Appel de la méthode parente avec les groupes pour le comportement panel
        return super().split(X, y, groups)



# Classe de validation croisée out-of-sample par entité
class PanelOutOfSampleSplitPerEntity(PanelOutOfSampleSplit):
    """Panel out-of-sample split that yields train/test indices separately for each entity.
    
    This class extends PanelOutOfSampleSplit to provide entity-specific splits,
    returning a dictionary mapping each entity to its (train_indices, test_indices).
    This is useful when you need to analyze or process each entity separately.
    
    The returned structure allows for:
    - Entity-specific model training and evaluation
    - Handling entities with different data availability
    - Parallel processing by entity
    
    Examples:
        >>> # Create panel data
        >>> entities = ['AAPL', 'GOOGL', 'MSFT']
        >>> dates = pd.date_range('2020-01-01', periods=50, freq='D')
        >>> idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'price': np.random.randn(150)}, index=idx)
        
        >>> # Split per entity
        >>> splitter = PanelOutOfSampleSplitPerEntity(test_size=5, gap=1)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     entity_train = X.iloc[train_idx]
        ...     entity_test = X.iloc[test_idx]
        ...     # Process each entity separately
    """
    
    # Méthode de séparation des données avec indices groupés par entité
    def split(self, X, y=None, groups=None):
        """Generate train/test indices separately for each entity.
        
        Args:
            X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity, time)
            y (array-like, optional): Target values with same index as X. Defaults to None.
            groups (array-like, optional): Group labels. If None, extracts from panel structure.
                Defaults to None.
                
        Yields:
            tuple: (train_indices, test_indices) for each entity that has test data
                in the current split. Entities are processed sequentially.
        """
        # Extraction des groupes si non fournis
        if groups is None:
            groups = _extract_groups_from_panel(X)
        
        # Positions des lignes de chaque entité (entités scalaires ou tuples)
        group_mappings = _precompute_group_mappings(_labels_array(groups))

        # Génération des splits avec regroupement par entité
        for train_indices, test_indices in super().split(X, y, groups):
            # Traitement entité par entité
            for group, group_indices in group_mappings.items():
                
                # Extraction des indices d'entraînement et de test pour cette entité
                entity_train = train_indices[np.isin(train_indices, group_indices)]
                entity_test = test_indices[np.isin(test_indices, group_indices)]
                
                # Inclusion seulement si l'entité a des données de test
                if len(entity_test) > 0:
                    yield (entity_train, entity_test)


# Classe de validation croisée in-sample par entité
class PanelInSampleSplitPerEntity(PanelInSampleSplit):
    """Panel in-sample split that yields train/test indices separately for each entity.
    
    This class extends PanelInSampleSplit to provide entity-specific splits where
    the training set includes the test period. Returns a dictionary mapping each
    entity to its (train_indices, test_indices).
    
    This is particularly useful for:
    - Historical backtesting where future information is available
    - Entity-specific model calibration and evaluation
    - Analyzing model performance across different entities
    
    Examples:
        >>> # Create panel data
        >>> entities = ['AAPL', 'GOOGL', 'MSFT']
        >>> dates = pd.date_range('2020-01-01', periods=50, freq='D')
        >>> idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'price': np.random.randn(150)}, index=idx)
        
        >>> # In-sample split per entity
        >>> test_dates = ['2020-02-01']
        >>> splitter = PanelInSampleSplitPerEntity(test_indices=test_dates, test_size=5)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     # Training includes test period for historical analysis
        ...     entity_train = X.iloc[train_idx]  # Includes test period
        ...     entity_test = X.iloc[test_idx]
    """
    
    # Méthode de séparation des données avec indices groupés par entité (in-sample)
    def split(self, X, y=None, groups=None):
        """Generate train/test indices separately for each entity (in-sample).
        
        Args:
            X (pd.DataFrame or pd.Series): Panel data with MultiIndex (entity, time)
            y (array-like, optional): Target values with same index as X. Defaults to None.
            groups (array-like, optional): Group labels. If None, extracts from panel structure.
                Defaults to None.
                
        Yields:
            tuple: (train_indices, test_indices) for each entity that has test data
                in the current split. Training indices include the test period for
                in-sample validation. Entities are processed sequentially.
        """
        # Extraction des groupes si non fournis
        if groups is None:
            groups = _extract_groups_from_panel(X)
        
        # Positions des lignes de chaque entité (entités scalaires ou tuples)
        group_mappings = _precompute_group_mappings(_labels_array(groups))

        # Génération des splits avec regroupement par entité
        for train_indices, test_indices in super().split(X, y, groups):
            # Traitement entité par entité
            for group, group_indices in group_mappings.items():
                
                # Extraction des indices d'entraînement et de test pour cette entité
                entity_train = train_indices[np.isin(train_indices, group_indices)]
                entity_test = test_indices[np.isin(test_indices, group_indices)]
                
                # Inclusion seulement si l'entité a des données de test
                if len(entity_test) > 0:
                    yield (entity_train, entity_test)