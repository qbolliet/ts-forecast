# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
import warnings

# Sklearn
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

# /!\ Faire du batch et du online learning en créant de nouvelles classes mais en harcodant juste le paramètre max_train_size d'une classe plus générale

# /!\ Opérer les distinctions suivantes pour "test_indices":
# - Si est différent de None, est prioritaire sur n_splits pour construire les périodes de tests
# - Si c'est une liste de numérique, sélectionne ces indices comme début de chaque période de test
# - Si c'est une liste de strings ou de date ou de tuple, recherche ces éléments "CONTENUS" dans un index du jeu de données et utiilise cela comme début de la période

def _resolve_panel_test_indices(test_indices, X):
    """Resolve test_indices to numeric positions for panel data."""
    if test_indices is None:
        return None
    
    if isinstance(test_indices, (list, np.ndarray)):
        if len(test_indices) == 0:
            return None
            
        # If numeric indices
        if isinstance(test_indices[0], (int, np.integer)):
            return np.array(test_indices)
        
        # If string, date, or tuple indices for panel data
        elif isinstance(test_indices[0], (str, pd.Timestamp, tuple)) or hasattr(test_indices[0], 'date'):
            try:
                positions = []
                for idx in test_indices:
                    if hasattr(X, 'index'):
                        if isinstance(idx, tuple):
                            # For MultiIndex panel data (entity, time)
                            if idx in X.index:
                                positions.append(X.index.get_loc(idx))
                            else:
                                raise ValueError(f"Index {idx} not found in data")
                        else:
                            # For single level index, search for matching values
                            if hasattr(X.index, 'get_level_values'):
                                # MultiIndex case - search in time level (level 1)
                                time_level = X.index.get_level_values(1)
                                matching_locs = []
                                for i, time_val in enumerate(time_level):
                                    if time_val == idx:
                                        matching_locs.append(i)
                                if matching_locs:
                                    positions.extend(matching_locs)
                                else:
                                    raise ValueError(f"Index {idx} not found in time level")
                            else:
                                # Single index case
                                if idx in X.index:
                                    positions.append(X.index.get_loc(idx))
                                else:
                                    raise ValueError(f"Index {idx} not found in data")
                    else:
                        raise ValueError("Cannot use string/date/tuple indices without pandas index")
                return np.array(positions)
            except Exception as e:
                raise ValueError(f"Error resolving test_indices: {e}")
    
    return np.array([test_indices]) if np.isscalar(test_indices) else np.array(test_indices)

def _get_panel_structure(X):
    """Extract panel structure from data."""
    if not hasattr(X, 'index'):
        raise ValueError("Panel data requires pandas DataFrame/Series with MultiIndex")
    
    if not hasattr(X.index, 'get_level_values'):
        raise ValueError("Panel data requires MultiIndex (entity, time)")
    
    entities = X.index.get_level_values(0).unique()
    return entities

# Classe de CrossVal "Out of sample"
class PanelOutOfSampleSplit(_BaseKFold) :

    # Initialisation
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None, gap=0):
        # Initialisation du parent
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Initialisation des attributs
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test set."""
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        
        # For panel data with gap and max_train_size, we need custom logic
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        entities = _get_panel_structure(X)
        
        for test_indices in self._iter_test_indices(X, y, groups):
            # Calculate train indices considering gap and max_train_size
            train_indices = []
            
            if self.gap > 0 or self.max_train_size is not None:
                # Custom train logic for panel data
                for entity in entities:
                    try:
                        entity_locs = X.index.get_locs([entity, slice(None)])
                        entity_start = entity_locs.start
                        entity_stop = entity_locs.stop
                        
                        # Find the earliest test index for this entity
                        entity_test_indices = test_indices[(test_indices >= entity_start) & (test_indices < entity_stop)]
                        
                        if len(entity_test_indices) > 0:
                            earliest_test = min(entity_test_indices)
                            train_end = earliest_test - self.gap
                            
                            if train_end > entity_start:
                                if self.max_train_size and self.max_train_size < (train_end - entity_start):
                                    train_start = train_end - self.max_train_size
                                    train_indices.extend(range(train_start, train_end))
                                else:
                                    train_indices.extend(range(entity_start, train_end))
                    except (KeyError, IndexError):
                        continue
                
                yield (np.array(train_indices), test_indices)
            else:
                # Use default sklearn behavior (complement of test indices)
                all_indices = np.arange(n_samples)
                train_indices = all_indices[~np.isin(all_indices, test_indices)]
                yield (train_indices, test_indices)

    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for panel data splits."""
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        
        # Get panel structure
        entities = _get_panel_structure(X)
        
        # Resolve test_indices
        resolved_test_indices = _resolve_panel_test_indices(self.test_indices, X)
        
        if resolved_test_indices is not None:
            # Use test_indices to determine splits
            test_size = self.test_size if self.test_size is not None else 1
            
            # Validate test_indices
            if np.any(resolved_test_indices >= n_samples):
                raise ValueError("test_indices contains indices beyond data length")
            if np.any(resolved_test_indices < 0):
                raise ValueError("test_indices contains negative indices")
            
            # Group test indices by entity if they represent time points
            if isinstance(self.test_indices[0], (str, pd.Timestamp)) and hasattr(self.test_indices[0], 'date'):
                # For each test time point, find indices for all entities
                for test_time in self.test_indices:
                    entity_test_indices = []
                    
                    for entity in entities:
                        try:
                            # Find the test index for this entity at this time
                            entity_test_idx = X.index.get_loc((entity, test_time))
                            entity_test_end = min(entity_test_idx + test_size, n_samples)
                            entity_test_indices.extend(range(entity_test_idx, entity_test_end))
                        except KeyError:
                            # Entity doesn't have data at this time point
                            continue
                    
                    if entity_test_indices:
                        yield np.array(entity_test_indices)
            else:
                # Handle direct numeric indices
                for test_start in sorted(resolved_test_indices):
                    test_end = min(test_start + test_size, n_samples)
                    yield np.arange(test_start, test_end)
        else:
            # Default behavior: use n_splits to create time-based splits
            n_splits = self.n_splits
            test_size = self.test_size if self.test_size is not None else 1
            
            # For panel data, create splits based on time periods
            # Assuming data is sorted by entity then time
            time_points = X.index.get_level_values(1).unique()
            n_time_points = len(time_points)
            
            if n_splits >= n_time_points:
                raise ValueError(f"Cannot have n_splits={n_splits} >= n_time_points={n_time_points}")
            
            # Create splits using the last n_splits time periods
            for i in range(n_splits):
                test_time_idx = n_time_points - n_splits + i
                test_time = time_points[test_time_idx]
                
                test_indices = []
                
                for entity in entities:
                    try:
                        # Find test index for this entity at test time
                        entity_test_idx = X.index.get_loc((entity, test_time))
                        entity_locs = X.index.get_locs([entity, slice(None)])
                        entity_test_end = min(entity_test_idx + test_size, entity_locs.stop)
                        test_indices.extend(range(entity_test_idx, entity_test_end))
                    except KeyError:
                        continue
                
                if test_indices:
                    yield np.array(test_indices)

# Opérer les distinctions suivantes pour "test_indices"
# - Définit le début de la période de train et de test
# - Dans ce contexte, test_size désigne la taille de la période de test à partir de cette date et max_train_size, la taille de la période d'entrainement avant cette date (moins la taille du test)

# Classe de CrossVal "In sample" pour un entraînement en panel
class PanelInSampleSplit(_BaseKFold) :

    # Initialisation
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None):
        # Initialisation du parent
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Initialisation des attributs
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test set (in-sample)."""
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        
        # For in-sample panel data, train and test are the same with potential size limits
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        entities = _get_panel_structure(X)
        
        for test_indices in self._iter_test_indices(X, y, groups):
            # For in-sample, we need to calculate appropriate train indices that include the test period
            train_indices = []
            
            if self.max_train_size is not None:
                # Apply max_train_size constraint while including test period
                for entity in entities:
                    try:
                        entity_locs = X.index.get_locs([entity, slice(None)])
                        entity_start = entity_locs.start
                        entity_stop = entity_locs.stop
                        
                        # Find test indices for this entity
                        entity_test_indices = test_indices[(test_indices >= entity_start) & (test_indices < entity_stop)]
                        
                        if len(entity_test_indices) > 0:
                            # For in-sample, train should go up to and include the test period
                            max_test_idx = max(entity_test_indices)
                            train_end = max_test_idx + 1  # Include test period
                            
                            # Apply max_train_size if specified, but ensure we include test period
                            if self.max_train_size < (train_end - entity_start):
                                train_start = max(entity_start, train_end - self.max_train_size)
                                train_indices.extend(range(train_start, train_end))
                            else:
                                train_indices.extend(range(entity_start, train_end))
                    except (KeyError, IndexError):
                        continue
                
                yield (np.array(train_indices), test_indices)
            else:
                # No max_train_size: use all data up to and including test period
                train_indices = []
                for entity in entities:
                    try:
                        entity_locs = X.index.get_locs([entity, slice(None)])
                        entity_start = entity_locs.start
                        
                        # Find test indices for this entity
                        entity_test_indices = test_indices[(test_indices >= entity_start) & (test_indices < entity_locs.stop)]
                        
                        if len(entity_test_indices) > 0:
                            # Include all data up to and including test period
                            max_test_idx = max(entity_test_indices)
                            train_indices.extend(range(entity_start, max_test_idx + 1))
                    except (KeyError, IndexError):
                        continue
                
                yield (np.array(train_indices), test_indices)

    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for panel data splits (in-sample)."""
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        
        # Get panel structure
        entities = _get_panel_structure(X)
        
        # Resolve test_indices
        resolved_test_indices = _resolve_panel_test_indices(self.test_indices, X)
        
        if resolved_test_indices is not None:
            # Use the first test_indices as the boundary for in-sample
            if isinstance(self.test_indices[0], (str, pd.Timestamp)) and hasattr(self.test_indices[0], 'date'):
                first_test_time = self.test_indices[0]
                test_indices = []
                
                for entity in entities:
                    try:
                        # Find first test index for this entity
                        entity_test_idx = X.index.get_loc((entity, first_test_time))
                        # For in-sample, test starts at this point
                        test_size = self.test_size if self.test_size is not None else 1
                        entity_test_end = min(entity_test_idx + test_size, n_samples)
                        test_indices.extend(range(entity_test_idx, entity_test_end))
                    except KeyError:
                        continue
                
                if test_indices:
                    yield np.array(test_indices)
            else:
                # Handle direct numeric indices - use first one for in-sample
                first_test_idx = min(resolved_test_indices)
                test_size = self.test_size if self.test_size is not None else 1
                test_end = min(first_test_idx + test_size, n_samples)
                yield np.arange(first_test_idx, test_end)
        else:
            # Default in-sample: use last portion of data
            test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)
            test_start = max(0, n_samples - test_size)
            yield np.arange(test_start, n_samples)

# Revoir comment inclure les classes OCOM comme un argument supplémentaire des précédentes