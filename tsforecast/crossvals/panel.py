# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
import warnings

# Local imports
from .base import OutOfSampleSplit, InSampleSplit

# Sklearn
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


def _get_panel_structure(X):
    """Extract panel structure from data."""
    if not hasattr(X, 'index'):
        raise ValueError("Panel data requires pandas DataFrame/Series with MultiIndex")
    
    if not hasattr(X.index, 'get_level_values'):
        raise ValueError("Panel data requires MultiIndex (entity, time)")
    
    entities = X.index.get_level_values(0).unique()
    return entities

# Classe de CrossVal "Out of sample"
class PanelOutOfSampleSplit(OutOfSampleSplit):
    """Panel data out-of-sample cross-validation split.
    
    Inherits from OutOfSampleSplit and applies panel data specific logic
    using groups to represent entities.
    """
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test set."""
        if groups is None:
            # Extract groups from panel data structure
            groups = self._extract_groups_from_panel(X)
        
        # Call parent with groups for panel behavior
        return super().split(X, y, groups)
    
    def _extract_groups_from_panel(self, X):
        """Extract entity groups from panel data structure."""
        entities = _get_panel_structure(X)
        n_samples = _num_samples(X)
        groups = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            if hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
                entity = X.index[i][0]  # First level is entity
                groups[i] = entity
            else:
                raise ValueError("Panel data requires MultiIndex (entity, time)")
        
        return groups

# Classe de CrossVal "In sample" pour un entraînement en panel
class PanelInSampleSplit(InSampleSplit):
    """Panel data in-sample cross-validation split.
    
    Inherits from InSampleSplit and applies panel data specific logic
    using groups to represent entities.
    """
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test set (in-sample)."""
        if groups is None:
            # Extract groups from panel data structure
            groups = self._extract_groups_from_panel(X)
        
        # Call parent with groups for panel behavior
        return super().split(X, y, groups)
    
    def _extract_groups_from_panel(self, X):
        """Extract entity groups from panel data structure."""
        entities = _get_panel_structure(X)
        n_samples = _num_samples(X)
        groups = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            if hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
                entity = X.index[i][0]  # First level is entity
                groups[i] = entity
            else:
                raise ValueError("Panel data requires MultiIndex (entity, time)")
        
        return groups



# Additional panel classes that return indices per entity
class PanelOutOfSampleSplitPerEntity(PanelOutOfSampleSplit):
    """Panel out-of-sample split that yields train/test indices separately for each entity."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test set per entity."""
        if groups is None:
            groups = self._extract_groups_from_panel(X)
        
        unique_groups = np.unique(groups)
        
        for train_indices, test_indices in super().split(X, y, groups):
            # Yield indices grouped by entity
            entity_splits = {}
            
            for group in unique_groups:
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]
                
                entity_train = train_indices[np.isin(train_indices, group_indices)]
                entity_test = test_indices[np.isin(test_indices, group_indices)]
                
                if len(entity_test) > 0:  # Only include entities that have test data
                    entity_splits[group] = (entity_train, entity_test)
            
            yield entity_splits


class PanelInSampleSplitPerEntity(PanelInSampleSplit):
    """Panel in-sample split that yields train/test indices separately for each entity."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split panel data into training and test set per entity."""
        if groups is None:
            groups = self._extract_groups_from_panel(X)
        
        unique_groups = np.unique(groups)
        
        for train_indices, test_indices in super().split(X, y, groups):
            # Yield indices grouped by entity
            entity_splits = {}
            
            for group in unique_groups:
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]
                
                entity_train = train_indices[np.isin(train_indices, group_indices)]
                entity_test = test_indices[np.isin(test_indices, group_indices)]
                
                if len(entity_test) > 0:  # Only include entities that have test data
                    entity_splits[group] = (entity_train, entity_test)
            
            yield entity_splits