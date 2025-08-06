# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from typing import Any, List, Optional, Union
import warnings

# Sklearn
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold


# Méthode auxiliaire de résolution des indices de test à partir de la liste en entrée en positions numériques
def _resolve_test_positions(test_indices: Optional[Union[List[Any], np.ndarray]], X: Union[pd.Series, pd.DataFrame]) -> Optional[np.ndarray]:
    """Resolve test_indices to numeric positions.
    
    This function works for both time series and panel data by converting various
    index formats to numeric positions that can be used for array indexing.
    
    Args:
        test_indices: List of indices to resolve. Can be dates, strings, tuples
            (for panel data), or integers. If None, returns None.
            For panel data: 
            - Tuples (entity, date) for specific entity-date combinations
            - Dates/strings for all entities at those dates
        X: Input data with index to match against. Must be a pandas Series or
            DataFrame with appropriate index structure.
        
    Returns:
        Array of numeric positions corresponding to the test_indices, or None
        if no indices provided.
        
    Raises:
        ValueError: If indices cannot be resolved, are not found in the data,
            or there is a type mismatch between index types and data structure.
            
    Examples:
        >>> # Time series example
        >>> dates = pd.date_range('2020-01-01', periods=10, freq='D')
        >>> X = pd.Series(range(10), index=dates)
        >>> test_dates = ['2020-01-05', '2020-01-08']
        >>> positions = _resolve_test_positions(test_dates, X)
        >>> print(positions)  # [4, 7]
        
        >>> # Panel data example with tuples (specific entity-date)
        >>> entities = ['A', 'B']
        >>> dates = pd.date_range('2020-01-01', periods=5, freq='D')
        >>> idx = pd.MultiIndex.from_product([entities, dates])
        >>> X = pd.DataFrame({'value': range(10)}, index=idx)
        >>> test_tuples = [('A', '2020-01-03'), ('B', '2020-01-04')]
        >>> positions = _resolve_test_positions(test_tuples, X)
        >>> print(positions)  # [2, 8]
        
        >>> # Panel data example with dates only (all entities at those dates)
        >>> test_dates = ['2020-01-03', '2020-01-04']
        >>> positions = _resolve_test_positions(test_dates, X)
        >>> print(positions)  # [2, 7, 3, 8] (A and B for each date)
    """

    # Si aucun indice n'est renseigné, ne renvoie rien
    if test_indices is None:
        return None
    
    # Si 'test_indices' est une liste ou un array, ses éléments sont convertis en positions dans le jeu de données
    if isinstance(test_indices, (list, np.ndarray)):
        # Si est vide, ne retourne rien
        if len(test_indices) == 0:
            return None
            
        # Si contient des entiers, on fait l'hypothèse qu'il s'agit déjà d'une liste de positions dans le jeu de données
        if isinstance(test_indices[0], (int, np.integer)):
            return np.array(test_indices)
        
        # Si contient des strings ou des dates (gestion du cas des séries temporelles) ou contient des tuples (cas des données de panel), recherche les positions correspondantes
        elif isinstance(test_indices[0], (str, pd.Timestamp)) or hasattr(test_indices[0], 'date') or isinstance(test_indices[0], tuple):
            try:
                # Vérification que X est bien une Série ou un DataFrame
                if not hasattr(X, 'index'):
                    raise ValueError("Cannot use string/date/tuple indices without pandas index")
                
                # Détection du type de données : panel (MultiIndex) vs série temporelle (Index simple)
                is_panel_data = hasattr(X.index, 'nlevels') and X.index.nlevels > 1
                
                # Initialisation des positions
                positions = []
                
                # Parcours des indices
                for idx in test_indices:
                    # Gestion spécifique pour les données de panel avec MultiIndex
                    if is_panel_data:
                        if isinstance(idx, tuple):
                            # Cas classique : tuple (entity, date) pour une position spécifique
                            if idx in X.index:
                                positions.append(X.index.get_loc(idx))
                            else:
                                raise ValueError(f"Panel index {idx} not found in data")
                        else:
                            # Cas où le test_indice est une date seule pour toutes les entités
                            # Recherche de toutes les positions où la date (niveau 1) correspond, pour totes les entités
                            date_level_values = X.index.get_level_values(1)  # Niveau des dates
                            matching_positions = np.where(date_level_values == idx)[0]
                            
                            if len(matching_positions) == 0:
                                raise ValueError(f"Date {idx} not found in panel data")
                            
                            positions.extend(matching_positions.tolist())
                    
                    # Gestion pour les séries temporelles avec Index simple
                    elif not is_panel_data and not isinstance(idx, tuple):
                        # Vérification que l'indice recherché est présent
                        if idx in X.index:
                            # Ajout de la position de l'indice
                            positions.append(X.index.get_loc(idx))
                        else:
                            raise ValueError(f"Time series index {idx} not found in data")
                    else:
                        raise ValueError(f"Index type mismatch: time series data cannot use tuple indices")
                
                return np.array(positions)
            except Exception as e:
                raise ValueError(f"Error resolving test_indices: {e}")
    
    return np.array([test_indices]) if np.isscalar(test_indices) else np.array(test_indices)


# Méthode auxiliaire de pré-calcul des mappings de groupes pour optimiser les recherches répétées
def _precompute_group_mappings(groups):
    """Precompute group mappings to optimize repeated operations.
    
    This function calculates the indices corresponding to each group once,
    avoiding repeated O(n) searches in loops and improving performance
    for group-based operations.
    
    Args:
        groups: Group labels for each sample. Can be array-like containing
            group identifiers.
        
    Returns:
        dict: Dictionary mapping each unique group to its corresponding indices
            as numpy arrays.
        
    Examples:
        >>> import numpy as np
        >>> groups = np.array(['A', 'A', 'B', 'B', 'A'])
        >>> mappings = _precompute_group_mappings(groups)
        >>> mappings['A']
        array([0, 1, 4])
        >>> mappings['B']
        array([2, 3])
        
        >>> # Usage in cross-validation
        >>> groups = np.array(['entity1', 'entity1', 'entity2', 'entity2'])
        >>> mappings = _precompute_group_mappings(groups)
        >>> entity1_indices = mappings['entity1']  # Fast lookup
    """
    # Initialisation du dictionnaire de mapping
    group_mappings = {}
    # Identification des groupes uniques pour éviter les recalculs
    unique_groups = np.unique(groups)
    
    # Pré-calcul des indices pour chaque groupe (opération O(n) faite une seule fois)
    for group in unique_groups:
        # Vectorisation de la recherche des indices du groupe
        group_mappings[group] = np.where(groups == group)[0]
    
    return group_mappings

# Méthode auxiliaire d'optimisation des opérations sur MultiIndex pour données de panel
def _precompute_multiindex_mappings(X):
    """Precompute mappings for efficient MultiIndex operations.
    
    This function extracts and caches MultiIndex information to optimize
    repeated operations on panel data, avoiding redundant index parsing.
    
    Args:
        X: DataFrame or Series with MultiIndex. Should have at least 2 levels
            where the first level represents groups and subsequent levels
            represent time or other dimensions.
        
    Returns:
        dict: Dictionary containing precomputed mappings with keys:
            - 'group_ranges': Index ranges for each group
            - 'group_positions': Numeric positions for each group
            - 'level_0_values': Values from the first index level
            - 'level_1_values': Values from the second index level
        
    Examples:
        >>> import pandas as pd
        >>> entities = ['A', 'B']
        >>> dates = pd.date_range('2020-01-01', periods=3, freq='D')
        >>> idx = pd.MultiIndex.from_product([entities, dates],
        ...                                 names=['entity', 'date'])
        >>> X = pd.DataFrame({'value': range(6)}, index=idx)
        >>> mappings = _precompute_multiindex_mappings(X)
        >>> mappings['group_ranges']['A']  # Index range for entity A
        >>> mappings['group_positions']['A']  # Numeric positions for entity A
        >>> mappings['level_0_values']  # All entity values
    """
    mappings = {}
    
    # Vérification que X a bien un MultiIndex
    if hasattr(X, 'index') and hasattr(X.index, 'nlevels') and X.index.nlevels >= 2:
        # Utilisation de pandas GroupBy pour un traitement optimisé
        grouped = X.groupby(level=0)
        
        # Pré-calcul des ranges d'indices pour chaque groupe
        mappings['group_ranges'] = {}
        mappings['group_positions'] = {}
        
        # Extraction efficace des groupes et de leurs positions
        for group_name, group_data in grouped:
            # Stockage des indices du groupe
            mappings['group_ranges'][group_name] = group_data.index
            # Calcul des positions numériques une seule fois
            mappings['group_positions'][group_name] = X.index.get_indexer(group_data.index)
            
        # Extraction des niveaux pour éviter les accès répétés
        mappings['level_0_values'] = X.index.get_level_values(0)
        mappings['level_1_values'] = X.index.get_level_values(1)
        
    return mappings

# Méthode auxiliaire de vérification et tri des données par groupe et date
def _verify_and_sort_data(X, groups=None):
    """Verify that data is sorted by group and then by date, and sort if necessary.
    
    This function ensures data is properly sorted for time-aware cross-validation.
    For time series data, it sorts by date. For panel data, it sorts by group
    first, then by date within each group.
    
    Args:
        X: Input data as pandas Series or DataFrame. Should have appropriate
            index structure (DatetimeIndex for time series, MultiIndex for panel).
        groups: Group labels for panel data. None for time series data,
            array-like for panel data. Each element should correspond to the
            group of the corresponding sample.
        
    Returns:
        tuple: A 3-tuple containing:
            - X_sorted: Sorted version of input data
            - groups_sorted: Sorted group labels (None for time series)
            - sort_indices: Array mapping original positions to sorted positions
        
    Raises:
        ValueError: If data structure is incompatible with sorting requirements
            or if panel data doesn't have the required MultiIndex structure.
            
    Examples:
        >>> # Time series example
        >>> dates = pd.date_range('2020-01-01', periods=5, freq='D')
        >>> X = pd.Series([1, 2, 3, 4, 5], index=dates[::-1])  # Reverse order
        >>> X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X)
        >>> # Data will be sorted by date
        
        >>> # Panel data example
        >>> entities = ['B', 'A', 'B', 'A']  # Mixed order
        >>> dates = ['2020-01-02', '2020-01-01', '2020-01-01', '2020-01-02']
        >>> idx = list(zip(entities, dates))
        >>> X = pd.DataFrame({'value': [1, 2, 3, 4]}, 
        ...                  index=pd.MultiIndex.from_tuples(idx))
        >>> groups = np.array(entities)
        >>> X_sorted, groups_sorted, sort_indices = _verify_and_sort_data(X, groups)
        >>> # Data will be sorted by entity, then by date within entity
    """
    
    if groups is None:
        # Série temporelle : vérification que l'index est trié par date
        if hasattr(X, 'index'):
            if not X.index.is_monotonic_increasing:
                warnings.warn("Time series data is not sorted by date. Sorting automatically.")
                sort_indices = X.index.argsort()
                X_sorted = X.iloc[sort_indices] if hasattr(X, 'iloc') else X[sort_indices]
                return X_sorted, None, sort_indices
            else:
                # Données déjà triées
                return X, None, np.arange(len(X))
        else:
            # Pas d'index pandas, on assume que les données sont correctement ordonnées
            return X, None, np.arange(len(X))
    else:
        # Données de panel : vérification que les données sont triées par groupe puis par date
        if not hasattr(X, 'index') or not hasattr(X.index, 'nlevels') or X.index.nlevels < 2:
            raise ValueError("Panel data requires MultiIndex with at least 2 levels (group, date)")
        
        # Vérification du tri par groupe puis par date
        is_sorted = True
        previous_group = None
        previous_date = None
        
        for idx in X.index:
            current_group, current_date = idx[0], idx[1]
            
            if previous_group is not None:
                # Vérification que les groupes sont regroupés
                if current_group < previous_group:
                    is_sorted = False
                    break
                # Vérification que les dates sont croissantes au sein d'un groupe
                elif current_group == previous_group and current_date < previous_date:
                    is_sorted = False
                    break
            
            previous_group, previous_date = current_group, current_date
        
        if not is_sorted:
            warnings.warn("Panel data is not sorted by group then by date. Sorting automatically.")
            # Tri optimisé par groupe puis par date
            sort_indices = X.index.to_frame(name=['entity_col', 'date_col']).sort_values(['entity_col', 'date_col']).index
            # Extraction des positions
            sort_positions = X.index.get_indexer(sort_indices)
            
            X_sorted = X.iloc[sort_positions] if hasattr(X, 'iloc') else X[sort_positions]
            groups_sorted = groups[sort_positions] if groups is not None else None
            
            return X_sorted, groups_sorted, np.array(sort_positions)
        else:
            # Données déjà triées
            return X, groups, np.arange(len(X))

# Classe de base de pour crossval out of sample
class OutOfSampleSplit(_BaseKFold):
    """Base class for out-of-sample cross-validation splits.
    
    Out-of-sample cross-validation excludes the test period from the training data,
    which simulates real-world forecasting scenarios where future data is not available
    for training. This class supports both time series and panel data.
    
    This class provides the foundation for creating cross-validation splits where:
    - Training data comes strictly before the test period (respecting temporal order)
    - A gap can be inserted between training and test periods
    - Training set size can be limited using max_train_size
    - Supports both time series (single entity) and panel data (multiple entities)
    
    Args:
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        test_indices (list, optional): Specific indices to use as test periods.
            Can be dates, strings, tuples (for panel data), or integers. If None,
            uses the last portions of the data. Defaults to None.
        max_train_size (int, optional): Maximum size of training set. If None,
            uses all available data before the test period. Defaults to None.
        test_size (int, optional): Size of each test set. If None, uses 1 for
            specific test_indices or calculated size for default splits. Defaults to None.
        gap (int, optional): Number of periods to skip between training and test sets.
            Useful to avoid data leakage in forecasting scenarios. Defaults to 0.
    
    Examples:
        >>> # Time series out-of-sample split with gap
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> X = pd.DataFrame({'feature': range(100)}, index=dates)
        >>> splitter = OutOfSampleSplit(n_splits=3, test_size=10, gap=5)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        ...     print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        >>> # Panel data with specific test periods
        >>> entities = ['A', 'B']
        >>> dates = pd.date_range('2020-01-01', periods=50, freq='D')
        >>> idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        >>> X = pd.DataFrame({'feature': range(100)}, index=idx)
        >>> test_periods = ['2020-02-15', '2020-02-20']
        >>> splitter = OutOfSampleSplit(test_indices=test_periods, test_size=3, gap=2)
        >>> groups = np.repeat(entities, 50)
        >>> for train_idx, test_idx in splitter.split(X, groups=groups):
        ...     print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        
        >>> # Using max_train_size to limit training data
        >>> splitter = OutOfSampleSplit(test_size=10, max_train_size=50, gap=1)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     print(f"Training period limited to {len(train_idx)} samples")
    """
    # Initialisation
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None, gap=0):
        """Initialize OutOfSampleSplit cross-validator.
        
        Args:
            n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
            test_indices (list, optional): Specific indices to use as test periods. Defaults to None.
            max_train_size (int, optional): Maximum size of training set. Defaults to None.
            test_size (int, optional): Size of each test set. Defaults to None.
            gap (int, optional): Number of periods between training and test sets. Defaults to 0.
        
        Examples:
            >>> # Basic initialization
            >>> splitter = OutOfSampleSplit(n_splits=3, test_size=10, gap=2)
            
            >>> # With specific test indices
            >>> test_dates = ['2023-01-15', '2023-02-15', '2023-03-15']
            >>> splitter = OutOfSampleSplit(test_indices=test_dates, test_size=5)
        """
        # Initialisation du parent
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Initialisation des attributs
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    # Méthode de séparation des indices d'entrainement et de test
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets for out-of-sample validation.
        
        This method creates time-aware splits where training data comes strictly before
        test data, respecting temporal ordering. Supports both time series and panel data.
        
        Args:
            X (array-like): Input features. Can be pandas DataFrame/Series with DatetimeIndex
                for time series or MultiIndex for panel data.
            y (array-like, optional): Target values. Must have same index as X if provided.
                Defaults to None.
            groups (array-like, optional): Group labels for panel data. Each element
                should correspond to the group of the corresponding sample. Defaults to None.
        
        Yields:
            tuple: (train_indices, test_indices) where:
                - train_indices: Array of indices for training set (before test period)
                - test_indices: Array of indices for test set
        
        Raises:
            ValueError: If X and y have different indices when both are provided.
            
        Examples:
            >>> # Time series example
            >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
            >>> X = pd.DataFrame({'feature': range(100)}, index=dates)
            >>> splitter = OutOfSampleSplit(test_size=10, gap=5)
            >>> for train_idx, test_idx in splitter.split(X):
            ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            ...     print(f"Last train date: {X_train.index[-1]}, First test date: {X_test.index[0]}")
            
            >>> # Panel data example  
            >>> entities = ['A', 'B']
            >>> dates = pd.date_range('2020-01-01', periods=50, freq='D')
            >>> idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
            >>> X = pd.DataFrame({'feature': range(100)}, index=idx)
            >>> groups = np.repeat(entities, 50)
            >>> for train_idx, test_idx in splitter.split(X, groups=groups):
            ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        """

        if y is not None:
            # Vérification que X et y ont les mêmes indices
            if hasattr(X, 'index') and hasattr(y, 'index'):
                if not X.index.equals(y.index):
                    raise ValueError("'X' and 'y' should have the same indexes")
        
        return self._split(X, y, groups)
    
    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X, y=None, groups=None):
        """Internal method to generate train/test splits with data sorting verification.
        
        This method handles the core logic for creating out-of-sample splits,
        including data validation, sorting, and index remapping.
        
        Args:
            X: Input features array, DataFrame, or Series.
            y: Target values with same structure as X. Not used in splitting
                logic but validated for index consistency.
            groups: Group labels for panel data. None for time series data.
            
        Yields:
            Tuple of (train_indices, test_indices) mapped back to original
            data order if sorting was performed.
            
        Note:
            This is an internal method, typically called by split().
            See split() method for usage examples.
        """
        # Validation des arguments
        (X,) = indexable(X)
        
        # Vérification et tri des données si nécessaire
        X_sorted, groups_sorted, sort_mapping = _verify_and_sort_data(X, groups)
        
        # Parcours des indices de test
        for test_indices in self._iter_test_indices(X_sorted, y, groups_sorted):
            # Calcul des indices d'entraînement en utilisant 'gap' et 'max_train_size'
            train_indices = self._get_train_indices(X_sorted, test_indices, groups_sorted)
            
            # Remap des indices vers l'ordre original si les données ont été triées
            if not np.array_equal(sort_mapping, np.arange(len(X))):
                # Création du mapping inverse
                inverse_mapping = np.argsort(sort_mapping)
                train_indices = inverse_mapping[train_indices]
                test_indices = inverse_mapping[test_indices]
            
            yield (train_indices, test_indices)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement correspondant aux indices de test
    def _get_train_indices(self, X, test_indices, groups):
        """Calculate training indices for out-of-sample validation.
        
        For out-of-sample validation, training data comes strictly before the test period,
        with an optional gap between them to avoid data leakage. Handles both time series
        and panel data structures.
        
        Args:
            X: Input features array or DataFrame.
            test_indices: Array of test sample indices.
            groups: Group labels for panel data. None for time series data.
            
        Returns:
            Array of training indices that come before the test period, respecting
            the gap parameter and max_train_size constraint.
            
        Note:
            This is an internal method typically called by _split(). Training indices
            will always be before test_indices with gap consideration.
        """
       
        # Si les groupes ne sont pas spécifiés, on applique une logique de série temporelle
        if groups is None:
            return self._get_timeseries_train_indices(test_indices)
        # Si les groupes sont spécifiés, on applique une logique de panel
        else:
            return self._get_group_train_indices(test_indices, groups)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement pour les séries temporelles
    def _get_timeseries_train_indices(self, test_indices):
        """Calculate training indices for time series data in out-of-sample validation.
        
        For time series, training data includes all observations before the test period,
        with an optional gap and maximum training size limit.
        
        Args:
            test_indices: Array of test sample indices.
            
        Returns:
            Array of training indices before the test period, from either the
            beginning of data or max_train_size positions before the test period.
            
        Note:
            Returns indices from 0 to min(test_indices) - gap - 1, or from
            (min(test_indices) - gap - max_train_size) to (min(test_indices) - gap - 1)
            if max_train_size is specified.
        """
        # Extraction de l'indice de test minimal
        min_test_idx = min(test_indices)
        # Calcul de la position de l'indice de la fin de période d'entraînement
        train_end = min_test_idx - self.gap
        
        # Si la fin de la période de test n'existe pas, retourne un array vide
        if train_end <= 0:
            return np.array([])
        
        # On fait l'hypothèse que les indices sont ordonnés
        if self.max_train_size and self.max_train_size < train_end:
            # Détermination du premier indice de la période d'entraînement
            train_start = train_end - self.max_train_size
            return np.arange(train_start, train_end)
        else:
            # Si la longueur de la période d'entraînement n'est pas spécifiée, par défaut on sélectionne toutes les observations depuis la première
            return np.arange(0, train_end)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement pour les données de panel (version optimisée)
    def _get_group_train_indices(self, test_indices, groups):
        """Calculate training indices for panel data with groups in out-of-sample validation.
        
        Optimized version using precomputed mappings and vectorized operations
        to avoid repeated searches and improve performance for panel data.
        
        Args:
            test_indices: Array of test sample indices.
            groups: Array of group labels for each sample.
            
        Returns:
            Array of training indices for all groups combined, where each group's
            training data comes before its respective test periods.
            
        Raises:
            ValueError: If no test indices are found for any group.
            
        Note:
            This method processes each group separately, finding training indices
            that come before the group's test period while respecting gap and
            max_train_size constraints.
        """
        # Pré-calcul des mappings de groupes pour optimiser les opérations répétées
        group_mappings = _precompute_group_mappings(groups)
        # Initialisation de la liste des indices d'entraînement
        train_indices = []
        
        # Parcours optimisé des groupes en utilisant les mappings pré-calculés
        for group, group_indices in group_mappings.items():
            # Identification vectorisée des indices de test du groupe (plus efficace que np.isin)
            # Utilisation d'un masque booléen pour améliorer les performances
            test_mask = np.isin(test_indices, group_indices)
            group_test_indices = test_indices[test_mask]
            
            # Vérification que des indices de test ont été effectivement trouvés
            if len(group_test_indices) > 0:
                # Calcul vectorisé des valeurs min/max (évite les appels multiples)
                min_test_idx = np.min(group_test_indices)
                group_start = np.min(group_indices)
                # Calcul de la fin de la période d'entraînement
                train_end = min_test_idx - self.gap
                
                # Vérification que l'indice de la fin de la période d'entraînement est bien supérieur au premier indice du groupe
                if train_end > group_start:
                    if self.max_train_size and self.max_train_size < (train_end - group_start):
                        # Calcul du début de la période d'entraînement
                        train_start = train_end - self.max_train_size
                        # Utilisation de np.arange pour générer les indices plus efficacement
                        train_indices.extend(np.arange(train_start, train_end))
                    else:
                        # Si la période d'entraînement n'est pas spécifiée, on considère tous les indices du groupe
                        # Utilisation de np.arange pour une génération plus efficace
                        train_indices.extend(np.arange(group_start, train_end))
            else:
                raise ValueError(f"Cannot find test indices for the group : {group}")
        
        # Conversion finale optimisée en évitant les listes intermédiaires
        return np.array(train_indices)
    
    # Méthode auxiliaire d'identification des indices de test
    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for out-of-sample cross-validation splits.
        
        This method creates test indices for each split, supporting both user-specified
        test periods and automatic generation based on n_splits.
        
        Args:
            X (array-like): Input features
            y (array-like, optional): Target values. Defaults to None.
            groups (array-like, optional): Group labels for panel data. Defaults to None.
            
        Yields:
            np.ndarray: Array of test indices for each split
            
        Examples:
            >>> # This is an internal method that generates test indices
            >>> # For time series: yields consecutive test periods
            >>> # For panel data: yields test periods across all entities
        """
        # Validation des arguments
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        
        # Conversion des indices de test en positions
        resolved_test_positions = _resolve_test_positions(self.test_indices, X)
        
        # Si des positions sont identifiées, on les utilise pour déterminer 
        if resolved_test_positions is not None:
            # Si la taille du test n'est pas spécifiée, utilise 1 par défaut
            test_size = self.test_size if self.test_size is not None else 1
            
            # Validation des positions de test
            if np.any(resolved_test_positions >= n_samples):
                raise ValueError("test_indices contains indices beyond data length")
            if np.any(resolved_test_positions < 0):
                raise ValueError("test_indices contains negative indices")
            
            # Si aucun groupe n'est spécifié, on applique la logique des séries temporelles
            if groups is None:
                yield from self._iter_timeseries_test_indices(resolved_test_positions, test_size, n_samples)
            # Sinon on applique la logique de panel
            else:
                yield from self._iter_group_test_indices(X, resolved_test_positions, groups)
        # Sinon, reproduit le comportement par défaut de sklearn en utilisant n_splits et en créant des segments de test equidistants
        else:
            yield from self._iter_default_test_indices(X, groups)
    
    # Méthode auxiliaire d'identification des indices de test pour les séries temporelles
    def _iter_timeseries_test_indices(self, resolved_test_positions, test_size, n_samples):
        """Generate test indices for time series data.
        
        Creates test periods of specified size starting from resolved test positions,
        ensuring they don't exceed the data boundaries.
        
        Args:
            resolved_test_positions: Array of resolved test start positions.
            test_size: Size of each test period.
            n_samples: Total number of samples in the dataset.
            
        Yields:
            Array of test indices for each test period, with each period having
            up to test_size consecutive indices.
        """
        for test_start in sorted(resolved_test_positions):
            # Calcul de la fin de la période de test
            test_end = min(test_start + test_size, n_samples)
            yield np.arange(test_start, test_end)
    
    # Méthode auxiliaire d'identification des indices de test au sein de chaque groupe (version optimisée)
    def _iter_group_test_indices(self, X, resolved_test_positions, groups):
        """Generate test indices for group-aware splits.
        
        Creates test periods for panel data, handling both temporal indices
        (dates/strings) and direct position indices. Uses precomputed mappings
        for efficient group operations.
        
        Args:
            X: Input features DataFrame or Series, typically with MultiIndex
                for panel data.
            resolved_test_positions: Array of resolved test positions.
            groups: Array of group labels for each sample.
            
        Yields:
            Array of test indices for each test period across all groups,
            respecting group boundaries and test_size constraints.
            
        Note:
            For temporal indices, attempts to find the specified time period
            for each group. Issues warnings for missing periods but continues
            processing other groups.
        """
        # Calcul du nombre d'observations
        n_samples = _num_samples(X)
        # Si la taille du test n'est pas spécifiée, utilise 1 par défaut
        test_size = self.test_size if self.test_size is not None else 1
        
        # Pré-calcul des mappings pour optimiser les opérations répétées
        group_mappings = _precompute_group_mappings(groups)
        multiindex_mappings = _precompute_multiindex_mappings(X)
        
        # /!\ Traitement optimisé pour les DataFrames avec MultiIndex (entity x date)
        if isinstance(self.test_indices[0], (str, pd.Timestamp)) and hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
            # Extraction optimisée des valeurs des niveaux (une seule fois)
            level_0_values = multiindex_mappings.get('level_0_values', X.index.get_level_values(0))
            
            # Parcours des indices de test
            for test_time in self.test_indices:
                # Initialisation des indices de test des groupes
                group_test_indices = []
                
                # Parcours optimisé des groupes en utilisant les mappings pré-calculés
                for group in group_mappings.keys():
                    # Tentative de résolution de l'indice de test pour chaque groupe
                    try:
                        if hasattr(X.index, 'get_loc'):
                            # Extraction de la position correspondant au début de la période de test pour le groupe
                            group_test_idx = X.index.get_loc((group, test_time))
                            
                            # Utilisation des mappings pré-calculés pour déterminer les limites du groupe
                            if 'group_positions' in multiindex_mappings and group in multiindex_mappings['group_positions']:
                                # Utilisation des positions pré-calculées (plus efficace)
                                group_positions = multiindex_mappings['group_positions'][group]
                                group_end = max(group_positions) + 1  # +1 pour la limite exclusive
                            else:
                                # Fallback avec vectorisation pandas (plus efficace que list comprehension)
                                group_mask = level_0_values == group
                                group_positions = np.where(group_mask)[0]
                                group_end = max(group_positions) + 1 if len(group_positions) > 0 else n_samples
                            
                            # Calcul de la position de fin du test en respectant les limites du groupe
                            group_test_end = min(group_test_idx + test_size, group_end, n_samples)
                            
                            # Ajout des indices de test
                            group_test_indices.extend(range(group_test_idx, group_test_end))
                    # Gestion des erreurs sans interrompre le processus
                    except KeyError:
                        warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                        continue
                
                # Conversion en np.array et yield si des indices sont trouvés
                if group_test_indices:
                    yield np.array(group_test_indices)
        else:
            # Traitement optimisé pour les positions directes
            # Parcours des positions de test avec recherche optimisée
            for test_start in sorted(resolved_test_positions):
                # Identification du groupe auquel appartient cette position
                group_end = n_samples  # Valeur par défaut
                
                # Recherche optimisée du groupe en utilisant les mappings pré-calculés
                for group, group_indices in group_mappings.items():
                    # Vérification vectorisée de l'appartenance (plus efficace que 'in')
                    if test_start in group_indices:
                        # Calcul optimisé de la fin du groupe
                        group_end = max(group_indices) + 1  # +1 pour la limite exclusive
                        break
                
                # Calcul de la position de fin de période de test
                test_end = min(test_start + test_size, group_end, n_samples)
                
                yield np.arange(test_start, test_end)
    
    # Méthode auxiliaire d'identification des indices de test en utilisant n_split comme par défaut dans TimeSeriesSplit de sklearn et en utilisant la dernière portion des données
    def _iter_default_test_indices(self, X, groups):
        """Generate default test indices using n_splits.
        
        Creates evenly spaced test periods from the end of the data when no
        specific test_indices are provided. Handles both time series and panel data.
        
        Args:
            X: Input features array, DataFrame, or Series.
            groups: Group labels for panel data. None for time series data.
            
        Yields:
            Array of test indices for each split, with test periods taken from
            the most recent portions of the data.
            
        Raises:
            ValueError: If the number of splits is too large for the available
                data or if there are insufficient time points for panel data.
        """
        # Comportement par défaut : utilise la dernière portion des données
        # Calcul du nombre d'observations
        n_samples = _num_samples(X)
        
        # Distinction des cas de série temporelle et de panel
        if groups is None:
            # Cas des séries-temporelles
            # Calcul du nombre de segments (équivaut au nombre de séparations + 1)
            n_folds = self.n_splits + 1
            # Si la longueur de la période de test n'est pas spécifiée
            test_size = self.test_size if self.test_size is not None else n_samples // n_folds

            # Vérification que le nombre de segments est valide
            if n_folds > n_samples:
                raise ValueError(f"Cannot have number of folds={n_folds} greater than the number of samples={n_samples}")
            # Vérification que le nombre de séprations est cohérent avec le nombre d'observations
            if n_samples - (test_size * self.n_splits) <= 0: # On peut ajouter -self.gap si on veut s'assurer des données d'entraînement
                raise ValueError(f"Too many splits={self.n_splits} for number of samples={n_samples} with test_size={test_size}")
            
            # Calcul des débuts de période de test (ce sont à chaque fois les dates les plus récentes qui sont considérées)
            test_starts = range(n_samples - self.n_splits * test_size, n_samples, test_size)
            # Parcours des débuts de periode de test
            for test_start in test_starts:
                # Ajout de la fin de période de test
                yield np.arange(test_start, test_start + test_size)
        else:
            # Cas des données de panel avec des groupes
            # Identification des différents groupes uniques
            unique_groups = np.unique(groups)
            # La taille de la période de test vaut 1 par défaut si elle n'est pas spécifiée
            test_size = self.test_size if self.test_size is not None else 1
            
            # Recherche des indices de test (on fait l'hypothèse de données regroupées par entitées et triées par date par ordre croissant)
            # /!\ On fait l'hypothèse que le panel est un pd.DataFrame avec un multi-index (entity x date)
            if hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
                # Recherche des dates
                time_points = X.index.get_level_values(1).unique()
                # Calcul du nombre de dates différentes
                n_time_points = len(time_points)
                
                # Vérification que le nombre de séparations est cohérent avec le nombre de dates
                if n_time_points - (test_size * self.n_splits) <= 0: # On peut ajouter - self.gap si on veut s'assurer des données d'entraînement
                    raise ValueError(f"Too many splits={self.n_splits} for number of time points={n_time_points} with test_size={test_size}")
            
                # Parcours des séparations
                for i in range(self.n_splits):
                    # Identification de la date de début de la période de test
                    # Identification de l'indice de la date (ce sont à chaque fois les dates les plus récentes qui sont considérées)
                    test_time_idx = n_time_points - self.n_splits * test_size + i * test_size
                    # Extarction de la date
                    test_time = time_points[test_time_idx]
                    
                    # Initialisation des indices de test
                    test_indices = []

                    # Pré-calcul des mappings pour optimiser les opérations répétées
                    multiindex_mappings = _precompute_multiindex_mappings(X)
                    level_0_values = multiindex_mappings.get('level_0_values')
                    
                    # Parcours optimisé des groupes
                    for group in unique_groups:
                        # Extraction des indices de début et de fin de la période de test
                        try:
                            # Extraction de la position correspondant au début de la période de test pour le groupe
                            group_test_idx = X.index.get_loc((group, test_time))

                            # Détermination optimisée des limites du groupe en utilisant les mappings pré-calculés
                            if 'group_positions' in multiindex_mappings and group in multiindex_mappings['group_positions']:
                                # Utilisation des positions pré-calculées (plus efficace)
                                group_positions = multiindex_mappings['group_positions'][group]
                            elif level_0_values is not None:
                                # Méthode vectorisée avec valeurs des niveaux pré-extraites
                                group_mask = level_0_values == group
                                group_positions = np.where(group_mask)[0]
                            else:
                                # Fallback pour les cas sans MultiIndex
                                group_positions = []
                            
                            # Identification de la dernière position du groupe
                            group_end = max(group_positions) + 1 if len(group_positions) > 0 else n_samples

                            # Identification de l'indice de fin de période de test
                            group_test_end = min(group_test_idx + test_size, group_end)
                            # Ajout aux indices de test
                            test_indices.extend(range(group_test_idx, group_test_end))
                        # On ignore la période de test si on ne la trouve pas dans les données
                        except KeyError:
                            warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                            continue
                    
                    # Conversion en np.array
                    if test_indices:
                        yield np.array(test_indices)


# Classe de base pour la validation croisée in-sample
class InSampleSplit(_BaseKFold):
    """Base class for in-sample cross-validation splits.
    
    In-sample cross-validation includes the test period in the training data,
    which is useful for evaluating model performance on historical data where
    future information might leak into the training set.
    
    This class supports both time series and panel data:
    - Time series: Single entity with temporal ordering
    - Panel data: Multiple entities with temporal ordering within each entity
    
    Args:
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        test_indices (list, optional): Specific indices to use as test periods.
            Can be dates, strings, tuples (for panel data), or integers. Defaults to None.
        max_train_size (int, optional): Maximum size of training set. If None,
            uses all available data up to the test period. Defaults to None.
        test_size (int, optional): Size of each test set. If None, uses 1 for
            specific test_indices or calculated size for default splits. Defaults to None.
    
    Examples:
        >>> # Time series in-sample split
        >>> splitter = InSampleSplit(n_splits=3, test_size=5)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     # Training includes test period
        ...     pass
        
        >>> # Panel data with specific test periods
        >>> test_periods = [('entity1', '2023-01-01'), ('entity2', '2023-01-01')]
        >>> splitter = InSampleSplit(test_indices=test_periods, test_size=3)
        >>> for train_idx, test_idx in splitter.split(X, groups=groups):
        ...     pass
    """
    
    # Initialisation de la classe
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None):
        """Initialize InSampleSplit cross-validator for in-sample validation.
        
        Args:
            n_splits: Number of splits for cross-validation. Only used when
                test_indices is None for calculating default test_size.
            test_indices: Specific indices to use as test periods. Can be dates,
                strings, tuples (for panel data), or integers. If None, uses
                default behavior with last portion of data.
            max_train_size: Maximum size of training set. If None, uses all
                available data up to and including the test period.
            test_size: Size of each test set. If None, calculated based on
                n_splits for default behavior or uses 1 for specific test_indices.
                
        Examples:
            >>> # Basic initialization for time series
            >>> splitter = InSampleSplit(test_size=10)
            
            >>> # With specific test dates for panel data
            >>> test_dates = ['2023-01-15', '2023-02-15']
            >>> splitter = InSampleSplit(test_indices=test_dates, test_size=5)
            
            >>> # With maximum training size constraint
            >>> splitter = InSampleSplit(test_size=5, max_train_size=100)
        """
        # Initialisation de la classe parent
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Initialisation des attributs spécifiques
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size

    # Méthode principale de séparation des données
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets for in-sample validation.
        
        In in-sample validation, the training set includes the test period, which allows
        the model to learn from future information relative to the test period.
        This is useful for evaluating model performance on historical data.
        
        Args:
            X: Input features. Can be pandas DataFrame/Series with DatetimeIndex
                for time series or MultiIndex for panel data, or array-like data.
            y: Target values. Must have same index as X if provided.
            groups: Group labels for panel data. Each element should correspond
                to the group of the corresponding sample. None for time series data.
        
        Yields:
            Tuple of (train_indices, test_indices) where:
                - train_indices: Array of indices for training set (includes test period)
                - test_indices: Array of indices for test set
        
        Raises:
            ValueError: If X and y have different indices when both are provided.
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # Time series example
            >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
            >>> X = pd.DataFrame({'feature': range(100)}, index=dates)
            >>> splitter = InSampleSplit(test_size=10)
            >>> for train_idx, test_idx in splitter.split(X):
            ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            ...     print(f"Training includes test period: {len(train_idx)} samples")
            
            >>> # Panel data example
            >>> entities = ['A', 'B']
            >>> dates = pd.date_range('2020-01-01', periods=50, freq='D')
            >>> idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
            >>> X = pd.DataFrame({'feature': range(100)}, index=idx)
            >>> groups = np.repeat(entities, 50)
            >>> splitter = InSampleSplit(test_size=5)
            >>> for train_idx, test_idx in splitter.split(X, groups=groups):
            ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        """
        # Validation des indices de X et y
        if y is not None:
            if hasattr(X, 'index') and hasattr(y, 'index'):
                if not X.index.equals(y.index):
                    raise ValueError("'X' and 'y' should have the same indexes")
        
        return self._split(X, y, groups)
    
    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X, y=None, groups=None):
        """Internal method to generate train/test splits with data sorting verification.
        
        Handles the core logic for creating in-sample splits, including data
        validation, sorting, and index remapping back to original order.
        
        Args:
            X: Input features array, DataFrame, or Series.
            y: Target values with same structure as X. Not used in splitting
                logic but validated for index consistency.
            groups: Group labels for panel data. None for time series data.
            
        Yields:
            Tuple of (train_indices, test_indices) mapped back to original
            data order if sorting was performed.
        """
        # Validation des arguments
        (X,) = indexable(X)
        
        # Vérification et tri des données si nécessaire
        X_sorted, groups_sorted, sort_mapping = _verify_and_sort_data(X, groups)
        
        # Génération des indices de test et calcul des indices d'entraînement correspondants
        for test_indices in self._iter_test_indices(X_sorted, y, groups_sorted):
            # Pour la validation in-sample, l'entraînement inclut la période de test
            train_indices = self._get_train_indices(X_sorted, test_indices, groups_sorted)
            
            # Remappage des indices vers l'ordre original si les données ont été triées
            if not np.array_equal(sort_mapping, np.arange(len(X))):
                # Création du mapping inverse pour retrouver l'ordre original
                inverse_mapping = np.argsort(sort_mapping)
                train_indices = inverse_mapping[train_indices]
                test_indices = inverse_mapping[test_indices]
            
            yield (train_indices, test_indices)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement incluant la période de test
    def _get_train_indices(self, X, test_indices, groups):
        """Calculate training indices including test period for in-sample validation.
        
        For in-sample validation, the training set extends up to and includes the test period,
        allowing the model to learn from future information relative to the test period.
        
        Args:
            X: Input features array or DataFrame.
            test_indices: Array of test sample indices.
            groups: Group labels for panel data. None for time series data.
            
        Returns:
            Array of training indices that include the test period, respecting
            max_train_size constraints while ensuring test period is included.
        """
        # Si les groupes ne sont pas spécifiés, on applique une logique de série temporelle
        if groups is None:
            return self._get_timeseries_train_indices(test_indices)
        # Si les groupes sont spécifiés, on applique une logique de panel
        else:
            return self._get_group_train_indices(test_indices, groups)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement pour les séries temporelles
    def _get_timeseries_train_indices(self, test_indices):
        """Calculate training indices for time series data in in-sample validation.
        
        For time series in-sample validation, training data includes all observations
        up to and including the test period, with optional max_train_size constraint.
        
        Args:
            test_indices: Array of test sample indices.
            
        Returns:
            Array of training indices including the test period, from either the
            beginning of data or from (max_test_idx + 1 - max_train_size) to
            (max_test_idx + 1) if max_train_size is specified.
        """
        # Identification du dernier indice de test
        max_test_idx = max(test_indices)
        # La période de test étant incluse dans la période d'entraînement, sa fin est postérieure
        train_end = max_test_idx + 1  # Inclusion de la période de test
        min_train_size = len(test_indices)  # Taille minimale = taille du test
        
        # Gestion de la taille maximale d'entraînement
        if self.max_train_size is not None:
            # La période de test étant incluse dans la période d'entrainement, sa longueur vaut au moins celle-ci
            actual_train_size = max(self.max_train_size, min_train_size)
            # Calcul du début de la période d'entraînement
            train_start = max(0, train_end - actual_train_size)
            return np.arange(train_start, train_end)
        else:
            # Si aucune limite, utilise toutes les données depuis le début jusqu'à la date de fin d'entraînement
            return np.arange(0, train_end)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement pour les données de panel
    def _get_group_train_indices(self, test_indices, groups):
        """Calculate training indices for panel data with groups in in-sample validation.
        
        For panel data in-sample validation, training indices are calculated separately
        for each group, including the test period for each group. Uses precomputed
        mappings for efficient group operations.
        
        Args:
            test_indices: Array of test sample indices.
            groups: Array of group labels for each sample.
            
        Returns:
            Array of training indices for all groups combined, where each group's
            training data includes its respective test periods.
            
        Raises:
            ValueError: If no test indices are found for any group.
        """
        # Pré-calcul des mappings de groupes pour optimiser les opérations répétées
        group_mappings = _precompute_group_mappings(groups)
        # Initialisation de la liste des indices d'entraînement
        train_indices = []
        
        # Parcours optimisé des groupes en utilisant les mappings pré-calculés
        for group, group_indices in group_mappings.items():
            # Identification vectorisée des indices de test du groupe (plus efficace que np.isin)
            # Utilisation d'un masque booléen pour améliorer les performances
            test_mask = np.isin(test_indices, group_indices)
            group_test_indices = test_indices[test_mask]
            
            # Traitement seulement si des indices de test existent pour ce groupe
            if len(group_test_indices) > 0:
                # Identification du dernier indice de test
                max_test_idx = max(group_test_indices)
                # Identification du premier indice du groupe
                group_start = min(group_indices)
                # La période de test étant incluse dans la période d'entraînement, sa fin est postérieure
                train_end = max_test_idx + 1  # Inclusion de la période de test
                
                # Gestion de la taille maximale d'entraînement pour ce groupe
                if self.max_train_size is not None:
                    # La période de test étant incluse dans la période d'entrainement, sa longueur vaut au moins celle-ci
                    min_train_size = len(group_test_indices)
                    actual_train_size = max(self.max_train_size, min_train_size)
                    # Calcul du début de la période d'entraînement
                    train_start = max(group_start, train_end - actual_train_size)
                    train_indices.extend(range(train_start, train_end))
                else:
                    # Si aucune limite, utilise toutes les données du groupe avant la fin du test
                    train_indices.extend(range(group_start, train_end))
            else:
                raise ValueError(f"Cannot find test indices for the group : {group}")
        
        return np.array(train_indices)
    
    # Méthode auxiliaire de génération des indices de test pour la validation in-sample
    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for in-sample cross-validation splits.
        
        For in-sample validation, creates a single test set rather than iterating
        through multiple splits. This is the key difference from out-of-sample validation.
        
        Args:
            X: Input features array, DataFrame, or Series.
            y: Target values. Not used in splitting logic but validated
                for index consistency if provided.
            groups: Group labels for panel data. None for time series data.
            
        Yields:
            Array of test indices for the single in-sample split. Behavior depends
            on test_indices parameter:
            - Multiple test_indices: All specified indices included, test_size ignored
            - Single test_index: Uses test_size parameter to define period size
            - No test_indices: Uses default behavior with calculated test_size
            
        Raises:
            ValueError: If test_indices contain positions beyond data length
                or negative indices.
        """
        # Validation des arguments
        (X,) = indexable(X)
        # Nombre d'observations de l'échantillon
        n_samples = _num_samples(X)
        
        # Résolution des indices de test en positions numériques
        resolved_test_positions = _resolve_test_positions(self.test_indices, X)
        
        # Traitement avec des indices de test spécifiés
        if resolved_test_positions is not None:
            # La période de test vaut 1 par défaut si elle n'est pas spécifiée
            test_size = self.test_size if self.test_size is not None else 1
            
            # Validation des positions de test
            if np.any(resolved_test_positions >= n_samples):
                raise ValueError("test_indices contains indices beyond data length")
            if np.any(resolved_test_positions < 0):
                raise ValueError("test_indices contains negative indices")
            
            # Logique pour les séries temporelles (pas de groupes)
            if groups is None:
                yield from self._iter_timeseries_test_indices(resolved_test_positions, test_size, n_samples)
            else:
                # Logique pour les données de panel (avec groupes)
                yield from self._iter_group_test_indices(X, resolved_test_positions, groups)
        else:
            # Comportement par défaut : utilise la dernière portion des données
            # Calcul du nombre d'observations
            n_samples = _num_samples(X)
            
            # Distinction des cas de série temporelle et de panel
            if groups is None:
                yield from self._iter_default_timeseries_test_indices(X, n_samples)
            else:
                yield from self._iter_default_group_test_indices(X, groups)
    
    # Méthode auxiliaire d'identification des indices de test pour les séries temporelles
    def _iter_timeseries_test_indices(self, resolved_test_positions, test_size, n_samples):
        """Generate test indices for time series data in in-sample validation.
        
        For time series in-sample validation, the behavior depends on the number
        of test indices provided. Multiple indices are all included; single index
        uses test_size parameter.
        
        Args:
            resolved_test_positions: Array of resolved test positions.
            test_size: Size of test set (ignored if multiple test indices provided).
            n_samples: Total number of samples in the dataset.
            
        Yields:
            Array of test indices. If multiple test positions are provided, yields
            all valid positions. If single position, yields consecutive indices
            up to test_size.
        """
        # Vérification du nombre d'indices de test fournis
        if len(resolved_test_positions) > 1:
            # Plusieurs indices de test : inclusion de tous, test_size ignoré
            test_indices = []
            for test_idx in sorted(resolved_test_positions):
                if test_idx < n_samples:
                    test_indices.append(test_idx)
            if test_indices:
                yield np.array(test_indices)
        else:
            # Un seul indice de test : utilisation du comportement actuel avec test_size
            first_test_idx = min(resolved_test_positions)
            # Calcul de la fin de la période de test
            test_end = min(first_test_idx + test_size, n_samples)
            yield np.arange(first_test_idx, test_end)
    
    # Méthode auxiliaire d'identification des indices de test au sein de chaque groupe
    def _iter_group_test_indices(self, X, resolved_test_positions, groups):
        """Generate test indices for group-aware splits in in-sample validation.
        
        For panel data in-sample validation, handles temporal indices and direct
        positions. Behavior depends on number of test indices: multiple indices
        are all included, single index uses test_size parameter.
        
        Args:
            X: Input features DataFrame or Series, typically with MultiIndex.
            resolved_test_positions: Array of resolved test positions.
            groups: Array of group labels for each sample.
            
        Yields:
            Array of test indices for all groups. For multiple test indices,
            includes all valid positions across groups. For single index,
            creates test periods of test_size within each group.
            
        Note:
            Issues warnings for missing periods but continues processing.
            Returns sorted unique indices to avoid duplicates.
        """
        # Calcul du nombre d'observations
        n_samples = _num_samples(X)
        # Si la taille du test n'est pas spécifiée, utilise 1 par défaut
        test_size = self.test_size if self.test_size is not None else 1
        
        # Gestion des indices temporels pour données de panel
        if isinstance(self.test_indices[0], (str, pd.Timestamp)) and hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
            # Identification des groupes uniques
            unique_groups = np.unique(groups)
            # Initialisation des indices de test pour tous les groupes
            test_indices = []
            
            # Vérification du nombre d'indices de test fournis
            if len(self.test_indices) > 1:
                # Plusieurs indices de test : inclusion de tous, test_size ignoré
                for test_time in self.test_indices:
                    # Collecte des indices de test pour tous les groupes à cette date
                    for group in unique_groups:
                        try:
                            # Identification de l'indice
                            group_test_idx = X.index.get_loc((group, test_time))
                            # Ajout de l'indice individuel
                            test_indices.append(group_test_idx)
                        # On ignore la période de test si on ne la trouve pas dans les données
                        except KeyError:
                            warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                            continue
            else:
                # Un seul indice de test : utilisation du comportement avec test_size
                first_test_time = self.test_indices[0]
                # Pré-calcul des mappings pour optimiser les opérations répétées
                multiindex_mappings = _precompute_multiindex_mappings(X)
                level_0_values = multiindex_mappings.get('level_0_values')
                
                # Collecte optimisée des indices de test pour tous les groupes
                for group in unique_groups:
                    # Identification des index dans le jeu de données
                    try:
                        # Identification de l'indice
                        group_test_idx = X.index.get_loc((group, first_test_time))
                        # Détermination optimisée des limites du groupe en utilisant les mappings pré-calculés
                        if 'group_positions' in multiindex_mappings and group in multiindex_mappings['group_positions']:
                            # Utilisation des positions pré-calculées (plus efficace)
                            group_positions = multiindex_mappings['group_positions'][group]
                        elif level_0_values is not None:
                            # Méthode vectorisée avec valeurs des niveaux pré-extraites
                            group_mask = level_0_values == group
                            group_positions = np.where(group_mask)[0]
                        else:
                            # Fallback pour les cas sans MultiIndex
                            group_positions = []
                        
                        # Identification de la dernière position du groupe
                        group_end = max(group_positions) + 1 if len(group_positions) > 0 else n_samples

                        # Identification de l'indice de fin de période de test
                        group_test_end = min(group_test_idx + test_size, group_end)

                        # Ajout des indices de test
                        test_indices.extend(range(group_test_idx, group_test_end))
                    # On ignore la période de test si on ne la trouve pas dans les données
                    except KeyError:
                        warnings.warn(f"Cannot find test period '{first_test_time}' for entity '{group}'")
                        continue
            
            # Conversion en np.array et yield unique
            if test_indices:
                yield np.array(sorted(set(test_indices)))  # Tri et suppression des doublons
        else:
            # Utilisation directe des positions
            # Vérification du nombre d'indices de test fournis
            if len(resolved_test_positions) > 1:
                # Plusieurs indices de test : inclusion de tous, test_size ignoré
                test_indices = []
                for test_idx in sorted(resolved_test_positions):
                    if test_idx < n_samples:
                        test_indices.append(test_idx)
                if test_indices:
                    yield np.array(test_indices)
            else:
                # Un seul indice de test : utilisation du comportement actuel avec test_size
                first_test_idx = min(resolved_test_positions)
                # Calcul de la fin de la période de test
                test_end = min(first_test_idx + test_size, n_samples)
                yield np.arange(first_test_idx, test_end)
    
    # Méthode auxiliaire d'identification des indices de test par défaut pour les séries temporelles
    def _iter_default_timeseries_test_indices(self, X, n_samples):
        """Generate default test indices for time series in in-sample validation.
        
        Uses the last portion of the data for testing when no specific test_indices
        are provided. Test size is calculated based on n_splits or uses provided test_size.
        
        Args:
            X: Input features array, DataFrame, or Series.
            n_samples: Total number of samples in the dataset.
            
        Yields:
            Array of test indices from the end of the time series, with size
            determined by test_size parameter or calculated from n_splits.
        """
        # Cas des séries-temporelles - utilisation de la dernière portion
        test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)
        test_start = max(0, n_samples - test_size)
        yield np.arange(test_start, n_samples)
    
    # Méthode auxiliaire d'identification des indices de test par défaut pour les données de panel
    def _iter_default_group_test_indices(self, X, groups):
        """Generate default test indices for panel data in in-sample validation.
        
        Uses the last time period available for all groups when no specific
        test_indices are provided. Assumes MultiIndex structure with entity-date format.
        
        Args:
            X: Input features DataFrame with MultiIndex, typically with
                (entity, date) structure.
            groups: Array of group labels for each sample.
            
        Yields:
            Array of test indices for all groups from the last available
            time period, with size controlled by test_size parameter.
            
        Note:
            Issues warnings for groups where the test period cannot be found
            but continues processing other groups.
        """
        # Cas des données de panel avec des groupes
        # Identification des différents groupes uniques
        unique_groups = np.unique(groups)
        # La taille de la période de test vaut 1 par défaut si elle n'est pas spécifiée
        test_size = self.test_size if self.test_size is not None else 1
        
        # Recherche des indices de test (on fait l'hypothèse de données regroupées par entitées et triées par date par ordre croissant)
        # /!\ On fait l'hypothèse que le panel est un pd.DataFrame avec un multi-index (entity x date)
        if hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
            # Recherche des dates
            time_points = X.index.get_level_values(1).unique()
            
            # Utilisation de la dernière date disponible pour validation in-sample
            test_time = time_points[-1]
            
            # Initialisation des indices de test
            test_indices = []

            # Pré-calcul des mappings pour optimiser les opérations répétées
            multiindex_mappings = _precompute_multiindex_mappings(X)
            level_0_values = multiindex_mappings.get('level_0_values')
            
            # Parcours optimisé des groupes
            for group in unique_groups:
                # Extraction des indices de début et de fin de la période de test
                try:
                    # Extraction de la position correspondant au début de la période de test pour le groupe
                    group_test_idx = X.index.get_loc((group, test_time))

                    # Détermination optimisée des limites du groupe en utilisant les mappings pré-calculés
                    if 'group_positions' in multiindex_mappings and group in multiindex_mappings['group_positions']:
                        # Utilisation des positions pré-calculées (plus efficace)
                        group_positions = multiindex_mappings['group_positions'][group]
                    elif level_0_values is not None:
                        # Méthode vectorisée avec valeurs des niveaux pré-extraites
                        group_mask = level_0_values == group
                        group_positions = np.where(group_mask)[0]
                    else:
                        # Fallback pour les cas sans MultiIndex
                        group_positions = []
                    # Identification de la dernière position du groupe
                    group_end = max(group_positions) + 1  # +1 pour la limite exclusive

                    # Identification de l'indice de fin de période de test
                    group_test_end = min(group_test_idx + test_size, group_end)
                    # Ajout aux indices de test
                    test_indices.extend(range(group_test_idx, group_test_end))
                # On ignore la période de test si on ne la trouve pas dans les données
                except KeyError:
                    warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                    continue
            
            # Conversion en np.array et yield unique
            if test_indices:
                yield np.array(test_indices)