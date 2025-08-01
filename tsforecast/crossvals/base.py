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
    
    This function works for both time series and panel data:
    - Time series: resolves date/string indices to numeric positions
    - Panel data: resolves tuple indices (group, date) to numeric positions
    
    Args:
        test_indices: List of indices to resolve (can be dates, strings, tuples, or integers)
        X: Input data with index to match against
        
    Returns:
        Array of numeric positions corresponding to the test_indices, or None if no indices provided
        
    Raises:
        ValueError: If indices cannot be resolved or are not found in the data
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
                    if is_panel_data and isinstance(idx, tuple):
                        # Vérification que l'indice recherché est présent
                        if idx in X.index:
                            # Ajout de la position de l'indice
                            positions.append(X.index.get_loc(idx))
                        else:
                            raise ValueError(f"Panel index {idx} not found in data")
                    # Gestion pour les séries temporelles avec Index simple
                    elif not is_panel_data and not isinstance(idx, tuple):
                        # Vérification que l'indice recherché est présent
                        if idx in X.index:
                            # Ajout de la position de l'indice
                            positions.append(X.index.get_loc(idx))
                        else:
                            raise ValueError(f"Time series index {idx} not found in data")
                    else:
                        raise ValueError(f"Index type mismatch: {'panel' if is_panel_data else 'time series'} data expects {'tuple' if is_panel_data else 'scalar'} indices, got {type(idx)}")
                
                return np.array(positions)
            except Exception as e:
                raise ValueError(f"Error resolving test_indices: {e}")
    
    return np.array([test_indices]) if np.isscalar(test_indices) else np.array(test_indices)

# Méthode auxiliaire de vérification et tri des données par groupe et date
def _verify_and_sort_data(X, groups=None):
    """Verify that data is sorted by group and then by date, and sort if necessary.
    
    Args:
        X: Input data (pandas Series or DataFrame)
        groups: Group labels (None for time series, array-like for panel data)
        
    Returns:
        Tuple (X_sorted, groups_sorted, sort_indices) where sort_indices maps original to sorted positions
        
    Raises:
        ValueError: If data structure is incompatible with sorting requirements
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
            # Tri par groupe puis par date
            sort_indices = X.index.to_frame().sort_values([X.index.names[0], X.index.names[1]]).index
            sort_positions = [X.index.get_loc(idx) for idx in sort_indices]
            
            X_sorted = X.iloc[sort_positions] if hasattr(X, 'iloc') else X[sort_positions]
            groups_sorted = groups[sort_positions] if groups is not None else None
            
            return X_sorted, groups_sorted, np.array(sort_positions)
        else:
            # Données déjà triées
            return X, groups, np.arange(len(X))

# Classe de base de pour crossval out of sample
class OutOfSampleSplit(_BaseKFold):
    """Base class for out-of-sample cross-validation splits."""
    # Initialisation
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None, gap=0):
        # Initialisation du parent
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Initialisation des attributs
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    # Méthode de séparation des indices d'entrainement et de test
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""

        if y is not None:
            # Vérification que X et y ont les mêmes indices
            if hasattr(X, 'index') and hasattr(y, 'index'):
                if not X.index.equals(y.index):
                    raise ValueError("'X' and 'y' should have the same indexes")
        
        return self._split(X, y, groups)
    
    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
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
        """Calculate train indices considering gap and max_train_size."""
       
        # Si les groupes ne sont pas spécifiés, on applique une logique de série temporelle
        if groups is None:
            return self._get_timeseries_train_indices(test_indices)
        # Si les groupes sont spécifiés, on applique une logique de panel
        else:
            return self._get_group_train_indices(test_indices, groups)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement pour les séries temporelles
    def _get_timeseries_train_indices(self, test_indices):
        """Calculate train indices for time series data."""
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
    
    # Méthode auxiliaire d'extraction des indices d'entraînement pour les données de panel
    def _get_group_train_indices(self, test_indices, groups):
        """Calculate train indices for panel data with groups."""
        # Détermination des groupes
        unique_groups = np.unique(groups)
        # Initialisation de la liste des indices d'entraînement
        train_indices = []
        
        # Parcours des groupes
        for group in unique_groups:
            # Identification des indices du groupe 
            group_mask = groups == group
            group_indices = np.where(group_mask)[0]
            # Identification des indices de test du groupe
            group_test_indices = test_indices[np.isin(test_indices, group_indices)]
            
            # Vérification que des indices de test ont été effectivement trouvés
            if len(group_test_indices) > 0:
                # Minimum des indices de test
                min_test_idx = min(group_test_indices)
                # Premier indice du groupe
                group_start = min(group_indices)
                # Calcul de la fin de la période d'entraînement
                train_end = min_test_idx - self.gap
                
                # Vérification que l'indice de la fin de la période d'entraînement est bien supérieur au premier indice du groupe
                if train_end > group_start:
                    if self.max_train_size and self.max_train_size < (train_end - group_start):
                        # Calcul du début de la période d'entraînement
                        train_start = train_end - self.max_train_size
                        train_indices.extend(range(train_start, train_end))
                    else:
                        # Si la période d'entraînement n'est pas spécifiée, on considère tous les indices du groupe
                        train_indices.extend(range(group_start, train_end))
            else:
                raise ValueError(f"Cannot find test indices for the group : {group}")
        
        return np.array(train_indices)
    
    # Méthode auxiliaire d'identification des indices de test
    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for splits."""
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
        """Generate test indices for time series data."""
        for test_start in sorted(resolved_test_positions):
            # Calcul de la fin de la période de test
            test_end = min(test_start + test_size, n_samples)
            yield np.arange(test_start, test_end)
    
    # Méthode auxiliaire d'identification des indices de test au sein de chaque groupe.
    def _iter_group_test_indices(self, X, resolved_test_positions, groups):
        """Generate test indices for group-aware splits."""
        # Calcul du nombre d'observations
        n_samples = _num_samples(X)
        # Identification des groupes uniques
        unique_groups = np.unique(groups)
        # Si la taille du test n'est pas spécifiée, utilise 1 par défaut
        test_size = self.test_size if self.test_size is not None else 1
        
        # /!\ On fait l'hypothèse que le panel est un pd.DataFrame avec un multi-index (entity x date)
        if isinstance(self.test_indices[0], (str, pd.Timestamp)) and hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
            # Parcours des indices de test
            for test_time in self.test_indices:
                # Initialisation des indices de test des groupes
                group_test_indices = []
                # Parcours des groupes
                for group in unique_groups:
                    # Tentative de résolution de l'indice de test pour chaque groupe
                    try:
                        if hasattr(X.index, 'get_loc'):
                            # Extraction de la position correspondant au début de la période de test pour le groupe
                            group_test_idx = X.index.get_loc((group, test_time))
                            
                            # Détermination des limites du groupe pour éviter de dépasser
                            group_mask = [idx[0] == group for idx in X.index] if hasattr(X.index, '__iter__') else []
                            group_positions = [i for i, mask in enumerate(group_mask) if mask]
                            
                            if group_positions:
                                group_end = max(group_positions) + 1  # +1 pour la limite exclusive
                                # Calcul de la position de fin du test en respectant les limites du groupe
                                group_test_end = min(group_test_idx + test_size, group_end, n_samples)
                            else:
                                # Fallback si on ne peut pas déterminer les limites du groupe
                                group_test_end = min(group_test_idx + test_size, n_samples)
                            
                            # Ajout des indices de test
                            group_test_indices.extend(range(group_test_idx, group_test_end))
                    # On ignore la période de test si on ne la trouve pas dans les données
                    except KeyError:
                        warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                        continue
                # Conversion en np.array
                if group_test_indices:
                    yield np.array(group_test_indices)
        else:
            # On utilise directement les positions
            # Détermination des groupes uniques
            unique_groups = np.unique(groups)
            # Parcours des positions de test
            for test_start in sorted(resolved_test_positions):
                # Identification du groupe auqeul appartient cette position
                current_group = None
                # Parcours des groupes
                for group in unique_groups:
                    # Détermination des limites du groupe pour ne pas dépasser
                    group_mask = groups == group
                    group_positions = np.where(group_mask)[0]
                    # Si la position de début de la période de test appartient au groupe, le groupe est identifié et sa dernière position retenue
                    if test_start in group_positions:
                        # Identification du groupe
                        current_group = group
                        # Identification de la dernière position du groupe
                        group_end = max(group_positions) + 1  # +1 pour la limite exclusive
                        # Interruption de la recherche
                        break
                
                # Calcul de la position de fin de période de test
                if current_group is not None:
                    # Calcul de la position de fin du test en respectant les limites du groupe
                    test_end = min(test_start + test_size, group_end, n_samples)
                else:
                    # Fallback si on ne peut pas identifier le groupe
                    test_end = min(test_start + test_size, n_samples)
                
                yield np.arange(test_start, test_end)
    
    # Méthode auxiliaire d'identification des indices de test en utilisant n_split comme par défaut dans TimeSeriesSplit de sklearn et en utilisant la dernière portion des données
    def _iter_default_test_indices(self, X, groups):
        """Generate default test indices using n_splits."""
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

                    # Parcours des groupes
                    for group in unique_groups:
                        # Extraction des indices de début et de fin de la période de test
                        try:
                            # Extraction de la position correspondant au début de la période de test pour le groupe
                            group_test_idx = X.index.get_loc((group, test_time))

                            # Détermination des limites du groupe pour éviter de dépasser
                            group_mask = [idx[0] == group for idx in X.index] if hasattr(X.index, '__iter__') else []
                            group_positions = [i for i, mask in enumerate(group_mask) if mask]
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
        """Initialize InSampleSplit.
        
        Args:
            n_splits (int, optional): Number of splits. Defaults to 5.
            test_indices (list, optional): Specific test indices. Defaults to None.
            max_train_size (int, optional): Maximum training size. Defaults to None.
            test_size (int, optional): Test set size. Defaults to None.
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
        
        Args:
            X (array-like): Input features. Can be pandas DataFrame/Series with DatetimeIndex
                for time series or MultiIndex for panel data.
            y (array-like, optional): Target values. Must have same index as X if provided.
                Defaults to None.
            groups (array-like, optional): Group labels for panel data. Each element
                should correspond to the group of the corresponding sample. Defaults to None.
        
        Yields:
            tuple: (train_indices, test_indices) where:
                - train_indices: Array of indices for training set (includes test period)
                - test_indices: Array of indices for test set
        
        Raises:
            ValueError: If X and y have different indices when both are provided.
            
        Examples:
            >>> # Time series example
            >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
            >>> X = pd.DataFrame({'feature': range(100)}, index=dates)
            >>> splitter = InSampleSplit(test_size=10)
            >>> for train_idx, test_idx in splitter.split(X):
            ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            >>> # Panel data example  
            >>> entities = ['A', 'B'] * 50
            >>> dates = pd.date_range('2020-01-01', periods=50, freq='D')
            >>> idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
            >>> X = pd.DataFrame({'feature': range(100)}, index=idx)
            >>> groups = [ent for ent in entities for _ in dates]
            >>> for train_idx, test_idx in splitter.split(X, groups=groups):
            ...     pass
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
        
        Args:
            X (array-like): Input features
            y (array-like, optional): Target values. Defaults to None.
            groups (array-like, optional): Group labels. Defaults to None.
            
        Yields:
            tuple: (train_indices, test_indices) mapped back to original data order
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
            X (array-like): Input features
            test_indices (array): Indices of test samples
            groups (array-like, optional): Group labels for panel data
            
        Returns:
            np.ndarray: Array of training indices that include the test period
        """
        # Logique pour les séries temporelles (pas de groupes)
        if groups is None:
            # Logique simple pour séries temporelles
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
                # Si aucune limite, utilise toutes les données depuis le début
                return np.arange(0, train_end)
        else:
            # Logique pour les données de panel (avec groupes)
            # Identification des groupes distincts
            unique_groups = np.unique(groups)
            # Initialisation des indices d'entraînement
            train_indices = []
            
            # Parcours des groupes
            for group in unique_groups:
                # Identification des indices du groupe
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]
                # Identification des indices de test du groupe
                group_test_indices = test_indices[np.isin(test_indices, group_indices)]
                
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
                        # Si aucune limite, utilise toutes les données du groupe
                        train_indices.extend(range(group_start, train_end))
            
            return np.array(train_indices)
    
    # Méthode auxiliaire de génération des indices de test pour la validation in-sample
    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for in-sample cross-validation splits.
        
        For in-sample validation, test indices are typically chosen from a specific period
        or the end of the available data, depending on the configuration.
        
        Args:
            X (array-like): Input features
            y (array-like, optional): Target values. Defaults to None.
            groups (array-like, optional): Group labels for panel data. Defaults to None.
            
        Yields:
            np.ndarray: Array of test indices for each split
            
        Raises:
            ValueError: If test_indices contain invalid positions
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
                # Parcours des indices de test
                for test_start in sorted(resolved_test_positions):
                    # Calcul de la fin de la période de test
                    test_end = min(test_start + test_size, n_samples)
                    yield np.arange(test_start, test_end)
            else:
                # Logique pour les données de panel (avec groupes)
                # Gestion des indices temporels pour données de panel
                if isinstance(self.test_indices[0], (str, pd.Timestamp)) and hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
                    # Identification des indices du groupe
                    unique_groups = np.unique(groups)
                    # Parcours des indices de test
                    for test_time in self.test_indices:
                        # Initialisation des indices de test des groupes
                        group_test_indices = []
                        
                        # Collecte des indices de test pour tous les groupe
                        for group in unique_groups:
                            # Identification des index dans le jeu de données
                            try:
                                # Identification de l'indice
                                group_test_idx = X.index.get_loc((group, test_time))
                                # Détermination des limites du groupe pour éviter de dépasser
                                group_mask = [idx[0] == group for idx in X.index] if hasattr(X.index, '__iter__') else []
                                group_positions = [i for i, mask in enumerate(group_mask) if mask]
                                
                                if group_positions:
                                    group_end = max(group_positions) + 1  # +1 pour la limite exclusive
                                    # Calcul de la position de fin du test en respectant les limites du groupe
                                    group_test_end = min(group_test_idx + test_size, group_end, n_samples)
                                else:
                                    # Fallback si on ne peut pas déterminer les limites du groupe
                                    group_test_end = min(group_test_idx + test_size, n_samples)

                                # Ajout des indices de test
                                group_test_indices.extend(range(group_test_idx, group_test_end))
                            # On ignore la période de test si on ne la trouve pas dans les données
                            except KeyError:
                                warnings.warn(f"Cannot find test period '{test_time}' for entity '{group}'")
                                continue
                        
                        # Conversion en np.array
                        if group_test_indices:
                            yield np.array(group_test_indices)
                else:
                    # On utilise directement les positions
                    # Détermination des groupes uniques
                    unique_groups = np.unique(groups)
                    # Parcours des positions de test
                    for test_start in sorted(resolved_test_positions):
                        # Identification du groupe auqeul appartient cette position
                        current_group = None
                        # Parcours des groupes
                        for group in unique_groups:
                            # Détermination des limites du groupe pour ne pas dépasser
                            group_mask = groups == group
                            group_positions = np.where(group_mask)[0]
                            # Si la position de début de la période de test appartient au groupe, le groupe est identifié et sa dernière position retenue
                            if test_start in group_positions:
                                # Identification du groupe
                                current_group = group
                                # Identification de la dernière position du groupe
                                group_end = max(group_positions) + 1  # +1 pour la limite exclusive
                                # Interruption de la recherche
                                break
                        
                        # Calcul de la position de fin de période de test
                        if current_group is not None:
                            # Calcul de la position de fin du test en respectant les limites du groupe
                            test_end = min(test_start + test_size, group_end, n_samples)
                        else:
                            # Fallback si on ne peut pas identifier le groupe
                            test_end = min(test_start + test_size, n_samples)
                        
                        yield np.arange(test_start, test_end)
        else:
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

                        # Parcours des groupes
                        for group in unique_groups:
                            # Extraction des indices de début et de fin de la période de test
                            try:
                                # Extraction de la position correspondant au début de la période de test pour le groupe
                                group_test_idx = X.index.get_loc((group, test_time))

                                # Détermination des limites du groupe pour éviter de dépasser
                                group_mask = [idx[0] == group for idx in X.index] if hasattr(X.index, '__iter__') else []
                                group_positions = [i for i, mask in enumerate(group_mask) if mask]
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
                        
                        # Conversion en np.array
                        if test_indices:
                            yield np.array(test_indices)