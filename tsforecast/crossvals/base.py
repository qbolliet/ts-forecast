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

# TODO
# - Changer le naming de "_resolve_test_indices" et des variables liées pour mieux signifier que l'output est des POSITIONS associées aux indices
# - Revoir la structure des méthodes de OutOfSampleSplit. En particulier, j'ai l'impression qu'il y a des méthodes spécifiques aux groupes pour "_iter_test_indices" mais pas pour "_get_train_indices", faire la structure la plus claire et la plus propre
# - Faire apparaître dans "_resolve_test_indices" que cela ne fonctionne que pour les séries temporelles et pas les données de panel. Dans ce cadre, est-il utile de le passer résultat de cette fonction en argument de "_iter_group_test_indices". Par ailleurs, ne vaut-il pas mieux traiter les deux types de données dans "_resolve_test_indices" et alléger le code de "_iter_group_test_indices"

# Méthode auxiliaire de résulution des indices de test à partir de la liste en entrée
def _resolve_test_indices(test_indices: Optional[Union[List[Any], np.ndarray]], X: Union[pd.Series, pd.DataFrame]) -> Optional[np.ndarray]:
    """Resolve test_indices to numeric positions."""

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
                # Initialisation des positions
                positions = []
                # Parcours des indices
                for idx in test_indices:
                    # Vérification que X est bien une Série ou un DataFrame
                    if hasattr(X, 'index'):
                        # Vérification que l'indice recherché est présent
                        if idx in X.index:
                            # Ajout de la position de l'indice
                            positions.append(X.index.get_loc(idx))
                        else:
                            raise ValueError(f"Index {idx} not found in data")
                    else:
                        raise ValueError("Cannot use string/date indices without pandas index")
                return np.array(positions)
            except Exception as e:
                raise ValueError(f"Error resolving test_indices: {e}")
    
    return np.array([test_indices]) if np.isscalar(test_indices) else np.array(test_indices)

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
        
        # Parcours des indices de test
        for test_indices in self._iter_test_indices(X, y, groups):
            # Calcul des indices d'entraînement en utilisant 'gap' et 'max_train_size'
            train_indices = self._get_train_indices(X, test_indices, groups)
            yield (train_indices, test_indices)
    
    # Méthode auxiliaire d'extraction des indices d'entraînement correspondant aux indices de test
    def _get_train_indices(self, X, test_indices, groups):
        """Calculate train indices considering gap and max_train_size."""
       
        # Si les groupes ne sont pas spécifiés, on applique une logique de série temporelle, où l'on recherche un seul groupe d'indices d'entraînement
        if groups is None:
            # Extraction de l'indice de test minimal
            min_test_idx = min(test_indices)
            # Calcul de la position de l'indice de la fin de période d'entraînement
            train_end = min_test_idx - self.gap
            
            # Si la fin de la période de test d'existe pas, retourne un array vide
            if train_end <= 0:
                return np.array([])
            
            # On fait l'hypothèse que les indices sont ordonnés
            # /!\ Peut on la vérifier automatiquement ?
            if self.max_train_size and self.max_train_size < train_end:
                # Détermination du premier indice de la période d'entraînement
                train_start = train_end - self.max_train_size
                return np.arange(train_start, train_end)
            else:
                # Si la longueur de la période d'entraînement n'est pas spécifiée, par défaut on sélectionne toutes les observations depuis la première
                return np.arange(0, train_end)
        # Si les groupes sont spécifiés, on applique une logique de panel où l'on recherche un ensemble d'indices d'entraînement pour chaque groupe
        else:
            # Détermination des groupes
            unique_groups = np.unique(groups)
            # Initialisation de la liste des indices d'entraînement
            train_indices = []
            # Parcours des groupes
            # /!\ On peut peut-être faire l'hypothèse que "test_indices" est un tableau en deux dimensions, ordonné comme "unique_groups" et utiliser moins de isin pour réidentifier les indices
            for group in unique_groups:
                # Identification des indices du groupe 
                group_mask = groups == group
                group_indices = np.where(group_mask)[0] # /!\ Je ne suis pas sur qu'une liste d'indices soit retourné par cette ligne
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
                    
                    # Vérification que l'indice de la fin de la période d'entraînement est bien supérieur au premier indice du groupe, sinon retourne un array vide
                    if train_end > group_start:
                        # On fait l'hypothèse que les indices sont ordonnés
                        # /!\ Peut on la vérifier automatiquement ?
                        if self.max_train_size and self.max_train_size < (train_end - group_start):
                            # Calcul du début de la période d'entraînement
                            train_start = train_end - self.max_train_size
                            train_indices.extend(range(train_start, train_end))
                        else:
                            # Si la période d'entraînement n'est pas spécifiée, on considère tous les indices du groupe
                            train_indices.extend(range(group_start, train_end))
                else :
                    raise ValueError(f"Cannot find test indices for the group : {group}")
            
            return np.array(train_indices)
    
    # Méthode auxiliaire d'identification des indices de test
    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for splits."""
        # Validation des arguments
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        
        # Conversion des indices de test en positions
        resolved_test_indices = _resolve_test_indices(self.test_indices, X)
        
        # Si des positions sont identifiées, on les utilise pour déterminer 
        if resolved_test_indices is not None:
            # Si la taille du test n'est pas spécifiée, utilise 1 par défaut
            test_size = self.test_size if self.test_size is not None else 1
            
            # Validation des indices de test
            if np.any(resolved_test_indices >= n_samples):
                raise ValueError("test_indices contains indices beyond data length")
            if np.any(resolved_test_indices < 0):
                raise ValueError("test_indices contains negative indices")
            
            # Si aucun groupe n'est spécifié, on applique la logique des séries temporelles
            if groups is None:
                # Simple time series splits
                for test_start in sorted(resolved_test_indices):
                    # Calcul de la fin de la période de test
                    test_end = min(test_start + test_size, n_samples)
                    yield np.arange(test_start, test_end)
            # Sinon on applique la logique de panel
            else:
                yield from self._iter_group_test_indices(X, resolved_test_indices, groups)
        # Sinon, reproduit le comportement par défaut de sklearn en utilisant n_splits et en créant des segments de test equidistants
        else:
            yield from self._iter_default_test_indices(X, groups)
    
    # Méthode auxiliaire d'identification des indices de test au sein de chaque groupe.
    def _iter_group_test_indices(self, X, resolved_test_indices, groups):
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
                            # Extraction de la position correspondante
                            group_test_idx = X.index.get_loc((group, test_time))
                            # Calcul de la position de fin du test 
                            # TODO n_samples est trop lâche il faut s'assurer que ne vaut pas plus que le dernier indice du groupe +1
                            # TODO il faut aussi vérifier que le jeu de données est trié par groupe puis date de sorte à ce que les observations d'un même groupe soient regroupées et que les dates soient ordonnées par ordre croissant
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
            for test_start in sorted(resolved_test_indices):
                # Détermination de la période de test
                # /!\ n_samples est encore trop optimiste cf commentaire précédent
                test_end = min(test_start + test_size, n_samples)
                yield np.arange(test_start, test_end)
    
    def _iter_default_test_indices(self, X, groups):
        """Generate default test indices using n_splits."""
        n_samples = _num_samples(X)
        
        if groups is None:
            # Simple time series default
            n_folds = self.n_splits + 1
            test_size = self.test_size if self.test_size is not None else n_samples // n_folds
            
            if n_folds > n_samples:
                raise ValueError(f"Cannot have number of folds={n_folds} greater than the number of samples={n_samples}")
            if n_samples - self.gap - (test_size * self.n_splits) <= 0:
                raise ValueError(f"Too many splits={self.n_splits} for number of samples={n_samples} with test_size={test_size} and gap={self.gap}")
            
            test_starts = range(n_samples - self.n_splits * test_size, n_samples, test_size)
            for test_start in test_starts:
                yield np.arange(test_start, test_start + test_size)
        else:
            # Group-aware default (panel data)
            unique_groups = np.unique(groups)
            test_size = self.test_size if self.test_size is not None else 1
            
            # Find time points (assuming sorted data)
            if hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
                time_points = X.index.get_level_values(1).unique()
                n_time_points = len(time_points)
                
                if self.n_splits >= n_time_points:
                    raise ValueError(f"Cannot have n_splits={self.n_splits} >= n_time_points={n_time_points}")
                
                for i in range(self.n_splits):
                    test_time_idx = n_time_points - self.n_splits + i
                    test_time = time_points[test_time_idx]
                    
                    test_indices = []
                    for group in unique_groups:
                        try:
                            group_test_idx = X.index.get_loc((group, test_time))
                            group_test_end = min(group_test_idx + test_size, n_samples)
                            test_indices.extend(range(group_test_idx, group_test_end))
                        except KeyError:
                            continue
                    
                    if test_indices:
                        yield np.array(test_indices)


class InSampleSplit(_BaseKFold):
    """Base class for in-sample cross-validation splits."""
    
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        if y is not None:
            if hasattr(X, 'index') and hasattr(y, 'index'):
                if not X.index.equals(y.index):
                    raise ValueError("'X' and 'y' should have the same indexes")
        
        return self._split(X, y, groups)
    
    def _split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        (X,) = indexable(X)
        
        for test_indices in self._iter_test_indices(X, y, groups):
            # For in-sample, train includes test period
            train_indices = self._get_train_indices(X, test_indices, groups)
            yield (train_indices, test_indices)
    
    def _get_train_indices(self, X, test_indices, groups):
        """Calculate train indices including test period for in-sample validation."""
        if groups is None:
            # Simple time series logic
            max_test_idx = max(test_indices)
            train_end = max_test_idx + 1  # Include test period
            min_train_size = len(test_indices)  # At minimum test size
            
            if self.max_train_size is not None:
                actual_train_size = max(self.max_train_size, min_train_size)
                train_start = max(0, train_end - actual_train_size)
                return np.arange(train_start, train_end)
            else:
                return np.arange(0, train_end)
        else:
            # Group-aware logic (for panel data)
            unique_groups = np.unique(groups)
            train_indices = []
            
            for group in unique_groups:
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]
                group_test_indices = test_indices[np.isin(test_indices, group_indices)]
                
                if len(group_test_indices) > 0:
                    max_test_idx = max(group_test_indices)
                    group_start = min(group_indices)
                    train_end = max_test_idx + 1  # Include test period
                    
                    if self.max_train_size is not None:
                        min_train_size = len(group_test_indices)
                        actual_train_size = max(self.max_train_size, min_train_size)
                        train_start = max(group_start, train_end - actual_train_size)
                        train_indices.extend(range(train_start, train_end))
                    else:
                        train_indices.extend(range(group_start, train_end))
            
            return np.array(train_indices)
    
    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for in-sample splits."""
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        
        # Resolve test_indices
        resolved_test_indices = _resolve_test_indices(self.test_indices, X)
        
        if resolved_test_indices is not None:
            test_size = self.test_size if self.test_size is not None else 1
            
            # Validation
            if np.any(resolved_test_indices >= n_samples):
                raise ValueError("test_indices contains indices beyond data length")
            if np.any(resolved_test_indices < 0):
                raise ValueError("test_indices contains negative indices")
            
            if groups is None:
                # For in-sample, use first test index as boundary
                first_test_idx = min(resolved_test_indices)
                test_end = min(first_test_idx + test_size, n_samples)
                yield np.arange(first_test_idx, test_end)
            else:
                # Group-aware in-sample (use first test time)
                if isinstance(self.test_indices[0], (str, pd.Timestamp)) and hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
                    first_test_time = self.test_indices[0]
                    unique_groups = np.unique(groups)
                    test_indices = []
                    
                    for group in unique_groups:
                        try:
                            group_test_idx = X.index.get_loc((group, first_test_time))
                            group_test_end = min(group_test_idx + test_size, n_samples)
                            test_indices.extend(range(group_test_idx, group_test_end))
                        except KeyError:
                            continue
                    
                    if test_indices:
                        yield np.array(test_indices)
                else:
                    # Handle direct numeric indices
                    first_test_idx = min(resolved_test_indices)
                    test_end = min(first_test_idx + test_size, n_samples)
                    yield np.arange(first_test_idx, test_end)
        else:
            # Default in-sample: use last portion of data
            if groups is None:
                test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)
                test_start = max(0, n_samples - test_size)
                yield np.arange(test_start, n_samples)
            else:
                # Group-aware default in-sample
                unique_groups = np.unique(groups)
                test_size = self.test_size if self.test_size is not None else 1
                
                if hasattr(X, 'index') and hasattr(X.index, 'get_level_values'):
                    time_points = X.index.get_level_values(1).unique()
                    last_time = time_points[-1]
                    
                    test_indices = []
                    for group in unique_groups:
                        try:
                            group_test_idx = X.index.get_loc((group, last_time))
                            group_test_end = min(group_test_idx + test_size, n_samples)
                            test_indices.extend(range(group_test_idx, group_test_end))
                        except KeyError:
                            continue
                    
                    if test_indices:
                        yield np.array(test_indices)