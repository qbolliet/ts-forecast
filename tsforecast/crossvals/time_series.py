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

# /!\ Les deux classes ont une méthode "split" similaire que l'on pourrait revoir en héritant d'une même classe

# /!\ A AJOUTER DANS LA DOCSTRING ET LES EXEMPLES (IMPLEMENTE TEL QUEL) : 
# Opérer les distinctions suivantes pour "test_indices":
# - Si est différent de None, est prioritaire sur n_splits pour construire les périodes de tests
# - Si c'est une liste de numérique, sélectionne ces indices comme début de chaque période de test
# - Si c'est une liste de strings ou de date, recherche ces éléments dans l'index du jeu de données et utiilise cela comme début de la période

# Fonction auxiliaire d'identification des indices de tests
def _resolve_test_indices(test_indices : Optional[Union[List[Any], np.ndarray]], X : Union[pd.Series, pd.DataFrame]):
    """Resolve test_indices to numeric positions."""
    # Distinction suivant le type d'indices
    if test_indices is None:
        return None
    # Si les indices de tests sont une liste ou un array, recherche des indices dans l'array 
    if isinstance(test_indices, (list, np.ndarray)):
        # Ne retourne rien si la liste est vide
        if len(test_indices) == 0:
            return None
            
        # Retourne les entiers tels quels
        if isinstance(test_indices[0], (int, np.integer)):
            return np.array(test_indices)
        
        # Sinon, recherche des positions dans l'indice.
        # Comme il s'agit d'une série temporelle, on s'attend à trouver des dates
        elif isinstance(test_indices[0], (str, pd.Timestamp)) or hasattr(test_indices[0], 'date'):
            try:
                # Initialisation de la liste des positions
                positions = []
                # Parcours de la liste des éléments à trouver
                for idx in test_indices:
                    # Recherche de l'élément dans l'indice
                    if hasattr(X, 'index'):
                        if idx in X.index:
                            positions.append(X.index.get_loc(idx))
                        else:
                            raise ValueError(f"Index {idx} not found in data")
                    else:
                        raise ValueError("Cannot use string/date indices without pandas index")
                return np.array(positions)
            except Exception as e:
                raise ValueError(f"Error resolving test_indices: {e}")
    
    return np.array([test_indices]) if np.isscalar(test_indices) else np.array(test_indices)

# Classe de CrossVal "Out of sample"
class TSOutOfSampleSplit(_BaseKFold) :

    # Initialisation
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None, gap=0):
        # Initialisation du parent
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Initialisation des attributs
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    # Méthode de séparation en échantillons d'entraînement et de prédiction
    # /!\ Utiliser groups pour faire la méthode OCOM dans une autre classe dont la méthode spécifie les groupes avec les individus du panel
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        # Les groupes ne sont pas utilisés
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        # Renvoie une erreur si y est renseigné et ne possède pas les mêmes indices que X
        if y is not None :
            if not pd.testing.assert_index_equal(X, y) :
                raise ValueError("'X' and 'y' should have the same indexes")
        
        return self._split(X)
    
    # Méthode auxiliaire de séparation des données en échantillons d'entraînement et de prédiction
    def _split(self, X):
        """Generate indices to split data into training and test set."""
        # Validation des arguments
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        # Résolution des indices de début de période de test
        resolved_test_indices = _resolve_test_indices(self.test_indices, X)
        
        # Si les indices de test sont renseignés, ils sont utilisés pour déterminer les séparations entre périodes d'entraînement et de test
        if resolved_test_indices is not None:
            # Utilisation des indices de tests pour déterminer le nombre de séparations
            self.n_splits = len(resolved_test_indices)
            # Longueur des périodes de tests
            test_size = self.test_size if self.test_size is not None else 1
            
            # Validation des test_indices
            if np.any(resolved_test_indices >= n_samples):
                raise ValueError("test_indices contains indices beyond data length")
            if np.any(resolved_test_indices < 0):
                raise ValueError("test_indices contains negative indices")
            
            # Initialisation des indices sur lesquels opérer la recherche
            indices = np.arange(n_samples)
            
            # Parcours des débuts de période de test
            for test_start in sorted(resolved_test_indices):
                # Calcul de la fin de la période de test
                test_end = min(test_start + test_size, n_samples)
                # Calcul de la fin de la période d'entraînement
                train_end = test_start - self.gap
                
                # Gestion du cas limite
                if train_end <= 0:
                    continue
                
                # Si une période d'entraînement maximale et valide est spécifiée, elle est utilisée
                if self.max_train_size and self.max_train_size < train_end:
                    # Calcul de la date de début de période d'entraînement
                    train_start = train_end - self.max_train_size
                    yield (
                        indices[train_start:train_end],
                        indices[test_start:test_end],
                    )
                else:
                    yield (
                        indices[:train_end],
                        indices[test_start:test_end],
                    )
        else:
            # Utilisation de la logique de sklearn TimeSeriesSplit
            n_folds = self.n_splits + 1
            # Utilisation de la période de tests si elle est renseignée, sinon on scinde l'ensemble des observations en segments de même longueur pour chaque période de test
            test_size = (
                self.test_size if self.test_size is not None else n_samples // n_folds
            )
            
            # Vérification que l'on a un nombre suffisant d'abservations pour chaque séquencement
            if n_folds > n_samples:
                raise ValueError(
                    f"Cannot have number of folds={n_folds} greater"
                    f" than the number of samples={n_samples}."
                )
            if n_samples - self.gap - (test_size * self.n_splits) <= 0:
                raise ValueError(
                    f"Too many splits={self.n_splits} for number of samples"
                    f"={n_samples} with test_size={test_size} and gap={self.gap}."
                )
            # Initialisation des indices
            indices = np.arange(n_samples)
            # Initialisayion des indices de débuts
            test_starts = range(n_samples - self.n_splits * test_size, n_samples, test_size)
            # Parcours des débuts de période de test
            for test_start in test_starts:
                # Calcul de la date de fin de période d'entraînement
                train_end = test_start - self.gap
                # Si une période d'entraînement maximale et valide est spécifiée, elle est utilisée
                if self.max_train_size and self.max_train_size < train_end:
                    yield (
                        indices[train_end - self.max_train_size : train_end],
                        indices[test_start : test_start + test_size],
                    )
                else:
                    yield (
                        indices[:train_end],
                        indices[test_start : test_start + test_size],
                    )


# Opérer les distinctions suivantes pour "test_indices"
# - Définit le début de la période de train et de test
# - Dans ce contexte, test_size désigne la taille de la période de test à partir de cette date et max_train_size, la taille de la période d'entrainement avant cette date (moins la taille du test). La période d'entrainement comprend dans tous les cas au minimum la période de test


# Classe de CrossVal "In sample"
class TSInSampleSplit(_BaseKFold) :

    # Initialisation
    def __init__(self, n_splits=5, *, test_indices=None, max_train_size=None, test_size=None):
        # Initialisation du parent
        super().__init__(n_splits, shuffle=False, random_state=None)
        # Initialisation des attributs
        self.test_indices = test_indices
        self.max_train_size = max_train_size
        self.test_size = test_size

    # Méthode de séparation en échantillons d'entraînement et de prédiction
    # Utiliser groups pour faire la méthode OCOM dans une autre classe dont la méthode spécifie les groupes avec les individus du panel
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        # Les groupes ne sont pas utilisés
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        # Renvoie une erreur si y est renseigné et ne possède pas les mêmes indices que X
        if y is not None :
            if not pd.testing.assert_index_equal(X, y) :
                raise ValueError("'X' and 'y' should have the same indexes")
        
        return self._split(X)
    
    # Méthode auxiliaire de séparation des données d'entraînement et de test
    def _split(self, X):
        """Generate indices to split data into training and test set."""
        # Validation des arguments
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        
        # Résolution des 'test_indices'
        resolved_test_indices = _resolve_test_indices(self.test_indices, X)
        

        # Si les indices de test sont renseignés, ils sont utilisés pour déterminer les séparations entre périodes d'entraînement et de test
        if resolved_test_indices is not None:
            # Utilisation des indices de tests pour déterminer le nombre de séparations
            self.n_splits = len(resolved_test_indices)
            # Longueur des périodes de tests
            test_size = self.test_size if self.test_size is not None else 1
            
            # Validation des test_indices
            if np.any(resolved_test_indices >= n_samples):
                raise ValueError("test_indices contains indices beyond data length")
            if np.any(resolved_test_indices < 0):
                raise ValueError("test_indices contains negative indices")
            
            # Parcours de tous les indices de test résolus
            for test_start in sorted(resolved_test_indices):
                # Calcul de la fin de la période de test
                test_end = min(test_start + test_size, n_samples)
                
                # Les indices de test commencent au test_start
                test_indices = np.arange(test_start, test_end)
                
                # Calcul de la période d'entraînement qui doit inclure au minimum la période de test
                min_train_size = len(test_indices)  # Au minimum la taille de la période de test
                
                # Si max_train_size est spécifié, utilise le maximum entre max_train_size et min_train_size
                if self.max_train_size is not None:
                    actual_train_size = max(self.max_train_size, min_train_size)
                    # Calcul du début de la période d'entraînement en partant de test_end
                    train_start = max(0, test_end - actual_train_size)
                    train_indices = np.arange(train_start, test_end)
                else:
                    # Si pas de max_train_size, utilise toutes les observations disponibles avant + la période de test
                    train_indices = np.arange(0, test_end)
                
                yield (train_indices, test_indices)
        else:
            # Utilisation de n_splits pour créer des divisions in-sample
            n_folds = self.n_splits + 1
            # Longueur des périodes de tests
            test_size = self.test_size if self.test_size is not None else n_samples // n_folds
            
            # Vérification que l'on a un nombre suffisant d'observations pour chaque séquencement
            if n_folds > n_samples:
                raise ValueError(
                    f"Cannot have number of folds={n_folds} greater"
                    f" than the number of samples={n_samples}."
                )
            if test_size * self.n_splits > n_samples:
                raise ValueError(
                    f"Too many splits={self.n_splits} for number of samples"
                    f"={n_samples} with test_size={test_size}."
                )
            
            # Initialisation des indices de débuts de test
            test_starts = range(n_samples - self.n_splits * test_size, n_samples, test_size)
            
            # Parcours des débuts de période de test
            for test_start in test_starts:
                # Calcul de la fin de la période de test
                test_end = min(test_start + test_size, n_samples)
                
                # Les indices de test
                test_indices = np.arange(test_start, test_end)
                
                # Calcul de la période d'entraînement qui doit inclure au minimum la période de test
                min_train_size = len(test_indices)
                
                # Si max_train_size est spécifié, utilise le maximum entre max_train_size et min_train_size
                if self.max_train_size is not None:
                    actual_train_size = max(self.max_train_size, min_train_size)
                    # Calcul du début de la période d'entraînement en partant de test_end
                    train_start_idx = max(0, test_end - actual_train_size)
                    train_indices = np.arange(train_start_idx, test_end)
                else:
                    # Si pas de max_train_size, utilise toutes les observations disponibles avant + la période de test
                    train_indices = np.arange(0, test_end)
                
                yield (train_indices, test_indices)