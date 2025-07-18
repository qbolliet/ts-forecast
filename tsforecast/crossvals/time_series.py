# Importation des modules
# Modules de base

# Sklearn
from sklearn.model_selection._split import _BaseKFold

# /!\ Opérer les distinctions suivantes pour "test_indices":
# - Si est différent de None, est prioritaire sur n_splits pour construire les périodes de tests
# - Si c'est une liste de numérique, sélectionne ces indices comme début de chaque période de test
# - Si c'est une liste de strings ou de date, recherche ces éléments dans l'index du jeu de données et utiilise cela comme début de la période

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
    # Utiliser groups pour faire la méthode OCOM dans une autre classe dont la méthode spécifie les groupes avec les individus du panel
    def split(self, X, y = None, groups = None):
        return super().split(X, y, groups)


# Opérer les distinctions suivantes pour "test_indices"
# - Définit le début de la période de train et de test
# - Dans ce contexte, test_size désigne la taille de la période de test à partir de cette date et max_train_size, la taille de la période d'entrainement avant cette date (moins la taille du test)


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
    def split(self, X, y = None, groups = None):
        return super().split(X, y, groups)