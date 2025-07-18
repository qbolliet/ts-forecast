# Importation des modules
# Modules de base

# Sklearn
from sklearn.model_selection._split import _BaseKFold

# /!\ Faire du batch et du online learning en créant de nouvelles classes mais en harcodant juste le paramètre max_train_size d'une classe plus générale

# /!\ Opérer les distinctions suivantes pour "test_indices":
# - Si est différent de None, est prioritaire sur n_splits pour construire les périodes de tests
# - Si c'est une liste de numérique, sélectionne ces indices comme début de chaque période de test
# - Si c'est une liste de strings ou de date ou de tuple, recherche ces éléments "CONTENUS" dans un index du jeu de données et utiilise cela comme début de la période

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

    # Méthode de séparation en échantillons d'entraînement et de prédiction
    # Voir si on ne peut pas utiliser groups dans l'héritage pour faire comprendre la structure de panel au modèle
    # Dans ce cas pourrait hériter de la version de séries temporelle en spécifiant un argument supplémentaire
    def split(self, X, y = None, groups = None):
        return super().split(X, y, groups)

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
    
    # Méthode de séparation en échantillons d'entraînement et de prédiction
    # Voir si on ne peut pas utiliser groups dans l'héritage pour faire comprendre la structure de panel au modèle
    # Dans ce cas pourrait hériter de la version de séries temporelle en spécifiant un argument supplémentaire
    def split(self, X, y = None, groups = None):
        return super().split(X, y, groups)

# Revoir comment inclure les classes OCOM comme un argument supplémentaire des précédentes