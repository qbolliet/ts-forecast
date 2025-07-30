# Importation des modules
# Modules de base
import warnings

# Local imports
from .base import OutOfSampleSplit, InSampleSplit

# Classe de CrossVal "Out of sample"
class TSOutOfSampleSplit(OutOfSampleSplit):
    """Time series out-of-sample cross-validation split.
    
    Inherits from OutOfSampleSplit and applies time series specific logic
    where groups parameter is ignored.
    """
    
    # Méthode de séparation des données en échantillons d'entraînement et de test
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        # Si des groupes sont spécifiés, ils sont ignorés
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        
        # Appel de la méthode du parent avec groups=None pour avoir le comportement sur des séries temporelles
        return super().split(X, y, groups=None)


# Classe de CrossVal "In sample"
class TSInSampleSplit(InSampleSplit):
    """Time series in-sample cross-validation split.
    
    Inherits from InSampleSplit and applies time series specific logic
    where groups parameter is ignored.
    """
    # Méthode de séparation des données en échantillons d'entraînement et de test
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        # Si des groupes sont spécifiés, ils sont ignorés
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        
        # Appel de la méthode du parent avec groups=None pour avoir le comportement sur des séries temporelles
        return super().split(X, y, groups=None)