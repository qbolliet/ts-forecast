# Importation des modules
# Modules de base
import numpy as np
# Sklearn
from sklearn.base import BaseEstimator, RegressorMixin
# Opera
from opera import Mixture

# Wrapper permettant l'intégration des modèles du package "opera" dans un syntaxe "sklearn-like"
class OperaAdapter(BaseEstimator, RegressorMixin):
    """Wrapper sklearn pour opera avec support partial_fit."""

    # Initialisation
    def __init__(self, model, loss_type="mse", loss_gradient=False):
        self.model = model
        self.loss_type = loss_type
        self.loss_gradient = loss_gradient
        self.mixture_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y):
        """
        X: predictions d'experts (n_samples, n_experts)
        y: targets réels (n_samples,)
        """
        self.mixture_ = Mixture(
            y=y,
            experts=X,
            model=self.model,
            loss_type=self.loss_type,
            loss_gradient=self.loss_gradient
        )
        self.is_fitted_ = True
        return self
    
    def partial_fit(self, X, y):
        """Update incrémentiel - clé pour online learning."""
        if not self.is_fitted_:
            return self.fit(X, y)
        
        self.mixture_.update(new_experts=X, new_y=y)
        return self
    
    def predict(self, X):
        """Prédiction sans update des coefficients."""
        if not self.is_fitted_:
            raise ValueError("Model must be fit before prediction")
        return self.mixture_.predict(new_experts=X)
    