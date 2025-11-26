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
    def __init__(self, model="BOA", coefficients="uniform", loss_type="mse", loss_gradient=True, parameters=None):
        # Initialisation des paramètres du modèle Mixture
        self.model = model
        self.coefficients = coefficients
        self.loss_type = loss_type
        self.loss_gradient = loss_gradient
        self.parameters = parameters
        # Initialisation du modèle de mixture
        self.mixture_ = None
        # Initialisation du booléen du statut d'entraînement
        self.is_fitted_ = False

    # Méthode d'entraînement
    def fit(self, X, y):
        """
        X: predictions d'experts (n_samples, n_experts)
        y: targets réels (n_samples,)
        """
        # Initialisation du modèle (l'entraînement initial est compris dans l'initialisation)
        self.mixture_ = Mixture(
            y=y,
            experts=X,
            model=self.model,
            coefficients=self.coefficients,
            loss_type=self.loss_type,
            loss_gradient=self.loss_gradient,
            parameters=self.parameters
        )
        # Mise à jour du statut d'entraînement
        self.is_fitted_ = True
        
        return self

    # Méthode d'entraînement partiel
    def partial_fit(self, X, y):
        """Update incrémental"""
        # Initialisation de l'entraînement du modèle s'il ne l'a jamais été
        if not self.is_fitted_:
            return self.fit(X, y)
        # Mise à jour du modèle
        self.mixture_.update(new_experts=X, new_y=y)
        
        return self

    # Méthode de prédiction
    def predict(self, X):
        """Prédiction sans update des coefficients."""
        # Vérification que l'estimateur est entraîné
        if not self.is_fitted_:
            raise ValueError("Model must be fit before prediction")
        # Réalisation des prédictions
        return self.mixture_.predict(new_experts=X)
    