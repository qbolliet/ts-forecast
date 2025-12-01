# Importation des modules
# Modules de base
from typing import Dict, List, Union
# Sklearn
from sklearn.base import BaseEstimator
# 
from hierarchicalforecast import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
from hierarchicalforecast.utils import aggregate

# Wrapper permettant l'intégration des modèles du package "hierarchicalforecast" dans un syntaxe "sklearn-like"
class HierarchicalForecastAdapter(BaseEstimator):
    """Wrapper unifiant forecast + réconciliation."""
    # Initialisation
    def __init__(self, base_forecaster, reconciliation_method='mint_shrink',
                 hierarchy_spec=None):
        self.base_forecaster = base_forecaster
        self.reconciliation_method = reconciliation_method
        self.hierarchy_spec = hierarchy_spec

    # Méthode d'entraînement du jeu de données
    def fit(self, X, y=None):
        """
        X: DataFrame avec colonnes hiérarchiques + ds + y
        """
        # Agréger selon hiérarchie
        self.Y_df_, self.S_, self.tags_ = aggregate(
            X, self.hierarchy_spec
        )
        
        # Fit forecaster de base
        self.base_forecaster.fit(self.Y_df_)
        
        # Stocker fitted values si nécessaire
        if hasattr(self.base_forecaster, 'forecast_fitted_values'):
            self.Y_fitted_ = self.base_forecaster.forecast_fitted_values()
        else:
            self.Y_fitted_ = self.Y_df_
        
        return self

    # Méthode de prédiction
    def predict(self, X):
        """
        h: horizon de prévision
        """
        # Générer forecasts de base
        Y_hat = self.base_forecaster.forecast(h=h)
        
        # Réconcilier
        reconciler = MinTrace(method=self.reconciliation_method)
        hrec = HierarchicalReconciliation(reconcilers=[reconciler])
        
        Y_rec = hrec.reconcile(
            Y_hat_df=Y_hat,
            Y_df=self.Y_fitted_,
            S=self.S_,
            tags=self.tags_
        )
        
        return Y_rec