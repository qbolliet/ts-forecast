# Importation des modules
# Modules de base
import pandas as pd
# Sklearn
from sklearn.base import BaseEstimator
# Darts
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel


# Wrapper permettant l'intégration des modèles du package "darts" dans un syntaxe "sklearn-like"
class DartsAdapter(BaseEstimator):
    """Adapter pour GlobalForecastingModel de darts."""
    # Initialisation
    def __init__(self, model : GlobalForecastingModel):
        self.model = model

    # Méthode d'entraînement
    def fit(self, X, y=None, **fit_params):
        # Si des covariables ne sont pas précisées, alors on considère le problème de prévision comme univarié
        if y is None:
            y, X = X, None

        # Conversion en time-series
        self.series_ = TimeSeries.from_dataframe(
            pd.DataFrame({'value': y}),
            value_cols='value'
        )

        # Par défaut les X sont considérés comme des observées passées. 
        # Darts spécifie les modèles pour lesquels ces arguments sont supportés ici (https://unit8co.github.io/darts/#forecasting-models)
        # Les cas où des valeurs futures sont connues ou des valeurs statisques sont utilisées ne sont pas traitées dans ce cadre
        if X is not None:
            # Création de la série temporelle des co-variables
            self.covariates_ = TimeSeries.from_dataframe(X)
            fit_params['past_covariates'] = self.covariates_

        # Entraînement du modèle
        self.model.fit(self.series_, **fit_params)
        return self

    # Méthode de prédiction
    def predict(self, X):

        # Prédiction en utilisant les covariables sans décaler l'horizon de prédiction
        pred_series = self.model.predict(
            n=0,
            past_covariates=TimeSeries.from_dataframe(X)
        )
        
        # TimeSeries → array
        return pred_series.pd_series()

