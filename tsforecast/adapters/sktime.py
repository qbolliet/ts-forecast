"""Adapter for sktime forecasters to sklearn API."""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Any
# Sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# Wrapper permettant l'intégration des forecasters du package "sktime" dans un syntaxe "sklearn-like"
# /!\ La méthode "score" (nécessaire pour GridSearchCV) et l'héritage de RegressorMixin font que l'on wrappe des régresseurs et non des classifieurs par défaut
class SktimeAdapter(BaseEstimator, RegressorMixin):
    """
    Adapter to use sktime forecasters with sklearn API.
    
    This adapter converts standard sklearn DataFrames to sktime's expected format
    and handles the forecasting horizon (fh) parameter.
    
    Args:
        forecaster: A sktime forecaster instance
        fh: Forecasting horizon (default: 0 for same-period prediction)
    
    Examples:
        >>> from sktime.forecasting.arima import ARIMA
        >>> from sklearn.model_selection import cross_val_score
        >>> 
        >>> # Création de l'adapter
        >>> estimator = SktimeForecasterAdapter(ARIMA(), fh=0)
        >>> 
        >>> # Utilisation avec l'API sklearn
        >>> estimator.fit(X_train, y_train)
        >>> y_pred = estimator.predict(X_test)
        >>> 
        >>> # Utilisation avec cross_val_score
        >>> scores = cross_val_score(estimator, X, y, cv=cv)
    """
    # Initialisation
    def __init__(self, forecaster: Any, fh: int =0):
        # Initialisation des attributs
        self.forecaster = forecaster
        self.fh = fh

    # Méthode auxiliaire de conversion des données au format attendu par les sktime forecasters
    def _convert_to_sktime_format(self, X, y=None):
        """
        Convert sklearn-style DataFrame to sktime format.
        
        Args:
            X: Features DataFrame with MultiIndex (entity, date) or DatetimeIndex
            y: Target Series (optional)
        
        Returns:
            X_sktime: DataFrame with pd.Series columns for sktime
            y_sktime: Series in sktime format (if y provided)
        """
        # Conversion de X : création d'une pd.Series par colonne
        if isinstance(X, pd.DataFrame):
            X_sktime = pd.DataFrame({
                col: pd.Series(X[col].values, index=X.index)
                for col in X.columns
            })
        else:
            X_sktime = X
        
        # Conversion de y si présent
        if y is not None:
            if isinstance(y, pd.Series):
                # Création d'un DataFrame s'il s'agit de données de panel
                y_sktime = y.to_frame() if isinstance(y.index, pd.MultiIndex) else y
            else:
                # Conversion d'un array numpy en Series
                y_sktime = pd.Series(y, index=X.index if hasattr(X, 'index') else None)
                # Conversion en DataFrame s'il s'agit de données de panel
                y_sktime = y_sktime.to_frame() if isinstance(y_sktime.index, pd.MultiIndex) else y_sktime 
        else:
            y_sktime = None
            
        return X_sktime, y_sktime

    # Méthode d'entraînement
    def fit(self, X, y):
        """
        Fit the sktime forecaster.
        
        Args:
            X: Features DataFrame
            y: Target values
        
        Returns:
            self
        """
        # Conversion au format sktime
        X_sktime, y_sktime = self._convert_to_sktime_format(X, y)
        
        # Entraînement du forecaster avec fh
        self.forecaster.fit(y=y_sktime, X=X_sktime, fh=self.fh)
        
        return self

    # Méthode de prédiction
    def predict(self, X):
        """
        Make predictions using the sktime forecaster.
        
        Args:
            X: Features DataFrame
        
        Returns:
            Predictions array
        """
        # Vérification que le modèle est entraîné
        check_is_fitted(self.forecaster)
        
        # Conversion au format sktime
        X_sktime, _ = self._convert_to_sktime_format(X)
        
        # Prédiction
        y_pred = self.forecaster.predict(X=X_sktime, fh=self.fh)
        
        return y_pred

    # Méthode d'extraction des paramètres 
    # /!\ Ajouter en docstring que les paramètres du forecaster sont ajoutés avec le préfixe "forecaster_"
    def get_params(self, deep=True):
        """
        Get parameters for GridSearchCV compatibility.
        
        Args:
            deep: If True, return parameters of sub-objects
        
        Returns:
            Parameter dictionary
        """
        # Initialisation du dictionnaire des paramètres avec l'horizon de prédiction
        params = {'fh': self.fh}

        # Extraction des paramètres du forecaster
        if deep and hasattr(self.forecaster, 'get_params'):
            # Ajout des paramètres du forecaster avec le préfixe 'forecaster__'
            forecaster_params = self.forecaster.get_params(deep=True)
            params.update({f'forecaster__{k}': v for k, v in forecaster_params.items()})
        
        return params

    # Méthode d'initialisation des paramètres
    # /!\ Ajouter en docstring que les paramètres du forecaster sont attendus avec le préfixe "forecaster_"
    def set_params(self, **params):
        """
        Set parameters for GridSearchCV compatibility.
        
        Args:
            **params: Parameters to set
        
        Returns:
            self
        """
        # Séparation des paramètres de l'adapter et du forecaster
        adapter_params = {}
        forecaster_params = {}

        # Parcours des paramètres
        for key, value in params.items():
            if key.startswith('forecaster__'):
                # Paramètre du forecaster
                forecaster_params[key.replace('forecaster__', '')] = value
            else:
                # Paramètre de l'adapter
                adapter_params[key] = value
        
        # Application des paramètres de l'adapter
        for key, value in adapter_params.items():
            setattr(self, key, value)
        
        # Application des paramètres du forecaster
        if forecaster_params and hasattr(self.forecaster, 'set_params'):
            self.forecaster.set_params(**forecaster_params)
        
        return self
