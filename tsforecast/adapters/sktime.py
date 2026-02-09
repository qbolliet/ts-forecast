"""Adapter for sktime forecasters to sklearn API."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


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
    
    def __init__(self, forecaster, fh=0):
        self.forecaster = forecaster
        self.fh = fh
    
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
                y_sktime = y
            else:
                # Conversion d'un array numpy en Series
                y_sktime = pd.Series(y, index=X.index if hasattr(X, 'index') else None)
        else:
            y_sktime = None
            
        return X_sktime, y_sktime
    
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
        
        # Conversion en array numpy pour compatibilité sklearn
        if isinstance(y_pred, pd.Series):
            return y_pred.values
        return y_pred
    
    def score(self, X, y):
        """
        Calculate R² score (sklearn compatibility).
        
        Args:
            X: Features DataFrame
            y: True target values
        
        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def get_params(self, deep=True):
        """
        Get parameters for GridSearchCV compatibility.
        
        Args:
            deep: If True, return parameters of sub-objects
        
        Returns:
            Parameter dictionary
        """
        params = {'fh': self.fh}
        
        if deep and hasattr(self.forecaster, 'get_params'):
            # Ajout des paramètres du forecaster avec le préfixe 'forecaster__'
            forecaster_params = self.forecaster.get_params(deep=True)
            params.update({f'forecaster__{k}': v for k, v in forecaster_params.items()})
        
        return params
    
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
