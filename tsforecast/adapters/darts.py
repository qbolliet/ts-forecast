"""Darts adapter for sklearn-compatible time series forecasting.

This module provides a wrapper around the darts library's GlobalForecastingModel
to enable seamless integration with sklearn pipelines and tools like GridSearchCV.
"""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
# Sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# Wrapper permettant l'intégration des modèles du package "darts" dans un syntaxe "sklearn-like"
# /!\ La méthode "score" (nécessaire pour GridSearchCV) et l'héritage de RegressorMixin font que l'on wrappe des régresseurs et non des classifieurs par défaut
class DartsAdapter(BaseEstimator, RegressorMixin):
    """Sklearn-compatible adapter for darts GlobalForecastingModel.

    This adapter wraps any GlobalForecastingModel from the darts library,
    allowing it to be used in sklearn pipelines and with sklearn tools like
    GridSearchCV and cross_val_score.

    The adapter handles conversion between pandas Series/DataFrames and darts
    TimeSeries objects, and supports both univariate and multivariate forecasting
    with past covariates.

    Args:
        model: An instantiated GlobalForecastingModel from darts (e.g.,
            LinearRegressionModel, NaiveDrift, ExponentialSmoothing, etc.).

    Attributes:
        model: The wrapped darts GlobalForecastingModel instance.
        series_: The darts TimeSeries created from training target data.
        covariates_: The darts TimeSeries for past covariates (if provided).

    Raises:
        ImportError: If darts package is not installed.

    Examples:
        Basic univariate forecasting::

            import pandas as pd
            from tsforecast.adapters import DartsAdapter
            from darts.models import NaiveDrift

            # Create time series data
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            y = pd.Series(range(100), index=dates)

            # Initialize and fit
            model = NaiveDrift()
            adapter = DartsAdapter(model=model)
            adapter.fit(None, y[:80])

            # Predict - X represents the covariates timeframe
            # For simple forecasting, pass DataFrame with target dates
            X_future = pd.DataFrame(index=dates[80:])
            predictions = adapter.predict(X_future)

        With past covariates::

            from darts.models import LinearRegressionModel
            import numpy as np

            # Create covariates (e.g., weather features)
            X = pd.DataFrame({
                'temperature': np.random.randn(100),
                'humidity': np.random.randn(100)
            }, index=dates)

            # Fit with covariates
            model = LinearRegressionModel(lags=10)
            adapter = DartsAdapter(model=model)
            adapter.fit(X[:80], y[:80])

            # Predict with future covariates
            predictions = adapter.predict(X[80:])

        Using with GridSearchCV::

            from sklearn.model_selection import GridSearchCV
            from darts.models import LinearRegressionModel

            # Create base model
            base_model = LinearRegressionModel(lags=10)
            adapter = DartsAdapter(model=base_model)

            # Define parameter grid for the wrapped model
            # Note: Use 'model__' prefix for nested parameters
            param_grid = {
                'model__lags': [5, 10, 15],
                'model__output_chunk_length': [1, 3, 5]
            }

            # GridSearchCV will clone and fit multiple models
            grid_search = GridSearchCV(
                adapter,
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
    """
    # Initialisation
    def __init__(self, model: Any) -> None:
        """Initialize the DartsAdapter.

        Args:
            model: An instantiated GlobalForecastingModel from darts.

        Examples:
            Creating an adapter with different models::

                from darts.models import NaiveDrift, ExponentialSmoothing
                from tsforecast.adapters import DartsAdapter

                # Simple baseline model
                adapter1 = DartsAdapter(model=NaiveDrift())

                # Exponential smoothing
                adapter2 = DartsAdapter(
                    model=ExponentialSmoothing(seasonal_periods=7)
                )
        """
        self.model = model

    # Méthode d'entraînement
    def fit(
        self,
        X: Optional[Union[pd.Series, pd.DataFrame]],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any
    ) -> "DartsAdapter":
        """Fit the darts model on time series data.

        If y is None, treats X as the target series (univariate case).
        If y is provided, X is treated as past covariates.

        For panel data with MultiIndex, only the last level (temporal) is
        kept for darts compatibility. All other levels are dropped. The user
        is responsible for proper data preparation.

        Args:
            X: Past covariates as DataFrame or target series if y is None.
                Must have DatetimeIndex or compatible index.
            y: Target time series to forecast. If None, X is used as target.
            **fit_params: Additional parameters passed to model.fit().
                Common parameters include 'future_covariates' for models
                that support them.

        Returns:
            self: The fitted adapter instance.

        Raises:
            ImportError: If darts package is not installed.
            ValueError: If data format is invalid or model cannot be fit.

        Examples:
            Univariate fitting::

                import pandas as pd
                from tsforecast.adapters import DartsAdapter
                from darts.models import ExponentialSmoothing

                dates = pd.date_range('2020-01-01', periods=100, freq='D')
                y = pd.Series(range(100), index=dates)

                adapter = DartsAdapter(model=ExponentialSmoothing())
                # X is None, y is the target
                adapter.fit(None, y)

            Fitting with past covariates::

                from darts.models import LinearRegressionModel
                import numpy as np

                X = pd.DataFrame({
                    'feature1': np.random.randn(100),
                    'feature2': np.random.randn(100)
                }, index=dates)

                adapter = DartsAdapter(model=LinearRegressionModel(lags=5))
                # X contains covariates, y is the target
                adapter.fit(X, y)
        """
        # Lazy import of darts
        try:
            from darts import TimeSeries
            from darts.models.forecasting.forecasting_model import GlobalForecastingModel
        except ImportError as e:
            raise ImportError(
                "DartsAdapter requires the 'darts' package. "
                "Install it with: pip install ts-forecast[adapters-darts] "
                "or: pip install darts"
            ) from e

        # Gestion du cas univarié : si y est None, X devient la série cible
        if y is None:
            y, X = X, None

        # Transformation des panel data (MultiIndex) en série multivariée avec static_covariates
        if isinstance(y.index, pd.MultiIndex):
            # Extraction des noms de niveaux (entités + temporel)
            level_names = y.index.names
            entity_levels = level_names[:-1]  # Tous sauf le dernier (temporel)
            
            # Unstacking pour transformer en multivarié : colonnes = entités, index = dates
            y_multivariate = y.unstack(level=list(range(len(entity_levels))))
            
            # Création des static_covariates à partir des entités (noms de colonnes)
            if len(entity_levels) == 1:
                # Cas simple : un seul niveau d'entités
                static_cov_df = pd.DataFrame(
                    {entity_levels[0]: y_multivariate.columns},
                    index=y_multivariate.columns
                )
            else:
                # Cas multiple : plusieurs niveaux d'entités (MultiIndex dans les colonnes)
                static_cov_df = pd.DataFrame(
                    y_multivariate.columns.tolist(),
                    index=y_multivariate.columns,
                    columns=entity_levels
                )
            
            y_processed = y_multivariate
        else:
            # Série temporelle simple : pas de transformation
            y_processed = y

        # Conversion pandas → TimeSeries Darts (univarié ou multivarié)
        self.series_ = TimeSeries.from_dataframe(
            y_processed,
        )

        # Gestion des covariables temporelles (past_covariates)
        if X is not None:
            # Transformation identique pour X si nécessaire
            if isinstance(X.index, pd.MultiIndex):
                entity_levels_x = X.index.names[:-1]
                X_multivariate = X.unstack(level=list(range(len(entity_levels_x))))
                X_processed = X_multivariate
            else:
                X_processed = X
                
            self.covariates_ = TimeSeries.from_dataframe(X_processed)
            fit_params['past_covariates'] = self.covariates_

        # Entraînement du modèle
        self.model.fit(self.series_, **fit_params)

        # Stockage des attributs fitted pour la compatibilité sklearn
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else 0
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        
        return self

    # Méthode de prédiction
    def predict(self, X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Generate forecasts using the fitted darts model.

        For panel data with MultiIndex, converts to multivariate format before
        prediction, then restacks to original panel structure. The user is
        responsible for ensuring X matches the training data structure.

        Args:
            X: Covariates for the prediction period as DataFrame with
                DatetimeIndex or MultiIndex, or DataFrame with target prediction
                dates. The index defines the time points and entities for which
                predictions are generated.

        Returns:
            Predicted values as pandas Series with index matching input X.

        Raises:
            ImportError: If darts package is not installed.
            ValueError: If model is not fitted.

        Examples:
            Generating predictions::

                # After fitting the model
                adapter.fit(None, y_train)

                # Create DataFrame for prediction period
                future_dates = pd.date_range('2020-04-01', periods=30, freq='D')
                X_future = pd.DataFrame(index=future_dates)

                # Get predictions
                predictions = adapter.predict(X_future)
                print(predictions.head())

            Panel data predictions::

                # After fitting on panel data
                adapter.fit(None, y_train)  # y_train has (entity, date) MultiIndex

                # Create prediction index matching training structure
                entities = ['store_1', 'store_2']
                future_dates = pd.date_range('2020-04-01', periods=30, freq='D')
                X_future = pd.DataFrame(
                    index=pd.MultiIndex.from_product([entities, future_dates])
                )

                # Get predictions with same MultiIndex structure
                predictions = adapter.predict(X_future)
        """
        # Lazy import of darts
        try:
            from darts import TimeSeries
        except ImportError as e:
            raise ImportError(
                "DartsAdapter requires the 'darts' package. "
                "Install it with: pip install ts-forecast[adapters-darts] "
                "or: pip install darts"
            ) from e

        # Stockage de l'index original pour reconstruction finale
        original_index = X.index
        is_panel = isinstance(X.index, pd.MultiIndex)

        # Transformation des panel data en multivarié si nécessaire
        if is_panel:
            # Extraction des niveaux d'entités
            entity_levels = list(range(X.index.nlevels - 1))
            # Unstacking pour transformer en multivarié
            X_multivariate = X.unstack(level=entity_levels)
            X_processed = X_multivariate
        else:
            X_processed = X

        # Prédiction avec covariables
        pred_series = self.model.predict(
            n=0,
            past_covariates=TimeSeries.from_dataframe(X_processed)
        )


        # Conversion TimeSeries → DataFrame/Series pandas
        if pred_series.n_components > 1:
            # Résultat multivarié : conversion en DataFrame
            pred_df = pred_series.pd_dataframe()
        else:
            # Résultat univarié : conversion en Series
            pred_df = pred_series.pd_series().to_frame()

        # Reconstruction de la structure d'index originale pour panel data
        if is_panel:
            # Restacking pour recréer le MultiIndex (entités, dates)
            pred_stacked = pred_df.stack(level=list(range(pred_df.columns.nlevels)))
            # Réindexation pour correspondre exactement à l'index original
            result = pred_stacked.reindex(original_index)
        else:
            # Série simple : conversion en Series si nécessaire
            if isinstance(pred_df, pd.DataFrame):
                result = pred_df.iloc[:, 0]
            else:
                result = pred_df
            # Réindexation pour correspondre à l'index original
            result = result.reindex(original_index)

        return result

    # Méthode d'extraction des paramètres
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for GridSearchCV compatibility.

        Model parameters are prefixed with 'model__' to enable
        nested parameter tuning in GridSearchCV.

        Args:
            deep: If True, return parameters of sub-objects.

        Returns:
            Parameter dictionary with 'model' and model sub-parameters.

        Examples:
            Getting parameters::

                from darts.models import LinearRegressionModel
                from tsforecast.adapters import DartsAdapter

                model = LinearRegressionModel(lags=5)
                adapter = DartsAdapter(model=model)

                # Get all parameters
                params = adapter.get_params(deep=True)
                print(params)
                # {'model': LinearRegressionModel(...), 'model__lags': 5, ...}

            Using with GridSearchCV::

                from sklearn.model_selection import GridSearchCV

                param_grid = {
                    'model__lags': [3, 5, 7],
                    'model__output_chunk_length': [1, 3]
                }
                grid = GridSearchCV(adapter, param_grid, cv=3)
        """
        # Initialisation du dictionnaire des paramètres
        params = {'model': self.model}

        # Extraction des paramètres du modèle
        if deep and hasattr(self.model, 'get_params'):
            # Ajout des paramètres du modèle avec le préfixe 'model__'
            model_params = self.model.get_params(deep=True)
            params.update({f'model__{k}': v for k, v in model_params.items()})

        return params

    # Méthode d'initialisation des paramètres
    def set_params(self, **params: Any) -> "DartsAdapter":
        """Set parameters for GridSearchCV compatibility.

        Model parameters should be prefixed with 'model__' for
        nested parameter setting in GridSearchCV.

        Args:
            **params: Parameters to set (model params must have 'model__' prefix).

        Returns:
            self: The adapter instance with updated parameters.

        Examples:
            Setting parameters directly::

                adapter = DartsAdapter(model=LinearRegressionModel())

                # Set adapter parameter
                adapter.set_params(model=LinearRegressionModel(lags=10))

                # Set nested model parameter
                adapter.set_params(model__lags=7)

            Used internally by GridSearchCV::

                # GridSearchCV automatically uses set_params
                param_grid = {'model__lags': [3, 5, 7]}
                grid = GridSearchCV(adapter, param_grid, cv=3)
                grid.fit(X, y)
        """
        # Séparation des paramètres de l'adapter et du modèle
        adapter_params = {}
        model_params = {}

        # Parcours des paramètres
        for key, value in params.items():
            if key.startswith('model__'):
                # Paramètre du modèle
                model_params[key.replace('model__', '')] = value
            else:
                # Paramètre de l'adapter
                adapter_params[key] = value

        # Application des paramètres de l'adapter
        for key, value in adapter_params.items():
            setattr(self, key, value)

        # Application des paramètres du modèle
        if model_params and hasattr(self.model, 'set_params'):
            self.model.set_params(**model_params)

        return self
        
    