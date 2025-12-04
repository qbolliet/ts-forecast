"""Darts adapter for sklearn-compatible time series forecasting.

This module provides a wrapper around the darts library's GlobalForecastingModel
to enable seamless integration with sklearn pipelines and tools like GridSearchCV.
"""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Any, Optional, Union
# Sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


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

        # Conversion de la série cible en TimeSeries darts
        self.series_ = TimeSeries.from_dataframe(
            pd.DataFrame({'value': y}),
            value_cols='value'
        )

        # Gestion des covariables passées (past covariates)
        # Darts spécifie les modèles supportant ces arguments :
        # https://unit8co.github.io/darts/#forecasting-models
    # Les cas où des valeurs futures sont connues ou des valeurs statisques sont utilisées ne sont pas traitées dans ce cadre mais à travers les 'fit_params'
        if X is not None:
            self.covariates_ = TimeSeries.from_dataframe(X)
            fit_params['past_covariates'] = self.covariates_

        # Entraînement du modèle
        self.model.fit(self.series_, **fit_params)
        return self

    # Méthode de prédiction
    def predict(self, X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Generate forecasts using the fitted darts model.

        Args:
            X: Covariates for the prediction period as DataFrame with
                DatetimeIndex, or DataFrame with target prediction dates.
                The index defines the time points for which predictions
                are generated.

        Returns:
            Predicted values as pandas Series with DatetimeIndex matching
            the input period.

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

            Predictions with covariates::

                # Fit with covariates
                adapter.fit(X_train, y_train)

                # Predict using future covariates
                # X_test must have same columns as X_train
                predictions = adapter.predict(X_test)
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

        # Prédiction en utilisant les covariables sans décaler l'horizon
        pred_series = self.model.predict(
            n=0,
            past_covariates=TimeSeries.from_dataframe(X)
        )

        # Conversion TimeSeries → pandas Series
        return pred_series.pd_series()

    # Méthode de caclul de métriques
    def score(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None
    ) -> float:
        """Calculate R² score for the model predictions.

        This method is required for GridSearchCV compatibility and follows
        sklearn convention where higher values are better.

        Args:
            X: Covariates for prediction period or DataFrame with prediction dates.
            y: True target values.
            sample_weight: Sample weights for scoring. Defaults to None.

        Returns:
            R² score (coefficient of determination). Returns 1.0 for perfect
            predictions, can be negative for very poor predictions.

        Examples:
            Evaluating model performance::

                from sklearn.model_selection import train_test_split

                # Split data
                y_train, y_test = y[:80], y[80:]

                # Fit and score
                adapter.fit(None, y_train)
                r2 = adapter.score(
                    pd.DataFrame(index=y_test.index),
                    y_test
                )
                print(f"R² score: {r2:.3f}")

            Using with cross-validation::

                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(
                    adapter,
                    X,
                    y,
                    cv=5,
                    scoring='r2'
                )
                print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std():.3f})")
        """
        # Calcul des prédictions
        y_pred = self.predict(X)
        # Calcul de la métrique
        return r2_score(y, y_pred, sample_weight=sample_weight)
