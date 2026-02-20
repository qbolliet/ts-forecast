"""Darts adapter for sklearn-compatible time series forecasting.

This module provides a wrapper around the darts library's GlobalForecastingModel
to enable seamless integration with sklearn pipelines and tools like GridSearchCV.
"""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
import warnings
# Sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# Wrapper permettant l'intégration des modèles du package "darts" dans un syntaxe "sklearn-like"
# /!\ La méthode "score" (nécessaire pour GridSearchCV) et l'héritage de RegressorMixin font que l'on wrappe des régresseurs et non des classifieurs par défaut
class DartsAdapter(BaseEstimator, RegressorMixin):
    """Sklearn-compatible adapter for darts GlobalForecastingModel.

    This adapter wraps any GlobalForecastingModel from the darts library,
    enforcing a pure regression paradigm: y[t] = f(X[t]). Auto-regression
    on the target series is disabled, and covariates are passed as
    ``future_covariates`` with ``lags_future_covariates=[0]`` so that darts
    uses X[t] to predict y[t] without requiring temporal continuity with
    the training period.

    This design eliminates gap-related issues in cross-validation setups
    where a temporal gap exists between training and test periods (e.g.
    due to publication delays or prediction horizons). The user is
    responsible for aligning X and y upstream (e.g. via ``X.shift(-horizon)``).

    At fit time, the adapter automatically overrides the following darts
    model parameters:

    - ``lags=None`` (no auto-regression on target)
    - ``lags_past_covariates=None`` (covariates are not treated as past)
    - ``lags_future_covariates=[0]`` (y[t] predicted from X[t])
    - ``output_chunk_length=1`` (each prediction is independent)

    Args:
        model: An instantiated GlobalForecastingModel from darts (e.g.,
            LinearRegressionModel, RandomForest, LightGBMModel, etc.).

    Attributes:
        model: The wrapped darts GlobalForecastingModel instance.
        series_: The darts TimeSeries created from training target data.
        covariates_: The darts TimeSeries for future covariates (if provided).

    Raises:
        ImportError: If darts package is not installed.

    Examples:
        Basic usage with sklearn cross-validation::

            import pandas as pd
            import numpy as np
            from tsforecast.adapters import DartsAdapter
            from darts.models import LinearRegressionModel

            # Upstream preprocessing: shift X to align with horizon
            X = X.shift(-horizon)

            # Wrap darts model (no need to configure lags)
            adapter = DartsAdapter(model=LinearRegressionModel())
            adapter.fit(X_train, y_train)
            predictions = adapter.predict(X_test)

            # Score works with any sklearn-compatible cross-validation
            r2 = adapter.score(X_test, y_test)

        Using with GridSearchCV::

            from sklearn.model_selection import GridSearchCV

            adapter = DartsAdapter(model=LinearRegressionModel())
            param_grid = {
                'model__output_chunk_length': [1, 3, 5],
            }
            grid_search = GridSearchCV(adapter, param_grid, cv=cv)
            grid_search.fit(X, y)
    """
    # Initialisation
    def __init__(self, model: Any) -> None:
        """Initialize the DartsAdapter.

        This adapter enforces a pure regression paradigm: y[t] = f(X[t]).
        Auto-regression on target (lags) is disabled, and X is treated as
        future_covariates so that darts can access covariate values at
        prediction time without requiring temporal continuity with the
        training period.

        The preprocessing (horizon shift, publication delays, etc.) must be
        handled upstream by the user before calling fit/predict.

        Args:
            model: An instantiated GlobalForecastingModel from darts.
                The adapter will override ``lags``, ``lags_past_covariates``,
                ``lags_future_covariates`` and ``output_chunk_length`` at fit
                time to enforce the sklearn-like regression paradigm.

        Examples:
            Basic usage (y[t] predicted from X[t])::

                from darts.models import LinearRegressionModel
                from tsforecast.adapters import DartsAdapter

                adapter = DartsAdapter(model=LinearRegressionModel())
                adapter.fit(X_train, y_train)
                predictions = adapter.predict(X_test)
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

        Enforces a pure regression paradigm: the model learns to predict
        y[t] from X[t] (and optional lags), without auto-regression on the
        target series. X is passed to darts as ``future_covariates``.

        The adapter automatically overrides the following model parameters:

        - ``lags=None`` (no auto-regression)
        - ``lags_past_covariates=None`` (X is not treated as past covariates)
        - ``lags_future_covariates=[0]`` (y[t] predicted from X[t])
        - ``output_chunk_length=1`` (each prediction is independent)

        For panel data with MultiIndex, only the last level (temporal) is
        kept for darts compatibility. All other levels are dropped.

        Args:
            X: Covariates as DataFrame with DatetimeIndex or MultiIndex.
                Treated as future_covariates by darts.
            y: Target time series to forecast. If None, X is used as target
                and no covariates are provided.
            **fit_params: Additional parameters passed to model.fit().

        Returns:
            self: The fitted adapter instance.

        Raises:
            ImportError: If darts package is not installed.
            ValueError: If data format is invalid or model cannot be fit.

        Examples:
            Fitting with covariates::

                from darts.models import LinearRegressionModel
                from tsforecast.adapters import DartsAdapter
                import numpy as np
                import pandas as pd

                dates = pd.date_range('2020-01-01', periods=100, freq='MS')
                X = pd.DataFrame({
                    'feature1': np.random.randn(100),
                    'feature2': np.random.randn(100)
                }, index=dates)
                y = pd.Series(np.random.randn(100), index=dates)

                adapter = DartsAdapter(model=LinearRegressionModel())
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

        # Réinstanciation du modèle avec les paramètres forcés pour le paradigme régresseur pur
        # Darts ne supporte pas set_params : on réinstancie la classe avec les paramètres modifiés
        # (même approche que ForecastingModel.gridsearch)
        # - lags=None : désactivation de l'auto-régression
        # - lags_past_covariates=None : désactivation des past_covariates
        # - lags_future_covariates=[0] : y[t] = f(X[t])
        # - output_chunk_length=1 : chaque prédiction est indépendante
        model_class = type(self.model)
        model_params = self.model.model_params
        forced_params = {
            'lags': None,
            'lags_past_covariates': None,
            'lags_future_covariates': [0],
            'output_chunk_length': 1,
        }
        # Détection des paramètres qui seront écrasés
        overridden = {
            k: (model_params[k], v)
            for k, v in forced_params.items()
            if k in model_params and model_params[k] != v
        }
        if overridden:
            details = ', '.join(
                f"{k}={orig!r} → {new!r}" for k, (orig, new) in overridden.items()
            )
            warnings.warn(
                f"DartsAdapter overrides the following model parameters to enforce "
                f"the pure regression paradigm (y[t] = f(X[t])): {details}. "
                f"Pass these values explicitly to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        # Initialisation du modèle avec les bons paramètres
        self.model = model_class(**{**model_params, **forced_params})

        # Gestion des covariables (traitées comme future_covariates)
        if X is not None:
            # Transformation identique pour X si nécessaire
            if isinstance(X.index, pd.MultiIndex):
                entity_levels_x = X.index.names[:-1]
                X_multivariate = X.unstack(level=list(range(len(entity_levels_x))))
                X_processed = X_multivariate
            else:
                X_processed = X

            self.covariates_ = TimeSeries.from_dataframe(X_processed)
            fit_params['future_covariates'] = self.covariates_

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

        Computes ``n`` automatically as the number of time steps between the
        end of the training series and the last date in X. X is passed to
        darts as ``future_covariates``.

        If a temporal gap exists between the training period and X (e.g. due
        to cross-validation with ``gap > 0``), the missing covariate dates
        are filled with zeros. The corresponding predictions are discarded
        during the final reindexing step, so the fill values have no effect
        on the returned result.

        For panel data with MultiIndex, converts to multivariate format before
        prediction, then restacks to original panel structure.

        Args:
            X: Covariates for the prediction period as DataFrame with
                DatetimeIndex or MultiIndex. The temporal index defines the
                time points for which predictions are generated.

        Returns:
            Predicted values as pandas Series with index matching input X.

        Raises:
            ImportError: If darts package is not installed.
            ValueError: If model is not fitted.

        Examples:
            Generating predictions::

                adapter.fit(X_train, y_train)
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
        
        # Calcul automatique du nombre de pas de prédiction (n)
        # n = nombre de pas entre la fin de la série d'entraînement et la dernière date de X
        freq = self.series_.freq
        last_train_date = self.series_.end_time()
        last_test_date = X_processed.index.max()
        n = len(pd.date_range(
            start=last_train_date + freq,
            end=last_test_date,
            freq=freq
        ))

        # Construction des arguments de prédiction
        predict_kwargs = {'n': n}

        # Passage des covariables comme future_covariates
        # Création d'un index continu couvrant la période complète (gap + test)
        # Les dates du gap sont remplies avec 0 (valeurs ignorées car les
        # prédictions correspondantes sont écartées par le reindex final)
        if X is not None and hasattr(self, 'covariates_'):
            full_index = pd.date_range(
                start=last_train_date + freq,
                end=last_test_date,
                freq=freq
            )
            X_full = X_processed.reindex(full_index, fill_value=0)
            predict_kwargs['future_covariates'] = TimeSeries.from_dataframe(
                X_full
            )
        
        # Prédiction
        pred_series = self.model.predict(**predict_kwargs)


        # Conversion TimeSeries → DataFrame/Series pandas
        if pred_series.n_components > 1:
            # Résultat multivarié : conversion en DataFrame
            pred_df = pred_series.to_dataframe()
        else:
            # Résultat univarié : conversion en Series
            pred_df = pred_series.to_series().to_frame()
        
        # Reconstruction de la structure d'index originale pour panel data
        if is_panel:
            # Comptage de nombre de niveaux décrivant les entités du panel
            n_entity_levels = pred_df.columns.nlevels  # 1 si (entity,), 2 si (region, entity), etc.
            # Restacking pour recréer le MultiIndex (entités, dates)
            pred_stacked = pred_df.stack(level=list(range(n_entity_levels)))
            # Réordonnancement des niveaux en mettant la date à la fin
            new_order = list(range(1, n_entity_levels + 1)) + [0]
            pred_stacked.index = pred_stacked.index.reorder_levels(new_order)
            pred_stacked = pred_stacked.sort_index()
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

        Returns adapter-level parameters (``model``) and, if ``deep=True``,
        model sub-parameters prefixed with ``model__``. Sub-parameters are
        extracted from the darts ``model_params`` attribute.

        Args:
            deep: If True, return parameters of sub-objects.

        Returns:
            Parameter dictionary.

        Examples:
            Getting parameters::

                from darts.models import LinearRegressionModel
                from tsforecast.adapters import DartsAdapter

                adapter = DartsAdapter(model=LinearRegressionModel())
                params = adapter.get_params(deep=True)
                # {'model': LinearRegressionModel(...),
                #  'model__lags': ..., ...}
        """
        # Paramètres de l'adapter
        params = {
            'model': self.model,
        }

        # Extraction des paramètres du modèle via l'attribut darts model_params
        if deep and hasattr(self.model, 'model_params'):
            model_params = self.model.model_params
            params.update({f'model__{k}': v for k, v in model_params.items()})

        return params

    # Méthode d'initialisation des paramètres
    def set_params(self, **params: Any) -> "DartsAdapter":
        """Set parameters for GridSearchCV compatibility.

        Model parameters should be prefixed with ``model__`` for nested
        parameter setting. Since darts models do not support in-place
        parameter mutation, the model is reinstantiated with the updated
        parameters (same approach as ``ForecastingModel.gridsearch``).

        Args:
            **params: Parameters to set. Use ``model__`` prefix for darts
                model parameters (e.g. ``model__output_chunk_length=5``).
                Use ``model`` (without prefix) to replace the entire model
                instance.

        Returns:
            self: The adapter instance with updated parameters.

        Examples:
            Setting parameters directly::

                adapter = DartsAdapter(model=LinearRegressionModel())

                # Remplacement complet du modèle
                adapter.set_params(model=LinearRegressionModel(lags=10))

                # Modification d'un paramètre du modèle (réinstanciation)
                adapter.set_params(model__output_chunk_length=5)
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

        # Réinstanciation du modèle darts avec les paramètres modifiés
        if model_params and hasattr(self.model, 'model_params'):
            model_class = type(self.model)
            current_params = self.model.model_params
            self.model = model_class(**{**current_params, **model_params})

        return self