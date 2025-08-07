"""Mixed frequency transformer for time series data.

This module provides transformers to handle mixed frequency data through
aggregation and imputation strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from joblib import Parallel, delayed

from ..utils.frequency_detection import FrequencyDetector, FrequencyConverter
from .release_delays import ReleaseDelayTransformer


class MixedFrequencyTransformer(BaseEstimator, TransformerMixin):
    """Transform mixed frequency time series data to a common frequency.
    
    This transformer handles datasets with mixed frequencies by aggregating
    high-frequency data and imputing low-frequency data using machine learning
    models.
    
    Parameters:
        target_frequency (str): Target frequency for all series.
            Options: 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
        estimator (Optional[BaseEstimator]): Estimator for continuous variable imputation.
            If None, forward-fill is used.
        classifier (Optional[BaseEstimator]): Classifier for categorical variable imputation.
            If None, mode imputation is used.
        transformer (Optional[TransformerMixin]): Transformer to apply before imputation.
        handle_nan (bool): Whether the estimator can handle NaN values. Default is False.
        metric (Union[str, Callable]): Metric for model selection. Default is 'mse'.
        categorical_metric (Union[str, Callable]): Metric for categorical models. 
            Default is 'accuracy'.
        validation_size (float): Proportion of data for validation. Default is 0.2.
        time_col (str): Name of time column. Default is 'date'.
        panel_cols (Optional[List[str]]): Panel dimension columns.
        n_jobs (int): Number of parallel jobs. Default is 1.
        random_state (Optional[int]): Random state for reproducibility.
        
    Attributes:
        frequency_map_ (Dict): Detected frequencies for each series.
        variable_types_ (Dict): Detected variable types.
        imputation_models_ (Dict): Fitted imputation models.
        aggregation_functions_ (Dict): Aggregation functions for each variable.
        is_fitted_ (bool): Whether the transformer has been fitted.
    """
    
    def __init__(self,
                 target_frequency: str,
                 estimator: Optional[BaseEstimator] = None,
                 classifier: Optional[BaseEstimator] = None,
                 transformer: Optional[TransformerMixin] = None,
                 handle_nan: bool = False,
                 metric: Union[str, Callable] = 'mse',
                 categorical_metric: Union[str, Callable] = 'accuracy',
                 validation_size: float = 0.2,
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 n_jobs: int = 1,
                 random_state: Optional[int] = None):
        """Initialize the MixedFrequencyTransformer."""
        self.target_frequency = target_frequency
        self.estimator = estimator
        self.classifier = classifier
        self.transformer = transformer
        self.handle_nan = handle_nan
        self.metric = metric
        self.categorical_metric = categorical_metric
        self.validation_size = validation_size
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Initialisation des détecteurs
        self._freq_detector = FrequencyDetector()
        self._freq_converter = FrequencyConverter()
        self._var_detector = VariableTypeDetector()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MixedFrequencyTransformer':
        """Fit the transformer on the training data.
        
        Args:
            X: Input features with mixed frequencies
            y: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        # Validation des données
        X = self._validate_input(X)
        
        # Détection des fréquences
        self.frequency_map_ = self._freq_detector.detect_dataset_frequency(
            X, self.time_col, self.panel_cols
        )
        
        # Détection des types de variables
        self.variable_types_ = self._detect_variable_types(X)
        
        # Détermination des fonctions d'agrégation
        self.aggregation_functions_ = self._determine_aggregation_functions()
        
        # Initialisation du dictionnaire des modèles d'imputation
        self.imputation_models_ = {}
        
        # Préparation des données pour l'imputation
        # Application temporaire d'un ReleaseDelayTransformer pour avoir des données complètes
        delay_transformer = ReleaseDelayTransformer(
            prediction_date='today',
            time_col=self.time_col,
            panel_cols=self.panel_cols
        )
        
        # Fit du transformer de délais
        delay_transformer.fit(X)
        
        # Annulation des délais pour l'entraînement
        X_no_delays = delay_transformer.inverse_transform(X)
        
        # Entraînement des modèles d'imputation pour les variables basse fréquence
        self._fit_imputation_models(X_no_delays, y)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data to the target frequency.
        
        Args:
            X: Input data with mixed frequencies
            
        Returns:
            Transformed data at target frequency
        """
        check_is_fitted(self, 'is_fitted_')
        
        # Validation des données
        X = self._validate_input(X)
        X_transformed = X.copy()
        
        # Identification des colonnes à transformer
        high_freq_cols = []
        low_freq_cols = []
        
        for col, freq in self.frequency_map_.items():
            if self._freq_converter.is_higher_frequency(freq, self.target_frequency):
                high_freq_cols.append(col)
            elif freq != self.target_frequency:
                low_freq_cols.append(col)
        
        # Agrégation des données haute fréquence
        if high_freq_cols:
            X_transformed = self._aggregate_high_frequency(X_transformed, high_freq_cols)
        
        # Imputation des données basse fréquence
        if low_freq_cols:
            X_transformed = self._impute_low_frequency(X_transformed, low_freq_cols)
        
        # Alignement final à la fréquence cible
        X_transformed = self._align_to_target_frequency(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step, optimized for training data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed features
        """
        # Ajustement du transformer
        self.fit(X, y)
        
        # Pour les données d'entraînement, application d'une stratégie enrichie
        X_transformed = X.copy()
        
        # Annulation temporaire des délais pour enrichir les données
        delay_transformer = ReleaseDelayTransformer(
            prediction_date='today',
            time_col=self.time_col,
            panel_cols=self.panel_cols
        )
        delay_transformer.fit(X)
        X_no_delays = delay_transformer.inverse_transform(X)
        
        # Transformation enrichie avec imputation
        X_enriched = self._transform_with_enrichment(X_no_delays)
        
        # Réapplication des délais originaux
        X_final = delay_transformer.transform(X_enriched)
        
        return X_final
    
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data format.
        
        Args:
            X: Input data
            
        Returns:
            Validated DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Vérification de la cohérence avec les paramètres
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found: {missing_cols}")
        
        return X
    
    def _detect_variable_types(self, X: pd.DataFrame) -> Dict[str, str]:
        """Detect variable types for each column.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary mapping columns to types
        """
        variable_types = {}
        
        for col in X.columns:
            if col != self.time_col and (not self.panel_cols or col not in self.panel_cols):
                var_type = self._var_detector.detect_type(X[col])
                variable_types[col] = var_type
        
        return variable_types
    
    def _determine_aggregation_functions(self) -> Dict[str, Callable]:
        """Determine aggregation functions based on variable types.
        
        Returns:
            Dictionary mapping columns to aggregation functions
        """
        agg_functions = {}
        
        for col, var_type in self.variable_types_.items():
            if var_type == 'continuous':
                # Moyenne pour les variables continues
                agg_functions[col] = 'mean'
            elif var_type == 'binary' or var_type == 'categorical':
                # Mode pour les variables catégorielles et binaires
                agg_functions[col] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
            else:
                # Par défaut, moyenne
                agg_functions[col] = 'mean'
        
        return agg_functions
    
    def _fit_imputation_models(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit imputation models for low-frequency variables.
        
        Args:
            X: Training data without delays
            y: Target variable (optional)
        """
        # Identification des variables basse fréquence nécessitant imputation
        low_freq_vars = {}
        for col, freq in self.frequency_map_.items():
            if not self._freq_converter.is_higher_frequency(freq, self.target_frequency) and \
               freq != self.target_frequency:
                low_freq_vars[col] = freq
        
        if not low_freq_vars:
            return
        
        # Entraînement parallèle des modèles si possible
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_imputation_model)(X, col, freq)
                for col, freq in low_freq_vars.items()
            )
            
            # Stockage des résultats
            for col, model in results:
                if model is not None:
                    self.imputation_models_[col] = model
        else:
            # Entraînement séquentiel
            for col, freq in low_freq_vars.items():
                _, model = self._fit_single_imputation_model(X, col, freq)
                if model is not None:
                    self.imputation_models_[col] = model
    
    def _fit_single_imputation_model(self, 
                                    X: pd.DataFrame, 
                                    column: str, 
                                    source_freq: str) -> Tuple[str, Any]:
        """Fit imputation model for a single variable.
        
        Args:
            X: Training data
            column: Column to impute
            source_freq: Original frequency of the column
            
        Returns:
            Tuple of (column_name, fitted_model)
        """
        # Préparation des données pour l'entraînement
        # Agrégation des variables haute fréquence à la fréquence source
        X_aggregated = self._aggregate_to_frequency(X, source_freq)
        
        # Sélection des features (toutes sauf la cible)
        feature_cols = [c for c in X_aggregated.columns 
                       if c != column and c != self.time_col 
                       and (not self.panel_cols or c not in self.panel_cols)]
        
        if not feature_cols:
            warnings.warn(f"No features available for imputing {column}")
            return column, None
        
        # Préparation des données d'entraînement
        mask = X_aggregated[column].notna()
        X_train = X_aggregated.loc[mask, feature_cols]
        y_train = X_aggregated.loc[mask, column]
        
        if len(X_train) < 10:  # Pas assez de données pour entraîner
            warnings.warn(f"Insufficient data for training imputation model for {column}")
            return column, None
        
        # Sélection du modèle selon le type de variable
        var_type = self.variable_types_[column]
        if var_type == 'continuous':
            model = clone(self.estimator) if self.estimator else None
        else:
            model = clone(self.classifier) if self.classifier else None
        
        if model is None:
            # Utilisation de stratégies simples par défaut
            return column, f"simple_{var_type}"
        
        # Application du transformer si fourni
        if self.transformer:
            X_train = self.transformer.fit_transform(X_train)
        
        # Entraînement du modèle
        try:
            # Si le modèle ne tolère pas les NaN, imputation préalable simple
            if not self.handle_nan:
                X_train = X_train.fillna(X_train.mean() if var_type == 'continuous' else X_train.mode().iloc[0])
            
            model.fit(X_train, y_train)
            return column, model
        except Exception as e:
            warnings.warn(f"Failed to train imputation model for {column}: {str(e)}")
            return column, None
    
    def _aggregate_high_frequency(self, 
                                 X: pd.DataFrame, 
                                 columns: List[str]) -> pd.DataFrame:
        """Aggregate high-frequency variables to target frequency.
        
        Args:
            X: Input data
            columns: Columns to aggregate
            
        Returns:
            Data with aggregated columns
        """
        X_agg = X.copy()
        
        # Création d'un resampler selon la fréquence cible
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'annual': 'A'
        }
        
        target_freq_pandas = freq_map.get(self.target_frequency, 'D')
        
        # Agrégation selon le type de données (panel ou série temporelle)
        if self.panel_cols:
            # Agrégation par groupe panel
            for panel_values, group_df in X.groupby(self.panel_cols):
                group_idx = group_df.index
                
                # Configuration de l'index temporel
                if self.time_col in group_df.columns:
                    group_df = group_df.set_index(self.time_col)
                
                # Agrégation pour chaque colonne haute fréquence
                for col in columns:
                    if col in group_df.columns:
                        agg_func = self.aggregation_functions_.get(col, 'mean')
                        
                        # Resampling et agrégation
                        aggregated = group_df[col].resample(target_freq_pandas).agg(agg_func)
                        
                        # Réindexation sur les dates originales (approximation)
                        # TODO: améliorer l'alignement des dates
                        X_agg.loc[group_idx, col] = aggregated.reindex(group_df.index, method='nearest')
        else:
            # Agrégation pour série temporelle simple
            if self.time_col in X.columns:
                X_temp = X.set_index(self.time_col)
            else:
                X_temp = X
            
            for col in columns:
                if col in X_temp.columns:
                    agg_func = self.aggregation_functions_.get(col, 'mean')
                    aggregated = X_temp[col].resample(target_freq_pandas).agg(agg_func)
                    X_agg[col] = aggregated.reindex(X_temp.index, method='nearest')
        
        return X_agg
    
    def _impute_low_frequency(self, 
                             X: pd.DataFrame, 
                             columns: List[str]) -> pd.DataFrame:
        """Impute low-frequency variables to target frequency.
        
        Args:
            X: Input data
            columns: Columns to impute
            
        Returns:
            Data with imputed columns
        """
        X_imputed = X.copy()
        
        # Imputation progressive par ordre croissant de délais
        # Récupération des délais si disponibles
        delay_order = self._get_imputation_order(columns)
        
        for col in delay_order:
            if col in self.imputation_models_:
                model = self.imputation_models_[col]
                
                if isinstance(model, str) and model.startswith('simple_'):
                    # Stratégie simple d'imputation
                    X_imputed = self._simple_impute(X_imputed, col, model)
                else:
                    # Imputation par modèle ML
                    X_imputed = self._model_impute(X_imputed, col, model)
            else:
                # Forward fill par défaut
                X_imputed[col] = X_imputed[col].fillna(method='ffill')
        
        return X_imputed
    
    def _get_imputation_order(self, columns: List[str]) -> List[str]:
        """Determine optimal order for progressive imputation.
        
        Args:
            columns: Columns to order
            
        Returns:
            Ordered list of columns
        """
        # Ordre basé sur le nombre de valeurs manquantes (du moins au plus)
        missing_counts = {}
        for col in columns:
            if col in self.frequency_map_:
                # Estimation basée sur le ratio de fréquences
                source_freq = self.frequency_map_[col]
                try:
                    ratio = self._freq_converter.get_conversion_factor(
                        source_freq, self.target_frequency
                    )
                    missing_counts[col] = ratio
                except:
                    missing_counts[col] = float('inf')
        
        # Tri par nombre croissant de valeurs manquantes estimées
        return sorted(columns, key=lambda x: missing_counts.get(x, float('inf')))
    
    def _simple_impute(self, 
                      X: pd.DataFrame, 
                      column: str, 
                      strategy: str) -> pd.DataFrame:
        """Apply simple imputation strategy.
        
        Args:
            X: Input data
            column: Column to impute
            strategy: Imputation strategy
            
        Returns:
            Data with imputed column
        """
        var_type = strategy.replace('simple_', '')
        
        if var_type == 'continuous':
            # Interpolation linéaire
            X[column] = X[column].interpolate(method='linear', limit_direction='both')
        elif var_type == 'binary' or var_type == 'categorical':
            # Forward fill pour les catégories
            X[column] = X[column].fillna(method='ffill')
            # Backward fill pour les valeurs restantes
            X[column] = X[column].fillna(method='bfill')
        
        return X
    
    def _model_impute(self, 
                     X: pd.DataFrame, 
                     column: str, 
                     model: BaseEstimator) -> pd.DataFrame:
        """Impute using fitted ML model.
        
        Args:
            X: Input data
            column: Column to impute
            model: Fitted model
            
        Returns:
            Data with imputed column
        """
        # Identification des valeurs manquantes
        missing_mask = X[column].isna()
        
        if not missing_mask.any():
            return X
        
        # Préparation des features
        feature_cols = [c for c in X.columns 
                       if c != column and c != self.time_col 
                       and (not self.panel_cols or c not in self.panel_cols)]
        
        X_features = X.loc[missing_mask, feature_cols]
        
        # Application du transformer si nécessaire
        if self.transformer:
            X_features = self.transformer.transform(X_features)
        
        # Gestion des NaN dans les features si nécessaire
        if not self.handle_nan:
            # Imputation simple des features
            for col in X_features.columns:
                if X_features[col].isna().any():
                    if self.variable_types_.get(col, 'continuous') == 'continuous':
                        X_features[col] = X_features[col].fillna(X_features[col].mean())
                    else:
                        mode_val = X_features[col].mode()
                        if len(mode_val) > 0:
                            X_features[col] = X_features[col].fillna(mode_val[0])
        
        # Prédiction
        try:
            predictions = model.predict(X_features)
            X.loc[missing_mask, column] = predictions
        except Exception as e:
            warnings.warn(f"Failed to impute {column} using model: {str(e)}")
            # Fallback sur stratégie simple
            X = self._simple_impute(X, column, f"simple_{self.variable_types_[column]}")
        
        return X
    
    def _aggregate_to_frequency(self, 
                               X: pd.DataFrame, 
                               target_freq: str) -> pd.DataFrame:
        """Aggregate entire dataset to a specific frequency.
        
        Args:
            X: Input data
            target_freq: Target frequency
            
        Returns:
            Aggregated data
        """
        # Utilisation temporaire d'une copie du transformer avec la fréquence cible
        temp_transformer = MixedFrequencyTransformer(
            target_frequency=target_freq,
            time_col=self.time_col,
            panel_cols=self.panel_cols
        )
        
        # Copie des attributs nécessaires
        temp_transformer.frequency_map_ = self.frequency_map_
        temp_transformer.variable_types_ = self.variable_types_
        temp_transformer.aggregation_functions_ = self.aggregation_functions_
        temp_transformer.is_fitted_ = True
        
        # Agrégation uniquement (pas d'imputation)
        high_freq_cols = [col for col, freq in self.frequency_map_.items()
                         if self._freq_converter.is_higher_frequency(freq, target_freq)]
        
        if high_freq_cols:
            return temp_transformer._aggregate_high_frequency(X, high_freq_cols)
        
        return X
    
    def _align_to_target_frequency(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align the dataset to the target frequency grid.
        
        Args:
            X: Input data
            
        Returns:
            Aligned data
        """
        # Création d'un index régulier à la fréquence cible
        if self.time_col in X.columns:
            time_index = X[self.time_col]
        else:
            time_index = X.index
        
        aligned_index = self._freq_converter.align_to_frequency(
            time_index, self.target_frequency
        )
        
        # Réindexation du DataFrame
        # TODO: Améliorer cette logique pour mieux gérer les données panel
        
        return X
    
    def _transform_with_enrichment(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform with data enrichment for training.
        
        Args:
            X: Input data without delays
            
        Returns:
            Enriched transformed data
        """
        # Application de la transformation standard
        X_transformed = self.transform(X)
        
        # Enrichissement supplémentaire pour les données d'entraînement
        # Ajout de features dérivées, lags, etc.
        # TODO: Implémenter des stratégies d'enrichissement
        
        return X_transformed


class VariableTypeDetector:
    """Detect variable types in time series data."""
    
    def __init__(self, 
                 binary_threshold: int = 2,
                 categorical_threshold: int = 10,
                 numeric_threshold: float = 0.95):
        """Initialize the detector.
        
        Args:
            binary_threshold: Max unique values for binary classification
            categorical_threshold: Max unique values for categorical classification
            numeric_threshold: Min ratio of numeric values for continuous classification
        """
        self.binary_threshold = binary_threshold
        self.categorical_threshold = categorical_threshold
        self.numeric_threshold = numeric_threshold
    
    def detect_type(self, series: pd.Series) -> str:
        """Detect the type of a variable.
        
        Args:
            series: Data series to analyze
            
        Returns:
            Variable type: 'continuous', 'binary', or 'categorical'
        """
        # Suppression des valeurs manquantes pour l'analyse
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return 'continuous'  # Par défaut
        
        # Nombre de valeurs uniques
        n_unique = clean_series.nunique()
        
        # Détection des variables binaires
        if n_unique <= self.binary_threshold:
            return 'binary'
        
        # Tentative de conversion numérique
        try:
            numeric_series = pd.to_numeric(clean_series, errors='coerce')
            numeric_ratio = numeric_series.notna().sum() / len(clean_series)
            
            if numeric_ratio >= self.numeric_threshold:
                # Variable principalement numérique
                return 'continuous'
            elif n_unique <= self.categorical_threshold:
                # Variable catégorielle
                return 'categorical'
            else:
                # Trop de catégories, traiter comme continue
                return 'continuous'
        except:
            # Échec de conversion, vérifier le nombre de catégories
            if n_unique <= self.categorical_threshold:
                return 'categorical'
            else:
                return 'continuous'
    
    def get_appropriate_estimator(self, 
                                 var_type: str,
                                 default_estimator: Optional[BaseEstimator] = None,
                                 default_classifier: Optional[BaseEstimator] = None) -> Optional[BaseEstimator]:
        """Get appropriate estimator for variable type.
        
        Args:
            var_type: Variable type
            default_estimator: Default estimator for continuous variables
            default_classifier: Default classifier for categorical variables
            
        Returns:
            Appropriate estimator or None
        """
        if var_type == 'continuous':
            if default_estimator is not None:
                return clone(default_estimator)
            else:
                # Utilisation d'un estimateur simple par défaut
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # binary or categorical
            if default_classifier is not None:
                return clone(default_classifier)
            else:
                # Utilisation d'un classifieur simple par défaut
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)