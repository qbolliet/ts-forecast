"""High frequency imputer for mixed frequency time series data.

This module provides the HighFrequencyImputer class to impute high-frequency values
for low-frequency series in mixed-frequency datasets using machine learning models.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from ..xy.transformers import XYTransformerMixin
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import (
    normalize_frequency,
    is_higher_frequency,
    get_frequency_order,
)
from .detector import FrequencyDetector, detect_frequency


# Type aliases
LowFreqHandling = Literal['interpolate', 'impute']
VariableCategory = Literal['aggregate', 'interpolate', 'impute', 'target_freq']


class HighFrequencyImputer(BaseEstimator, XYTransformerMixin):
    """Impute high-frequency values for low-frequency series in mixed-frequency datasets.

    This XY transformer handles mixed-frequency datasets by:
    1. Making data additive via a user-provided transformer
    2. Aggregating high-frequency variables to the target frequency
    3. Interpolating or imputing low-frequency variables
    4. Learning ML relationships between variables
    5. Imputing missing sub-period values
    6. Handling publication delays if provided

    Parameters:
        target_frequency: Target frequency for imputation. Must not be higher
            than the lowest frequency in the data. Examples: 'M', 'Q', 'monthly'.
        additive_transformer: Transformer to make data additive before imputation
            (e.g., log transformer, differencing). Must support fit_transform()
            and inverse_transform(). If None, data is assumed to already be additive.
        estimator: Estimator(s) for prediction. Can be:
            - Single estimator: Applied to all variables
            - Dict[variable_name, estimator]: Variable-specific models
            - None: Uses linear interpolation fallback
        low_frequency_handling: Strategy for variables at frequencies lower than
            target. Maps variable names to:
            - 'interpolate': Linear/time interpolation (no ML model)
            - 'impute': Use cascading ML imputation
            Default strategy for unlisted variables is 'interpolate'.
        delays: Publication delays DataFrame with columns:
            - variable: Variable name
            - delay: Delay value
            - unit: Delay unit ('D', 's', etc.)
            - reference_point: 'start' or 'end'
            If None, no delay handling unless impute_delayed_values=True.
        impute_delayed_values: Whether to impute values affected by publication
            delays. Default is False. If True and delays=None, attempts to infer
            delays from trailing NaN patterns.
        fit_per_entity: For panel data, whether to train separate models per
            entity (True) or a single model across all entities (False).
        time_col: Name of the time column.
        panel_cols: Panel identifier columns. If None, treats data as simple
            time series.

    Attributes:
        detected_frequencies_: Detected frequency per variable.
        variable_categories_: Category for each variable ('aggregate', 'interpolate',
            'impute', 'target_freq').
        imputation_order_: Order of variables for cascading imputation.
        imputation_models_: Fitted imputation models per variable.
        inferred_delays_: Delays inferred from NaN patterns.
        additive_transformer_: Fitted additive transformer.
        is_panel_: Whether data is panel data.
        feature_columns_: X columns (features).
        target_column_: y column if provided.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> from tsforecast.frequency import HighFrequencyImputer
        >>>
        >>> # Create mixed-frequency data
        >>> dates = pd.date_range('2023-01-01', periods=12, freq='M')
        >>> df = pd.DataFrame({
        ...     'monthly_var': range(12),
        ...     'quarterly_var': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        ... }, index=dates)
        >>>
        >>> # Impute quarterly to monthly
        >>> imputer = HighFrequencyImputer(
        ...     target_frequency='M',
        ...     estimator=LinearRegression(),
        ...     low_frequency_handling={'quarterly_var': 'impute'}
        ... )
        >>> imputed = imputer.fit_transform(df)
    """

    def __init__(
        self,
        target_frequency: str,
        additive_transformer: Optional[TransformerMixin] = None,
        estimator: Union[BaseEstimator, Dict[str, BaseEstimator], None] = None,
        low_frequency_handling: Optional[Dict[str, LowFreqHandling]] = None,
        delays: Optional[pd.DataFrame] = None,
        impute_delayed_values: bool = False,
        fit_per_entity: bool = False,
        time_col: str = 'date',
        panel_cols: Optional[List[str]] = None,
    ):
        """Initialize the HighFrequencyImputer."""
        self.target_frequency = target_frequency
        self.additive_transformer = additive_transformer
        self.estimator = estimator
        # Ne pas modifier les paramètres pour la compatibilité sklearn clone()
        self.low_frequency_handling = low_frequency_handling
        self.delays = delays
        self.impute_delayed_values = impute_delayed_values
        self.fit_per_entity = fit_per_entity
        self.time_col = time_col
        self.panel_cols = panel_cols

    @property
    def _freq_detector(self) -> FrequencyDetector:
        """Lazy initialization of frequency detector."""
        if not hasattr(self, '_freq_detector_cache'):
            self._freq_detector_cache = FrequencyDetector()
        return self._freq_detector_cache

    @property
    def _freq_converter(self) -> FrequencyConverter:
        """Lazy initialization of frequency converter."""
        if not hasattr(self, '_freq_converter_cache'):
            self._freq_converter_cache = FrequencyConverter()
        return self._freq_converter_cache

    def _validate_parameters(self) -> None:
        """Validate constructor parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Validation de target_frequency
        try:
            normalize_frequency(self.target_frequency)
        except ValueError as e:
            raise ValueError(f"Invalid target_frequency '{self.target_frequency}': {e}")

        # Validation de estimator
        if self.estimator is not None:
            if isinstance(self.estimator, dict):
                for var_name, est in self.estimator.items():
                    if not isinstance(est, BaseEstimator):
                        raise ValueError(
                            f"Estimator for '{var_name}' must be a sklearn BaseEstimator, "
                            f"got {type(est).__name__}"
                        )
            elif not isinstance(self.estimator, BaseEstimator):
                raise ValueError(
                    f"estimator must be a sklearn BaseEstimator or Dict[str, BaseEstimator], "
                    f"got {type(self.estimator).__name__}"
                )

        # Validation de low_frequency_handling
        valid_strategies = {'interpolate', 'impute'}
        low_freq_handling = self.low_frequency_handling or {}
        for var_name, strategy in low_freq_handling.items():
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy '{strategy}' for variable '{var_name}'. "
                    f"Must be one of {valid_strategies}"
                )

        # Validation de delays DataFrame
        if self.delays is not None:
            required_cols = {'variable', 'delay', 'unit', 'reference_point'}
            missing_cols = required_cols - set(self.delays.columns)
            if missing_cols:
                raise ValueError(
                    f"delays DataFrame missing required columns: {missing_cols}"
                )

    def _validate_fit_data(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Validate and prepare data for fitting.

        Args:
            X: Input features.
            y: Target variable (optional).

        Returns:
            Validated and prepared DataFrame.

        Raises:
            ValueError: If data validation fails.
        """
        # Vérification du type
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        # Vérification de l'index temporel
        if not isinstance(X.index, pd.DatetimeIndex):
            if self.time_col in X.columns:
                X = X.set_index(self.time_col)
            else:
                raise ValueError(
                    f"X must have a DatetimeIndex or a time column '{self.time_col}'"
                )

        # Vérification des panel_cols
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found in X: {missing_cols}")

        return X

    def _detect_frequencies(self, X: pd.DataFrame) -> Dict[str, str]:
        """Detect frequency for each column in the DataFrame.

        Args:
            X: Input DataFrame with DatetimeIndex.

        Returns:
            Dictionary mapping column names to detected frequencies.
        """
        frequencies = {}

        # Colonnes à exclure (panel identifiers)
        exclude_cols = set(self.panel_cols or [])

        for col in X.columns:
            if col in exclude_cols:
                continue

            try:
                # Détection de la fréquence pour cette colonne
                series = X[col].dropna()
                if len(series) < 2:
                    # Pas assez de données pour détecter la fréquence
                    warnings.warn(
                        f"Not enough data to detect frequency for column '{col}'"
                    )
                    continue

                freq = detect_frequency(series, literal=False)
                if freq:
                    # Normalisation de la fréquence (extraction base sans position)
                    freq_base = freq.split('-')[0]  # Supprime anchor comme -DEC
                    if freq_base.endswith('S') or freq_base.endswith('E'):
                        freq_base = freq_base[:-1]  # Supprime S/E
                    frequencies[col] = normalize_frequency(freq_base)
            except Exception as e:
                warnings.warn(f"Could not detect frequency for column '{col}': {e}")

        return frequencies

    def _validate_target_frequency(self, detected_frequencies: Dict[str, str]) -> None:
        """Validate that target frequency is not higher than any data frequency.

        Args:
            detected_frequencies: Dictionary of detected frequencies per column.

        Raises:
            ValueError: If target frequency is higher than any data frequency.
        """
        target_freq_normalized = normalize_frequency(self.target_frequency)

        for col, freq in detected_frequencies.items():
            # Vérifier si la fréquence cible est plus haute que la fréquence de la variable
            if is_higher_frequency(target_freq_normalized, freq):
                raise ValueError(
                    f"Target frequency '{self.target_frequency}' is higher than "
                    f"frequency '{freq}' of column '{col}'. Target frequency must be "
                    f"equal to or lower than all data frequencies."
                )

    def _classify_variables(
        self, detected_frequencies: Dict[str, str]
    ) -> Dict[str, VariableCategory]:
        """Classify each variable by its relationship to target frequency.

        Args:
            detected_frequencies: Dictionary of detected frequencies per column.

        Returns:
            Dictionary mapping column names to their category.
        """
        target_freq_normalized = normalize_frequency(self.target_frequency)
        categories: Dict[str, VariableCategory] = {}
        low_freq_handling = self.low_frequency_handling or {}

        for col, freq in detected_frequencies.items():
            # Comparaison des fréquences
            if is_higher_frequency(freq, target_freq_normalized):
                # Variable à fréquence plus haute -> agrégation nécessaire
                categories[col] = 'aggregate'
            elif freq == target_freq_normalized:
                # Variable à la fréquence cible
                categories[col] = 'target_freq'
            else:
                # Variable à fréquence plus basse -> selon low_frequency_handling
                strategy = low_freq_handling.get(col, 'interpolate')
                categories[col] = strategy  # type: ignore

        return categories

    def _determine_imputation_order(
        self,
        variable_categories: Dict[str, VariableCategory],
        detected_frequencies: Dict[str, str],
    ) -> List[str]:
        """Determine order of variables for cascading imputation.

        Variables are sorted by frequency (lowest frequency first) to ensure
        that dependencies are resolved before imputation.

        Args:
            variable_categories: Category for each variable.
            detected_frequencies: Detected frequency for each variable.

        Returns:
            Ordered list of variable names to impute.
        """
        # Filtrer les variables marquées pour imputation
        impute_vars = [
            col for col, cat in variable_categories.items() if cat == 'impute'
        ]

        # Trier par fréquence (plus basse d'abord = ordre numérique le plus élevé)
        impute_vars.sort(
            key=lambda col: get_frequency_order(detected_frequencies.get(col, 'D')),
            reverse=True,  # Ordre décroissant = fréquence la plus basse d'abord
        )

        return impute_vars

    def _get_estimator_for_variable(self, variable: str) -> Optional[BaseEstimator]:
        """Get the appropriate estimator for a variable.

        Args:
            variable: Variable name.

        Returns:
            Cloned estimator for the variable, or None if no estimator available.
        """
        if self.estimator is None:
            return None

        if isinstance(self.estimator, dict):
            est = self.estimator.get(variable)
            if est is None:
                # Fallback vers l'estimateur par défaut s'il existe
                est = self.estimator.get('__default__')
            return clone(est) if est is not None else None

        return clone(self.estimator)

    def _aggregate_to_target(
        self, X: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Aggregate high-frequency columns to target frequency.

        Args:
            X: Input DataFrame.
            columns: Columns to aggregate.

        Returns:
            DataFrame with aggregated columns.
        """
        if not columns:
            return X

        result = X.copy()
        target_freq = normalize_frequency(self.target_frequency)

        for col in columns:
            if col not in X.columns:
                continue

            # Agrégation par somme (données additives)
            aggregated = self._freq_converter.aggregate_to_lower_frequency(
                X[col], target_freq, method='sum'
            )

            # Réindexation sur l'index original
            result[col] = aggregated.reindex(X.index, method='ffill')

        return result

    def _interpolate_to_target(
        self, X: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Interpolate low-frequency columns to target frequency.

        Args:
            X: Input DataFrame.
            columns: Columns to interpolate.

        Returns:
            DataFrame with interpolated columns.
        """
        if not columns:
            return X

        result = X.copy()
        target_freq = normalize_frequency(self.target_frequency)

        for col in columns:
            if col not in X.columns:
                continue

            # Interpolation linéaire vers la fréquence cible
            interpolated = self._freq_converter.interpolate_to_higher_frequency(
                X[col], target_freq, method='linear', fill_method='ffill'
            )

            # Réindexation sur l'index original
            result[col] = interpolated.reindex(X.index, method='ffill')

        return result

    def _fit_imputation_models(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Fit imputation models for variables marked for imputation.

        Args:
            X: Training data (already transformed and aggregated).
            y: Target variable (optional).

        Returns:
            Dictionary mapping variable names to fitted models.
        """
        models: Dict[str, Any] = {}

        for variable in self.imputation_order_:
            estimator = self._get_estimator_for_variable(variable)
            if estimator is None:
                # Pas de modèle disponible, utiliser interpolation comme fallback
                warnings.warn(
                    f"No estimator available for variable '{variable}', "
                    f"using linear interpolation as fallback"
                )
                models[variable] = 'interpolate_fallback'
                continue

            # Préparation des features (toutes les autres colonnes)
            feature_cols = [
                c for c in X.columns
                if c != variable and c not in (self.panel_cols or [])
            ]

            if not feature_cols:
                warnings.warn(
                    f"No features available for imputing '{variable}', "
                    f"using linear interpolation as fallback"
                )
                models[variable] = 'interpolate_fallback'
                continue

            # Masque des observations non-NaN pour la variable cible
            mask = X[variable].notna()
            X_train = X.loc[mask, feature_cols]
            y_train = X.loc[mask, variable]

            if len(X_train) < 2:
                warnings.warn(
                    f"Not enough training data for variable '{variable}', "
                    f"using linear interpolation as fallback"
                )
                models[variable] = 'interpolate_fallback'
                continue

            # Imputation simple des NaN dans les features
            X_train = X_train.fillna(X_train.mean())

            if self.fit_per_entity and self.panel_cols:
                # Entraînement par entité
                models[variable] = self._fit_models_per_entity(
                    X, variable, feature_cols, estimator
                )
            else:
                # Entraînement global
                try:
                    estimator.fit(X_train, y_train)
                    models[variable] = {
                        'model': estimator,
                        'feature_cols': feature_cols,
                    }
                except Exception as e:
                    warnings.warn(
                        f"Failed to fit model for variable '{variable}': {e}. "
                        f"Using linear interpolation as fallback"
                    )
                    models[variable] = 'interpolate_fallback'

        return models

    def _fit_models_per_entity(
        self,
        X: pd.DataFrame,
        variable: str,
        feature_cols: List[str],
        estimator: BaseEstimator,
    ) -> Dict[str, Any]:
        """Fit separate models for each panel entity.

        Args:
            X: Training data.
            variable: Variable to impute.
            feature_cols: Feature column names.
            estimator: Base estimator to clone for each entity.

        Returns:
            Dictionary with entity-specific models.
        """
        entity_models: Dict[str, Any] = {}

        # Groupement par entité
        for entity_id, group in X.groupby(self.panel_cols):
            mask = group[variable].notna()
            X_train = group.loc[mask, feature_cols]
            y_train = group.loc[mask, variable]

            if len(X_train) < 2:
                warnings.warn(
                    f"Not enough training data for entity '{entity_id}' "
                    f"and variable '{variable}', using global model"
                )
                continue

            # Imputation simple des NaN
            X_train = X_train.fillna(X_train.mean())

            try:
                entity_estimator = clone(estimator)
                entity_estimator.fit(X_train, y_train)
                entity_models[entity_id] = entity_estimator
            except Exception as e:
                warnings.warn(
                    f"Failed to fit model for entity '{entity_id}': {e}"
                )

        return {
            'entity_models': entity_models,
            'feature_cols': feature_cols,
            'base_estimator': estimator,
        }

    def _infer_delays_from_nan_patterns(self, X: pd.DataFrame) -> Dict[str, float]:
        """Infer publication delays from trailing NaN patterns.

        Args:
            X: Input DataFrame.

        Returns:
            Dictionary mapping variable names to inferred delays (in periods).
        """
        inferred_delays: Dict[str, float] = {}

        for col in X.columns:
            if col in (self.panel_cols or []):
                continue

            series = X[col]
            # Compter les NaN trailing
            trailing_nan_count = 0
            for val in series.iloc[::-1]:
                if pd.isna(val):
                    trailing_nan_count += 1
                else:
                    break

            if trailing_nan_count > 0:
                inferred_delays[col] = float(trailing_nan_count)

        return inferred_delays

    def _create_delays_df(self) -> pd.DataFrame:
        """Create delays DataFrame from inferred delays.

        Returns:
            DataFrame with delay information.
        """
        if not self.inferred_delays_:
            return pd.DataFrame(columns=['variable', 'delay', 'unit', 'reference_point'])

        records = []
        for variable, delay in self.inferred_delays_.items():
            records.append({
                'variable': variable,
                'delay': delay,
                'unit': 'periods',  # Délai en nombre de périodes
                'reference_point': 'end',
            })

        return pd.DataFrame(records)

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Learn transformation parameters from X and y.

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_targets).
        """
        # Validation des paramètres
        self._validate_parameters()

        # Validation et préparation des données
        X = self._validate_fit_data(X, y)

        # Stockage des colonnes
        self.feature_columns_ = list(X.columns)
        self.target_column_ = y.name if y is not None else None

        # Détection si données panel
        self.is_panel_ = bool(self.panel_cols)

        # Détection des fréquences
        self.detected_frequencies_ = self._detect_frequencies(X)

        if not self.detected_frequencies_:
            raise ValueError("Could not detect frequency for any column")

        # Validation de la fréquence cible
        self._validate_target_frequency(self.detected_frequencies_)

        # Classification des variables
        self.variable_categories_ = self._classify_variables(self.detected_frequencies_)

        # Détermination de l'ordre d'imputation
        self.imputation_order_ = self._determine_imputation_order(
            self.variable_categories_, self.detected_frequencies_
        )

        # Fit du transformer additif si fourni
        if self.additive_transformer is not None:
            self.additive_transformer_ = clone(self.additive_transformer)
            X_work = self.additive_transformer_.fit_transform(X)
            if isinstance(X_work, tuple):
                X_work = X_work[0]  # Si c'est un XY transformer
        else:
            self.additive_transformer_ = None
            X_work = X.copy()

        # Agrégation des variables haute fréquence
        aggregate_cols = [
            col for col, cat in self.variable_categories_.items() if cat == 'aggregate'
        ]
        X_work = self._aggregate_to_target(X_work, aggregate_cols)

        # Interpolation des variables marquées pour interpolation
        interpolate_cols = [
            col for col, cat in self.variable_categories_.items() if cat == 'interpolate'
        ]
        X_work = self._interpolate_to_target(X_work, interpolate_cols)

        # Entraînement des modèles d'imputation
        self.imputation_models_ = self._fit_imputation_models(X_work, y)

        # Inférence des délais si nécessaire
        if self.impute_delayed_values and self.delays is None:
            self.inferred_delays_ = self._infer_delays_from_nan_patterns(X)
        else:
            self.inferred_delays_ = {}

    def _transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features X.

        Args:
            X: Features to transform.

        Returns:
            Transformed features.
        """
        # Validation des données
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        # Stockage de l'original pour inverse_transform
        self._original_X_ = X.copy()

        # Préparation de l'index
        if not isinstance(X.index, pd.DatetimeIndex):
            if self.time_col in X.columns:
                X = X.set_index(self.time_col)
            else:
                raise ValueError("X must have a DatetimeIndex")

        # Application de la transformation additive
        if self.additive_transformer_ is not None:
            X_work = self.additive_transformer_.transform(X)
            if isinstance(X_work, tuple):
                X_work = X_work[0]
        else:
            X_work = X.copy()

        # Agrégation des variables haute fréquence
        aggregate_cols = [
            col for col, cat in self.variable_categories_.items() if cat == 'aggregate'
        ]
        X_work = self._aggregate_to_target(X_work, aggregate_cols)

        # Interpolation des variables marquées pour interpolation
        interpolate_cols = [
            col for col, cat in self.variable_categories_.items() if cat == 'interpolate'
        ]
        X_work = self._interpolate_to_target(X_work, interpolate_cols)

        # Imputation cascadée
        X_work = self._apply_cascading_imputation(X_work)

        # Gestion des délais
        if self.impute_delayed_values:
            X_work = self._impute_delayed_values(X_work)

        return X_work

    def _apply_cascading_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cascading imputation for variables marked for imputation.

        Args:
            X: DataFrame with aggregated and interpolated variables.

        Returns:
            DataFrame with imputed variables.
        """
        result = X.copy()

        for variable in self.imputation_order_:
            model_info = self.imputation_models_.get(variable)

            if model_info is None or model_info == 'interpolate_fallback':
                # Fallback vers interpolation linéaire
                result[variable] = result[variable].interpolate(
                    method='linear', limit_direction='both'
                )
                continue

            # Identification des valeurs manquantes
            missing_mask = result[variable].isna()
            if not missing_mask.any():
                continue

            feature_cols = model_info.get('feature_cols', [])
            X_features = result.loc[missing_mask, feature_cols]

            # Imputation des NaN dans les features
            X_features = X_features.fillna(X_features.mean())

            if 'entity_models' in model_info:
                # Prédiction par entité
                predictions = self._predict_per_entity(
                    model_info, X_features, result.loc[missing_mask]
                )
            else:
                # Prédiction globale
                model = model_info['model']
                try:
                    predictions = model.predict(X_features)
                except Exception as e:
                    warnings.warn(
                        f"Prediction failed for variable '{variable}': {e}. "
                        f"Using interpolation fallback."
                    )
                    result[variable] = result[variable].interpolate(
                        method='linear', limit_direction='both'
                    )
                    continue

            result.loc[missing_mask, variable] = predictions

        return result

    def _predict_per_entity(
        self,
        model_info: Dict[str, Any],
        X_features: pd.DataFrame,
        X_full: pd.DataFrame,
    ) -> np.ndarray:
        """Make predictions using entity-specific models.

        Args:
            model_info: Model information dictionary.
            X_features: Features for prediction.
            X_full: Full DataFrame with entity columns.

        Returns:
            Predictions array.
        """
        entity_models = model_info['entity_models']
        base_estimator = model_info.get('base_estimator')
        predictions = np.full(len(X_features), np.nan)

        # Groupement par entité
        for entity_id, group_idx in X_full.groupby(self.panel_cols).groups.items():
            # Filtrer les indices présents dans X_features
            common_idx = X_features.index.intersection(group_idx)
            if len(common_idx) == 0:
                continue

            X_entity = X_features.loc[common_idx]
            model = entity_models.get(entity_id)

            if model is None:
                # Utiliser le modèle de base s'il existe
                if base_estimator is not None:
                    try:
                        entity_pred = base_estimator.predict(X_entity)
                    except Exception:
                        entity_pred = np.full(len(X_entity), np.nan)
                else:
                    entity_pred = np.full(len(X_entity), np.nan)
            else:
                try:
                    entity_pred = model.predict(X_entity)
                except Exception:
                    entity_pred = np.full(len(X_entity), np.nan)

            # Assignation aux bonnes positions
            for i, idx in enumerate(common_idx):
                pos = X_features.index.get_loc(idx)
                predictions[pos] = entity_pred[i]

        return predictions

    def _impute_delayed_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute values affected by publication delays.

        Args:
            X: DataFrame with imputed values.

        Returns:
            DataFrame with delayed values imputed.
        """
        result = X.copy()

        # Utiliser les délais fournis ou inférés
        delays_to_use = self.delays if self.delays is not None else self._create_delays_df()

        if delays_to_use.empty:
            return result

        for _, row in delays_to_use.iterrows():
            variable = row['variable']
            delay = row['delay']

            if variable not in result.columns:
                continue

            # Identifier les positions affectées par le délai
            # (dernières 'delay' observations)
            n_delay = int(delay)
            if n_delay <= 0:
                continue

            delayed_idx = result.index[-n_delay:]
            missing_mask = result.loc[delayed_idx, variable].isna()

            if not missing_mask.any():
                continue

            # Imputation des valeurs retardées
            model_info = self.imputation_models_.get(variable)

            if model_info is None or model_info == 'interpolate_fallback':
                # Interpolation linéaire
                result[variable] = result[variable].interpolate(
                    method='linear', limit_direction='both'
                )
            else:
                feature_cols = model_info.get('feature_cols', [])
                X_features = result.loc[delayed_idx[missing_mask], feature_cols]

                if not X_features.empty:
                    X_features = X_features.fillna(X_features.mean())
                    try:
                        if 'entity_models' in model_info:
                            predictions = self._predict_per_entity(
                                model_info, X_features, result.loc[delayed_idx[missing_mask]]
                            )
                        else:
                            predictions = model_info['model'].predict(X_features)
                        result.loc[delayed_idx[missing_mask], variable] = predictions
                    except Exception as e:
                        warnings.warn(
                            f"Failed to impute delayed values for '{variable}': {e}"
                        )

        return result

    def _transform_y(self, y: pd.Series) -> pd.Series:
        """Transform target y.

        Args:
            y: Targets to transform.

        Returns:
            Transformed targets.
        """
        # Stockage de l'original
        self._original_y_ = y.copy()

        # Application de la transformation additive si applicable
        if self.additive_transformer_ is not None:
            # Certains transformers peuvent transformer y aussi
            if hasattr(self.additive_transformer_, 'transform'):
                try:
                    # Essayer de transformer y seul
                    y_transformed = y.copy()
                except Exception:
                    y_transformed = y.copy()
            else:
                y_transformed = y.copy()
        else:
            y_transformed = y.copy()

        return y_transformed

    def _inverse_transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform features.

        Args:
            X: Transformed features.

        Returns:
            Original features (at target frequency - disaggregation is lossy).
        """
        result = X.copy()

        # Application de l'inverse du transformer additif
        if self.additive_transformer_ is not None:
            if hasattr(self.additive_transformer_, 'inverse_transform'):
                try:
                    result = self.additive_transformer_.inverse_transform(result)
                    if isinstance(result, tuple):
                        result = result[0]
                except Exception as e:
                    warnings.warn(
                        f"Failed to inverse transform with additive transformer: {e}"
                    )

        return result

    def _inverse_transform_y(self, y: pd.Series) -> pd.Series:
        """Inverse transform targets.

        Args:
            y: Transformed targets.

        Returns:
            Original targets.
        """
        result = y.copy()

        # Application de l'inverse du transformer additif si applicable
        if self.additive_transformer_ is not None:
            if hasattr(self.additive_transformer_, 'inverse_transform'):
                try:
                    # Certains transformers peuvent inverse transform y
                    pass  # Pour l'instant, retourner tel quel
                except Exception:
                    pass

        return result
