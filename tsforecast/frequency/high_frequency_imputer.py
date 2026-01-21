"""High frequency imputer for mixed frequency time series data.

This module provides the HighFrequencyImputer class to impute high-frequency values
for low-frequency series in mixed-frequency datasets using machine learning models.
"""
# Importation des modules
# Modules de base
import warnings
from typing import Dict, List, Literal, Optional, Union, Any, Tuple
# Manipulation de données
import numpy as np
import pandas as pd
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
# Utilitaires du package
from ..xy.transformers import XYPanelTimeSeriesTransformer
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import (
    normalize_frequency,
    is_higher_frequency,
    get_frequency_order,
)
from ..panel.utils import get_unique_panel_entities
from .detector import FrequencyDetector, detect_frequency


# Type aliases
VariableCategory = Literal['aggregate', 'impute', 'target_freq']


# Classe d'imputation des valeurs de variables
class HighFrequencyImputer(XYPanelTimeSeriesTransformer):
    """Impute high-frequency values for low-frequency series in mixed-frequency datasets.

    This XY transformer handles mixed-frequency datasets by:
    1. Making data additive via a user-provided transformer
    2. Aggregating high-frequency variables to the target frequency
    3. Imputing low-frequency variables
    4. Learning ML relationships between variables
    5. Imputing missing sub-period values
    6. Handling publication delays if provided

    Parameters:
        target_frequency: Target frequency for imputation. Can be:
            - str: Single frequency applied to all series/entities (e.g., 'M', 'Q', 'monthly')
            - Dict[entity_id, str]: Entity-specific target frequencies for panel data
            Must not be higher than the lowest frequency in the data.
        additive_transformer: Transformer to make data additive before imputation
            (e.g., log transformer, differencing). Must support fit_transform()
            and inverse_transform(). If None, data is assumed to already be additive.
        estimator: Estimator(s) for prediction. Can be:
            - Single estimator: Applied to all variables
            - Dict[variable_name, estimator]: Variable-specific models
        delays: Publication delays DataFrame with columns:
            - variable: Variable name
            - delay: Delay value
            - unit: Delay unit ('D', 's', etc.)
            - reference_point: 'start' or 'end'
            If None, no delay handling unless impute_delayed_values=True.
        impute_delayed_values: Whether to impute values affected by publication
            delays. Default is False. If True and delays=None, attempts to infer
            delays from trailing NaN patterns.
        on_frequency_mismatch: How to handle cases where target_frequency is higher
            than data frequencies. Options:
            - 'error': Raise ValueError (default)
            - 'warn': Issue warning and adjust target_frequency to highest available

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
        adjusted_target_frequency_: Actual target frequency used after validation
            (may differ from input if on_frequency_mismatch='warn').

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
        ...     estimator=LinearRegression()
        ... )
        >>> imputed = imputer.fit_transform(df)
    """
    # Initialisation
    def __init__(
        self,
        target_frequency: Union[str, Dict[Union[str, tuple], str]],
        estimator: Union[BaseEstimator, Dict[str, BaseEstimator]],
        additive_transformer: Optional[TransformerMixin] = None,
        impute_delayed_values: bool = False,
        delays: Optional[pd.DataFrame] = None,
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None
    ):
        """Initialize the HighFrequencyImputer."""
        # Initialisation du parent
        super().__init__(time_col=time_col, panel_cols=panel_cols, validate_input=True, strict_validation=True, auto_sort=False, convert_cols_to_index=True)
        # Validation des paramètres
        # Validation de la fréquence cible
        target_frequency = self._validate_target_frequency_format(target_frequency)
        
        # Validation de l'estimateur
        self._validate_estimator(estimator)
        
        # Validation du transformer additif
        if additive_transformer is not None:
            self._validate_additive_transformer(additive_transformer)

        # Validation des délais de publication
        if delays is not None:
            required_cols = ['variable', 'delay', 'unit', 'reference_point']
            missing_cols = set(required_cols) - set(delays.columns)
            if missing_cols:
                raise ValueError(
                    f"delays DataFrame missing required columns: {missing_cols}"
                )
        
        # Validation du paramètre on_frequency_mismatch
        if on_frequency_mismatch not in ['error', 'warn']:
            raise ValueError(
                f"on_frequency_mismatch must be 'error' or 'warn', "
                f"got '{on_frequency_mismatch}'"
            )

        # Instanciation des attributs
        self.target_frequency = target_frequency
        self.additive_transformer = additive_transformer
        self.estimator = estimator
        self.delays = delays
        self.impute_delayed_values = impute_delayed_values
        self.on_frequency_mismatch = on_frequency_mismatch

    # Validation du format de la fréquence cible
    def _validate_target_frequency_format(
        self, 
        target_frequency: Union[str, Dict[Union[str, tuple], str]]
    ) -> Union[str, Dict[Union[str, tuple], str]]:
        """Validate the format and values of target_frequency parameter.

        Args:
            target_frequency: Target frequency (string or dict mapping entities to frequencies)

        Raises:
            ValueError: If target_frequency format is invalid or contains invalid frequencies
        """
        # Cas d'une fréquence unique (string)
        if isinstance(target_frequency, str):
            try:
                return normalize_frequency(target_frequency)
            except ValueError as e:
                raise ValueError(f"Invalid target_frequency '{target_frequency}': {e}")
        
        # Cas d'un dictionnaire de fréquences par entité
        elif isinstance(target_frequency, dict):
            # Vérification que le dictionnaire est non vide
            if not target_frequency:
                raise ValueError("target_frequency dict cannot be empty")
            # Initialisation du dictionnaire de fréquences validées
            validated_freqs = {}
            # Initialisation du dictionnaire des fréquences invalides
            invalid_freqs = {}

            # Validation de chaque fréquence dans le dictionnaire
            for entity, freq in target_frequency.items():
                if not isinstance(freq, str):
                    raise ValueError(
                        f"Frequency for entity '{entity}' must be a string, "
                        f"got {type(freq).__name__}"
                    )
                try:
                    validated_freqs[entity] = normalize_frequency(freq)
                except ValueError as e:
                    invalid_freqs[entity] = str(e)
            
            # Levée d'erreur si des fréquences invalides
            if invalid_freqs:
                # Construction du message d'erreur
                error_msg = "Invalid frequencies in target_frequency dict:\n"
                for entity, error in invalid_freqs.items():
                    error_msg += f"  - Entity '{entity}': {error}\n"
                # Erreur
                raise ValueError(error_msg.rstrip())
            else:
                return validated_freqs
        
        # Format invalide
        else:
            raise TypeError(
                f"target_frequency must be a string or dict, "
                f"got {type(target_frequency).__name__}"
            )

    # Validation de l'estimateur via duck typing
    def _validate_estimator(
        self, 
        estimator: Union[BaseEstimator, Dict[str, BaseEstimator]]
    ) -> None:
        """Validate estimator has required methods (fit and predict).

        Args:
            estimator: Estimator or dict of estimators to validate

        Raises:
            ValueError: If estimator lacks required methods
        """
        # Cas d'un dictionnaire d'estimateurs
        if isinstance(estimator, dict):
            # Vérification que le dictionnaire n'est pas vide
            if not estimator:
                raise ValueError("estimator dict cannot be empty")
            
            # Validation de chaque estimateur
            for var_name, est in estimator.items():
                if not hasattr(est, 'fit') or not callable(getattr(est, 'fit')):
                    raise ValueError(
                        f"Estimator for '{var_name}' must have a 'fit' method, "
                        f"got {type(est).__name__}"
                    )
                if not hasattr(est, 'predict') or not callable(getattr(est, 'predict')):
                    raise ValueError(
                        f"Estimator for '{var_name}' must have a 'predict' method, "
                        f"got {type(est).__name__}"
                    )
        
        # Cas d'un estimateur unique
        else:
            if not hasattr(estimator, 'fit') or not callable(getattr(estimator, 'fit')):
                raise ValueError(
                    f"estimator must have a 'fit' method, "
                    f"got {type(estimator).__name__}"
                )
            if not hasattr(estimator, 'predict') or not callable(getattr(estimator, 'predict')):
                raise ValueError(
                    f"estimator must have a 'predict' method, "
                    f"got {type(estimator).__name__}"
                )

    # Validation du transformer additif via duck typing
    def _validate_additive_transformer(
        self, 
        transformer: TransformerMixin
    ) -> None:
        """Validate additive_transformer has required methods.

        Args:
            transformer: Transformer to validate

        Raises:
            ValueError: If transformer lacks required methods
        """
        # Initialisation de la liste des méthodes requises
        required_methods = ['fit', 'transform', 'inverse_transform']
        # Initialisation de la liste des méthodes manquantes
        missing_methods = []

        # Vérification de la présence des méthodes requises
        for method_name in required_methods:
            if not hasattr(transformer, method_name) or not callable(getattr(transformer, method_name)):
                missing_methods.append(method_name)
        
        # Levée d'erreur si des méthodes manquent
        if missing_methods:
            raise ValueError(
                f"additive_transformer must have methods: {', '.join(required_methods)}. "
                f"Missing: {', '.join(missing_methods)}. "
                f"Got {type(transformer).__name__}"
            )

    # Détecteur de fréquence
    @property
    def _freq_detector(self) -> FrequencyDetector:
        """Lazy initialization of frequency detector."""
        # Initialisation d'une instance du détecteur de fréquence si elle n'existe pas déjà 
        if not hasattr(self, '_freq_detector_cache'):
            self._freq_detector_cache = FrequencyDetector()
        # Retourne cette instance
        return self._freq_detector_cache

    # Convertisseur de fréquences
    @property
    def _freq_converter(self) -> FrequencyConverter:
        """Lazy initialization of frequency converter."""
        # Initialisation d'une instance du convertisseur de fréquence si elle n'existe pas déjà 
        if not hasattr(self, '_freq_converter_cache'):
            self._freq_converter_cache = FrequencyConverter()
        # Retourne cette instance
        return self._freq_converter_cache

    # Obtention de la fréquence la plus élevée pour les séries temporelles
    def _get_highest_frequency_timeseries(
        self, 
        detected_frequencies: Dict[str, str]
    ) -> str:
        """Get the highest (most granular) frequency for time series data.

        Args:
            detected_frequencies: Dictionary mapping columns to frequencies

        Returns:
            Highest frequency string

        Raises:
            ValueError: If no valid frequencies found
        """
        # Extraction des fréquences valides
        valid_freqs = [freq for freq in detected_frequencies.values() if freq is not None]
        # Vérification que la liste est non vide
        if not valid_freqs:
            raise ValueError("No valid frequencies detected in the dataset")
        
        # Détermination de la fréquence avec l'ordre le plus bas (plus granulaire)
        freq_orders = {}
        for freq in set(valid_freqs):
            try:
                # Normalisation de la fréquence (extraction de la partie base)
                base_freq = freq.split('-')[0] if '-' in freq else freq
                freq_orders[freq] = get_frequency_order(base_freq)
            except ValueError:
                # Tentative avec la fréquence complète en cas d'échec
                try:
                    freq_orders[freq] = get_frequency_order(freq)
                except ValueError:
                    continue
        
        if not freq_orders:
            raise ValueError("Could not determine frequency order for detected frequencies")
        
        # Retour de la fréquence avec l'ordre le plus bas
        return min(freq_orders.keys(), key=lambda x: freq_orders[x])

    # Obtention de la fréquence la plus élevée pour une entité donnée
    def _get_highest_frequency_entity(
        self, 
        entity: Union[str, tuple],
        detected_frequencies: Dict[Tuple[Union[str, tuple], str], str]
    ) -> str:
        """Get the highest frequency for a specific entity in panel data.

        Args:
            entity: Entity identifier
            detected_frequencies: Dictionary mapping (entity, variable) to frequency

        Returns:
            Highest frequency for the entity

        Raises:
            ValueError: If no valid frequencies found for the entity
        """
        # Extraction des fréquences pour cette entité
        entity_freqs = {}
        for (ent, var), freq in detected_frequencies.items():
            if ent == entity and freq is not None:
                entity_freqs[var] = freq
        
        if not entity_freqs:
            raise ValueError(f"No valid frequencies detected for entity '{entity}'")
        
        # Utilisation de la méthode pour séries temporelles
        return self._get_highest_frequency_timeseries(entity_freqs)

    # Méthode auxiliaire de vérification de la fréquence cible
    def _validate_target_frequency(
        self
    ) -> Union[str, Dict[Union[str, tuple], str]]:
        """Validate that target frequency is appropriate for the data.

        For time series: Validates target frequency is not higher than the highest
        frequency in the data.
        
        For panel data: Validates each entity has at least one series with frequency
        >= target frequency.

        Args:
            detected_frequencies: Dictionary of detected frequencies per column or
                (entity, column) for panel data.

        Returns:
            Adjusted target frequency (may differ from input if on_frequency_mismatch='warn')

        Raises:
            ValueError: If target frequency is invalid and on_frequency_mismatch='error'
        """
        # Cas 1: Données de séries temporelles simples
        if not self.is_panel_:
            return self._validate_target_frequency_timeseries()
        
        # Cas 2: Données de panel
        else:
            return self._validate_target_frequency_panel()

    # Validation de la fréquence cible pour les séries temporelles
    def _validate_target_frequency_timeseries(
        self
    ) -> str:
        """Validate target frequency for time series data.

        Args:
            detected_frequencies: Dictionary mapping columns to frequencies

        Returns:
            Adjusted target frequency

        Raises:
            ValueError: If target frequency is invalid and on_frequency_mismatch='error'
        """
        # Vérification que target_frequency est un string
        if isinstance(self.effective_target_frequency, dict):
            raise ValueError(
                "target_frequency cannot be a dict for simple time series data. "
                "Use a string frequency instead."
            )
        
        # Obtention de la fréquence la plus élevée
        highest_freq = self._get_highest_frequency_timeseries(self.detected_frequencies_)
        
        # Vérification si la fréquence cible est plus haute que la plus haute fréquence
        if is_higher_frequency(self.effective_target_frequency, highest_freq):
            # Construction du message d'erreur
            error_msg = (
                f"Target frequency '{self.effective_target_frequency}' is higher than "
                f"the highest frequency '{highest_freq}' in the data. "
                f"Target frequency must be equal to or lower than the highest frequency."
            )
            
            # Gestion selon le paramètre on_frequency_mismatch
            if self.on_frequency_mismatch == 'error':
                raise ValueError(error_msg)
            else:  # 'warn'
                warnings.warn(
                    f"{error_msg} Adjusting target_frequency to '{highest_freq}'.",
                    UserWarning
                )
                return highest_freq
        
        # Fréquence cible valide
        return self.self.effective_target_frequency

    # Validation de la fréquence cible pour les données de panel
    def _validate_target_frequency_panel(
        self
    ) -> Union[str, Dict[Union[str, tuple], str]]:
        """Validate target frequency for panel data.

        Args:
            detected_frequencies: Dictionary mapping (entity, variable) to frequency

        Returns:
            Adjusted target frequency (dict or string)

        Raises:
            ValueError: If target frequency is invalid and on_frequency_mismatch='error'
        """
        # Vérification que toutes les entités ont une fréquence cible
        missing_entities = set(self.entities_) - set(self.effective_target_frequency.keys())
        if missing_entities:
            raise ValueError(
                f"target_frequency dict is missing entries for entities: "
                f"{missing_entities}"
            )
        
        # Vérification des entités supplémentaires dans target_frequency
        extra_entities = set(self.effective_target_frequency.keys()) - set(self.entities_)
        if extra_entities:
            warnings.warn(
                f"target_frequency dict contains entries for entities not in data: "
                f"{extra_entities}. These will be ignored.",
                UserWarning
            )
        
        # Validation de chaque fréquence cible par entité
        # Initialisation de la liste des entités invalides
        invalid_entities = []
        # Initialisation du dictionnaire des fréquences ajustées
        adjusted_freqs = {}
        
        # Parcours des entités
        for entity in self.entities_:
            # Extraction de la fréquence cible associée à l'entité
            target_freq = self.effective_target_frequency[entity]
            
            try:
                # Obtention de la fréquence la plus élevée pour cette entité
                highest_freq = self._get_highest_frequency_entity(entity, self.detected_frequencies_)
                
                # Vérification si la fréquence cible est plus haute
                if is_higher_frequency(target_freq, highest_freq):
                    # Ajout aux fréquences invalides
                    invalid_entities.append((entity, target_freq, highest_freq))
                    # Ajustement de la fréquence
                    adjusted_freqs[entity] = highest_freq
                else:
                    adjusted_freqs[entity] = target_freq
                    
            except ValueError as e:
                # Entité sans fréquence valide détectée
                warnings.warn(f"Entity '{entity}': {e}", UserWarning)
                continue
        
        # Traitement des entités invalides
        if invalid_entities:
            # Construction du message d'erreur
            error_msg = (
                f"Target frequencies are higher than highest frequencies for "
                f"{len(invalid_entities)} entities:\n"
            )
            for entity, target, highest in invalid_entities[:5]:
                error_msg += (
                    f"  - Entity '{entity}': target '{target}' > highest '{highest}'\n"
                )
            if len(invalid_entities) > 5:
                error_msg += f"  ... and {len(invalid_entities) - 5} more entities\n"
            
            # Gestion selon le paramètre on_frequency_mismatch
            if self.on_frequency_mismatch == 'error':
                raise ValueError(error_msg.rstrip())
            else:  # 'warn'
                warnings.warn(
                    f"{error_msg.rstrip()}\n"
                    f"Adjusting target frequencies to entity-specific highest frequencies.",
                    UserWarning
                )
                return adjusted_freqs
        
        # Fréquences cibles valides
        return adjusted_freqs

    # Méthode auxiliaire de classification des variables selon leur fréquence
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

    # Méthode d'entraînement
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Learn transformation parameters from X and y.

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_targets).
        """
        # Construction du jeu de données de travail
        if y is not None:
            # Vérification que X et y sont de même longueur
            if len(X) != len(y):
                raise ValueError("X and y should be of equal length")
            # Construction du jeu de données de travail en concaténant X et y
            X_work = pd.concat([X, y.to_frame()], axis=1)
        else:
            X_work = X
        
        # Identification des entités du jeu de données
        self.entities_ = get_unique_panel_entities(X)

        # Si le jeu de données est un jeu de données de panel et 'target_frequency' une chaîne de caractères, on , alors on 
        if self.is_panel_ and isinstance(self.target_frequency, str):
            self.effective_target_frequency = {entity: self.target_frequency for entity in self.entities_}
        else:
            self.effective_target_frequency = self.target_frequency.copy()

        # Détection des fréquences
        self.detected_frequencies_ = detect_frequency(data=X_work)

        # Validation de la fréquence cible
        self.effective_target_frequency = self.validate_target_frequency()


        ##################################################################################################################
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
        self.effective_target_frequency = self._validate_target_frequency(X=X_work)

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

    # -------------------------------------------------------------------------
    # Transformation des features X
    # -------------------------------------------------------------------------
    def _transform_X(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Transform features X.

        Args:
            X: Features to transform.
            y: Targets (optional, not used in this transformer but required
                by XYTransformerMixin interface).

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

    # -------------------------------------------------------------------------
    # Transformation des targets y
    # -------------------------------------------------------------------------
    def _transform_y(self, X: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        """Transform target y.

        Args:
            X: Features (for conditional transformations, not used here).
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

    # -------------------------------------------------------------------------
    # Transformation inverse des features X
    # -------------------------------------------------------------------------
    def _inverse_transform_X(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Inverse transform features.

        Args:
            X: Transformed features.
            y: Transformed targets (optional, not used here).

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

    # -------------------------------------------------------------------------
    # Transformation inverse des targets y
    # -------------------------------------------------------------------------
    def _inverse_transform_y(self, X: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        """Inverse transform targets.

        Args:
            X: Transformed features (optional, not used here).
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
