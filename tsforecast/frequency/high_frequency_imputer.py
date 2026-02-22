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
from ..panel.utils import get_unique_panel_entities, normalize_entity_key
from .detector import FrequencyDetector, detect_frequency
from .provenance import ImputationProvenanceTracker, ProvenanceType
from .imputation_window import ImputationWindowCalculator, ImputationScope
from ..delays.data_manager import compare_and_detect_delays


# Type aliases
VariableCategory = Literal['aggregate', 'impute', 'target_freq']


# Classe d'imputation des valeurs de variables
class HighFrequencyImputer(XYPanelTimeSeriesTransformer):
    """Impute high-frequency values for low-frequency series in mixed-frequency datasets.

    This XY transformer handles mixed-frequency datasets using a cascading imputation
    approach that respects frequency hierarchies and tracks value provenance:

    1. Making data additive via a user-provided transformer
    2. Computing the P1 window (where all series have true values)
    3. Aggregating high-frequency variables to lower frequencies
    4. Cascading imputation from lowest to highest frequency
    5. Optionally refitting models with imputed values (cascade_refitting)
    6. Handling publication delays if provided
    7. Tracking provenance of each imputed value

    The cascade algorithm processes variables by frequency level, from lowest (e.g., quarterly)
    to highest (e.g., daily). At each level:
    - Features are aggregated to match the variable's frequency
    - Models are trained on the P1 window (optionally extended)
    - Predictions are made for missing values
    - If cascade_refitting=True, models are retrained after each frequency stage

    Parameters:
        target_frequency: Target frequency for imputation. Can be:
            - str: Single frequency applied to all series/entities
            - Dict[entity_id, str]: Entity-specific target frequencies for panel data
        estimator: Estimator(s) for prediction. Can be:
            - Single estimator: Applied to all variables
            - Dict[variable_name, estimator]: Variable-specific models
        additive_transformer: Transformer to make data additive before imputation.
        cascade_refitting: If True, refit models using imputed values after each
            frequency stage for cascade imputation.
        keep_lower_frequencies: If True, output includes all intermediate frequencies
            in a MultiIndex structure (Entity, Frequency, Date) for panel or
            (Frequency, Date) for time series.
        delays: Publication delays DataFrame with columns:
            column, delay, unit, reference_point.
        impute_delayed_values: Whether to impute values affected by publication delays.
        on_frequency_mismatch: How to handle target_frequency higher than data ('error'/'warn').
        attrition_threshold: Minimum ratio of columns with data (0-1) for extended window.
        imputation_scope: Training window scope ('strict', 'extended_backward',
            'extended_forward', 'extended_both').
        train_on_partial_coverage: If True, use imputed values for training outside P1.

    Attributes:
        detected_frequencies_: Detected frequency per variable or (entity, variable).
        variable_categories_: Category per (entity, variable) tuple:
            'aggregate', 'impute', or 'target_freq'.
        imputation_order_: Ordered list of variables for cascading imputation.
        imputation_models_: Fitted imputation models per variable.
        imputation_provenance_: DataFrame tracking origin of each value
            ('original', 'model_on_true', 'model_on_mixed', 'aggregated').
        imputation_window_: Tuple (start, end) of the P1 window where all series have data.
        training_window_: Tuple (start, end) of the extended training window.
        frequency_progression_: Dict mapping variables to their frequency stages.
        inferred_delays_: DataFrame with delays inferred from data using compare_and_detect_delays
            (if impute_delayed_values=True and delays=None).
        additive_transformer_: Fitted additive transformer.
        is_panel_: Whether data is panel data.
        feature_columns_: X columns (features).
        target_column_: y column if provided.
        effective_target_frequency_: Actual target frequency used after validation.
        entities_: Unique entities in panel data.

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
        >>> # Impute quarterly to monthly with cascade
        >>> imputer = HighFrequencyImputer(
        ...     target_frequency='M',
        ...     estimator=LinearRegression(),
        ...     cascade_refitting=True,
        ...     imputation_scope='extended_both',
        ...     attrition_threshold=0.5
        ... )
        >>> imputed = imputer.fit_transform(df)
        >>>
        >>> # Access provenance information
        >>> print(imputer.imputation_provenance_)
    """
    # Initialisation
    def __init__(
        self,
        target_frequency: Union[str, Dict[Union[str, tuple], str]],
        estimator: Optional[Union[BaseEstimator, Dict[str, BaseEstimator]]]=None,
        additive_transformer: Optional[TransformerMixin] = None,
        cascade_refitting: bool = True,
        keep_lower_frequencies: bool = True,
        impute_delayed_values: bool = False,
        delays: Optional[pd.DataFrame] = None,
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
        attrition_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        train_on_partial_coverage: bool = False,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
    ):
        """Initialize the HighFrequencyImputer.

        Args:
            target_frequency: Target frequency for imputation. Can be:
                - str: Single frequency applied to all series/entities (e.g., 'M', 'Q', 'monthly')
                - Dict[entity_id, str]: Entity-specific target frequencies for panel data
                Must not be higher than the lowest frequency in the data.
            estimator: Estimator(s) for prediction. Can be:
                - Single estimator: Applied to all variables
                - Dict[variable_name, estimator]: Variable-specific models, a model associated to '__default__' key can be provided
            additive_transformer: Transformer to make data additive before imputation
                (e.g., log transformer, differencing). Must support fit_transform()
                and inverse_transform(). If None, data is assumed to already be additive.
            cascade_refitting: If True, refit models using imputed values after each
                frequency stage. This enables more accurate imputation in later stages.
            keep_lower_frequencies: If True, output includes all intermediate frequencies
                in a MultiIndex structure. If False, only target frequency is returned.
            impute_delayed_values: Whether to impute values affected by publication
                delays. Default is False. If True and delays=None, attempts to infer
                delays from trailing NaN patterns.
            delays: Publication delays DataFrame with columns:
                - column: Column name
                - delay: Delay value
                - unit: Delay unit ('D', 's', etc.)
                - reference_point: 'start' or 'end'
                If None, no delay handling unless impute_delayed_values=True.
            on_frequency_mismatch: How to handle cases where target_frequency is higher
                than data frequencies. Options:
                - 'error': Raise ValueError (default)
                - 'warn': Issue warning and adjust target_frequency to highest available
            attrition_threshold: Minimum percentage of columns (0-1) that must have
                non-null values to be included in the extended training window.
                Minimum 2 columns required regardless of threshold. Default 0.5.
            imputation_scope: Defines the training window scope:
                - 'strict': Use only P1 window (where ALL series have data)
                - 'extended_backward': Extend P1 backwards where threshold is met
                - 'extended_forward': Extend P1 forwards where threshold is met
                - 'extended_both': Extend P1 in both directions
            train_on_partial_coverage: If True, use imputed values for training models
                outside P1 window. If False, only use true values for training.
            time_col: Name of the time column (if data has time in column not index).
            panel_cols: List of column names identifying panel entities.
        """
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
            required_cols = ['column', 'delay', 'unit', 'reference_point']
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

        # Validation du seuil d'attrition
        if not 0 <= attrition_threshold <= 1:
            raise ValueError(
                f"attrition_threshold must be between 0 and 1, got {attrition_threshold}"
            )

        # Validation du scope d'imputation
        valid_scopes = ('strict', 'extended_backward', 'extended_forward', 'extended_both')
        if imputation_scope not in valid_scopes:
            raise ValueError(
                f"imputation_scope must be one of {valid_scopes}, got '{imputation_scope}'"
            )

        # Instanciation des attributs
        self.target_frequency = target_frequency
        self.additive_transformer = additive_transformer
        self.estimator = estimator
        self.delays = delays
        self.cascade_refitting = cascade_refitting
        self.keep_lower_frequencies = keep_lower_frequencies
        self.impute_delayed_values = impute_delayed_values
        self.on_frequency_mismatch = on_frequency_mismatch
        self.attrition_threshold = attrition_threshold
        self.imputation_scope = imputation_scope
        self.train_on_partial_coverage = train_on_partial_coverage

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
        # Normalisation de l'entité
        normalized_entity = normalize_entity_key(entity)
        # Extraction des fréquences pour cette entité
        entity_freqs = {}
        for key, freq in detected_frequencies.items():
            # Extraction de la variable
            var = key[-1]
            # Extraction de l'entité
            ent = normalize_entity_key(key[:-1])
            if ent == normalized_entity and freq is not None:
                entity_freqs[var] = freq
        
        # Erreur si aucune fréquence n'est détectée
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
        return self.effective_target_frequency

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
        self
    ) -> Dict[Union[str, Tuple], VariableCategory]:
        """Classify each variable by its relationship to target frequency.

        For time series data, returns a Dict mapping column names to categories.
        For panel data, returns a Dict mapping (entity, variable) tuples to categories.

        Returns:
            Dictionary mapping variable identifiers to their category:
            - For time series: {column_name: category}
            - For panel: {(entity, variable): category}

        Categories:
            - 'aggregate': Variable has higher frequency than target, needs aggregation
            - 'impute': Variable has lower frequency than target, needs imputation
            - 'target_freq': Variable already at target frequency
        """
        # Initialisation du dictionnaire de catégories
        categories: Dict[Union[str, Tuple], VariableCategory] = {}

        # Cas de données de panel
        if self.is_panel_:
            # Parcours des fréquences détectées
            # Clé = (entity..., variable), valeur = fréquence
            for key, freq in self.detected_frequencies_.items():
                # Décomposition de la clé: les niveaux d'entité sont tous sauf le dernier
                # Le dernier élément est le nom de la variable
                if isinstance(key, tuple):
                    entity = key[:-1] if len(key) > 2 else key[0]
                    col = key[-1]
                else:
                    # Cas dégénéré: pas de tuple (ne devrait pas arriver pour des données de panel)
                    entity = None
                    col = key

                # Extraction de la fréquence cible pour cette entité
                if isinstance(self.effective_target_frequency, dict):
                    # Normalisation de la clé d'entité
                    entity_key = entity if isinstance(entity, tuple) else (entity,)
                    target_freq = self.effective_target_frequency.get(entity_key)
                    if target_freq is None:
                        # Essai avec l'entité non-tuplée
                        target_freq = self.effective_target_frequency.get(entity)
                else:
                    target_freq = self.effective_target_frequency

                if target_freq is None:
                    continue

                # Comparaison des fréquences (avec normalisation)
                freq_normalized = normalize_frequency(freq)
                target_normalized = normalize_frequency(target_freq)

                if is_higher_frequency(freq, target_freq):
                    # Variable à fréquence plus haute -> agrégation nécessaire
                    categories[key] = 'aggregate'
                elif freq_normalized == target_normalized:
                    # Variable à la fréquence cible (comparaison normalisée)
                    categories[key] = 'target_freq'
                else:
                    # Variable à fréquence plus basse -> imputation nécessaire
                    categories[key] = 'impute'

        # Cas de données de séries temporelles
        else:
            # Parcours des fréquences détectées
            for col, freq in self.detected_frequencies_.items():
                # Comparaison des fréquences (avec normalisation)
                freq_normalized = normalize_frequency(freq)
                target_normalized = normalize_frequency(self.effective_target_frequency)

                if is_higher_frequency(freq, self.effective_target_frequency):
                    # Variable à fréquence plus haute -> agrégation nécessaire
                    categories[col] = 'aggregate'
                elif freq_normalized == target_normalized:
                    # Variable à la fréquence cible (comparaison normalisée)
                    categories[col] = 'target_freq'
                else:
                    # Variable à fréquence plus basse -> imputation nécessaire
                    categories[col] = 'impute'

        return categories
    
    # Méthode auxiliaire de détermination de l'ordre d'imputation des variables
    def _determine_imputation_order(
        self
    ) -> List[Union[str, Tuple]]:
        """Determine order of variables for cascading imputation.

        Sorting logic:
        1. Sort by frequency (lowest frequency first, e.g., quarterly before monthly)
        2. For panel data with variable frequencies per entity:
           - First, variables with the lowest frequencies
           - Among those, variables affecting the fewest entities

        This ensures that:
        - Lower frequency variables are imputed first (they have more data points)
        - Variables are processed efficiently without redundant computations

        Returns:
            Ordered list of variable identifiers to impute.
            - For time series: List of column names
            - For panel: List of (entity, variable) tuples or variable names

        Examples:
            >>> # Time series: ['quarterly_var', 'monthly_var', 'weekly_var']
            >>> # Panel: [('A', 'quarterly_var'), ('B', 'quarterly_var'), ('A', 'monthly_var')]
        """
        # Extraction des variables marquées pour imputation
        impute_vars = [
            key for key, cat in self.variable_categories_.items() if cat == 'impute'
        ]

        if not impute_vars:
            return []

        # Cas de données de séries temporelles simples
        if not self.is_panel_:
            # Tri par fréquence (plus basse d'abord = ordre numérique le plus élevé)
            impute_vars.sort(
                key=lambda col: get_frequency_order(
                    self.detected_frequencies_.get(col, 'D')
                ),
                reverse=True,  # Ordre décroissant = fréquence la plus basse d'abord
            )
            return impute_vars

        # Cas de données de panel
        # Étape 1: Regroupement des variables par nom unique
        # Cela permet de déterminer quelles variables sont présentes dans plusieurs entités
        # Initialisation des dictionnaire résultats
        var_to_entities: Dict[str, List[Tuple]] = {}
        var_to_frequencies: Dict[str, List[float]] = {}

        # Parcours des variables à imputer
        for key in impute_vars:
            if isinstance(key, tuple):
                # Extraction du nom de variable (dernier élément)
                var_name = key[-1]
            else:
                var_name = key

            # Ajout de l'entité à la liste pour cette variable
            if var_name not in var_to_entities:
                var_to_entities[var_name] = []
                var_to_frequencies[var_name] = []

            var_to_entities[var_name].append(key)

            # Extraction de la fréquence pour cette variable/entité
            freq = self.detected_frequencies_.get(key, 'D')
            freq_order = get_frequency_order(freq)
            var_to_frequencies[var_name].append(freq_order)

        # Étape 2: Calcul des métriques de tri par variable
        # - Fréquence représentative (médiane des fréquences par entité)
        # - Fréquence la plus faible (maximimum des ordres de fréquences par entité)
        # - Fréquence moyenne (plus sensible aux valeurs extrêmes)
        # - Nombre d'entités affectées
        # Initialisation de la liste 
        var_metrics: List[Tuple[str, float, int, float, int]] = []

        # Parcours des variables
        for var_name in var_to_entities.keys():
            # Fréquence représentative: médiane des ordres de fréquence sur l'ensemble des entités
            freq_orders = var_to_frequencies[var_name]
            representative_freq = np.median(freq_orders)
            # Fréquence minimale
            min_freq = np.max(freq_orders)
            # Fréquence moyenne
            mean_freq = np.mean(freq_orders)

            # Nombre d'entités
            n_entities = len(var_to_entities[var_name])

            var_metrics.append((var_name, representative_freq, min_freq, mean_freq, n_entities))

        # Étape 3: Tri des variables
        # Critère 1: Fréquence la plus basse d'abord (ordre numérique le plus élevé)
        # Critère 2: Moins d'entités d'abord (en cas d'égalité de fréquence)
        var_metrics.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))  # -freq médiane pour décroissant, -freq minimale pour décroissant, -freq moyenne pour décroissant, +entities pour croissant

        # Étape 4: Construction de la liste finale ordonnée
        # Pour chaque variable, on ajoute toutes ses clés (entité, variable)
        # triées par fréquence locale
        # Initialisation de la 
        ordered_impute_vars = []

        # Parcours des variables
        for var_name, _, _, _, _ in var_metrics:
            # Récupération des clés pour cette variable
            var_keys = var_to_entities[var_name]

            # Tri interne par fréquence (plus basse d'abord)
            var_keys.sort(
                key=lambda k: get_frequency_order(self.detected_frequencies_.get(k, 'D')),
                reverse=True
            )

            ordered_impute_vars.extend(var_keys)

        return ordered_impute_vars

    # Méthode auxiliaire d'extraction de l'estimateur associé à une variable
    def _get_estimator_for_variable(self, variable: str) -> Optional[BaseEstimator]:
        """Get the appropriate estimator for a variable.

        Args:
            variable: Variable name.

        Returns:
            Cloned estimator for the variable, or None if no estimator available.
        """
        # Cas où l'estimateur n'est pas spécifié
        if self.estimator is None:
            return None

        # Cas où un estimateur est fourni pour différentes variables
        if isinstance(self.estimator, dict):
            # Extraction de l'estimateur
            est = self.estimator.get(variable)
            if est is None:
                # Fallback vers l'estimateur par défaut s'il existe
                est = self.estimator.get('__default__')
            # Clonage de l'estimateur
            return clone(est) if est is not None else None

        # Clonage de l'estimateur
        return clone(self.estimator)

    # Regroupement des clés par entité et variable
    def _group_keys_by_entity_and_variable(
        self,
        keys: List[Union[str, Tuple]]
    ) -> Dict[tuple, List[str]]:
        """Group variable keys by entity, extracting column names.

        Args:
            keys: List of (entity..., variable) tuples or plain column names.

        Returns:
            Dict mapping entity tuples to lists of column names.
            For time series (non-panel), returns {(): [col_names]}.
        """
        # Initialisation du dictionnaire résultat
        grouped: Dict[tuple, List[str]] = {}

        # Parcours des clés
        for key in keys:
            if isinstance(key, tuple):
                # Extraction de l'entité (tous les niveaux sauf le dernier)
                entity = normalize_entity_key(key[:-1])
                col = key[-1]
            else:
                # Séries temporelles : pas d'entité
                entity = ()
                col = key
            # Ajout de l'entité
            if entity not in grouped:
                grouped[entity] = []
            # Ajout de la colonne à l'entité
            if col not in grouped[entity]:
                grouped[entity].append(col)

        return grouped

    # Obtention de la fréquence cible pour une entité
    def _get_entity_target_frequency(
        self,
        entity: tuple
    ) -> str:
        """Get the target frequency for a specific entity.

        Args:
            entity: Entity tuple (e.g., ('FR',) or ('FR', 'GDP')).

        Returns:
            Normalized target frequency string for the entity.
        """
        # Cas d'un dictionnaire de fréquences
        if isinstance(self.effective_target_frequency, dict):
            # Essai avec le tuple complet
            freq = self.effective_target_frequency.get(entity)
            if freq is None and len(entity) == 1:
                # Essai avec l'entité extraite du tuple
                freq = self.effective_target_frequency.get(entity[0])
            # Cas d'erreur lorsque la fréquence n'est pas trouvée
            if freq is None:
                raise ValueError(
                    f"No target frequency found for entity {entity}"
                )
            return freq

        # Cas d'une fréquence unique (string)
        return self.effective_target_frequency

    # Obtention du masque booléen pour les lignes d'une entité
    def _get_entity_mask(
        self,
        X: pd.DataFrame,
        entity: tuple
    ) -> np.ndarray:
        """Get a boolean mask for rows belonging to a specific entity.

        Args:
            X: DataFrame with MultiIndex (entity levels + time level).
            entity: Entity tuple to filter on.

        Returns:
            Boolean numpy array of shape (len(X),).
        """
        # Extraction de l'index des données
        index = X.index
        # Nombre de niveaux d'entité = nombre total de niveaux - 1 (dernier = temps)
        n_entity_levels = index.nlevels - 1

        if n_entity_levels == 1:
            # Un seul niveau d'entité
            return (index.get_level_values(0) == entity[0])
        else:
            # Plusieurs niveaux d'entité : combinaison de masques
            mask = np.ones(len(X), dtype=bool)
            for i in range(n_entity_levels):
                mask &= (index.get_level_values(i) == entity[i])

            return mask

    # Méthode auxiliaire d'agrégation des variables à fréquence élevée à la fréquence cible
    def _aggregate_to_target(
        self, X: pd.DataFrame, aggregate_keys: List[Union[str, Tuple]]
    ) -> pd.DataFrame:
        """Aggregate high-frequency columns to target frequency.

        For panel data, aggregation is performed per entity to respect
        entity-specific target frequencies.

        Args:
            X: Input DataFrame.
            aggregate_keys: Variable keys to aggregate (column names or
                (entity..., variable) tuples).

        Returns:
            DataFrame with aggregated columns.
        """
        # Retourne le jeu de données original si aucune clé n'est spécifiée
        if not aggregate_keys:
            return X

        # Copie du jeu de données
        result = X.copy()

        # Cas séries temporelles : agrégation globale
        if not self.is_panel_:
            # Extraction des noms de colonnes
            columns = self._extract_column_names(aggregate_keys)
            
            # Parcours des colonnes
            for col in columns:
                if col not in X.columns:
                    continue
                # Agrégation par somme (données additives)
                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    X[col], self.effective_target_frequency, method='sum'
                )
                # Réindexation sur l'index original
                result[col] = aggregated.reindex(X.index)

            # Suppression des lignes de Nan éventuellement introduites
            result.dropna(axis=0, how='all', inplace=True)

            return result

        # Cas panel : agrégation par entité
        grouped = self._group_keys_by_entity_and_variable(aggregate_keys)

        # Parcours des entités
        for entity, cols in grouped.items():
            # Fréquence cible pour cette entité
            target_freq = self._get_entity_target_frequency(entity)
            # Masque des lignes de cette entité
            entity_mask = self._get_entity_mask(X, entity)

            # Parcours des colonnes
            for col in cols:
                if col not in X.columns:
                    continue

                # Extraction de la série pour cette entité
                entity_series = X.loc[entity_mask, col]
                # Suppression des niveaux panel pour obtenir un DatetimeIndex
                entity_series = entity_series.droplevel(
                    list(range(X.index.nlevels - 1))
                )

                # Agrégation par somme (données additives)
                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    entity_series, target_freq, method='sum'
                )

                # Réindexation sur l'index temporel original de l'entité
                reindexed = aggregated.reindex(entity_series.index)

                # Réassignation dans le DataFrame résultat
                result.loc[entity_mask, col] = reindexed.values

        # Suppression des lignes de Nan éventuellement introduites
        result.dropna(axis=0, how='all', inplace=True)

        return result

    # Méthode auxiliaire d'interpolation à la fréquence cible
    def _interpolate_to_target(
        self, X: pd.DataFrame, interpolate_keys: List[Union[str, Tuple]]
    ) -> pd.DataFrame:
        """Interpolate low-frequency columns to target frequency.

        For panel data, interpolation is performed per entity to respect
        entity-specific target frequencies.

        Args:
            X: Input DataFrame.
            interpolate_keys: Variable keys to interpolate (column names or
                (entity..., variable) tuples).

        Returns:
            DataFrame with interpolated columns.
        """
        # Retourne le jeu de données original si aucune clé n'est spécifiée
        if not interpolate_keys:
            return X

        # Copie du jeu de données
        result = X.copy()

        # Cas séries temporelles : interpolation globale
        if not self.is_panel_:
            # Extraction des noms de colonnes
            columns = self._extract_column_names(interpolate_keys)
            # Normalisation de la fréquence cible
            target_freq = normalize_frequency(self.effective_target_frequency)

            # Parcours des colonnes
            for col in columns:
                if col not in X.columns:
                    continue
                # Interpolation linéaire vers la fréquence cible
                # /!\ Faire que la fill method dépende de la position dans la string frequency
                interpolated = self._freq_converter.interpolate_to_higher_frequency(
                    X[col], target_freq, method='linear', fill_method='ffill'
                )
                # Réindexation sur l'index original
                # /!\ Faire que la fill method dépende de la position dans la string frequency
                result[col] = interpolated.reindex(X.index)

            return result

        # Cas panel : interpolation par entité
        grouped = self._group_keys_by_entity_and_variable(interpolate_keys)

        for entity, cols in grouped.items():
            # Fréquence cible pour cette entité
            target_freq = self._get_entity_target_frequency(entity)
            # Masque des lignes de cette entité
            entity_mask = self._get_entity_mask(X, entity)

            # Parcours des colonnes
            for col in cols:
                if col not in X.columns:
                    continue

                # Extraction de la série pour cette entité
                entity_series = X.loc[entity_mask, col]
                # Suppression des niveaux panel pour obtenir un DatetimeIndex
                entity_series = entity_series.droplevel(
                    list(range(X.index.nlevels - 1))
                )

                # Interpolation linéaire vers la fréquence cible
                # /!\ Faire que la fill method dépende de la position dans la string frequency
                interpolated = self._freq_converter.interpolate_to_higher_frequency(
                    entity_series, target_freq, method='linear', fill_method='ffill'
                )

                # Réindexation sur l'index temporel original de l'entité
                reindexed = interpolated.reindex(entity_series.index)

                # Réassignation dans le DataFrame résultat
                result.loc[entity_mask, col] = reindexed.values

        return result

    # Marquage de la provenance des valeurs agrégées
    def _mark_aggregated_provenance(
        self,
        tracker: ImputationProvenanceTracker,
        X: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]]
    ) -> None:
        """Mark aggregated values in the provenance tracker.

        For panel data, marks only the rows of the concerned entity.
        For time series, marks the entire index.

        Args:
            tracker: Provenance tracker instance.
            X: DataFrame with current data.
            aggregate_keys: Variable keys that were aggregated.
        """
        if not aggregate_keys:
            return

        # Cas panel : marquage par entité
        if self.is_panel_:
            grouped = self._group_keys_by_entity_and_variable(aggregate_keys)
            for entity, cols in grouped.items():
                entity_mask = self._get_entity_mask(X, entity)
                entity_index = X.index[entity_mask]
                for col in cols:
                    if col in X.columns:
                        tracker.mark_aggregated(col, entity_index)
        else:
            # Cas séries temporelles : marquage global
            columns = self._extract_column_names(aggregate_keys)
            for col in columns:
                if col in X.columns:
                    tracker.mark_aggregated(col, X.index)

    # Méthode auxilaire d'entraînement des modèles d'imputation
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
        # Initialisation du dictionnaire résultat
        models: Dict[str, Any] = {}

        # Parcours des variables 
        for variable in self.imputation_order_:
            # Extraction de l'estimateur associé à la variable
            estimator = self._get_estimator_for_variable(variable)
            # Cas où l'estimateur n'est pas spécifié
            if estimator is None:
                # Pas de modèle disponible, utilisation de l'interpolation linaire comme fallback
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
            # Interpolation si aucune feature n'est disponible pour prédire la variable
            if not feature_cols:
                warnings.warn(
                    f"No features available for imputing '{variable}', "
                    f"using linear interpolation as fallback"
                )
                models[variable] = 'interpolate_fallback'
                continue

            # Masque des observations non-NaN pour la variable cible
            mask = X[variable].notna()
            # Séparation du X et du y
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
            # /!\ Faire un test en supprimant ou remplacer par un Imputer sklearn
            X_train = X_train.fillna(X_train.mean())

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

    # Méthode auxiliaire d'entraînement d'un modèle par entité
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
        # Initialisation du dictionnaire des modèles associés à chaque entité
        entity_models: Dict[str, Any] = {}

        # Groupement par entité
        for entity_id, group in X.groupby(self.panel_cols):
            # Masque des observations pour lesquelles la target n'est pas disponible
            mask = group[variable].notna()
            # Séparation en X et y
            X_train = group.loc[mask, feature_cols]
            y_train = group.loc[mask, variable]

            if len(X_train) < 2:
                warnings.warn(
                    f"Not enough training data for entity '{entity_id}' "
                    f"and variable '{variable}', using global model"
                )
                continue

            # Imputation simple des NaN
            # /!\ Faire un test en supprimant
            X_train = X_train.fillna(X_train.mean())

            # Entraînement d'un modèle spécifiquement sur chaque entité
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

    # Méthode d'inférence des délais depuis les données en argument
    def _infer_delays_from_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Infer publication delays from data using compare_and_detect_delays.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with delay information from compare_and_detect_delays
            (columns: column, delay, unit, reference_point, observation_date, etc.).
        """
        # Utilisation de compare_and_detect_delays avec existing_data=None
        # pour identifier les observations les plus récentes
        try:
            delays_df = compare_and_detect_delays(
                new_data=X,
                existing_data=None,
                download_date=None,  # Utilise la date actuelle
                detection_mode='new_only',
                reference_point='end',
                delay_unit='D',  # Par défaut en jours
                time_col=self.time_col,
                panel_cols=self.panel_cols
            )
            return delays_df
        except Exception as e:
            warnings.warn(
                f"Failed to infer delays from data: {e}. "
                f"No delays will be inferred.",
                UserWarning
            )
            return pd.DataFrame(columns=['column', 'delay', 'unit', 'reference_point'])

    # Méthode d'entraînement
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Learn transformation parameters from X and y.

        Implements the cascade imputation fitting algorithm:
        1. Detect frequencies for all variables
        2. Classify variables (aggregate, impute, target_freq)
        3. Calculate P1 window (where all series have data)
        4. Initialize provenance tracking
        5. Fit models for each variable in imputation order

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_targets).
        """
        # Stockage des colonnes
        self.feature_columns_ = list(X.columns)
        self.target_column_ = y.name if y is not None else None

        # Détection si données panel
        self.is_panel_ = bool(self.panel_cols) or isinstance(X.index, pd.MultiIndex)

        # Construction du jeu de données de travail
        if y is not None:
            # Vérification que X et y sont de même longueur
            if len(X) != len(y):
                raise ValueError("X and y should be of equal length")
            # Construction du jeu de données de travail en concaténant X et y
            X_work = pd.concat([X, y.to_frame()], axis=1)
        else:
            X_work = X.copy()

        # Identification des entités du jeu de données
        if self.is_panel_ and isinstance(X.index, pd.MultiIndex):
            self.entities_ = get_unique_panel_entities(X)
        else:
            # Pour les séries temporelles simples, pas d'entités
            self.entities_ = None

        # Si le jeu de données est un panel et 'target_frequency' est un string, créer un dict
        if self.is_panel_ and isinstance(self.target_frequency, str) and self.entities_:
            self.effective_target_frequency = {
                entity: self.target_frequency for entity in self.entities_
            }
        elif isinstance(self.target_frequency, dict):
            self.effective_target_frequency = self.target_frequency.copy()
        else:
            self.effective_target_frequency = self.target_frequency

        # Détection des fréquences des différentes colonnes
        self.detected_frequencies_ = detect_frequency(data=X_work)
        # Vérification que les fréquences des colonnes ont bien pu être détectées
        if not self.detected_frequencies_:
            raise ValueError("Could not detect frequency for any column")

        # Validation de la fréquence cible
        self.effective_target_frequency = self._validate_target_frequency()

        # Classification des variables
        self.variable_categories_ = self._classify_variables()

        # Détermination de l'ordre d'imputation
        self.imputation_order_ = self._determine_imputation_order()

        # Initialisation du tracker de provenance
        self._provenance_tracker = ImputationProvenanceTracker()
        self._provenance_tracker.initialize(X_work, panel_cols=self.panel_cols)

        # Calcul de la fenêtre d'imputation et de la fenêtre d'entraînement
        # Initialisation du calculateur
        self._p1_calculator = ImputationWindowCalculator(
            attrition_threshold=self.attrition_threshold,
            imputation_scope=self.imputation_scope,
            min_columns=2,
            exclude_delay_nans=True
        )
        try:
            # Entrainement du calculateur
            self._p1_calculator.fit(X_work, delays=self.delays, panel_cols=self.panel_cols)
            # Calcul de la fenêtre d'imputation
            self.imputation_window_ = (self._p1_calculator.p1_start_, self._p1_calculator.p1_end_)
            # Calcul de la fenêtre d'entraînement
            self.training_window_ = (
                self._p1_calculator.training_start_,
                self._p1_calculator.training_end_
            )
        except ValueError as e:
            # Si pas de fenêtre d'imputation valide, utiliser toutes les données
            warnings.warn(
                f"Could not calculate P1 window: {e}. Using all available data.",
                UserWarning
            )
            # Fenêtre d'imputation par défaut (ensemble de la période)
            self.imputation_window_ = (X_work.index.min(), X_work.index.max())
            # Fenêtre d'entraînement par défaut (ensemble de la période)
            self.training_window_ = self.imputation_window_

        # Fit du transformer additif si fourni
        if self.additive_transformer is not None:
            # Copie indépendante du transformer
            self.additive_transformer_ = clone(self.additive_transformer)
            # Entraînement et transformation des données pour les rendre additives
            X_work = self.additive_transformer_.fit_transform(X)
            # Extraction de la composante X si c'est un XYTransformer
            if isinstance(X_work, tuple):
                X_work = X_work[0]  # Si c'est un XY transformer
        else:
            # Copie indépendante des données si elles ne sont pas transformées
            self.additive_transformer_ = None
            X_work = X.copy()

        # Agrégation des variables haute fréquence
        aggregate_keys = [
            key for key, cat in self.variable_categories_.items() if cat == 'aggregate'
        ]
        X_work = self._aggregate_to_target(X_work, aggregate_keys)

        # Marquage des valeurs agrégées dans le tracker de provenance
        self._mark_aggregated_provenance(self._provenance_tracker, X_work, aggregate_keys)

        # Entraînement des modèles d'imputation avec la logique de cascade
        self.imputation_models_ = self._fit_cascade_imputation_models(X_work)

        # Initialisation de la progression des fréquences
        self.frequency_progression_ = self._compute_frequency_progression()

        # Inférence des délais si nécessaire
        if self.impute_delayed_values and self.delays is None:
            self.inferred_delays_ = self._infer_delays_from_data(X)
        else:
            self.inferred_delays_ = pd.DataFrame()

        # Stockage de la matrice de provenance (sera mise à jour pendant transform)
        self.imputation_provenance_ = self._provenance_tracker.get_provenance_matrix()

    # Méthode auxiliaire d'extraction des noms de colonnes
    # /!\ Peut sans doute être supprimé
    def _extract_column_names(
        self,
        keys: List[Union[str, Tuple]]
    ) -> List[str]:
        """Extract unique column names from variable keys.

        Args:
            keys: List of variable identifiers (column names or (entity, column) tuples)

        Returns:
            List of unique column names
        """
        column_names = set()
        for key in keys:
            if isinstance(key, tuple):
                # Dernier élément est le nom de colonne
                column_names.add(key[-1])
            else:
                column_names.add(key)
        return list(column_names)

    # Méthode auxiliaire de
    # /!\ A revoir, dans le cadre de keep_lower_frequencies, on devrait transisitonner progressivement toutes les variables vers la cible ?
    def _compute_frequency_progression(self) -> Dict[str, List[str]]:
        """Compute the frequency progression for each variable.

        Returns:
            Dict mapping variable names to list of frequency stages.
        """
        progression = {}

        for key in self.imputation_order_:
            # Extraction du nom de variable
            if isinstance(key, tuple):
                var_name = key[-1]
            else:
                var_name = key

            # Extraction de la fréquence source
            source_freq = self.detected_frequencies_.get(key, 'D')

            # Extraction de la fréquence cible
            if self.is_panel_ and isinstance(key, tuple):
                entity = key[:-1] if len(key) > 2 else key[0]
                if isinstance(self.effective_target_frequency, dict):
                    target_freq = self.effective_target_frequency.get(entity, 'D')
                else:
                    target_freq = self.effective_target_frequency
            else:
                target_freq = self.effective_target_frequency

            # Liste des fréquences intermédiaires (à implémenter si keep_lower_frequencies=True)
            if var_name not in progression:
                progression[var_name] = [source_freq, target_freq]

        return progression

    def _fit_cascade_imputation_models(
        self,
        X: pd.DataFrame
    ) -> Dict[Union[str, Tuple], Any]:
        """Fit imputation models using cascade algorithm.

        The cascade algorithm:
        1. Group variables by frequency level
        2. For each frequency level (from lowest to highest):
           a. Aggregate features to match variable frequency
           b. Train models on P1 window (optionally extended)
           c. If cascade_refitting=True and not first level, use imputed values too

        Args:
            X: Training data (already transformed).

        Returns:
            Dictionary mapping variable identifiers to fitted models.
        """
        models: Dict[Union[str, Tuple], Any] = {}

        if not self.imputation_order_:
            return models

        # Groupement des variables par niveau de fréquence
        freq_groups = self._group_variables_by_frequency()

        # Suivi des données de travail (mises à jour si cascade_refitting=True)
        X_work = X.copy()

        # Traitement par palier de fréquence
        for freq_level, variables_at_level in freq_groups.items():
            # Entraînement des modèles pour toutes les variables à ce niveau
            for var_key in variables_at_level:
                var_name = var_key[-1] if isinstance(var_key, tuple) else var_key

                # Obtention de l'estimateur pour cette variable
                estimator = self._get_estimator_for_variable(var_name)
                if estimator is None:
                    warnings.warn(
                        f"No estimator available for variable '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    models[var_key] = 'interpolate_fallback'
                    continue

                # Préparation des features
                feature_cols = [
                    c for c in X_work.columns
                    if c != var_name and c not in (self.panel_cols or [])
                ]

                if not feature_cols:
                    warnings.warn(
                        f"No features available for imputing '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    models[var_key] = 'interpolate_fallback'
                    continue

                # Création du masque d'entraînement
                if self._p1_calculator._is_fitted:
                    training_mask = self._p1_calculator.get_training_mask(X_work, column=var_name)
                else:
                    training_mask = X_work[var_name].notna()

                # Si train_on_partial_coverage=False, filtrer uniquement les vraies valeurs
                if not self.train_on_partial_coverage:
                    # Utiliser uniquement les valeurs originales (non-imputées)
                    original_mask = self._provenance_tracker.get_mask(
                        ProvenanceType.ORIGINAL, column=var_name
                    )
                    training_mask = training_mask & original_mask

                # Extraction des données d'entraînement
                X_train = X_work.loc[training_mask, feature_cols]
                y_train = X_work.loc[training_mask, var_name]

                if len(X_train) < 2:
                    warnings.warn(
                        f"Not enough training data for variable '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    models[var_key] = 'interpolate_fallback'
                    continue

                # Imputation simple des NaN dans les features
                X_train = X_train.fillna(X_train.mean())

                # Entraînement du modèle
                try:
                    estimator.fit(X_train, y_train)
                    models[var_key] = {
                        'model': estimator,
                        'feature_cols': feature_cols,
                        'freq_level': freq_level,
                        'trained_on_imputed': self.train_on_partial_coverage and freq_level > 0,
                    }
                except Exception as e:
                    warnings.warn(
                        f"Failed to fit model for variable '{var_name}': {e}. "
                        f"Using linear interpolation as fallback"
                    )
                    models[var_key] = 'interpolate_fallback'

            # Si cascade_refitting=True, appliquer l'imputation intermédiaire et mettre à jour X_work
            if self.cascade_refitting and freq_level < max(freq_groups.keys()):
                X_work = self._apply_intermediate_imputation(X_work, models, variables_at_level)

        return models

    def _group_variables_by_frequency(self) -> Dict[int, List[Union[str, Tuple]]]:
        """Group variables to impute by frequency level.

        Returns:
            Dict mapping frequency level (int) to list of variable keys.
            Lower frequency = higher level number.
        """
        freq_groups: Dict[int, List[Union[str, Tuple]]] = {}

        for var_key in self.imputation_order_:
            freq = self.detected_frequencies_.get(var_key, 'D')
            freq_order = int(get_frequency_order(freq))

            if freq_order not in freq_groups:
                freq_groups[freq_order] = []
            freq_groups[freq_order].append(var_key)

        # Tri par niveau de fréquence décroissant (plus basse fréquence d'abord)
        return dict(sorted(freq_groups.items(), reverse=True))

    def _apply_intermediate_imputation(
        self,
        X: pd.DataFrame,
        models: Dict[Union[str, Tuple], Any],
        variables: List[Union[str, Tuple]]
    ) -> pd.DataFrame:
        """Apply intermediate imputation for refitting models.

        Args:
            X: Current working DataFrame.
            models: Fitted models so far.
            variables: Variables to impute at this level.

        Returns:
            Updated DataFrame with intermediate imputations.
        """
        result = X.copy()

        for var_key in variables:
            var_name = var_key[-1] if isinstance(var_key, tuple) else var_key
            model_info = models.get(var_key)

            if model_info is None or model_info == 'interpolate_fallback':
                # Fallback vers interpolation
                if var_name in result.columns:
                    result[var_name] = result[var_name].interpolate(
                        method='linear', limit_direction='both'
                    )
                continue

            # Identification des valeurs manquantes
            if var_name not in result.columns:
                continue

            missing_mask = result[var_name].isna()
            if not missing_mask.any():
                continue

            feature_cols = model_info.get('feature_cols', [])
            X_features = result.loc[missing_mask, feature_cols]
            X_features = X_features.fillna(X_features.mean())

            try:
                model = model_info['model']
                predictions = model.predict(X_features)
                result.loc[missing_mask, var_name] = predictions

                # Marquage de la provenance
                trained_on_imputed = model_info.get('trained_on_imputed', False)
                self._provenance_tracker.mark_model_imputed(
                    var_name, result.index[missing_mask], trained_on_imputed=trained_on_imputed
                )
            except Exception as e:
                warnings.warn(f"Intermediate imputation failed for '{var_name}': {e}")
                result[var_name] = result[var_name].interpolate(
                    method='linear', limit_direction='both'
                )

        return result

    # -------------------------------------------------------------------------
    # Interface de transformation principale
    # -------------------------------------------------------------------------
    def _transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Transform X and optionally y using cascade imputation.

        This method implements the abstract _transform from XYPanelTimeSeriesTransformer.
        It concatenates X and y, applies transformations, then splits them.

        Args:
            X: Features to transform.
            y: Targets to transform (optional).

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        # Validation des données
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        # Stockage des originaux pour inverse_transform
        self._original_X_ = X.copy()
        self._original_y_ = y.copy() if y is not None else None

        # Identification du nom de la colonne y pour la scission ultérieure
        y_col_name = None
        if y is not None:
            y_col_name = y.name if y.name is not None else '__target__'

        # Concaténation de X et y si y est fourni
        if y is not None:
            y_frame = y.to_frame(name=y_col_name)
            data_work = pd.concat([X, y_frame], axis=1)
        else:
            data_work = X.copy()

        # Préparation de l'index
        if not isinstance(data_work.index, (pd.DatetimeIndex, pd.MultiIndex)):
            if self.time_col and self.time_col in data_work.columns:
                data_work = data_work.set_index(self.time_col)
            else:
                raise ValueError("Data must have a DatetimeIndex or MultiIndex")

        # Initialisation du tracker de provenance pour la transformation
        transform_tracker = ImputationProvenanceTracker()
        transform_tracker.initialize(data_work, panel_cols=self.panel_cols)

        # Application de la transformation additive
        if self.additive_transformer_ is not None:
            data_transformed = self.additive_transformer_.transform(data_work)
            if isinstance(data_transformed, tuple):
                data_transformed = data_transformed[0]
        else:
            data_transformed = data_work.copy()

        # Agrégation des variables haute fréquence
        aggregate_keys = [
            key for key, cat in self.variable_categories_.items() if cat == 'aggregate'
        ]
        data_transformed = self._aggregate_to_target(data_transformed, aggregate_keys)

        # Marquage des valeurs agrégées
        self._mark_aggregated_provenance(transform_tracker, data_transformed, aggregate_keys)

        # Imputation cascadée
        data_transformed, intermediate_results = self._apply_cascading_imputation(
            data_transformed, transform_tracker
        )

        # Gestion des délais
        if self.impute_delayed_values:
            data_transformed = self._impute_delayed_values(data_transformed)

        # Construction de la sortie selon keep_lower_frequencies
        if self.keep_lower_frequencies and intermediate_results:
            data_result = self._build_multifreq_output(data_transformed, intermediate_results)
        else:
            data_result = data_transformed

        # Mise à jour de la matrice de provenance
        self.imputation_provenance_ = transform_tracker.get_provenance_matrix()

        # Scission de X et y si y était fourni
        if y is not None and y_col_name in data_result.columns:
            y_transformed = data_result[y_col_name]
            X_transformed = data_result.drop(columns=[y_col_name])
            return X_transformed, y_transformed
        else:
            return data_result

    def _apply_cascading_imputation(
        self,
        X: pd.DataFrame,
        provenance_tracker: ImputationProvenanceTracker
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Apply cascading imputation for variables marked for imputation.

        Args:
            X: DataFrame with aggregated variables.
            provenance_tracker: Tracker to record provenance of imputed values.

        Returns:
            Tuple of:
            - Final DataFrame with imputed variables
            - Dict mapping frequency levels to intermediate DataFrames
              (only if keep_lower_frequencies=True)
        """
        result = X.copy()
        intermediate_results: Dict[str, pd.DataFrame] = {}

        # Groupement des variables par niveau de fréquence
        freq_groups = self._group_variables_by_frequency()

        for freq_level, variables_at_level in freq_groups.items():
            # Stockage des résultats intermédiaires si nécessaire
            if self.keep_lower_frequencies:
                intermediate_results[f"freq_{freq_level}"] = result.copy()

            # Imputation des variables à ce niveau
            for var_key in variables_at_level:
                var_name = var_key[-1] if isinstance(var_key, tuple) else var_key
                model_info = self.imputation_models_.get(var_key)

                if model_info is None or model_info == 'interpolate_fallback':
                    # Fallback vers interpolation linéaire
                    if var_name in result.columns:
                        result[var_name] = result[var_name].interpolate(
                            method='linear', limit_direction='both'
                        )
                    continue

                # Identification des valeurs manquantes
                if var_name not in result.columns:
                    continue

                missing_mask = result[var_name].isna()
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
                            f"Prediction failed for variable '{var_name}': {e}. "
                            f"Using interpolation fallback."
                        )
                        result[var_name] = result[var_name].interpolate(
                            method='linear', limit_direction='both'
                        )
                        continue

                result.loc[missing_mask, var_name] = predictions

                # Marquage de la provenance
                trained_on_imputed = model_info.get('trained_on_imputed', False)
                provenance_tracker.mark_model_imputed(
                    var_name,
                    result.index[missing_mask],
                    trained_on_imputed=trained_on_imputed
                )

        return result, intermediate_results

    def _build_multifreq_output(
        self,
        final_result: pd.DataFrame,
        intermediate_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Build MultiIndex output with all frequency levels.

        Args:
            final_result: Final DataFrame at target frequency.
            intermediate_results: Dict mapping frequency levels to DataFrames.

        Returns:
            DataFrame with MultiIndex:
            - Time series: (Frequency, Date)
            - Panel: (Entity, Frequency, Date)
        """
        # Collecte de tous les DataFrames avec leur étiquette de fréquence
        all_frames = []

        for freq_label, df in intermediate_results.items():
            # Ajout du niveau de fréquence à l'index
            df_copy = df.copy()
            df_copy['_frequency_level'] = freq_label
            all_frames.append(df_copy)

        # Ajout du résultat final
        final_copy = final_result.copy()
        final_copy['_frequency_level'] = 'target'
        all_frames.append(final_copy)

        # Concaténation
        combined = pd.concat(all_frames, ignore_index=False)

        # Reconstruction du MultiIndex
        if self.is_panel_:
            # Panel: (Entity, Frequency, Date)
            if isinstance(combined.index, pd.MultiIndex):
                # Extraction des niveaux existants
                entity_values = combined.index.get_level_values(0)
                date_values = combined.index.get_level_values(-1)
                freq_values = combined['_frequency_level']

                new_index = pd.MultiIndex.from_arrays(
                    [entity_values, freq_values, date_values],
                    names=['entity', 'frequency', 'date']
                )
            else:
                # Index simple avec colonne de fréquence
                new_index = pd.MultiIndex.from_arrays(
                    [combined['_frequency_level'], combined.index],
                    names=['frequency', 'date']
                )
        else:
            # Time series: (Frequency, Date)
            new_index = pd.MultiIndex.from_arrays(
                [combined['_frequency_level'], combined.index],
                names=['frequency', 'date']
            )

        # Application du nouvel index
        combined.index = new_index
        combined = combined.drop(columns=['_frequency_level'])

        return combined

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
        delays_to_use = self.delays if self.delays is not None else self.inferred_delays_

        if delays_to_use.empty:
            return result

        for _, row in delays_to_use.iterrows():
            column = row['column']
            delay = row['delay']

            if column not in result.columns:
                continue

            # Identifier les positions affectées par le délai
            # (dernières 'delay' observations)
            n_delay = int(delay)
            if n_delay <= 0:
                continue

            delayed_idx = result.index[-n_delay:]
            missing_mask = result.loc[delayed_idx, column].isna()

            if not missing_mask.any():
                continue

            # Imputation des valeurs retardées
            model_info = self.imputation_models_.get(column)

            if model_info is None or model_info == 'interpolate_fallback':
                # Interpolation linéaire
                result[column] = result[column].interpolate(
                    method='linear', limit_direction='both'
                )
            else:
                feature_cols = model_info.get('feature_cols', [])
                # Utilisation de .values pour obtenir un tableau booléen numpy
                missing_idx = delayed_idx[missing_mask.values]
                X_features = result.loc[missing_idx, feature_cols]

                if not X_features.empty:
                    X_features = X_features.fillna(X_features.mean())
                    try:
                        if 'entity_models' in model_info:
                            predictions = self._predict_per_entity(
                                model_info, X_features, result.loc[missing_idx]
                            )
                        else:
                            predictions = model_info['model'].predict(X_features)
                        result.loc[missing_idx, column] = predictions
                    except Exception as e:
                        warnings.warn(
                            f"Failed to impute delayed values for '{column}': {e}"
                        )

        return result

    # -------------------------------------------------------------------------
    # Transformation inverse
    # -------------------------------------------------------------------------
    def _inverse_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Inverse transform X and optionally y.

        Note: Disaggregation from lower to higher frequency is lossy.
        The imputed sub-period values cannot be exactly reversed to original
        low-frequency values.

        Args:
            X: Transformed features.
            y: Transformed targets (optional).

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.
        """
        # Identification du nom de la colonne y pour la scission ultérieure
        y_col_name = None
        if y is not None:
            y_col_name = y.name if y.name is not None else '__target__'

        # Concaténation de X et y si y est fourni
        if y is not None:
            y_frame = y.to_frame(name=y_col_name)
            data_work = pd.concat([X, y_frame], axis=1)
        else:
            data_work = X.copy()

        # Application de l'inverse du transformer additif
        if self.additive_transformer_ is not None:
            if hasattr(self.additive_transformer_, 'inverse_transform'):
                try:
                    data_result = self.additive_transformer_.inverse_transform(data_work)
                    if isinstance(data_result, tuple):
                        data_result = data_result[0]
                except Exception as e:
                    warnings.warn(
                        f"Failed to inverse transform with additive transformer: {e}"
                    )
                    data_result = data_work
            else:
                data_result = data_work
        else:
            data_result = data_work

        # Scission de X et y si y était fourni
        if y is not None and y_col_name in data_result.columns:
            y_original = data_result[y_col_name]
            X_original = data_result.drop(columns=[y_col_name])
            return X_original, y_original
        else:
            return data_result
