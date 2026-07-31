"""High frequency imputer for mixed frequency time series data.

This module provides the HighFrequencyImputer class to impute high-frequency values
for low-frequency series in mixed-frequency datasets using machine learning models.
"""
# Importation des modules
# Modules de base
import warnings
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Union, Any, Tuple
# Manipulation de données
import numpy as np
import pandas as pd
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
# Utilitaires du package
from ..xy.transformers import XYPanelTimeSeriesTransformer
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import (
    normalize_frequency,
    is_higher_frequency,
    get_frequency_order,
)
from ..panel.utils import is_panel_data, get_unique_panel_entities, normalize_entity_key
from .detector import FrequencyDetector, detect_frequency
from .provenance import ImputationProvenanceTracker, ProvenanceType
from .imputation_window import ImputationWindowCalculator, ImputationScope
from .target_frequency_validator import TargetFrequencyValidator
from .frequency_aligner import FrequencyAligner
from .regularizer import IndexRegularizer


# Type aliases
VariableCategory = Literal['aggregate', 'impute', 'target_freq']


# Classe d'imputation des valeurs de variables
class HighFrequencyImputer(XYPanelTimeSeriesTransformer):
    """Impute high-frequency values for low-frequency series in mixed-frequency datasets.

    This XY transformer handles mixed-frequency datasets using a cascading imputation
    approach that respects frequency hierarchies and tracks value provenance:

    1. Making data additive via a user-provided transformer
    2. Computing the imputation window (where all series have true values)
    3. Aggregating high-frequency variables to lower frequencies
    4. Cascading imputation from lowest to highest frequency
    5. Optionally refitting models with imputed values (cascade_refitting)
    6. Handling publication delays if provided
    7. Tracking provenance of each imputed value

    The cascade algorithm processes variables by frequency level, from lowest (e.g., quarterly)
    to highest (e.g., daily). At each level:
    - Features are aggregated to match the variable's frequency
    - Models are trained on the imputation window (optionally extended)
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
        coverage_threshold: Minimum ratio of columns with data (0-1) for extended window.
        imputation_scope: Training window scope ('strict', 'extended_backward',
            'extended_forward', 'extended_both').
        train_on_partial_coverage: If True, use imputed values for training outside P1.
        train_on_partial_fit_order: Order for imputing variables when
            train_on_partial_coverage is True:
            - 'random': Sort by frequency level then entity count (default)
            - 'cv': Use cross-validation to impute easiest variables first
        scale_features: If True, divide X_train by the frequency conversion
            factor when scaling y_train during training.

    Attributes:
        detected_frequencies_: Detected frequency per variable or (entity, variable).
        variable_categories_: Category per (entity, variable) tuple:
            'aggregate', 'impute', or 'target_freq'.
        imputation_order_: Ordered list of variables for cascading imputation.
        imputation_models_: Fitted imputation models per variable.
        model_fitting_order_: List of (pred_freq, var_key) tuples recording the
            exact order in which models were fitted, for replay during transform.
        imputation_provenance_: DataFrame tracking origin of each value
            ('original', 'model_on_true', 'model_on_mixed', 'aggregated').
        imputation_window_: Tuple (start, end) of the imputation window where
            all series have data.
        training_window_: Tuple (start, end) of the extended training window.
        frequency_progression_: Dict mapping variables to their frequency stages.
        inferred_delays_: DataFrame with delays inferred from data using
            compare_and_detect_delays (if impute_delayed_values=True and delays=None).
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
        ...     coverage_threshold=0.5
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
        coverage_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        train_on_partial_coverage: bool = False,
        train_on_partial_fit_order: Literal['random', 'cv'] = 'random',
        scale_features: bool = True,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
    ):
        """Initialize the HighFrequencyImputer.

        Args:
            target_frequency: Target frequency for imputation. Can be:
                - str: Single frequency (e.g., 'M', 'Q', 'monthly')
                - Dict[entity_id, str]: Entity-specific frequencies for panel data
                Must not be higher than the lowest frequency in the data.
            estimator: Estimator(s) for prediction. Can be:
                - Single estimator: Applied to all variables
                - Dict[variable_name, estimator]: Variable-specific models,
                  a model associated to '__default__' key can be provided
            additive_transformer: Transformer to make data additive before
                imputation (e.g., log transformer, differencing). Must support
                fit_transform() and inverse_transform(). If None, data is
                assumed to already be additive.
            cascade_refitting: If True, refit models using imputed values after
                each frequency stage. Enables more accurate imputation in
                later stages.
            keep_lower_frequencies: If True, output includes all intermediate
                frequencies in a MultiIndex structure. If False, only target
                frequency is returned.
            impute_delayed_values: Whether to impute values affected by
                publication delays. Default is False.
            delays: Publication delays DataFrame with columns:
                column, delay, unit, reference_point.
            on_frequency_mismatch: How to handle target_frequency higher than
                data frequencies ('error'/'warn').
            coverage_threshold: Minimum percentage of columns (0-1) that
                must have non-null values for extended training window.
            imputation_scope: Training window scope.
            train_on_partial_coverage: If True, use imputed values for
                training models outside the imputation window.
            train_on_partial_fit_order: Order for variable imputation:
                - 'random': By frequency level then entity count (default)
                - 'cv': Cross-validation to find easiest variables first
            scale_features: If True, divide X_train by the frequency
                conversion factor alongside y_train.
            time_col: Name of the time column (if in columns not index).
            panel_cols: List of column names identifying panel entities.
        """
        # Initialisation du parent
        super().__init__(
            time_col=time_col, panel_cols=panel_cols,
            validate_input=True, strict_validation=True,
            auto_sort=False, convert_cols_to_index=True
        )

        # Validation des paramètres
        target_frequency = self._validate_target_frequency_format(target_frequency)
        self._validate_estimator(estimator)
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

        if on_frequency_mismatch not in ['error', 'warn']:
            raise ValueError(
                f"on_frequency_mismatch must be 'error' or 'warn', "
                f"got '{on_frequency_mismatch}'"
            )
        if not 0 <= coverage_threshold <= 1:
            raise ValueError(
                f"coverage_threshold must be between 0 and 1, got {coverage_threshold}"
            )
        valid_scopes = ('strict', 'extended_backward', 'extended_forward', 'extended_both')
        if imputation_scope not in valid_scopes:
            raise ValueError(
                f"imputation_scope must be one of {valid_scopes}, got '{imputation_scope}'"
            )
        if train_on_partial_fit_order not in ('random', 'cv'):
            raise ValueError(
                f"train_on_partial_fit_order must be 'random' or 'cv', "
                f"got '{train_on_partial_fit_order}'"
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
        self.coverage_threshold = coverage_threshold
        self.imputation_scope = imputation_scope
        self.train_on_partial_coverage = train_on_partial_coverage
        self.train_on_partial_fit_order = train_on_partial_fit_order
        self.scale_features = scale_features

    # -------------------------------------------------------------------------
    # Validation des paramètres d'entrée
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de validation du format de la fréquence cible
    def _validate_target_frequency_format(
        self,
        target_frequency: Union[str, Dict[Union[str, tuple], str]]
    ) -> Union[str, Dict[Union[str, tuple], str]]:
        """Validate the format and values of target_frequency parameter.

        Args:
            target_frequency: Target frequency (string or dict mapping
                entities to frequencies).

        Returns:
            Normalized target_frequency.

        Raises:
            ValueError: If target_frequency format is invalid or contains
                invalid frequencies.
        """
        # Distinction suivant le type de la fréquence cible
        # Cas où la fréquence cible est une chaîne de caractères
        if isinstance(target_frequency, str):
            # Normalisation
            try:
                return normalize_frequency(target_frequency, return_format="base")
            except ValueError as e:
                raise ValueError(f"Invalid target_frequency '{target_frequency}': {e}")
        
        # Cas où la fréquence cible est un dictionnaire
        elif isinstance(target_frequency, dict):
            # Vérification que le dictionnaire est non vide
            if not target_frequency:
                raise ValueError("target_frequency dict cannot be empty")
            
            # Initialisation des dictionnaires de fréquences valides et invalides
            validated_freqs = {}
            invalid_freqs = {}

            # Parcours des fréquences associées à chaque entité
            for entity, freq in target_frequency.items():
                # Vérification que la fréquence est une chaîne de caractères
                if not isinstance(freq, str):
                    raise ValueError(
                        f"Frequency for entity '{entity}' must be a string, "
                        f"got {type(freq).__name__}"
                    )
                # Normalisation
                try:
                    validated_freqs[entity] = normalize_frequency(freq)
                except ValueError as e:
                    invalid_freqs[entity] = str(e)

            # Construction du message d'erreur s'il existe des fréquences invalides
            if invalid_freqs:
                error_msg = "Invalid frequencies in target_frequency dict:\n"
                for entity, error in invalid_freqs.items():
                    error_msg += f"  - Entity '{entity}': {error}\n"
                raise ValueError(error_msg.rstrip())
            return validated_freqs

        else:
            raise TypeError(
                f"target_frequency must be a string or dict, "
                f"got {type(target_frequency).__name__}"
            )

    # Méthode auxiliaire de validation de l'estimateur
    def _validate_estimator(
        self,
        estimator: Optional[Union[BaseEstimator, Dict[str, BaseEstimator]]]
    ) -> None:
        """Validate estimator has required methods (fit and predict).

        Args:
            estimator: Estimator or dict of estimators to validate.

        Raises:
            ValueError: If estimator lacks required methods.
        """
        # Cas où l'estimateur n'est pas spécifié
        if estimator is None:
            return
        
        # Distinction suivant le type de l'estimateur
        # Cas où il s'agit d'un dictionnaire
        if isinstance(estimator, dict):
            # Vérification que le dictionnaire est non vide
            if not estimator:
                raise ValueError("estimator dict cannot be empty")
            # Parcours des éléments du dictionnaire
            for var_name, est in estimator.items():
                # Vérification qu'il possède la méthode "fit"
                if not hasattr(est, 'fit') or not callable(getattr(est, 'fit')):
                    raise ValueError(
                        f"Estimator for '{var_name}' must have a 'fit' method, "
                        f"got {type(est).__name__}"
                    )
                # Vérification qu'il possède la méthode "predict"
                if not hasattr(est, 'predict') or not callable(getattr(est, 'predict')):
                    raise ValueError(
                        f"Estimator for '{var_name}' must have a 'predict' method, "
                        f"got {type(est).__name__}"
                    )
        # Cas où il s'agit d'un estimateur
        else:
            # Vérification qu'il possède la méthode "fit"
            if not hasattr(estimator, 'fit') or not callable(getattr(estimator, 'fit')):
                raise ValueError(
                    f"estimator must have a 'fit' method, "
                    f"got {type(estimator).__name__}"
                )
            # Vérification qu'il possède la méthode "predict"
            if not hasattr(estimator, 'predict') or not callable(getattr(estimator, 'predict')):
                raise ValueError(
                    f"estimator must have a 'predict' method, "
                    f"got {type(estimator).__name__}"
                )

    # Méthode auxiliaire de validation du transformer
    def _validate_additive_transformer(
        self,
        transformer: TransformerMixin
    ) -> None:
        """Validate additive_transformer has required methods.

        Args:
            transformer: Transformer to validate.

        Raises:
            ValueError: If transformer lacks required methods.
        """
        # Liste des méthodes requises
        required_methods = ['fit', 'transform', 'inverse_transform']
        # Initialisation de la liste des méthodes manquantes
        missing_methods = []

        # Parcours des méthodes requises
        for method_name in required_methods:
            # Vérification que le transformer possède la méthode en attribut
            if not hasattr(transformer, method_name) or not callable(getattr(transformer, method_name)):
                missing_methods.append(method_name)

        # Construction du message d'erreur si des méthodes sont manquantes
        if missing_methods:
            raise ValueError(
                f"additive_transformer must have methods: {', '.join(required_methods)}. "
                f"Missing: {', '.join(missing_methods)}. "
                f"Got {type(transformer).__name__}"
            )

    # -------------------------------------------------------------------------
    # Propriétés composées
    # -------------------------------------------------------------------------
    # Instance du validateur de la fréquence cible
    @property
    def _target_freq_validator(self) -> TargetFrequencyValidator:
        """Lazy initialization of target frequency validator."""
        # Initialisation si l'attribut n'existe pas déjà
        if not hasattr(self, '_target_freq_validator_cache'):
            self._target_freq_validator_cache = TargetFrequencyValidator()
        return self._target_freq_validator_cache

    # Instance de l'aligneur de fréquence
    @property
    def _freq_aligner(self) -> FrequencyAligner:
        """Lazy initialization of frequency aligner."""
        # Initialisation si l'attribut n'existe pas déjà
        if not hasattr(self, '_freq_aligner_cache'):
            self._freq_aligner_cache = FrequencyAligner()
        return self._freq_aligner_cache

    # Instance du détecteur de fréquence
    @property
    def _freq_detector(self) -> FrequencyDetector:
        """Lazy initialization of frequency detector."""
        # Initialisation si l'attribut n'existe pas déjà
        if not hasattr(self, '_freq_detector_cache'):
            self._freq_detector_cache = FrequencyDetector()
        return self._freq_detector_cache

    # Instance du convertisseur de fréquence
    @property
    def _freq_converter(self) -> FrequencyConverter:
        """Lazy initialization of frequency converter."""
        # Initialisation si l'attribut n'existe pas déjà
        if not hasattr(self, '_freq_converter_cache'):
            self._freq_converter_cache = FrequencyConverter()
        return self._freq_converter_cache
    
    # Instance du convertisseur de fréquence
    @property
    def _index_regularizer(self) -> IndexRegularizer:
        """Lazy initialization of index regularizer."""
        # Initialisation si l'attribut n'existe pas déjà
        if not hasattr(self, '_index_regularizer_cache'):
            self._index_regularizer_cache = IndexRegularizer()
        return self._index_regularizer_cache

    # -------------------------------------------------------------------------
    # Méthodes auxiliaires
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de classification des variables suivant qu'elles doivent être agrégées, imputées ou rester inchangées
    def _classify_variables(
        self
    ) -> Dict[Union[str, Tuple], VariableCategory]:
        """Classify each variable by its relationship to target frequency.

        Returns:
            Dictionary mapping variable identifiers to their category:
            - 'aggregate': Higher frequency than target, needs aggregation
            - 'impute': Lower frequency than target, needs imputation
            - 'target_freq': Already at target frequency
        """
        # Initialisation du dictionnaire contenant les catégories
        categories: Dict[Union[str, Tuple], VariableCategory] = {}

        # Distinction des données de panel et de séries temporelles
        # Cas des données de panel
        if self.is_panel_:
            # Parcours des fréquences détectées
            for key, freq in self.detected_frequencies_.items():
                # Extraction de la colonne et de l'entité
                if isinstance(key, tuple):
                    entity = key[:-1] if len(key) > 2 else key[0]
                    col = key[-1]
                else:
                    entity = None
                    col = key

                # Extraction de la fréquence cible associée à l'entité
                if isinstance(self.effective_target_frequency_, dict):
                    # Normalisation de l'entité sous forme de tuple
                    entity_key = entity if isinstance(entity, tuple) else (entity,)
                    # Extraction de la fréquence
                    target_freq = self.effective_target_frequency_.get(entity_key)
                    # Si la fréquence est nulle, tentative d'extraction avec la clé brute
                    if target_freq is None:
                        target_freq = self.effective_target_frequency_.get(entity)
                else:
                    target_freq = self.effective_target_frequency_

                # Vérification que la fréquence cible est spécifiée
                if target_freq is None:
                    continue
                
                # Normalisation de la fréquence détectée
                freq_normalized = normalize_frequency(freq)
                # Normalisation de la fréquenc cible
                target_normalized = normalize_frequency(target_freq)

                # Comparaison des fréquences
                if is_higher_frequency(freq, target_freq):
                    # Agrégation si la fréquence cible est plus faible que la fréquence source
                    categories[key] = 'aggregate'
                elif freq_normalized == target_normalized:
                    # Cas d'égalité
                    categories[key] = 'target_freq'
                else:
                    # Imputation si la fréquence cible est plus élevée que la fréquence source
                    categories[key] = 'impute'
        # Cas des séries temporelles
        else:
            # Parcours des fréquences détectées
            for col, freq in self.detected_frequencies_.items():
                # Normalisation de la fréquence source
                freq_normalized = normalize_frequency(freq)
                # Normalisation de la fréquence cible
                target_normalized = normalize_frequency(self.effective_target_frequency_)

                # Comparaison des fréquences
                if is_higher_frequency(freq, self.effective_target_frequency_):
                    # Agrégation si la fréquence cible est plus faible que la fréquence source
                    categories[col] = 'aggregate'
                elif freq_normalized == target_normalized:
                    # Cas d'égalité
                    categories[col] = 'target_freq'
                else:
                    # Imputation si la fréquence cible est plus élevée que la fréquence source
                    categories[col] = 'impute'

        return categories

    # Méthode auxiliaire de classification des variables par rapport à une fréquence cible
    # /!\ Regarder si cela fait sens que dans la méthode précédente les clés du dictionnaire soient les variables / entités X variable et ici les statuts
    # /!\ Voir la méthode précédente ne pourrait pas être un cas particulier de celle-ci
    def _classify_variables_at_frequency(
        self,
        prediction_frequency: Union[str, Dict],
    ) -> Dict[str, List[Union[str, Tuple]]]:
        """Classify variables relative to a specific prediction frequency.

        Args:
            prediction_frequency: Frequency at which predictions will be made
                (str for TS, dict for panel).

        Returns:
            Dict with keys 'aggregate', 'impute', 'target_freq', each
            containing a list of variable keys.
        """
        # Initialisation du dictionnaire résultat
        result: Dict[str, List[Union[str, Tuple]]] = {
            'aggregate': [], 'impute': [], 'target_freq': []
        }

        # Distinction suivant la structure des données
        # Données de panel
        if self.is_panel_:
            # Parcours des fréquences détectées
            for key, freq in self.detected_frequencies_.items():
                # Extraction de la colonne et de l'entité
                if isinstance(key, tuple):
                    entity = key[:-1] if len(key) > 2 else key[0]
                else:
                    entity = None

                # Extraction de la fréquence cible associée à l'entité
                if isinstance(prediction_frequency, dict):
                    # Normalisation de l'entité sous forme de tuple
                    entity_key = entity if isinstance(entity, tuple) else (entity,)
                    # Extraction de la fréquence 
                    pred_freq = prediction_frequency.get(entity_key)
                    # Si la fréquence est nulle, tentative d'extraction avec la clé brute
                    if pred_freq is None:
                        pred_freq = prediction_frequency.get(entity)
                else:
                    pred_freq = prediction_frequency

                # Vérification que la fréquence cible est spécifiée
                if pred_freq is None:
                    continue
                
                # Normalisation de la fréquence détectée
                freq_normalized = normalize_frequency(freq)
                # Normalisation de la fréquence cible
                pred_normalized = normalize_frequency(pred_freq)

                # Comparaison des fréquences
                if is_higher_frequency(freq, pred_freq):
                    # Agrégation si la fréquence cible est plus faible que la fréquence source
                    result['aggregate'].append(key)
                elif freq_normalized == pred_normalized:
                    # Cas d'égalité
                    result['target_freq'].append(key)
                else:
                    # Imputation si la fréquence cible est plus élevée que la fréquence source
                    result['impute'].append(key)
        # Cas des séries temporelles
        else:
            # Cas d'erreur si la fréquence cible est un dictionnaire dans ce cas
            if not isinstance(prediction_frequency, str):
                raise TypeError("'prediction_frequency' should be a string when applied to time series")
            # Parcours des fréquences détectées
            for col, freq in self.detected_frequencies_.items():
                # Normalisation de la fréquence source
                freq_normalized = normalize_frequency(freq)
                # Normalisation de la fréquence cible
                pred_normalized = normalize_frequency(prediction_frequency)

                # Comparaison des fréquences
                if is_higher_frequency(freq, prediction_frequency):
                    # Agrégation si la fréquence cible est plus faible que la fréquence source
                    result['aggregate'].append(col)
                elif freq_normalized == pred_normalized:
                    # Cas d'égalité
                    result['target_freq'].append(col)
                else:
                    # Imputation si la fréquence cible est plus élevée que la fréquence source
                    result['impute'].append(col)

        return result

    # Méthode auxiliaire de détermination de l'ordre d'imputation des variables suivant leur fréquence
    def _determine_imputation_order(
        self
    ) -> List[Union[str, Tuple]]:
        """Determine order of variables for cascading imputation.

        Sorting logic:
        1. Sort by frequency (lowest frequency first)
        2. For panel data: variables with fewer entities first among same freq

        Returns:
            Ordered list of variable identifiers to impute.
        """
        # Extraction des variables à imputer de la liste
        # /!\ Est ce que ce genre d'opérations sur les valeurs du dictionnaire ne donne pas envie de le formater comme le résultat de "_classify_variables_at_frequency"
        impute_vars = [
            key for key, cat in self.variable_categories_.items() if cat == 'impute'
        ]

        # Cas où il n'y a pas de variables à imputer
        if not impute_vars:
            return []

        # Distinction suivant la structure des données
        # Cas de données de séries temporelles
        if not self.is_panel_:
            # Tri avec les fréquences les plus faibles en premier
            impute_vars.sort(
                key=lambda col: get_frequency_order(
                    self.detected_frequencies_[col]
                ),
                reverse=True,
            )
            return impute_vars
        # Cas de données de panel
        else :
            # Initialisation du dictionnaire associant une variable à la liste de ses entités
            var_to_entities: Dict[str, List[Tuple]] = {}
            # Initialisation du dictionnaire associant une variable à la liste de ses fréquences
            var_to_frequencies: Dict[str, List[float]] = {}

            # Parcours des variables à imputer
            for key in impute_vars:
                # Extraction du nom de la variable de la clé
                var_name = key[-1] if isinstance(key, tuple) else key
                # Ajout de la variable si elle ne fait pas déjà partie des clés des dictionnaires
                if var_name not in var_to_entities:
                    var_to_entities[var_name] = []
                    var_to_frequencies[var_name] = []
                # Ajout de l'entité
                var_to_entities[var_name].append(key)
                # Extraction de la fréquence détectée
                freq = self.detected_frequencies_[key]
                # Ajout de l'ordre de la fréquence
                var_to_frequencies[var_name].append(get_frequency_order(freq))

            # Initialisation de la liste des métriques sur lesquelles trier les données
            var_metrics: List[Tuple[str, float, float, int]] = []
            # Parcours des variables
            for var_name in var_to_entities:
                # Extraction des ordres des fréquences associées à la variable
                freq_orders = var_to_frequencies[var_name]
                # Calcul de métriques sur l'ordre des fréquences
                # Fréquence médiane
                representative_freq = np.median(freq_orders)
                # Fréquence la plus faible
                min_freq = np.max(freq_orders)
                # Nombre d'entités associées à la variable
                n_entities = len(var_to_entities[var_name])
                var_metrics.append((var_name, representative_freq, min_freq, n_entities))

            # Tri des variables suivant les métriques (d'abord les fréquences les plus faibles, puis celles ayant le moins d'entités)
            var_metrics.sort(key=lambda x: (-x[1], -x[2], x[3]))

            # Initialisation de la liste de l'ordre des variables à imputer
            ordered_impute_vars = []
            # Parcours des variables ordonnées
            for var_name, _, _, _ in var_metrics:
                # Extraction des entités liées à la variable
                var_keys = var_to_entities[var_name]
                # Tri par fréquence décroissante au sein des entité
                var_keys.sort(
                    key=lambda k: get_frequency_order(self.detected_frequencies_[k]),
                    reverse=True
                )
                # Ajout aux variables à imputer
                ordered_impute_vars.extend(var_keys)

            return ordered_impute_vars

    # Méthode auxiliaire de détermination de l'ordre d'imputation par validation croisée
    def _determine_variable_order_cv(
        self,
        X: pd.DataFrame,
        impute_vars: List[Union[str, Tuple]],
    ) -> List[Union[str, Tuple]]:
        """Determine variable order using cross-validation MAPE.

        Variables with the lowest MAPE (easiest to predict) are placed first.
        Falls back to the 'random' ordering if fewer than 10 observations
        are available for training.

        Args:
            X: Working data.
            impute_vars: List of variable keys to order.

        Returns:
            Ordered list, easiest variables first.
        """
        # Vérification qu'il y a au moins deux variables à ordonner
        if len(impute_vars) <= 1:
            return impute_vars

        # Initialisation de la liste associant un score à chaque variable / entité X variable
        scored_vars: List[Tuple[Union[str, Tuple], float]] = []

        # Parcours des variables à imputer
        for var_key in impute_vars:
            # Extraction du nom de la variable de la clé
            var_name = var_key[-1] if isinstance(var_key, tuple) else var_key

            # Préparation des données d'entraînement dans la fenêtre stricte
            if hasattr(self, '_imputation_window_calc') and self._imputation_window_calc._is_fitted:
                mask = self._imputation_window_calc.get_imputation_window_mask(X)
            else:
                mask = pd.Series(True, index=X.index)

            # Extraction des colonnes de features
            feature_cols = [c for c in X.columns if c != var_name and c not in (self.panel_cols or [])]

            # Score infini si la série est univariée
            if not feature_cols:
                scored_vars.append((var_key, float('inf')))
                continue
            
            # Séparation en X et y
            X_sub = X.loc[mask, feature_cols]
            y_sub = X.loc[mask, var_name]

            # Fallback si < 10 observations
            if len(X_sub) < 10:
                scored_vars.append((var_key, get_frequency_order(
                    self.detected_frequencies_[var_key]
                )))
                continue

            # CV 5-fold avec MAPE
            # Extraction de l'estimateur
            estimator = self._get_estimator_for_variable(var_name)
            # Score infini si l'estimateur n'est pas spécifié
            if estimator is None:
                scored_vars.append((var_key, float('inf')))
                continue
            
            # Initialisation de la KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            # Initialisation de la liste des scores
            mapes = []
            # Parcours des sous-ensembles d'entraînement et de validation
            for train_idx, val_idx in kf.split(X_sub):
                # Extraction des observations d'entraînement et de validation
                X_train, X_val = X_sub.iloc[train_idx], X_sub.iloc[val_idx]
                y_train, y_val = y_sub.iloc[train_idx], y_sub.iloc[val_idx]
                # Estimation et prédiction sur la validation
                try:
                    est = clone(estimator)
                    est.fit(X_train, y_train)
                    preds = est.predict(X_val)
                    # Évite la division par zéro
                    non_zero = y_val != 0
                    if non_zero.sum() > 0:
                        mapes.append(mean_absolute_percentage_error(
                            y_val[non_zero], preds[non_zero]
                        ))
                except Exception:
                    mapes.append(float('inf'))
            # Moyenne des erreurs
            avg_mape = np.mean(mapes) if mapes else float('inf')
            # Ajout à la liste des scores
            scored_vars.append((var_key, avg_mape))

        # Tri par MAPE croissant (variables les plus faciles en premier)
        scored_vars.sort(key=lambda x: x[1])
        return [v for v, _ in scored_vars]

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

        # Cas où l'estimateur est un dictionnaire
        if isinstance(self.estimator, dict):
            est = self.estimator.get(variable)
            # Si pas spécifié, extraction de l'estimateur par défaut
            if est is None:
                est = self.estimator.get('__default__')
            # Clonage de l'estimateur
            return clone(est) if est is not None else None
        
        # Clonage de l'estimateur
        return clone(self.estimator)

    # Méthode auxiliaire de construction de la liste des fréquences auxquelles réaliser une prédiction pour atteindre la fréquence cible
    def _build_frequency_prediction_list(
        self,
    ) -> List[Union[str, Dict]]:
        """Build the ordered list of frequencies at which to predict.

        Returns:
            List of frequencies, sorted from lowest to target.
            Each element is a str (TS) or Dict (panel).

        If ``cascade_refitting=False`` and ``keep_lower_frequencies=False``,
        returns a single-element list with the target frequency.
        Otherwise returns all intermediate frequencies from lowest detected
        up to the target, sorted by increasing granularity.

        For panel data, entities with heterogeneous base frequencies are
        separated into distinct dicts.
        """
        # Cas simple : pas de cascade ni de fréquences intermédiaires
        if not self.cascade_refitting and not self.keep_lower_frequencies:
            # Retourne directement la fréquence cible
            return [self.effective_target_frequency_]
        # Cas où il faut conserver des fréquences plus faibles
        else:
            # Construction de la liste des fréquences uniques à imputer (qui correspondent aux fréquences plus faibles que la fréquence cible)
            # Initialisation de l'ensemble des fréquences à imputer
            impute_freqs = set()
            # Parcours des variables
            for key, cat in self.variable_categories_.items():
                # Sléection des variables à imputer (à fréquence plus faible que la fréquence cible)
                if cat == 'impute':
                    # Extraction de la fréquence
                    freq = self.detected_frequencies_[key]
                    # Ajout à la liste
                    impute_freqs.add(normalize_frequency(freq, return_format='base'))

            # Retourne la fréquence cible s'il n'y a rien à imputer
            if not impute_freqs:
                return [self.effective_target_frequency_]

            # Tri des fréquences de la plus basse (order élevé) à la plus haute
            sorted_freqs = sorted(impute_freqs, key=get_frequency_order, reverse=True)

            # Cas des séries temporelles
            if not self.is_panel_:
                # Ajout de la fréquence cible si pas déjà présente
                target_norm = normalize_frequency(self.effective_target_frequency_, return_format='base')
                if target_norm not in sorted_freqs:
                    sorted_freqs.append(target_norm)
                return sorted_freqs
            # Cas des données de panel
            else:
                # Panel : on retourne des dictionnaires homogènes suivant la fréquence à prédire
                # Initialisation de la liste résultat
                freq_list = []
                # Parcours des fréquences
                for freq in sorted_freqs:
                    # Initialisation du dictionnaire pour la fréquence
                    freq_dict = {}
                    # Parcours des entités
                    for entity in self.effective_target_frequency_.keys():
                        # Ajout de la fréquence si la fréquence cible est supérieure ou égale à la fréquence cible
                        if is_higher_frequency(self.effective_target_frequency_[entity], freq) or (normalize_frequency(self.effective_target_frequency_, return_format='base') == freq):
                            freq_dict[entity] = freq
                    # Ajout du dictionnaire à la liste
                    freq_list.append(freq_dict)

                # Ajout de la fréquence cible
                # Distinction suivant son type (devrait normalement être un dictionnaire pour des données de panel)
                if isinstance(self.effective_target_frequency_, dict):
                    # Extraction des fréquences cibles qui ne sont pas déjà présentes
                    missing_target_freqs = set(normalize_frequency(target_freq, return_format='base') for target_freq in self.effective_target_frequency_.values()) - set()
                    if missing_target_freqs:
                        freq_dict = {entity: target_freq for entity, target_freq in self.effective_target_frequency_.items() if target_freq in missing_target_freqs }
                        freq_list.append(freq_dict)
                else:
                    # Création d'un dictionnaire avec la fréquence cible pour chaque entité
                    if normalize_frequency(self.effective_target_frequency_, return_format='base') not in impute_freqs:
                        freq_dict = {entity: self.effective_target_frequency_ for entity in (self.entities_ or [])}
                        freq_list.append(freq_dict)

                return freq_list

    # Méthode auxiliaire de préparation des données d'entraînement du modèle d'imputation
    # /!\ Revoir le nom de l'argument X_work
    # /!\ Voir si on n'appelle pas la méthode à tord plusieurs fois sur des données de panel et s'il ne serait pas préférable d'extraire d'abord les noms de variables de la clé pour l'appeler une seule fois par variable
    def _prepare_training_data(
        self,
        X_work: pd.DataFrame,
        var_key: Union[str, Tuple],
        pred_freq: Union[str, Dict],
        imputation_mask: pd.Series,
        cascade_imputed: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series, float]:
        """Prepare X_train, y_train and scale factor for a variable.

        Args:
            X_work: Current working data.
            var_key: Variable key to prepare training data for.
            pred_freq: Prediction frequency.
            imputation_mask: Mask for the imputation window.
            cascade_imputed: Previously imputed cascade data, or None.

        Returns:
            Tuple of (X_train, y_train, scale_factor).
        """
        # Extraction du nom de la variable
        var_name = var_key[-1] if isinstance(var_key, tuple) else var_key

        # Détermination des colonnes de features
        feature_cols = [
            c for c in X_work.columns
            if c != var_name
        ]

        # Si la série est univariée, renvoie des éléments vides
        if not feature_cols:
            return pd.DataFrame(), pd.Series(dtype=float), 1.0

        # Masque d'entraînement : fenêtre d'entraînement + valeurs non-null pour y
        if hasattr(self, '_imputation_window_calc') and self._imputation_window_calc._is_fitted:
            # Extraction du mask d'imputation
            training_mask = self._imputation_window_calc.get_imputation_window_mask(X_work)
            # Hybridation avec la période de disponibilité de y
            # /!\ Que se passe-t-il lorsque X_work a un niveau de panel supplémentaire lié aux fréquences quand "keep_lower_frequencies"=True ou "cascade_refitting"=True
            # /!\ Peut-être faire une version de la méthode avec des données en argument et que l'on peut réentrainer, il n'y a pas besoin de vérifier que le caluclateur est fitted dans ce cas
            # /!\ une autre option serait d'instancier une nouvelle classe et de l'entrainer sur X_work
            training_mask &= X_work[var_name].notna()
        else:
            training_mask = X_work[var_name].notna()

        # Restriction aux données originales si train_on_partial_coverage=False
        if not self.train_on_partial_coverage:
            if hasattr(self, '_provenance_tracker'):
                original_mask = self._provenance_tracker.get_mask(
                    ProvenanceType.ORIGINAL, column=var_name
                )
                # /!\ Vérifier la compatibilité des index
                training_mask = training_mask & original_mask

        # Ajout des données de cascade si cascade_refitting=True
        if self.cascade_refitting and cascade_imputed is not None:
            # Parcours des colonnes
            for col in feature_cols:
                if col in cascade_imputed.columns:
                    # Remplissage les NaN de X_work avec les valeurs imputées
                    null_mask = X_work[col].isna()
                    X_work.loc[null_mask, col] = cascade_imputed.loc[
                        null_mask & cascade_imputed[col].notna(), col
                    ]

        # Extraction des données d'entraînement
        X_train = X_work.loc[training_mask, feature_cols]
        y_train = X_work.loc[training_mask, var_name]

        # Calcul du facteur de mise à l'échelle
        # Extraction de la fréquence de prédiction s'il s'agit d'un dictionnaire
        if isinstance(pred_freq, dict):
            if isinstance(var_key, tuple):
                # Création des deux systèmes de clé
                entity = var_key[:-1] if len(var_key) > 2 else var_key[0]
                entity_key = entity if isinstance(entity, tuple) else (entity,)
                # Extraction de la fréquence
                pf = pred_freq.get(entity_key) or pred_freq.get(entity)
            else:
                pf = list(pred_freq.values())[0]
        else:
            pf = pred_freq

        # Calcul du facteur de mise à l'échelle
        scale_factor = self._freq_converter.get_conversion_factor(
            normalize_frequency(self.detected_frequencies_[var_key]),
            normalize_frequency(pf)
        )

        return X_train, y_train, scale_factor

    # Méthode auxiliaire d'application du facteur de mise à l'échelle aux données
    def _apply_frequency_scaling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_factor: float,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply frequency scaling to training data.

        Args:
            X_train: Training features.
            y_train: Training target.
            scale_factor: Conversion factor.

        Returns:
            Tuple of (scaled_X_train, scaled_y_train).
        """
        # Retourne les données inchangées si le facteur est égal à 1
        if scale_factor == 1.0:
            return X_train, y_train

        # Mise à l'échelle si demandé
        if self.scale_features:
            X_scaled = X_train / scale_factor
            y_scaled = y_train / scale_factor
        else:
            X_scaled = X_train
            y_scaled = y_train

        return X_scaled, y_scaled

    # Méthode auxiliaire de détermination des échantillons de prédiction
    def _determine_prediction_samples(
        self,
        X_work: pd.DataFrame,
        var_name: str,
        imputation_mask: pd.Series,
    ) -> List[Tuple[pd.Index, List[str]]]:
        """Determine prediction sample groups by available covariates.

        Within the imputation window, groups NaN observations by which
        covariates are available. Orders from most covariates to fewest.

        Args:
            X_work: Working data.
            var_name: Target variable name.
            imputation_mask: Boolean mask for imputation window.

        Returns:
            List of (index, feature_cols) tuples, ordered from most
            features to fewest.
        """
        # Détection des observations manquantes de la variable à prédire dans la fenêtre d'imputation
        missing_in_window = imputation_mask & X_work[var_name].isna()
        
        # Retourne une liste vide s'il n'y a aucune observation à imputer
        if not missing_in_window.any():
            return []

        # Extraction des features
        feature_cols = [
            c for c in X_work.columns
            if c != var_name
        ]
        # Extraction des index à imputer
        missing_idx = X_work.index[missing_in_window]

        # Regroupement par pattern de covariables disponibles
        groups: Dict[tuple, List] = {}
        # Parcours des index
        for idx in missing_idx:
            # Extraction des features dispobibles pour l'index
            available = tuple(
                col for col in feature_cols if pd.notna(X_work.at[idx, col])
            )
            # Ajout de la clé au dictionnaire si elle n'existe pas déjà
            if available not in groups:
                groups[available] = []
            # Ajout de l'index
            groups[available].append(idx)

        # Tri par nombre de covariables décroissant
        result = []
        for available_cols, indices in sorted(groups.items(), key=lambda x: -len(x[0])):
            result.append((pd.Index(indices), list(available_cols)))

        return result

    # Méthode auxiliaire de notation de la provenance des variables agrégées
    def _mark_aggregated_provenance(
        self,
        tracker: ImputationProvenanceTracker,
        X: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]]
    ) -> None:
        """Mark aggregated values in the provenance tracker.

        Args:
            tracker: Provenance tracker instance.
            X: DataFrame with current data.
            aggregate_keys: Variable keys that were aggregated.
        """
        # Ne fait rien si aucune clé d'agrégation n'est fournie
        if not aggregate_keys:
            return

        # Distinction suivant la structure des données
        # Cas de données de panel
        if self.is_panel_:
            # Regroupement des clés par entités
            grouped = self._freq_aligner.group_keys_by_entity_and_variable(aggregate_keys)
            # Parcours des entités et de leurs colonnes associées
            for entity, cols in grouped.items():
                # Extraction des observations liées à l'entité
                entity_mask = self._freq_aligner.get_entity_mask(X, entity)
                entity_index = X.index[entity_mask]
                # Parcours des colonnes
                for col in cols:
                    # Marquage comme agrégé
                    if col in X.columns:
                        tracker.mark_aggregated(col, entity_index)
        # Cas de données de séries temporelles
        else:
            # Extraction des noms de colonne des tuples
            columns = self._freq_aligner.extract_column_names(aggregate_keys)
            # Parcours des colonnes
            for col in columns:
                # Marquage comme aggrégé
                if col in X.columns:
                    tracker.mark_aggregated(col, X.index)

    # /!\ Voir s'il ne faut pas supprimer cette méthode et se reposer uniquement sur le imputation_scope pour imputer les variables sujettes à délais de publication
    def _infer_delays_from_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Infer publication delays from data.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with delay information.
        """
        # Import différé pour éviter l'import circulaire avec delays.data_manager
        from ..delays.data_manager import compare_and_detect_delays

        try:
            delays_df = compare_and_detect_delays(
                new_data=X,
                existing_data=None,
                download_date=None,
                detection_mode='new_only',
                reference_point='end',
                delay_unit='D',
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

    # Méthode auxiliaire de regroupement des variables à imputer par fréquence
    def _group_variables_by_frequency(self) -> Dict[int, List[Union[str, Tuple]]]:
        """Group variables to impute by frequency level.

        Returns:
            Dict mapping frequency level (int) to list of variable keys.
            Lower frequency = higher level number.
        """
        # Initialisation du dictionnaire résultat associant à chaque fréquence une liste de variables / entité X variable
        # Chaque fréquence sera représentée par son ordre
        # /!\ Voir si on ne peut pas faire en sorte que les clés soient les fréquences détectées mais qu'elles soient ordonnées par frequency_order ?
        freq_groups: Dict[int, List[Union[str, Tuple]]] = {}

        # Parcours des clés désignant les variables à imputer
        for var_key in self.imputation_order_:
            # Détection de la fréquence associée
            freq = self.detected_frequencies_[var_key]
            # Extraction de l'ordre associé à la fréquence
            freq_order = int(get_frequency_order(freq))

            # Ajout de l'ordre au dictionnaire résultat s'il n'existe pas déjà
            if freq_order not in freq_groups:
                freq_groups[freq_order] = []
            # Ajout de la clé à la liste des variables à cette fréquence
            freq_groups[freq_order].append(var_key)

        # Tri du dictionnaire par ordre
        return dict(sorted(freq_groups.items(), reverse=True))

    # Méthode auxiliaire d'imputation de l'évolution de la fréquence d'une variable au gré de ses imputations
    # /!\ Peut-être supprimer
    def _compute_frequency_progression(self) -> Dict[str, List[str]]:
        """Compute the frequency progression for each variable.

        Returns:
            Dict mapping variable names to list of frequency stages.
        """
        # Initialisation du dictionnaire résultat
        progression = {}

        # Parcours des variables à imputer
        for key in self.imputation_order_:
            # Extraction du nom de la variable
            var_name = key[-1] if isinstance(key, tuple) else key
            # Extraction de la fréquence source
            source_freq = self.detected_frequencies_[key]

            # Extraction de la fréquence cible
            if self.is_panel_ and isinstance(key, tuple):
                # Normalisation de l'entité sous forme de tuple
                entity_key = normalize_entity_key(key[:-1])
                if isinstance(self.effective_target_frequency_, dict):
                    target_freq = (
                        self.effective_target_frequency_.get(entity_key)
                        or self.effective_target_frequency_.get(entity_key[0])
                    )
                else:
                    target_freq = self.effective_target_frequency_
            else:
                target_freq = self.effective_target_frequency_

            # Ajout de la progression associée à la variable
            # /!\ Etrange car ce n'est pas forcément la même par entité, j'ai l'impression qu'on ne garde ici que la première occurence
            if var_name not in progression:
                progression[var_name] = [source_freq, target_freq]

        return progression

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'entraînement
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Learn transformation parameters from X and y.

        Implements the cascade imputation fitting algorithm:

        PHASE 0: Setup (columns, panel detection, frequency detection, validation)
        PHASE 1: Imputation window calculation
        PHASE 2: Additive transformer
        PHASE 3: Build frequency prediction list
        PHASE 4: Initialize provenance
        PHASE 5: Iterate over frequency prediction list and fit models
        PHASE 6: Finalization

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_targets).
        """
        # =================================================================
        # PHASE 0 — Setup
        # =================================================================
        self.feature_columns_ = list(X.columns)
        self.target_column_ = y.name if y is not None else None
        self.is_panel_ = bool(self.panel_cols) or is_panel_data(data=X)

        # Construction du jeu de données de travail
        if y is not None:
            # Concaténation de X et y si y spécifié
            # Vérification que X et y ont la même longueur
            if len(X) != len(y):
                raise ValueError("X and y should be of equal length")
            X_work = pd.concat([X, y.to_frame()], axis=1)
        else:
            X_work = X.copy()

        # Identification des entités
        if self.is_panel_ and isinstance(X.index, pd.MultiIndex):
            self.entities_ = get_unique_panel_entities(X)
        else:
            self.entities_ = None

        # Expansion de target_frequency en dict si panel + string
        if self.is_panel_ and isinstance(self.target_frequency, str) and self.entities_:
            self.effective_target_frequency_ = {
                entity: self.target_frequency for entity in self.entities_
            }
        elif isinstance(self.target_frequency, dict):
            self.effective_target_frequency_ = self.target_frequency.copy()
        else:
            self.effective_target_frequency_ = self.target_frequency

        # Détection des fréquences
        self.detected_frequencies_ = detect_frequency(data=X_work)
        if not self.detected_frequencies_:
            raise ValueError("Could not detect frequency for any column")

        # Validation de la fréquence cible via TargetFrequencyValidator
        self.effective_target_frequency_ = self._target_freq_validator.validate(
            target_frequency=self.effective_target_frequency_,
            detected_frequencies=self.detected_frequencies_,
            on_frequency_mismatch=self.on_frequency_mismatch,
        )

        # Classification et ordre d'imputation
        self.variable_categories_ = self._classify_variables()
        self.imputation_order_ = self._determine_imputation_order()

        # =================================================================
        # PHASE 1 — Imputation window
        # =================================================================
        self._imputation_window_calc = ImputationWindowCalculator(
            coverage_threshold=self.coverage_threshold,
            imputation_scope=self.imputation_scope,
            min_columns=2,
        )
        try:
            self._imputation_window_calc.fit(X_work)
            self.imputation_window_ = (
                self._imputation_window_calc.imputation_window_start_,
                self._imputation_window_calc.imputation_window_end_
            )
        except ValueError as e:
            warnings.warn(
                f"Could not calculate imputation window: {e}. Using all available data.",
                UserWarning
            )
            if isinstance(X_work.index, pd.MultiIndex):
                time_idx = X_work.index.get_level_values(-1)
            else:
                time_idx = X_work.index
            self.imputation_window_ = (time_idx.min(), time_idx.max())

        # =================================================================
        # PHASE 2 — Additive transformer
        # =================================================================
        if self.additive_transformer is not None:
            self.additive_transformer_ = clone(self.additive_transformer)
            X_work = self.additive_transformer_.fit_transform(X_work)
            if isinstance(X_work, tuple):
                X_work = X_work[0]
        else:
            self.additive_transformer_ = None

        # =================================================================
        # PHASE 3 — Frequency prediction list
        # =================================================================
        freq_prediction_list = self._build_frequency_prediction_list()

        # =================================================================
        # PHASE 4 — Provenance initialization
        # =================================================================
        self._provenance_tracker = ImputationProvenanceTracker()
        self._provenance_tracker.initialize(X_work, panel_cols=self.panel_cols)

        # =================================================================
        # PHASE 5 — Iterate over frequency prediction list
        # =================================================================
        self.imputation_models_ = OrderedDict()
        self.model_fitting_order_ = []
        cascade_imputed = None

        for pred_freq in freq_prediction_list:
            # 5a. Classification des variables relative à pred_freq
            var_classification = self._classify_variables_at_frequency(pred_freq)
            aggregate_keys = var_classification['aggregate']
            impute_keys = var_classification['impute']

            # 5b. Ordonnancement des variables à imputer
            if self.train_on_partial_fit_order == 'cv' and self.train_on_partial_coverage:
                ordered_impute_keys = self._determine_variable_order_cv(X_work, impute_keys)
            else:
                # Tri par fréquence (plus basse d'abord)
                ordered_impute_keys = sorted(
                    impute_keys,
                    key=lambda k: get_frequency_order(
                        self.detected_frequencies_.get(k, 'D')
                    ),
                    reverse=True,
                )

            # 5c. Agrégation des variables haute fréquence
            X_at_pred_freq = self._freq_aligner.aggregate_to_target(
                X_work, aggregate_keys,
                pred_freq if isinstance(pred_freq, (str, dict)) else pred_freq,
                self.is_panel_
            )
            self._mark_aggregated_provenance(
                self._provenance_tracker, X_at_pred_freq, aggregate_keys
            )

            # 5d. Pour chaque variable à imputer
            for var_key in ordered_impute_keys:
                var_name = var_key[-1] if isinstance(var_key, tuple) else var_key

                estimator = self._get_estimator_for_variable(var_name)
                if estimator is None:
                    warnings.warn(
                        f"No estimator available for variable '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    self.imputation_models_[var_key] = 'interpolate_fallback'
                    self.model_fitting_order_.append((pred_freq, var_key))
                    continue

                # Détermination des échantillons de prédiction
                if self._imputation_window_calc._is_fitted:
                    imputation_mask = self._imputation_window_calc.get_imputation_window_mask(
                        X_at_pred_freq
                    )
                else:
                    imputation_mask = pd.Series(True, index=X_at_pred_freq.index)

                # Préparation des données d'entraînement
                X_train, y_train, scale_factor = self._prepare_training_data(
                    X_at_pred_freq, var_key, pred_freq, imputation_mask, cascade_imputed
                )

                if len(X_train) < 2:
                    warnings.warn(
                        f"Not enough training data for variable '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    self.imputation_models_[var_key] = 'interpolate_fallback'
                    self.model_fitting_order_.append((pred_freq, var_key))
                    continue

                # 5e. Frequency scaling
                X_train_scaled, y_train_scaled = self._apply_frequency_scaling(
                    X_train, y_train, scale_factor
                )

                # Imputation simple des NaN dans les features
                X_train_scaled = X_train_scaled.fillna(X_train_scaled.mean())

                # Suppression des lignes avec y NaN
                valid_mask = y_train_scaled.notna()
                X_train_scaled = X_train_scaled.loc[valid_mask]
                y_train_scaled = y_train_scaled.loc[valid_mask]

                if len(X_train_scaled) < 2:
                    self.imputation_models_[var_key] = 'interpolate_fallback'
                    self.model_fitting_order_.append((pred_freq, var_key))
                    continue

                # 5f. Fit du modèle + stockage metadata
                feature_cols = list(X_train_scaled.columns)
                try:
                    estimator.fit(X_train_scaled, y_train_scaled)
                    self.imputation_models_[var_key] = {
                        'model': estimator,
                        'feature_cols': feature_cols,
                        'scale_factor': scale_factor,
                        'pred_freq': pred_freq,
                        'trained_on_imputed': (
                            self.train_on_partial_coverage
                            and cascade_imputed is not None
                        ),
                    }
                except Exception as e:
                    warnings.warn(
                        f"Failed to fit model for variable '{var_name}': {e}. "
                        f"Using linear interpolation as fallback"
                    )
                    self.imputation_models_[var_key] = 'interpolate_fallback'

                self.model_fitting_order_.append((pred_freq, var_key))

                # 5g. Cascade refitting : application des prédictions intermédiaires
                if self.cascade_refitting:
                    model_info = self.imputation_models_.get(var_key)
                    if model_info and model_info != 'interpolate_fallback':
                        missing_mask = X_at_pred_freq[var_name].isna()
                        if missing_mask.any():
                            X_predict = X_at_pred_freq.loc[missing_mask, feature_cols]
                            X_predict = X_predict.fillna(X_predict.mean())
                            try:
                                preds = model_info['model'].predict(X_predict)
                                # Inverse scaling
                                preds = preds * scale_factor
                                X_at_pred_freq.loc[missing_mask, var_name] = preds

                                if cascade_imputed is None:
                                    cascade_imputed = pd.DataFrame(
                                        index=X_at_pred_freq.index
                                    )
                                cascade_imputed[var_name] = X_at_pred_freq[var_name]

                                # Marquage provenance
                                self._provenance_tracker.mark_model_imputed(
                                    var_name,
                                    X_at_pred_freq.index[missing_mask],
                                    trained_on_imputed=model_info.get('trained_on_imputed', False)
                                )
                            except Exception as e:
                                warnings.warn(
                                    f"Intermediate imputation failed for '{var_name}': {e}"
                                )

        # =================================================================
        # PHASE 6 — Finalisation
        # =================================================================
        self.frequency_progression_ = self._compute_frequency_progression()

        if self.impute_delayed_values and self.delays is None:
            self.inferred_delays_ = self._infer_delays_from_data(X)
        else:
            self.inferred_delays_ = pd.DataFrame()

        self.imputation_provenance_ = self._provenance_tracker.get_provenance_matrix()

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------
    def _transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Transform X and optionally y using cascade imputation.

        Replays the model fitting order from fit to ensure consistent results.

        Args:
            X: Features to transform.
            y: Targets to transform (optional).

        Returns:
            X_transformed if y is None.
            (X_transformed, y_transformed) if y is provided.
        """
        # 1. Setup
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        self._original_X_ = X.copy()
        self._original_y_ = y.copy() if y is not None else None

        y_col_name = None
        if y is not None:
            y_col_name = y.name if y.name is not None else '__target__'
            data_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            data_work = X.copy()

        if not isinstance(data_work.index, (pd.DatetimeIndex, pd.MultiIndex)):
            if self.time_col and self.time_col in data_work.columns:
                data_work = data_work.set_index(self.time_col)
            else:
                raise ValueError("Data must have a DatetimeIndex or MultiIndex")

        # Init provenance tracker pour la transformation
        transform_tracker = ImputationProvenanceTracker()
        transform_tracker.initialize(data_work, panel_cols=self.panel_cols)

        # 2. Appliquer additive transformer
        if self.additive_transformer_ is not None:
            data_transformed = self.additive_transformer_.transform(data_work)
            if isinstance(data_transformed, tuple):
                data_transformed = data_transformed[0]
        else:
            data_transformed = data_work.copy()

        # 3. Rejouer model_fitting_order_ dans le même ordre
        intermediate_results: Dict[str, pd.DataFrame] = {}
        current_pred_freq = None

        for pred_freq, var_key in self.model_fitting_order_:
            var_name = var_key[-1] if isinstance(var_key, tuple) else var_key

            # Agrégation au pred_freq si changement de fréquence
            if pred_freq != current_pred_freq:
                current_pred_freq = pred_freq
                var_classification = self._classify_variables_at_frequency(pred_freq)
                aggregate_keys = var_classification['aggregate']
                data_transformed = self._freq_aligner.aggregate_to_target(
                    data_transformed, aggregate_keys, pred_freq, self.is_panel_
                )
                self._mark_aggregated_provenance(
                    transform_tracker, data_transformed, aggregate_keys
                )

                # Stocker résultat intermédiaire si keep_lower_frequencies
                if self.keep_lower_frequencies:
                    freq_label = (
                        str(pred_freq) if isinstance(pred_freq, str)
                        else str(list(pred_freq.values())[0]) if pred_freq else 'unknown'
                    )
                    intermediate_results[freq_label] = data_transformed.copy()

            # Récupérer model_info
            model_info = self.imputation_models_.get(var_key)

            if model_info is None or model_info == 'interpolate_fallback':
                if var_name in data_transformed.columns:
                    data_transformed[var_name] = data_transformed[var_name].interpolate(
                        method='linear', limit_direction='both'
                    )
                continue

            # Identification des valeurs manquantes
            if var_name not in data_transformed.columns:
                continue

            missing_mask = data_transformed[var_name].isna()
            if not missing_mask.any():
                continue

            feature_cols = model_info.get('feature_cols', [])
            scale_factor = model_info.get('scale_factor', 1.0)
            X_features = data_transformed.loc[missing_mask, feature_cols]
            X_features = X_features.fillna(X_features.mean())

            try:
                model = model_info['model']
                predictions = model.predict(X_features)
                # Inverse scaling : multiplier par scale_factor
                predictions = predictions * scale_factor
                data_transformed.loc[missing_mask, var_name] = predictions

                # Marquage provenance
                trained_on_imputed = model_info.get('trained_on_imputed', False)
                transform_tracker.mark_model_imputed(
                    var_name,
                    data_transformed.index[missing_mask],
                    trained_on_imputed=trained_on_imputed
                )
            except Exception as e:
                warnings.warn(
                    f"Prediction failed for variable '{var_name}': {e}. "
                    f"Using interpolation fallback."
                )
                data_transformed[var_name] = data_transformed[var_name].interpolate(
                    method='linear', limit_direction='both'
                )

        # 4. Imputer valeurs retardées si demandé
        if self.impute_delayed_values:
            data_transformed = self._impute_delayed_values(data_transformed)

        # 5. Construire sortie MultiIndex si keep_lower_frequencies
        if self.keep_lower_frequencies and intermediate_results:
            data_result = self._build_multifreq_output(
                data_transformed, intermediate_results
            )
        else:
            data_result = data_transformed

        # 6. Mise à jour de la provenance
        self.imputation_provenance_ = transform_tracker.get_provenance_matrix()

        # 7. Scission X et y
        if y is not None and y_col_name in data_result.columns:
            y_transformed = data_result[y_col_name]
            X_transformed = data_result.drop(columns=[y_col_name])
            return X_transformed, y_transformed
        else:
            return data_result

    # -------------------------------------------------------------------------
    # Méthodes auxiliaires de transform
    # -------------------------------------------------------------------------
    def _build_multifreq_output(
        self,
        final_result: pd.DataFrame,
        intermediate_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Build MultiIndex output with all frequency levels.

        Args:
            final_result: Final DataFrame at target frequency.
            intermediate_results: Dict mapping frequency labels to DataFrames.

        Returns:
            DataFrame with MultiIndex:
            - Time series: (Frequency, Date)
            - Panel: (Entity, Frequency, Date)
        """
        all_frames = []

        for freq_label, df in intermediate_results.items():
            df_copy = df.copy()
            df_copy['_frequency_level'] = freq_label
            all_frames.append(df_copy)

        final_copy = final_result.copy()
        final_copy['_frequency_level'] = 'target'
        all_frames.append(final_copy)

        combined = pd.concat(all_frames, ignore_index=False)

        if self.is_panel_:
            if isinstance(combined.index, pd.MultiIndex):
                entity_values = combined.index.get_level_values(0)
                date_values = combined.index.get_level_values(-1)
                freq_values = combined['_frequency_level']

                new_index = pd.MultiIndex.from_arrays(
                    [entity_values, freq_values, date_values],
                    names=['entity', 'frequency', 'date']
                )
            else:
                new_index = pd.MultiIndex.from_arrays(
                    [combined['_frequency_level'], combined.index],
                    names=['frequency', 'date']
                )
        else:
            new_index = pd.MultiIndex.from_arrays(
                [combined['_frequency_level'], combined.index],
                names=['frequency', 'date']
            )

        combined.index = new_index
        combined = combined.drop(columns=['_frequency_level'])

        return combined

    # /!\ Revoir s'il ne faut pas supprimer cette méthode et utiliser seulement le imputation_scope
    def _impute_delayed_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute values affected by publication delays.

        Args:
            X: DataFrame with imputed values.

        Returns:
            DataFrame with delayed values imputed.
        """
        result = X.copy()
        delays_to_use = self.delays if self.delays is not None else self.inferred_delays_

        if delays_to_use.empty:
            return result

        for _, row in delays_to_use.iterrows():
            column = row['column']
            delay = row['delay']

            if column not in result.columns:
                continue

            n_delay = int(delay)
            if n_delay <= 0:
                continue

            delayed_idx = result.index[-n_delay:]
            missing_mask = result.loc[delayed_idx, column].isna()

            if not missing_mask.any():
                continue

            model_info = self.imputation_models_.get(column)

            if model_info is None or model_info == 'interpolate_fallback':
                result[column] = result[column].interpolate(
                    method='linear', limit_direction='both'
                )
            else:
                feature_cols = model_info.get('feature_cols', [])
                scale_factor = model_info.get('scale_factor', 1.0)
                missing_idx = delayed_idx[missing_mask.values]
                X_features = result.loc[missing_idx, feature_cols]

                if not X_features.empty:
                    X_features = X_features.fillna(X_features.mean())
                    try:
                        predictions = model_info['model'].predict(X_features)
                        predictions = predictions * scale_factor
                        result.loc[missing_idx, column] = predictions
                    except Exception as e:
                        warnings.warn(
                            f"Failed to impute delayed values for '{column}': {e}"
                        )

        return result

    # -------------------------------------------------------------------------
    # Transformation inverse
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de transformation inverse
    # /!\ A revoir car il faut utiliser le provenance tracker (l'inversion de l'additive transformer doit être faite dans la transformation à la fin)
    def _inverse_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Inverse transform X and optionally y.

        Note: Disaggregation from lower to higher frequency is lossy.

        Args:
            X: Transformed features.
            y: Transformed targets (optional).

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.
        """
        y_col_name = None
        if y is not None:
            y_col_name = y.name if y.name is not None else '__target__'
            data_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            data_work = X.copy()

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

        if y is not None and y_col_name in data_result.columns:
            y_original = data_result[y_col_name]
            X_original = data_result.drop(columns=[y_col_name])
            return X_original, y_original
        else:
            return data_result
