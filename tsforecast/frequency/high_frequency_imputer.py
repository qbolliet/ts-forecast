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
from .detector import FrequencyDetector, detect_frequency, detect_index_frequency
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
    2. Computing the imputation window (where all series have true values),
       optionally extended forward to cover delayed series ends
       (imputation_scope='extended_forward' + coverage_threshold)
    3. Aggregating high-frequency variables to lower frequencies
    4. Cascading imputation from lowest to highest frequency
    5. Optionally refitting models with imputed values (cascade_refitting)
    6. Tracking provenance of each imputed value

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
            frequency stage for cascade imputation. Drives the number of fits:
            one ``estimator.fit`` per (stage, variable imputable at that stage)
            when True, one per imputable variable when False — the model is
            then reused at the following stages, only its ``scale_factor``
            changing, predictions being rescaled accordingly.
        keep_lower_frequencies: If True, output includes all intermediate
            frequencies in a MultiIndex structure: (entity, frequency, date)
            for panel data, (frequency, date) for time series. Every level,
            including the target-frequency one, is labeled with its real
            frequency string — never 'target' — so the target level is
            identified via effective_target_frequency_. If False, the
            output is the plain frame of the last cascade stage (target
            frequency), with no frequency level in its index.
        delays: Deprecated. Publication delays are now handled by
            imputation_scope='extended_forward' combined with
            coverage_threshold, which extends the training window to cover
            delayed series ends directly (review §2.9). Still structurally
            validated at init for backward compatibility, but never
            consumed at fit/transform time.
        impute_delayed_values: Deprecated. If True, emits a
            DeprecationWarning at fit time and otherwise has no effect: use
            imputation_scope='extended_forward' (with coverage_threshold)
            instead, which handles delayed series ends through the standard
            extended-window mechanism.
        on_frequency_mismatch: How to handle target_frequency higher than data ('error'/'warn').
        coverage_threshold: Minimum ratio of columns with data (0-1) for extended window.
        imputation_scope: Training window scope ('strict', 'extended_backward',
            'extended_forward', 'extended_both').
        train_on_partial_coverage: If True, use imputed values for training outside P1.
        train_on_partial_fit_order: Order for imputing variables when
            train_on_partial_coverage is True:
            - 'random': Sort by frequency level then entity count (default)
            - 'cv': Use cross-validation to impute easiest variables first
        scale_features: If True, divide X_train by the number of sub-periods
            as well as y_train, which is always divided. Set it to True when
            the training covariates were aggregated by summation to the
            variable's frequency, so that both sides of the model are
            expressed at the sub-period scale used at prediction time.
        enforce_period_totals: If True (default), the sub-periods predicted
            for one period of a lower-frequency variable are rescaled
            proportionally so that they sum back to the value observed for
            that period (review §2.6, option 1). Only the additive
            constraint is optional: an imputed variable is ALWAYS predicted
            over its whole period, anchor dates included, so that its column
            never mixes the low-frequency total with sub-period values. Set
            it to False to keep the raw model output, homogeneous in scale
            but free of any additive constraint. Periods that are only
            partly predicted, that hold no observation, or whose predictions
            sum to zero are never rescaled.
        restore_original_values: If True, ``inverse_transform`` refills the
            cells that were non-NaN in the input of the last ``transform``
            with their exact original values, on top of the provenance-based
            restoration (review §2.10). Needed to recover the anchor dates
            of a lower-frequency variable: they carry a true observation in
            the input, but the target level of the output spreads them over
            their period, so their provenance there is DISAGGREGATED and the
            ORIGINAL mask alone sets them back to NaN. Default is False,
            which keeps ``inverse_transform`` a pure function of the
            transformed frame and its provenance.

    Attributes:
        detected_frequencies_: Detected frequency per variable or (entity, variable).
        variable_categories_: Category per (entity, variable) tuple:
            'aggregate', 'impute', or 'target_freq'.
        imputation_order_: Ordered list of variables for cascading imputation.
        imputation_models_: Fitted imputation models, keyed by cascade stage:
            ``stage_key = (freq_label, group_key)`` where ``freq_label`` is
            :meth:`_freq_label` of the stage frequency (the frequency string
            for a time series, a frozenset of the entity -> frequency items
            for a panel) and ``group_key`` is:
            - for a time series: the variable (column) name;
            - for a panel (review §2.4): the variable name alone when every
              entity shares the same detected frequency for it at this
              stage, otherwise ``(variable name, detected frequency)`` when
              entities disagree — NOT ``(entity, variable)``. The model
              backing a panel entry is GLOBAL, fitted once on every entity
              of the group rather than once per entity (which would only
              refit and overwrite the exact same model).
            A variable imputed at two stages therefore holds two distinct
            entries. Each value is either ``'interpolate_fallback'`` or a
            dict with keys ``model``, ``feature_cols``, ``feature_means``
            (per-covariate means of the TRAINING set, reapplied to fill the
            missing covariates at prediction time, review §2.7),
            ``scale_factor`` (sub-period count of the stage),
            ``fit_scale_factor`` (the one baked into the model at fit time),
            ``pred_freq`` and ``trained_on_imputed``.
        model_fitting_order_: List of those same ``stage_key`` tuples, in the
            exact order in which stages were registered, for replay during
            transform. ``len(imputation_models_) == len(model_fitting_order_)``
            after any fit.
        stage_groups_: Metadata of each ``stage_key``, needed at replay time
            and populated for EVERY registered stage, interpolation fallbacks
            included (their ``imputation_models_`` entry is a bare string and
            carries none). Each value is a dict with keys ``var_name``,
            ``f_var`` (the group's normalized source frequency, which drives
            the disaggregation periods) and ``entities`` (the entity tuples
            of the group, None for a time series).
        freq_prediction_list_: Ordered list of the prediction frequencies
            (cascade stages) built at fit time. ``transform`` replays exactly
            these stages, rebuilding each stage frame from the input data via
            :meth:`_build_stage_frame`, so that fit and transform work on
            identical stage frames for identical data.
        imputation_provenance_fit_: DataFrame tracking origin of each value
            ('original', 'model_on_true', 'model_on_mixed', 'aggregated',
            'disaggregated') as seen at the end of ``fit``. Single-level
            matrix, indexed like the fit input: ``fit`` never builds a
            multi-frequency output, so there is no per-level breakdown to
            produce here (contrast with ``imputation_provenance_`` below).
        imputation_provenance_: Same categories as above, but reflecting the
            LAST ``transform`` call — distinct from ``imputation_provenance_fit_``,
            which ``transform`` never touches (review §2.8.1). With
            ``keep_lower_frequencies=False`` this is a single matrix at the
            target frequency. With ``keep_lower_frequencies=True`` it is
            stacked exactly like the transformed output — MultiIndex
            ``(frequency, date)`` for a time series, ``(entity, frequency,
            date)`` for a panel, one row-block per cascade stage — so that a
            cell's provenance always describes the value actually present at
            that (frequency, date) in the output, instead of one single
            level's provenance being reused (falsely) to describe every
            level (review §2.8.4).
        imputation_window_: Tuple (start, end) of the imputation window where
            all series have data.
        training_window_: Tuple (start, end) of the extended training window.
        frequency_progression_: Dict mapping variables to their frequency stages.
        additive_transformer_: Fitted additive transformer.
        is_panel_: Whether data is panel data.
        feature_columns_: X columns (features).
        target_column_: y column if provided.
        effective_target_frequency_: Actual target frequency used after validation.
        entities_: Unique entities in panel data.
        _source_frequency_label_: Frequency label of the index seen at fit
            time, in the very format used for the ``frequency`` level of a
            multi-frequency output. ``inverse_transform`` keeps that level
            and drops the others (review §2.10). None when the index
            frequency could not be detected — the target level is then used
            instead.
        _original_X_ / _original_y_: Snapshot of the input of the LAST
            ``transform`` call, overwritten at each call. ``transform`` is
            therefore stateful: it records what it was given so that
            ``inverse_transform`` can restore the source index, its names
            and — when ``restore_original_values=True`` — the original
            values themselves.

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
        enforce_period_totals: bool = True,
        restore_original_values: bool = False,
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
                each frequency stage — one fit per (stage, variable imputable
                at that stage). Enables more accurate imputation in later
                stages. If False, each imputable variable is fitted once, at
                the first stage where it is imputable, then reused at the
                following stages with the scale factor of the stage.
            keep_lower_frequencies: If True, output includes all intermediate
                frequencies in a MultiIndex structure — (entity, frequency,
                date) for panel data, (frequency, date) for time series —
                every level labeled with its real frequency string (the
                target level is identified via effective_target_frequency_,
                not by a dedicated 'target' label). If False, only the
                target frequency is returned as a plain frame.
            impute_delayed_values: Deprecated. If True, emits a
                DeprecationWarning at fit time and otherwise has no effect:
                use imputation_scope='extended_forward' (with
                coverage_threshold) instead, which extends the training
                window to cover delayed series ends through the standard
                extended-window mechanism. Default is False.
            delays: Deprecated. Publication delays are now handled by
                imputation_scope='extended_forward' combined with
                coverage_threshold. Still structurally validated at init
                for backward compatibility, but never consumed at
                fit/transform time.
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
            scale_features: If True, divide X_train by the number of
                sub-periods alongside y_train, which is always divided.
            enforce_period_totals: If True (default), rescale the sub-periods
                predicted for one period of a lower-frequency variable so
                that they sum back to the value observed for that period
                (review §2.6, option 1). Independently of this flag, an
                imputed variable is always predicted over its whole period —
                anchor dates included — so that its column never mixes the
                low-frequency total with sub-period values. False keeps the
                raw model output, without any additive constraint.
            restore_original_values: If True, ``inverse_transform`` refills
                every cell that was non-NaN in the input of the last
                ``transform`` with its exact original value, recovering in
                particular the anchor dates of the lower-frequency variables
                (DISAGGREGATED at the target level, hence dropped by the
                ORIGINAL mask). Default is False.
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
        self.enforce_period_totals = enforce_period_totals
        self.restore_original_values = restore_original_values

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

    # Méthode auxiliaire de construction d'un label hashable pour une fréquence de prédiction
    @staticmethod
    def _freq_label(pred_freq: Union[str, Dict]) -> Union[str, frozenset]:
        """Build a hashable, canonical label for a prediction frequency.

        Panel stages carry a dict entity -> frequency, which cannot be used
        as a dictionary key: it is turned into a frozenset of its items,
        canonical whatever the insertion order.

        Args:
            pred_freq: Prediction frequency of a cascade stage (str for a
                time series, dict entity -> frequency for a panel).

        Returns:
            The frequency string itself, or a frozenset of the dict items.

        Examples:
            >>> HighFrequencyImputer._freq_label('M')
            'M'
            >>> HighFrequencyImputer._freq_label({('France',): 'M'})
            frozenset({(('France',), 'M')})
        """
        # Les dictionnaires par entité ne sont pas hashables
        if isinstance(pred_freq, dict):
            return frozenset(pred_freq.items())
        return pred_freq

    # Méthode auxiliaire de construction d'un label de fréquence lisible pour une étape
    def _stage_frequency_label(self, pred_freq: Union[str, Dict]) -> str:
        """Build a human-readable frequency label for a cascade stage.

        Used both as the ``stage_frames`` key and as the value of the
        ``frequency`` level in the multi-frequency output (see review
        §2.5): every level, including the last (target) one, keeps its
        real frequency label — no level is ever named ``'target'``. The
        target level remains identifiable via ``effective_target_frequency_``.

        Labeling rule for panel stages (``pred_freq`` a dict entity ->
        frequency): if every entity of the stage shares the same
        frequency, that shared frequency is the label; otherwise a
        composite label lists each entity's frequency, sorted by entity
        key for determinism.

        Args:
            pred_freq: Prediction frequency of the stage (str for a time
                series, dict entity -> frequency for a panel).

        Returns:
            Frequency label string.

        Examples:
            >>> imputer._stage_frequency_label('M')
            'M'
            >>> imputer._stage_frequency_label({('FR',): 'Q', ('DE',): 'Q'})
            'Q'
            >>> imputer._stage_frequency_label({('FR',): 'Q', ('DE',): 'M'})
            "('DE',)=M+('FR',)=Q"
        """
        # Cas d'une fréquence unique (séries temporelles)
        if not isinstance(pred_freq, dict):
            return str(pred_freq)

        # Cas d'un panel : fréquence partagée par toutes les entités de l'étape
        unique_freqs = {
            normalize_frequency(freq, return_format='base')
            for freq in pred_freq.values()
        }
        if len(unique_freqs) == 1:
            return unique_freqs.pop()

        # Fréquences hétérogènes : label composite, trié pour être déterministe
        parts = sorted(f"{entity}={freq}" for entity, freq in pred_freq.items())
        return "+".join(parts)

    # Méthode auxiliaire de calcul du facteur de mise à l'échelle d'une étape
    def _stage_scale_factor(
        self,
        var_key: Union[str, Tuple],
        pred_freq: Union[str, Dict],
    ) -> float:
        """Count the stage sub-periods held by one period of the variable.

        The scale factor is the number of prediction-frequency (high)
        sub-periods in one period of the variable's own detected (low)
        frequency — 12 for a yearly variable predicted monthly. The
        calendar count is exact where the duration ratio of
        ``get_conversion_factor`` would give 12.17.

        Args:
            var_key: Variable key (column name, or ``(entity..., column)``).
            pred_freq: Prediction frequency of the stage.

        Returns:
            Number of stage sub-periods per variable period.
        """
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

        return self._freq_converter.count_subperiods(
            self.detected_frequencies_[var_key],
            pf,
        )

    # -------------------------------------------------------------------------
    # Désagrégation des valeurs basse fréquence (revue §2.6)
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'appartenance des dates à leur période basse fréquence
    def _period_membership(
        self,
        index: pd.Index,
        f_var: str,
        entities: Optional[List[tuple]] = None,
    ) -> pd.Series:
        """Map each date of a grid to the low-frequency period holding it.

        The reference frequency is always the variable's own DETECTED
        frequency, never the frequency of the cascade stage: a yearly
        variable imputed at the quarter then at the month is rescaled on
        its yearly total at both stages.

        Args:
            index: Index of the stage frame — DatetimeIndex for a time
                series, MultiIndex ``(entity..., date)`` for a panel.
            f_var: Detected frequency of the variable (any representation:
                ``'Q'``, ``'QS'``, ``'quarterly'``...).
            entities: Entity tuples the disaggregation is restricted to
                (panel only). Rows of other entities get NaN and are never
                rescaled. None means every row.

        Returns:
            Series indexed like ``index``, holding a ``pd.Period`` for a
            time series and an ``(entity_tuple, Period)`` pair for a panel —
            two entities never share a rescaling block. NaN marks the rows
            excluded from the disaggregation.

        Examples:
            >>> # imputer._period_membership(monthly_index, 'Q').iloc[:3].tolist()
            >>> # [Period('2018Q1'), Period('2018Q1'), Period('2018Q1')]
        """
        # Normalisation en base de fréquence : seule forme acceptée par to_period
        base = normalize_frequency(f_var, return_format='base')

        # Cas des séries temporelles : la période suffit à identifier le bloc
        if not isinstance(index, pd.MultiIndex):
            return pd.Series(list(index.to_period(base)), index=index, dtype=object)

        # Cas des données de panel : l'entité fait partie de la clé de bloc
        periods = index.get_level_values(-1).to_period(base)
        entity_index = index.droplevel(-1)
        entity_tuples = [
            key if isinstance(key, tuple) else (key,)
            for key in entity_index
        ]
        membership = pd.Series(
            list(zip(entity_tuples, periods)), index=index, dtype=object
        )

        # Restriction aux entités du groupe : les autres lignes ne sont pas recalées
        if entities is not None:
            wanted = {
                normalize_entity_key(entity) for entity in entities
            }
            outside = [entity not in wanted for entity in entity_tuples]
            membership[outside] = np.nan

        return membership

    # Méthode auxiliaire de recalage additif des prédictions sur les totaux observés
    def _rescale_to_period_totals(
        self,
        predictions: pd.Series,
        y_original: pd.Series,
        period_membership: pd.Series,
        context: str = '',
    ) -> Tuple[pd.Series, pd.Series]:
        """Rescale predictions so each period sums to its observed total.

        Implements the additive constraint of review §2.6 (option 1): the
        sub-periods predicted for one period of a lower-frequency variable
        are multiplied by ``observed total / predicted total``, so that the
        column carries a genuine disaggregation of the observation instead
        of a free-floating prediction.

        A period is left untouched — raw predictions kept — when it is only
        partly predicted (at least one NaN sub-period, typically a period
        straddling the edge of the grid), when it holds no observation at
        all (delayed series end), or when its predictions sum to zero while
        the observed total does not. A period whose predicted total has the
        opposite sign of its observed total IS rescaled, the constraint
        taking precedence, but every sub-period then flips sign: a warning
        is emitted.

        Args:
            predictions: Predicted sub-period values, indexed like the stage
                frame rows they were produced for.
            y_original: Original column of the variable, at its own
                frequency (anchor dates non-null, the rest NaN). Read from
                the untouched input frame, never from a stage frame.
            period_membership: Output of :meth:`_period_membership`, aligned
                on the stage frame index.
            context: Label used in warning messages, e.g. "'pib' at stage M".

        Returns:
            Tuple of (rescaled predictions, boolean mask of the cells that
            were actually rescaled). The mask drives the DISAGGREGATED
            provenance marking: cells left out keep a MODEL_ON_* provenance.

        Examples:
            >>> # A quarter observed at 2800 with predictions summing to 1400
            >>> # comes back doubled, and sums to 2800 exactly.
        """
        # Totaux observés par période basse fréquence : une observation par
        # période en principe, la somme reste le choix additif si l'index en
        # porte plusieurs
        observed = y_original.dropna()
        observed_periods = period_membership.reindex(observed.index)
        period_totals: Dict[Any, float] = {}
        for value, period_key in zip(observed.to_numpy(), observed_periods.to_numpy()):
            # Observations hors du périmètre de désagrégation (NaN)
            if period_key is None or isinstance(period_key, float):
                continue
            period_totals[period_key] = period_totals.get(period_key, 0.0) + float(value)

        # Initialisation du résultat et du masque des cellules recalées
        rescaled = predictions.copy()
        rescaled_mask = pd.Series(False, index=predictions.index)

        # Regroupement positionnel des lignes prédites par période, en une seule
        # passe : un test d'égalité sur une Series de tuples aurait une sémantique
        # de diffusion ambiguë selon la longueur des clés
        membership = period_membership.reindex(predictions.index)
        rows_by_period: Dict[Any, List[int]] = {}
        for position, period_key in enumerate(membership.to_numpy()):
            # Lignes hors du périmètre de désagrégation (NaN)
            if period_key is None or isinstance(period_key, float):
                continue
            rows_by_period.setdefault(period_key, []).append(position)

        # Recensement des cas dégénérés pour un avertissement agrégé
        zero_sum_periods: List[Any] = []
        flipped_periods: List[Any] = []

        # Recalage additif : les prédictions de chaque période basse fréquence
        # sont ajustées pour que leur somme égale la valeur observée
        for period_key, period_value in period_totals.items():
            positions = rows_by_period.get(period_key)
            if not positions:
                continue
            block = predictions.iloc[positions]

            # Périodes partiellement prédites : aucune contrainte imposable
            if not block.notna().all():
                continue

            total = block.sum()

            # Somme nulle : le ratio est indéfini, prédictions brutes conservées
            if total == 0:
                if period_value != 0:
                    zero_sum_periods.append(period_key)
                continue

            # Application du ratio de recalage
            rescaled.iloc[positions] = block * (period_value / total)
            rescaled_mask.iloc[positions] = True

            # Signe opposé : la contrainte est imposée mais le profil est inversé
            if period_value != 0 and np.sign(total) != np.sign(period_value):
                flipped_periods.append(period_key)

        # Avertissements agrégés (un par variable et par étape, pas par période)
        suffix = f" for {context}" if context else ""
        if zero_sum_periods:
            warnings.warn(
                f"Additive rescaling skipped{suffix}: predictions sum to zero for "
                f"{len(zero_sum_periods)} period(s) with a non-zero observed total "
                f"(e.g. {zero_sum_periods[0]}). Raw predictions kept, their period "
                f"totals do not match the observations.",
                UserWarning
            )
        if flipped_periods:
            warnings.warn(
                f"Additive rescaling flipped the predicted profile{suffix} for "
                f"{len(flipped_periods)} period(s) (e.g. {flipped_periods[0]}): the "
                f"predictions sum to the opposite sign of the observed total. Period "
                f"totals are exact but every sub-period changed sign.",
                UserWarning
            )

        return rescaled, rescaled_mask

    # Méthode auxiliaire d'application de la contrainte additive à une étape
    def _apply_period_totals(
        self,
        values: pd.Series,
        X_input: pd.DataFrame,
        stage_group: Optional[Dict[str, Any]],
        context: str = '',
    ) -> Tuple[pd.Series, pd.Series]:
        """Enforce the additive period constraint on one stage's values.

        Single entry point of the disaggregation: builds the period
        membership from the group's source frequency then delegates the
        rescaling. A no-op returning an all-False mask when
        ``enforce_period_totals`` is False or when the group metadata is
        missing — the caller then marks the cells MODEL_ON_* as before.

        Args:
            values: Values produced for the variable at this stage (model
                predictions or interpolation fallback), indexed like the
                stage frame rows they cover.
            X_input: Untouched input frame of the replay, holding the
                variable's original low-frequency observations.
            stage_group: Entry of :attr:`stage_groups_` for the stage key.
            context: Label used in warning messages.

        Returns:
            Tuple of (values, boolean mask of the disaggregated cells).
        """
        # Sortie immédiate si la contrainte est désactivée ou non applicable
        if not self.enforce_period_totals or stage_group is None:
            return values, pd.Series(False, index=values.index)

        var_name = stage_group['var_name']
        if var_name not in X_input.columns:
            return values, pd.Series(False, index=values.index)

        # Construction de l'appartenance aux périodes de la fréquence source
        period_membership = self._period_membership(
            X_input.index, stage_group['f_var'], stage_group.get('entities')
        )

        return self._rescale_to_period_totals(
            values, X_input[var_name], period_membership, context=context
        )

    # Méthode auxiliaire de construction du masque des lignes à désagréger
    def _disaggregation_mask(
        self,
        X_input: pd.DataFrame,
        stage_group: Optional[Dict[str, Any]],
        var_name: str,
    ) -> pd.Series:
        """Build the mask of rows a variable is disaggregated on at a stage.

        Unlike the historical ``X_input[var_name].isna()``, this mask covers
        the ANCHOR dates too: the whole period is predicted and the
        low-frequency total is overwritten, which is what keeps the column
        from mixing two scales (review §2.6). For a panel the mask is
        restricted to the entities of the fitted group.

        Args:
            X_input: Untouched input frame of the replay.
            stage_group: Entry of :attr:`stage_groups_` for the stage key.
            var_name: Column being imputed.

        Returns:
            Boolean Series aligned on ``X_input.index``.
        """
        # Cas des séries temporelles : toute la grille est concernée
        entities = stage_group.get('entities') if stage_group else None
        if not self.is_panel_ or not entities:
            return pd.Series(True, index=X_input.index)

        # Cas des données de panel : restriction aux entités du groupe
        mask = np.zeros(len(X_input), dtype=bool)
        for entity in entities:
            mask |= self._freq_aligner.get_entity_mask(X_input, entity)

        return pd.Series(mask, index=X_input.index)

    # Méthode auxiliaire de récupération du modèle déjà entraîné pour une variable
    def _model_for_var(
        self,
        group_key: Union[str, Tuple],
    ) -> Optional[Dict[str, Any]]:
        """Find the model already fitted for a variable, whatever the stage.

        Backs the ``cascade_refitting=False`` regime (review §5.2): a
        variable is fitted once, at the first stage where it is imputable,
        then reused at the following stages. Fallback entries
        (``'interpolate_fallback'``, plain strings) are ignored so that a
        variable which could not be fitted at an earlier stage still gets
        its chance at a later one.

        Args:
            group_key: Registry group key — the variable name, or
                ``(variable name, detected frequency)`` for a panel where the
                frequency differs by entity (see review §2.4).

        Returns:
            The most recently registered model entry for the variable, or
            None when none was ever fitted.
        """
        # Parcours du registre sur la composante group_key de la clé d'étape
        found: Optional[Dict[str, Any]] = None
        for stage_key, model_info in self.imputation_models_.items():
            if stage_key[1] == group_key and isinstance(model_info, dict):
                found = model_info

        return found

    # Méthode auxiliaire de prédiction à l'échelle de l'étape
    def _stage_predictions(
        self,
        model_info: Dict[str, Any],
        X_features: pd.DataFrame,
    ) -> np.ndarray:
        """Predict at the scale of the stage the entry was registered for.

        The scale factor is baked into the model at fit time (``y_train``
        is divided by it), never applied at prediction time. A model reused
        across stages (``cascade_refitting=False``) therefore predicts at
        the scale of the stage it was fitted on, and its output must be
        brought back to the current stage: a yearly variable fitted at the
        quarterly stage predicts quarterly values, three times the monthly
        ones expected at the monthly stage.

        Args:
            model_info: Registry entry of the stage, holding the fitted
                model, its ``fit_scale_factor`` and the stage
                ``scale_factor``.
            X_features: Features to predict on, at the stage frequency.

        Returns:
            Predictions at the scale of the stage.
        """
        # Prédiction brute, à l'échelle de l'étape d'entraînement
        predictions = model_info['model'].predict(X_features)

        # Report de l'échelle d'entraînement vers celle de l'étape : le rapport
        # vaut 1.0 pour un modèle entraîné pour l'étape courante
        stage_scale = model_info.get('scale_factor')
        fit_scale = model_info.get('fit_scale_factor', stage_scale)
        if not stage_scale or not fit_scale or stage_scale == fit_scale:
            return predictions

        return predictions * (fit_scale / stage_scale)

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

    # Méthode auxiliaire de construction du frame d'une étape de prédiction
    def _build_stage_frame(
        self,
        X_original: pd.DataFrame,
        imputed_store: Dict[str, pd.Series],
        pred_freq: Union[str, Dict],
    ) -> pd.DataFrame:
        """Build the working frame for one prediction-frequency stage.

        The frame is always rebuilt from the ORIGINAL data (never from the
        previous stage's frame) so that aggregation artefacts do not
        accumulate across stages. Previously imputed values are injected
        before aggregation, original values always taking precedence.

        Args:
            X_original: Data as seen at fit/transform entry (after the
                additive transformer). Never modified.
            imputed_store: Mapping column name -> Series of values imputed
                at earlier stages (indexed like ``X_original``). Keyed by
                plain variable name: for a panel, the underlying model is
                global (see review §2.4) and its predictions already span
                every entity of the variable.
            pred_freq: Prediction frequency of the stage (str for time
                series, dict entity -> frequency for panel).

        Returns:
            Frame aligned on ``X_original``'s index, with higher-frequency
            columns aggregated to ``pred_freq`` (labels anchored on the
            source index position).

        Examples:
            >>> # Stage frames are rebuilt from the original data
            >>> # X_stage = imputer._build_stage_frame(X_work, {}, 'Q')
        """
        # Reconstruction depuis les données d'origine
        X_stage = X_original.copy()

        # Injection des imputations des étapes précédentes
        for col, imputed in imputed_store.items():
            # Vérification que la colonne existe dans le jeu de données
            if col not in X_stage.columns:
                continue
            # Les valeurs d'origine priment sur les valeurs imputées
            X_stage[col] = X_stage[col].combine_first(imputed)

        # Agrégation des colonnes plus fréquentes que la fréquence de l'étape
        classification = self._classify_variables_at_frequency(pred_freq)
        return self._freq_aligner.aggregate_to_target(
            X_stage, classification['aggregate'], pred_freq, self.is_panel_
        )

    # Méthode auxiliaire de préparation des données d'entraînement du modèle d'imputation
    def _prepare_training_data(
        self,
        X_stage: pd.DataFrame,
        X_original: pd.DataFrame,
        var_key: Union[str, Tuple],
        pred_freq: Union[str, Dict],
    ) -> Tuple[pd.DataFrame, pd.Series, float, pd.Series]:
        """Prepare X_train, y_train and scale factors for a variable.

        Covariates are **aggregated to the variable's own frequency**
        ``f_var`` (and not merely sampled at the variable's anchor dates):
        the model is fitted on ``f_var``-scale sums of the covariates
        against the ``f_var``-scale target. Predictions, on the other hand,
        are made on covariates carried at the stage frequency ``pred_freq``.
        The gap between both scales is precisely what ``scale_factor`` is
        meant to absorb, which is why this aggregation conditions the
        validity of the frequency scaling (see review §2.1 and §5.1).

        The variable's own column is always read from ``X_original``: a
        variable is never trained on its own imputations from earlier
        cascade stages.

        Args:
            X_stage: Stage frame built by :meth:`_build_stage_frame`
                (covariates carried at ``pred_freq``).
            X_original: Data as seen at fit entry (after the additive
                transformer), holding the variable's original values.
            var_key: Variable key to prepare training data for.
            pred_freq: Prediction frequency of the stage.

        Returns:
            Tuple of (X_train, y_train, scale_factor, feature_factors),
            where X_train holds the covariates aggregated to the variable's
            own frequency, scale_factor is the number of prediction-frequency
            sub-periods per variable period, and feature_factors is the
            per-covariate sub-period count used to bring each aggregated
            column back to the scale it carries at prediction time.
        """
        # Extraction du nom de la variable
        var_name = var_key[-1] if isinstance(var_key, tuple) else var_key

        # Détermination des colonnes de features
        feature_cols = [
            c for c in X_stage.columns
            if c != var_name
        ]

        # Si la série est univariée, renvoie des éléments vides
        if not feature_cols:
            return pd.DataFrame(), pd.Series(dtype=float), 1.0, pd.Series(dtype=float)

        # Valeurs d'origine de la variable : jamais enrichies de ses propres imputations
        y_source = X_original[var_name]

        # Masque d'entraînement : fenêtre d'entraînement + valeurs non-null pour y
        if hasattr(self, '_imputation_window_calc') and self._imputation_window_calc._is_fitted:
            # Extraction du mask d'imputation
            training_mask = self._imputation_window_calc.get_imputation_window_mask(X_stage)
            # Hybridation avec la période de disponibilité de y
            training_mask &= y_source.notna()
        else:
            training_mask = y_source.notna()

        # Restriction aux données originales si train_on_partial_coverage=False
        if not self.train_on_partial_coverage:
            if hasattr(self, '_provenance_tracker'):
                # DISAGGREGATED est admis au même titre qu'ORIGINAL : les dates-ancres
                # d'une variable désagrégée à une étape antérieure changent de
                # provenance (revue §2.6) alors que "y_source" reste lu dans les
                # données d'origine — les exclure viderait le masque à l'étape
                # suivante et enverrait tout en repli
                original_mask = self._provenance_tracker.get_mask(
                    [ProvenanceType.ORIGINAL, ProvenanceType.DISAGGREGATED],
                    column=var_name
                )
                # /!\ Vérifier la compatibilité des index
                training_mask = training_mask & original_mask

        # Agrégation des covariables à la fréquence propre de la variable
        f_var = self.detected_frequencies_[var_key]
        # Sélection des covariables strictement plus fréquentes que la variable
        feature_agg_keys = [
            key for key in self._classify_variables_at_frequency(f_var)['aggregate']
            if (key[-1] if isinstance(key, tuple) else key) != var_name
        ]
        X_features = self._freq_aligner.aggregate_to_target(
            X_stage, feature_agg_keys, f_var, self.is_panel_
        )

        # Extraction des données d'entraînement
        X_train = X_features.loc[training_mask, feature_cols]
        y_train = y_source.loc[training_mask]

        # Calcul du facteur de mise à l'échelle : nombre de sous-périodes de la
        # fréquence de prédiction (haute) dans une période de la fréquence
        # détectée de la variable (basse) — ex. variable annuelle prédite au
        # mensuel : 12
        scale_factor = self._stage_scale_factor(var_key, pred_freq)

        # Facteurs propres à chaque covariable : une colonne agrégée à f_var
        # totalise autant d'observations que sa fréquence en compte dans une
        # période de f_var, et non systématiquement scale_factor
        feature_factors = self._covariate_subperiod_counts(
            X_train.columns, f_var, scale_factor
        )

        return X_train, y_train, scale_factor, feature_factors

    # Méthode auxiliaire de comptage des sous-périodes propres à chaque covariable
    def _covariate_subperiod_counts(
        self,
        columns: pd.Index,
        f_var: str,
        default: float,
    ) -> pd.Series:
        """Count how many observations each covariate contributes per period.

        A covariate aggregated by summation to the variable's frequency
        ``f_var`` totals as many observations as its own frequency holds in
        one ``f_var`` period: 12 for a monthly covariate within a year, but
        only 4 for a quarterly one. Dividing every column by the same
        ``scale_factor`` would therefore leave the coarser covariates three
        times above the scale they carry at prediction time.

        Args:
            columns: Covariate columns of the training frame.
            f_var: Frequency of the variable being imputed.
            default: Fallback count for covariates whose frequency is
                unknown (typically the stage scale factor).

        Returns:
            Series of sub-period counts indexed by column name.
        """
        # Table nom de colonne -> fréquence détectée, tous panels confondus
        column_frequencies = {
            (key[-1] if isinstance(key, tuple) else key): freq
            for key, freq in self.detected_frequencies_.items()
        }

        # Comptage colonne par colonne, avec repli sur le facteur de l'étape
        counts = {}
        for column in columns:
            f_col = column_frequencies.get(column)
            try:
                counts[column] = self._freq_converter.count_subperiods(f_var, f_col)
            except (ValueError, TypeError):
                counts[column] = default

        return pd.Series(counts, dtype=float)

    # Méthode auxiliaire d'application du facteur de mise à l'échelle aux données
    def _apply_frequency_scaling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_factor: float,
        feature_factors: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Scale training data to the prediction-frequency scale.

        The target is always divided by the number of sub-periods so the
        model directly predicts values at the prediction frequency.
        Features are divided too when scale_features=True (i.e. when they
        were aggregated by summation to the variable's low frequency), each
        one by its own sub-period count when ``feature_factors`` is given.

        Because the model already returns sub-period values, predictions
        must never be multiplied back by the scale factor.

        Args:
            X_train: Training features (aggregated at the variable's frequency).
            y_train: Training target (at the variable's low frequency).
            scale_factor: Number of prediction-frequency sub-periods per
                variable-frequency period (e.g. 12 for yearly -> monthly).
            feature_factors: Per-covariate sub-period counts indexed by
                column name. Defaults to ``scale_factor`` for every column,
                which only holds when all covariates share the prediction
                frequency.

        Returns:
            Tuple of (scaled X_train, scaled y_train).
        """
        # Retour direct si aucune mise à l'échelle n'est nécessaire
        if scale_factor == 1.0:
            return X_train, y_train

        # La cible est TOUJOURS ramenée à l'échelle d'une sous-période
        y_scaled = y_train / scale_factor

        # Les features ne le sont que si elles ont été agrégées par somme,
        # chacune par le nombre d'observations qu'elle y a contribuées
        if not self.scale_features:
            return X_train, y_scaled
        if feature_factors is None:
            return X_train / scale_factor, y_scaled

        divisors = feature_factors.reindex(X_train.columns).fillna(scale_factor)
        return X_train / divisors, y_scaled

    # Méthode auxiliaire de détermination des échantillons de prédiction
    def _determine_prediction_samples(
        self,
        X_stage: pd.DataFrame,
        rows_mask: pd.Series,
        feature_cols: List[str],
    ) -> List[Tuple[pd.Index, List[str]]]:
        """Group the rows to predict by their pattern of available covariates.

        The rows in scope are given by ``rows_mask`` rather than derived
        here from ``isna()``: since the disaggregation of review §2.6, the
        scope covers the anchor dates too — a variable's own observed value
        no longer excludes its row from prediction. The caller therefore
        intersects the disaggregation scope with the imputation window (see
        :meth:`_prediction_masks`).

        Groups are ordered from the richest covariate pattern to the
        poorest, the empty pattern last: callers skip that final group
        instead of imputing from nothing (review §2.7).

        Args:
            X_stage: Stage frame the prediction reads its covariates from.
            rows_mask: Boolean mask, aligned on ``X_stage.index``, of the
                rows the variable may be imputed on.
            feature_cols: Columns the model was fitted on. Columns absent
                from ``X_stage`` count as unavailable everywhere.

        Returns:
            List of ``(index, available_cols)`` tuples ordered from most
            available covariates to fewest. The union of the indexes is
            exactly the rows of ``rows_mask``.

        Examples:
            >>> # samples = imputer._determine_prediction_samples(
            ... #     X_stage, rows_mask, ['covariate_a', 'covariate_b']
            ... # )
            >>> # [(Index([...]), ['covariate_a', 'covariate_b']),
            >>> #  (Index([...]), ['covariate_a']), (Index([...]), [])]
        """
        # Retour immédiat s'il n'y a aucune ligne à prédire
        rows_mask = rows_mask.reindex(X_stage.index).fillna(False).astype(bool)
        if not rows_mask.any():
            return []

        target_index = X_stage.index[rows_mask.to_numpy()]

        # Sans covariable connue, toutes les lignes partagent le motif vide
        known_cols = [c for c in feature_cols if c in X_stage.columns]
        if not known_cols:
            return [(target_index, [])]

        # Motif de disponibilité des covariables, ligne par ligne
        availability = X_stage.loc[rows_mask, known_cols].notna().to_numpy()

        # Regroupement des lignes partageant exactement le même motif
        patterns, inverse = np.unique(availability, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()

        # Tri par nombre de covariables décroissant : les groupes les plus
        # riches sont traités en premier, le motif vide en dernier
        order = np.argsort(-patterns.sum(axis=1), kind='stable')

        return [
            (
                target_index[inverse == pattern_id],
                [col for col, present in zip(known_cols, patterns[pattern_id]) if present],
            )
            for pattern_id in order
        ]

    # Méthode auxiliaire de construction des masques de désagrégation et de prédiction
    def _prediction_masks(
        self,
        X_stage: pd.DataFrame,
        X_input: pd.DataFrame,
        stage_group: Optional[Dict[str, Any]],
        var_name: str,
    ) -> Tuple[pd.Series, pd.Series]:
        """Build the disaggregation scope and the predictable rows of a stage.

        The scope is the whole grid of the group, anchor dates included
        (review §2.6): the variable is re-expressed at the stage frequency
        over all of it, so no row of the scope may keep a low-frequency
        total. The predictable rows are the scope restricted to the
        imputation window (review §2.7): nothing is ever predicted outside
        the window, where covariate coverage is insufficient by
        construction.

        Callers therefore blank the whole scope and write back only the rows
        that were actually predicted — a row of the scope left unpredicted
        ends up NaN rather than carrying its period total.

        Args:
            X_stage: Stage frame the masks are aligned on.
            X_input: Untouched input frame of the stage.
            stage_group: Entry of :attr:`stage_groups_` for the stage key.
            var_name: Column being imputed.

        Returns:
            Tuple of (disaggregation scope, predictable rows), both boolean
            Series aligned on ``X_stage.index``, the second included in the
            first.
        """
        # Périmètre de désagrégation, aligné sur le frame de l'étape
        scope = self._disaggregation_mask(X_input, stage_group, var_name)
        scope = scope.reindex(X_stage.index).fillna(False).astype(bool)

        # Restriction à la fenêtre d'imputation quand le calculateur est entraîné
        predictable = scope
        if hasattr(self, '_imputation_window_calc') and self._imputation_window_calc._is_fitted:
            window = self._imputation_window_calc.get_imputation_window_mask(X_stage)
            predictable = scope & window.reindex(X_stage.index).fillna(False).astype(bool)

        return scope, predictable

    # Méthode auxiliaire de prédiction d'une variable sur les lignes prédictibles
    def _predict_stage_values(
        self,
        model_info: Dict[str, Any],
        X_stage: pd.DataFrame,
        rows_mask: pd.Series,
        context: str = '',
    ) -> pd.Series:
        """Predict a variable on the rows it can actually be predicted on.

        Shared by the intermediate cascade of :meth:`_fit` and the replay of
        :meth:`_transform` so both paths produce identical predictions on
        identical data. Two rules, both from review §2.7:

        1. rows are grouped by their pattern of available covariates and a
           group without a single usable covariate is left unimputed —
           nothing is fabricated where nothing was observed;
        2. the remaining missing covariates are filled with the means of the
           TRAINING SET, stored at fit time in ``model_info['feature_means']``.
           Filling with the mean of the rows being predicted made the result
           depend on the prediction sample.

        Args:
            model_info: Registry entry of the stage.
            X_stage: Stage frame holding the covariates.
            rows_mask: Rows the variable may be imputed on, from
                :meth:`_prediction_masks`.
            context: Label used in warning messages.

        Returns:
            Predictions indexed by the rows actually predicted — a subset of
            ``rows_mask``, empty when no row carries any covariate.
        """
        feature_cols = list(model_info.get('feature_cols', []))
        # Statistiques de remplissage du jeu d'entraînement (§2.7). Un registre
        # produit par une version antérieure n'en porte pas : repli sur zéro,
        # neutre pour les estimateurs linéaires centrés
        feature_means = model_info.get('feature_means')
        if feature_means is None:
            feature_means = pd.Series(0.0, index=feature_cols)

        # Regroupement des lignes à prédire par motif de covariables disponibles
        prediction_samples = self._determine_prediction_samples(
            X_stage, rows_mask, feature_cols
        )

        predicted: List[pd.Series] = []
        skipped: List[pd.Index] = []

        # Parcours des groupes, du plus riche en covariables au plus pauvre
        for sample_index, available_cols in prediction_samples:
            # Restriction aux features connues du modèle
            usable = [c for c in feature_cols if c in available_cols]
            # Aucune covariable observée : la date n'est pas imputée
            if not usable:
                skipped.append(sample_index)
                continue

            # Le motif décide SI l'on impute, les statistiques du fit décident
            # AVEC QUOI l'on complète les covariables manquantes du groupe
            X_pred = X_stage.loc[sample_index, feature_cols].fillna(feature_means)
            predicted.append(
                pd.Series(
                    self._stage_predictions(model_info, X_pred),
                    index=sample_index,
                )
            )

        # Avertissement agrégé sur les dates laissées non imputées
        if skipped:
            skipped_index = skipped[0].append(skipped[1:]) if len(skipped) > 1 else skipped[0]
            suffix = f" for {context}" if context else ""
            warnings.warn(
                f"No covariate available{suffix} on {len(skipped_index)} date(s) "
                f"(e.g. {skipped_index[0]}): they are left unimputed rather than "
                f"predicted from training means alone.",
                UserWarning
            )

        # Concaténation des groupes prédits, réordonnée comme le frame d'étape.
        # L'index vide conserve le type de celui du frame : les appelants y
        # indexent directement, un RangeIndex vide y lèverait un KeyError
        if not predicted:
            return pd.Series(dtype=float, index=X_stage.index[:0])

        values = pd.concat(predicted)
        ordered_index = X_stage.index[X_stage.index.isin(values.index)]

        return values.reindex(ordered_index)

    # Méthode auxiliaire de notation de la provenance des variables agrégées
    def _mark_aggregated_provenance(
        self,
        tracker: ImputationProvenanceTracker,
        X: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]]
    ) -> None:
        """Mark aggregated values in the provenance tracker.

        Only the cells that actually CARRY an aggregate are marked: an
        aggregation to a lower frequency leaves the sub-period rows empty,
        and marking them too (review §2.8.2) would claim NaN cells were
        aggregated. Cells already marked ORIGINAL are excluded too: since
        aggregation runs at every stage, a column that already held a true
        observation at this index would otherwise have its ORIGINAL
        provenance overwritten by AGGREGATED, even though nothing was
        actually aggregated to obtain it (review §2.8.2).

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
                    # Marquage comme agrégé des seules cellules porteuses d'un
                    # agrégat, à l'exclusion de celles déjà ORIGINAL
                    if col in X.columns:
                        candidate = entity_index[X.loc[entity_index, col].notna()]
                        original_mask = tracker.get_mask(
                            ProvenanceType.ORIGINAL, column=col
                        ).reindex(candidate).fillna(False)
                        marked = candidate[~original_mask]
                        if len(marked) > 0:
                            tracker.mark_aggregated(col, marked)
        # Cas de données de séries temporelles
        else:
            # Extraction des noms de colonne des tuples
            columns = self._freq_aligner.extract_column_names(aggregate_keys)
            # Parcours des colonnes
            for col in columns:
                # Marquage comme agrégé des seules cellules porteuses d'un
                # agrégat, à l'exclusion de celles déjà ORIGINAL
                if col in X.columns:
                    candidate = X.index[X[col].notna()]
                    original_mask = tracker.get_mask(
                        ProvenanceType.ORIGINAL, column=col
                    ).reindex(candidate).fillna(False)
                    marked = candidate[~original_mask]
                    if len(marked) > 0:
                        tracker.mark_aggregated(col, marked)

    # Méthode auxiliaire de marquage des cellules imputées à une étape
    @staticmethod
    def _mark_imputed_cells(
        tracker: ImputationProvenanceTracker,
        column: str,
        index: pd.Index,
        disaggregated_mask: pd.Series,
        trained_on_imputed: bool,
    ) -> None:
        """Mark imputed cells, splitting disaggregated ones from model ones.

        A cell belonging to a period whose sub-periods were rescaled onto an
        observed total is DISAGGREGATED — it carries a real observation,
        spread out. Any other imputed cell keeps its MODEL_ON_* provenance.

        Args:
            tracker: Provenance tracker to write into.
            column: Column being imputed.
            index: Index of every cell that received a value.
            disaggregated_mask: Boolean mask, aligned on ``index``, of the
                cells produced by a rescaled disaggregation.
            trained_on_imputed: Whether the model saw imputed values at fit
                time (drives MODEL_ON_MIXED vs MODEL_ON_TRUE).
        """
        # Cellules recalées sur un total observé
        disaggregated_index = index[disaggregated_mask.to_numpy()]
        if len(disaggregated_index) > 0:
            tracker.mark_disaggregated(column, disaggregated_index)

        # Cellules restantes : prédictions du modèle sans contrainte additive
        model_index = index[~disaggregated_mask.to_numpy()]
        if len(model_index) > 0:
            tracker.mark_model_imputed(
                column, model_index, trained_on_imputed=trained_on_imputed
            )

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

        Each cascade stage works on a frame rebuilt from the entry data by
        :meth:`_build_stage_frame` (never from the previous stage's frame),
        so that aggregation artefacts do not accumulate. Intermediate
        predictions feed ``imputed_store``, which enriches the covariates of
        the following stages; ``X_work`` is never modified.

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

        # Label de fréquence de l'index d'entrée : il identifie, à l'inversion,
        # le niveau à conserver dans une sortie multi-fréquences (§2.10). Le
        # passage par "_stage_frequency_label" garantit le même format de label
        # que celui porté par le niveau "frequency" de la sortie
        try:
            index_freq = detect_index_frequency(X_work.index, return_format='base')
            self._source_frequency_label_ = self._stage_frequency_label(index_freq)
        except (ValueError, TypeError):
            # Index irrégulier ou trop court : le repli sur le niveau cible suffit
            self._source_frequency_label_ = None

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
        # Métadonnées de groupe par clé d'étape : le replay en a besoin pour
        # désagréger (fréquence source, entités concernées), y compris sur les
        # replis dont l'entrée de registre est une simple chaîne
        self.stage_groups_ = OrderedDict()
        # Conservation de la liste des étapes : le replay de "_transform" la rejoue à l'identique
        self.freq_prediction_list_ = freq_prediction_list
        # Imputations déjà réalisées, par nom de variable : elles alimentent les
        # covariables des étapes suivantes, jamais X_work
        imputed_store: Dict[str, pd.Series] = {}

        for pred_freq in freq_prediction_list:
            # 5a. Classification des variables relative à pred_freq
            var_classification = self._classify_variables_at_frequency(pred_freq)
            aggregate_keys = var_classification['aggregate']
            impute_keys = var_classification['impute']

            # 5b. Construction du frame de l'étape, reconstruit depuis les données d'origine
            X_stage = self._build_stage_frame(X_work, imputed_store, pred_freq)
            # Marquage de la provenance sur le frame d'étape (index d'origine)
            self._mark_aggregated_provenance(
                self._provenance_tracker, X_stage, aggregate_keys
            )

            # 5c. Ordonnancement des variables à imputer
            if self.train_on_partial_fit_order == 'cv' and self.train_on_partial_coverage:
                ordered_impute_keys = self._determine_variable_order_cv(X_stage, impute_keys)
            else:
                # Tri par fréquence (plus basse d'abord)
                ordered_impute_keys = sorted(
                    impute_keys,
                    key=lambda k: get_frequency_order(
                        self.detected_frequencies_.get(k, 'D')
                    ),
                    reverse=True,
                )

            # 5d. Regroupement des clés d'imputation par variable au sein de l'étape :
            # pour un panel, "ordered_impute_keys" contient une clé par couple
            # (entité, variable), mais le modèle entraîné reste GLOBAL au panel
            # (conforme à la spécification "l'entraînement se fait sur l'ensemble des
            # entités") — itérer par entité ne ferait donc que refitter et réécraser
            # exactement le même modèle plusieurs fois. Le regroupement se fait sur
            # (variable, fréquence détectée) et non la seule variable : le facteur
            # d'échelle et l'agrégation d'entraînement dépendent de la fréquence, qui
            # peut différer selon l'entité pour une même variable.
            # Piste inverse, non retenue ici (de vrais modèles par entité) :
            # restreindre "training_mask" et "missing_mask" à l'entité via
            # "self._freq_aligner.get_entity_mask".
            vars_in_stage: "OrderedDict[Tuple[str, str], List[Union[str, Tuple]]]" = OrderedDict()
            for var_key in ordered_impute_keys:
                var_name = var_key[-1] if isinstance(var_key, tuple) else var_key
                freq_norm = normalize_frequency(
                    self.detected_frequencies_[var_key], return_format='base'
                )
                vars_in_stage.setdefault((var_name, freq_norm), []).append(var_key)

            # La fréquence ne rejoint la clé de groupe que si elle diffère
            # effectivement selon l'entité pour une même variable à cette étape
            freqs_per_var: Dict[str, set] = {}
            for var_name, freq_norm in vars_in_stage:
                freqs_per_var.setdefault(var_name, set()).add(freq_norm)

            # 5d'. Un seul fit par groupe (variable[, fréquence]), sur l'ensemble
            # des entités du groupe
            for (var_name, freq_norm), var_keys in vars_in_stage.items():
                group_key = (
                    var_name if len(freqs_per_var[var_name]) == 1
                    else (var_name, freq_norm)
                )
                # Clé d'origine représentative du groupe (une des clés (entité,
                # variable) qui le composent) : seuls le nom de colonne et la
                # fréquence détectée comptent pour préparer les données
                # d'entraînement, le modèle étant global au panel
                repr_var_key = var_keys[0]
                # Clé d'étape : un même couple (étape, groupe) ne peut être
                # entraîné qu'une fois, et une variable imputée à deux étapes
                # obtient deux entrées distinctes
                stage_key = (self._freq_label(pred_freq), group_key)

                # Métadonnées du groupe, enregistrées avant tout branchement :
                # le replay en a besoin même quand l'étape finit en repli
                self.stage_groups_[stage_key] = {
                    'var_name': var_name,
                    'f_var': freq_norm,
                    'entities': (
                        [normalize_entity_key(k[:-1]) for k in var_keys]
                        if self.is_panel_ and isinstance(var_keys[0], tuple)
                        else None
                    ),
                }
                stage_group = self.stage_groups_[stage_key]

                # Sans réentraînement, un seul fit par variable : les étapes
                # suivantes réutilisent le modèle avec le facteur de l'étape
                if not self.cascade_refitting:
                    base = self._model_for_var(group_key)
                    if base is not None:
                        self.imputation_models_[stage_key] = {
                            **base,
                            'scale_factor': self._stage_scale_factor(repr_var_key, pred_freq),
                            'pred_freq': pred_freq,
                        }
                        self.model_fitting_order_.append(stage_key)
                        continue

                estimator = self._get_estimator_for_variable(var_name)
                if estimator is None:
                    warnings.warn(
                        f"No estimator available for variable '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    self.imputation_models_[stage_key] = 'interpolate_fallback'
                    self.model_fitting_order_.append(stage_key)
                    continue

                # Préparation des données d'entraînement
                X_train, y_train, scale_factor, feature_factors = self._prepare_training_data(
                    X_stage, X_work, repr_var_key, pred_freq
                )

                if len(X_train) < 2:
                    warnings.warn(
                        f"Not enough training data for variable '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    self.imputation_models_[stage_key] = 'interpolate_fallback'
                    self.model_fitting_order_.append(stage_key)
                    continue

                # 5e. Frequency scaling
                X_train_scaled, y_train_scaled = self._apply_frequency_scaling(
                    X_train, y_train, scale_factor, feature_factors
                )

                # Imputation simple des NaN dans les features. Les statistiques
                # employées sont celles du JEU D'ENTRAÎNEMENT et sont conservées
                # dans le registre : la prédiction les réapplique à l'identique
                # (§2.7), là où la moyenne des lignes à prédire rendait le
                # résultat dépendant du champ de prédiction
                feature_means = X_train_scaled.mean()
                X_train_scaled = X_train_scaled.fillna(feature_means)
                # Écartement des covariables sans aucune observation sur la fenêtre
                # d'entraînement : la moyenne de remplissage y reste NaN
                X_train_scaled = X_train_scaled.dropna(axis=1, how='all')

                # Suppression des lignes avec y NaN
                valid_mask = y_train_scaled.notna()
                X_train_scaled = X_train_scaled.loc[valid_mask]
                y_train_scaled = y_train_scaled.loc[valid_mask]

                # Repli si le jeu d'entraînement est trop petit ou sans covariable exploitable
                if len(X_train_scaled) < 2 or X_train_scaled.shape[1] == 0:
                    self.imputation_models_[stage_key] = 'interpolate_fallback'
                    self.model_fitting_order_.append(stage_key)
                    continue

                # 5f. Fit du modèle + stockage metadata
                feature_cols = list(X_train_scaled.columns)
                try:
                    estimator.fit(X_train_scaled, y_train_scaled)
                    self.imputation_models_[stage_key] = {
                        'model': estimator,
                        'feature_cols': feature_cols,
                        # Moyennes du jeu d'entraînement, réappliquées à la
                        # prédiction pour compléter les covariables manquantes
                        'feature_means': feature_means.reindex(feature_cols),
                        'scale_factor': scale_factor,
                        # Facteur cuit dans le modèle : y_train en a été divisé.
                        # Il ne bouge plus, quand "scale_factor" suit l'étape
                        'fit_scale_factor': scale_factor,
                        'pred_freq': pred_freq,
                        'trained_on_imputed': (
                            self.train_on_partial_coverage
                            and bool(imputed_store)
                        ),
                    }
                except Exception as e:
                    warnings.warn(
                        f"Failed to fit model for variable '{var_name}': {e}. "
                        f"Using linear interpolation as fallback"
                    )
                    self.imputation_models_[stage_key] = 'interpolate_fallback'

                self.model_fitting_order_.append(stage_key)

                # 5g. Cascade refitting : application des prédictions intermédiaires,
                # sur l'ensemble des entités du groupe en une seule passe (le modèle
                # est global au panel, cf. 5d)
                if self.cascade_refitting:
                    model_info = self.imputation_models_.get(stage_key)
                    if model_info and model_info != 'interpolate_fallback':
                        stage_context = (
                            f"'{var_name}' at stage "
                            f"{self._stage_frequency_label(pred_freq)}"
                        )
                        # Périmètre de désagrégation (toute la grille du groupe,
                        # dates-ancres comprises, §2.6) et lignes prédictibles
                        # (restreintes à la fenêtre d'imputation, §2.7)
                        scope_mask, predict_mask = self._prediction_masks(
                            X_stage, X_work, stage_group, var_name
                        )
                        if scope_mask.any():
                            try:
                                # Les prédictions sont déjà à l'échelle de la fréquence
                                # de prédiction : aucune remise à l'échelle inverse
                                preds = self._predict_stage_values(
                                    model_info, X_stage, predict_mask,
                                    context=stage_context
                                )
                                # Contrainte additive : la somme des sous-périodes de
                                # chaque période égale la valeur observée
                                preds, disagg_mask = self._apply_period_totals(
                                    preds, X_work, stage_group,
                                    context=stage_context
                                )
                                # Cascade intra-étape : les variables suivantes de l'étape
                                # voient les valeurs imputées. Le périmètre de
                                # désagrégation est d'abord vidé en entier : une ancre
                                # laissée non prédite ne doit pas conserver le total de
                                # sa période dans une colonne à l'échelle de la
                                # sous-période (§2.6)
                                X_stage.loc[scope_mask, var_name] = np.nan
                                X_stage.loc[preds.index, var_name] = preds
                                # Alimentation du magasin d'imputations pour les étapes
                                # suivantes (jamais X_work), par nom de variable : les
                                # imputations déjà déposées par un autre groupe de la
                                # même variable (fréquences hétérogènes selon l'entité)
                                # sont préservées
                                existing_imputed = imputed_store.get(var_name)
                                imputed_store[var_name] = (
                                    preds if existing_imputed is None
                                    else existing_imputed.combine_first(preds)
                                )

                                # Marquage provenance : DISAGGREGATED sur les cellules
                                # effectivement recalées, MODEL_ON_* sur les autres.
                                # "preds.index" couvre toutes les entités du groupe, ce
                                # qui est cohérent avec le modèle GLOBAL au panel (§2.4) :
                                # marquer par "var_name" seul, sans restriction d'entité,
                                # ne perd donc aucune information (review §2.8.3)
                                self._mark_imputed_cells(
                                    self._provenance_tracker, var_name, preds.index,
                                    disagg_mask, model_info.get('trained_on_imputed', False)
                                )
                            except Exception as e:
                                warnings.warn(
                                    f"Intermediate imputation failed for '{var_name}': {e}"
                                )

        # =================================================================
        # PHASE 6 — Finalisation
        # =================================================================
        self.frequency_progression_ = self._compute_frequency_progression()

        if self.impute_delayed_values:
            warnings.warn(
                "impute_delayed_values is deprecated and has no effect: publication "
                "delays are now handled by imputation_scope='extended_forward' "
                "combined with coverage_threshold, which imputes delayed series ends "
                "directly from the extended training window (review §2.9).",
                DeprecationWarning
            )

        # Attribut distinct de celui écrit par "_transform" (review §2.8.1) :
        # sans cela, un "fit_transform" perdrait la trace du fit, écrasée par
        # celle du transform qui suit immédiatement
        self.imputation_provenance_fit_ = self._provenance_tracker.get_provenance_matrix()

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------
    def _transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Transform X and optionally y using cascade imputation.

        Replays the cascade stages recorded at fit time
        (``freq_prediction_list_``) and, within each stage, the models of
        ``model_fitting_order_``. Each stage frame is rebuilt from the input
        data by :meth:`_build_stage_frame` — the very method used by
        :meth:`_fit` — so fit and transform work on identical stage frames
        for identical data. The input frame itself is never modified.

        Note:
            This method is stateful: it snapshots its input in
            ``_original_X_`` / ``_original_y_`` and rewrites
            ``imputation_provenance_``, both overwritten at each call.
            :meth:`_inverse_transform` reads them back to restore the source
            index and the original values, and therefore always inverts the
            LAST transform.

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

        # Mémorisation de l'entrée du DERNIER transform : la transformation
        # inverse y lit l'index source, ses noms de niveaux et, sur demande
        # (restore_original_values), les valeurs d'origine exactes (§2.10).
        # Cet instantané est donc écrasé à chaque appel de transform
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

        # 3. Rejouer les étapes du fit dans le même ordre, avec les mêmes frames d'étape
        # Frame d'entrée du replay : jamais modifié, il sert de base à chaque étape
        X_input = data_transformed
        # Imputations déjà réalisées, alimentant les covariables des étapes suivantes
        imputed_store: Dict[str, pd.Series] = {}
        # Frames d'étape APRÈS imputations, par label de fréquence réel (§2.5) :
        # alimenté après la boucle d'imputation de chaque étape, jamais avant
        stage_frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        # Instantané de la matrice de provenance par label de fréquence réel,
        # capturé au même moment que "stage_frames" : "transform_tracker" est
        # unique et continue d'être écrit aux étapes suivantes, donc sans cet
        # instantané la provenance d'un niveau bas serait écrasée par celle
        # d'une étape ultérieure (review §2.8.4)
        provenance_frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()

        for stage_idx, pred_freq in enumerate(self.freq_prediction_list_):
            # Classification des variables relative à pred_freq
            var_classification = self._classify_variables_at_frequency(pred_freq)
            aggregate_keys = var_classification['aggregate']

            # Construction du frame d'étape : même méthode qu'au fit, mêmes frames
            X_stage = self._build_stage_frame(X_input, imputed_store, pred_freq)
            self._mark_aggregated_provenance(
                transform_tracker, X_stage, aggregate_keys
            )

            # Extraction des étapes d'entraînement associées à cette fréquence
            registry_label = self._freq_label(pred_freq)
            stage_keys = [
                key for key in self.model_fitting_order_
                if key[0] == registry_label
            ]

            # Label de fréquence réel de l'étape (jamais 'target', cf. §2.5)
            freq_label = self._stage_frequency_label(pred_freq)
            is_final_stage = stage_idx == len(self.freq_prediction_list_) - 1

            # Parcours des variables imputées à cette étape
            for stage_key in stage_keys:
                # "group_key" est la variable seule, ou (variable, fréquence
                # détectée) pour un panel hétérogène en fréquence (cf. §2.4) : la
                # variable est toujours la première composante, contrairement aux
                # anciennes clés (entité..., variable). Ne pas réutiliser le nom
                # "freq_label" ici : il porte le label lisible de l'étape,
                # utilisé après la boucle pour "stage_frames"
                _, group_key = stage_key
                var_name = group_key[0] if isinstance(group_key, tuple) else group_key

                # Récupérer model_info et les métadonnées de groupe (présentes
                # même pour les replis, contrairement au registre de modèles)
                model_info = self.imputation_models_.get(stage_key)
                stage_group = (getattr(self, 'stage_groups_', None) or {}).get(stage_key)
                stage_context = f"'{var_name}' at stage {freq_label}"

                if model_info is None or model_info == 'interpolate_fallback':
                    # Le repli n'alimente pas imputed_store : le fit n'en produit aucune
                    # valeur, les frames d'étape resteraient asymétriques
                    if var_name in X_stage.columns:
                        # L'ancre sert de point d'appui à l'interpolation, puis le
                        # recalage proportionnel la ramène à l'échelle de la
                        # sous-période : sans cela la colonne repliée garderait le
                        # mélange d'échelles du §2.6
                        interpolated = X_stage[var_name].interpolate(
                            method='linear', limit_direction='both'
                        )
                        # Même restriction que le chemin modèle : rien n'est
                        # produit hors de la fenêtre d'imputation (§2.7)
                        scope_mask, predict_mask = self._prediction_masks(
                            X_stage, X_input, stage_group, var_name
                        )
                        rescaled, disagg_mask = self._apply_period_totals(
                            interpolated.loc[predict_mask], X_input,
                            stage_group, context=stage_context
                        )
                        X_stage.loc[scope_mask, var_name] = np.nan
                        X_stage.loc[rescaled.index, var_name] = rescaled
                        self._mark_imputed_cells(
                            transform_tracker, var_name, rescaled.index,
                            disagg_mask, False
                        )
                    continue

                # Identification des valeurs manquantes
                if var_name not in X_stage.columns:
                    continue

                # Périmètre de désagrégation (toute la grille du groupe, dates-ancres
                # comprises, pour que la colonne ne mélange pas le total de la période
                # basse fréquence et des valeurs de sous-période, §2.6) et lignes
                # prédictibles (restreintes à la fenêtre d'imputation, §2.7)
                scope_mask, predict_mask = self._prediction_masks(
                    X_stage, X_input, stage_group, var_name
                )
                if not scope_mask.any():
                    continue

                try:
                    # Les prédictions sont déjà à l'échelle de la fréquence de
                    # prédiction : aucune remise à l'échelle inverse
                    predictions = self._predict_stage_values(
                        model_info, X_stage, predict_mask, context=stage_context
                    )
                    # Contrainte additive : la somme des sous-périodes de chaque
                    # période égale la valeur observée
                    predictions, disagg_mask = self._apply_period_totals(
                        predictions, X_input, stage_group, context=stage_context
                    )
                    # Cascade intra-étape : le frame de l'étape porte le résultat.
                    # Le périmètre de désagrégation est d'abord vidé en entier, une
                    # ancre laissée non prédite ne devant pas conserver le total de
                    # sa période dans une colonne à l'échelle de la sous-période (§2.6)
                    X_stage.loc[scope_mask, var_name] = np.nan
                    X_stage.loc[predictions.index, var_name] = predictions
                    # Alimentation des étapes suivantes, symétrique du bloc 5g du
                    # fit : sans réentraînement, les covariables des étapes
                    # suivantes restent celles vues à l'entraînement. Clé par nom
                    # de variable : les imputations déjà déposées par un autre
                    # groupe de la même variable (fréquences hétérogènes selon
                    # l'entité) sont préservées
                    if self.cascade_refitting:
                        existing_imputed = imputed_store.get(var_name)
                        imputed_store[var_name] = (
                            predictions if existing_imputed is None
                            else existing_imputed.combine_first(predictions)
                        )

                    # Marquage provenance : DISAGGREGATED sur les cellules
                    # effectivement recalées, MODEL_ON_* sur les autres. Même
                    # remarque qu'au fit (§2.8.3) : "predictions.index" couvre
                    # toutes les entités du groupe, cohérent avec le modèle GLOBAL
                    self._mark_imputed_cells(
                        transform_tracker, var_name, predictions.index,
                        disagg_mask, model_info.get('trained_on_imputed', False)
                    )
                except Exception as e:
                    warnings.warn(
                        f"Prediction failed for variable '{var_name}': {e}. "
                        f"Using interpolation fallback."
                    )
                    X_stage[var_name] = X_stage[var_name].interpolate(
                        method='linear', limit_direction='both'
                    )

            # Stockage du frame d'étape APRÈS les imputations de l'étape (§2.5) :
            # un niveau par étape ayant produit au moins un modèle, plus la
            # dernière étape (fréquence cible) même sans imputation propre,
            # nécessaire à la sortie sans MultiIndex ci-dessous
            if stage_keys or is_final_stage:
                stage_frames[freq_label] = X_stage.copy()
                # Même instantané pour la provenance, capturé au même instant
                # que le frame d'étape (§2.8.4)
                provenance_frames[freq_label] = transform_tracker.get_provenance_matrix()

            # Le frame de la dernière étape porte le résultat à la fréquence cible
            data_transformed = X_stage
            final_stage_label = freq_label

        # 4. Construire sortie MultiIndex si keep_lower_frequencies
        if self.keep_lower_frequencies and stage_frames:
            data_result = self._build_multifreq_output(stage_frames)
        else:
            data_result = stage_frames[final_stage_label]

        # 5. Mise à jour de la provenance : une matrice par niveau de
        # fréquence, empilée avec la même structure d'index que
        # "data_result" (§2.8.4), quand keep_lower_frequencies=True ; sinon
        # la seule matrice au niveau cible, comme avant
        if self.keep_lower_frequencies and provenance_frames:
            self.imputation_provenance_ = self._build_multifreq_output(provenance_frames)
        else:
            self.imputation_provenance_ = transform_tracker.get_provenance_matrix()

        # 6. Scission X et y
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
        stage_frames: "OrderedDict[str, pd.DataFrame]",
    ) -> pd.DataFrame:
        """Stack every useful cascade stage into one multi-frequency frame.

        Backs ``keep_lower_frequencies=True`` (review §2.5). Each stage
        frame is expected to already carry the imputations performed at
        that stage — callers must populate ``stage_frames`` *after* the
        imputation loop of a stage, never right after aggregation, or the
        corresponding level would miss that stage's imputed values.

        Args:
            stage_frames: Stage frames keyed by real frequency label (see
                :meth:`_stage_frequency_label`), one entry per stage that
                produced at least one imputation model, plus the final
                (target-frequency) stage. Insertion order fixes the
                stacking order and therefore the level order in the
                output index.

        Returns:
            DataFrame with a MultiIndex:
            - Time series: ``(frequency, date)``
            - Panel: ``(entity, frequency, date)``
            Every level — including the target-frequency one — keeps its
            real frequency label; no level is named ``'target'``. The
            target level is identified via ``effective_target_frequency_``,
            not via a dedicated label.
        """
        all_frames = []

        for freq_label, df in stage_frames.items():
            df_copy = df.copy()
            df_copy['_frequency_level'] = freq_label
            all_frames.append(df_copy)

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

    # -------------------------------------------------------------------------
    # Transformation inverse
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de sélection du niveau de fréquence à inverser
    def _select_inverse_frequency_level(
        self,
        data: pd.DataFrame,
        provenance: pd.DataFrame,
    ) -> Optional[str]:
        """Pick the frequency level to keep when inverting a stacked output.

        The level of the source index (``_source_frequency_label_``) is the
        one to keep: it is where the values sit at the granularity of the
        data given to ``fit`` (review §2.10). It may be missing when the
        index frequency could not be detected, or when it never was a
        cascade stage; the target level is then the best proxy, being the
        only level always produced.

        Args:
            data: Frame to invert, possibly carrying a ``frequency`` level.
            provenance: Provenance matrix of the last transform, possibly
                carrying a ``frequency`` level too — both are stacked or
                not depending on ``keep_lower_frequencies``.

        Returns:
            The frequency label to keep, or None when neither frame carries
            a frequency level (nothing to select).
        """
        # Recensement des labels disponibles, côté données comme côté provenance
        available = []
        for frame in (data, provenance):
            if self._has_frequency_level(frame):
                available.extend(
                    frame.index.get_level_values('frequency').unique().tolist()
                )
        if not available:
            return None

        # Priorité au niveau de l'index source
        if self._source_frequency_label_ is not None:
            if self._source_frequency_label_ in available:
                return self._source_frequency_label_

        # Repli sur le niveau cible, toujours produit par la cascade
        target_label = self._stage_frequency_label(self.effective_target_frequency_)
        if target_label in available:
            return target_label

        # Dernier repli : le dernier niveau empilé, avec avertissement
        warnings.warn(
            f"Neither the source frequency level "
            f"({self._source_frequency_label_}) nor the target one "
            f"({target_label}) is present in the data to invert. "
            f"Falling back on the last stacked level '{available[-1]}'."
        )
        return available[-1]

    # Méthode auxiliaire de vérification de la présence du niveau de fréquence
    @staticmethod
    def _has_frequency_level(frame: pd.DataFrame) -> bool:
        """Tell whether a frame carries the multi-frequency ``frequency`` level.

        Args:
            frame: Frame to inspect.

        Returns:
            True if its index is a MultiIndex holding a ``frequency`` level.
        """
        return (
            isinstance(frame.index, pd.MultiIndex)
            and 'frequency' in (frame.index.names or [])
        )

    # Méthode auxiliaire de suppression du niveau de fréquence
    def _drop_frequency_level(
        self,
        frame: pd.DataFrame,
        label: Optional[str],
    ) -> pd.DataFrame:
        """Reduce a stacked frame to one frequency level and restore its index.

        Panel and time series are handled the same way: the level is
        addressed by NAME, never by position. The index names of the
        remaining levels are restored from the last transform input, the
        stacking having renamed them (``['entity', 'frequency', 'date']``).

        Args:
            frame: Frame to reduce, stacked or not.
            label: Frequency label to keep (see
                :meth:`_select_inverse_frequency_level`). None leaves the
                frame untouched.

        Returns:
            The frame restricted to ``label``, without the frequency level,
            or the frame itself when it carries no such level.
        """
        # Frame déjà à un seul niveau de fréquence
        if label is None or not self._has_frequency_level(frame):
            return frame

        # Extraction du niveau demandé (absent du frame : rien à extraire)
        if label not in frame.index.get_level_values('frequency'):
            return frame
        reduced = frame.xs(label, level='frequency')

        # Restauration des noms de niveaux de l'index source, l'empilement
        # les ayant remplacés par ('entity', 'frequency', 'date')
        source = getattr(self, '_original_X_', None)
        if source is not None:
            source_names = list(source.index.names)
            if len(source_names) == reduced.index.nlevels:
                reduced = reduced.rename_axis(source_names)

        return reduced

    # Méthode auxiliaire de restauration des valeurs d'origine exactes
    def _restore_original_values(
        self,
        data_result: pd.DataFrame,
        y_col_name: Optional[str],
    ) -> pd.DataFrame:
        """Refill the cells observed in the last transform input.

        Backs ``restore_original_values=True`` (review §2.10): the ORIGINAL
        mask alone drops the anchor dates of a lower-frequency variable,
        which the target level holds as DISAGGREGATED even though the input
        carried a true observation there.

        Args:
            data_result: Frame restored from the provenance mask, at the
                source index.
            y_col_name: Column name given to y in the working frame, so
                that the snapshot of y is realigned on it.

        Returns:
            The frame with every cell observed in the snapshot set back to
            its original value.
        """
        # Reconstruction de l'instantané de l'entrée du dernier transform
        snapshot = self._original_X_
        if self._original_y_ is not None and y_col_name is not None:
            snapshot = pd.concat(
                [snapshot, self._original_y_.to_frame(name=y_col_name)], axis=1
            )

        # Restriction aux colonnes et à l'index communs : la sortie inverse
        # peut porter d'autres colonnes (variables ajoutées) ou un index réduit
        common_cols = [c for c in data_result.columns if c in snapshot.columns]
        if not common_cols:
            return data_result
        aligned = snapshot[common_cols].reindex(index=data_result.index)

        # Les valeurs observées priment sur celles restaurées par la provenance
        data_result = data_result.copy()
        data_result[common_cols] = aligned.combine_first(data_result[common_cols])
        return data_result

    # Méthode auxiliaire de transformation inverse
    def _inverse_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Restore the original data structure from an imputed dataset.

        Mirror of :meth:`_transform`, driven by the provenance matrix of the
        LAST transform (``imputation_provenance_``) rather than by the fit
        one: an ``inverse_transform`` following a ``transform`` on new data
        must undo what that very call produced (review §2.10). Steps:

        1. If the input carries a frequency level
           (``keep_lower_frequencies=True``), keep only the level matching
           the frequency of the source index and restore its index names.
        2. Set back to NaN every cell whose provenance is not ORIGINAL.
        3. Apply the additive transformer inverse LAST, mirroring
           ``transform``, which applies it first.

        Note:
            Step 2 drops the anchor dates of the lower-frequency variables:
            the target level spreads them over their period, so their
            provenance there is DISAGGREGATED — they are readable as
            ORIGINAL on their own frequency level, or restorable exactly
            with ``restore_original_values=True``. Disaggregation from a
            lower to a higher frequency stays lossy in any case: the
            sub-period values themselves cannot be recovered.

        Args:
            X: Transformed features, as returned by ``transform``.
            y: Transformed targets (optional).

        Returns:
            X_original if y is None.
            (X_original, y_original) if y is provided.

        Raises:
            ValueError: If ``transform`` was never called: the provenance
                matrix it writes is what identifies the imputed cells.

        Examples:
            >>> imputer = HighFrequencyImputer(target_frequency='M')
            >>> transformed = imputer.fit_transform(df)
            >>> restored = imputer.inverse_transform(transformed)
            >>> restored.index.equals(df.index)
            True
            >>> # Imputed cells are back to NaN, observed ones are unchanged
            >>> restored['pib_trimestriel'].isna().sum() >= df['pib_trimestriel'].isna().sum()
            True
        """
        # 0. Garde : la provenance du dernier transform est indispensable
        if not hasattr(self, 'imputation_provenance_'):
            raise ValueError(
                "inverse_transform requires a previous call to transform: the "
                "provenance matrix of the last transform (imputation_provenance_) "
                "identifies the cells to set back to NaN. Call transform(X) or "
                "fit_transform(X) first."
            )

        # Concaténation X / y, symétrique de "_transform"
        y_col_name = None
        if y is not None:
            y_col_name = y.name if y.name is not None else '__target__'
            data_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            data_work = X.copy()

        provenance = self.imputation_provenance_

        # 1. Suppression du niveau de fréquence éventuel, sur les données ET
        # sur la provenance : chacune peut le porter ou non, selon la valeur
        # de keep_lower_frequencies au dernier transform
        level_label = self._select_inverse_frequency_level(data_work, provenance)
        data_work = self._drop_frequency_level(data_work, level_label)
        provenance = self._drop_frequency_level(provenance, level_label)

        # 2. Restauration des NaN pour toute cellule non originale
        original_mask = (provenance == ProvenanceType.ORIGINAL).reindex(
            index=data_work.index, columns=data_work.columns
        )
        # Colonnes hors périmètre de la provenance (identifiants de panel) :
        # jamais masquées, elles ne portent aucune valeur imputée
        untracked = [c for c in data_work.columns if c not in provenance.columns]
        if untracked:
            original_mask[untracked] = True
        original_mask = original_mask.fillna(False).astype(bool)
        data_work = data_work.where(original_mask)

        # 3. Inversion de la transformation additive, en DERNIER (miroir de
        # transform, qui l'applique en premier)
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

        # 4. Restauration optionnelle des valeurs d'origine exactes
        if self.restore_original_values:
            data_result = self._restore_original_values(data_result, y_col_name)

        # 5. Scission X et y, symétrique de "_transform"
        if y is not None and y_col_name in data_result.columns:
            y_original = data_result[y_col_name]
            X_original = data_result.drop(columns=[y_col_name])
            return X_original, y_original
        else:
            return data_result
