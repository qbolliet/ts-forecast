"""Mixed-frequency imputer """
# Importation des modules
# Modules de base
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)
# Manipulation de données
import pandas as pd
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
# Utilitaires du package
from ..xy.transformers import XYPanelTimeSeriesTransformer
from ..utils.frequency.utils import (
    detect_frequency,
    detect_index_frequency,
    is_higher_frequency,
    normalize_frequency,
)
from ..panel.utils import (
    get_unique_panel_entities,
    is_panel_data,
    normalize_entity_key,
    split_variable_key,
)
from .aggregation_constraint import (
    ConstraintSetting,
    validate_aggregation_constraint,
)
from .covariate_materializer import CovariateMaterializer
from .imputation_plan2 import ImputationPlan
from .imputation_window import (
    ImputationScope,
    ImputationWindowCalculator,
    TrainingScope,
)
from .provenance import ImputationProvenanceTracker
from .stage_scaler import ScaleMode, StageScaler
from .target_frequency_validator import TargetFrequencyValidator
from .training_set_builder import TrainingSetBuilder
from .variable_orderer import VariableOrderer

# Type aliases
VariableCategory = Literal['aggregate', 'impute', 'target_freq']
EntityKey = Tuple[Any, ...]
IntermediateFrequencies = Literal[False, 'covariates_only', True]

# Valeurs admissibles des littéraux.
_IMPUTATION_SCOPES: Tuple[str, ...] = get_args(ImputationScope)
_TRAINING_SCOPES: Tuple[str, ...] = get_args(TrainingScope)
_COVARIATE_STRATEGIES: Tuple[str, ...] = ('tolerate_nan', 'interpolate', 'model')
_COVARIATE_FALLBACKS: Tuple[str, ...] = ('interpolate', 'tolerate_nan')
_COVARIATE_ELIGIBILITIES: Tuple[str, ...] = ('any_entity', 'all_entities')
_FIT_PREDICT_ORDERS: Tuple[str, ...] = ('frequency', 'cv')
_FREQUENCY_MISMATCH_POLICIES: Tuple[str, ...] = ('error', 'warn')
_INTERMEDIATE_MODALITIES: Tuple[Any, ...] = (False, 'covariates_only', True)

# Attributs lus par "check_is_fitted"
_FITTED_ATTRIBUTES: Tuple[str, ...] = (
    'effective_target_frequency_',
    'detected_frequencies_',
    'variable_categories_',
    'frequency_progression_',
    'imputation_plan_',
)

# Lot livrant "transform" / "inverse_transform" ([SPEC] §17) /!\
_TRANSFORM_LOT = 'L12'


# Classe d'imputation multi-fréquences à deux axes orthogonaux
class HighFrequencyImputer2(XYPanelTimeSeriesTransformer):
    """Impute low-frequency columns onto a higher target frequency.

    The parameter space rests on **two orthogonal axes**, each answering one
    question and one only:

    - Axis 1, ``covariate_strategy``: how is a **covariate** observed less
      frequently than the current grid made available to the model?
    - Axis 2, ``impute_intermediate_frequencies``: does the imputed variable
      travel through **intermediate frequencies**, and does its final model
      train on its own imputations?

    Axis 1 governs the columns handed to the estimator, axis 2 governs the
    rows of its target. They compose without knowing each other.

    **Hard prerequisite of** ``covariate_strategy='tolerate_nan'``: the
    estimator **must tolerate NaN**. This modality hands the covariates to
    the model exactly as observed, holes included; a bare
    ``LinearRegression`` raises on them and sends the whole group to the
    interpolation fallback. Wrap the estimator in a ``Pipeline`` carrying a
    ``SimpleImputer``, or pick another strategy.

    **Downstream look of** ``covariate_strategy='interpolate'``: linear
    interpolation between two anchors reads the **future** anchor. A value
    materialized for 2021-03-31 out of annual anchors at 2021-12-31 and
    2022-12-31 therefore embeds information unavailable in real time. This is
    deliberate — the imputer reconstructs history, it does not forecast — but
    it forbids using the output as-is to simulate a real-time run.
    ``interpolation_method`` and ``interpolation_anchor`` tune the shape of
    that reconstruction, never its direction.

    **Inert combinations, documented and never warned about**
    — a ``UserWarning`` per combination would make
    hyperparameter search unbearable:

    - ``impute_intermediate_frequencies='covariates_only'`` **without**
      ``covariate_strategy='model'`` changes **no** final value: the
      covariates are materialized from their own observations, the
      intermediate stages cost compute and show up in the multi-frequency
      output, nothing else.
    - ``covariate_fallback`` is inert outside ``covariate_strategy='model'``.
    - ``fit_predict_order`` — and with it ``cv``, ``cv_scoring`` and
      ``min_cv_train_size`` as ordering devices — is inert outside
      ``covariate_strategy='model'``.
    - ``training_coverage_threshold`` without ``training_scope`` is inert:
      the training window then follows ``imputation_scope`` and
      ``coverage_threshold``.

    ``keep_lower_frequencies`` is a **pure display parameter**: it governs how
    the frequency levels of the output are stacked, never the logic. Under
    ``impute_intermediate_frequencies=False`` there is **no intermediate
    level to stack** — the output carries the source level and the target
    level, and nothing in between.

    **The price of** ``impute_intermediate_frequencies=False``: on a time
    series, ``y_train`` of an annual variable observed at three anchors holds
    **three rows**. ``min_cv_train_size`` and the estimator's own size guards
    are the price of the modality, not a defect; the interpolation fallback
    stays the safety net when a fit cannot happen. On
    a panel, the mutualization below widens that count without touching the
    origin filter.

    **Inter-entity mutualization of the training set**: the training set of a variable at a stage
    gathers **every entity observing that variable**, each contributing at
    the frequency at which it observes it, brought back to the stage scale by
    a divisor of its own block. Two consequences to keep in mind:

    - **Assumed bias**: mutualizing assumes comparable levels across
      entities. A country ten times bigger pulls the target, and
      ``scale_features`` corrects the **frequency** scale only, never the
      entity one. The escape hatch needs no parameter: fit one imputer per
      entity.
    - **Provenance is contagious across entities**: an entity contributing
      ``'interpolated'`` or ``'model'`` cells degrades ``target_taint``,
      hence the provenance of **every** cell the stage produces — including
      those of the other entities.

    Detected frequencies are indexed **per (entity, column)**:
    on a panel the same column may carry a different frequency for
    each entity, and every reasoning about it is per entity. Per-feature
    hyperparameter dicts (``scale_features``, ``interpolation_method``,
    ``interpolation_anchor``, ``estimator``, ``aggregation_constraint``) stay
    keyed by **column name**, never by ``(entity, column)``.

    Args:
        target_frequency: Target frequency, as a string applying to every
            entity, or a dict ``{entity: frequency}``. A dict must name
            **every** entity of the panel, or ``fit`` raises naming the
            missing ones.
        estimator: Estimator applied to every variable, or a dict
            ``{column: estimator}`` with an optional ``'__default__'`` key.
            ``None`` (default) sends every variable to interpolation, with a
            single warning at ``fit``.
        additive_transformer: Transformer making the data additive before any
            imputation (log, differencing, ...). Must expose ``fit_transform``
            **and** ``inverse_transform``. Additivity is the contract of the
            whole class, and this is its only escape hatch.
        covariate_strategy: Axis 1. ``'tolerate_nan'`` hands covariates over
            as observed (see the hard prerequisite above); ``'interpolate'``
            (default) materializes them by interpolation; ``'model'`` imputes
            them by model, cascading over the variables in
            ``fit_predict_order``.
        covariate_fallback: Way used when the ``'model'`` route fails. Inert
            outside ``covariate_strategy='model'``.
        covariate_eligibility: How a covariate's availability is aggregated
            over the entities of a panel. ``'any_entity'`` (default) keeps a
            column observed by at least one entity; ``'all_entities'`` is the
            conservative choice for estimators that do not tolerate NaN.
        interpolation_method: Interpolation method, global or per column.
        interpolation_anchor: Position of a value inside its period, in
            ``[0, 1]``, global or per column. ``None`` keeps the detected
            anchoring.
        impute_intermediate_frequencies: Axis 2. ``False`` (default) goes
            straight to the target frequency; ``'covariates_only'`` walks the
            intermediate stages but keeps ``y_train`` free of model-produced
            rows; ``True`` also trains on them. **Never tested for truth**:
            ``'covariates_only'`` is truthy.
        fit_predict_order: Order in which variables are imputed,
            ``'frequency'`` (default) or ``'cv'``. Inert outside
            ``covariate_strategy='model'``.
        cv: sklearn cross-validation strategy used by the ``'cv'`` order:
            ``None``, an int ``>= 2``, a splitter, or an iterable of splits.
            Resolved by ``check_cv`` at ``fit`` only.
        cv_scoring: Scoring of the ``'cv'`` order, higher is better.
        min_cv_train_size: Minimum number of scorable observations for a
            variable to be cross-validated. Below it, the variable falls back
            to the ``'frequency'`` ordering group.
        imputation_scope: Scope of the **prediction** window.
        coverage_threshold: Coverage ratio, in ``[0, 1]``, gating the
            extensions of the prediction window.
        training_scope: Scope of the **training** window. ``None`` (default)
            follows ``imputation_scope``. Widening it **adds rows, never
            columns**: feature selection stays governed by availability at
            prediction time.
        training_coverage_threshold: Coverage ratio of the training window's
            extensions. ``None`` follows ``coverage_threshold``. Inert
            without ``training_scope``.
        scale_features: ``False``, ``'constant'`` (default), ``'calendar'``,
            or a dict of these values keyed by column.
        aggregation_constraint: ``'sum'`` (default), ``None``, or a dict of
            these two values keyed by column with an optional
            ``'__default__'`` key.
        keep_lower_frequencies: Pure display parameter, see above.
        on_frequency_mismatch: ``'error'`` (default) or ``'warn'`` when
            ``target_frequency`` is higher than the data allows.
        restore_original_values: If True, ``inverse_transform`` refills every
            cell that was non-NaN in the input with its exact original value.
        time_col: Name of the time column when it is not in the index.
        panel_cols: Columns identifying the panel entities on a flat frame.
        verbose: If True, print progress messages prefixed
            ``[HighFrequencyImputer2]``.

    Attributes:
        effective_target_frequency_: Normalized target frequency, scalar or
            dict keyed by entity tuple.
        detected_frequencies_: Frequency detected at ``fit``:
            ``{column: frequency}`` on a time series,
            ``{(entity..., column): frequency}`` on a panel — **entities may
            diverge for one and the same column**.
        variable_categories_: Variable keys per category ``'aggregate'`` /
            ``'impute'`` / ``'target_freq'``, classified **per (entity,
            column) pair** on a panel.
        frequency_progression_: Ordered list of stage frequencies.
        imputation_order_: Variable order per stage. **Empty outside**
            ``covariate_strategy='model'``.
        imputation_plan_: :class:`ImputationPlan` — the complete fitted state.
        imputation_models_: Read-only view ``{(stage, variable): estimator}``
            over the plan.
        imputation_window_mask_: Boolean ``pd.Series`` of the prediction
            window (MultiIndex ``(entity..., date)`` on a panel).
        training_window_mask_: Boolean ``pd.Series`` of the training window.
        strict_window_mask_: Boolean ``pd.Series`` of the strict window.
        imputation_window_: Readable ``(start, end)`` bounds of the
            **prediction** window, or a dict of them per entity.
        training_window_: Readable ``(start, end)`` bounds of the **training**
            window. Note the difference with ``hfi``, where
            ``imputation_window_`` carried the strict bounds: here each
            attribute carries the bounds of its own mask, and
            ``strict_window_mask_`` is the sole holder of the strict window.
        imputation_provenance_: Provenance matrix after ``fit``, then after
            ``transform``.
        feature_columns_: Columns of ``X`` as received.
        target_column_: Name under which ``y`` was merged, or None.
        entities_: Entity keys of the panel, or None on a time series.
        is_panel_: Whether the data was handled as a panel.
        cv_: Splitter resolved by ``check_cv``. Present **only** under
            ``fit_predict_order='cv'``.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> data = pd.DataFrame(
        ...     {'m1': range(24), 'a1': float('nan')}, index=dates, dtype=float
        ... )
        >>> data.loc[['2021-12-31', '2022-12-31'], 'a1'] = [120.0, 132.0]
        >>> imputer = HighFrequencyImputer2(
        ...     target_frequency='M', estimator=LinearRegression()
        ... )
        >>> imputer.fit(data)                       # doctest: +SKIP
        >>> imputer.frequency_progression_          # doctest: +SKIP
        ['M']
    """

    # Initialisation
    def __init__(
        self,
        target_frequency: Union[str, Dict[Union[str, tuple], str]],
        estimator: Optional[Union[BaseEstimator, Dict[str, BaseEstimator]]] = None,
        additive_transformer: Optional[TransformerMixin] = None,
        # --- Axe 1 : matérialisation des covariables ---
        covariate_strategy: Literal['tolerate_nan', 'interpolate', 'model'] = 'interpolate',
        covariate_fallback: Literal['interpolate', 'tolerate_nan'] = 'interpolate',
        covariate_eligibility: Literal['any_entity', 'all_entities'] = 'any_entity',
        interpolation_method: Union[str, Dict[str, str]] = 'linear',
        interpolation_anchor: Union[None, float, Dict[str, Optional[float]]] = None,
        # --- Axe 2 : fréquences intermédiaires ---
        impute_intermediate_frequencies: IntermediateFrequencies = False,
        # --- Ordre d'imputation ---
        fit_predict_order: Literal['frequency', 'cv'] = 'frequency',
        cv: Union[int, Any, Iterable, None] = None,
        cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error',
        min_cv_train_size: int = 10,
        # --- Fenêtres ---
        imputation_scope: ImputationScope = 'strict',
        coverage_threshold: float = 0.5,
        training_scope: Optional[TrainingScope] = None,
        training_coverage_threshold: Optional[float] = None,
        # --- Échelle et contraintes  ---
        scale_features: Union[Literal[False], ScaleMode,
                              Dict[str, Union[Literal[False], ScaleMode]]] = 'constant',
        aggregation_constraint: Union[ConstraintSetting,
                                      Dict[str, ConstraintSetting]] = 'sum',
        # --- Sortie et divers ---
        keep_lower_frequencies: bool = True,
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
        restore_original_values: bool = False,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """Validate the parameters and store them untouched.

        Raises:
            ValueError: If a parameter is not admissible. The message always
                lists the admissible values or forms.
            TypeError: If a parameter has the wrong type (``target_frequency``
                neither str nor dict, a non-boolean boolean, a non-integer
                ``min_cv_train_size``).
        """
        # Initialisation du parent : les quatre drapeaux de validation sont
        # figés par cette classe et n'appartiennent pas à son espace de
        # paramètres publics
        super().__init__(
            time_col=time_col, panel_cols=panel_cols,
            validate_input=True, strict_validation=True,
            auto_sort=False, convert_cols_to_index=True
        )

        # Validation des paramètres
        self._validate_target_frequency_format(target_frequency)
        self._validate_estimator(estimator)
        if additive_transformer is not None:
            self._validate_additive_transformer(additive_transformer)

        # Validation des littéraux : message listant les valeurs admises
        self._validate_literal(
            'covariate_strategy', covariate_strategy, _COVARIATE_STRATEGIES
        )
        self._validate_literal(
            'covariate_fallback', covariate_fallback, _COVARIATE_FALLBACKS
        )
        self._validate_literal(
            'covariate_eligibility', covariate_eligibility, _COVARIATE_ELIGIBILITIES
        )
        self._validate_literal(
            'fit_predict_order', fit_predict_order, _FIT_PREDICT_ORDERS
        )
        self._validate_literal(
            'on_frequency_mismatch', on_frequency_mismatch,
            _FREQUENCY_MISMATCH_POLICIES
        )
        self._validate_literal(
            'imputation_scope', imputation_scope, _IMPUTATION_SCOPES
        )
        if training_scope is not None:
            self._validate_literal(
                'training_scope', training_scope, _TRAINING_SCOPES
            )

        # Validation des paramètres d'imputation des fréquences intermédiaires
        self._validate_intermediate_frequencies(impute_intermediate_frequencies)

        # Validation des formes par feature, déléguée aux validateurs
        # statiques des composants consommateurs : une seule implémentation,
        # donc aucune dérive possible entre ce que la classe accepte et ce que
        # le composant accepte
        CovariateMaterializer._validate_interpolation_method(interpolation_method)
        CovariateMaterializer._validate_interpolation_anchor(interpolation_anchor)
        StageScaler._validate_scale_features(scale_features)
        validate_aggregation_constraint(aggregation_constraint)

        # Validation de la stratégie de validation croisée. "check_cv" n'est
        # appelé qu'au fit.
        self._validate_cv(cv)
        self._validate_cv_scoring(cv_scoring)

        # Validation des bornes numériques
        if not isinstance(min_cv_train_size, int) or isinstance(min_cv_train_size, bool):
            raise TypeError(
                f"min_cv_train_size must be an int, "
                f"got {type(min_cv_train_size).__name__}"
            )
        if min_cv_train_size < 1:
            raise ValueError(
                f"min_cv_train_size must be >= 1, got {min_cv_train_size}"
            )
        self._validate_unit_interval('coverage_threshold', coverage_threshold)
        if training_coverage_threshold is not None:
            self._validate_unit_interval(
                'training_coverage_threshold', training_coverage_threshold
            )

        # Validation groupée des booléens : un 'frequency'/1/None passé par
        # erreur se propagerait sinon silencieusement jusqu'à un "if" qui
        # l'évalue
        boolean_params = {
            'keep_lower_frequencies': keep_lower_frequencies,
            'restore_original_values': restore_original_values,
            'verbose': verbose,
        }
        for param_name, param_value in boolean_params.items():
            if not isinstance(param_value, bool):
                raise TypeError(
                    f"{param_name} must be a bool, got {type(param_value).__name__}"
                )

        # Instanciation des attributs
        self.target_frequency = target_frequency
        self.estimator = estimator
        self.additive_transformer = additive_transformer
        self.covariate_strategy = covariate_strategy
        self.covariate_fallback = covariate_fallback
        self.covariate_eligibility = covariate_eligibility
        self.interpolation_method = interpolation_method
        self.interpolation_anchor = interpolation_anchor
        self.impute_intermediate_frequencies = impute_intermediate_frequencies
        self.fit_predict_order = fit_predict_order
        self.cv = cv
        self.cv_scoring = cv_scoring
        self.min_cv_train_size = min_cv_train_size
        self.imputation_scope = imputation_scope
        self.coverage_threshold = coverage_threshold
        self.training_scope = training_scope
        self.training_coverage_threshold = training_coverage_threshold
        self.scale_features = scale_features
        self.aggregation_constraint = aggregation_constraint
        self.keep_lower_frequencies = keep_lower_frequencies
        self.on_frequency_mismatch = on_frequency_mismatch
        self.restore_original_values = restore_original_values
        self.verbose = verbose

    # -------------------------------------------------------------------------
    # Journalisation
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de journalisation conditionnelle
    def _log(self, message: str) -> None:
        """Print a progress message when verbose mode is enabled.

        Args:
            message: Message to print, prefixed with the class tag.
        """
        # Silence total hors mode verbeux : aucune sortie standard produite
        if self.verbose:
            print(f"[HighFrequencyImputer2] {message}")

    # -------------------------------------------------------------------------
    # Validation des paramètres d'entrée
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de validation d'un littéral
    @staticmethod
    def _validate_literal(
        name: str,
        value: Any,
        admissible: Sequence[Any],
    ) -> None:
        """Check that a parameter belongs to its ``Literal``.

        Args:
            name: Parameter name, quoted in the message.
            value: Value handed to ``__init__``.
            admissible: Admissible values, listed in the message.

        Raises:
            ValueError: If the value is not one of the admissible ones.
        """
        # Appartenance au littéral, message énumérant les valeurs admises
        if value not in admissible:
            raise ValueError(
                f"{name} must be one of {tuple(admissible)}, got {value!r}"
            )

    # Méthode auxiliaire de validation de la modalité de l'axe 2
    @staticmethod
    def _validate_intermediate_frequencies(value: Any) -> None:
        """Check ``impute_intermediate_frequencies`` without truth testing.

        The three modalities are compared by identity for the booleans and by
        equality for the string. A membership test against a tuple would
        conflate ``0`` with ``False`` and ``1`` with ``True``; a truth test
        would silently promote ``'covariates_only'`` — which is truthy — into
        ``True``.

        Args:
            value: Value handed to ``__init__``.

        Raises:
            ValueError: If the value is not one of the three modalities.
        """
        # Comparaison modalité par modalité, jamais par vérité booléenne
        if value is False or value is True or value == 'covariates_only':
            return
        raise ValueError(
            f"impute_intermediate_frequencies must be one of "
            f"{_INTERMEDIATE_MODALITIES}, got {value!r}"
        )

    # Méthode auxiliaire de validation d'un réel de l'intervalle unité
    @staticmethod
    def _validate_unit_interval(name: str, value: Any) -> None:
        """Check that a parameter is a float within ``[0, 1]``.

        Args:
            name: Parameter name, quoted in the message.
            value: Value handed to ``__init__``.

        Raises:
            TypeError: If the value is not a real number.
            ValueError: If the value falls outside ``[0, 1]``.
        """
        # Exclusion explicite des booléens, que Python range parmi les entiers
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be a float, got {type(value).__name__}"
            )
        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} must be a float between 0 and 1, got {value}"
            )

    # Méthode auxiliaire de validation de la stratégie de validation croisée
    @staticmethod
    def _validate_cv(cv: Any) -> None:
        """Check the format of ``cv`` without resolving it.

        ``check_cv`` is deliberately not called here: it would materialize a
        splitter object that ``get_params`` would then have to hand back
        unchanged. Resolution happens at ``fit``, into ``cv_``.

        Args:
            cv: Value handed to ``__init__``.

        Raises:
            ValueError: If ``cv`` is an int below 2, or is neither None, an
                int, a splitter, nor an iterable.
        """
        # Absence de stratégie : sklearn choisira son défaut au fit
        if cv is None:
            return
        # Forme entière : nombre de plis
        if isinstance(cv, int) and not isinstance(cv, bool):
            if cv < 2:
                raise ValueError(f"cv must be an int >= 2, got {cv}")
            return
        # Forme objet : contrat de splitter sklearn
        if hasattr(cv, 'split') and hasattr(cv, 'get_n_splits'):
            return
        # Forme itérable : suite de couples d'index. Les chaînes sont exclues
        # explicitement — itérables au sens de Python, jamais une suite de
        # découpages
        if hasattr(cv, '__iter__') and not isinstance(cv, (str, bytes)):
            return
        raise ValueError(
            f"cv must be None, an int >= 2, a splitter exposing 'split' and "
            f"'get_n_splits', or an iterable of splits, got "
            f"{type(cv).__name__}"
        )

    # Méthode auxiliaire de validation du score de validation croisée
    @staticmethod
    def _validate_cv_scoring(cv_scoring: Any) -> None:
        """Check that ``cv_scoring`` is a string or a callable.

        Args:
            cv_scoring: Value handed to ``__init__``.

        Raises:
            ValueError: If the value is neither a string nor a callable.
        """
        # Chaîne du registre sklearn, ou scorer appelable
        if isinstance(cv_scoring, str) or callable(cv_scoring):
            return
        raise ValueError(
            f"cv_scoring must be a str or a callable, "
            f"got {type(cv_scoring).__name__}"
        )

    # Méthode auxiliaire de validation du format de la fréquence cible
    def _validate_target_frequency_format(
        self,
        target_frequency: Union[str, Dict[Union[str, tuple], str]]
    ) -> Union[str, Dict[Union[str, tuple], str]]:
        """Validate the format and values of the ``target_frequency`` parameter.

        Args:
            target_frequency: Target frequency (string or dict mapping
                entities to frequencies). Dict entity keys may be given in
                scalar form for a single-level panel (``{'FR': 'M'}``).

        Returns:
            Normalized target_frequency. For a dict, BOTH sides are
            normalized: the frequency values via ``normalize_frequency`` and
            the entity keys into tuples via ``normalize_entity_key``, so that
            ``{'FR': 'M'}`` becomes ``{('FR',): 'M'}``. This is the only place
            where user-supplied entity keys enter the imputer, and every
            downstream consumer then indexes ``effective_target_frequency_``
            by tuple without any defensive fallback.

        Raises:
            ValueError: If the format is invalid or a frequency is not
                normalizable.
            TypeError: If target_frequency is neither a string nor a dict.
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
            validated_freqs: Dict[EntityKey, str] = {}
            invalid_freqs: Dict[Any, str] = {}

            # Parcours des fréquences associées à chaque entité
            for entity, freq in target_frequency.items():
                # Vérification que la fréquence est une chaîne de caractères
                if not isinstance(freq, str):
                    raise ValueError(
                        f"Frequency for entity '{entity}' must be a string, "
                        f"got {type(freq).__name__}"
                    )
                # Normalisation de la clé d'entité en tuple ET de la fréquence
                try:
                    validated_freqs[normalize_entity_key(entity)] = normalize_frequency(freq)
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
        """Validate that the estimator exposes ``fit`` and ``predict``.

        Args:
            estimator: Estimator or dict of estimators to validate. The dict
                form admits a ``'__default__'`` key, treated like any other.

        Raises:
            ValueError: If an estimator lacks a required method, or if the
                dict form is empty.
        """
        # Cas où l'estimateur n'est pas spécifié : l'avertissement unique est
        # émis au fit, pas ici, pour ne pas se répéter à chaque clone
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
        """Validate that the additive transformer exposes its two methods.

        The contract is ``fit_transform`` **and** ``inverse_transform``:
        the transformer is fitted and applied in one call at
        phase 2 of the fit, and inverted at ``inverse_transform``.

        Args:
            transformer: Transformer to validate.

        Raises:
            ValueError: If the transformer lacks a required method.
        """
        # Liste des méthodes requises
        required_methods = ['fit_transform', 'inverse_transform']
        # Initialisation de la liste des méthodes manquantes
        missing_methods = []

        # Parcours des méthodes requises
        for method_name in required_methods:
            # Vérification que le transformer possède la méthode en attribut
            if not hasattr(transformer, method_name) or not callable(
                getattr(transformer, method_name)
            ):
                missing_methods.append(method_name)

        # Construction du message d'erreur si des méthodes sont manquantes
        if missing_methods:
            raise ValueError(
                f"additive_transformer must have methods: "
                f"{', '.join(required_methods)}. "
                f"Missing: {', '.join(missing_methods)}. "
                f"Got {type(transformer).__name__}"
            )

    # -------------------------------------------------------------------------
    # Conformité sklearn
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de vérification de l'ajustement
    def _check_fitted(self) -> None:
        """Raise a clean ``NotFittedError`` when the imputer is not fitted.

        The attribute list is explicit. The default sklearn convention
        — any attribute ending in ``_`` — would be satisfied by the parent
        class, which sets ``is_panel_``, ``n_features_`` and
        ``feature_names_`` before ``_fit`` runs: an interrupted fit would then
        look fitted.

        Raises:
            NotFittedError: If ``fit`` has not completed.
        """
        # Liste explicite plutôt que la convention du suffixe
        check_is_fitted(self, attributes=list(_FITTED_ATTRIBUTES))

    # Vue en lecture seule des modèles ajustés du plan
    @property
    def imputation_models_(self) -> Dict[Tuple[Any, Any], Any]:
        """Return the ``{(stage, variable): estimator}`` view over the plan.

        Returns:
            Mapping from ``(stage label, variable key)`` to the fitted model.

        Raises:
            AttributeError: If ``fit`` has not run. An ``AttributeError`` —
                and not a ``NotFittedError`` — keeps ``hasattr(imputer,
                'imputation_models_')`` False before the fit, so that this
                property never fools ``check_is_fitted``.
        """
        # Lecture directe du dictionnaire d'instance : la property, portée par
        # la classe, n'apparaît jamais dans "__dict__"
        plan = self.__dict__.get('imputation_plan_')
        if plan is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                f"'imputation_plan_'. Call fit() first."
            )
        return plan.models()

    # -------------------------------------------------------------------------
    # Alignement de la cible
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de résolution du nom de colonne de la cible
    def _resolve_target_column_name(self, y: pd.Series) -> str:
        """Resolve the column name used for ``y`` once merged into a frame.

        Single naming rule shared by ``_fit``, ``_transform`` and
        ``_inverse_transform``. Two independent rules previously let the
        fit-time name and the transform-time name diverge, so the target was
        never found among the stage columns and silently skipped imputation.

        Args:
            y: Target series, possibly unnamed.

        Returns:
            ``y.name`` if set, else the fallback ``'__target__'``.
        """
        return y.name if y.name is not None else '__target__'

    # Méthode auxiliaire d'alignement de l'index de la cible sur celui de X
    def _align_target_index(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Align ``y``'s index onto ``X``'s, tolerating the col->index step.

        The check is an **index equality** check, not a length check.
        Two series of the same length carrying different labels are a caller
        bug, not a case to fix silently — ``pd.concat`` aligns on index value,
        so reindexing would grow the working frame with NaN rows instead of
        raising.

        When ``time_col``/``panel_cols`` are used, the base transformer
        converts ``X``'s columns into a Multi/DatetimeIndex before ``_fit``
        ever sees it, but never touches ``y``. If ``y`` still carries the
        index recorded before that conversion, it is repositioned onto
        ``X.index``.

        Args:
            X: Working features, already validated and converted.
            y: Target series as received by ``fit``.

        Returns:
            ``y`` repositioned onto ``X.index`` when that is safe to do.

        Raises:
            ValueError: If the two indices neither match nor derive from the
                same pre-conversion index.
        """
        # Cas nominal : les deux index coïncident déjà
        if X.index.equals(y.index):
            return y

        # Cas colonnes -> index : la conversion appliquée à X, jamais à y,
        # est réappliquée à y en réutilisant les métadonnées de conversion
        conversion_meta = getattr(self, 'conversion_metadata_', None)
        if (
            conversion_meta is not None
            and conversion_meta.get('index_was_replaced')
            and conversion_meta['original_index'].equals(y.index)
        ):
            return y.set_axis(X.index)

        # Aucun des deux cas : décompte des libellés divergents, pour que le
        # message distingue une longueur différente d'un désaccord de valeurs
        if len(X) != len(y):
            detail = f"lengths differ ({len(X)} vs {len(y)})"
        else:
            mismatches = int((~X.index.isin(y.index)).sum())
            detail = (
                f"same length ({len(X)}) but {mismatches} label(s) of X's "
                f"index are absent from y's"
            )
        raise ValueError(
            f"X and y have different indices: {detail}. y must share X's "
            f"index, or — for column-based panel/time-series data "
            f"(time_col/panel_cols) — the index X had before those columns "
            f"were converted to the index."
        )

    # -------------------------------------------------------------------------
    # Fréquences détectées et classification des variables
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'extraction des fréquences d'une colonne par entité
    def _column_frequencies_by_entity(self, column: str) -> Dict[EntityKey, str]:
        """Return the frequency of one column for each entity observing it.

        Args:
            column: Bare column name.

        Returns:
            Mapping from entity key tuple to detected frequency. A time
            series yields the single key ``()``; a column with no detected
            frequency yields an empty dict.
        """
        # Itération + filtre plutôt qu'une compréhension indexée par le nom nu :
        # un panel peut porter la même colonne à des fréquences différentes
        # selon l'entité, et l'indexer par nom nu ferait gagner la dernière
        # entité rencontrée — donc dépendre de l'ordre des colonnes en entrée
        return {
            split_variable_key(key)[0]: freq
            for key, freq in self.detected_frequencies_.items()
            if split_variable_key(key)[1] == column
        }

    # Méthode auxiliaire de conversion des fréquences détectées à la forme des consommateurs
    def _detected_frequencies_by_column(
        self,
    ) -> Dict[str, Union[str, Dict[EntityKey, str]]]:
        """Return the detected frequencies keyed by column.

        Two shapes coexist on purpose. ``detected_frequencies_`` is the public
        attribute, keyed by ``(entity..., column)`` on a
        panel. ``CovariateMaterializer``, ``StageScaler`` and
        ``TrainingSetBuilder`` all read the other shape,
        ``{column: frequency | {entity: frequency}}``. This adapter is the
        single crossing point between the two, so no consumer ever
        re-implements the split.

        Returns:
            Mapping from column name to a scalar frequency (time series, or a
            panel where every entity agrees) or to a per-entity mapping.

        Examples:
            >>> imputer._detected_frequencies_by_column()   # doctest: +SKIP
            {'m1': 'M', 'q1': 'Q', 'v': {('FR',): 'Y', ('DE',): 'Q'}}
        """
        # Cas des séries temporelles : les clés sont déjà des noms de colonnes
        if not self.is_panel_:
            return dict(self.detected_frequencies_)

        # Cas du panel : regroupement par colonne, puis repli sur la forme
        # scalaire quand toutes les entités s'accordent
        by_column: Dict[str, Dict[EntityKey, str]] = {}
        for key, freq in self.detected_frequencies_.items():
            entity, column = split_variable_key(key)
            by_column.setdefault(column, {})[entity] = freq

        result: Dict[str, Union[str, Dict[EntityKey, str]]] = {}
        for column, per_entity in by_column.items():
            unique = set(per_entity.values())
            result[column] = per_entity if len(unique) > 1 else unique.pop()
        return result

    # Méthode auxiliaire de classification des variables relativement à une fréquence
    def _classify_variables_at_frequency(
        self,
        prediction_frequency: Union[str, Dict[EntityKey, str]],
    ) -> Dict[str, List[Union[str, tuple]]]:
        """Classify variables relative to a prediction frequency.

        On a panel the classification is done **per (entity, column) pair**,
        never per column: the same column may be annual for ``FR``, quarterly
        for ``DE`` and monthly for ``IT``, hence imputable for the first two
        and already at target for the third.

        Args:
            prediction_frequency: Frequency at which predictions will be made
                (str for a time series, dict entity -> frequency for a panel).

        Returns:
            Dict with keys ``'aggregate'``, ``'impute'`` and
            ``'target_freq'``, each holding a list of variable keys.

        Raises:
            TypeError: If a dict is passed for time series data.
        """
        # Initialisation du dictionnaire résultat
        result: Dict[str, List[Union[str, tuple]]] = {
            'aggregate': [], 'impute': [], 'target_freq': []
        }

        # Distinction suivant la structure des données
        # Données de panel
        if self.is_panel_:
            # Parcours des fréquences détectées, couple par couple
            for key, freq in self.detected_frequencies_.items():
                # Décomposition de la clé : entité toujours en tuple
                entity, _ = split_variable_key(key)

                # Extraction de la fréquence cible associée à l'entité
                if isinstance(prediction_frequency, dict):
                    pred_freq = prediction_frequency.get(entity)
                else:
                    pred_freq = prediction_frequency

                # Vérification que la fréquence cible est spécifiée
                if pred_freq is None:
                    continue

                # Normalisation des deux fréquences comparées
                freq_normalized = normalize_frequency(freq)
                pred_normalized = normalize_frequency(pred_freq)

                # Comparaison des fréquences, POUR CETTE ENTITÉ
                if is_higher_frequency(freq, pred_freq):
                    # Agrégation si la fréquence source est plus fine que la cible
                    result['aggregate'].append(key)
                elif freq_normalized == pred_normalized:
                    # Cas d'égalité : déjà à la fréquence cible
                    result['target_freq'].append(key)
                else:
                    # Imputation si la fréquence source est plus basse que la cible
                    result['impute'].append(key)
        # Cas des séries temporelles
        else:
            # Cas d'erreur si la fréquence cible est un dictionnaire
            if not isinstance(prediction_frequency, str):
                raise TypeError(
                    "'prediction_frequency' should be a string when applied "
                    "to time series"
                )
            # Parcours des fréquences détectées
            for col, freq in self.detected_frequencies_.items():
                # Normalisation des deux fréquences comparées
                freq_normalized = normalize_frequency(freq)
                pred_normalized = normalize_frequency(prediction_frequency)

                # Comparaison des fréquences
                if is_higher_frequency(freq, prediction_frequency):
                    result['aggregate'].append(col)
                elif freq_normalized == pred_normalized:
                    result['target_freq'].append(col)
                else:
                    result['impute'].append(col)

        return result

    # Méthode auxiliaire de regroupement des couples imputables par fréquence source
    def _imputable_groups(
        self,
        prediction_frequency: Union[str, Dict[EntityKey, str]],
    ) -> Dict[Tuple[str, str], Tuple[EntityKey, ...]]:
        """Group the imputable ``(entity, column)`` pairs by source frequency.

        Each ``(column, f_var)`` group yields one plan step at the stage,
        all of them sharing the model fitted on the mutualized training set.

        Args:
            prediction_frequency: Frequency of the stage.

        Returns:
            Mapping from ``(column, source frequency)`` to the tuple of
            entity keys of the group, each tuple sorted for determinism. On a
            time series the entity key is ``()``.

        Examples:
            >>> imputer._imputable_groups('M')          # doctest: +SKIP
            {('v', 'Y'): (('FR',),), ('v', 'Q'): (('DE',),)}
        """
        # Regroupement des couples imputables par (colonne, fréquence source)
        groups: Dict[Tuple[str, str], List[EntityKey]] = {}
        categories = self._classify_variables_at_frequency(prediction_frequency)
        for key in categories['impute']:
            entity, column = split_variable_key(key)
            source_freq = normalize_frequency(
                self.detected_frequencies_[key], return_format='base'
            )
            groups.setdefault((column, source_freq), []).append(entity)

        # Tri des entités de chaque groupe : l'ordre du plan ne doit dépendre
        # ni de l'ordre des colonnes ni de celui des lignes en entrée
        return {
            group_key: tuple(sorted(entities, key=repr))
            for group_key, entities in groups.items()
        }

    # -------------------------------------------------------------------------
    # Fenêtres
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de mise en correspondance des bornes (start, end) d'une fenêtre
    @staticmethod
    def _zip_window_bounds(
        start: Union[pd.Timestamp, Dict[EntityKey, Optional[pd.Timestamp]], None],
        end: Union[pd.Timestamp, Dict[EntityKey, Optional[pd.Timestamp]], None],
    ) -> Union[Tuple, Dict[EntityKey, Tuple]]:
        """Pair per-entity (or scalar) start/end bounds into tuples.

        Args:
            start: Window start(s), as returned by
                ``ImputationWindowCalculator`` (scalar for a time series, dict
                keyed by entity tuple for a panel).
            end: Window end(s), same shape as ``start``.

        Returns:
            ``(start, end)`` tuple, or dict mapping entities to
            ``(start, end)``.
        """
        # Cas des données de panel : appariement par entité
        if isinstance(start, dict):
            return {entity: (start[entity], end.get(entity)) for entity in start}
        # Cas des séries temporelles
        return (start, end)

    # Méthode auxiliaire de sélection des colonnes de features
    def _select_feature_columns(
        self,
        candidates: Sequence[str],
        training_frame: pd.DataFrame,
        prediction_frame: pd.DataFrame,
    ) -> Tuple[str, ...]:
        """Select the covariates kept for one stage, per the window rules.

        **Widening ``training_scope`` adds rows, never columns**.
        Feature selection stays governed by availability at prediction
        time, independently of the training window. The two adjustments of
        follow from it:

        - a column is kept only if it is non-empty on **both** windows — a
          column observed only on rows the training scope added would train a
          coefficient the prediction grid can never feed;
        - per-entity eligibility is delegated to
          ``CovariateMaterializer.eligible_columns``, which carries
          ``covariate_eligibility``.

        The symmetric row rule — training rows carrying no observed covariate
        at all are dropped — is applied by :meth:`_drop_empty_training_rows`.

        Args:
            candidates: Candidate covariate columns, in output order.
            training_frame: Frame restricted to the training grid.
            prediction_frame: Frame restricted to the prediction grid.

        Returns:
            Kept columns, in the order of ``candidates``.

        Examples:
            >>> imputer._select_feature_columns(       # doctest: +SKIP
            ...     ['m1', 'q1'], train_frame, pred_frame
            ... )
            ('m1',)
        """
        # Éligibilité par entité, portée par le matérialiseur
        eligible = set(
            self._covariate_materializer.eligible_columns(
                candidates, prediction_frame
            )
        )

        # Non-vacuité sur les deux fenêtres : une colonne vide sur l'une des
        # deux grilles n'apporte aucune information exploitable
        kept = []
        for column in candidates:
            if column not in eligible:
                continue
            if column not in training_frame.columns:
                continue
            if column not in prediction_frame.columns:
                continue
            if training_frame[column].notna().any() and prediction_frame[column].notna().any():
                kept.append(column)
        return tuple(kept)

    # Méthode auxiliaire d'élimination des lignes d'entraînement sans covariable
    @staticmethod
    def _drop_empty_training_rows(
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Drop the training rows carrying no observed covariate at all.

        Second adjustment : symmetric to
        :meth:`_select_feature_columns`: a row whose every covariate is NaN
        teaches the estimator nothing and, under an estimator that does not
        tolerate NaN, sends the whole fit to the fallback.

        Args:
            X_train: Materialized training features.
            y_train: Raw training target, sharing ``X_train``'s index.

        Returns:
            The pair restricted to the rows carrying at least one observed
            covariate. An empty ``X_train`` (no covariate at all) is returned
            untouched: there is no row to discriminate on.
        """
        # Aucune covariable : rien à filtrer, la garde de taille du fit joue
        if X_train.shape[1] == 0:
            return X_train, y_train

        # Conservation des lignes portant au moins une covariable observée
        kept_rows = X_train.notna().any(axis=1)
        return X_train.loc[kept_rows], y_train.loc[kept_rows]

    # -------------------------------------------------------------------------
    # Progression de fréquences
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de construction de la progression de fréquences
    def _build_frequency_progression(self) -> List[Union[str, Dict[EntityKey, str]]]:
        """Build the ordered list of stage frequencies.

        Under ``impute_intermediate_frequencies is False`` the progression is
        the target frequency alone: the imputed variable jumps straight from
        its own frequency to the target, with no intermediate stage.

        Returns:
            Ordered list of stage frequencies, the target frequency last.

        Raises:
            NotImplementedError: Under ``'covariates_only'`` and ``True``.
        """
        # Modalité sans étape intermédiaire : une seule étape, la cible.
        if self.impute_intermediate_frequencies is False:
            return [self.effective_target_frequency_]

        # TODO (lot L11) : progression complète (§5.2, points 1 et 3) —
        # ensemble F des fréquences des couples (entité, colonne) imputables
        # plus la cible, privé de la plus basse, trié du plus bas au plus
        # haut, la cible en dernier. Point d'extension unique : le reste du
        # fit parcourt déjà "frequency_progression_" sans hypothèse sur sa
        # longueur
        raise NotImplementedError(
            f"impute_intermediate_frequencies="
            f"{self.impute_intermediate_frequencies!r} is delivered by lot "
            f"L11; only False is supported so far."
        )

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'entraînement
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Learn the imputation plan from X and y.

        Phases of the fitting logic:

        - PHASE 0: setup — transform-state purge, columns, panel
          detection, ``y`` alignment and naming, frequency detection,
          ``target_frequency`` normalization and validation, variable
          classification per (entity, column) pair.
        - PHASE 1: the three window masks, each caller naming its
          ``kind``.
        - PHASE 2: additive transformer.
        - PHASE 3: frequency progression.
        - PHASE 4: provenance tracker, initialized after the additive
          transformer.

        PHASE 5 (stage execution) is lot L10 and PHASE 6 lot L12; the
        attributes they fill are initialized empty here.

        Args:
            X: Features of shape (n_samples, n_features).
            y: Target of shape (n_samples,), optional.

        Raises:
            ValueError: If no frequency can be detected, if a
                ``target_frequency`` dict misses entities (B16), or if the
                indices of X and y disagree (B14).
        """
        # =================================================================
        # PHASE 0 — Setup
        # =================================================================
        # Purge de l'état écrit par un précédent "transform"
        for stale_attr in ('imputation_provenance_', '_original_X_', '_original_y_'):
            if stale_attr in self.__dict__:
                delattr(self, stale_attr)

        # Extraction des colonnes d'intérêt
        self.feature_columns_ = list(X.columns)
        # Identification de la structure des données
        self.is_panel_ = bool(self.panel_cols) or is_panel_data(data=X)

        # Construction du jeu de données de travail
        if y is not None:
            # Alignement de l'index de y sur celui de X.
            y = self._align_target_index(X, y)
            # Extraction du nom de y
            y_col_name = self._resolve_target_column_name(y)
            # Concaténation en un unique jeu de données
            X_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            y_col_name = None
            X_work = X.copy()
        # Nom de la colonne que l'on cherche à prédire
        self.target_column_ = y_col_name

        # Identification des entités
        if self.is_panel_ and isinstance(X.index, pd.MultiIndex):
            self.entities_ = get_unique_panel_entities(X)
        else:
            self.entities_ = None

        # Label de fréquence de l'index d'entrée : il identifie, à l'inversion,
        # le niveau à conserver dans une sortie multi-fréquences
        try:
            index_freq = detect_index_frequency(X_work.index, return_format='base')
            self._source_index_frequency_label = self._stage_frequency_label(index_freq)
        except (ValueError, TypeError):
            # Index irrégulier ou trop court : le repli sur le niveau cible suffit
            self._source_index_frequency_label = None

        # Avertissement unique si aucun estimateur n'est fourni
        if self.estimator is None:
            warnings.warn(
                "No estimator was provided (estimator=None): every variable "
                "will fall back to linear interpolation.",
                UserWarning
            )

        # Normalisation de target_frequency (la fréquence de prédiction)
        normalized_target_frequency = self._validate_target_frequency_format(
            self.target_frequency
        )

        # Expansion de target_frequency en dict si le jeu de données est un panel et la fréquence cible une chaîne de caractères
        if self.is_panel_ and isinstance(normalized_target_frequency, str) and self.entities_:
            self.effective_target_frequency_ = {
                entity: normalized_target_frequency for entity in self.entities_
            }
        elif isinstance(normalized_target_frequency, dict):
            # Un dict incomplet est une erreur nommant les entités
            # manquantes, et non un silence dont la conséquence n'apparaît
            # qu'à la classification, sous la forme d'entités jamais imputées
            self._check_target_frequency_covers_entities(normalized_target_frequency)
            self.effective_target_frequency_ = normalized_target_frequency.copy()
        else:
            self.effective_target_frequency_ = normalized_target_frequency

        # Détection des fréquences par (entité, colonne) : sur un panel, la
        # même colonne peut porter une fréquence différente selon l'entité
        # et les clés sont alors des tuples (entité..., colonne)
        raw_frequencies = detect_frequency(data=X_work)

        # Écartement des couples sans fréquence détectable : une entité
        # n'observant jamais une colonne n'a pas de fréquence pour elle.
        # "covariate_eligibility" côté covariables et "impute_intermediate_frequency" /!\ a remplacer par le bon nom de méthode une fois implémentée
        # côté cible en tirent chacun les conséquences
        self._undetected_frequencies_ = tuple(
            key for key, freq in raw_frequencies.items() if freq is None
        )
        self.detected_frequencies_ = {
            key: freq for key, freq in raw_frequencies.items() if freq is not None
        }
        # Vérification que le dictionnaire des fréquences détectées est non vide
        if not self.detected_frequencies_:
            raise ValueError("Could not detect frequency for any column")
        # Avertissement pour les colonnes manquantes par entité
        if self._undetected_frequencies_:
            self._log(
                f"[fit] {len(self._undetected_frequencies_)} (entity, column) "
                f"pair(s) never observed, left out of the classification: "
                f"{self._undetected_frequencies_}"
            )

        # Validation de la fréquence cible contre les fréquences détectées
        self.effective_target_frequency_ = self._target_freq_validator.validate(
            target_frequency=self.effective_target_frequency_,
            detected_frequencies=self.detected_frequencies_,
            on_frequency_mismatch=self.on_frequency_mismatch,
        )

        # Classification des variables, par couple (entité, colonne)
        self.variable_categories_ = self._classify_variables_at_frequency(
            self.effective_target_frequency_
        )

        # Résolution de la stratégie de validation croisée : "cv_" n'existe
        # que sous fit_predict_order='cv'
        if self.fit_predict_order == 'cv':
            self._variable_orderer = VariableOrderer(
                fit_predict_order=self.fit_predict_order,
                cv=self.cv,
                cv_scoring=self.cv_scoring,
                min_cv_train_size=self.min_cv_train_size,
            ).fit()
            self.cv_ = self._variable_orderer.cv_
        else:
            self._variable_orderer = VariableOrderer(
                fit_predict_order=self.fit_predict_order,
                cv=self.cv,
                cv_scoring=self.cv_scoring,
                min_cv_train_size=self.min_cv_train_size,
            )
            # Purge d'un "cv_" laissé par un fit précédent sous un autre ordre
            if 'cv_' in self.__dict__:
                delattr(self, 'cv_')

        # Ordre d'imputation : vide hors covariate_strategy='model',
        # l'ordre n'ayant alors aucun effet observable. Sous
        # 'model', l'ordre par étape est produit en phase 5
        self.imputation_order_ = {}

        # Instanciation du matérialiseur : producteur unique des features,
        # porteur des trois registres et de la contrainte d'agrégation
        self._covariate_materializer = CovariateMaterializer(
            covariate_strategy=self.covariate_strategy,
            covariate_fallback=self.covariate_fallback,
            covariate_eligibility=self.covariate_eligibility,
            interpolation_method=self.interpolation_method,
            interpolation_anchor=self.interpolation_anchor,
            aggregation_constraint=self.aggregation_constraint,
        )

        # =================================================================
        # PHASE 1 — Fenêtres : les trois masques
        # =================================================================
        # Initialisation du calculateur de fenêtre
        window_calc, window_error = self._fit_imputation_window(X_work)

        # Cas d'échec du calcul : le calculateur non entraîné est conservé,
        # les gardes "_is_fitted" des consommateurs le neutralisent
        if window_calc is None:
            self._imputation_window_calc = self._make_window_calculator()
            warnings.warn(
                f"Could not calculate imputation window: {window_error}. "
                f"Using all available data.",
                UserWarning
            )
            # Repli sur les bornes extrêmes de l'index
            if isinstance(X_work.index, pd.MultiIndex):
                time_idx = X_work.index.get_level_values(-1)
            else:
                time_idx = X_work.index
            self.imputation_window_ = (time_idx.min(), time_idx.max())
            self.training_window_ = self.imputation_window_
            # Aucune restriction : les trois masques valent True partout
            permissive = pd.Series(True, index=X_work.index)
            self.strict_window_mask_ = permissive
            self.imputation_window_mask_ = permissive.copy()
            self.training_window_mask_ = permissive.copy()
        else:
            self._imputation_window_calc = window_calc

            # Chaque appelant nomme explicitement son masque
            self.strict_window_mask_ = window_calc.get_imputation_window_mask(
                X_work, kind='strict'
            )
            self.imputation_window_mask_ = window_calc.get_imputation_window_mask(
                X_work, kind='imputation'
            )
            self.training_window_mask_ = window_calc.get_imputation_window_mask(
                X_work, kind='training'
            )

            # Bornes lisibles : chaque attribut porte celles de son masque.
            self.imputation_window_ = self._zip_window_bounds(
                window_calc.imputation_window_start_,
                window_calc.imputation_window_end_,
            )
            self.training_window_ = self._zip_window_bounds(
                window_calc.training_window_start_,
                window_calc.training_window_end_,
            )

            # Avertissement global si aucune fenêtre stricte n'existe : sans
            # lui, tous les entraînements échouent silencieusement un à un et
            # tout finit en repli par interpolation 
            # /!\ Cela ne me semble problématique que pour la stratégie "cv" qui est la seule à utiliser la fenêtre stricte
            # /!\ D'ailleurs je ne suis pas sûr que ça soit souhaitable de se restreindre à la période stricte pour "cv" et ne pas prendre toute la fenêtre d'entraînement car cela peut biaiser les résultats non ? Il y a moins de valeurs manquantes.
            start = window_calc.imputation_strict_window_start_
            no_window = (
                start is None if not isinstance(start, dict)
                else all(v is None for v in start.values())
            )
            if no_window:
                warnings.warn(
                    "No strict imputation window found: no model can be trained; "
                    "all imputations will fall back to interpolation.",
                    UserWarning
                )

        # Câblage du constructeur de jeu d'entraînement mutualisé.
        # Le "kind" du masque est nommé ici, une fois pour toutes : le
        # composant reçoit un callable, jamais le calculateur, et ne peut donc
        # pas lire un autre masque que celui d'entraînement.
        # La contrainte d'agrégation ne lui est pas passée : elle ne joue
        # aucun rôle dans la composition du jeu
        # et lui parvient de toute façon par le matérialiseur injecté
        self._training_set_builder = TrainingSetBuilder(
            materializer=self._covariate_materializer,
            training_mask=lambda frequencies: (
                self._imputation_window_calc.get_mask_at_frequency(
                    frequencies, kind='training'
                )
            ),
            log=self._log if self.verbose else None,
        )

        # =================================================================
        # PHASE 2 — Transformateur additif
        # =================================================================
        # Passage en représentation additive, unique échappatoire à
        # l'hypothèse d'additivité de la classe entière
        if self.additive_transformer is not None:
            # Initialisation du transformer
            self.additive_transformer_ = clone(self.additive_transformer)
            # Additivité des données
            X_work = self.additive_transformer_.fit_transform(X_work)
            # Déballage du couple (X, y) que renvoie un transformateur XY
            if isinstance(X_work, tuple):
                X_work = X_work[0]
        else:
            self.additive_transformer_ = None

        # =================================================================
        # PHASE 3 — Progression de fréquences
        # =================================================================
        # Progression des fréquences à imputer
        self.frequency_progression_ = self._build_frequency_progression()
        # Logging
        self._log(
            f"[fit] Frequency progression: "
            f"{[self._stage_frequency_label(f) for f in self.frequency_progression_]}"
        )

        # =================================================================
        # PHASE 4 — Provenance
        # =================================================================
        # Initialisation après le transformateur additif : le tracker
        # scanne le jeu de données réellement imputé, et non celui d'avant la
        # transformation, dont les cellules non nulles ne coïncident pas
        # nécessairement.
        self._provenance_tracker = ImputationProvenanceTracker()
        self._provenance_tracker.initialize(X_work, panel_cols=self.panel_cols)

        # =================================================================
        # PHASE 5 — Exécution des étapes : lot L10
        # =================================================================
        # Le plan est l'état ajusté complet : il est initialisé vide ici et
        # rempli étape par étape par le lot L10
        self.imputation_plan_ = ImputationPlan()

        # =================================================================
        # PHASE 6 — Finalisation : lot L12
        # =================================================================
        self.imputation_provenance_ = self._provenance_tracker.get_provenance_matrix()

    # Méthode auxiliaire de vérification de la couverture des entités
    def _check_target_frequency_covers_entities(
        self,
        normalized_target_frequency: Dict[EntityKey, str],
    ) -> None:
        """Check that a ``target_frequency`` dict names every entity.

        Args:
            normalized_target_frequency: Target frequency dict, entity keys
                already normalized into tuples.

        Raises:
            ValueError: If entities of the panel are missing from the dict.
                The message names them — a silent gap would only surface much
                later, as entities the classification never imputes.
        """
        # Contrôle sans objet hors panel, ou tant que les entités sont inconnues
        if not self.is_panel_ or not self.entities_:
            return

        # Énumération des entités absentes du dictionnaire
        missing = [
            entity for entity in self.entities_
            if normalize_entity_key(entity) not in normalized_target_frequency
        ]
        if missing:
            raise ValueError(
                f"target_frequency dict is incomplete: no frequency given for "
                f"{len(missing)} entity/entities "
                f"{tuple(missing)}. A dict target_frequency must name every "
                f"entity of the panel."
            )

    # Méthode auxiliaire de construction d'un calculateur de fenêtre
    def _make_window_calculator(self) -> ImputationWindowCalculator:
        """Build a window calculator carrying the imputer's hyperparameters.

        Returns:
            Unfitted :class:`ImputationWindowCalculator` configured with the
            four window parameters.
        """
        # Les deux paramètres d'entraînement retombent sur ceux de prédiction
        # quand ils valent None : le calculateur porte lui-même cette règle
        return ImputationWindowCalculator(
            coverage_threshold=self.coverage_threshold,
            imputation_scope=self.imputation_scope,
            training_scope=self.training_scope,
            training_coverage_threshold=self.training_coverage_threshold,
            min_columns=2,
        )

    # Méthode auxiliaire d'ajustement d'un calculateur de fenêtre d'imputation
    def _fit_imputation_window(
        self,
        data: pd.DataFrame,
    ) -> Tuple[Optional[ImputationWindowCalculator], Optional[ValueError]]:
        """Fit a fresh window calculator on the given data.

        The window is a constraint on data availability, not an estimated
        parameter: it is a deterministic function of the frame it is computed
        on. ``_fit`` and ``_transform`` therefore share this factory and each
        compute the window on their own data, which is what preserves
        ``fit_transform(X) == fit(X).transform(X)``.

        Args:
            data: Frame the window is computed on. Must be the frame BEFORE
                the additive transformer, on both paths, or the two windows
                would not coincide.

        Returns:
            Tuple ``(calculator, error)``: the fitted calculator and None, or
            None and the ``ValueError`` the calculator raised.
        """
        # Instanciation avec les hyperparamètres de l'imputeur
        calculator = self._make_window_calculator()
        # Estimation de la fenêtre
        try:
            calculator.fit(data)
        except ValueError as error:
            return None, error

        return calculator, None

    # Méthode auxiliaire de construction d'un label de fréquence lisible pour une étape
    def _stage_frequency_label(self, pred_freq: Union[str, Dict]) -> str:
        """Build a human-readable frequency label for a stage.

        Labeling rule for panel stages: if every entity of the stage shares
        the same frequency, that shared frequency is the label; otherwise a
        composite label lists each entity's frequency, sorted by entity key
        for determinism.

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
        return '+'.join(parts)

    # Validateur de fréquence cible, à initialisation paresseuse
    @property
    def _target_freq_validator(self) -> TargetFrequencyValidator:
        """Return the memoized target-frequency validator.

        Returns:
            Shared :class:`TargetFrequencyValidator` instance. The attribute
            name carries no trailing underscore, so it never fools
            ``check_is_fitted``.
        """
        # Mémoïsation paresseuse : le validateur est sans état
        if getattr(self, '_target_freq_validator_cache', None) is None:
            self._target_freq_validator_cache = TargetFrequencyValidator()
        return self._target_freq_validator_cache

    # -------------------------------------------------------------------------
    # Transform et inversion
    # -------------------------------------------------------------------------
    # Surcharge de transform pour la vérification d'ajustement explicite
    def transform(self, X, y=None):
        """Transform X, and optionally y.

        Args:
            X: Features to transform.
            y: Target to transform (optional).

        Returns:
            The transformed features, or the pair ``(X, y)`` when ``y`` is
            given.

        Raises:
            NotFittedError: If ``fit`` has not run (B20).
            NotImplementedError: Always, until lot L12.
        """
        # Vérification d'ajustement avant toute autre chose : sans elle, le
        # NotImplementedError de "_transform" masquerait le NotFittedError
        self._check_fitted()
        return super().transform(X, y)

    # Surcharge d'inverse_transform pour la vérification d'ajustement explicite
    def inverse_transform(self, X, y=None):
        """Invert the transformation of X, and optionally y.

        Args:
            X: Transformed features.
            y: Transformed target (optional).

        Returns:
            The original features, or the pair ``(X, y)`` when ``y`` is given.

        Raises:
            NotFittedError: If ``fit`` has not run (B20).
            NotImplementedError: Always, until lot L12.
        """
        # Même ordre que "transform" : ajustement d'abord
        self._check_fitted()
        return super().inverse_transform(X, y)

    # Méthode de transformation, livrée par le lot L12
    def _transform(self, X, y=None):
        """Replay the fitted plan on new data.

        Deliberately NOT implemented in this lot. A provisional version would
        survive and drift away from the fit — defects B7/B27, the very motive
        of this architecture: ``fit`` and ``transform`` must share ONE
        implementation of step execution, which lot L10 delivers first.

        Args:
            X: Features to transform.
            y: Target to transform (optional).

        Raises:
            NotImplementedError: Always, until lot L12.
        """
        raise NotImplementedError(
            f"HighFrequencyImputer2.transform is delivered by lot "
            f"{_TRANSFORM_LOT} ([SPEC] §12.4). This lot (L9) delivers "
            f"__init__ and fit phases 0 to 4 only."
        )

    # Méthode de transformation inverse, livrée par le lot L12
    def _inverse_transform(self, X, y=None):
        """Undo the imputation and the additive transformer.

        Deliberately NOT implemented in this lot, for the same reason as
        :meth:`_transform`.

        Args:
            X: Transformed features.
            y: Transformed target (optional).

        Raises:
            NotImplementedError: Always, until lot L12.
        """
        raise NotImplementedError(
            f"HighFrequencyImputer2.inverse_transform is delivered by lot "
            f"{_TRANSFORM_LOT} ([SPEC] §12.4). This lot (L9) delivers "
            f"__init__ and fit phases 0 to 4 only."
        )
