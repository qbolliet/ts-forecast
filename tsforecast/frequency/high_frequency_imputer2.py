"""Mixed-frequency imputer """
# Importation des modules
# Modules de base
import warnings
from dataclasses import dataclass, replace
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
import numpy as np
import pandas as pd
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
# Utilitaires du package
from ..xy.transformers import XYPanelTimeSeriesTransformer
from ..utils.frequency.utils import (
    detect_frequency,
    detect_index_frequency,
    get_frequency_order,
    is_higher_frequency,
    normalize_frequency,
)
from ..panel.utils import (
    get_unique_panel_entities,
    iter_entity_blocks,
    is_panel_data,
    normalize_entity_key,
    split_variable_key,
)
from .aggregation_constraint import (
    AggregationConstraint,
    ConstraintSetting,
    validate_aggregation_constraint,
)
from .covariate_materializer import CovariateMaterializer
from .imputation_plan import INTERPOLATE_FALLBACK
from .imputation_plan2 import (
    ImputationPlan,
    ImputationStep,
    MaterializationWay,
    append_step,
    to_entity_tuple,
)
from .imputation_window import (
    ImputationScope,
    ImputationWindowCalculator,
    TrainingScope,
)
from .provenance import (
    CellOrigin,
    ImputationProvenanceTracker,
    ProvenanceType,
    Taint,
    max_origin,
    origin_to_taint,
    resolve_model_provenance,
)
from .stage_scaler import ScaleMode, StageScaler
from .target_frequency_validator import TargetFrequencyValidator
from .training_set_builder import TrainingSet, TrainingSetBuilder
from .variable_orderer import VariableOrderer, VariableSpec

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

# Origines de cellule admissibles dans "y_train", par modalité de l'axe 2.
# Table plutôt qu'une suite de tests : la modalité n'est
# jamais évaluée comme un booléen ('covariates_only' est truthy)
ELIGIBLE_ORIGINS: Dict[Any, Tuple[CellOrigin, ...]] = {
    False: ('observed',),
    'covariates_only': ('observed',),
    True: ('observed', 'interpolated', 'model'),
}

# Lot livrant "transform" / "inverse_transform" ([SPEC] §17) /!\
_TRANSFORM_LOT = 'L12'


# Contexte d'ajustement d'une variable à une étape
@dataclass(frozen=True)
class _VariableFit:
    """Everything one (stage, variable) needs before an estimator is fitted.

    Composed by :meth:`HighFrequencyImputer2._prepare_variable` and consumed
    by two callers, which is exactly why it exists: the ordering of PHASE 5b
    and the fit of PHASE 5c must rank and estimate on the SAME problem - same
    covariates, same materialization, same rows, same scale. Whatever the
    ordering scores, the fit then estimates.

    The two callers share the code, never the result: PHASE 5b runs before any
    variable of the stage has been imputed, so a context it composed is stale
    as soon as the first variable has been written to the mirror. PHASE 5c
    therefore recomposes its own.

    Attributes:
        training: The mutualized :class:`TrainingSet`, raw target included -
            the source of ``row_origin`` and of the ``ways`` the plan step
            freezes.
        blocks: Mapping entity -> ``f_block(e)``, composition of the
            mutualized set.
        pred_grid: Prediction grid of the stage, restricted to the entities of
            the variable's groups.
        feature_cols: Covariates retained by :meth:`_select_feature_columns`,
            i.e. those available at prediction time.
        ways: Materialization way of each covariate, decided on the prediction
            grid and imposed on both grids.
        X_train: Scaled features, rows carrying no observed covariate and rows
            of missing target already dropped.
        y_train: Scaled target, sharing ``X_train``'s index.
    """

    training: TrainingSet
    blocks: Dict[EntityKey, str]
    pred_grid: pd.Index
    feature_cols: Tuple[str, ...]
    ways: Dict[str, MaterializationWay]
    X_train: pd.DataFrame
    y_train: pd.Series


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
            intermediate stages but applies the very same origin filter as
            ``False`` — ``y_train`` holds observed anchors only, so the benefit
            of the cascade reaches the covariates and never the target;
            ``True`` also trains on the target's own earlier imputations and on
            the interpolated cells that replaced them on failure. **Never
            tested for truth**: ``'covariates_only'`` is truthy.
        fit_predict_order: Order in which variables are imputed,
            ``'frequency'`` (default) or ``'cv'``. Inert outside
            ``covariate_strategy='model'``. Under ``'cv'`` each variable is
            scored on the very training set it will then be fitted on — the
            covariates still available at prediction time, materialized the
            same way, on the mutualized rows of the **training** window and
            at the same scale — so the ranking and the estimation see one
            single problem per variable.
        cv: sklearn cross-validation strategy used by the ``'cv'`` order:
            ``None``, an int ``>= 2``, a splitter, or an iterable of splits.
            Resolved by ``check_cv`` at ``fit`` only.
        cv_scoring: Scoring of the ``'cv'`` order, higher is better.
        min_cv_train_size: Minimum number of scorable observations for a
            variable to be cross-validated. Below it — and likewise when no
            covariate at all survives the selection — the variable falls back
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
            Diagnostic only: nothing in the fit reads it — models train on
            the **training** window, whose scope it merely coincides with
            under the default ``training_scope=None``.
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

        - PHASE 5: stage execution — one model per (stage, variable),
          shared by its source-frequency groups, then one plan step per
          group.
        - PHASE 6: finalization — frozen plan and output attributes. The
          multi-frequency output belongs to ``transform``.

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
        raw_frequencies = self._detect_frequencies_robustly(X_work)

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

        # Accumulateur des avertissements de la phase 5 : ils sont émis en un
        # seul message en fin de phase, jamais un par variable et par étape
        self._warnings: List[str] = []

        # Contrainte d'agrégation, portée par un composant unique : il recale
        # les prédictions des étapes ET, injecté dans le matérialiseur, les
        # covariables interpolées — une seule implémentation, aucune dérive
        self._aggregation_constraint = AggregationConstraint(
            aggregation_constraint=self.aggregation_constraint,
            context='HighFrequencyImputer2',
        )

        # Instanciation du matérialiseur : producteur unique des features,
        # porteur des trois registres et de la contrainte d'agrégation
        self._covariate_materializer = CovariateMaterializer(
            covariate_strategy=self.covariate_strategy,
            covariate_fallback=self.covariate_fallback,
            covariate_eligibility=self.covariate_eligibility,
            interpolation_method=self.interpolation_method,
            interpolation_anchor=self.interpolation_anchor,
            aggregation_constraint=self.aggregation_constraint,
            aggregation_constraint_applier=self._aggregation_constraint,
        )
        # Purge des registres : un fit ne lit jamais le miroir d'un précédent
        self._covariate_materializer.reset()

        # Metteur à l'échelle des étapes : sans état, il rend un diviseur par
        # appel — jamais un scalaire figé pour tout un jeu
        self._stage_scaler = StageScaler(
            scale_features=self.scale_features,
            column_frequencies=self._detected_frequencies_by_column(),
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

            # Avertissement global si la fenêtre d'ENTRAÎNEMENT est vide :
            # sans lui, tous les entraînements échouent silencieusement un à un
            # et tout finit en repli par interpolation. C'est bien cette
            # fenêtre que lit le jeu d'entraînement, et non la stricte, qui
            # n'est plus qu'un attribut de diagnostic ; sous le scope par
            # défaut (training_scope=None) les deux coïncident, et un scope
            # élargi éteint légitimement l'avertissement
            if not bool(self.training_window_mask_.to_numpy(dtype=bool).any()):
                warnings.warn(
                    "The training window is empty: no model can be trained; "
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
            training_mask=self._training_mask_at,
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
        # PHASE 5 — Exécution des étapes
        # =================================================================
        # Le plan est l'état ajusté complet : initialisé vide, il est rempli
        # étape par étape, chaque étape étant construite PUIS exécutée au fil
        # de l'eau — l'étape k dépend des imputations de l'étape k-1
        self.imputation_plan_ = ImputationPlan()
        # Parcours des fréquences
        for stage_freq in self.frequency_progression_:
            # Exécution de l'imputation à l'étape donnée
            self._execute_stage(X_work, stage_freq)

        # Avertissements de la phase, agrégés en UN SEUL message
        if self._warnings:
            warnings.warn(
                f"{len(self._warnings)} imputation step(s) degraded during the "
                f"fit:\n  - " + "\n  - ".join(self._warnings),
                UserWarning,
            )

        # =================================================================
        # PHASE 6 — Finalisation
        # =================================================================
        # Le plan est figé et les attributs de sortie sont renseignés. La
        # sortie multi-fréquences relève du "transform", seul producteur de
        # frame
        self.imputation_provenance_ = self._provenance_tracker.get_provenance_matrix()

    # -------------------------------------------------------------------------
    # PHASE 5 — Exécution des étapes
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de résolution de l'estimateur d'une colonne
    def _estimator_for(self, column: str) -> Optional[BaseEstimator]:
        """Resolve the estimator of one column, under the dict form too.

        Args:
            column: Column being imputed.

        Returns:
            A fresh clone of the estimator covering that column, or None when
            none does — the interpolation fallback then applies.

        Examples:
            >>> imputer._estimator_for('a1')          # doctest: +SKIP
            LinearRegression()
        """
        # Initialisation de l'estimateur à la valeur de l'attribut
        estimator = self.estimator
        # Dictionnaire indexé par colonne : la colonne nommée, puis le défaut explicite
        if isinstance(estimator, dict):
            estimator = estimator.get(column, estimator.get('__default__'))
        # Cas où l'estimateur n'est pas spécifiée
        if estimator is None:
            return None
        # Copie indépendant de l'estimateur sklearn
        return clone(estimator)

    # Méthode auxiliaire d'extraction des entités portées par un index /!\ Est ce que ce n'est pas faisable avec une fonction de tsforecast/panel/utils ?
    @staticmethod
    def _index_entities(index: pd.Index) -> List[EntityKey]:
        """List the distinct entities of an index, in order of appearance.

        Args:
            index: ``DatetimeIndex`` of a time series, or panel
                ``MultiIndex`` ``(entity..., date)``.

        Returns:
            Normalized entity keys; ``[()]`` for a time series.
        """
        # Série temporelle : entité dégénérée unique
        if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
            return [()]

        # Panel : parcours des niveaux d'entité, dédupliqué sans tri
        levels = [index.get_level_values(level) for level in range(index.nlevels - 1)]
        seen: Dict[EntityKey, None] = {}
        for values in zip(*levels):
            seen.setdefault(normalize_entity_key(tuple(values)), None)
        return list(seen)

    # Méthode auxiliaire de restriction d'un index aux entités d'un groupe
    @staticmethod
    def _restrict_to_entities(
        index: pd.Index,
        entities: Optional[Sequence[EntityKey]],
    ) -> pd.Index:
        """Restrict an index to the rows of a group of entities.

        Args:
            index: Index to restrict.
            entities: Entity keys kept, None keeping everything (time series).

        Returns:
            The restricted index, in its original order.
        """
        # Série temporelle, ou groupe couvrant tout le panel
        if entities is None or not isinstance(index, pd.MultiIndex):
            return index

        # Appartenance ligne à ligne aux entités du groupe
        kept = {normalize_entity_key(entity) for entity in entities}
        levels = [index.get_level_values(level) for level in range(index.nlevels - 1)]
        rows = [
            normalize_entity_key(tuple(values)) in kept for values in zip(*levels)
        ]
        return index[np.asarray(rows, dtype=bool)]

    # Méthode auxiliaire de lecture d'un masque de fenêtre à la fréquence d'étape
    # /!\ Cette logique n'est pas déjà présente ailleurs ?
    def _stage_mask(
        self,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
        kind: Literal['strict', 'imputation', 'training'],
    ) -> pd.Series:
        """Read one window mask at the frequency of a stage.

        The ``kind`` is named by the caller, never defaulted.
        An entity the calculator omits — one without a valid fitted mask — is
        left unrestricted rather than silently losing every one of its rows.

        Args:
            X_work: Working frame, the fallback grid.
            stage_freq: Frequency of the stage, scalar or per entity.
            kind: Window read: ``'strict'``, ``'imputation'`` or
                ``'training'``.

        Returns:
            Boolean Series at the stage frequency, on a ``DatetimeIndex`` for
            a time series and on a ``(entity..., date)`` ``MultiIndex``
            otherwise.
        """
        # Calculateur non ajusté ou fréquence inconvertible : aucune
        # restriction, la garde de la phase 1 ayant déjà averti
        try:
            mask = self._imputation_window_calc.get_mask_at_frequency(
                stage_freq, kind=kind
            )
        except (ValueError, KeyError, TypeError) as error:
            self._log(f"[fit] {kind} mask unavailable ({error}); every row kept")
            return pd.Series(True, index=X_work.index)

        # Entités omises par le calculateur : rattachées sans restriction
        covered = set(self._index_entities(mask.index))
        missing = [
            entity for entity in self._index_entities(X_work.index)
            if entity not in covered
        ]
        if missing:
            extra = self._restrict_to_entities(X_work.index, missing)
            mask = pd.concat([mask, pd.Series(True, index=extra)])
        return mask

    # Méthode auxiliaire de la grille de prédiction d'une étape
    def _prediction_grid(
        self,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
        entities: Optional[Sequence[EntityKey]],
    ) -> pd.Index:
        """Build the prediction grid of a stage, restricted to one group.

        The grid spans the whole imputation window of the entities, anchor
        rows included: a period holding no observation is
        predicted like any other.

        Args:
            X_work: Working frame.
            stage_freq: Frequency of the stage.
            entities: Entities of the group, None for a time series.

        Returns:
            Index of the rows the step writes on.
        """
        # Masque d'imputation, "kind" nommé explicitement
        mask = self._stage_mask(X_work, stage_freq, kind='imputation')
        grid = mask.index[mask.to_numpy(dtype=bool)]
        return self._restrict_to_entities(grid, entities)

    # Méthode auxiliaire de liaison de fréquence des blocs
    @staticmethod
    def _block_binding(
        blocks: Dict[EntityKey, str],
        index: pd.Index,
    ) -> Union[str, Dict[EntityKey, str]]:
        """Shape the block frequencies the way :class:`StageScaler` reads them.

        A per-entity mapping needs an index carrying an entity level; a time
        series has none, and its single degenerate block is handed over as the
        scalar frequency instead.

        Args:
            blocks: Mapping entity -> block frequency.
            index: Grid the divisors are spread over.

        Returns:
            The mapping itself, or the single block frequency of a time
            series.
        """
        # Grille sans niveau d'entité : forme scalaire obligatoire
        if not isinstance(index, pd.MultiIndex) and len(blocks) == 1:
            return next(iter(blocks.values()))
        return dict(blocks)

    # Méthode auxiliaire d'ordonnancement des colonnes d'une étape
    def _order_columns(
        self,
        by_column: Dict[str, Dict[Tuple[str, str], Tuple[EntityKey, ...]]],
        X_work: pd.DataFrame,
        stage_label: str,
        stage_freq: Union[str, Dict[EntityKey, str]],
        stage_frame: pd.DataFrame,
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
    ) -> List[str]:
        """Order the imputable columns of one stage.

        The order is computed by :class:`VariableOrderer` only under
        ``covariate_strategy='model'``: it is the only modality where ranks 2
        and 3 of the precedence are reachable, hence the only one where the
        order changes a value. Otherwise the input column
        order is used and ``imputation_order_`` stays empty.

        Under the ``'cv'`` order, each column is scored on the very
        ``(X_train, y_train)`` pair phase 5c will fit it on, composed here by
        :meth:`_prepare_variable`: same covariate selection, same
        materialization ways, same mutualized rows and same scale. Scoring a
        raw view of ``X_work`` instead would rank a variable on covariates
        the selection may drop before the fit, on panel rows whose
        magnitudes are those of different frequencies, and on the strict
        window rather than the training one.

        One thing the ordering cannot see, and it is inherent: nothing of the
        stage has been imputed yet, so the sets are those of the start of the
        stage, whereas the variable of rank *k* will be fitted on what ranks
        *1..k-1* produced. The order defines the context that would define
        the order — the ranking is a heuristic, and phase 5c recomposes its
        own context rather than reusing these.

        Args:
            by_column: Imputable groups of the stage, keyed by column.
            X_work: Working frame.
            stage_label: Readable label of the stage.
            stage_freq: Frequency of the stage.
            stage_frame: Stage frame of 5a.
            freqs_by_column: Detected frequencies, keyed by column.

        Returns:
            The columns to impute, in processing order.
        """
        # Ordre d'entrée des colonnes : le défaut, sans effet sur les valeurs
        columns = [column for column in X_work.columns if column in by_column]
        if self.covariate_strategy != 'model' or len(columns) <= 1:
            return columns

        # Une spécification par colonnes : la fréquence retenue est la plus
        # basse de ses groupes, celle qui décide de son rang
        specs: Dict[str, VariableSpec] = {}
        # Parcours des colonnes
        for column in columns:
            groups = by_column[column]
            lowest = max(
                (source_freq for _column, source_freq in groups),
                key=get_frequency_order,
            )
            entities = tuple(
                sorted({e for group in groups.values() for e in group}, key=repr)
            )
            specs[column] = VariableSpec(
                name=column,
                frequency=lowest,
                entities=entities if self.is_panel_ else None,
            )

        # Ajustement paresseux de l'ordonnanceur : l'avertissement croisé qu'il
        # émet ne concerne que la validation croisée et n'aurait aucun sens
        # sous l'ordre 'frequency'
        if not hasattr(self._variable_orderer, 'cv_'):
            with warnings.catch_warnings():
                if self.fit_predict_order != 'cv':
                    warnings.simplefilter('ignore', UserWarning)
                self._variable_orderer.fit()

        # Jeux de scoring de l'ordre 'cv' : ceux-là mêmes que la 5c ajustera.
        # Rien n'est préparé sous l'ordre 'frequency', qui n'en lit aucun
        training_sets: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]] = None
        if self.fit_predict_order == 'cv':
            # Initialisation à un dictionnaire vide
            training_sets = {}
            # Parcours des colonnes
            for column in columns:
                # Union des entités des groupes, à l'identique de la 5c : la
                # grille de prédiction, donc la sélection des covariables, en
                # dépendent
                group_entities = sorted(
                    {
                        entity
                        for group in by_column[column].values()
                        for entity in group
                    },
                    key=repr,
                )
                prepared = self._prepare_variable(
                    column=column,
                    entities=group_entities if self.is_panel_ else None,
                    stage_freq=stage_freq,
                    X_work=X_work,
                    stage_frame=stage_frame,
                    freqs_by_column=freqs_by_column,
                )
                training_sets[column] = (prepared.X_train, prepared.y_train)

        # Les lignes scorées sont celles du jeu mutualisé, déjà restreintes à la fenêtre 'training'
        ordered = list(self._variable_orderer.order(
            specs,
            estimator=self.estimator,
            log=self._log if self.verbose else None,
            training_sets=training_sets,
        ))
        self.imputation_order_[stage_label] = list(ordered)
        return ordered

    # Méthode d'exécution d'une étape de fréquence
    def _execute_stage(
        self,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
    ) -> None:
        """Run one frequency stage of the progression.

        Three moves, in this exact order: the stage frame, the imputable
        variables and their order, then one pass per variable.

        Args:
            X_work: Working frame, after the additive transformer.
            stage_freq: Frequency of the stage.
        """
        # Étiquette lisible de l'étape, clé du plan et des journaux
        stage_label = self._stage_frequency_label(stage_freq)
        # Détection des fréquences
        freqs_by_column = self._detected_frequencies_by_column()

        # Groupes imputables : un par (colonne, fréquence source)
        groups = self._imputable_groups(stage_freq)
        if not groups:
            self._log(f"[fit] stage {stage_label}: no imputable variable")
            return

        # 5a. Frame d'étape : données d'origine, agrégations exactes et miroir
        stage_mask = self._stage_mask(X_work, stage_freq, kind='imputation')
        stage_frame = self._covariate_materializer.stage_frame(
            grid_index=stage_mask.index,
            stage_freq=stage_freq,
            detected_frequencies=freqs_by_column,
            source_data=X_work,
        )

        # 5b. Variables imputables à l'étape, regroupées puis ordonnées
        by_column: Dict[str, Dict[Tuple[str, str], Tuple[EntityKey, ...]]] = {}
        for group_key, entities in groups.items():
            by_column.setdefault(group_key[0], {})[group_key] = entities
        ordered = self._order_columns(
            by_column,
            X_work,
            stage_label,
            stage_freq=stage_freq,
            stage_frame=stage_frame,
            freqs_by_column=freqs_by_column,
        )

        # 5c. Une variable à la fois, un seul ajustement par (étape, variable)
        for column in ordered:
            self._fit_variable(
                column=column,
                groups=by_column[column],
                stage_freq=stage_freq,
                stage_label=stage_label,
                X_work=X_work,
                stage_frame=stage_frame,
                freqs_by_column=freqs_by_column,
            )

    # Méthode auxiliaire de composition du contexte d'ajustement d'une variable
    def _prepare_variable(
        self,
        *,
        column: str,
        entities: Optional[Sequence[EntityKey]],
        stage_freq: Union[str, Dict[EntityKey, str]],
        X_work: pd.DataFrame,
        stage_frame: pd.DataFrame,
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
    ) -> _VariableFit:
        """Compose everything one (stage, variable) needs before fitting.

        The single implementation of the mutualized training grid, the covariates that will still
        be there at prediction time, the way each of them is materialized,
        and the two scales. **The ordering of phase 5b and the fit of phase
        5c both go through it**, so the order ranks variables on the very
        sets they are then fitted on - not on a raw view of the data whose
        covariates the selection may never keep, whose panel rows carry
        incomparable magnitudes, and whose rows come from another window.

        Nothing is written to the three stores: the stage frames are views
        and every materialization here runs under ``record=False``. The
        prediction grid is deliberately NOT materialized - that production
        only feeds ``covariate_taint``, which is the caller's business.

        Args:
            column: Column being imputed.
            entities: Entities of the variable's groups, None for a time
                series. The prediction grid is restricted to them, so a
                caller passing a different set gets a different covariate
                selection.
            stage_freq: Frequency of the stage.
            X_work: Working frame.
            stage_frame: Stage frame of 5a.
            freqs_by_column: Detected frequencies, keyed by column.

        Returns:
            The :class:`_VariableFit` of that (stage, variable).
        """
        # Constructeur du jeu de données d'entraînement
        builder = self._training_set_builder
        # Matérialiseur des résultats des étapes précédentes
        materializer = self._covariate_materializer
        # Origines acceptées pour la target et les covariables
        eligible = ELIGIBLE_ORIGINS[self.impute_intermediate_frequencies]

        # Jeu mutualisé, sonde sans covariable : elle ne matérialise rien et
        # rend déjà les blocs, la cible brute et la grille d'entraînement
        probe = builder.build(
            column=column,
            feature_cols=(),
            stage_freq=stage_freq,
            detected_frequencies=freqs_by_column,
            source_data=X_work,
            eligible_origins=eligible,
        )
        blocks = dict(probe.blocks)

        # Grille de prédiction des entités concernées
        pred_grid = self._prediction_grid(X_work, stage_freq, entities)

        # Vues des deux fenêtres, chacune à sa fréquence : la grille
        # d'entraînement est au pas des blocs, la grille de prédiction au pas
        # de l'étape
        train_view = materializer.stage_frame(
            grid_index=probe.X.index,
            stage_freq=self._block_binding(blocks, probe.X.index) if blocks else stage_freq,
            detected_frequencies=freqs_by_column,
            source_data=X_work,
        )
        pred_view = stage_frame.reindex(pred_grid)

        # Sélection des covariables : non-vacuité sur les deux fenêtres, puis
        # éligibilité par entité
        candidates = [name for name in X_work.columns if name != column]
        feature_cols = self._select_feature_columns(candidates, train_view, pred_view)

        # Voie de matérialisation : décidée une seule, sur la grille de
        # prédiction puis imposée aux deux grilles : une covariable servie par
        # le repli au predict doit être préparée par le même chemin au fit,
        # même lorsque ses ancres suffiraient. Le matérialiseur ramène ensuite
        # la voie à ce que la fréquence de chaque bloc autorise
        ways = materializer.decide_ways(
            columns=feature_cols,
            grid_index=pred_grid,
            stage_freq=stage_freq,
            detected_frequencies=freqs_by_column,
        )

        # Jeu mutualisé complet, matérialisé par les voies imposées
        training = builder.build(
            column=column,
            feature_cols=feature_cols,
            stage_freq=stage_freq,
            detected_frequencies=freqs_by_column,
            source_data=X_work,
            eligible_origins=eligible,
            materialization=ways,
        )

        # Mise à l'échelle : diviseur de la cible par ligne, diviseurs des
        # features par bloc
        X_train, y_train = training.X, training.y
        if len(training) > 0:
            binding = self._block_binding(blocks, training.X.index)
            y_train = self._stage_scaler.apply(
                y_train,
                self._stage_scaler.target_divisor(
                    column,
                    source_freq=binding,
                    pred_freq=stage_freq,
                    index=training.X.index,
                    produced_freq=training.row_frequency,
                ),
            )
            if feature_cols:
                X_train = self._stage_scaler.apply(
                    X_train,
                    self._stage_scaler.feature_divisors(
                        columns=feature_cols,
                        column_frequencies=freqs_by_column,
                        ways=ways,
                        grid_freq=binding,
                        stage_freq=stage_freq,
                        index=training.X.index,
                    ),
                )

        # Lignes sans aucune covariable observée, puis cible manquante
        X_train, y_train = self._drop_empty_training_rows(X_train, y_train)
        usable = y_train.notna()
        X_train, y_train = X_train.loc[usable], y_train.loc[usable]

        # La division par un diviseur nommé autrement efface le nom de la
        # cible : il est rétabli, seul lien entre la série ajustée (ou scorée)
        # et la colonne qu'elle impute
        y_train = y_train.rename(column)

        return _VariableFit(
            training=training,
            blocks=blocks,
            pred_grid=pred_grid,
            feature_cols=feature_cols,
            ways=dict(ways),
            X_train=X_train,
            y_train=y_train,
        )

    # Méthode d'ajustement d'une variable à une étape
    def _fit_variable(
        self,
        *,
        column: str,
        groups: Dict[Tuple[str, str], Tuple[EntityKey, ...]],
        stage_freq: Union[str, Dict[EntityKey, str]],
        stage_label: str,
        X_work: pd.DataFrame,
        stage_frame: pd.DataFrame,
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
    ) -> None:
        """Fit one model for one (stage, variable), then execute its groups.

        The training set is mutualized across every entity observing the
        column: it does not depend on the source-frequency
        group, hence one single fit shared by the plan steps, which differ
        only by their ``source_frequency``, their entities and their
        rescaling.

        Args:
            column: Column being imputed.
            groups: Imputable groups of that column, keyed by
                ``(column, source frequency)``.
            stage_freq: Frequency of the stage.
            stage_label: Readable label of the stage.
            X_work: Working frame.
            stage_frame: Stage frame of 5a.
            freqs_by_column: Detected frequencies, keyed by column.
        """
        # Union des entités des groupes : la grille de prédiction, et à
        # travers elle la sélection des covariables, en dépendent
        group_entities = sorted(
            {entity for entities in groups.values() for entity in entities}, key=repr
        )
        entities = group_entities if self.is_panel_ else None

        # Contexte d'ajustement. Il est recomposé ici même lorsque la 5b l'a
        # déjà composé pour ordonner : depuis, les variables de rang inférieur
        # ont écrit dans le miroir, et c'est précisément ce que la cascade
        # cherche à exploiter
        prepared = self._prepare_variable(
            column=column,
            entities=entities,
            stage_freq=stage_freq,
            X_work=X_work,
            stage_frame=stage_frame,
            freqs_by_column=freqs_by_column,
        )
        training = prepared.training
        blocks = prepared.blocks
        pred_grid = prepared.pred_grid
        feature_cols = prepared.feature_cols
        ways = prepared.ways
        X_train, y_train = prepared.X_train, prepared.y_train

        # Souillure de la cible : lue par le filtre d'origine, jamais codée en
        # dur — le lot de l'axe 2 n'aura qu'à élargir ELIGIBLE_ORIGINS
        target_taint = origin_to_taint(
            max_origin([origin for origin in training.row_origin if origin is not None])
        )

        # Matérialisation de la grille de prédiction en mode rejeu des voies
        # retenues sur la grille d'entraînement. Elle ne sert qu'à la souillure
        # des covariables : aucune donnée d'entraînement n'en sort, ce qui est
        # la raison pour laquelle la 5b n'en a pas besoin pour ordonner
        pred_origins: Dict[str, CellOrigin] = {}
        if feature_cols and len(training) > 0:
            _X_pred, _ways, pred_origins = self._covariate_materializer.materialize(
                columns=feature_cols,
                grid_index=pred_grid,
                stage_freq=stage_freq,
                detected_frequencies=freqs_by_column,
                source_data=X_work,
                materialization=ways,
                record=False,
            )

        # Souillure des covariables : origines des cellules effectivement lues
        # sur les deux grilles, restreintes aux feature_cols du modèle — jamais
        # l'état global du registre
        covariate_taint = origin_to_taint(max_origin(
            [training.column_origins.get(name, 'observed') for name in feature_cols]
            + [pred_origins.get(name, 'observed') for name in feature_cols]
        ))

        # Ajustement unique, quel que soit le nombre de groupes
        model, is_fallback = self._fit_estimator(
            column, X_train, y_train, feature_cols, stage_label
        )
        # Logging
        self._log(
            f"[fit] stage {stage_label}, {column!r}: {len(y_train)} pooled "
            f"training rows from blocks {blocks}, {len(groups)} group(s)"
        )

        # Une étape de plan par groupe, partageant le modèle ajusté ci-dessus
        multi_frequency = len(groups) > 1
        # Parcours des groupes
        for group_key, group in groups.items():
            # Extraction des entités du groupe
            group_entities_or_none = group if self.is_panel_ else None
            # Construction de l'étape
            step = self._build_step(
                column=column,
                source_frequency=group_key[1],
                entities=group_entities_or_none,
                grid=self._restrict_to_entities(pred_grid, group_entities_or_none),
                stage_freq=stage_freq,
                stage_label=stage_label,
                model=model,
                feature_cols=() if is_fallback else feature_cols,
                materialization={} if is_fallback else dict(training.ways),
                covariate_taint='none' if is_fallback else covariate_taint,
                target_taint='none' if is_fallback else target_taint,
                is_fallback=is_fallback,
                training_blocks=blocks,
                multi_frequency=multi_frequency,
            )
            # Exécution, puis gel : un échec de prédiction dégrade l'étape en
            # repli, de sorte que le plan dise ce qui a réellement été fait
            degraded = self._execute_step(step, X_work=X_work, stage_freq=stage_freq)
            if degraded:
                step = replace(
                    step,
                    model=INTERPOLATE_FALLBACK,
                    feature_cols=(),
                    materialization={},
                    covariate_taint='none',
                    target_taint='none',
                    is_fallback=True,
                )
            self.imputation_plan_ = append_step(self.imputation_plan_, step)

    # Méthode auxiliaire d'ajustement de l'estimateur d'une variable
    def _fit_estimator(
        self,
        column: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: Tuple[str, ...],
        stage_label: str,
    ) -> Tuple[Any, bool]:
        """Fit the estimator of one variable, or fall back on interpolation.

        Args:
            column: Column being imputed.
            X_train: Scaled training features.
            y_train: Scaled training target.
            feature_cols: Covariates retained.
            stage_label: Readable label of the stage, for the messages.

        Returns:
            Tuple ``(model, is_fallback)``: the fitted estimator and False, or
            the ``INTERPOLATE_FALLBACK`` sentinel and True. Every failure
            message is accumulated, never emitted here.
        """
        # Extraction de l'estimateur
        estimator = self._estimator_for(column)
        # Aucun estimateur pour cette colonne : le repli est la règle, et
        # l'avertissement global de la phase 0 a déjà été émis
        if estimator is None:
            if self.estimator is not None:
                # Warning
                self._warnings.append(
                    f"{column!r} at stage {stage_label}: no estimator covers "
                    f"this column, interpolation fallback"
                )
            return INTERPOLATE_FALLBACK, True

        # Jeu d'entraînement inexploitable : aucune covariable, ou aucune ligne
        if not feature_cols or len(y_train) == 0:
            # Warning
            self._warnings.append(
                f"{column!r} at stage {stage_label}: empty training set "
                f"({len(y_train)} row(s), {len(feature_cols)} covariate(s)), "
                f"interpolation fallback"
            )
            return INTERPOLATE_FALLBACK, True

        # Échec d'ajustement : repli, jamais une exception qui ferait perdre
        # les autres variables de l'étape
        try:
            estimator.fit(X_train, y_train)
        except Exception as error:
            # Warning
            self._warnings.append(
                f"{column!r} at stage {stage_label}: estimator fit failed "
                f"({type(error).__name__}: {error}), interpolation fallback"
            )
            return INTERPOLATE_FALLBACK, True

        return estimator, False

    # Méthode auxiliaire de construction d'une étape de plan
    def _build_step(
        self,
        *,
        column: str,
        source_frequency: str,
        entities: Optional[Sequence[EntityKey]],
        grid: pd.Index,
        stage_freq: Union[str, Dict[EntityKey, str]],
        stage_label: str,
        model: Any,
        feature_cols: Sequence[str],
        materialization: Dict[str, Any],
        covariate_taint: Taint,
        target_taint: Taint,
        is_fallback: bool,
        training_blocks: Dict[EntityKey, str],
        multi_frequency: bool,
    ) -> ImputationStep:
        """Freeze one plan step of a (stage, variable, source frequency) group.

        Args:
            column: Column imputed by the step.
            source_frequency: Detected frequency of the column for the group.
            entities: Entities of the group, None for a time series.
            grid: Prediction grid of the group, read by the ``'calendar'``
                scale mode.
            stage_freq: Frequency of the stage.
            stage_label: Readable label of the stage.
            model: Estimator shared by every group of the variable, or the
                interpolation sentinel.
            feature_cols: Covariates of the model, empty for a fallback.
            materialization: Way retained for each covariate.
            covariate_taint: Worst covariate taint of the step.
            target_taint: Worst target taint of the step.
            is_fallback: Whether the step interpolates instead of predicting.
            training_blocks: Composition of the mutualized training set.
            multi_frequency: Whether the column carries several source
                frequencies at this stage, which is what makes the frequency
                part of the registry key.

        Returns:
            The frozen :class:`ImputationStep`.
        """
        # Facteur d'échelle du groupe. La cible ayant été mise à l'échelle
        # ligne à ligne, les prédictions sortent déjà au pas de l'étape :
        # "scale_factor" et "fit_scale_factor" coïncident et le report vaut 1.0
        try:
            scale = self._stage_scaler.fit_scale_factor(
                column,
                source_freq=source_frequency,
                pred_freq=stage_freq,
                index=grid,
            )
        except (ValueError, KeyError, TypeError):
            scale = 1.0

        # Clé de registre : la fréquence n'entre dans la clé que lorsque les
        # entités divergent sur la fréquence de cette colonne
        var_key = (column, source_frequency) if multi_frequency else column

        return ImputationStep(
            pred_freq_label=stage_label,
            pred_freq=stage_freq,
            var_key=var_key,
            var_name=column,
            model=model,
            feature_cols=tuple(feature_cols),
            scale_factor=scale,
            fit_scale_factor=scale,
            source_frequency=source_frequency,
            entities=to_entity_tuple(entities),
            covariate_taint=covariate_taint,
            target_taint=target_taint,
            materialization=materialization,
            is_fallback=is_fallback,
            interpolation_method=self._covariate_materializer.resolve_method(column),
            interpolation_anchor=self._covariate_materializer.resolve_anchor(column),
            training_blocks=training_blocks,
        )

    # Méthode unique d'exécution d'une étape du plan
    def _execute_step(
        self,
        step: ImputationStep,
        *,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
    ) -> bool:
        """Execute one frozen plan step.

        The step is already frozen: this method decides nothing and only
        writes — values, rescaling, provenance and the three stores. ``fit``
        builds the step then calls it; ``transform`` will replay the plan
        through this very method, which is what makes the two paths identical
        by construction.

        Args:
            step: Frozen step to execute.
            X_work: Working frame the values are read from.
            stage_freq: Frequency of the stage.

        Returns:
            True when the step had to fall back on interpolation at execution
            time — a prediction failure — so the caller can degrade the plan
            step accordingly. False otherwise.
        """
        # Extraction du matérisaliseur des étapes précédentes
        materializer = self._covariate_materializer
        # Extraction du nom de la colonne
        column = step.var_name
        # Détection des fréquences de la colonne (éventuellement pour différentes entités)
        freqs_by_column = self._detected_frequencies_by_column()

        # Grille du groupe : la fenêtre d'imputation de ses entités, ancres comprises
        grid = self._prediction_grid(X_work, stage_freq, step.entities)
        if len(grid) == 0:
            return False

        # Production des valeurs : modèle, ou repli d'interpolation
        values: Optional[pd.Series] = None
        if not step.is_fallback:
            values = self._predict_step(step, grid, X_work, freqs_by_column, stage_freq)
        degraded = values is None and not step.is_fallback

        if values is None:
            # « Le repli matérialise » : "interpolate_column" alimente déjà les
            # trois registres, avec l'origine 'interpolated'
            values = materializer.interpolate_column(
                column=column,
                grid_index=grid,
                stage_freq=stage_freq,
                detected_frequencies=freqs_by_column,
                source_data=X_work,
            )
            origin: CellOrigin = 'interpolated'
            provenance = ProvenanceType.INTERPOLATED
        else:
            origin = 'model'
            provenance = resolve_model_provenance(
                step.covariate_taint, step.target_taint
            )

        # Recalage aux totaux de la fréquence source du groupe : annuels pour
        # une entité annuelle, trimestriels pour une entité trimestrielle
        observations = X_work[column].reindex(
            self._restrict_to_entities(X_work.index, step.entities)
        )
        values, _rescaled_mask = self._aggregation_constraint.rescale(
            values, observations, step.source_frequency, column=column,
            grid_freq=stage_freq,
        )

        # Ecriture : cellules effectivement produites
        written = values[values.notna()]
        if written.empty:
            return degraded

        # Marquage de provenance : identique pour les cellules recalées et non
        # recalées, lignes d'ancres comprises — le recalage ne change aucune
        # provenance. Une ligne d'ancre
        # ré-exprimée ne reste jamais original : elle ne porte plus
        # l'observation
        self._provenance_tracker.mark_imputed(column, written.index, provenance)

        # Mise à jour des trois registres, y compris en repli
        materializer.record_production(
            column,
            written,
            pd.Series(origin, index=written.index),
            pd.Series(
                self._stage_frequency_of(stage_freq, written.index),
                index=written.index,
            ),
        )
        return degraded

    # Méthode auxiliaire de prédiction d'une étape à modèle
    def _predict_step(
        self,
        step: ImputationStep,
        grid: pd.Index,
        X_work: pd.DataFrame,
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
        stage_freq: Union[str, Dict[EntityKey, str]],
    ) -> Optional[pd.Series]:
        """Produce the predictions of one model step on its grid.

        Args:
            step: Frozen step, holding the model and the ways to replay.
            grid: Prediction grid of the group.
            X_work: Working frame.
            freqs_by_column: Detected frequencies, keyed by column.
            stage_freq: Frequency of the stage.

        Returns:
            The predictions at the scale of the stage, or None when the
            prediction failed — the caller then falls back on interpolation.
        """
        # Covariables produites en mode rejeu des voies de l'étape : c'est le
        # seul chemin autorisé pour X_pred.
        # "record=False" : les registres ne portent qie ce qui a été imputé,
        # jamais ce qui a été préparé comme feature. Y inscrire la
        # matérialisation d'une covariable écraserait l'imputation que son
        # propre modèle vient d'écrire, et rendrait le résultat dépendant de
        # l'ordre de traitement
        X_pred, _ways, _origins = self._covariate_materializer.materialize(
            columns=step.feature_cols,
            grid_index=grid,
            stage_freq=stage_freq,
            detected_frequencies=freqs_by_column,
            source_data=X_work,
            materialization=step.materialization,
            record=False,
        )

        # Mise à l'échelle des features à la fréquence source du groupe, puis
        # prédiction
        try:
            divisors = self._stage_scaler.feature_divisors(
                columns=step.feature_cols,
                column_frequencies=freqs_by_column,
                ways=step.materialization,
                grid_freq=stage_freq,
                stage_freq=stage_freq,
                index=grid,
            )
            predictions = step.model.predict(
                self._stage_scaler.apply(X_pred, divisors)
            )
        except Exception as error:
            # Warning
            self._warnings.append(
                f"{step.var_name!r} at stage {step.pred_freq_label}: prediction "
                f"failed ({type(error).__name__}: {error}), interpolation fallback"
            )
            return None

        # Report d'échelle du modèle vers l'étape : le rapport vaut 1.0 tant
        # que le modèle est ajusté pour l'étape courante, ce qui est toujours
        # le cas hors réutilisation inter-étapes
        values = pd.Series(
            np.asarray(predictions, dtype=float).ravel(), index=grid, name=step.var_name
        )
        return self._stage_scaler.invert(
            self._stage_scaler.apply(values, step.scale_factor), step.fit_scale_factor
        )

    # Méthode auxiliaire de la fréquence d'étape ligne à ligne
    @staticmethod
    def _stage_frequency_of(
        stage_freq: Union[str, Dict[EntityKey, str]],
        index: pd.Index,
    ) -> List[str]:
        """Give the stage frequency of each row of a grid.

        Args:
            stage_freq: Frequency of the stage, scalar or per entity.
            index: Grid the frequencies are read on.

        Returns:
            One frequency string per row of ``index``.
        """
        # Fréquence unique : la même pour toutes les lignes
        if not isinstance(stage_freq, dict):
            return [normalize_frequency(stage_freq, return_format='base')] * len(index)

        # Fréquence par entité : lecture ligne à ligne
        levels = [index.get_level_values(level) for level in range(index.nlevels - 1)]
        return [
            normalize_frequency(
                stage_freq[normalize_entity_key(tuple(values))], return_format='base'
            )
            for values in zip(*levels)
        ]

    # Méthode auxiliaire du masque d'entraînement lu à la fréquence des blocs
    def _training_mask_at(
        self,
        frequencies: Dict[EntityKey, str],
    ) -> Optional[pd.Series]:
        """Read the ``'training'`` window mask at one frequency per block.

        Injected into :class:`TrainingSetBuilder`, which therefore receives a
        callable and never the calculator: it cannot read a window other than
        the training one, whose ``kind`` is named here once and for all.

        Args:
            frequencies: Mapping entity -> block frequency ``f_block(e)``.

        Returns:
            Boolean mask at those frequencies, on a ``DatetimeIndex`` for a
            time series and on a ``(entity..., date)`` MultiIndex otherwise;
            None when no window is computable — an unfitted calculator, or a
            block frequency it cannot convert to — which the builder reads as
            "no restriction at all".
        """
        # Série temporelle : le calculateur attend la fréquence scalaire du
        # bloc unique, une liaison par entité n'ayant pas de sens sans entité
        binding: Union[str, Dict[EntityKey, str]] = frequencies
        if not self.is_panel_ and isinstance(frequencies, dict):
            binding = next(iter(frequencies.values()))
        try:
            return self._imputation_window_calc.get_mask_at_frequency(
                binding, kind='training'
            )
        except (ValueError, KeyError, TypeError) as error:
            # Logging
            self._log(f"[fit] training mask unavailable ({error}); every row kept")
            return None

    # Méthode auxiliaire de détection tolérante des fréquences
    def _detect_frequencies_robustly(
        self,
        X_work: pd.DataFrame,
    ) -> Dict[Union[str, tuple], Optional[str]]:
        """Detect the frequency of every (entity, column) pair, undetectable ones included.

        ``detect_frequency`` raises on a series holding fewer than two
        observations — an entirely NaN column, or an entity observing a column
        once — which would abort the whole fit over one unusable pair. The
        classification already has a place for such a pair: ``None``, which
        :attr:`_undetected_frequencies_` collects and leaves out. This method
        is the crossing point between the two conventions: the global
        detection first, and a pair-by-pair retry when it refuses.

        Args:
            X_work: Working frame, time series or panel.

        Returns:
            Mapping from column name (time series) or ``(entity..., column)``
            tuple (panel) to the detected frequency, ``None`` where none could
            be detected.
        """
        # Chemin nominal : une seule détection sur tout le jeu
        try:
            return detect_frequency(data=X_work)
        except (ValueError, TypeError) as error:
            self._log(
                f"[fit] dataset-wide frequency detection refused ({error}); "
                f"falling back on a pair-by-pair detection"
            )

        # Repli : couple par couple, une paire indétectable valant None
        detected: Dict[Union[str, tuple], Optional[str]] = {}
        for entity, _mask, block in iter_entity_blocks(X_work):
            prefix = normalize_entity_key(entity) if self.is_panel_ else ()
            for column in block.columns:
                key = (*prefix, column) if prefix else column
                try:
                    detected[key] = detect_frequency(data=block[column])
                except (ValueError, TypeError):
                    detected[key] = None
        return detected

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
