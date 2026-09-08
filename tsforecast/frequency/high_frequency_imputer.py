"""Mixed-frequency imputer """
# Importation des modules
# Modules de base
import warnings
from collections import OrderedDict
from contextlib import contextmanager
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
    resolve_aggregation_constraint,
    validate_aggregation_constraint,
)
from .covariate_materializer import CovariateMaterializer
from .imputation_plan import (
    INTERPOLATE_FALLBACK,
    ImputationPlan,
    ImputationStep,
    MaterializationWay,
    append_step,
    to_entity_tuple,
)
from .imputation_plan import _scale_equal as _scale_factors_equal
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
)
from .stage_scaler import ScaleMode, StageScaler
from .target_frequency_validator import TargetFrequencyValidator
from .training_set_builder import (
    TrainingSet,
    TrainingSetBuilder,
    split_training_index,
)
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
    'unanchored_pairs_',
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


# Contexte d'ajustement d'une variable à une étape
@dataclass(frozen=True)
class _VariableFit:
    """Everything one (stage, variable) needs before an estimator is fitted.

    Composed by :meth:`HighFrequencyImputer._prepare_variable` and consumed
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
class HighFrequencyImputer(XYPanelTimeSeriesTransformer):
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
      ``covariate_strategy='model'`` changes **no** final value: materialization
      ranks 2 and 3 are then structurally out of reach — a covariate is served
      from its own observations before any register is read — so the stage
      carry that gives ``'covariates_only'`` its whole point never happens. The
      intermediate stages still cost compute and still show up in the
      multi-frequency output when ``keep_lower_frequencies=True``, and that is
      all they do. ``True``, by contrast, has an effect under **every**
      strategy: it changes ``y_train`` itself.
    - ``covariate_fallback`` is inert outside ``covariate_strategy='model'``.
    - ``fit_predict_order`` — and with it ``cv``, ``cv_scoring`` and
      ``min_cv_train_size`` as ordering devices — is inert outside
      ``covariate_strategy='model'``.
    - ``training_coverage_threshold`` without ``training_scope`` is inert:
      the training window then follows ``imputation_scope`` and
      ``coverage_threshold``.

    **What the parameter space no longer expresses**: the legacy mode "one
    single fit, reused at the following stages with that stage's scale
    factor" — the ``False`` branch of a cascade refitting switch — is
    **abandoned**, switch included. It bought compute
    at the price of a fit/predict asymmetry that was never mastered. Should
    the need come back it returns as an **internal optimization** —
    memoizing a model whose training set and materialization ways have not
    changed between two stages — never as public semantics. It is not to be
    confused with the single fit per (stage, variable) already in place,
    which shares a model between the plan steps of **one** stage only and
    never across stages.

    ``keep_lower_frequencies`` is a **pure display parameter**: it governs how
    the frequency levels of the output are stacked, never the logic — the
    values of the target level are the same either way. The levels stacked are
    the stages of the progression, so under
    ``impute_intermediate_frequencies=False`` there is **no intermediate level
    to stack**: the output carries the target level and nothing else.
    The frequency level sits on the entity side of the index, the shape:
    ``(frequency, date)`` on a time series, ``(entity..., 'frequency',
    'date')`` on a panel.

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
            ``'covariates_only'`` and ``True`` build the very same stage plan
            and differ only by that origin filter; ``False`` and
            ``'covariates_only'`` apply the very same filter and differ only
            by the plan. ``'covariates_only'`` is inert on the final values
            outside ``covariate_strategy='model'``, ``True`` never is (see
            the inert combinations above). **Not orthogonal to**
            ``aggregation_constraint`` under ``True``: an earlier stage's
            imputation falling on the date of an anchor — the 31st of
            December is both a quarter end and a year end — is dropped from
            ``y_train`` under ``'sum'``, the rescaling making it an exact
            linear combination of the other rows, and kept under ``None``,
            where the training index then gains a ``'frequency'`` level. The
            link is algebraic, not conventional, and is assumed as such.
            Under ``False`` and ``'covariates_only'`` no coincidence is
            possible and ``aggregation_constraint`` has no effect whatsoever
            on the composition of ``y_train``.
        impute_unobserved_entities: Whether an entity that never observes an
            imputable column may receive a complete imputation of it, learned
            on the other entities of the panel. ``False`` (default) leaves
            such a pair out of every prediction grid: its cells stay NaN and
            ``ORIGINAL``. ``True`` makes the pair imputable at **every
            stage of its entity's progression**, through one dedicated plan
            step per stage whose ``source_frequency`` is ``None``. It is the
            target-side counterpart of ``covariate_eligibility``, and it is
            **independent of axis 2**: one single rule, whose consequences
            follow the progression — under ``False`` that progression holds
            the target frequency alone, which is the degenerate case, and
            under the two other modalities the intermediate stages fill the
            mirror and the intermediate levels of the output that the entity
            would otherwise leave with a hole. The value reached at the target
            frequency is the same either way. And
            ``frequency_progression_`` stays untouched throughout — the pair
            travels the stages, it never adds one. Three semantic differences
            these cells carry, and which ``MODEL_UNANCHORED`` reports: no
            anchor, hence **no rescaling** whatever
            ``aggregation_constraint`` says, and **no divisor** — they are
            free predictions, not disaggregations of an observed total. This
            holds at every stage, the intermediate ones included: a cell
            anchored on an earlier prediction of its own column, for its own
            entity, is anchored on nothing. Rescaling the finer level onto the
            coarser one would moreover give these cells a cross-level
            coherence the anchored ones do not have — two levels of an
            anchored entity are each rescaled onto the shared observed total,
            never onto one another. The
            entity contributes nothing to the training set either : it has no true value to bring. Its only
            failure path is the absence of usable covariates on the target
            grid: interpolation cannot be the fallback of an entity with
            nothing to interpolate, so the cells stay NaN and ``ORIGINAL``
            and a single aggregated warning names the pairs at the end of
            ``fit``.
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
            ``'__default__'`` key. Beyond the rescaling of the predictions, it
            also governs the composition of ``y_train`` under
            ``impute_intermediate_frequencies=True``, and therefore **ceases
            to be orthogonal** to it: under ``'sum'`` an earlier stage's
            imputation coinciding with an anchor is dropped — the rescaling
            imposes that the sub-periods sum to the observed total, so the
            coarser row is the sum of the finer ones and keeping both counts
            the period total twice; under ``None`` nothing is rescaled, the
            collinearity is broken, every cell is kept and the training index
            gains a ``'frequency'`` level on the entity side. The read is
            per column, the dict form included. Under ``False`` and
            ``'covariates_only'`` the effect is nil: ``y_train`` holds
            anchors only and no coincidence can occur.
        keep_lower_frequencies: Pure display parameter, see above.
        on_frequency_mismatch: ``'error'`` (default) or ``'warn'`` when
            ``target_frequency`` is higher than the data allows.
        restore_original_values: If True, ``inverse_transform`` refills every
            cell that was non-NaN in the input with its exact original value.
        time_col: Name of the time column when it is not in the index.
        panel_cols: Columns identifying the panel entities on a flat frame.
        verbose: If True, print progress messages prefixed
            ``[HighFrequencyImputer]``.

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
        frequency_progression_: Ordered list of stage frequencies. **Not
            changed** by ``impute_unobserved_entities``: an unanchored pair
            has no source frequency, so it joins no frequency set and adds no
            stage.
        unanchored_pairs_: Tuple of the ``(entity..., column)`` pairs actually
            imputed without any anchor under
            ``impute_unobserved_entities=True``. Always written, empty under
            the default; a subset of the pairs no frequency could be detected
            for.
        imputation_order_: Variable order per stage. **Empty outside**
            ``covariate_strategy='model'``.
        imputation_cv_scores_: ``{stage_label: {variable: cv_score}}`` — the
            cross-validated scores that decided ``imputation_order_``. Empty
            outside ``covariate_strategy='model'`` **and**
            ``fit_predict_order='cv'``; a stage whose order fell back entirely
            to frequency carries no entry. Exposed for tracking.
        provenance_statistics_: Counts and percentages per ``ProvenanceType``,
            ``{'overall': {...}, column: {...}}`` — the output of
            :meth:`ImputationProvenanceTracker.compute_statistics` computed on
            the last pass, kept so it need not be rebuilt. Exposed for
            tracking (see :func:`tsforecast.tracking.imputation_metrics`).
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
            window. Unlike earlier designs where ``imputation_window_``
            carried the strict bounds, here each attribute carries the bounds
            of its own mask, and
            ``strict_window_mask_`` is the sole holder of the strict window.
        imputation_provenance_: Provenance matrix. Written by ``fit``, then
            overwritten by every ``transform``: it always describes the
            last pass, which is what ``inverse_transform`` undoes. It follows
            the shape of the output — stacked on the frequency level under
            ``keep_lower_frequencies=True``, flat otherwise — and ``fit``
            purges the transform's copy in its first phase.
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
        >>> imputer = HighFrequencyImputer(
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
        impute_unobserved_entities: bool = False,
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
            'impute_unobserved_entities': impute_unobserved_entities,
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
        self.impute_unobserved_entities = impute_unobserved_entities
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
            print(f"[HighFrequencyImputer] {message}")

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
    # Prédicat d'ajustement consulté par "check_is_fitted"
    def __sklearn_is_fitted__(self) -> bool:
        """Report whether ``fit`` has completed.

        ``check_is_fitted`` consults this predicate before its suffix scan, so
        every ``check_is_fitted(self)`` call — those the parent already makes
        in ``transform`` and ``inverse_transform`` included — becomes strict
        without an override here.

        The attribute list is explicit. The default sklearn convention
        — any attribute ending in ``_`` — would be satisfied by the parent
        class, which sets ``is_panel_``, ``n_features_`` and
        ``feature_names_`` before ``_fit`` runs: an interrupted fit would then
        look fitted.

        Returns:
            True once all fitted attributes are present.
        """
        # Liste explicite plutôt que la convention du suffixe
        return all(hasattr(self, attr) for attr in _FITTED_ATTRIBUTES)

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
        # Couples apparus au transform : aucune fréquence de fit ne les porte,
        # la détection faite sur les données du transform est leur seule
        # source. Les couples du fit, eux, ne sont jamais recouverts
        pairs: Dict[Union[str, tuple], str] = {
            **getattr(self, '_transform_frequencies', {}),
            **self.detected_frequencies_,
        }

        # Cas des séries temporelles : les clés sont déjà des noms de colonnes
        if not self.is_panel_:
            return dict(pairs)

        # Cas du panel : regroupement par colonne, puis repli sur la forme
        # scalaire quand toutes les entités s'accordent
        by_column: Dict[str, Dict[EntityKey, str]] = {}
        for key, freq in pairs.items():
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
    ) -> Dict[Tuple[str, Optional[str]], Tuple[EntityKey, ...]]:
        """Group the imputable ``(entity, column)`` pairs by source frequency.

        Each ``(column, f_var)`` group yields one plan step at the stage,
        all of them sharing the model fitted on the mutualized training set.

        Under ``impute_unobserved_entities=True`` a further group
        ``(column, None)`` gathers the entities that never observe the column,
        at every stage binding them. This is the **single** entry point of
        that capability: the classification is left untouched, which is
        exactly what keeps ``variable_categories_`` and
        ``frequency_progression_`` identical with and without the parameter --
        such a pair has no source frequency, so it joins no frequency set and,
        travelling the stages, adds none.

        Args:
            prediction_frequency: Frequency of the stage.

        Returns:
            Mapping from ``(column, source frequency)`` to the tuple of
            entity keys of the group, each tuple sorted for determinism. On a
            time series the entity key is ``()``. The source frequency is
            ``None`` for an unanchored group.

        Examples:
            >>> imputer._imputable_groups('M')          # doctest: +SKIP
            {('v', 'Y'): (('FR',),), ('v', None): (('IT',),)}
        """
        # Regroupement des couples imputables par (colonne, fréquence source)
        groups: Dict[Tuple[str, Optional[str]], List[EntityKey]] = {}
        categories = self._classify_variables_at_frequency(prediction_frequency)
        for key in categories['impute']:
            entity, column = split_variable_key(key)
            source_freq = normalize_frequency(
                self.detected_frequencies_[key], return_format='base'
            )
            groups.setdefault((column, source_freq), []).append(entity)

        # Groupes sans ancre, ajoutés après les groupes ancrés : l'ordre du
        # plan reste celui des fréquences sources, la nouveauté en queue
        for entity, column in self._unanchored_pairs_at(prediction_frequency):
            groups.setdefault((column, None), []).append(entity)

        # Tri des entités de chaque groupe : l'ordre du plan ne doit dépendre
        # ni de l'ordre des colonnes ni de celui des lignes en entrée
        return {
            group_key: tuple(sorted(entities, key=repr))
            for group_key, entities in groups.items()
        }

    # Méthode auxiliaire de lecture de la fréquence cible d'une entité
    def _entity_target_frequency(self, entity: Optional[EntityKey]) -> Optional[str]:
        """Read the target frequency bound to one entity.

        Args:
            entity: Entity key, None on a time series.

        Returns:
            The normalized target frequency of that entity, or None when the
            binding does not name it.

        Examples:
            >>> imputer._entity_target_frequency(('FR',))   # doctest: +SKIP
            'M'
        """
        # Les deux formes de la cible, scalaire et par entité, lues ici seulement
        target = self.effective_target_frequency_
        frequency = target.get(entity) if isinstance(target, dict) else target
        if frequency is None:
            return None
        return normalize_frequency(frequency, return_format='base')

    # Méthode auxiliaire de sélection des couples imputables sans aucune ancre
    def _unanchored_pairs_at(
        self,
        prediction_frequency: Union[str, Dict[EntityKey, str]],
    ) -> List[Tuple[EntityKey, str]]:
        """Select the never-observed pairs imputable at one stage.

        A pair the fit could detect no frequency for joins every stage of its
        entity's progression, not only the last: the rule is the general one,
        of which the single stage of
        ``impute_intermediate_frequencies=False`` is the degenerate case.
        Reading the binding rather than the target frequency is what keeps the
        parameter independent of axis 2 -- one rule, whose consequences follow
        the progression, instead of a behavior that changes with the modality.

        Two conditions gate a pair, beyond ``impute_unobserved_entities``:

        1. the stage binds the entity, which a stage its target-frequency
           group does not travel through never does;
        2. at least one other entity observes the column, without which the
           mutualized training set would be empty and the step would have no
           model to share.

        No condition is put on the level the entity reaches: the cells
        produced at an intermediate stage are anchored on nothing, exactly
        like those of the last one -- being anchored on an earlier prediction
        of itself is not being anchored -- so every stage rescales nothing and
        emits ``MODEL_UNANCHORED``.

        Args:
            prediction_frequency: Frequency of the stage, scalar or per
                entity.

        Returns:
            List of ``(entity key, column)`` pairs, in the order of
            ``_undetected_frequencies_``. Empty under the default parameter,
            and empty on a time series, which has no other entity to learn
            from.

        Examples:
            >>> imputer._unanchored_pairs_at('M')       # doctest: +SKIP
            [(('IT',), 'v')]
        """
        # Capacité fermée par défaut, et sans objet sur une série temporelle :
        # l'imputation sans ancre est apprise sur les AUTRES entités
        if not self.impute_unobserved_entities or not self.is_panel_:
            return []

        # Parcours des couples jamais observés
        pairs: List[Tuple[EntityKey, str]] = []
        for key in self._undetected_frequencies_:
            entity, column = split_variable_key(key)

            # Fréquence de prédiction liée à l'entité par l'étape
            if isinstance(prediction_frequency, dict):
                pred_freq = prediction_frequency.get(entity)
            else:
                pred_freq = prediction_frequency
            if pred_freq is None:
                continue

            # Colonne observée par au moins une autre entité : sans elle, le
            # jeu mutualisé est vide et il n'y a aucun modèle à partager
            if not self._column_frequencies_by_entity(column):
                continue

            pairs.append((entity, column))
        return pairs

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
    # Méthode auxiliaire de construction de la progression d'un groupe d'entités
    def _group_frequency_progression(
        self,
        entities: Optional[Sequence[EntityKey]],
        target_frequency: str,
    ) -> List[str]:
        """Build the stage frequencies of one target-frequency group.

        Args:
            entities: Entity keys of the group, ``None`` for a time series —
                the degenerate one-entity case, where nothing is filtered.
            target_frequency: Target frequency shared by the group.

        Returns:
            Stage frequencies in base form, from the lowest to the highest,
            the target frequency last. The lowest frequency of ``F`` is never
            a stage: nothing is imputable at it.

        Examples:
            >>> imputer._group_frequency_progression(None, 'M')  # doctest: +SKIP
            ['Q', 'M']
        """
        # Fréquence cible du groupe, forme canonique de comparaison
        f_target = normalize_frequency(target_frequency, return_format='base')

        # Ensemble des fréquences des couples (entité, colonne) imputables du
        # périmètre du groupe, plus la fréquence cible
        admissible = None if entities is None else set(entities)
        categories = self._classify_variables_at_frequency(
            self.effective_target_frequency_
        )
        frequencies = {f_target}
        for key in categories['impute']:
            entity, _ = split_variable_key(key)
            if admissible is not None and entity not in admissible:
                continue
            frequencies.add(
                normalize_frequency(
                    self.detected_frequencies_[key], return_format='base'
                )
            )

        # Fréquence la plus basse de l'ensemble précédemment défini : jamais une étape, rien n'y étant à
        # imputer. "get_frequency_order" croît quand la fréquence baisse
        lowest = max(frequencies, key=get_frequency_order)

        # Tri de la plus basse à la plus haute, borné des deux côtés :
        # strictement au-dessus de la plus basse source, au plus à la cible
        stages = [
            frequency
            for frequency in sorted(
                frequencies, key=get_frequency_order, reverse=True
            )
            if is_higher_frequency(frequency, lowest)
            and (frequency == f_target or is_higher_frequency(f_target, frequency))
        ]

        # Garantie que la cible est le dernier élément de la progression
        if f_target in stages:
            stages.remove(f_target)
        stages.append(f_target)
        return stages

    # Méthode auxiliaire de construction de la progression de fréquences
    def _build_frequency_progression(self) -> List[Union[str, Dict[EntityKey, str]]]:
        """Build the ordered list of stage frequencies.

        Under ``impute_intermediate_frequencies is False`` the progression is
        the target frequency alone: the imputed variable jumps straight from
        its own frequency to the target, with no intermediate stage. Under the
        two other modalities the progression is complete and **identical**:
        ``'covariates_only'`` and ``True`` share the very same stage plan and
        differ only by the origin filter of ``y_train``, held by
        ``ELIGIBLE_ORIGINS``.

        On a panel the progression is computed **per group of entities sharing
        the same target frequency**, then merged into global stages: an entity
        whose group does not travel through a stage is simply absent from that
        stage's binding, which leaves it unclassified — hence never imputed,
        and never written — at that stage.

        Returns:
            Ordered list of stage frequencies, the target frequency last. A
            time series yields a list of strings; a panel yields a list of
            ``{entity: frequency}`` bindings.

        Examples:
            >>> imputer.frequency_progression_          # doctest: +SKIP
            [{('FR',): 'Q', ('DE',): 'Q'}, {('FR',): 'M'}]
        """
        # Modalité sans étape intermédiaire : une seule étape, la cible.
        if self.impute_intermediate_frequencies is False:
            return [self.effective_target_frequency_]

        target = self.effective_target_frequency_

        # Cas dégénéré de la série temporelle : un seul groupe, cible scalaire
        if not isinstance(target, dict):
            return list(self._group_frequency_progression(None, target))

        # Panel : un groupe d'entités par fréquence cible partagée
        groups: Dict[str, List[EntityKey]] = {}
        for entity, frequency in target.items():
            groups.setdefault(
                normalize_frequency(frequency, return_format='base'), []
            ).append(entity)
        per_group = {
            f_target: self._group_frequency_progression(entities, f_target)
            for f_target, entities in groups.items()
        }

        # Fusion en étapes globales : union ordonnée des fréquences d'étape,
        # chaque étape ne liant que les entités dont le groupe la traverse
        stage_order = sorted(
            {stage for stages in per_group.values() for stage in stages},
            key=get_frequency_order,
            reverse=True,
        )
        return [
            {
                entity: stage
                for f_target, entities in groups.items()
                if stage in per_group[f_target]
                for entity in entities
            }
            for stage in stage_order
        ]

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
        - PHASE 6: finalization — frozen plan, provenance matrix and
          ``unanchored_pairs_``. The multi-frequency output belongs to
          ``transform``.

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
        # "CovariateMaterializer.eligible_columns" côté covariables et
        # "_group_frequency_progression" côté cible en tirent chacun les
        # conséquences
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

        # Scores de validation croisée qui ont décidé de cet ordre, par étape
        # puis par variable. Vide hors covariate_strategy='model' +
        # fit_predict_order='cv'. Exposé pour le tracking.
        self.imputation_cv_scores_: Dict[str, Dict[Any, float]] = {}

        # Accumulateur des avertissements de la phase 5 : ils sont émis en un
        # seul message en fin de phase, jamais un par variable et par étape
        self._warnings: List[str] = []

        # Couples imputés sans ancre, et couples que l'absence de covariable
        # exploitable a laissés NaN : les premiers alimentent
        # "unanchored_pairs_", les seconds l'avertissement agrégé
        self._unanchored_written: set = set()
        self._unanchored_failures: List[tuple] = []

        # Contrainte d'agrégation, portée par un composant unique : il recale
        # les prédictions des étapes ET, injecté dans le matérialiseur, les
        # covariables interpolées — une seule implémentation, aucune dérive
        self._aggregation_constraint = AggregationConstraint(
            aggregation_constraint=self.aggregation_constraint,
            context='HighFrequencyImputer',
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

        # Couples sans ancre restés vides : un seul avertissement les nommant
        # tous. L'interpolation ne peut pas leur servir de repli — il n'y a
        # rien à interpoler
        if self._unanchored_failures:
            named = sorted({pair for pair in self._unanchored_failures}, key=repr)
            warnings.warn(
                f"{len(named)} unobserved (entity, column) pair(s) could not be "
                f"imputed for lack of usable covariates on the target grid; "
                f"their cells stay NaN and ORIGINAL: {named}",
                UserWarning,
            )

        # =================================================================
        # PHASE 6 — Finalisation
        # =================================================================
        # Le plan est figé et les attributs de sortie sont renseignés. La
        # sortie multi-fréquences relève du "transform", seul producteur de
        # frame
        self.imputation_provenance_ = self._provenance_tracker.get_provenance_matrix()

        # Statistiques de provenance (comptes et pourcentages par
        # "ProvenanceType", au global et par colonne) : exposées telles quelles
        # pour le tracking, elles évitent à l'utilisateur de reconstruire un
        # tracker pour les recalculer
        self.provenance_statistics_ = self._provenance_tracker.compute_statistics()

        # Couples imputés sans ancre : toujours écrit, vide compris, ce qui
        # autorise sa lecture par "check_is_fitted"
        self.unanchored_pairs_ = tuple(sorted(self._unanchored_written, key=repr))

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
        kind: Literal['imputation', 'training'],
    ) -> pd.Series:
        """Read one window mask at the frequency of a stage.

        The ``kind`` is named by the caller, never defaulted. Only the two
        windows the fit actually restricts on are read here: the strict
        window is diagnostic (``strict_window_mask_``) and governs nothing
        in the stage execution.
        An entity the calculator omits — one without a valid fitted mask — is
        left unrestricted rather than silently losing every one of its rows.

        Args:
            X_work: Working frame, the fallback grid.
            stage_freq: Frequency of the stage, scalar or per entity.
            kind: Window read: ``'imputation'`` or ``'training'``.

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

    # Méthode auxiliaire de la grille non restreinte d'un groupe
    def _unrestricted_grid(
        self,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
        entities: Optional[Sequence[EntityKey]],
    ) -> pd.Index:
        """Build the stage grid of a group, window restriction lifted.

        Reserved to the unanchored steps, whose entities have
        no window to speak of: the prediction window of such an entity is
        empty, since the imputed column carries no anchor to bound it. Every
        row of the stage grid is kept; the covariates then decide, a row they
        cannot feed producing a NaN that is simply never written.

        Args:
            X_work: Working frame.
            stage_freq: Frequency of the stage.
            entities: Entities of the group, None for a time series.

        Returns:
            Index of the rows of those entities at the stage frequency.
        """
        # Index du masque, pris sans sa valeur booléenne : la grille de l'étape
        mask = self._stage_mask(X_work, stage_freq, kind='imputation')
        return self._restrict_to_entities(mask.index, entities)

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

    # Méthode auxiliaire de calcul des diviseurs de features par fréquence de ligne
    def _feature_divisors_per_row(
        self,
        *,
        feature_cols: Sequence[str],
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
        ways: Dict[str, MaterializationWay],
        binding: Union[str, Dict[EntityKey, str]],
        blocks: Dict[EntityKey, str],
        stage_freq: Union[str, Dict[EntityKey, str]],
        row_frequency: pd.Series,
        real_index: pd.Index,
        row_entities: Sequence[EntityKey],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Divide the covariates by the period their own row spans.

        The grid of a mutualized training set is homogeneous only as long as
        axis 2 injects nothing into it: a cell produced at an earlier, finer
        stage sits on a row of that stage's frequency, not of its block's. The
        covariates of such a row are materialized at the row frequency
        (:class:`TrainingSetBuilder`), so their divisor must be read there
        too — the very same unifying rule the target divisor follows.

        One call to :meth:`StageScaler.feature_divisors` per layer of rows
        sharing one frequency per entity, never one per row. A grid where
        every row already sits at its block frequency — the case as soon as
        axis 2 injects nothing — short-circuits back to the single historical
        call, block binding included.

        Args:
            feature_cols: Covariate columns to scale.
            freqs_by_column: Detected frequencies, keyed by column.
            ways: Materialization way retained for each covariate.
            binding: Block frequency binding of the training grid.
            blocks: Block frequency of each contributing entity.
            stage_freq: Frequency of the stage.
            row_frequency: Frequency each training row was produced at,
                indexed on the training grid — stamped with a frequency level
                or not.
            real_index: The ``(entity..., date)`` grid of the same rows, the
                only shape :class:`StageScaler` ever reads. The very object
                ``row_frequency.index`` is when no frequency level was stamped.
            row_entities: Entity key of each row, decomposed once by
                :func:`split_training_index`.

        Returns:
            Whatever :meth:`StageScaler.feature_divisors` returns for a
            homogeneous, unstamped grid; a ``DataFrame`` indexed like
            ``row_frequency`` and columned like ``feature_cols`` otherwise.
        """
        # Extraction de l'index
        index = row_frequency.index
        # Extraction des fréquences
        frequencies = row_frequency.to_numpy(dtype=object)
        # Grille estampillée : elle n'est jamais remise au scaler
        stamped = real_index is not index

        # Fréquence de bloc attendue pour chaque ligne
        expected = np.array(
            [blocks.get(entity) for entity in row_entities], dtype=object
        )

        # Grille homogène et non estampillée : le chemin d'origine, à la
        # liaison de bloc
        if not stamped and bool((frequencies == expected).all()):
            return self._stage_scaler.feature_divisors(
                columns=feature_cols,
                column_frequencies=freqs_by_column,
                ways=ways,
                grid_freq=binding,
                stage_freq=stage_freq,
                index=index,
            )

        # Grille hétérogène : un appel par couche, puis remontée au format par
        # ligne, seul capable de porter deux échelles sur une même colonne
        divisors = pd.DataFrame(
            np.nan, index=index, columns=list(feature_cols), dtype=float
        )
        for layer, in_layer in self._row_layers(row_entities, frequencies, blocks):
            # Écriture positionnelle : sous une grille estampillée, deux lignes
            # partagent une date et l'alignement par étiquette les confondrait
            positions = np.flatnonzero(in_layer)
            sub_real = real_index[in_layer]
            sub_divisors = self._stage_scaler.feature_divisors(
                columns=feature_cols,
                column_frequencies=freqs_by_column,
                ways=ways,
                grid_freq=self._block_binding(layer, sub_real),
                stage_freq=stage_freq,
                index=sub_real,
            )
            # Forme scalaire par colonne : diffusion sur les lignes concernées
            if isinstance(sub_divisors, pd.Series):
                for name in feature_cols:
                    divisors.iloc[
                        positions, divisors.columns.get_loc(name)
                    ] = float(sub_divisors[name])
            else:
                divisors.iloc[positions, :] = (
                    sub_divisors[list(feature_cols)].to_numpy(dtype=float)
                )
        return divisors

    # Méthode auxiliaire d'énumération des couches d'une grille d'entraînement
    @staticmethod
    def _row_layers(
        row_entities: Sequence[EntityKey],
        frequencies: np.ndarray,
        blocks: Dict[EntityKey, str],
    ) -> List[Tuple[Dict[EntityKey, str], np.ndarray]]:
        """Enumerate the layers of a training grid, one frequency per entity.

        Same rule as :meth:`TrainingSetBuilder._frequency_layers`, read here
        on the assembled grid rather than on the per-entity frames: the block
        frequency opens each entity's list, so a grid where axis 2 injected
        nothing yields exactly one layer.

        Args:
            row_entities: Entity key of each row, in grid order.
            frequencies: Production frequency of each row, in grid order.
            blocks: Block frequency of each contributing entity.

        Returns:
            Ordered list of ``(layer, selector)`` pairs, the selector being a
            boolean array over the grid. Empty layers are dropped.
        """
        # Fréquences de chaque entité, celle de son bloc en tête
        per_entity: Dict[EntityKey, List[str]] = {}
        for entity, frequency in zip(row_entities, frequencies):
            ordered = per_entity.setdefault(
                entity,
                [blocks[entity]] if entity in blocks else [],
            )
            if frequency not in ordered:
                ordered.append(frequency)

        # Couches : la i-ème fréquence de chaque entité qui en porte une
        layers: List[Tuple[Dict[EntityKey, str], np.ndarray]] = []
        depth = max((len(f) for f in per_entity.values()), default=0)
        for rank in range(depth):
            layer = {
                entity: ordered[rank]
                for entity, ordered in per_entity.items()
                if rank < len(ordered)
            }
            in_layer = np.array(
                [
                    layer.get(entity) == frequency
                    for entity, frequency in zip(row_entities, frequencies)
                ],
                dtype=bool,
            )
            if in_layer.any():
                layers.append((layer, in_layer))
        return layers

    # Méthode auxiliaire de calcul des diviseurs de cible par fréquence de ligne
    def _target_divisors_per_row(
        self,
        *,
        column: str,
        binding: Union[str, Dict[EntityKey, str]],
        blocks: Dict[EntityKey, str],
        stage_freq: Union[str, Dict[EntityKey, str]],
        row_frequency: pd.Series,
        real_index: pd.Index,
        row_entities: Sequence[EntityKey],
    ) -> Union[float, pd.Series]:
        """Divide the target by the period its own row spans.

        Strictly the historical single call
        (:meth:`StageScaler.target_divisor` with ``produced_freq``) as long as
        the grid carries no frequency level. Once it does, two rows share a
        date and the scaler — which groups by ``(stage frequency, production
        frequency)`` and writes its results back by label — would set both.
        The call is then split layer by layer, each layer being handed the
        real, duplicate-free ``(entity..., date)`` sub-grid, and the results
        written back positionally.

        Args:
            column: Column being imputed, for scale-mode resolution.
            binding: Block frequency binding of the training grid. Unread by
                the scaler as soon as ``produced_freq`` is given, and kept
                only to leave the historical call untouched.
            blocks: Block frequency of each contributing entity.
            stage_freq: Frequency of the stage.
            row_frequency: Frequency each training row was produced at.
            real_index: The ``(entity..., date)`` grid of the same rows.
            row_entities: Entity key of each row.

        Returns:
            A ``Series`` of per-row divisors indexed like ``row_frequency``,
            or the float ``1.0`` when the target is exempt from scaling.
        """
        # Extraction de l'index et des fréquences de ligne
        index = row_frequency.index
        frequencies = row_frequency.to_numpy(dtype=object)

        # Grille non estampillée : l'appel unique d'origine, inchangé
        if real_index is index:
            return self._stage_scaler.target_divisor(
                column,
                source_freq=binding,
                pred_freq=stage_freq,
                index=index,
                produced_freq=row_frequency,
            )

        # Grille estampillée : un appel par couche, sur la grille réelle
        divisors = pd.Series(np.nan, index=index, dtype=float)
        for layer, in_layer in self._row_layers(row_entities, frequencies, blocks):
            positions = np.flatnonzero(in_layer)
            sub_real = real_index[in_layer]
            produced = pd.Series(frequencies[in_layer], index=sub_real)
            sub_divisors = self._stage_scaler.target_divisor(
                column,
                source_freq=self._block_binding(layer, sub_real),
                pred_freq=stage_freq,
                index=sub_real,
                produced_freq=produced,
            )
            divisors.iloc[positions] = np.asarray(sub_divisors, dtype=float)
        return divisors

    # Méthode auxiliaire d'extraction de l'entité de chaque ligne d'une grille
    @staticmethod
    def _row_entities(
        index: pd.Index,
        has_frequency_level: bool = False,
    ) -> List[EntityKey]:
        """Return the entity key of every row of a grid.

        Thin pass-through over :func:`split_training_index`, the single
        decomposition of a training grid: under
        ``keep_coincident_cells`` the grid carries a frequency level on the
        entity side and the entity is ``key[:-2]``, not ``key[:-1]``.

        Args:
            index: Grid, ``MultiIndex`` ``(entity..., date)`` on a panel,
                ``(entity..., freq, date)`` when stamped.
            has_frequency_level: Whether the grid carries that extra level,
                read from :attr:`TrainingSet.has_frequency_level`.

        Returns:
            One entity key per row, in grid order. A time series yields the
            degenerate key ``()`` everywhere.
        """
        # Décomposition unique, partagée avec le constructeur du jeu
        _real, entities = split_training_index(
            index, has_frequency_level=has_frequency_level
        )
        return entities

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
            # Les groupes sans ancre n'ont pas de fréquence source à comparer :
            # ils sont écartés du rang, et une colonne qui n'aurait qu'eux se
            # range à la fréquence cible de ses entités
            anchored = [
                source_freq
                for _column, source_freq in groups
                if source_freq is not None
            ]
            entities = tuple(
                sorted({e for group in groups.values() for e in group}, key=repr)
            )
            lowest = (
                max(anchored, key=get_frequency_order) if anchored
                else self._entity_target_frequency(entities[0] if entities else None)
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
        # Report des scores CV qui ont décidé du rang, pour le tracking. Sous
        # l'ordre 'frequency' l'ordonnanceur laisse "scores_" vide
        stage_scores = dict(getattr(self._variable_orderer, 'scores_', {}) or {})
        if stage_scores:
            self.imputation_cv_scores_[stage_label] = stage_scores
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

        # Départage des cellules coïncidentes, lu par colonne : le
        # paramètre admet une forme dictionnaire, la valeur n'est donc jamais
        # globale. Le constructeur du jeu ne reçoit que ce booléen : il ignore
        # tout de "aggregation_constraint", comme il ignore tout de l'axe 2
        keep_coincident = resolve_aggregation_constraint(
            self.aggregation_constraint, column
        ) is None

        # Jeu mutualisé, sonde sans covariable : elle ne matérialise rien et
        # rend déjà les blocs, la cible brute et la grille d'entraînement
        probe = builder.build(
            column=column,
            feature_cols=(),
            stage_freq=stage_freq,
            detected_frequencies=freqs_by_column,
            source_data=X_work,
            eligible_origins=eligible,
            keep_coincident_cells=keep_coincident,
        )
        blocks = dict(probe.blocks)

        # Grille réelle de la sonde : le matérialiseur et le scaler ne lisent
        # jamais le niveau de fréquence. Dédoublonnée, deux cellules
        # coïncidentes partageant par définition une date
        probe_real, _probe_entities = split_training_index(
            probe.X.index, has_frequency_level=probe.has_frequency_level
        )
        probe_real = probe_real[~probe_real.duplicated()]

        # Grille de prédiction des entités concernées
        pred_grid = self._prediction_grid(X_work, stage_freq, entities)

        # Vues des deux fenêtres, chacune à sa fréquence : la grille
        # d'entraînement est au pas des blocs, la grille de prédiction au pas
        # de l'étape
        train_view = materializer.stage_frame(
            grid_index=probe_real,
            stage_freq=self._block_binding(blocks, probe_real) if blocks else stage_freq,
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
            keep_coincident_cells=keep_coincident,
        )

        # Mise à l'échelle : diviseur de la cible par ligne, diviseurs des
        # features par bloc
        X_train, y_train = training.X, training.y
        if len(training) > 0:
            # Décomposition unique de la grille d'entraînement : la grille
            # réelle que lisent le matérialiseur et le scaler, et l'entité de
            # chaque ligne — "key[:-2]" sous une grille estampillée
            real_index, row_entities = split_training_index(
                training.X.index,
                has_frequency_level=training.has_frequency_level,
            )
            binding = self._block_binding(blocks, real_index)
            y_train = self._stage_scaler.apply(
                y_train,
                self._target_divisors_per_row(
                    column=column,
                    binding=binding,
                    blocks=dict(training.blocks),
                    stage_freq=stage_freq,
                    row_frequency=training.row_frequency,
                    real_index=real_index,
                    row_entities=row_entities,
                ),
            )
            if feature_cols:
                X_train = self._stage_scaler.apply(
                    X_train,
                    self._feature_divisors_per_row(
                        feature_cols=feature_cols,
                        freqs_by_column=freqs_by_column,
                        ways=ways,
                        binding=binding,
                        blocks=dict(training.blocks),
                        stage_freq=stage_freq,
                        row_frequency=training.row_frequency,
                        real_index=real_index,
                        row_entities=row_entities,
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
                unanchored=group_key[1] is None,
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
        source_frequency: Optional[str],
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
        unanchored: bool = False,
    ) -> ImputationStep:
        """Freeze one plan step of a (stage, variable, source frequency) group.

        Args:
            column: Column imputed by the step.
            source_frequency: Detected frequency of the column for the group,
                None for an unanchored group.
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
            unanchored: Whether the entities of the group never observe the
                column, so the step predicts without any anchor.

        Returns:
            The frozen :class:`ImputationStep`.
        """
        # Groupe sans ancre : aucune fréquence source, donc aucune conversion.
        # La prédiction est produite directement à l'échelle de l'étape et les
        # deux facteurs valent 1.0 — le report reste neutre, exactement comme
        # pour les groupes ancrés dont les deux facteurs coïncident déjà
        if unanchored:
            scale = 1.0
        else:
            scale = self._group_scale_factor(column, source_frequency, stage_freq, grid)

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
            unanchored=unanchored,
        )

    # Méthode auxiliaire du facteur d'échelle d'un groupe ancré
    def _group_scale_factor(
        self,
        column: str,
        source_frequency: str,
        stage_freq: Union[str, Dict[EntityKey, str]],
        grid: pd.Index,
    ) -> Union[float, pd.Series]:
        """Compute the scale factor of one anchored group.

        The target having been scaled row by row, the predictions already come
        out at the pace of the stage: ``scale_factor`` and ``fit_scale_factor``
        coincide and the carry is 1.0.

        Args:
            column: Column imputed by the step.
            source_frequency: Detected frequency of the column for the group.
            stage_freq: Frequency of the stage.
            grid: Prediction grid of the group, read by the ``'calendar'``
                scale mode.

        Returns:
            The factor, or 1.0 when the scaler refuses the conversion.
        """
        try:
            return self._stage_scaler.fit_scale_factor(
                column,
                source_freq=source_frequency,
                pred_freq=stage_freq,
                index=grid,
            )
        except (ValueError, KeyError, TypeError):
            return 1.0

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

        An unanchored step departs from the common path on
        three points, and on three only: it is never rescaled to a period
        total, its cells carry ``MODEL_UNANCHORED``, and it has no
        interpolation fallback — there is nothing to interpolate for an
        entity that never observes the column, so a failure leaves its cells
        NaN and ``ORIGINAL`` and is reported by the aggregated warning of the
        end of the fit.

        Args:
            step: Frozen step to execute.
            X_work: Working frame the values are read from.
            stage_freq: Frequency of the stage.

        Returns:
            True when the step had to fall back on interpolation at execution
            time — a prediction failure — so the caller can degrade the plan
            step accordingly. False otherwise, an unanchored failure
            included: such a step is never degraded into a fallback, it is
            simply not executed.
        """
        # Extraction du matérisaliseur des étapes précédentes
        materializer = self._covariate_materializer
        # Extraction du nom de la colonne
        column = step.var_name
        # Détection des fréquences de la colonne (éventuellement pour différentes entités)
        freqs_by_column = self._detected_frequencies_by_column()

        # Grille du groupe : la fenêtre d'imputation de ses entités, ancres comprises
        grid = self._prediction_grid(X_work, stage_freq, step.entities)

        # Sans ancre, la fenêtre stricte d'une entité est vide par
        # construction : elle est l'intervalle où toutes ses colonnes sont
        # couvertes, et la colonne imputée n'y en couvre aucune. La subordonner
        # à la couverture de la colonne même que l'étape produit serait
        # circulaire ; l'entité est donc laissée sans restriction, exactement
        # comme "_stage_mask" traite déjà une entité que le calculateur omet
        if step.unanchored and len(grid) == 0:
            grid = self._unrestricted_grid(X_work, stage_freq, step.entities)

        if len(grid) == 0:
            return False

        # Production des valeurs : modèle, ou repli d'interpolation
        values: Optional[pd.Series] = None
        if not step.is_fallback:
            values = self._predict_step(step, grid, X_work, freqs_by_column, stage_freq)
        degraded = values is None and not step.is_fallback

        # Absence d'ancre : l'interpolation ne peut pas être le repli, faute
        # d'observation à interpoler. Les couples restent NaN et ORIGINAL, et
        # l'avertissement agrégé de la fin du fit les nomme
        if step.unanchored and values is None:
            self._unanchored_failures.extend(self._step_pairs(step))
            return False

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
            # Provenance tranchée par l'étape elle-même : c'est là, et là
            # seulement, que l'absence d'ancre prime sur les cinq familles
            provenance = step.emitted_provenance

        # Recalage aux totaux de la fréquence source du groupe : annuels pour
        # une entité annuelle, trimestriels pour une entité trimestrielle.
        # Court-circuité sans ancre, quelle que soit la valeur du paramètre :
        # il n'existe aucun total de période à imposer, et ces cellules sont
        # des prédictions libres là où les autres sont des désagrégations
        if not step.unanchored:
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
            # Sans ancre, une grille entièrement NaN : le couple est nommé par l'avertissement agrégé
            if step.unanchored:
                self._unanchored_failures.extend(self._step_pairs(step))
            return degraded

        # Couples effectivement imputés sans ancre, source de unanchored_pairs_
        if step.unanchored:
            self._unanchored_written.update(self._step_pairs(step))

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

    # Méthode auxiliaire des clés de couple d'une étape
    @staticmethod
    def _step_pairs(step: ImputationStep) -> List[tuple]:
        """Render the ``(entity..., column)`` keys covered by one step.

        The shape is that of the keys of ``detected_frequencies_`` and of the
        never-observed pairs, so ``unanchored_pairs_`` is directly comparable
        with them.

        Args:
            step: Executed step.

        Returns:
            One key per entity of the step, empty on a time series, which
            carries no entity level.

        Examples:
            >>> HighFrequencyImputer._step_pairs(step)     # doctest: +SKIP
            [('IT', 'v')]
        """
        # Série temporelle : aucune entité, donc aucun couple à nommer
        return [
            (*entity, step.var_name) for entity in (step.entities or ())
        ]

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
        # Report court-circuité : les deux facteurs sont le même
        # objet tant que le modèle est ajusté pour l'étape courante. Diviser
        # puis multiplier n'ajouterait que du bruit d'arrondi et, sous
        # 'calendar', désalignerait une Series figée sur la grille du fit
        # contre celle du transform — donc rendrait tout NaN
        if _scale_factors_equal(step.scale_factor, step.fit_scale_factor):
            return values
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
    # Transform — rejeu du plan figé
    # -------------------------------------------------------------------------
    # Gestionnaire de contexte installant l'état de rejeu
    @contextmanager
    def _replay_state(
        self,
        window_calc: ImputationWindowCalculator,
        tracker: ImputationProvenanceTracker,
        frequencies: Dict[Union[str, tuple], str],
    ):
        """Install the transform state around a replay, then restore the fit's.

        The step execution of ``fit`` reads its window calculator, its
        provenance tracker, its warning accumulators and its three registers
        from the instance. Swapping them here is what lets ``transform`` reuse
        :meth:`_execute_step` **as is** rather than carry a second execution
        loop.

        Args:
            window_calc: Window calculator recomputed on the transformed data..
            tracker: Provenance tracker of this transform, already initialized
                on the frame produced by the additive transformer.
            frequencies: Frequencies detected for the (entity, column) pairs
                the fit never saw, the only ones no fit value can be replayed
                for.

        Yields:
            None. The fit state is restored on the way out, exception
            included.
        """
        # Mémorisation de l'état du fit, restauré en sortie
        saved = (
            self._imputation_window_calc,
            self._provenance_tracker,
            self._warnings,
            self._unanchored_written,
            self._unanchored_failures,
            getattr(self, '_transform_frequencies', {}),
            getattr(self, '_replay_notes', []),
        )
        self._imputation_window_calc = window_calc
        self._provenance_tracker = tracker
        self._warnings = []
        self._unanchored_written = set()
        self._unanchored_failures = []
        self._transform_frequencies = dict(frequencies)
        self._replay_notes = []
        # Registres du transform : recalculés, jamais hérités du fit
        self._covariate_materializer.reset()
        try:
            yield
        finally:
            (
                self._imputation_window_calc,
                self._provenance_tracker,
                self._warnings,
                self._unanchored_written,
                self._unanchored_failures,
                self._transform_frequencies,
                self._replay_notes,
            ) = saved

    # Méthode auxiliaire de contrôle des fréquences au transform
    def _check_transform_frequencies(
        self,
        X_work: pd.DataFrame,
    ) -> Dict[Union[str, tuple], str]:
        """Compare the frequencies of the transform with those of the fit.

        The comparison is made **pair by pair** — ``(entity..., column)`` — and
        never column by column: on a panel the same column may legitimately
        carry a different frequency for each entity, and a
        per-column comparison would report a false divergence at every
        transform.

        Args:
            X_work: Working frame of the transform, before the additive
                transformer.

        Returns:
            Frequencies detected for the pairs the fit never saw — a new entity
            of a known column. Their frequency cannot be replayed, so this is
            the only one available.

        Raises:
            ValueError: If a column known at fit time is absent from the data
                being transformed. The message names the missing columns.
        """
        # Colonnes du fit, couples indétectables compris : une colonne
        # entièrement NaN au fit reste une colonne attendue au transform
        fit_columns = {
            split_variable_key(key)[1]
            for key in (*self.detected_frequencies_, *self._undetected_frequencies_)
        }
        # Colonnes disparues : erreur nommant les colonnes, jamais un silence
        missing = sorted(
            column for column in fit_columns if column not in X_work.columns
        )
        if missing:
            raise ValueError(
                f"transform is missing {len(missing)} column(s) seen at fit "
                f"time: {missing}. Every column of the fit must be present, "
                f"even entirely empty: it is part of the frozen plan. Extra "
                f"columns, on the other hand, are ignored."
            )

        # Redétection sur les données du transform
        detected = self._detect_frequencies_robustly(X_work)

        # Divergences, couple par couple, sur les seules clés communes
        diverging: List[Tuple[Any, str, str]] = []
        for key, freq in detected.items():
            fit_freq = self.detected_frequencies_.get(key)
            if freq is None or fit_freq is None:
                continue
            if (
                normalize_frequency(freq, return_format='base')
                != normalize_frequency(fit_freq, return_format='base')
            ):
                diverging.append((key, fit_freq, freq))

        # Avertissement uniquement, puis poursuite avec les fréquences du fit :
        # "_detected_frequencies_by_column" ne lit que "detected_frequencies_"
        if diverging:
            listing = ', '.join(
                f"{key!r}: fit={fit_freq}, transform={freq}"
                for key, fit_freq, freq in sorted(
                    diverging, key=lambda item: repr(item[0])
                )
            )
            warnings.warn(
                f"{len(diverging)} (entity, column) pair(s) carry a different "
                f"frequency at transform time; the fit frequencies are kept: "
                f"{listing}",
                UserWarning,
            )

        # Couples inconnus du fit, sur des colonnes du fit : leur fréquence est
        # celle qu'on vient de détecter, aucune autre n'existe
        return {
            key: freq
            for key, freq in detected.items()
            if freq is not None
            and key not in self.detected_frequencies_
            and split_variable_key(key)[1] in fit_columns
        }

    # Méthode auxiliaire de l'avertissement unique des lignes hors fenêtre
    def _warn_rows_outside_window(
        self,
        window_calc: ImputationWindowCalculator,
        X_work: pd.DataFrame,
    ) -> None:
        """Warn once about the rows the recomputed window leaves out.

        The window is a constraint on data availability, not a learned
        parameter: it is recomputed on the transformed data. Rows falling
        outside it are simply not predicted — nothing is ever blanked — and a
        single aggregated message names how many they are and which entities
        they belong to.

        Args:
            window_calc: Window calculator fitted on the transformed data.
            X_work: Working frame of the transform.
        """
        # Masque de prédiction, "kind" nommé explicitement
        try:
            mask = window_calc.get_imputation_window_mask(X_work, kind='imputation')
        except (ValueError, KeyError, TypeError):
            return

        # Lignes du périmètre laissées hors fenêtre
        outside = mask.index[~mask.to_numpy(dtype=bool)]
        if len(outside) == 0:
            return

        # Entités concernées, nommées dans le message agrégé
        entities = tuple(self._index_entities(outside)) if self.is_panel_ else ()
        warnings.warn(
            f"{len(outside)} row(s) of the data being transformed fall outside "
            f"the imputation window recomputed on them"
            + (f", for entities {entities}" if entities else "")
            + ". These rows keep their input values: nothing is ever blanked.",
            UserWarning,
        )

    # Méthode auxiliaire des entités apparues au transform
    def _new_entities(self, X_work: pd.DataFrame) -> Tuple[EntityKey, ...]:
        """List the entities of the transform the fit never saw.

        Args:
            X_work: Working frame of the transform.

        Returns:
            Normalized entity keys, in order of appearance. Empty on a time
            series.
        """
        # Série temporelle : aucune entité, donc aucune nouveauté possible
        if not self.is_panel_ or not self.entities_:
            return ()

        # Entités du fit, sous forme normalisée
        known = {normalize_entity_key(entity) for entity in self.entities_}
        return tuple(
            entity
            for entity in self._index_entities(X_work.index)
            if entity not in known
        )

    # Méthode auxiliaire d'extension de la liaison d'étape aux entités nouvelles
    def _extend_stage_binding(
        self,
        stage_freq: Union[str, Dict[EntityKey, str]],
        new_entities: Sequence[EntityKey],
        stage_label: str,
    ) -> Union[str, Dict[EntityKey, str]]:
        """Bind the entities born at transform time to a stage frequency.

        A new entity carries no ``target_frequency`` entry — a dict one is
        checked against the entities of the fit — so it can only join a stage
        whose entities agree on one single frequency. When they disagree no
        rule departs them, and the entity is left out of that stage and named
        by the aggregated warning.

        Args:
            stage_freq: Frequency of the stage, scalar or per entity.
            new_entities: Entities the fit never saw.
            stage_label: Readable label of the stage, for the message.

        Returns:
            The binding extended to the new entities, or the original one.
        """
        # Rien à étendre, ou étape scalaire (série temporelle)
        if not new_entities or not isinstance(stage_freq, dict):
            return stage_freq

        # Unanimité des entités de l'étape
        unique = {
            normalize_frequency(freq, return_format='base')
            for freq in stage_freq.values()
        }
        if len(unique) != 1:
            self._replay_notes.append(
                f"stage {stage_label}: its entities disagree on the stage "
                f"frequency, so the entities born at transform time "
                f"{tuple(new_entities)} cannot be bound to it and are skipped"
            )
            return stage_freq

        shared = unique.pop()
        return {**stage_freq, **{entity: shared for entity in new_entities}}

    # Méthode auxiliaire des étapes sans ancre des entités nouvelles
    def _new_entity_steps(
        self,
        steps: Sequence[ImputationStep],
        new_entities: Sequence[EntityKey],
        binding: Union[str, Dict[EntityKey, str]],
    ) -> List[ImputationStep]:
        """Derive the unanchored steps of the entities born at transform time.

        An entity absent from the fit is, by construction, an entity without
        any anchor: it falls under ``impute_unobserved_entities``, and under it
        only. The step is derived from a step of the same
        (stage, variable) — same model object (``is``), same ``feature_cols``,
        same materialization ways — so nothing is decided here: only the
        entities, the absence of source frequency and the neutral scale change.

        Such a step never enters ``imputation_plan_``: the plan is the state of
        the fit.

        Args:
            steps: Plan steps of the stage, in plan order.
            new_entities: Entities the fit never saw.
            binding: Stage binding, already extended (or not) to them.

        Returns:
            One step per column imputed at that stage, empty when the parameter
            is off or when no new entity could be bound.
        """
        # Paramètre éteint, ou aucune entité nouvelle : rien à dériver
        if not new_entities or not self.impute_unobserved_entities:
            return []

        # Entités effectivement liées à l'étape
        bound = [
            entity
            for entity in new_entities
            if not isinstance(binding, dict) or entity in binding
        ]
        if not bound:
            return []

        # Un représentant par colonne, dans l'ordre du plan
        derived: List[ImputationStep] = []
        seen: set = set()
        for step in steps:
            if step.var_name in seen:
                continue
            seen.add(step.var_name)
            derived.append(
                replace(
                    step,
                    source_frequency=None,
                    entities=to_entity_tuple(bound),
                    scale_factor=1.0,
                    fit_scale_factor=1.0,
                    unanchored=True,
                )
            )
        return derived

    # Méthode de rejeu du plan, étape de fréquence par étape de fréquence
    def _replay_plan(
        self,
        X_stage: pd.DataFrame,
    ) -> Tuple["OrderedDict[str, pd.DataFrame]", "OrderedDict[str, pd.DataFrame]"]:
        """Replay every frozen step of the plan, in the order of the fit.

        Nothing is decided here: the classification, the progression, the
        variable order, the models, the ``feature_cols``, the materialization
        ways and the taints all come from the plan. Only the windows, the stage
        frames, the interpolated values, the predictions, the rescaling and the
        provenance are recomputed.

        Args:
            X_stage: Working frame after the additive transformer. Never
                modified: the imputations travel through the registers of
                :class:`CovariateMaterializer`, exactly as at fit time.

        Returns:
            Tuple ``(frames, provenances)``, both keyed by stage label in
            progression order — the stacking order of the multi-frequency
            output.
        """
        # Initialisation des jeux de données et des provenances
        frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        provenances: "OrderedDict[str, pd.DataFrame]" = OrderedDict()

        # Étapes du plan, indexées par label d'étape
        by_stage = self.imputation_plan_.by_stage()
        # Entités nées au transform
        new_entities = self._new_entities(X_stage)
        # Entités nouvelles laissées de côté faute de paramètre : une note
        if new_entities and not self.impute_unobserved_entities:
            self._replay_notes.append(
                f"{len(new_entities)} entity/entities absent from the fit "
                f"{new_entities} are left untouched "
                f"(impute_unobserved_entities=False)"
            )

        # Report de fréquence : dernière étape traversée par chaque entité
        carried: Dict[EntityKey, str] = {}
        last_position = len(self.frequency_progression_) - 1
        # Parcours des étapes de la progression, rejouée telle quelle
        for position, stage_freq in enumerate(self.frequency_progression_):
            stage_label = self._stage_frequency_label(stage_freq)
            # Liaison de fréquence, étendue aux entités nouvelles (D34)
            binding = self._extend_stage_binding(stage_freq, new_entities, stage_label)

            # Étapes de plan de l'étape, puis celles des entités nouvelles
            steps = list(by_stage.get(stage_label, ()))
            steps = steps + self._new_entity_steps(steps, new_entities, binding)
            # Logging
            self._log(
                f"[transform] stage {stage_label}: replaying {len(steps)} step(s)"
            )
            # Exécution par LA MÊME méthode qu'au fit
            for step in steps:
                self._execute_step(step, X_work=X_stage, stage_freq=binding)

            # Liaison complète, pour la sortie : une entité que l'étape ne
            # concerne pas garde les lignes de son dernier passage, et une
            # liaison partielle ferait échouer la lecture du masque, qui
            # retomberait alors sans bruit sur tout l'index
            output_binding = self._output_binding(binding, carried, X_stage)

            # Grille de la frame : l'INDEX du masque, jamais ses seules lignes
            # vraies — les lignes hors fenêtre restent dans la sortie avec
            # leurs valeurs d'entrée, aucune observation n'est détruite
            grid = self._stage_mask(X_stage, output_binding, kind='imputation').index
            # Étape finale : aucune ligne d'entrée ne peut manquer à la sortie
            if position == last_position:
                absent = X_stage.index.difference(grid)
                if len(absent) > 0:
                    grid = grid.append(absent)

            # Frame d'étape reconstruite après les écritures, puis recouverte
            # par les imputations : "stage_frame" ne lit le miroir que sous
            # "covariate_strategy='model'", et la sortie doit le porter sous
            # toutes les stratégies
            frame = self._covariate_materializer.stage_frame(
                grid_index=grid,
                stage_freq=output_binding,
                detected_frequencies=self._detected_frequencies_by_column(),
                source_data=X_stage,
            )
            # Noms de l'index d'entrée : la grille vient d'un masque converti,
            # qui ne les porte pas, et la sortie doit rester lisible comme
            # l'entrée — empilée ou non
            if grid.nlevels == X_stage.index.nlevels:
                frame = frame.rename_axis(X_stage.index.names)
            frames[stage_label] = self._overlay_imputations(frame, output_binding)

            # Instantané de provenance, pris au même instant et réindexé sur la
            # grille de l'étape : la matrice partage ainsi l'index de la sortie,
            # que "_inverse_transform" lit ensuite masque contre masque
            snapshot = self._provenance_tracker.get_provenance_matrix()
            provenances[stage_label] = (
                snapshot.reindex(index=grid, columns=frame.columns)
                .fillna(ProvenanceType.ORIGINAL)
                .rename_axis(frame.index.names)
            )

            # Report : chaque entité de l'étape y inscrit sa fréquence, lue par
            # les étapes suivantes qui ne la concernent plus
            if isinstance(binding, dict):
                carried.update(binding)

        return frames, provenances

    # Méthode auxiliaire de la liaison de fréquence de la sortie d'une étape
    def _output_binding(
        self,
        binding: Union[str, Dict[EntityKey, str]],
        carried: Dict[EntityKey, str],
        X_stage: pd.DataFrame,
    ) -> Union[str, Dict[EntityKey, str]]:
        """Complete a stage binding with the entities the stage leaves out.

        A stage of a panel binds only the entities it concerns: the
        others have either already reached their target frequency or not yet
        entered the progression. The output, however, has to carry every
        entity, so each of them is bound to the frequency of its last stage —
        its target frequency when it has not travelled any stage yet.

        Args:
            binding: Execution binding of the stage, extended to the entities
                born at transform time.
            carried: Frequency of the last stage each entity took part in.
            X_stage: Working frame, source of the entity list.

        Returns:
            The binding itself on a time series; a mapping naming every entity
            of the data otherwise.
        """
        # Série temporelle : la forme scalaire est la seule qui ait un sens
        if not isinstance(binding, dict):
            return binding

        completed = dict(binding)
        for entity in self._index_entities(X_stage.index):
            if entity in completed:
                continue
            # Report du dernier passage, à défaut la cible de l'entité, à
            # défaut encore la fréquence de l'étape : une entité sans
            # fréquence sortirait de la grille de sortie, donc du résultat
            frequency = (
                carried.get(entity)
                or self._entity_target_frequency(entity)
                or next(iter(binding.values()), None)
            )
            if frequency is not None:
                completed[entity] = frequency
        return completed

    # Méthode auxiliaire de recouvrement d'une frame par les imputations
    def _overlay_imputations(
        self,
        frame: pd.DataFrame,
        output_binding: Union[str, Dict[EntityKey, str]],
    ) -> pd.DataFrame:
        """Lay the imputations of the mirror over a stage frame.

        ``stage_frame`` fabricates nothing: it carries the input values and
        the exact aggregations, and reads the mirror only under
        ``covariate_strategy='model'``. The output must show the imputations
        under every strategy, and they must win over the raw anchor of a
        lower-frequency column — an anchor row re-expressed no longer carries
        the total of its period.

        Only the cells produced at the frequency of the row are laid over: a
        value produced at an earlier, coarser stage would otherwise land on a
        finer grid at the magnitude of another period.

        Args:
            frame: Stage frame, as built by the materializer.
            output_binding: Complete frequency binding of the frame's grid.

        Returns:
            The frame, imputations included.
        """
        # Extraction du matérialiseur
        materializer = self._covariate_materializer
        # Aucune production : la frame est déjà la sortie de l'étape
        if not materializer.imputed_store:
            return frame

        # Fréquence d'étape de chaque ligne de la grille
        row_frequency = pd.Series(
            self._stage_frequency_of(output_binding, frame.index), index=frame.index
        )

        # Initialisation du jeu de données résultat
        result = frame.copy()
        # Parcours des colonnes
        for column in result.columns:
            # Cellules imputées
            produced = materializer.imputed_store.get(column)
            if produced is None:
                continue
            values = produced.reindex(frame.index)
            # Fréquence de production, cellule par cellule
            produced_frequency = materializer.imputed_freq_store[column].reindex(
                frame.index
            )
            # Cellules de l'étape courante, les seules à la bonne échelle
            kept = values.where(
                values.notna() & produced_frequency.eq(row_frequency)
            )
            result[column] = kept.combine_first(result[column])
        return result

    # Méthode de transformation : rejeu du plan figé
    def _transform(self, X, y=None):
        """Replay the fitted plan on new data.

        Phases 0'-4' recompute what depends on the data — ``y`` alignment and
        naming by the same functions as the fit, the additive transformer
        applied with the fitted object, the provenance tracker initialized
        after it, the windows recomputed and the frequency check
        — then every frozen step is replayed by :meth:`_execute_step`,
        the very method the fit calls. No second execution loop, and no
        training set is ever rebuilt: the ``TrainingSetBuilder`` is not on this
        path.

        Note:
            This method is stateful: it snapshots its input in ``_original_X_``
            / ``_original_y_`` and overwrites ``imputation_provenance_`` at
            each call. :meth:`_inverse_transform` reads them back and therefore
            always inverts the last transform. ``fit`` purges the three.

        Args:
            X: Features to transform.
            y: Target to transform (optional).

        Returns:
            The transformed features, or the pair ``(X, y)`` when ``y`` is
            given. Under ``keep_lower_frequencies=True`` the index carries a
            ``'frequency'`` level.

        Raises:
            ValueError: If ``X`` is not a DataFrame, if it carries no usable
                temporal index, or if a column of the fit is missing.

        Examples:
            >>> imputed = imputer.fit(df).transform(df)      # doctest: +SKIP
        """
        # =================================================================
        # PHASE 0' — Setup, contrat d'entrée et instantané
        # =================================================================
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        # Alignement de l'index de y sur celui de X, par la même fonction qu'au
        # fit : deux règles de nommage divergentes ont déjà fait perdre la cible
        # parmi les colonnes d'étapes
        if y is not None:
            y = self._align_target_index(X, y)

        # Instantané de l'entrée du dernier transform, lu par l'inversion
        self._original_X_ = X.copy()
        self._original_y_ = y.copy() if y is not None else None

        # Concaténation en un unique jeu de travail
        y_col_name: Optional[str] = None
        if y is not None:
            y_col_name = self._resolve_target_column_name(y)
            X_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            X_work = X.copy()

        # Index temporel exploitable
        if not isinstance(X_work.index, (pd.DatetimeIndex, pd.MultiIndex)):
            if self.time_col and self.time_col in X_work.columns:
                X_work = X_work.set_index(self.time_col)
            else:
                raise ValueError("Data must have a DatetimeIndex or MultiIndex")

        # Contrôle des fréquences (D11) : erreur sur colonne manquante,
        # avertissement unique sur divergence, silence sur colonne en trop
        new_frequencies = self._check_transform_frequencies(X_work)

        # =================================================================
        # PHASE 1' — Fenêtres recalculées
        # =================================================================
        # Calcul au même stade qu'au fit — avant le transformateur additif —
        # pour que le recalcul sur les données du fit redonne exactement la
        # fenêtre du fit, et donc "fit_transform(X) == fit(X).transform(X)"
        window_calc, window_error = self._fit_imputation_window(X_work)
        if window_calc is None:
            warnings.warn(
                f"Could not calculate the imputation window of the data being "
                f"transformed: {window_error}. Using all available data.",
                UserWarning,
            )
            # Calculateur non ajusté : les gardes des consommateurs le neutralisent
            window_calc = self._make_window_calculator()
        else:
            self._warn_rows_outside_window(window_calc, X_work)

        # =================================================================
        # PHASE 2' — Transformateur additif, avec l'objet ajusté
        # =================================================================
        if self.additive_transformer_ is not None:
            X_stage = self.additive_transformer_.transform(X_work)
            # Déballage du couple (X, y) que renvoie un transformateur XY
            if isinstance(X_stage, tuple):
                X_stage = X_stage[0]
        else:
            X_stage = X_work.copy()

        # =================================================================
        # PHASE 3'-4' — Provenance, initialisée après lui
        # =================================================================
        tracker = ImputationProvenanceTracker()
        tracker.initialize(X_stage, panel_cols=self.panel_cols)

        # =================================================================
        # PHASE 5' — Rejeu du plan
        # =================================================================
        with self._replay_state(window_calc, tracker, new_frequencies):
            frames, provenances = self._replay_plan(X_stage)
            # Avertissements accumulés pendant le rejeu, relevés avant que le
            # contexte ne restaure les accumulateurs du fit
            degraded = list(self._warnings)
            unanchored_failures = sorted(set(self._unanchored_failures), key=repr)
            notes = list(self._replay_notes)

        # Avertissements agrégés, un message par famille
        if degraded:
            warnings.warn(
                f"{len(degraded)} imputation step(s) degraded during the "
                f"transform:\n  - " + "\n  - ".join(degraded),
                UserWarning,
            )
        if unanchored_failures:
            warnings.warn(
                f"{len(unanchored_failures)} unobserved (entity, column) "
                f"pair(s) could not be imputed for lack of usable covariates "
                f"on the target grid; their cells stay NaN and ORIGINAL: "
                f"{unanchored_failures}",
                UserWarning,
            )
        if notes:
            warnings.warn(
                f"{len(notes)} case(s) of entities outside the frozen plan at "
                f"transform time:\n  - " + "\n  - ".join(notes),
                UserWarning,
            )

        # =================================================================
        # PHASE 6' — Sortie multi-fréquences et provenance
        # =================================================================
        if not frames:
            # Progression sans étape : rien n'a été produit
            data_result = X_stage
            self.imputation_provenance_ = tracker.get_provenance_matrix()
        elif self.keep_lower_frequencies:
            data_result = self._build_multifreq_output(frames)
            self.imputation_provenance_ = self._build_multifreq_output(provenances)
        else:
            final_label = list(frames)[-1]
            data_result = frames[final_label]
            self.imputation_provenance_ = provenances[final_label]

        # Scission X / y
        if y is not None and y_col_name in data_result.columns:
            return data_result.drop(columns=[y_col_name]), data_result[y_col_name]
        return data_result

    # -------------------------------------------------------------------------
    # Sortie multi-fréquences
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'empilage des niveaux de fréquence
    def _build_multifreq_output(
        self,
        stage_frames: "OrderedDict[str, pd.DataFrame]",
    ) -> pd.DataFrame:
        """Stack the stage frames into one multi-frequency frame.

        Backs ``keep_lower_frequencies=True``, a **pure display parameter**: it
        governs how the levels of the output are stacked, never the logic. The
        index adds the frequency level taht sits on the
        entity side and every entity level of a panel is preserved with its
        own name.

        Under ``impute_intermediate_frequencies=False`` the progression holds
        the target stage alone: the output then carries that single level,
        there being no intermediate one to stack.

        Args:
            stage_frames: Stage frames keyed by frequency label, in progression
                order. Each must already carry the imputations of its stage —
                :meth:`_replay_plan` builds them after the writes.

        Returns:
            Frame with a MultiIndex ``(frequency, date)`` on a time series and
            ``(entity..., 'frequency', 'date')`` on a panel.
        """
        # Empilage dans l'ordre d'insertion, l'étiquette de chaque bloc étant
        # construite à part pour ne jamais entrer en collision avec une colonne
        all_frames: List[pd.DataFrame] = []
        freq_labels: List[np.ndarray] = []
        for freq_label, frame in stage_frames.items():
            all_frames.append(frame)
            freq_labels.append(np.full(len(frame), freq_label, dtype=object))

        combined = pd.concat(all_frames, ignore_index=False)
        freq_values = np.concatenate(freq_labels)

        # Construction du MultiIndex de sortie
        if self.is_panel_ and isinstance(combined.index, pd.MultiIndex):
            # Conservation de tous les niveaux d'entité, noms compris
            n_entity = combined.index.nlevels - 1
            entity_arrays = [
                combined.index.get_level_values(level) for level in range(n_entity)
            ]
            entity_names = [
                combined.index.names[level]
                if combined.index.names[level] is not None
                else ('entity' if n_entity == 1 else f'entity_{level}')
                for level in range(n_entity)
            ]
            new_index = pd.MultiIndex.from_arrays(
                [*entity_arrays, freq_values, combined.index.get_level_values(-1)],
                names=[
                    *entity_names,
                    'frequency',
                    combined.index.names[-1] or 'date',
                ],
            )
        else:
            new_index = pd.MultiIndex.from_arrays(
                [freq_values, combined.index],
                names=['frequency', combined.index.name or 'date'],
            )

        return combined.set_axis(new_index)

    # -------------------------------------------------------------------------
    # Transformation inverse
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de présence du niveau de fréquence
    @staticmethod
    def _has_frequency_level(frame: pd.DataFrame) -> bool:
        """Tell whether a frame carries the multi-frequency ``frequency`` level.

        Args:
            frame: Frame to inspect.

        Returns:
            True if its index is a MultiIndex holding a ``'frequency'`` level.
        """
        return (
            isinstance(frame.index, pd.MultiIndex)
            and 'frequency' in (frame.index.names or [])
        )

    # Méthode auxiliaire de sélection du niveau de fréquence à inverser
    def _select_inverse_frequency_level(
        self,
        data: pd.DataFrame,
        provenance: pd.DataFrame,
    ) -> Optional[str]:
        """Pick the frequency level to keep when inverting a stacked output.

        The level of the source index is the one to keep: it is where the
        values sit at the granularity of the data given to ``fit``. It may be
        missing — an undetectable index frequency, or a frequency that never
        was a stage — and the target level is then the best proxy, being the
        only one always produced.

        Args:
            data: Frame to invert, stacked or not.
            provenance: Provenance matrix of the last transform, stacked or not
                — both follow ``keep_lower_frequencies``.

        Returns:
            The frequency label to keep, or None when neither frame carries a
            frequency level.
        """
        # Recensement des labels disponibles, données et provenance confondues
        available: List[str] = []
        for frame in (data, provenance):
            if self._has_frequency_level(frame):
                available.extend(
                    frame.index.get_level_values('frequency').unique().tolist()
                )
        if not available:
            return None

        # Priorité au niveau de l'index source
        source_label = getattr(self, '_source_index_frequency_label', None)
        if source_label is not None and source_label in available:
            return source_label

        # Repli sur le niveau cible, toujours produit
        target_label = self._stage_frequency_label(self.effective_target_frequency_)
        if target_label in available:
            return target_label

        # Dernier repli : le dernier niveau empilé, avec avertissement
        warnings.warn(
            f"Neither the source frequency level ({source_label}) nor the "
            f"target one ({target_label}) is present in the data to invert. "
            f"Falling back on the last stacked level {available[-1]!r}.",
            UserWarning,
        )
        return available[-1]

    # Méthode auxiliaire de suppression du niveau de fréquence
    def _drop_frequency_level(
        self,
        frame: pd.DataFrame,
        label: Optional[str],
    ) -> pd.DataFrame:
        """Reduce a stacked frame to one frequency level and restore its index.

        Panel and time series are handled alike: the level is addressed by
        name, never by position, and the names of the remaining levels are
        restored from the last transform input.

        Args:
            frame: Frame to reduce, stacked or not.
            label: Frequency label to keep. None leaves the frame untouched.

        Returns:
            The frame restricted to ``label``, without the frequency level.
        """
        # Frame déjà à un seul niveau de fréquence
        if label is None or not self._has_frequency_level(frame):
            return frame

        # Niveau absent du frame : rien à extraire
        if label not in frame.index.get_level_values('frequency'):
            return frame
        reduced = frame.xs(label, level='frequency')

        # Restauration des noms de niveaux de l'index source
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

        Backs ``restore_original_values=True``: the ``ORIGINAL`` mask alone
        drops the anchor dates of a lower-frequency variable, which the target
        level holds as an imputation even though the input carried a true
        observation there.

        Args:
            data_result: Frame restored from the provenance mask, at the source
                index.
            y_col_name: Column name given to ``y`` in the working frame.

        Returns:
            The frame with every cell observed in the snapshot set back to its
            original value.
        """
        # Reconstruction de l'instantané de l'entrée du dernier transform
        snapshot = self._original_X_
        if self._original_y_ is not None and y_col_name is not None:
            snapshot = pd.concat(
                [snapshot, self._original_y_.to_frame(name=y_col_name)], axis=1
            )

        # Restriction aux colonnes et à l'index communs
        common_cols = [c for c in data_result.columns if c in snapshot.columns]
        if not common_cols:
            return data_result
        aligned = snapshot[common_cols].reindex(index=data_result.index)

        # Les valeurs observées priment sur celles restaurées par la provenance
        data_result = data_result.copy()
        data_result[common_cols] = aligned.combine_first(data_result[common_cols])
        return data_result

    # Méthode de transformation inverse
    def _inverse_transform(self, X, y=None):
        """Undo the imputation and the additive transformer.

        Mirror of :meth:`_transform`, driven by the provenance matrix of the
        last transform rather than by the fit: inverting a transform run on new
        data must undo what that very call produced. Four moves: keep the
        frequency level of the source index and restore the index names,
        set back to NaN every cell whose provenance is not ``ORIGINAL``, invert
        the additive transformer last, then optionally restore the exact
        original values.

        Args:
            X: Transformed features, as returned by ``transform``.
            y: Transformed target (optional).

        Returns:
            The original features, or the pair ``(X, y)`` when ``y`` is given.

        Raises:
            ValueError: If ``transform`` was never called: the provenance
                matrix it writes is what identifies the imputed cells.

        Examples:
            >>> restored = imputer.inverse_transform(transformed)  # doctest: +SKIP
        """
        # 0. Garde : la provenance du dernier transform est indispensable
        if 'imputation_provenance_' not in self.__dict__:
            raise ValueError(
                "inverse_transform requires a previous call to transform: the "
                "provenance matrix of the last transform "
                "(imputation_provenance_) identifies the cells to set back to "
                "NaN. Call transform(X) or fit_transform(X) first."
            )

        # Concaténation X / y, symétrique de "_transform"
        y_col_name: Optional[str] = None
        if y is not None:
            y_col_name = self._resolve_target_column_name(y)
            data_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            data_work = X.copy()

        provenance = self.imputation_provenance_

        # 1. Retrait du niveau de fréquence, sur les données ET la provenance :
        # chacune le porte ou non, selon "keep_lower_frequencies"
        level_label = self._select_inverse_frequency_level(data_work, provenance)
        data_work = self._drop_frequency_level(data_work, level_label)
        provenance = self._drop_frequency_level(provenance, level_label)

        # 2. Remise à NaN de toute cellule non originale
        original_mask = (provenance == ProvenanceType.ORIGINAL).reindex(
            index=data_work.index, columns=data_work.columns
        )
        # Colonnes hors périmètre de la provenance : jamais masquées
        untracked = [c for c in data_work.columns if c not in provenance.columns]
        if untracked:
            original_mask[untracked] = True
        original_mask = original_mask.fillna(False).astype(bool)
        data_work = data_work.where(original_mask)

        # 3. Inversion de la transformation additive, en DERNIER (miroir du
        # transform, qui l'applique en premier)
        data_result = data_work
        if self.additive_transformer_ is not None and hasattr(
            self.additive_transformer_, 'inverse_transform'
        ):
            try:
                inverted = self.additive_transformer_.inverse_transform(data_work)
                data_result = inverted[0] if isinstance(inverted, tuple) else inverted
            except Exception as error:
                warnings.warn(
                    f"Failed to inverse transform with the additive "
                    f"transformer: {error}",
                    UserWarning,
                )

        # 4. Restauration optionnelle des valeurs d'origine exactes
        if self.restore_original_values:
            data_result = self._restore_original_values(data_result, y_col_name)

        # 5. Scission X / y, symétrique de "_transform"
        if y is not None and y_col_name in data_result.columns:
            return data_result.drop(columns=[y_col_name]), data_result[y_col_name]
        return data_result
