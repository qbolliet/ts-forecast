"""Mixed-frequency imputer ``HighFrequencyImputer``, which imputes low-frequency columns
onto a finer target grid in cascade and records the provenance of every imputed value.
Also holds the private per-variable fit record."""
# Importation des modules
# Modules de base
import warnings
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Mapping,
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
    get_frequency_order,
    is_higher_frequency,
    normalize_frequency,
)
from ..panel.utils import (
    factorize_row_keys,
    get_entity_codes,
    normalize_entity_key,
    split_variable_key,
)
from ..utils._constants import FREQUENCY_LEVEL_NAME
from ._components.aggregation_constraint import (
    ConstraintSetting,
    resolve_aggregation_constraint,
    validate_aggregation_constraint,
)
from ._components.covariate_materializer import (
    COVARIATE_ELIGIBILITIES,
    COVARIATE_FALLBACKS,
    COVARIATE_STRATEGIES,
    DEFAULT_MATERIALIZATION_KEY as DEFAULT_FEATURE_KEY,
    CovariateMaterializer,
)
from ._components.imputation_plan import (
    INTERPOLATE_FALLBACK,
    ImputationPlan,
    ImputationStep,
    MaterializationWay,
    append_step,
    to_entity_tuple,
)
from ._components.imputation_window import (
    ImputationScope,
    ImputationWindowCalculator,
    TrainingScope,
)
from ._components.provenance import (
    CELL_ORIGIN_DTYPE,
    CellOrigin,
    PROVENANCE_DTYPE,
    ProvenanceType,
    Taint,
    max_cell_origin,
    max_origin,
    origin_to_taint,
    provenance_statistics,
)
from ._components.stage_scaler import ScaleFeatures, StageScaler
from ._components.training_set_builder import (
    TrainingSet,
    split_training_index,
)
from ._components.variable_orderer import FIT_PREDICT_ORDERS, VariableOrderer, VariableSpec
from ._engine.params import (
    validate_additive_transformer,
    validate_cv,
    validate_cv_scoring,
    validate_estimator,
    validate_intermediate_frequencies,
    validate_literal,
    validate_min_cv_train_size,
    validate_target_frequency_format,
    validate_unit_interval,
)
from ._engine.frequency_layout import (
    misaligned_anchor_pairs,
    plan_frequencies,
    stage_frequency_label,
    stage_frequency_of,
)
from ._engine.preparation import (
    PreparedData,
    prepare_work_frame,
    resolve_target_column_name,
)
from ._engine.run_context import FittedComponents, _RunContext
from ._engine.warnings import WarningAccumulator

# Type aliases
VariableCategory = Literal['aggregate', 'impute', 'target_freq']
EntityKey = Tuple[Any, ...]
IntermediateFrequencies = Literal[False, 'covariates_only', True]

# Valeurs admissibles des littéraux.
_IMPUTATION_SCOPES: Tuple[str, ...] = get_args(ImputationScope)
_TRAINING_SCOPES: Tuple[str, ...] = get_args(TrainingScope)
_FREQUENCY_MISMATCH_POLICIES: Tuple[str, ...] = ('error', 'warn')

# Paramètres admettant la forme dictionnaire par colonne
_PER_FEATURE_PARAMS: Tuple[str, ...] = (
    'estimator',
    'scale_features',
    'interpolation_method',
    'interpolation_anchor',
    'aggregation_constraint',
)

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

    **Training and prediction passes, once fitted with** ``y``: the imputer
    then **augments a training set** — ``X`` and ``y`` on the target grid — and
    is never meant to transform test data. An ``XYPipeline`` calls
    ``fit_transform`` on its steps during its own ``fit`` and ``transform``
    during ``predict`` / ``score``, and the two passes differ accordingly (as
    sklearn allows, e.g. ``TargetEncoder``):

    - ``fit_transform(X, y)`` is the **training pass**: ``X`` and ``y`` come
      out on the target grid, rows added by the grid included — no row is
      dropped, a downstream step may still shift ``X`` towards several
      horizons; dropping the rows without target is the job of
      :class:`tsforecast.xy.DropNaNTarget`, right before the estimator;
    - ``transform(X, y=None)`` is the **prediction pass**: only the columns of
      ``X`` are imputed, by the frozen models, on the **input rows** (flat
      output, whatever ``keep_lower_frequencies``, no added row), and ``y`` —
      when given, as ``score`` does — comes back **unchanged**: a test target
      is never imputed nor used.

    ``y`` is imputed like any column but is **not a covariate** of the other
    columns by default (``target_as_covariate=False``): an imputed feature
    built from the target carries its noise, inflates every in-sample metric
    and biases the downstream model, and the prediction pass has no ``y`` to
    feed such a model with. ``target_as_covariate=True`` restores it, at the
    price of a prediction pass that raises whenever a model needs ``y``.
    Without ``y`` at ``fit``, nothing of this applies: ``transform`` replays
    the plan on any data and ``fit_transform(X) == fit(X).transform(X)``.

    **Unit of the dates**: every date a pass returns — output, provenance
    matrix, inversion, and ``y`` too — is stored at the unit of the dates of
    the ``X`` of that pass (``'s'``, ``'ms'``, ``'us'`` or ``'ns'``); a ``y``
    stored at another unit comes back at ``X``'s, its values unchanged. The
    fitted state holds the unit of the fit input, and a ``transform`` on data
    stored at another unit returns the unit of those data, with the same
    values.

    **Limit — scoring a target coarser than the grid**: the model then
    predicts sub-period values, while the test target keeps its observed
    period totals; a score compares the two on different scales. Aggregate
    downstream to the target frequency — e.g. rolling three-month sums for a
    quarterly target, by an XY step aggregating ``y`` at the training pass only —
    before scoring (see ``docs/concepts/mixed_frequency_imputation.md``).

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

    **What the parameter space does not express**: a mode "one single fit,
    reused at the following stages with that stage's scale factor". Every
    stage fits its own models; there is no switch to refit the cascade or not.
    Such a reuse would introduce a fit/predict asymmetry. It is not to be
    confused with the single fit per (stage, variable), which shares a model
    between the plan steps of **one** stage only and never across stages.

    ``keep_lower_frequencies`` is a **pure display parameter**: it governs how
    the frequency levels of the output are stacked, never the logic — the
    values of the target level are the same either way. The levels stacked are
    the stages of the progression, so under
    ``impute_intermediate_frequencies=False`` there is **no intermediate level
    to stack**: the output carries the target level and nothing else.
    The frequency level sits on the entity side of the index, the shape:
    ``(frequency, date)`` on a time series, ``(entity..., 'frequency',
    'date')`` on a panel. The name ``'frequency'``
    (``tsforecast.utils._constants.FREQUENCY_LEVEL_NAME``) is **reserved**:
    ``fit`` and ``transform`` reject an input whose index level, ``panel_cols``
    entry or ``time_col`` already bears it, so the stacked index never holds
    two levels of one name. ``inverse_transform`` recognizes a stacked frame
    by its depth and reads the level by position, never by name.

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
      entity. ``scale_features`` alone may differ per entity within the
      mutualized set (see below): it makes the **scales** of the entities
      comparable, never their seasonal profiles — a variable seasonally
      adjusted by one entity and raw for another still teaches one model two
      behaviours.
    - **Provenance is contagious across entities**: an entity contributing
      ``'interpolated'`` or ``'model'`` cells degrades ``target_taint``,
      hence the provenance of **every** cell the stage produces — including
      those of the other entities.

    Detected frequencies are indexed **per (entity, column)**:
    on a panel the same column may carry a different frequency for
    each entity, and every reasoning about it is per entity. Per-feature
    hyperparameter dicts (``scale_features``, ``interpolation_method``,
    ``interpolation_anchor``, ``estimator``, ``aggregation_constraint``) stay
    keyed by **column name**, never by ``(entity, column)``. Each admits a
    ``'__default__'`` key covering the columns it does not name; a column
    covered by neither takes the parameter's default. A key naming no column
    of the data is ignored, with a **single** ``UserWarning`` at ``fit``
    listing every such key: a dict may thus be shared by datasets whose
    columns differ, and a typo still shows.

    ``scale_features`` alone admits **per-entity modes**, as a nested dict
    whose top-level keys remain column names — a variable published
    seasonally adjusted by one country and raw by another::

        scale_features={
            'pib': {'DE': 'calendar', '__default__': 'constant'},
            '__default__': 'constant',
        }

    The nested dict is **closed on itself**: an entity it does not name takes
    the nested ``'__default__'``, else ``'constant'``, never the top-level
    ``'__default__'``. It admits ``'constant'`` and ``'calendar'`` only:
    ``False`` applies to a whole column, never to some of its entities. Its
    keys name **full** entities, as the index carries them — ``'FR'`` or
    ``('FR',)`` on a single-level panel, ``('Zone euro', 'DE')`` on a
    two-level one. A partial key such as ``'Zone euro'`` matches **no**
    entity; it is reported by the same warning, and the entities it was
    meant for fall back on the nested ``'__default__'``.

    **Unusable (entity, column) pairs are left out, never fatal**: a pair
    observed fewer than twice has no detectable frequency, and a monthly,
    quarterly or yearly pair observed off the calendar bounds of its own
    frequency — a fiscal year closing in June on a period-end index, a yearly
    value dated 1 December on a period-start index — has no consistent
    calendar reading. Neither is imputed nor used as a covariate, so it never
    degrades the fit of the other variables; the second kind is named by a
    single ``UserWarning`` at ``fit``. A calendar value dated elsewhere comes
    back once re-dated at the end (period-end index) or at the start
    (period-start index) of its calendar period; a non-calendar period, such
    as a fiscal year, cannot be imputed by this class.

    **A prediction that produces nothing is a failed prediction**: a step
    whose model predicts only NaN is degraded into the interpolation
    fallback and named by the aggregated warning of the degraded steps;
    scattered NaN predictions stay unwritten cells, named by the same warning.
    An unanchored step has no fallback: its pairs are named by their own
    aggregated warning.

    Args:
        target_frequency: Target frequency, as a string applying to every
            entity, or a dict ``{entity: frequency}`` on a panel only. A
            dict must name **every** entity of the panel, or ``fit`` raises
            naming the missing ones; on a time series, ``fit`` refuses it.
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
        interpolation_method: Interpolation method, global or as a dict
            ``{column: method}`` with an optional ``'__default__'`` key; a
            column covered by neither is interpolated ``'linear'``.
        interpolation_anchor: Position of a value inside its period, in
            ``[0, 1]``, global or as a dict ``{column: anchor}`` with an
            optional ``'__default__'`` key. ``None`` — the default, and the
            value of a column covered by neither — keeps the detected
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
        target_as_covariate: Whether ``y``, when given at ``fit``, may be a
            covariate of the other imputed columns. ``False`` (default) keeps
            it out of every ``feature_cols`` — it is still imputed itself, from
            the columns of ``X`` — so that the prediction pass, which never
            sees ``y``, replays the same models. ``True`` lets it in; the
            prediction pass then raises if a model of a column of ``X`` needs
            it. Inert without ``y``.
        impute_unobserved_entities: Whether an entity that never observes an
            imputable column may receive a complete imputation of it, learned
            on the other entities of the panel. ``False`` (default) leaves
            such a pair out of every prediction grid: its cells stay NaN and
            unmarked (NaN in ``imputation_provenance_``). ``True`` makes
            the pair imputable at **every stage of its entity's
            progression**, through one dedicated plan
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
            nothing to interpolate, so the cells stay NaN and unmarked
            and a single aggregated warning names the pairs at the end of
            ``fit``. At ``transform`` the parameter also governs the pairs
            the fit never observed — every column of an entity absent from
            the fit, or a column a known entity left empty: under ``False``
            they stay as given; under ``True`` each is served as if it had
            been present at fit time, with the frozen model of its column —
            **anchored** on its own observations when the transform observes
            it (frequency detected on the transformed data, rescaling,
            ``MODEL_*``), unanchored otherwise.
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
            or a dict of these values keyed by column with an optional
            ``'__default__'`` key; a column covered by neither is scaled
            ``'constant'``. A value of the dict may itself be a per-entity
            dict ``{entity: 'constant' | 'calendar'}`` with an optional
            nested ``'__default__'``: closed on itself, without ``False``,
            keyed by full entities (see the class docstring).
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
            ``{'overall': {...}, column: {...}}`` (see
            :func:`~tsforecast.imputation._components.provenance.provenance_statistics`) of
            the **last pass**, like ``imputation_provenance_``: written by
            ``fit``, then overwritten by every ``transform``, which counts the
            target level of its output whatever ``keep_lower_frequencies``
            stacks. Exposed for tracking (see
            :func:`tsforecast.tracking.imputation_metrics`).
        imputation_plan_: :class:`ImputationPlan` — the complete fitted state.
        additive_transformer_: Fitted clone of ``additive_transformer``, the
            parameter itself staying unfitted; None without one.
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
        imputation_provenance_: Provenance matrix, of the categorical dtype
            ``PROVENANCE_DTYPE`` (one ``ProvenanceType`` value per cell, NaN
            for a cell not marked). Written by ``fit``, then
            overwritten by every ``transform``: it always describes the
            last pass, which is what ``inverse_transform`` undoes. It follows
            the shape of the output — stacked on the frequency level under
            ``keep_lower_frequencies=True``, flat otherwise — and ``fit``
            purges the transform's copy in its first phase. The matrix of a
            ``transform`` describes the cells of its output: a column
            observed more finely than its row holds the exact aggregation of
            the period there, marked ``AGGREGATED``, and an empty cell is
            never marked. A ``transform`` that raises writes none of the
            per-pass attributes.
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
        >>> imputed = imputer.fit(data).transform(data)
        >>> imputer.frequency_progression_
        ['M']
        >>> float(imputed.loc['M', 'a1'].loc['2022'].sum().round(1))
        132.0
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
        # --- Rôle de y ---
        target_as_covariate: bool = False,
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
        scale_features: ScaleFeatures = 'constant',
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
        self.target_as_covariate = target_as_covariate
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

        # Validation des paramètres, rejouée en tête du fit : "set_params"
        # écrit les attributs sans repasser par "__init__"
        self._validate_params()

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
    # Validation de l'ensemble des paramètres publics
    def _validate_params(self) -> None:
        """Validate every public parameter, as stored on the instance.

        Called by ``__init__`` and again at the head of ``fit``:
        ``set_params`` — the path of ``GridSearchCV`` — writes the attributes
        without going through ``__init__``, so ``fit`` must refuse what
        ``__init__`` refuses, with the same message. Nothing is normalized
        nor stored here.

        Raises:
            ValueError: If a parameter is not admissible. The message always
                names the parameter and lists the admissible values or forms.
            TypeError: If a parameter has the wrong type.
        """
        # Validation de la fréquence cible, de l'estimateur et du transformer
        validate_target_frequency_format(self.target_frequency)
        validate_estimator(self.estimator)
        if self.additive_transformer is not None:
            validate_additive_transformer(self.additive_transformer)

        # Validation des littéraux : message listant les valeurs admises
        validate_literal(
            'covariate_strategy', self.covariate_strategy, COVARIATE_STRATEGIES
        )
        validate_literal(
            'covariate_fallback', self.covariate_fallback, COVARIATE_FALLBACKS
        )
        validate_literal(
            'covariate_eligibility', self.covariate_eligibility,
            COVARIATE_ELIGIBILITIES
        )
        validate_literal(
            'fit_predict_order', self.fit_predict_order, FIT_PREDICT_ORDERS
        )
        validate_literal(
            'on_frequency_mismatch', self.on_frequency_mismatch,
            _FREQUENCY_MISMATCH_POLICIES
        )
        validate_literal(
            'imputation_scope', self.imputation_scope, _IMPUTATION_SCOPES
        )
        if self.training_scope is not None:
            validate_literal(
                'training_scope', self.training_scope, _TRAINING_SCOPES
            )

        # Validation des paramètres d'imputation des fréquences intermédiaires
        validate_intermediate_frequencies(self.impute_intermediate_frequencies)

        # Validation des formes par feature, déléguée aux validateurs
        # statiques des composants consommateurs : une seule implémentation,
        # donc aucune dérive possible entre ce que la classe accepte et ce que
        # le composant accepte
        CovariateMaterializer._validate_interpolation_method(self.interpolation_method)
        CovariateMaterializer._validate_interpolation_anchor(self.interpolation_anchor)
        StageScaler._validate_scale_features(self.scale_features)
        validate_aggregation_constraint(self.aggregation_constraint)

        # Validation de la stratégie de validation croisée. "check_cv" n'est
        # appelé qu'au fit.
        validate_cv(self.cv)
        validate_cv_scoring(self.cv_scoring)

        # Validation des bornes numériques. Les entiers numpy sont admis
        # (grille construite par "np.arange"), les booléens jamais
        validate_min_cv_train_size(self.min_cv_train_size)
        validate_unit_interval('coverage_threshold', self.coverage_threshold)
        if self.training_coverage_threshold is not None:
            validate_unit_interval(
                'training_coverage_threshold', self.training_coverage_threshold
            )

        # Validation groupée des booléens : un 'frequency'/1/None passé par
        # erreur se propagerait sinon silencieusement jusqu'à un "if" qui
        # l'évalue. Le booléen numpy est admis, au même titre que par sklearn
        boolean_params = {
            'keep_lower_frequencies': self.keep_lower_frequencies,
            'restore_original_values': self.restore_original_values,
            'verbose': self.verbose,
            'impute_unobserved_entities': self.impute_unobserved_entities,
            'target_as_covariate': self.target_as_covariate,
        }
        for param_name, param_value in boolean_params.items():
            if not isinstance(param_value, (bool, np.bool_)):
                raise TypeError(
                    f"{param_name} must be a bool, got {type(param_value).__name__}"
                )

    # Méthode auxiliaire de contrôle des clés des dictionnaires par feature
    def _warn_unknown_feature_keys(
        self,
        columns: Iterable[Any],
        entities: Optional[Sequence[EntityKey]] = None,
    ) -> None:
        """Warn once about per-feature dict keys naming nothing in the data.

        The per-feature dicts (``estimator``, ``scale_features``,
        ``interpolation_method``, ``interpolation_anchor``,
        ``aggregation_constraint``) are keyed by column name. A key naming
        no column of the data is ignored — the column falls back to the
        ``'__default__'`` entry or to the default — with a **single**
        ``UserWarning`` listing every offending key per parameter: an error
        would break a dict shared by datasets whose columns differ (feature
        selection upstream, search over feature subsets), silence would hide
        a typo. An ``(entity, column)`` key is never a column name: settings
        are per column.

        ``scale_features`` alone admits per-entity modes, as a nested dict
        ``{column: {entity: mode}}``. A nested key naming no entity of
        the data — a partial key of a multi-level panel included, or any
        entity key on a time series — is reported in the same warning, and
        that entity falls back on the nested ``'__default__'``.

        Args:
            columns: Columns of the working frame, the target included.
            entities: Normalized entities of the panel, None on a time series.
        """
        # Ensemble des colonnes connues
        known = set(columns)
        # Recensement des clés inconnues, paramètre par paramètre
        unknown_by_param: Dict[str, List[Any]] = {}
        for name in _PER_FEATURE_PARAMS:
            setting = getattr(self, name)
            if not isinstance(setting, dict):
                continue
            unknown = [
                key for key in setting
                if key != DEFAULT_FEATURE_KEY and key not in known
            ]
            if unknown:
                unknown_by_param[name] = sorted(unknown, key=repr)

        # Entités inconnues des modalités par entité de "scale_features",
        # comparées sous forme normalisée : 'FR' et ('FR',) sont une entité
        known_entities = set(entities or ())
        unknown_entities: Dict[Any, List[Any]] = {}
        if isinstance(self.scale_features, dict):
            for column, modes in self.scale_features.items():
                if not isinstance(modes, Mapping):
                    continue
                unknown = [
                    key for key in modes
                    if key != DEFAULT_FEATURE_KEY
                    and normalize_entity_key(key) not in known_entities
                ]
                if unknown:
                    unknown_entities[column] = sorted(unknown, key=repr)

        if not unknown_by_param and not unknown_entities:
            return

        # Avertissement unique, listant toutes les clés ignorées
        parts: List[str] = []
        if unknown_by_param:
            details = '; '.join(
                f"{name}: {keys}" for name, keys in unknown_by_param.items()
            )
            parts.append(
                f"Per-feature settings name columns absent from the data and are "
                f"ignored ({details}); these columns fall back to the "
                f"'{DEFAULT_FEATURE_KEY}' entry or to the default."
            )
        if unknown_entities:
            details = '; '.join(
                f"{column!r}: {keys}" for column, keys in unknown_entities.items()
            )
            parts.append(
                f"scale_features names entities absent from the data and they "
                f"are ignored ({details}); these entities fall back to the nested "
                f"'{DEFAULT_FEATURE_KEY}' entry or to 'constant'. Entity keys "
                "name full entities: a partial key of a multi-level panel "
                "matches none."
            )
        if any(
            isinstance(key, tuple)
            for keys in unknown_by_param.values() for key in keys
        ):
            parts.append(
                "Keys are column names, never (entity, column) pairs: "
                "scale_features takes per-entity modes as a nested dict "
                "{column: {entity: mode}}; fit one imputer per entity for the "
                "other per-entity settings."
            )
        warnings.warn(' '.join(parts), UserWarning)

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
        class, which sets ``is_panel_``, ``n_features_in_`` and
        ``feature_names_in_`` before ``_fit`` runs: an interrupted fit would then
        look fitted.

        Returns:
            True once all fitted attributes are present.
        """
        # Liste explicite plutôt que la convention du suffixe
        return all(hasattr(self, attr) for attr in _FITTED_ATTRIBUTES)

    # Contrôle des colonnes au transform propre à l'imputeur
    def _check_feature_names(self, X: pd.DataFrame) -> None:
        """Skip the column check of the base class: the imputer has its own contract.

        At ``transform``, a column seen at ``fit`` and now missing is refused by
        ``_check_transform_frequencies``, which names it; an extra column belongs to
        no plan and is carried through untouched. The strict equality of the base
        class would refuse the extra columns.

        Args:
            X: Data passed to ``transform``, after validation.
        """
        # Contrat de colonnes appliqué par le rejeu du plan
        return None

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
    # Fenêtres
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de sélection des colonnes de features
    def _select_feature_columns(
        self,
        materializer: CovariateMaterializer,
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
            materializer: Materializer of the pass, carrier of
                ``covariate_eligibility``.
            candidates: Candidate covariate columns, in output order.
            training_frame: Frame restricted to the training grid.
            prediction_frame: Frame restricted to the prediction grid.

        Returns:
            Kept columns, in the order of ``candidates``.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from tsforecast.imputation import HighFrequencyImputer
            >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
            >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
            >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0) * 3 + 10
            >>> imputer = HighFrequencyImputer(
            ...     target_frequency='M', estimator=LinearRegression()
            ... ).fit(data)
            >>> pred_frame = data.assign(q1=np.nan)
            >>> materializer = imputer._fitted_components_.new_materializer()
            >>> imputer._select_feature_columns(materializer, ['m1', 'q1'], data, pred_frame)
            ('m1',)
        """
        # Éligibilité par entité, portée par le matérialiseur
        eligible = set(
            materializer.eligible_columns(candidates, prediction_frame)
        )

        # Non-vacuité sur les deux fenêtres : une colonne vide sur l'une des
        # deux grilles n'apporte aucune information exploitable. Les deux
        # trames d'étape portent toutes les colonnes du jeu de travail
        return tuple(
            column for column in candidates
            if column in eligible
            and training_frame[column].notna().any()
            and prediction_frame[column].notna().any()
        )

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
    # Fit
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de purge de l'état de la dernière passe
    def _purge_pass_state(self) -> None:
        """Drop the per-pass state a previous ``transform`` wrote.

        The provenance matrix, its statistics and the input snapshot always
        describe the last pass: ``fit`` purges those of a previous transform,
        then writes its own.
        """
        # Purge de l'état écrit par un précédent "transform"
        for stale_attr in (
            'imputation_provenance_', 'provenance_statistics_', '_original_X_',
            '_original_y_', '_stacked_index_nlevels',
        ):
            if stale_attr in self.__dict__:
                delattr(self, stale_attr)

    # Méthode auxiliaire de configuration de l'ordonnanceur du fit
    def _configure_variable_orderer(self) -> VariableOrderer:
        """Build the orderer of the fit and publish the attributes of the order.

        The orderer depends on the parameters alone. ``cv_`` exists only under
        ``fit_predict_order='cv'``; ``imputation_order_`` and
        ``imputation_cv_scores_`` start empty, filled by phase 5 under
        ``covariate_strategy='model'``.

        Returns:
            The orderer, which joins the execution context of the fit.
        """
        # Résolution de la stratégie de validation croisée : "cv_" n'existe
        # que sous fit_predict_order='cv'. L'ordonnanceur ne sert qu'au fit :
        # il rejoint le contexte d'exécution de la passe
        variable_orderer = VariableOrderer(
            fit_predict_order=self.fit_predict_order,
            cv=self.cv,
            cv_scoring=self.cv_scoring,
            min_cv_train_size=self.min_cv_train_size,
        )
        if self.fit_predict_order == 'cv':
            # Hors covariate_strategy='model', l'ordre et ses paramètres sont
            # inertes : l'avertissement croisé de l'ordonnanceur sur
            # "min_cv_train_size" n'y a pas d'objet et est étouffé
            with warnings.catch_warnings():
                if self.covariate_strategy != 'model':
                    warnings.simplefilter('ignore', UserWarning)
                variable_orderer.fit()
            self.cv_ = variable_orderer.cv_
        else:
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
        return variable_orderer

    # Méthode auxiliaire de publication de la préparation du fit
    def _publish_fit_preparation(self, X: pd.DataFrame, prepared: PreparedData) -> None:
        """Publish the fitted attributes the preparation of the fit derived.

        Columns, structure, target name and entities; frequency layout and
        the attributes read on it; components shared by every pass; window
        masks and bounds; fitted additive transformer; frequency progression.

        Args:
            X: Features of the fit.
            prepared: Preparation of the fit (``fitting=True``).
        """
        # Préparation du fit
        fit = prepared.fit
        # Colonnes d'intérêt, structure des données, nom de la cible, entités
        self.feature_columns_ = list(X.columns)
        self.is_panel_ = fit.is_panel
        self.target_column_ = prepared.target_column
        self.entities_ = fit.entities
        # Label de fréquence de l'index d'entrée, lu à l'inversion
        self._source_index_frequency_label = fit.source_index_frequency_label

        # Plan des fréquences, lu tel quel pour l'exécution des étapes et pour
        # chaque rejeu. Attributs ajustés publiés à partir du plan : les mêmes objets
        layout = fit.layout
        self._frequency_layout_ = layout
        self.effective_target_frequency_ = layout.target_frequency
        self.detected_frequencies_ = layout.detected_frequencies
        self.variable_categories_ = layout.categories

        # Composants configurés une fois pour toutes et relus tels quels par
        # chaque passe : la contrainte d'agrégation (elle recale les
        # prédictions des étapes et, injectée dans le matérialiseur, les
        # covariables interpolées — une seule implémentation, aucune dérive),
        # le metteur à l'échelle des étapes et la configuration du
        # matérialiseur. Chaque passe reçoit son propre matérialiseur, donc des
        # registres vides : un fit ne lit jamais le miroir d'un précédent
        self._fitted_components_ = FittedComponents.build(
            covariate_strategy=self.covariate_strategy,
            covariate_fallback=self.covariate_fallback,
            covariate_eligibility=self.covariate_eligibility,
            interpolation_method=self.interpolation_method,
            interpolation_anchor=self.interpolation_anchor,
            aggregation_constraint=self.aggregation_constraint,
            scale_features=self.scale_features,
            column_frequencies=layout.frequencies_by_column(),
        )

        # Trois masques de fenêtre, et bornes lisibles : chaque attribut porte
        # celles de son masque
        self.strict_window_mask_ = fit.strict_window_mask
        self.imputation_window_mask_ = fit.imputation_window_mask
        self.training_window_mask_ = fit.training_window_mask
        self.imputation_window_ = fit.imputation_window
        self.training_window_ = fit.training_window

        # Transformateur additif ajusté, None sans transformateur
        self.additive_transformer_ = fit.additive_transformer

        # Progression des fréquences à imputer, établie par le plan des
        # fréquences : elle ne dépend que des fréquences détectées et de la cible
        self.frequency_progression_ = layout.progression
        # Logging
        self._log(
            f"[fit] Frequency progression: "
            f"{[stage_frequency_label(f) for f in self.frequency_progression_]}"
        )

    # Méthode auxiliaire d'entraînement
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Learn the imputation plan from X and y.

        Phases of the fitting logic:

        - PHASE 0: parameters — revalidation, purge of the last pass, warning
          without an estimator, orderer of the variables.
        - PHASES 0-4: preparation, the same code as the transform's
          (:func:`~tsforecast.imputation._engine.preparation.prepare_work_frame`,
          ``fitting=True``): ``y`` alignment and naming, working frame,
          frequency layout
          (:func:`~tsforecast.imputation._engine.frequency_layout.plan_frequencies`),
          the three window masks, additive transformer fitted, provenance
          tracker initialized after it — then published as fitted attributes,
          with the components every pass shares.
        - PHASE 4: execution context of the fit (:class:`_RunContext`),
          dropped at the end of the pass.
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
                ``target_frequency`` dict is given on a time series or
                misses entities, if the
                indices of X and y disagree, if an index level bears
                the reserved name ``'frequency'``, or if the window
                calculator cannot be fitted — an empty window is not a
                failure, it imputes nothing and is warned about.
        """
        # PHASE 0 — Paramètres revalidés ("set_params" contourne "__init__"), précédente passe purgée
        self._validate_params()
        self._purge_pass_state()

        # PHASE 0 — Configuration sans données : avertissement sans estimateur, ordonnanceur
        if self.estimator is None:
            warnings.warn(
                "No estimator was provided (estimator=None): every variable "
                "will fall back to linear interpolation.",
                UserWarning
            )
        variable_orderer = self._configure_variable_orderer()

        # PHASES 0-4 — Préparation commune au transform, publiée en attributs ajustés
        prepared = prepare_work_frame(self, X, y, fitting=True)
        self._publish_fit_preparation(X, prepared)

        # PHASE 4 — Contexte d'exécution du fit, abandonné en fin de passe. Le
        # constructeur du jeu d'entraînement reçoit un callable au "kind" nommé
        # ici, jamais le calculateur : il ne lit que le masque d'entraînement
        run = _RunContext.for_fit(
            self._fitted_components_,
            window_calculator=prepared.window_calculator,
            provenance_tracker=prepared.provenance_tracker,
            variable_orderer=variable_orderer,
            training_mask=partial(self._training_mask_at, prepared.window_calculator),
            log=self._log if self.verbose else None,
        )

        # PHASE 5 — Exécution des étapes, chacune construite PUIS exécutée (l'étape
        # k dépend des imputations de l'étape k-1), recalages sous un bloc différé :
        # un avertissement par cas dégénéré, jamais un par variable × étape
        self.imputation_plan_ = ImputationPlan()
        with run.aggregation_constraint.deferred_warnings():
            for stage_freq in self.frequency_progression_:
                self._execute_stage(run, prepared.X_work, stage_freq)
        # Avertissements de la phase, un seul message par famille : étapes
        # dégradées, puis couples sans ancre restés vides
        run.pending_warnings.emit()

        # PHASE 6 — Finalisation : provenance et statistiques de la passe, couples
        # imputés sans ancre (toujours écrits, vides compris, pour "check_is_fitted"),
        # instantané de l'entrée, relu par "inverse_transform" comme dernière passe
        self.imputation_provenance_ = run.provenance_tracker.get_provenance_matrix()
        self.provenance_statistics_ = run.provenance_tracker.compute_statistics()
        self.unanchored_pairs_ = tuple(sorted(run.unanchored_written, key=repr))
        self._original_X_ = prepared.snapshot_X
        self._original_y_ = prepared.snapshot_y

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
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from tsforecast.imputation import HighFrequencyImputer
            >>> imputer = HighFrequencyImputer(
            ...     target_frequency='M', estimator={'a1': LinearRegression()}
            ... )
            >>> imputer._estimator_for('a1')
            LinearRegression()
            >>> imputer._estimator_for('m1') is None
            True
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
        run: _RunContext,
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
            run: Execution context of the pass, which provides the window
                calculator and the pass name used in the log.
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
            mask = run.window_calculator.get_mask_at_frequency(stage_freq, kind=kind)
        except (ValueError, KeyError, TypeError) as error:
            self._log(
                f"[{run.pending_warnings.pass_name}] {kind} mask unavailable "
                f"({error}); every row kept"
            )
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
        run: _RunContext,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
        entities: Optional[Sequence[EntityKey]],
    ) -> pd.Index:
        """Build the prediction grid of a stage, restricted to one group.

        The grid spans the whole imputation window of the entities, anchor
        rows included: a period holding no observation is
        predicted like any other.

        Args:
            run: Execution context of the pass.
            X_work: Working frame.
            stage_freq: Frequency of the stage.
            entities: Entities of the group, None for a time series.

        Returns:
            Index of the rows the step writes on.
        """
        # Masque d'imputation, "kind" nommé explicitement
        mask = self._stage_mask(run, X_work, stage_freq, kind='imputation')
        grid = mask.index[mask.to_numpy(dtype=bool)]
        return self._restrict_to_entities(grid, entities)

    # Méthode auxiliaire de la grille non restreinte d'un groupe
    def _unrestricted_grid(
        self,
        run: _RunContext,
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
            run: Execution context of the pass.
            X_work: Working frame.
            stage_freq: Frequency of the stage.
            entities: Entities of the group, None for a time series.

        Returns:
            Index of the rows of those entities at the stage frequency.
        """
        # Index du masque, pris sans sa valeur booléenne : la grille de l'étape
        mask = self._stage_mask(run, X_work, stage_freq, kind='imputation')
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
        stage_scaler: StageScaler,
        *,
        feature_cols: Sequence[str],
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
        ways: Dict[str, MaterializationWay],
        blocks: Dict[EntityKey, str],
        stage_freq: Union[str, Dict[EntityKey, str]],
        row_frequency: pd.Series,
        real_index: pd.Index,
        entity_codes: np.ndarray,
        entities: Sequence[EntityKey],
    ) -> pd.DataFrame:
        """Divide the covariates by the period their own row spans.

        The grid of a mutualized training set is homogeneous only as long as
        axis 2 injects nothing into it: a cell produced at an earlier, finer
        stage sits on a row of that stage's frequency, not of its block's. The
        covariates of such a row are materialized at the row frequency
        (:class:`TrainingSetBuilder`), so their divisor must be read there
        too — the very same unifying rule the target divisor follows.

        One call to :meth:`StageScaler.feature_divisors` per layer of rows
        sharing one frequency per entity, never one per row: a grid where
        axis 2 injected nothing is a single layer, hence a single call at the
        block binding.

        Args:
            stage_scaler: Stage scaler of the pass.
            feature_cols: Covariate columns to scale.
            freqs_by_column: Detected frequencies, keyed by column.
            ways: Materialization way retained for each covariate.
            blocks: Block frequency of each contributing entity.
            stage_freq: Frequency of the stage.
            row_frequency: Frequency each training row was produced at,
                indexed on the training grid — stamped with a frequency level
                or not.
            real_index: The ``(entity..., date)`` grid of the same rows, the
                only shape :class:`StageScaler` ever reads.
            entity_codes: Entity code of each row, decomposed once by
                :func:`split_training_index`.
            entities: Entity keys the codes point to.

        Returns:
            A ``DataFrame`` of per-row divisors, indexed like
            ``row_frequency`` and columned like ``feature_cols``.
        """
        # Extraction de l'index
        index = row_frequency.index

        # Un appel par couche, puis remontée au format par ligne, seul capable
        # de porter deux échelles sur une même colonne
        divisors = pd.DataFrame(
            np.nan, index=index, columns=list(feature_cols), dtype=float
        )
        for layer, in_layer in self._row_layers(entity_codes, entities, row_frequency, blocks):
            # Écriture positionnelle : sous une grille estampillée, deux lignes
            # partagent une date et l'alignement par étiquette les confondrait
            positions = np.flatnonzero(in_layer)
            sub_real = real_index[in_layer]
            sub_divisors = stage_scaler.feature_divisors(
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
        entity_codes: np.ndarray,
        entities: Sequence[EntityKey],
        row_frequency: pd.Series,
        blocks: Dict[EntityKey, str],
    ) -> List[Tuple[Dict[EntityKey, str], np.ndarray]]:
        """Enumerate the layers of a training grid, one frequency per entity.

        Same rule as :meth:`TrainingSetBuilder._frequency_layers`, read here
        on the assembled grid rather than on the per-entity frames: the block
        frequency opens each entity's list, so a grid where axis 2 injected
        nothing yields exactly one layer. The rows are read through integer
        codes of their entity and frequency: one pass per distinct (entity,
        frequency) pair, never per row.

        Args:
            entity_codes: Entity code of each row, in grid order.
            entities: Entity keys the codes point to.
            row_frequency: Production frequency of each row, in grid order.
            blocks: Block frequency of each contributing entity.

        Returns:
            Ordered list of ``(layer, selector)`` pairs, the selector being a
            boolean array over the grid. Empty layers are dropped.

        Examples:
            >>> layers = HighFrequencyImputer._row_layers(
            ...     np.array([0, 0, 1]), [('FR',), ('DE',)],
            ...     pd.Series(['Y', 'Q', 'Q'], dtype='str'), {('FR',): 'Y', ('DE',): 'Q'},
            ... )
            >>> [(layer, selector.tolist()) for layer, selector in layers]
            [({('FR',): 'Y', ('DE',): 'Q'}, [True, False, True]), ({('FR',): 'Q'}, [False, True, False])]
        """
        # Codes des fréquences de ligne ; une fréquence manquante n'appartient
        # à aucune couche
        freq_codes, freq_labels = pd.factorize(row_frequency, use_na_sentinel=True)
        label_codes = {label: code for code, label in enumerate(freq_labels)}

        # Fréquences de chaque entité, celle de son bloc en tête, puis celles
        # de ses lignes dans l'ordre de première apparition des couples
        per_entity: Dict[int, List[str]] = {}
        _pair_codes, first_rows = factorize_row_keys(entity_codes, freq_codes)
        for row in first_rows:
            entity_code = int(entity_codes[row])
            entity = entities[entity_code]
            ordered = per_entity.setdefault(
                entity_code,
                [blocks[entity]] if entity in blocks else [],
            )
            if freq_codes[row] >= 0 and freq_labels[freq_codes[row]] not in ordered:
                ordered.append(freq_labels[freq_codes[row]])

        # Couches : la i-ème fréquence de chaque entité qui en porte une ; une
        # ligne en fait partie si sa fréquence est celle de son entité au rang i
        layers: List[Tuple[Dict[EntityKey, str], np.ndarray]] = []
        depth = max((len(f) for f in per_entity.values()), default=0)
        for rank in range(depth):
            layer = {
                entities[entity_code]: ordered[rank]
                for entity_code, ordered in per_entity.items()
                if rank < len(ordered)
            }
            # Code de fréquence attendu par entité (-2 : aucun, jamais égal)
            expected = np.full(len(entities), -2, dtype=np.intp)
            for entity_code, ordered in per_entity.items():
                if rank < len(ordered):
                    expected[entity_code] = label_codes.get(ordered[rank], -2)
            in_layer = (freq_codes >= 0) & (freq_codes == expected[entity_codes])
            if in_layer.any():
                layers.append((layer, in_layer))
        return layers

    # Méthode auxiliaire de calcul des diviseurs de cible par fréquence de ligne
    def _target_divisors_per_row(
        self,
        stage_scaler: StageScaler,
        *,
        column: str,
        blocks: Dict[EntityKey, str],
        stage_freq: Union[str, Dict[EntityKey, str]],
        row_frequency: pd.Series,
        real_index: pd.Index,
        entity_codes: np.ndarray,
        entities: Sequence[EntityKey],
    ) -> pd.Series:
        """Divide the target by the period its own row spans.

        :meth:`StageScaler.target_divisor` with ``produced_freq``, called
        layer by layer: under a stamped grid two rows share a date, and the
        scaler — which groups by ``(stage frequency, production frequency)``
        and writes its results back by label — would set both. Each layer is
        handed the real, duplicate-free ``(entity..., date)`` sub-grid, and
        the results are written back positionally. A grid where axis 2
        injected nothing is a single layer, hence a single call.

        Args:
            stage_scaler: Stage scaler of the pass.
            column: Column being imputed, for scale-mode resolution.
            blocks: Block frequency of each contributing entity.
            stage_freq: Frequency of the stage.
            row_frequency: Frequency each training row was produced at.
            real_index: The ``(entity..., date)`` grid of the same rows.
            entity_codes: Entity code of each row.
            entities: Entity keys the codes point to.

        Returns:
            A ``Series`` of per-row divisors indexed like ``row_frequency``
            (all ``1.0`` when the target is exempt from scaling).
        """
        # Extraction de l'index et des fréquences de ligne
        index = row_frequency.index
        frequencies = row_frequency.array

        # Un appel par couche, sur la grille réelle
        divisors = pd.Series(np.nan, index=index, dtype=float)
        for layer, in_layer in self._row_layers(entity_codes, entities, row_frequency, blocks):
            positions = np.flatnonzero(in_layer)
            sub_real = real_index[in_layer]
            produced = pd.Series(frequencies[in_layer], index=sub_real)
            sub_divisors = stage_scaler.target_divisor(
                column,
                source_freq=self._block_binding(layer, sub_real),
                pred_freq=stage_freq,
                index=sub_real,
                produced_freq=produced,
            )
            divisors.iloc[positions] = np.asarray(sub_divisors, dtype=float)
        return divisors

    # Méthode auxiliaire d'ordonnancement des colonnes d'une étape
    def _order_columns(
        self,
        run: _RunContext,
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
            run: Execution context of the fit.
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
                else self._frequency_layout_.entity_target_frequency(entities[0] if entities else None)
            )
            specs[column] = VariableSpec(
                name=column,
                frequency=lowest,
                entities=entities if self.is_panel_ else None,
            )

        # Ajustement paresseux de l'ordonnanceur, sous l'ordre 'frequency'
        # seulement — la phase 0 l'ajuste déjà sous 'cv' : l'avertissement
        # croisé qu'il émet ne concerne que la validation croisée et n'aurait
        # ici aucun sens
        if not hasattr(run.variable_orderer, 'cv_'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                run.variable_orderer.fit()

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
                    run,
                    column=column,
                    entities=group_entities if self.is_panel_ else None,
                    stage_freq=stage_freq,
                    X_work=X_work,
                    stage_frame=stage_frame,
                    freqs_by_column=freqs_by_column,
                )
                training_sets[column] = (prepared.X_train, prepared.y_train)

        # Les lignes scorées sont celles du jeu mutualisé, déjà restreintes à la fenêtre 'training'
        ordered = list(run.variable_orderer.order(
            specs,
            estimator=self.estimator,
            log=self._log if self.verbose else None,
            training_sets=training_sets,
        ))
        self.imputation_order_[stage_label] = list(ordered)
        # Report des scores CV qui ont décidé du rang, pour le tracking. Sous
        # l'ordre 'frequency' l'ordonnanceur laisse "scores_" vide
        stage_scores = dict(getattr(run.variable_orderer, 'scores_', {}) or {})
        if stage_scores:
            self.imputation_cv_scores_[stage_label] = stage_scores
        return ordered

    # Méthode d'exécution d'une étape de fréquence
    def _execute_stage(
        self,
        run: _RunContext,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
    ) -> None:
        """Run one frequency stage of the progression.

        Three moves, in this exact order: the stage frame, the imputable
        variables and their order, then one pass per variable.

        Args:
            run: Execution context of the fit.
            X_work: Working frame, after the additive transformer.
            stage_freq: Frequency of the stage.
        """
        # Étiquette lisible de l'étape, clé du plan et des journaux
        stage_label = stage_frequency_label(stage_freq)
        # Détection des fréquences
        freqs_by_column = self._frequency_layout_.frequencies_by_column(run.transform_frequencies)

        # Groupes imputables : un par (colonne, fréquence source)
        groups = self._frequency_layout_.imputable_groups(
            stage_freq, impute_unobserved_entities=self.impute_unobserved_entities
        )
        if not groups:
            self._log(f"[fit] stage {stage_label}: no imputable variable")
            return

        # 5a. Frame d'étape : données d'origine, agrégations exactes et miroir
        stage_mask = self._stage_mask(
            run, X_work, stage_freq, kind='imputation'
        )
        stage_frame = run.materializer.stage_frame(
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
            run,
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
                run,
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
        run: _RunContext,
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
            run: Execution context of the fit.
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
        builder = run.training_set_builder
        # Matérialiseur des résultats des étapes précédentes
        materializer = run.materializer
        # Origines acceptées pour la target et les covariables
        eligible = ELIGIBLE_ORIGINS[self.impute_intermediate_frequencies]

        # Départage des cellules coïncidentes, lu par colonne : le
        # paramètre admet une forme dictionnaire, la valeur n'est donc jamais
        # globale. Le constructeur du jeu ne reçoit que ce booléen : il ignore
        # tout de "aggregation_constraint", comme il ignore tout de l'axe 2
        keep_coincident = resolve_aggregation_constraint(
            self.aggregation_constraint, column
        ) is None

        # Fréquences remises au constructeur : la colonne cible par entité,
        # couples indétectables explicités, pour qu'aucun bloc n'emprunte la
        # fréquence des autres entités
        training_freqs = self._frequency_layout_.training_frequencies(column, freqs_by_column)

        # Jeu mutualisé, sonde sans covariable : elle ne matérialise rien et
        # rend déjà les blocs, la cible brute et la grille d'entraînement
        probe = builder.build(
            column=column,
            feature_cols=(),
            stage_freq=stage_freq,
            detected_frequencies=training_freqs,
            source_data=X_work,
            eligible_origins=eligible,
            keep_coincident_cells=keep_coincident,
        )
        blocks = dict(probe.blocks)

        # Grille de prédiction des entités concernées
        pred_grid = self._prediction_grid(
            run, X_work, stage_freq, entities
        )

        # Vues des deux fenêtres, chacune à sa fréquence : la grille
        # d'entraînement au pas de chaque couche de lignes, la grille de
        # prédiction au pas de l'étape
        train_view = self._training_view(
            materializer, probe, blocks, stage_freq, freqs_by_column, X_work
        )
        pred_view = stage_frame.reindex(pred_grid)

        # Sélection des covariables : disponibilité sur les deux fenêtres, puis
        # éligibilité par entité. Seules les colonnes dotées d'une fréquence
        # détectée pour au moins une entité sont candidates : sans fréquence
        # (une seule observation, ancres hors de leur grille calendaire), une
        # colonne ne peut être ni agrégée ni interpolée et n'apporterait que
        # des NaN — un couple inutilisable ne fait pas échouer l'ajustement des
        # autres. La cible y n'est candidate que sous target_as_covariate=True :
        # la passe de prédiction ne la voit jamais, un modèle qui la lirait n'y
        # serait pas rejouable
        excluded_target = None if self.target_as_covariate else self.target_column_
        candidates = [
            name for name in X_work.columns
            if name != column and name != excluded_target and name in freqs_by_column
        ]
        feature_cols = self._select_feature_columns(
            materializer, candidates, train_view, pred_view
        )

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
            detected_frequencies=training_freqs,
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
            real_index, entity_codes, entities = split_training_index(
                training.X.index,
                has_frequency_level=training.has_frequency_level,
            )
            y_train = run.stage_scaler.apply(
                y_train,
                self._target_divisors_per_row(
                    run.stage_scaler,
                    column=column,
                    blocks=dict(training.blocks),
                    stage_freq=stage_freq,
                    row_frequency=training.row_frequency,
                    real_index=real_index,
                    entity_codes=entity_codes,
                    entities=entities,
                ),
            )
            if feature_cols:
                X_train = run.stage_scaler.apply(
                    X_train,
                    self._feature_divisors_per_row(
                        run.stage_scaler,
                        feature_cols=feature_cols,
                        freqs_by_column=freqs_by_column,
                        ways=ways,
                        blocks=dict(training.blocks),
                        stage_freq=stage_freq,
                        row_frequency=training.row_frequency,
                        real_index=real_index,
                        entity_codes=entity_codes,
                        entities=entities,
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

    # Méthode auxiliaire de la vue d'entraînement, lue couche par couche
    def _training_view(
        self,
        materializer: CovariateMaterializer,
        probe: TrainingSet,
        blocks: Dict[EntityKey, str],
        stage_freq: Union[str, Dict[EntityKey, str]],
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
        X_work: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the frame the covariate selection reads on the training grid.

        Read layer by layer: a row produced at an earlier stage,
        finer than its block, is read at its own frequency and never at the
        block's — read at the block frequency, a covariate full on those rows
        would look empty there and be dropped. A single layer, the case as
        soon as axis 2 injects nothing, gives the block binding on the whole
        grid.

        Args:
            materializer: Materializer of the pass, sole producer of the
                stage frames.
            probe: Covariate-free training set of the variable, carrying the
                rows, their production frequency and the blocks.
            blocks: Block frequency of each contributing entity.
            stage_freq: Frequency of the stage, used when there is no row.
            freqs_by_column: Detected frequencies, keyed by column.
            X_work: Working frame.

        Returns:
            The stage frames of every layer, stacked; rows are never written
            back, only read for their non-emptiness.
        """
        # Grille réelle de la sonde et entité de chaque ligne : le niveau de
        # fréquence n'est jamais lu par le matérialiseur
        real_index, entity_codes, entities = split_training_index(
            probe.X.index, has_frequency_level=probe.has_frequency_level
        )

        # Une trame par couche, sur sa sous-grille dédoublonnée : deux cellules
        # coïncidentes partagent par définition une date
        frames: List[pd.DataFrame] = []
        if blocks:
            layers = self._row_layers(entity_codes, entities, probe.row_frequency, blocks)
            for layer, in_layer in layers:
                sub_grid = real_index[in_layer]
                sub_grid = sub_grid[~sub_grid.duplicated()]
                frames.append(materializer.stage_frame(
                    grid_index=sub_grid,
                    stage_freq=self._block_binding(layer, sub_grid),
                    detected_frequencies=freqs_by_column,
                    source_data=X_work,
                ))

        # Aucune ligne : trame vide à la forme de l'étape
        if not frames:
            return materializer.stage_frame(
                grid_index=real_index[:0],
                stage_freq=stage_freq,
                detected_frequencies=freqs_by_column,
                source_data=X_work,
            )
        return pd.concat(frames)

    # Méthode d'ajustement d'une variable à une étape
    def _fit_variable(
        self,
        run: _RunContext,
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
            run: Execution context of the fit.
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
            run,
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
            max_cell_origin(training.row_origin)
        )

        # Matérialisation de la grille de prédiction en mode rejeu des voies
        # retenues sur la grille d'entraînement. Elle ne sert qu'à la souillure
        # des covariables : aucune donnée d'entraînement n'en sort, ce qui est
        # la raison pour laquelle la 5b n'en a pas besoin pour ordonner
        pred_origins: Dict[str, CellOrigin] = {}
        if feature_cols and len(training) > 0:
            _X_pred, _ways, pred_origins = run.materializer.materialize(
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

        # Ajustement unique, quel que soit le nombre de groupes. Grille de
        # prédiction vide et aucun groupe sans ancre (qui prédit, lui, hors
        # fenêtre) : ce fit n'écrit rien, et un repli éventuel n'est pas
        # annoncé — l'absence de fenêtre d'imputation l'est déjà par le
        # calculateur de fenêtre
        writes_something = len(pred_grid) > 0 or any(
            source_frequency is None for _column, source_frequency in groups
        )
        model, is_fallback = self._fit_estimator(
            run.pending_warnings, column, X_train, y_train, feature_cols, stage_label,
            announce=writes_something,
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
                run,
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
                n_training_rows=len(y_train),
                unanchored=group_key[1] is None,
            )
            # Exécution, puis gel : un échec de prédiction dégrade l'étape en
            # repli, de sorte que le plan dise ce qui a réellement été fait.
            # La volumétrie écrite n'est connue qu'après coup : elle est gelée
            # sur l'étape au même titre que la dégradation
            degraded, n_written = self._execute_step(
                run, step, X_work=X_work, stage_freq=stage_freq
            )
            step = replace(step, n_written=n_written)
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
        pending_warnings: WarningAccumulator,
        column: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: Tuple[str, ...],
        stage_label: str,
        announce: bool = True,
    ) -> Tuple[Any, bool]:
        """Fit the estimator of one variable, or fall back on interpolation.

        Args:
            pending_warnings: Pending warnings of the pass, which receive the
                announced fallbacks.
            column: Column being imputed.
            X_train: Scaled training features.
            y_train: Scaled training target.
            feature_cols: Covariates retained.
            stage_label: Readable label of the stage, for the messages.
            announce: Whether a fallback is worth announcing. False when the
                fit has nothing to write with the step — an empty prediction
                grid, whose cause the window calculator already reports.

        Returns:
            Tuple ``(model, is_fallback)``: the fitted estimator and False, or
            the ``INTERPOLATE_FALLBACK`` sentinel and True. Every failure
            message is accumulated, never emitted here.
        """
        # Repli d'interpolation, annoncé par l'avertissement agrégé de la phase
        def _fallback(reason: str) -> Tuple[Any, bool]:
            if announce:
                pending_warnings.add_degraded_step(
                    f"{column!r} at stage {stage_label}: {reason}, "
                    f"interpolation fallback"
                )
            return INTERPOLATE_FALLBACK, True

        # Extraction de l'estimateur
        estimator = self._estimator_for(column)
        # Aucun estimateur pour cette colonne : le repli est la règle, et
        # l'avertissement global de la phase 0 a déjà été émis
        if estimator is None:
            if self.estimator is not None:
                return _fallback("no estimator covers this column")
            return INTERPOLATE_FALLBACK, True

        # Jeu d'entraînement inexploitable : aucune covariable, ou aucune ligne
        if not feature_cols or len(y_train) == 0:
            return _fallback(
                f"empty training set ({len(y_train)} row(s), "
                f"{len(feature_cols)} covariate(s))"
            )

        # Échec d'ajustement : repli, jamais une exception qui ferait perdre
        # les autres variables de l'étape
        try:
            estimator.fit(X_train, y_train)
        except Exception as error:
            return _fallback(f"estimator fit failed ({type(error).__name__}: {error})")

        return estimator, False

    # Méthode auxiliaire de construction d'une étape de plan
    def _build_step(
        self,
        run: _RunContext,
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
        n_training_rows: int,
        unanchored: bool = False,
    ) -> ImputationStep:
        """Freeze one plan step of a (stage, variable, source frequency) group.

        Args:
            run: Execution context of the fit.
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
            n_training_rows: Size of the mutualized ``y_train`` the model was
                fitted on, shared by every group of the variable.
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
            scale = self._group_scale_factor(
                run.stage_scaler, column, source_frequency, stage_freq, grid
            )

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
            interpolation_method=run.materializer.resolve_method(column),
            interpolation_anchor=run.materializer.resolve_anchor(column),
            training_blocks=training_blocks,
            unanchored=unanchored,
            n_training_rows=n_training_rows,
        )

    # Méthode auxiliaire du facteur d'échelle d'un groupe ancré
    def _group_scale_factor(
        self,
        stage_scaler: StageScaler,
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
            stage_scaler: Stage scaler of the pass.
            column: Column imputed by the step.
            source_frequency: Detected frequency of the column for the group.
            stage_freq: Frequency of the stage.
            grid: Prediction grid of the group, read by the ``'calendar'``
                scale mode.

        Returns:
            The factor. A conversion the scaler refuses raises: a silent
            ``1.0`` would mis-scale every prediction of the group.
        """
        return stage_scaler.fit_scale_factor(
            column,
            source_freq=source_frequency,
            pred_freq=stage_freq,
            index=grid,
        )

    # Méthode unique d'exécution d'une étape du plan
    def _execute_step(
        self,
        run: _RunContext,
        step: ImputationStep,
        *,
        X_work: pd.DataFrame,
        stage_freq: Union[str, Dict[EntityKey, str]],
    ) -> Tuple[bool, int]:
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
        NaN and unmarked and is reported by the aggregated warning of the
        end of the pass.

        Args:
            run: Execution context of the pass, fit or replay.
            step: Frozen step to execute.
            X_work: Working frame the values are read from.
            stage_freq: Frequency of the stage.

        Returns:
            Tuple ``(degraded, n_written)``. ``degraded`` is True when the
            step had to fall back on interpolation at execution time — a
            prediction failure — so the caller can degrade the plan step
            accordingly; False otherwise, an unanchored failure included:
            such a step is never degraded into a fallback, it is simply not
            executed. ``n_written`` counts the cells actually written, after
            rescaling, and is 0 whenever nothing was produced. The replay at
            ``transform`` discards both.
        """
        # Extraction du matérisaliseur des étapes précédentes
        materializer = run.materializer
        # Extraction du nom de la colonne
        column = step.var_name
        # Détection des fréquences de la colonne (éventuellement pour différentes entités)
        freqs_by_column = self._frequency_layout_.frequencies_by_column(run.transform_frequencies)

        # Grille du groupe : la fenêtre d'imputation de ses entités, ancres comprises
        grid = self._prediction_grid(
            run, X_work, stage_freq, step.entities
        )

        # Sans ancre, la fenêtre stricte d'une entité est vide par
        # construction : elle est l'intervalle où toutes ses colonnes sont
        # couvertes, et la colonne imputée n'y en couvre aucune. La subordonner
        # à la couverture de la colonne même que l'étape produit serait
        # circulaire ; l'entité est donc laissée sans restriction, exactement
        # comme "_stage_mask" traite déjà une entité que le calculateur omet
        if step.unanchored and len(grid) == 0:
            grid = self._unrestricted_grid(
                run, X_work, stage_freq, step.entities
            )

        if len(grid) == 0:
            return False, 0

        # Production des valeurs : modèle, ou repli d'interpolation
        values: Optional[pd.Series] = None
        if not step.is_fallback:
            values = self._predict_step(
                run, step, grid, X_work, freqs_by_column, stage_freq
            )
        degraded = values is None and not step.is_fallback

        # Absence d'ancre : l'interpolation ne peut pas être le repli, faute
        # d'observation à interpoler. Les couples restent NaN et non marqués,
        # et l'avertissement agrégé de la fin de la passe les nomme
        if step.unanchored and values is None:
            run.pending_warnings.add_unimputed_pairs(self._step_pairs(step))
            return False, 0

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
            values, _rescaled_mask = run.aggregation_constraint.rescale(
                values, observations, step.source_frequency, column=column,
                grid_freq=stage_freq,
            )

        # Ecriture : cellules effectivement produites. Une étape sans ancre
        # n'arrive jamais ici vide : "_predict_step" rend None sur
        # une prédiction entièrement NaN, et aucun recalage ne vide la sienne
        written = values[values.notna()]
        if written.empty:
            return degraded, 0

        # Couples effectivement imputés sans ancre, source de unanchored_pairs_
        if step.unanchored:
            run.unanchored_written.update(self._step_pairs(step))

        # Marquage de provenance : identique pour les cellules recalées et non
        # recalées, lignes d'ancres comprises — le recalage ne change aucune
        # provenance. Une ligne d'ancre ré-exprimée ne reste jamais original : elle ne porte plus
        # l'observation. La grille densifiée peut porter des dates absentes de
        # l'entrée (index irrégulier, fenêtre étendue vers des périodes où
        # seules des observations basse fréquence isolées existent) : la
        # matrice, initialisée sur l'index d'entrée, est donc étendue d'abord
        run.provenance_tracker.extend_index(written.index)
        run.provenance_tracker.mark_imputed(column, written.index, provenance)

        # Mise à jour des trois registres, y compris en repli
        materializer.record_production(
            column,
            written,
            pd.Series(origin, index=written.index, dtype=CELL_ORIGIN_DTYPE),
            stage_frequency_of(stage_freq, written.index),
        )
        return degraded, len(written)

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
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from tsforecast.imputation import HighFrequencyImputer
            >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
            >>> idx = pd.MultiIndex.from_product([['FR', 'IT'], dates], names=['entity', 'date'])
            >>> panel = pd.DataFrame({'m1': np.arange(48.0), 'v': np.nan}, index=idx)
            >>> panel.loc[('FR', dates[11::12]), 'v'] = [120.0, 132.0]
            >>> imputer = HighFrequencyImputer(
            ...     target_frequency='M',
            ...     estimator=LinearRegression(),
            ...     impute_unobserved_entities=True,
            ... ).fit(panel)
            >>> HighFrequencyImputer._step_pairs(imputer.imputation_plan_.steps[-1])
            [('IT', 'v')]
        """
        # Série temporelle : aucune entité, donc aucun couple à nommer
        return [
            (*entity, step.var_name) for entity in (step.entities or ())
        ]

    # Méthode auxiliaire de prédiction d'une étape à modèle
    def _predict_step(
        self,
        run: _RunContext,
        step: ImputationStep,
        grid: pd.Index,
        X_work: pd.DataFrame,
        freqs_by_column: Dict[str, Union[str, Dict[EntityKey, str]]],
        stage_freq: Union[str, Dict[EntityKey, str]],
    ) -> Optional[pd.Series]:
        """Produce the predictions of one model step on its grid.

        Args:
            run: Execution context of the pass, fit or replay.
            step: Frozen step, holding the model and the ways to replay.
            grid: Prediction grid of the group.
            X_work: Working frame.
            freqs_by_column: Detected frequencies, keyed by column.
            stage_freq: Frequency of the stage.

        Returns:
            The predictions at the scale of the stage, or None when the
            prediction failed or produced nothing but NaN — the caller then
            falls back on interpolation, or, for an unanchored step, reports
            the pair. A failure is announced by the aggregated warning of the
            degraded steps, an unanchored step excepted: having no fallback,
            it is named by the warning of the unimputed pairs only.
        """
        # Covariables produites en mode rejeu des voies de l'étape : c'est le
        # seul chemin autorisé pour X_pred.
        # "record=False" : les registres ne portent qie ce qui a été imputé,
        # jamais ce qui a été préparé comme feature. Y inscrire la
        # matérialisation d'une covariable écraserait l'imputation que son
        # propre modèle vient d'écrire, et rendrait le résultat dépendant de
        # l'ordre de traitement
        X_pred, _ways, _origins = run.materializer.materialize(
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
            divisors = run.stage_scaler.feature_divisors(
                columns=step.feature_cols,
                column_frequencies=freqs_by_column,
                ways=step.materialization,
                grid_freq=stage_freq,
                stage_freq=stage_freq,
                index=grid,
            )
            predictions = step.model.predict(
                run.stage_scaler.apply(X_pred, divisors)
            )
        except Exception as error:
            # Sans ancre, aucun repli n'existe : seul l'avertissement des
            # couples non imputés le dira, jamais un « repli » qui n'a pas lieu
            if not step.unanchored:
                run.pending_warnings.add_degraded_step(
                    f"{step.var_name!r} at stage {step.pred_freq_label}: prediction "
                    f"failed ({type(error).__name__}: {error}), interpolation fallback"
                )
            return None

        # Prédictions à l'échelle de l'étape : le modèle est toujours ajusté
        # pour l'étape courante (aucune réutilisation inter-étapes), et la
        # cible l'a été ligne à ligne — aucune remise à l'échelle à appliquer
        values = pd.Series(
            np.asarray(predictions, dtype=float).ravel(), index=grid, name=step.var_name
        )

        # Prédictions entièrement NaN : rien n'est produit, l'étape est traitée
        # comme un échec de prédiction. Des NaN épars restent des cellules non
        # écrites, signalées par le même avertissement agrégé
        n_missing = int(values.isna().sum())
        if n_missing == len(values):
            if not step.unanchored:
                run.pending_warnings.add_degraded_step(
                    f"{step.var_name!r} at stage {step.pred_freq_label}: the model "
                    f"predicted only NaN, interpolation fallback"
                )
            return None
        if n_missing and not step.unanchored:
            run.pending_warnings.add_degraded_step(
                f"{step.var_name!r} at stage {step.pred_freq_label}: {n_missing} of "
                f"{len(values)} prediction(s) are NaN and stay unimputed"
            )
        return values

    # Méthode auxiliaire du masque d'entraînement lu à la fréquence des blocs
    def _training_mask_at(
        self,
        window_calculator: ImputationWindowCalculator,
        frequencies: Dict[EntityKey, str],
    ) -> Optional[pd.Series]:
        """Read the ``'training'`` window mask at one frequency per block.

        Injected into :class:`TrainingSetBuilder`, bound to the calculator of
        the fit: the builder therefore receives a callable and never the
        calculator, so it cannot read a window other than the training one,
        whose ``kind`` is named here once and for all.

        Args:
            window_calculator: Window calculator of the fit.
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
            return window_calculator.get_mask_at_frequency(binding, kind='training')
        except (ValueError, KeyError, TypeError) as error:
            # Logging
            self._log(f"[fit] training mask unavailable ({error}); every row kept")
            return None

    # -------------------------------------------------------------------------
    # Transform — rejeu du plan figé
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de contrôle des fréquences au transform
    def _check_transform_frequencies(
        self,
        X_work: pd.DataFrame,
        optional_columns: Iterable[str] = (),
    ) -> Tuple[Dict[Union[str, tuple], str], Tuple[Union[str, tuple], ...]]:
        """Compare the frequencies of the transform with those of the fit.

        The comparison is made **pair by pair** — ``(entity..., column)`` — and
        never column by column: on a panel the same column may legitimately
        carry a different frequency for each entity, and a
        per-column comparison would report a false divergence at every
        transform.

        Args:
            X_work: Working frame of the transform, before the additive
                transformer.
            optional_columns: Columns of the fit allowed to be absent — the
                target column in the prediction pass, which never sees ``y``.

        Returns:
            Tuple ``(frequencies, misaligned)``. ``frequencies`` maps the
            pairs the fit never observed — a column of a new entity, or one a
            known entity left empty at fit time — to the frequency detected on
            the data of the transform: theirs cannot be replayed, so this is
            the only one available. ``misaligned`` names those of them observed
            off the calendar bounds of their own frequency, left out exactly
            as the fit leaves its own out.

        Raises:
            ValueError: If a column known at fit time is absent from the data
                being transformed. The message names the missing columns.
        """
        # Plan des fréquences du fit
        layout = self._frequency_layout_
        # Couples connus du fit, mal placés compris : ils ne sont pas
        # "nouveaux" au transform, seulement écartés
        misaligned = layout.misaligned
        # Colonnes du fit, couples indétectables compris : une colonne
        # entièrement NaN au fit reste une colonne attendue au transform
        fit_columns = {
            split_variable_key(key)[1]
            for key in (
                *layout.detected_frequencies, *layout.undetected, *misaligned
            )
        }
        # Colonnes disparues : erreur nommant les colonnes, jamais un silence
        missing = sorted(
            column for column in fit_columns
            if column not in X_work.columns and column not in set(optional_columns)
        )
        if missing:
            raise ValueError(
                f"transform is missing {len(missing)} column(s) seen at fit "
                f"time: {missing}. Every column of the fit must be present, "
                f"even entirely empty: it is part of the frozen plan. Extra "
                f"columns, on the other hand, are ignored."
            )

        # Redétection sur les données du transform
        detected = detect_frequency(data=X_work)

        # Divergences, couple par couple, sur les seules clés communes
        diverging: List[Tuple[Any, str, str]] = []
        for key, freq in detected.items():
            fit_freq = layout.detected_frequencies.get(key)
            if freq is None or fit_freq is None:
                continue
            if (
                normalize_frequency(freq, return_format='base')
                != normalize_frequency(fit_freq, return_format='base')
            ):
                diverging.append((key, fit_freq, freq))

        # Avertissement uniquement, puis poursuite avec les fréquences du fit :
        # "FrequencyLayout.frequencies_by_column" ne recouvre jamais un couple du fit
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
        unknown = {
            key: freq
            for key, freq in detected.items()
            if freq is not None
            and key not in layout.detected_frequencies
            and key not in misaligned
            and split_variable_key(key)[1] in fit_columns
        }
        # Contrôle de calendrier du fit, appliqué à ces couples : observés hors
        # des bornes de leur période, ils n'ont pas de lecture cohérente
        if not unknown:
            return {}, ()
        off_calendar = misaligned_anchor_pairs(X_work, unknown, is_panel=self.is_panel_)
        return (
            {key: freq for key, freq in unknown.items() if key not in off_calendar},
            off_calendar,
        )

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
        they belong to. An entity — or a time series — with no row at all in
        the window is left out of that count: the calculator has already named
        it (no period covered by every column, or no determinable window), and
        a second message would only restate the same cause.

        Args:
            window_calc: Window calculator fitted on the transformed data.
            X_work: Working frame of the transform.
        """
        # Masque de prédiction, "kind" nommé explicitement
        try:
            mask = window_calc.get_imputation_window_mask(X_work, kind='imputation')
        except (ValueError, KeyError, TypeError):
            return

        # Lignes dans la fenêtre, et entités qui en ont au moins une : les
        # autres sont déjà nommées par l'avertissement du calculateur
        inside = pd.Series(mask.to_numpy(dtype=bool), index=mask.index)
        if isinstance(mask.index, pd.MultiIndex):
            levels = list(range(mask.index.nlevels - 1))
            has_window = inside.groupby(
                level=levels[0] if len(levels) == 1 else levels, sort=False
            ).transform('any')
        else:
            has_window = pd.Series(bool(inside.any()), index=mask.index)

        # Lignes du périmètre laissées hors d'une fenêtre qui existe
        outside = mask.index[(~inside & has_window).to_numpy(dtype=bool)]
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
        pending_warnings: WarningAccumulator,
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
            pending_warnings: Pending warnings of the replay, which name the
                entities left out.
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
            pending_warnings.add_outside_plan_case(
                f"stage {stage_label}: its entities disagree on the stage "
                f"frequency, so the entities born at transform time "
                f"{tuple(new_entities)} cannot be bound to it and are skipped"
            )
            return stage_freq

        shared = unique.pop()
        return {**stage_freq, **{entity: shared for entity in new_entities}}

    # Méthode auxiliaire des étapes rejouées à une étape de fréquence
    def _replay_steps(
        self,
        run: _RunContext,
        steps: Sequence[ImputationStep],
        new_entities: Sequence[EntityKey],
        binding: Union[str, Dict[EntityKey, str]],
        X_stage: pd.DataFrame,
    ) -> List[ImputationStep]:
        """Compose the steps one stage replays, for the pairs the fit never observed.

        Under ``impute_unobserved_entities=True``, a pair the fit never
        observed — a column of an entity absent from the fit, or a column a
        known entity left empty at fit time — is served as if it had been
        present at fit time with the data of the transform:

        - **observed** by the transform (a frequency is detected on its data):
          an **anchored** step, rescaled on its own observations, emitting
          the ``MODEL_*`` of the column's step — never an unanchored one,
          which would overwrite those observations and lose their
          totals. It is imputed at a stage only if its frequency is
          coarser than the stage's, as the classification of the fit does;
        - **never observed** by the transform: an **unanchored** step.

        Every derived step comes from a plan step of the same (stage,
        variable) — same model object (``is``), same ``feature_cols``, same
        materialization ways — so nothing is decided here: only the entities,
        the source frequency, the scale and the anchoring change. It is placed
        right after the plan steps of its column, which keeps the column order
        of the plan for its entity. Such a step never enters
        ``imputation_plan_``: the plan is the state of the fit. Under the
        default ``False`` the plan steps are replayed as they are, and those
        pairs stay as given.

        Args:
            run: Execution context of the replay.
            steps: Plan steps of the stage, in plan order.
            new_entities: Entities the fit never saw.
            binding: Stage binding, already extended (or not) to them.
            X_stage: Working frame, the grid of the anchored groups' scale.

        Returns:
            The steps to execute, in execution order.
        """
        # Paramètre éteint : le plan seul, les couples inconnus du fit intacts
        if not self.impute_unobserved_entities:
            return list(steps)

        # Entités nouvelles effectivement liées à l'étape
        bound = [entity for entity in new_entities if entity in binding]

        # Étapes du plan regroupées par colonne, dans l'ordre du plan
        by_column: "OrderedDict[str, List[ImputationStep]]" = OrderedDict()
        for step in steps:
            by_column.setdefault(step.var_name, []).append(step)

        # Initialisation de la liste des étapes
        composed: List[ImputationStep] = []
        for column, column_steps in by_column.items():
            # Couples observés par le transform, groupés par fréquence source
            anchored: Dict[str, List[EntityKey]] = {}
            # Entités nouvelles qui n'observent jamais la colonne
            unanchored: List[EntityKey] = []

            # Étapes sans ancre du fit : une entité que le transform observe
            # les quitte pour une étape ancrée
            for step in column_steps:
                if not step.unanchored:
                    composed.append(step)
                    continue
                remaining = []
                for entity in step.entities:
                    if not self._route_transform_pair(
                        run, entity, column, binding, anchored
                    ):
                        remaining.append(entity)
                if remaining:
                    composed.append(replace(step, entities=to_entity_tuple(remaining)))

            # Entités nouvelles : ancrées si observées, sans ancre sinon
            for entity in bound:
                if not self._route_transform_pair(
                    run, entity, column, binding, anchored
                ):
                    unanchored.append(entity)

            # Étapes dérivées, juste après celles de la colonne
            representative = column_steps[0]
            for frequency, entities in anchored.items():
                grid = self._prediction_grid(
                    run, X_stage, binding, entities
                )
                scale = self._group_scale_factor(
                    run.stage_scaler, column, frequency, binding, grid
                )
                composed.append(
                    replace(
                        representative,
                        source_frequency=frequency,
                        entities=to_entity_tuple(sorted(entities, key=repr)),
                        scale_factor=scale,
                        fit_scale_factor=scale,
                        unanchored=False,
                    )
                )
            if unanchored:
                composed.append(
                    replace(
                        representative,
                        source_frequency=None,
                        entities=to_entity_tuple(unanchored),
                        scale_factor=1.0,
                        fit_scale_factor=1.0,
                        unanchored=True,
                    )
                )
        return composed

    # Méthode auxiliaire d'aiguillage d'un couple inconnu du fit
    def _route_transform_pair(
        self,
        run: _RunContext,
        entity: EntityKey,
        column: str,
        binding: Dict[EntityKey, str],
        anchored: Dict[str, List[EntityKey]],
    ) -> bool:
        """Route one pair the fit never observed, by what the transform observes of it.

        Args:
            run: Execution context of the replay, holding what the transform
                observes of the pairs the fit never observed.
            entity: Entity of the pair.
            column: Column of the pair.
            binding: Stage binding, naming ``entity``.
            anchored: Anchored groups being composed, keyed by source
                frequency; extended in place.

        Returns:
            True when the transform observes the pair — it then joins an
            anchored group if its frequency is coarser than the stage, and is
            left as given otherwise (already at, or finer than, the stage
            frequency; or observed off the calendar bounds of its frequency).
            False when it never observes it: the pair is unanchored.
        """
        # Clé entité X colonne
        key = (*entity, column)
        # Couple observé hors des bornes calendaires de sa fréquence : écarté,
        # comme au fit, plutôt qu'imputé sur une lecture fausse
        if key in run.transform_misaligned:
            return True
        frequency = run.transform_frequencies.get(key)
        if frequency is None:
            return False
        # Classification du fit, couple par couple : imputable si sa fréquence
        # est plus grossière que celle de l'étape pour cette entité
        stage = binding[entity]
        if not is_higher_frequency(frequency, stage) and (
            normalize_frequency(frequency) != normalize_frequency(stage)
        ):
            source = normalize_frequency(frequency, return_format='base')
            anchored.setdefault(source, []).append(entity)
        return True

    # Méthode de rejeu du plan, étape de fréquence par étape de fréquence
    def _replay_plan(
        self,
        run: _RunContext,
        X_stage: pd.DataFrame,
        skip_columns: FrozenSet[str] = frozenset(),
    ) -> Tuple["OrderedDict[str, pd.DataFrame]", "OrderedDict[str, pd.DataFrame]"]:
        """Replay every frozen step of the plan, in the order of the fit.

        Nothing is decided here: the classification, the progression, the
        variable order, the models, the ``feature_cols``, the materialization
        ways and the taints all come from the plan. Only the windows, the stage
        frames, the interpolated values, the predictions, the rescaling and the
        provenance are recomputed.

        Args:
            run: Execution context of the replay.
            X_stage: Working frame after the additive transformer. Never
                modified: the imputations travel through the registers of
                :class:`CovariateMaterializer`, exactly as at fit time.
            skip_columns: Columns whose steps are not replayed — the target
                column in the prediction pass, absent from ``X_stage``. No
                other step reads them: ``target_as_covariate=False`` kept the
                target out of every ``feature_cols``, and the prediction pass
                refuses a plan where it is not.

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
            run.pending_warnings.add_outside_plan_case(
                f"{len(new_entities)} entity/entities absent from the fit "
                f"{new_entities} are left untouched "
                f"(impute_unobserved_entities=False)"
            )
        # Couples inconnus du fit observés hors des bornes de leur période :
        # écartés comme au fit, et nommés
        if run.transform_misaligned and self.impute_unobserved_entities:
            run.pending_warnings.add_outside_plan_case(
                f"{len(run.transform_misaligned)} (entity, column) pair(s) "
                f"unknown to the fit are observed off the calendar bounds of "
                f"their frequency and are left untouched: "
                f"{sorted(run.transform_misaligned, key=repr)}"
            )

        # Report de fréquence : dernière étape traversée par chaque entité
        carried: Dict[EntityKey, str] = {}
        # Parcours des étapes de la progression, rejouée telle quelle
        for stage_freq in self.frequency_progression_:
            stage_label = stage_frequency_label(stage_freq)
            # Liaison de fréquence, étendue aux entités nouvelles
            binding = self._extend_stage_binding(
                run.pending_warnings, stage_freq, new_entities, stage_label
            )

            # Étapes du plan de l'étape, complétées pour les couples que le fit
            # n'a jamais observés
            # (étapes des colonnes écartées retirées avant toute dérivation)
            stage_steps = tuple(
                step for step in by_stage.get(stage_label, ())
                if step.var_name not in skip_columns
            )
            steps = self._replay_steps(run, stage_steps, new_entities, binding, X_stage)
            # Logging
            self._log(
                f"[transform] stage {stage_label}: replaying {len(steps)} step(s)"
            )
            # Exécution par la même méthode qu'au fit
            for step in steps:
                self._execute_step(run, step, X_work=X_stage, stage_freq=binding)

            # Liaison complète, pour la sortie : une entité que l'étape ne
            # concerne pas garde les lignes de son dernier passage, et une
            # liaison partielle ferait échouer la lecture du masque, qui
            # retomberait alors sans bruit sur tout l'index
            output_binding = self._output_binding(binding, carried, X_stage)

            # Grille de la frame : l'index du masque, jamais ses seules lignes
            # vraies — les lignes hors fenêtre restent dans la sortie avec
            # leurs valeurs d'entrée. Une ligne d'entrée hors de la grille (une
            # cible plus grossière que l'index) n'y figure pas : elle n'y
            # porterait rien, sa valeur étant agrégée sur la ligne de sa
            # période, et l'inversion rend l'index d'entrée. L'ordre est
            # celui de l'entrée (on impose qu'il soit trié)
            grid = self._in_input_order(
                self._stage_mask(
                    run, X_stage, output_binding, kind='imputation'
                ).index,
                X_stage.index,
            )

            # Frame d'étape reconstruite après les écritures, puis recouverte
            # par les imputations : "stage_frame" ne lit le miroir que sous
            # "covariate_strategy='model'", et la sortie doit le porter sous
            # toutes les stratégies
            frame = run.materializer.stage_frame(
                grid_index=grid,
                stage_freq=output_binding,
                detected_frequencies=self._frequency_layout_.frequencies_by_column(
                    run.transform_frequencies
                ),
                source_data=X_stage,
            )
            # Noms de l'index d'entrée : la grille vient d'un masque converti,
            # qui ne les porte pas, et la sortie doit rester lisible comme
            # l'entrée — empilée ou non
            if grid.nlevels == X_stage.index.nlevels:
                frame = frame.rename_axis(X_stage.index.names)
            frames[stage_label] = self._overlay_imputations(
                run.materializer, frame, output_binding
            )

            # Instantané de provenance, pris au même instant et réindexé sur la
            # grille de l'étape : la matrice partage ainsi l'index de la sortie,
            # que "_inverse_transform" lit ensuite masque contre masque
            snapshot = run.provenance_tracker.get_provenance_matrix()
            provenances[stage_label] = self._stage_provenance(
                run.transform_frequencies,
                snapshot.reindex(index=grid, columns=frame.columns),
                frames[stage_label],
                output_binding,
            ).rename_axis(frame.index.names)

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
                or self._frequency_layout_.entity_target_frequency(entity)
                or next(iter(binding.values()), None)
            )
            if frequency is not None:
                completed[entity] = frequency
        return completed

    # Méthode auxiliaire de mise dans l'ordre de l'entrée d'une grille d'étape
    @classmethod
    def _in_input_order(cls, grid: pd.Index, reference: pd.Index) -> pd.Index:
        """Order a stage grid like the input: entity blocks in input order, dates sorted.

        The window masks are concatenated entity by entity — and an entity
        the calculator omits is appended last —, so their index carries no
        guaranteed order. The output must keep the order of its input: 
        an unsorted input is refused, and a component cannot produce what it
        refuses.

        Args:
            grid: Grid of a stage frame.
            reference: Index of the working frame, whose entity order wins.

        Returns:
            The grid, reordered.
        """
        # Série temporelle : l'ordre des dates suffit
        if not isinstance(grid, pd.MultiIndex):
            return grid.sort_values()

        # Panel : rang de chaque entité dans l'entrée, puis date
        rank = {
            entity: position
            for position, entity in enumerate(cls._index_entities(reference))
        }
        levels = [grid.get_level_values(level) for level in range(grid.nlevels - 1)]
        entity_rank = np.asarray([
            rank.get(normalize_entity_key(tuple(values)), len(rank))
            for values in zip(*levels)
        ])
        order = np.lexsort((grid.get_level_values(-1).to_numpy(), entity_rank))
        return grid[order]

    # Méthode auxiliaire de la provenance d'une frame d'étape
    def _stage_provenance(
        self,
        transform_frequencies: Mapping[Union[str, tuple], str],
        provenance: pd.DataFrame,
        frame: pd.DataFrame,
        output_binding: Union[str, Dict[EntityKey, str]],
    ) -> pd.DataFrame:
        """Make the tracker's marks describe the cells of one stage frame.

        The tracker marks the cells of the input grid, and a stage frame does
        not always carry the input value of a cell: a column observed more
        finely than the row is the exact **aggregation** of its period
        there, marked ``AGGREGATED`` — ``ORIGINAL`` would let the inversion
        hand a quarterly sum back as the monthly observation of the same
        date. A cell with a value no mark covers — an aggregation on a date
        absent from the input — is ``AGGREGATED`` too when the column is
        finer, ``ORIGINAL`` otherwise. And a cell left empty is never marked,
        whatever the tracker says of the input cell of the same date.

        Args:
            transform_frequencies: Frequencies the replay detected for the
                pairs the fit never observed.
            provenance: Snapshot of the tracker, reindexed on the frame.
            frame: Stage frame, imputations laid over.
            output_binding: Complete frequency binding of the frame's grid.

        Returns:
            The provenance of the frame, cell for cell.
        """
        has_value = frame.notna()
        # Colonnes que le suivi ne porte pas : ajoutées par la réindexation en
        # float, ramenées au dtype catégoriel avant tout marquage
        provenance = provenance.astype(PROVENANCE_DTYPE)

        # Cellules sans marque mais renseignées : valeurs reprises telles quelles
        provenance = provenance.mask(provenance.isna() & has_value, ProvenanceType.ORIGINAL)
        # Agrégations exactes d'une colonne plus fine que la ligne
        aggregated = (
            self._aggregated_cells(transform_frequencies, frame, output_binding)
            & has_value
        )
        provenance = provenance.mask(aggregated, ProvenanceType.AGGREGATED)
        # Cellule vide : jamais marquée, au rejeu comme au fit
        return provenance.where(has_value)

    # Méthode auxiliaire des cellules agrégées d'une frame d'étape
    def _aggregated_cells(
        self,
        transform_frequencies: Mapping[Union[str, tuple], str],
        frame: pd.DataFrame,
        output_binding: Union[str, Dict[EntityKey, str]],
    ) -> pd.DataFrame:
        """Locate the cells of a stage frame whose column is finer than the row.

        Such a column is never imputed at that stage: the stage frame
        carries its exact additive aggregation over the period of the row.

        Args:
            transform_frequencies: Frequencies the replay detected for the
                pairs the fit never observed.
            frame: Stage frame.
            output_binding: Complete frequency binding of the frame's grid.

        Returns:
            Boolean frame shaped like ``frame``.
        """
        # Fréquence d'étape et entité de chaque ligne, codées en couples
        # distincts sans construire de tuple par ligne
        row_frequency = stage_frequency_of(output_binding, frame.index)
        entity_codes, entities = get_entity_codes(frame.index)
        codes, first_rows = factorize_row_keys(entity_codes, row_frequency)
        pairs = [
            (entities[entity_codes[row]], row_frequency.iat[row]) for row in first_rows
        ]

        # Fréquence de chaque colonne, par entité sur un panel
        column_frequencies = self._frequency_layout_.frequencies_by_column(transform_frequencies)
        result = pd.DataFrame(False, index=frame.index, columns=frame.columns)
        # Détection des colonnes à plus haute fréquence que celle de l'étape
        for column in frame.columns:
            frequencies = column_frequencies.get(column)
            if frequencies is None:
                continue
            finer = []
            for entity, stage in pairs:
                own = (
                    frequencies.get(entity) if isinstance(frequencies, dict)
                    else frequencies
                )
                finer.append(own is not None and is_higher_frequency(own, stage))
            result[column] = np.asarray(finer, dtype=bool)[codes]
        return result

    # Méthode auxiliaire de recouvrement d'une frame par les imputations
    def _overlay_imputations(
        self,
        materializer: CovariateMaterializer,
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
            materializer: Materializer of the replay, holder of the mirror.
            frame: Stage frame, as built by the materializer.
            output_binding: Complete frequency binding of the frame's grid.

        Returns:
            The frame, imputations included.
        """
        # Aucune production : la frame est déjà la sortie de l'étape
        if not materializer.imputed_store:
            return frame

        # Fréquence d'étape de chaque ligne de la grille
        row_frequency = stage_frequency_of(output_binding, frame.index)

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
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Fit the plan, then run the training pass on the same data.

        Always the full replay of :meth:`_replay_pass`, ``y`` included: fitted
        with ``y``, this is the **training pass** — ``X`` and ``y`` on the target
        grid, the rows the grid adds included, none dropped. It therefore
        differs from ``fit(X, y).transform(X, y)``, which runs the prediction
        pass; without ``y`` the two are equal.

        Args:
            X: Features.
            y: Target (optional).

        Returns:
            The transformed features, or the pair ``(X, y)`` when ``y`` is
            given.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from tsforecast.imputation import HighFrequencyImputer
            >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
            >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
            >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0) * 3 + 10
            >>> y = pd.Series(np.nan, index=dates, name='y')
            >>> y.loc[dates[2::3]] = np.arange(8.0)
            >>> imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())
            >>> X_train, y_train = imputer.fit_transform(data, y)
            >>> int(y.notna().sum()), int(y_train.notna().sum())
            (8, 24)
        """
        # Entraînement du transformer
        self.fit(X, y)
        # Validation de X, comme le "transform" public
        if self.validate_input:
            X = self._validate_input(X)
        return self._replay_pass(X, y, prediction=False)

    # Aiguillage du transform public entre rejeu et passe de prédiction
    def _transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Run the prediction pass if the imputer was fitted with ``y``, else replay.

        Fitted with ``y``, the imputer augments a training set and ``transform``
        is the **prediction pass** of a pipeline (``predict`` / ``score``): only
        the columns of ``X`` are imputed, on the input rows, and ``y`` — when
        given — comes back unchanged. Fitted without ``y``, ``transform``
        replays the plan on any data.

        Args:
            X: Features to transform.
            y: Target, optional. Never imputed nor read by the prediction pass.

        Returns:
            The transformed features, or the pair ``(X, y)`` when ``y`` is
            given.
        """
        return self._replay_pass(X, y, prediction=self.target_column_ is not None)

    # Garde de la passe de prédiction : aucun modèle de X ne lit la cible
    def _check_prediction_pass_replayable(self) -> None:
        """Refuse a prediction pass whose models would need the target.

        Under ``target_as_covariate=True`` a model of a column of ``X`` may
        read ``y``, which the prediction pass never has: replaying it would
        feed the model a covariate it was never fitted without.

        Raises:
            ValueError: If a non-fallback step of a column other than the
                target lists the target among its ``feature_cols``.
        """
        # Identification des cas où la variable dépendante y est utilisée en covariable de l'imputation d'autres variables
        dependent = sorted({
            step.var_name for step in self.imputation_plan_
            if not step.is_fallback
            and step.var_name != self.target_column_
            and self.target_column_ in step.feature_cols
        })
        # Message d'erreur
        if dependent:
            raise ValueError(
                f"The prediction pass cannot impute {dependent}: their models "
                f"use the target {self.target_column_!r} as a covariate "
                f"(target_as_covariate=True), and transform never reads y once "
                f"the imputer is fitted with it. Refit with "
                f"target_as_covariate=False, or impute the training set with "
                f"fit_transform."
            )

    # Rejeu du plan : passe d'entraînement, rejeu autonome ou passe de prédiction
    def _replay_pass(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        prediction: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Replay the fitted plan on new data.

        Phases 0'-4' recompute what depends on the data by the very code of
        the fit (:func:`~tsforecast.imputation._engine.preparation.prepare_work_frame`,
        ``fitting=False``) — ``y`` alignment and naming, the frequency check
        against the fit, the windows recomputed before the additive
        transformer, the fitted transformer applied, the provenance tracker
        initialized after it — then every frozen step is replayed by
        :meth:`_execute_step`,
        the very method the fit calls, on an execution context
        (:class:`_RunContext`) built from the fitted state alone. No second
        execution loop, and no
        training set is ever rebuilt: the ``TrainingSetBuilder`` is not on this
        path.

        Note:
            This method is stateful: it snapshots its input in ``_original_X_``
            / ``_original_y_`` and overwrites ``imputation_provenance_`` and
            ``provenance_statistics_`` at each call. :meth:`_inverse_transform`
            reads them back and therefore always inverts the last pass. They
            are published together at the very end: a call that raises is not
            a pass and leaves the previous one invertible. ``fit`` purges
            them, then writes its own matrix, statistics and snapshot.

        In the **prediction pass** (``prediction=True``, imputer fitted with
        ``y``) the target column is absent from the working frame and its steps
        are not replayed; the output is the target level reindexed on the
        input rows, with the columns of ``X``, and ``y`` is returned as given.

        Args:
            X: Features to transform.
            y: Target to transform (optional); returned unchanged by the
                prediction pass.
            prediction: Whether to run the prediction pass.

        Returns:
            The transformed features, or the pair ``(X, y)`` when ``y`` is
            given. Under ``keep_lower_frequencies=True`` the index carries a
            ``'frequency'`` level — except in the prediction pass, always flat.

        Raises:
            ValueError: If ``X`` is not a DataFrame, if it carries no usable
                temporal index, if a column of the fit is missing, if an
                index level bears the reserved name ``'frequency'``, or if the
                prediction pass meets a model that needs the target.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from tsforecast.imputation import HighFrequencyImputer
            >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
            >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
            >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0) * 3 + 10
            >>> imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())
            >>> imputed = imputer.fit(data).transform(data)
            >>> imputed.loc['M', 'q1'].iloc[:3].round(1).tolist()
            [3.0, 3.3, 3.7]
        """
        # PHASES 0'-4' — Préparation commune au fit : contrôle des fréquences, fenêtres recalculées
        prepared = prepare_work_frame(self, X, y, fitting=False, prediction=prediction)
        # Cible alignée et son nom dans la frame de travail, relus par la sortie
        y, y_col_name = prepared.y, prepared.target_column

        # PHASE 5' — Rejeu du plan sur un contexte d'exécution neuf, construit
        # depuis l'état ajusté seul : composants partagés du fit, matérialiseur
        # et registres vides, fenêtre et traceur de ce transform. Le rejeu ne
        # lit ni n'altère aucun état d'exécution du fit
        run = _RunContext.for_transform(
            self._fitted_components_,
            window_calculator=prepared.window_calculator,
            provenance_tracker=prepared.provenance_tracker,
            transform_frequencies=prepared.new_frequencies,
            transform_misaligned=prepared.new_misaligned,
        )
        # Recalages du rejeu sous un bloc différé, comme au fit : un
        # avertissement par cas dégénéré pour tout le rejeu
        with run.aggregation_constraint.deferred_warnings():
            frames, provenances = self._replay_plan(
                run, prepared.X_work, skip_columns=prepared.skipped_columns
            )
        # Avertissements agrégés, un message par famille
        run.pending_warnings.emit()

        # PHASE 6' — Sortie multi-fréquences et provenance. La progression porte
        # toujours l'étape cible : la dernière frame est le niveau cible, quelle
        # que soit la forme de la sortie
        final_label = list(frames)[-1]
        # Profondeur de l'index empilé, notée au moment de l'empilement : c'est
        # elle, et non le nom d'un niveau, qui identifie à l'inversion une
        # sortie multi-fréquences
        stacked_nlevels: Optional[int] = None
        if prediction:
            # Passe de prédiction : niveau cible, sur les lignes d'entrée et les
            # colonnes de X — aucune ligne ajoutée, ni niveau de fréquence
            data_result = frames[final_label].reindex(index=X.index, columns=X.columns)
            # Colonne absente de l'étape : réindexée en float, ramenée au dtype catégoriel
            provenance = provenances[final_label].reindex(
                index=X.index, columns=X.columns
            ).astype(PROVENANCE_DTYPE)
        elif self.keep_lower_frequencies:
            data_result = self._build_multifreq_output(frames)
            provenance = self._build_multifreq_output(provenances)
            stacked_nlevels = data_result.index.nlevels
        else:
            data_result = frames[final_label]
            provenance = provenances[final_label]

        # Publication de l'état de la passe, d'un seul tenant et seulement
        # maintenant que rien ne peut plus lever. Les statistiques décrivent le
        # niveau cible : "keep_lower_frequencies" n'est qu'un affichage
        self._original_X_ = prepared.snapshot_X
        self._original_y_ = prepared.snapshot_y
        self._stacked_index_nlevels: Optional[int] = stacked_nlevels
        self.imputation_provenance_ = provenance
        self.provenance_statistics_ = provenance_statistics(
            provenance if prediction else provenances[final_label]
        )

        # Passe de prédiction : y rendu tel quel, jamais imputé
        if prediction:
            return data_result if y is None else (data_result, y)

        # Scission X / y, la cible gardant le nom qu'elle portait en entrée
        if y is not None and y_col_name in data_result.columns:
            return (
                data_result.drop(columns=[y_col_name]),
                data_result[y_col_name].rename(y.name),
            )
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
        all_frames = list(stage_frames.values())
        combined = pd.concat(all_frames, ignore_index=False)
        # Niveau de fréquence en dtype chaîne, quel que soit le nombre d'étapes :
        # l'étiquette de chaque étape répétée sur ses lignes, par position
        labels = pd.array(list(stage_frames), dtype='str')
        freq_values = pd.Index(
            labels.take(np.repeat(np.arange(len(labels)), [len(frame) for frame in all_frames]))
        )

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
                    FREQUENCY_LEVEL_NAME,
                    combined.index.names[-1] or 'date',
                ],
            )
        else:
            new_index = pd.MultiIndex.from_arrays(
                [freq_values, combined.index],
                names=[FREQUENCY_LEVEL_NAME, combined.index.name or 'date'],
            )

        return combined.set_axis(new_index)

    # -------------------------------------------------------------------------
    # Transformation inverse
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de présence du niveau de fréquence
    def _has_frequency_level(self, frame: pd.DataFrame) -> bool:
        """Tell whether a frame is a stacked multi-frequency frame.

        Read on the STRUCTURE, never on the level names: a stacked frame is
        a MultiIndex as deep as the output the last transform stacked, its
        frequency level sitting just before the date. Renaming the levels of
        the output, or an index level named like the frequency level, cannot
        fool the test.

        Args:
            frame: Frame to inspect.

        Returns:
            True if the last transform stacked its output and ``frame`` has
            that output's depth.
        """
        depth = getattr(self, '_stacked_index_nlevels', None)
        return (
            depth is not None
            and isinstance(frame.index, pd.MultiIndex)
            and frame.index.nlevels == depth
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

        The levels are read on the data to invert whenever they are stacked:
        a level only the provenance carries would select nothing in them. The
        provenance is read only for flat data.

        Args:
            data: Frame to invert, stacked or not.
            provenance: Provenance matrix of the last transform, stacked or not
                — both follow ``keep_lower_frequencies``.

        Returns:
            The frequency label to keep, or None when neither frame carries a
            frequency level.
        """
        # Labels disponibles : ceux des données, à défaut ceux de la provenance
        frame = data if self._has_frequency_level(data) else provenance
        if not self._has_frequency_level(frame):
            return None
        # Niveau de fréquence lu par position : l'avant-dernier
        available: List[str] = frame.index.get_level_values(-2).unique().tolist()

        # Priorité au niveau de l'index source
        source_label = getattr(self, '_source_index_frequency_label', None)
        if source_label is not None and source_label in available:
            return source_label

        # Repli sur le niveau cible, toujours produit
        target_label = stage_frequency_label(self.effective_target_frequency_)
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
        position — the one before the date —, never by name, and the names of
        the remaining levels are restored from the last transform input.

        Args:
            frame: Frame to reduce, stacked or not.
            label: Frequency label to keep. None leaves the frame untouched.

        Returns:
            The frame restricted to ``label``, without the frequency level.

        Raises:
            ValueError: If ``frame`` is stacked but carries no ``label`` level
                — the provenance of the last transform lacking a level the
                data to invert hold: those data are not an output of that
                transform.
        """
        # Frame déjà à un seul niveau de fréquence
        if label is None or not self._has_frequency_level(frame):
            return frame

        # Niveau de fréquence : l'avant-dernier, par position. Un niveau
        # absent ne laisse rien à inverser : une frame empilée et une frame
        # réduite ne s'alignent sur aucune cellule
        level = frame.index.nlevels - 2
        if label not in frame.index.get_level_values(level):
            raise ValueError(
                f"The frequency level {label!r} of the data to invert is not "
                f"one the last transform produced: inverse_transform undoes "
                f"the last pass, so it inverts an output of that pass only."
            )
        reduced = frame.xs(label, level=level)

        # Noms de niveaux de l'index source : l'instantané d'entrée de la passe
        # a, par construction, la profondeur de la frame réduite
        return reduced.rename_axis(self._original_X_.index.names)

    # Méthode auxiliaire de restauration des valeurs d'origine exactes
    def _restore_original_values(
        self,
        data_result: pd.DataFrame,
        y_col_name: Optional[str],
    ) -> pd.DataFrame:
        """Refill the cells observed in the input of the last pass.

        The last pass is the last ``transform``, or the ``fit`` when no
        ``transform`` has followed it: both snapshot their input.

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
        # Reconstruction de l'instantané de l'entrée de la dernière passe
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
        data must undo what that very call produced. Five moves: keep the
        frequency level of the source index and restore the index names,
        set back to NaN every cell whose provenance is not ``ORIGINAL``,
        reindex on the input of the last pass, invert the additive
        transformer last, then optionally restore the exact original values.

        The result is indexed like the input of the last pass, in its order:
        the rows the target grid added — a densified irregular index —
        are dropped, and the input rows the grid does not carry — a target
        coarser than the index — come back empty, or with their observed
        values under ``restore_original_values=True``.

        Args:
            X: Transformed features, as returned by ``transform``.
            y: Transformed target (optional).

        Returns:
            The original features, or the pair ``(X, y)`` when ``y`` is given,
            ``y`` under the name it was given.

        Raises:
            ValueError: If ``transform`` was never called: the provenance
                matrix it writes is what identifies the imputed cells. Also if
                the data to invert are stacked on a frequency level the last
                transform did not produce.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from tsforecast.imputation import HighFrequencyImputer
            >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
            >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
            >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0) * 3 + 10
            >>> imputer = HighFrequencyImputer(
            ...     target_frequency='M',
            ...     estimator=LinearRegression(),
            ...     restore_original_values=True,
            ... )
            >>> transformed = imputer.fit(data).transform(data)
            >>> restored = imputer.inverse_transform(transformed)
            >>> restored.shape, restored['q1'].dropna().head(3).tolist()
            ((24, 2), [10.0, 13.0, 16.0])
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
            y_col_name = resolve_target_column_name(y)
            data_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            data_work = X.copy()

        provenance = self.imputation_provenance_

        # 1. Retrait du niveau de fréquence, sur les données ET la provenance :
        # chacune le porte ou non, selon "keep_lower_frequencies"
        level_label = self._select_inverse_frequency_level(data_work, provenance)
        data_work = self._drop_frequency_level(data_work, level_label)
        provenance = self._drop_frequency_level(provenance, level_label)

        # 2. Remise à NaN de toute cellule non originale ; une cellule que la
        # provenance ne couvre pas n'est pas originale
        original_mask = (provenance == ProvenanceType.ORIGINAL).reindex(
            index=data_work.index, columns=data_work.columns, fill_value=False
        )
        # Colonnes hors périmètre de la provenance : jamais masquées
        untracked = [c for c in data_work.columns if c not in provenance.columns]
        if untracked:
            original_mask[untracked] = True
        data_work = data_work.where(original_mask)

        # 2'. Index de l'entrée de la dernière passe, dans son ordre : les
        # lignes ajoutées par la grille cible n'y portent plus rien
        data_work = data_work.reindex(self._original_X_.index)

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

        # 5. Scission X / y, symétrique de "_transform", la cible gardant le
        # nom qu'elle portait en entrée
        if y is not None and y_col_name in data_result.columns:
            return (
                data_result.drop(columns=[y_col_name]),
                data_result[y_col_name].rename(y.name),
            )
        return data_result
