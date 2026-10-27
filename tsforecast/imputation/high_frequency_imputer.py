"""Mixed-frequency imputer ``HighFrequencyImputer``, which imputes low-frequency columns
onto a finer target grid in cascade and records the provenance of every imputed value."""
# Importation des modules
# Modules de base
import warnings
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
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
from sklearn.base import BaseEstimator, TransformerMixin
# Utilitaires du package
from ..xy.transformers import XYPanelTimeSeriesTransformer
from ..panel.utils import normalize_entity_key
from ._components.aggregation_constraint import (
    ConstraintSetting,
    validate_aggregation_constraint,
)
from ._components.covariate_materializer import (
    COVARIATE_ELIGIBILITIES,
    COVARIATE_FALLBACKS,
    COVARIATE_STRATEGIES,
    DEFAULT_MATERIALIZATION_KEY as DEFAULT_FEATURE_KEY,
    CovariateMaterializer,
)
from ._components.imputation_plan import ImputationPlan, append_step
from ._components.imputation_window import ImputationScope, TrainingScope
from ._components.stage_scaler import ScaleFeatures, StageScaler
from ._components.variable_orderer import FIT_PREDICT_ORDERS, VariableOrderer
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
from ._engine.frequency_layout import stage_frequency_label
from ._engine.grids import training_mask_at
from ._engine.output import (
    assemble_output,
    drop_frequency_level,
    invert_additive_transformer,
    keep_original_cells,
    merge_target,
    restore_original_values,
    select_inverse_frequency_level,
    split_target,
)
from ._engine.preparation import (
    PreparedData,
    prepare_work_frame,
    resolve_target_column_name,
)
from ._engine.replay import replay_plan
from ._engine.run_context import FittedComponents, _RunContext
from ._engine.stage_runner import StageParams, StageRunner

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
        :func:`~tsforecast.imputation._engine.replay.check_transform_frequencies`,
        which names it; an extra column belongs to no plan and is carried through
        untouched. The strict equality of the base
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

    # Méthode auxiliaire de construction de l'exécuteur d'étapes d'une passe
    def _stage_runner(self, run: _RunContext) -> StageRunner:
        """Build the stage runner of one pass, fit or replay.

        The only place where the parameters of the imputer become the
        :class:`StageParams` the stage execution reads, so the fit and every
        replay execute their steps under the very same parameters.

        Args:
            run: Execution context of the pass.

        Returns:
            A :class:`StageRunner` on that context, the frequency layout of the
            fit and the parameters of the imputer.
        """
        params = StageParams(
            covariate_strategy=self.covariate_strategy,
            fit_predict_order=self.fit_predict_order,
            estimator=self.estimator,
            impute_intermediate_frequencies=self.impute_intermediate_frequencies,
            aggregation_constraint=self.aggregation_constraint,
            impute_unobserved_entities=self.impute_unobserved_entities,
            target_column=self.target_column_,
            target_as_covariate=self.target_as_covariate,
        )
        return StageRunner(
            run, self._frequency_layout_, params, log=self._log if self.verbose else None
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
        - PHASE 5: stage execution
          (:meth:`~tsforecast.imputation._engine.stage_runner.StageRunner.fit_stage`)
          — one model per (stage, variable), shared by its source-frequency
          groups, then one plan step per group, executed by the method every
          replay uses.
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
        # constructeur du jeu reçoit un callable au "kind" figé par "training_mask_at",
        # jamais le calculateur : il ne lit que le masque d'entraînement
        run = _RunContext.for_fit(
            self._fitted_components_,
            window_calculator=prepared.window_calculator,
            provenance_tracker=prepared.provenance_tracker,
            variable_orderer=variable_orderer,
            training_mask=partial(
                training_mask_at, prepared.window_calculator,
                is_panel=self.is_panel_, log=self._log,
            ),
            log=self._log if self.verbose else None,
        )

        # PHASE 5 — Exécution des étapes, chacune construite PUIS exécutée (l'étape
        # k dépend des imputations de l'étape k-1), recalages sous un bloc différé :
        # un avertissement par cas dégénéré, jamais un par variable × étape. Le
        # plan et l'ordre de chaque étape sont publiés au fil de la progression
        runner = self._stage_runner(run)
        self.imputation_plan_ = ImputationPlan()
        with run.aggregation_constraint.deferred_warnings():
            for stage_freq in self.frequency_progression_:
                stage = runner.fit_stage(prepared.X_work, stage_freq)
                for step in stage.steps:
                    self.imputation_plan_ = append_step(self.imputation_plan_, step)
                if stage.order is not None:
                    self.imputation_order_[stage.stage_label] = stage.order
                if stage.cv_scores:
                    self.imputation_cv_scores_[stage.stage_label] = stage.cv_scores
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
    # Transform — rejeu du plan figé
    # -------------------------------------------------------------------------
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
        initialized after it — then
        :func:`~tsforecast.imputation._engine.replay.replay_plan` replays every
        frozen step by
        :meth:`~tsforecast.imputation._engine.stage_runner.StageRunner.execute_step`,
        the very method the fit calls, on an execution context
        (:class:`_RunContext`) built from the fitted state alone, and
        :func:`~tsforecast.imputation._engine.output.assemble_output` shapes the
        output. No second execution loop, and no training set is ever rebuilt:
        the ``TrainingSetBuilder`` is not on this path.

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

        # PHASE 5' — Rejeu du plan sur un contexte d'exécution neuf, construit
        # depuis l'état ajusté seul : composants partagés du fit, matérialiseur
        # et registres vides, fenêtre et traceur de ce transform. Le rejeu ne
        # lit ni n'altère aucun état d'exécution du fit. Les étapes passent par
        # l'exécuteur d'étapes du fit, construit sur ce contexte
        run = _RunContext.for_transform(
            self._fitted_components_,
            window_calculator=prepared.window_calculator,
            provenance_tracker=prepared.provenance_tracker,
            transform_frequencies=prepared.new_frequencies,
            transform_misaligned=prepared.new_misaligned,
        )
        runner = self._stage_runner(run)
        # Recalages du rejeu sous un bloc différé, comme au fit : un
        # avertissement par cas dégénéré pour tout le rejeu
        with run.aggregation_constraint.deferred_warnings():
            frames, provenances = replay_plan(
                runner, self.imputation_plan_, prepared.X_work,
                entities=self.entities_, skip_columns=prepared.skipped_columns,
            )
        # Avertissements agrégés, un message par famille
        run.pending_warnings.emit()

        # PHASE 6' — Sortie de la passe : niveau cible, empilement multi-fréquences
        # ou lignes d'entrée de la passe de prédiction, et sa provenance
        output = assemble_output(
            frames, provenances,
            prediction=prediction,
            keep_lower_frequencies=self.keep_lower_frequencies,
            is_panel=self.is_panel_,
            input_index=X.index,
            input_columns=X.columns,
        )

        # Publication de l'état de la passe, d'un seul tenant et seulement
        # maintenant que rien ne peut plus lever
        self._original_X_ = prepared.snapshot_X
        self._original_y_ = prepared.snapshot_y
        self._stacked_index_nlevels: Optional[int] = output.stacked_nlevels
        self.imputation_provenance_ = output.provenance
        self.provenance_statistics_ = output.statistics

        # Passe de prédiction : y rendu tel quel, jamais imputé
        if prediction:
            return output.data if prepared.y is None else (output.data, prepared.y)
        # Scission X / y, la cible gardant le nom qu'elle portait en entrée
        return split_target(output.data, prepared.y, prepared.target_column)

    # -------------------------------------------------------------------------
    # Transformation inverse
    # -------------------------------------------------------------------------
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
        y_col_name = None if y is None else resolve_target_column_name(y)
        data_work = merge_target(X, y, y_col_name)

        # 1. Retrait du niveau de fréquence, sur les données ET la provenance :
        # chacune le porte ou non, selon "keep_lower_frequencies". Profondeur
        # empilée absente tant qu'aucun transform ne l'a écrite
        stacked_nlevels = getattr(self, '_stacked_index_nlevels', None)
        level_label = select_inverse_frequency_level(
            data_work, self.imputation_provenance_,
            stacked_nlevels=stacked_nlevels,
            source_label=getattr(self, '_source_index_frequency_label', None),
            target_frequency=self.effective_target_frequency_,
        )
        drop_level = partial(
            drop_frequency_level, label=level_label, stacked_nlevels=stacked_nlevels,
            index_names=self._original_X_.index.names,
        )
        data_work = drop_level(data_work)
        provenance = drop_level(self.imputation_provenance_)

        # 2. Remise à NaN de toute cellule non originale, puis index de l'entrée
        # de la dernière passe, dans son ordre : les lignes ajoutées par la
        # grille cible n'y portent plus rien
        data_work = keep_original_cells(data_work, provenance).reindex(
            self._original_X_.index
        )

        # 3. Inversion de la transformation additive, en DERNIER (miroir du
        # transform, qui l'applique en premier)
        data_result = invert_additive_transformer(self.additive_transformer_, data_work)

        # 4. Restauration optionnelle des valeurs d'origine exactes
        if self.restore_original_values:
            data_result = restore_original_values(
                data_result, self._original_X_, self._original_y_, y_col_name
            )

        # 5. Scission X / y, symétrique de "_transform", la cible gardant le
        # nom qu'elle portait en entrée
        return split_target(data_result, y, y_col_name)
