"""High frequency imputer for mixed frequency time series data.

This module provides the HighFrequencyImputer class to impute high-frequency values
for low-frequency series in mixed-frequency datasets using machine learning models.
"""
# Importation des modules
# Modules de base
import warnings
from collections import OrderedDict
from dataclasses import replace
from typing import Callable, Dict, List, Literal, Optional, Sequence, Union, Any, Tuple
# Manipulation de données
import numpy as np
import pandas as pd
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import check_scoring
# Utilitaires du package
from ..xy.transformers import XYPanelTimeSeriesTransformer
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import (
    normalize_frequency,
    is_higher_frequency,
    get_frequency_order,
)
from ..panel.utils import (
    is_panel_data,
    get_unique_panel_entities,
    normalize_entity_key,
    split_variable_key,
    extract_column_names,
    group_keys_by_entity_and_variable,
    get_entity_mask,
)
from ..utils.frequency.utils import detect_frequency, detect_index_frequency
from .imputation_plan import (
    ImputationStep,
    INTERPOLATE_FALLBACK,
    to_entity_tuple,
)
from .provenance import ImputationProvenanceTracker, ProvenanceType
from .imputation_window import ImputationWindowCalculator, ImputationScope
from .target_frequency_validator import TargetFrequencyValidator
from .frequency_aligner import FrequencyAligner

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

    Vocabulary correspondence (review §6): some early design discussions used
    a slightly different vocabulary than the final parameters below —

    | Design vocabulary  | Actual parameter(s)/attribute                     |
    |---------------------|----------------------------------------------------|
    | ``fit_on_imputed``  | ``train_on_partial_coverage`` + ``cascade_refitting`` |
    | ``keep_lower_frequencies`` | ``keep_lower_frequencies`` (unchanged)      |
    | ``P1``              | ``imputation_window_``                              |
    | ``refit``           | ``cascade_refitting``                               |

    Parameters:
        target_frequency: Target frequency for imputation. Can be:
            - str: Single frequency applied to all series/entities
            - Dict[entity_id, str]: Entity-specific target frequencies for
              panel data. Entity keys may be written in scalar form
              (``{'FR': 'M'}``); they are normalized to tuples at init (see
              the key format contract below).
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
            It also drives the intra-stage cascade, identically in ``fit`` and
            in ``transform`` : only when True do the
            variables imputed later in a stage see the values the earlier ones
            produced. A fallback step never feeds the following variables,
            whatever the flag — ``fit`` produces no value for it at all, so
            ``transform`` writes its interpolation to the output only, never to
            the covariates. As a consequence, with ``cascade_refitting=False``
            the fit writes nothing and ``imputation_provenance_fit_`` carries no
            MODEL_ON_*/DISAGGREGATED mark; ``imputation_provenance_``, written
            by ``transform``, carries them in every regime.
        covariate_eligibility: How a covariate's availability is aggregated over
            the entities of a panel when ``cascade_refitting=False`` restricts
            the features to those observed at prediction time. ``'any_entity'``
            (default) keeps a column available for at least one entity, leaving
            the other entities' rows NaN for the estimator to handle (see the
            NaN contract on ``estimator``); ``'all_entities'`` is the
            conservative setting for estimators that do not tolerate NaN, at
            the price of dropping a column for everyone. Ignored for a time
            series and when ``cascade_refitting=True``.
        keep_lower_frequencies: If True, output includes all intermediate
            frequencies in a MultiIndex structure: (entity..., frequency,
            date) for panel data, (frequency, date) for time series. Every level,
            including the target-frequency one, is labeled with its real
            frequency string — never 'target' — so the target level is
            identified via effective_target_frequency_. If False, the
            output is the plain frame of the last cascade stage (target
            frequency), with no frequency level in its index.
        on_frequency_mismatch: How to handle target_frequency higher than data ('error'/'warn').
        coverage_threshold: Minimum ratio of columns with data (0-1) for extended window.
        imputation_scope: Training window scope ('strict', 'extended_backward',
            'extended_forward', 'extended_both').
        train_on_partial_coverage: If True, use imputed values for training outside the strict window.
        train_on_partial_fit_order: Order in which variables are imputed at
            each cascade stage:
            - 'frequency': Sort by frequency level then entity count (default)
            - 'cv': Cross-validate each variable with its estimator and
              impute the easiest ones (highest score) first. Applies
              regardless of ``train_on_partial_coverage`` — the two
              parameters are independent.
        min_cv_train_size: Minimum number of scorable observations (target
            non-NaN, restricted to the strict imputation window) a variable
            must have for ``train_on_partial_fit_order='cv'`` to score it by
            cross-validation. Below this, the variable falls back to the
            'frequency' ordering group. Must be >= 2.
        cv_scoring: Scoring used by ``train_on_partial_fit_order='cv'``, in
            the ``sklearn`` sense: a registry string (e.g.
            ``'neg_mean_absolute_percentage_error'``, ``'r2'``), a scorer
            from ``sklearn.metrics.make_scorer``, or a callable
            ``scorer(estimator, X, y) -> float``, validated at scoring time
            via ``sklearn.metrics.check_scoring``. Higher is better in every
            case (sklearn convention), so the variable with the HIGHEST
            score is imputed first. Behavior change versus the previous
            hardcoded MAPE: rows with ``y == 0`` are no longer excluded —
            ``mean_absolute_percentage_error`` instead floors the
            denominator at ``eps``, so a zero target produces a very large
            error rather than being ignored. Prefer
            ``'neg_root_mean_squared_error'`` for series containing zeros.
        cv_n_splits: Number of folds used by ``train_on_partial_fit_order='cv'``.
            Must be >= 2.
        scale_features: If True, divide X_train by the number of sub-periods
            as well as y_train, which is always divided. Set it to True when
            the training covariates were aggregated by summation to the
            variable's frequency, so that both sides of the model are
            expressed at the sub-period scale used at prediction time. Each
            covariate gets its own divisor, read per entity, since a column
            aggregated on both sides does not necessarily carry the stage
            frequency at prediction time. For a daily or weekly stage the
            divisors are period-invariant averages (30, 91, 365 rather than
            the true calendar counts) — see :meth:`_stage_scale_factor`.
        enforce_period_totals: If True (default), the sub-periods predicted
            for one period of a lower-frequency variable are rescaled
            proportionally so that they sum back to the value observed for
            that period. Only the additive
            constraint is optional: an imputed variable is ALWAYS predicted
            over its whole period, anchor dates included, so that its column
            never mixes the low-frequency total with sub-period values. Set
            it to False to keep the raw model output, homogeneous in scale
            but free of any additive constraint. Periods that are only
            partly predicted, that hold no observation, or whose predictions
            sum to zero are never rescaled. The option drives the resclaing
            only, never the provenance marking: an anchor date is marked
            DISAGGREGATED either way, since it is the row a real observation
            was read from. Marking it MODEL_ON_* instead
            used to empty the training mask of the next cascade stage and
            send the whole variable to the interpolation fallback.
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

    Key format contract (panel data, review §3.4/§5.4):
        There is ONE representation of an entity — the **tuple** — used
        everywhere, including for a single entity level: ``('France',)``,
        never ``'France'``. A variable key is the FLATTENED
        ``(entity..., column)`` tuple produced by ``detect_frequency``; use
        :func:`tsforecast.panel.utils.split_variable_key` to split it back
        into ``(entity_tuple, column)``. Entity keys supplied by the user in
        a ``target_frequency`` dict are normalized to tuples by
        ``_validate_target_frequency_format``, so ``{'FR': 'M'}`` becomes
        ``{('FR',): 'M'}`` in ``effective_target_frequency_``. This
        normalization is recomputed at every ``fit`` call rather than stored
        on ``self.target_frequency`` (B3/§3.16): ``self.target_frequency``
        stays IDENTICAL to whatever was passed to ``__init__``, unnormalized,
        because ``sklearn.clone()`` requires ``get_params()[name] is`` the
        received value. Always read ``effective_target_frequency_`` after
        ``fit`` — never ``self.target_frequency`` — when tuple-normalized
        keys are needed. Consequently every internal lookup on
        ``effective_target_frequency_``, ``detected_frequencies_`` or a stage
        frequency dict is a SINGLE lookup: there is no defensive
        scalar/tuple fallback left, and a ``KeyError`` is a real bug.

    Attributes:
        detected_frequencies_: Detected frequency per variable — column name
            for a time series, flattened ``(entity..., column)`` tuple for a
            panel.
        variable_categories_: Variable keys per category (same key format as
            detected_frequencies_): ``Dict[VariableCategory, List[Union[str, Tuple]]]``
            with keys 'aggregate', 'impute', and 'target_freq'.
        imputation_order_: Ordered list of variables for cascading imputation,
            computed once in PHASE 3 of ``_fit``. INFORMATIVE ONLY (B22): the
            actual per-stage order used by the cascade is recomputed by
            PHASE 5 (``ordered_impute_keys``, possibly via
            ``train_on_partial_fit_order='cv'``), which never reads this
            attribute back. Kept for introspection/debugging, not consumed
            internally.
        imputation_plan_: SINGLE SOURCE OF TRUTH of the fitted cascade
            (review §5.3): the ordered list of
            :class:`~tsforecast.frequency.ImputationStep`, one entry per
            (stage, variable group) registered by ``_fit``, in fit order.
            ``_transform`` replays exactly this list; the four attributes
            below are read-only views derived from it, kept for the
            notebooks and tests that consume them. Each step being frozen,
            the plan can only be rebuilt by a new ``fit``.
        imputation_models_: DERIVED from ``imputation_plan_``, READ-ONLY (no
            setter: assigning to it raises ``AttributeError``). Fitted
            imputation models, keyed by cascade stage:
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
            dict with keys ``model``, ``feature_cols``,
            ``scale_factor`` (sub-period count of the stage),
            ``fit_scale_factor`` (the one baked into the model at fit time),
            ``pred_freq`` and ``trained_on_imputed``.
        model_fitting_order_: DERIVED from ``imputation_plan_``, READ-ONLY.
            List of those same ``stage_key`` tuples, in the exact order in
            which stages were registered.
            ``len(imputation_plan_) == len(imputation_models_)
            == len(model_fitting_order_)`` after any fit.
        stage_groups_: DERIVED from ``imputation_plan_``, READ-ONLY.
            Metadata of each ``stage_key``, needed at replay time and
            populated for EVERY registered stage, interpolation fallbacks
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
            ``(frequency, date)`` for a time series, ``(entity..., frequency,
            date)`` for a panel, one row-block per cascade stage — so that a
            cell's provenance always describes the value actually present at
            that (frequency, date) in the output, instead of one single
            level's provenance being reused (falsely) to describe every
            level (review §2.8.4).
        imputation_window_: Tuple (start, end) of the imputation window of the
            fit data, where all series have data. For a panel, each element is
            a dict keyed by entity tuple, straight from
            ``ImputationWindowCalculator``. It describes the training set;
            ``transform`` recomputes its own window on the data it imputes,
            the window being a constraint on covariate availability rather
            than an estimated parameter.
        training_window_: Tuple (start, end) of the extended training window.
        frequency_progression_: DERIVED from ``imputation_plan_``, READ-ONLY.
            Dict mapping each variable name to the ordered list of
            ``freq_label`` cascade stages (as carried by
            ``model_fitting_order_``) at which it was fitted, consecutive
            duplicates collapsed. Being read off the plan, it reflects every
            stage across all entities for a panel variable, not just the
            first one encountered.
        additive_transformer_: Fitted additive transformer.
        is_panel_: Whether data is panel data.
        feature_columns_: X columns (features).
        target_column_: y column if provided.
        effective_target_frequency_: Actual target frequency used after
            validation — a string for a time series, a dict keyed by entity
            TUPLE for a panel (user-supplied scalar keys are normalized at
            init, see the key format contract above).
        entities_: Unique entities in panel data, as tuples.
        _source_index_frequency_label: Frequency label of the index seen at fit
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
        covariate_eligibility: Literal['any_entity', 'all_entities'] = 'any_entity',
        keep_lower_frequencies: bool = True,
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
        coverage_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        train_on_partial_coverage: bool = False,
        train_on_partial_fit_order: Literal['frequency', 'cv'] = 'frequency',
        min_cv_train_size: int = 10,
        cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error',
        cv_n_splits: int = 5,
        scale_features: bool = True,
        enforce_period_totals: bool = True,
        restore_original_values: bool = False,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        verbose: bool = False,
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
                The estimator must tolerate NaN in X, or be wrapped in a
                Pipeline handling them (e.g. SimpleImputer) if needed — the imputer
                does not fill missing covariates itself.
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
            covariate_eligibility: How a covariate's availability is aggregated
                over the entities of a panel when ``cascade_refitting=False``
                filters the features down to those observed at prediction time.
                Ignored for a time series, and ignored altogether when
                ``cascade_refitting=True``.
                - ``'any_entity'`` (default): keep the column as soon as it is
                  available for at least one entity. The rows of the other
                  entities stay NaN and are handed to the estimator, which is
                  exactly the contract stated on ``estimator``. On a real panel,
                  dropping a covariate for everyone because a single entity
                  lacks it wastes information.
                - ``'all_entities'``: conservative, meant for estimators that do
                  NOT tolerate NaN. Under a bare LinearRegression a partially
                  NaN column makes ``predict`` raise and sends THE WHOLE GROUP
                  to the interpolation fallback — worse than dropping the
                  column. The transformer cannot guess the estimator's
                  tolerance, hence the parameter rather than a hard-wired rule.
            keep_lower_frequencies: If True, output includes all intermediate
                frequencies in a MultiIndex structure — (entity..., frequency,
                date) for panel data, (frequency, date) for time series —
                every level labeled with its real frequency string (the
                target level is identified via effective_target_frequency_,
                not by a dedicated 'target' label). If False, only the
                target frequency is returned as a plain frame.
            on_frequency_mismatch: How to handle target_frequency higher than
                data frequencies ('error'/'warn').
            coverage_threshold: Minimum percentage of columns (0-1) that
                must have non-null values for extended training window.
                INERT under the default ``imputation_scope='strict'`` (B22):
                it is only consulted by the ``'extended_*'`` scopes, which
                relax the strict (coverage == 1.0) window.
            imputation_scope: Training window scope.
            train_on_partial_coverage: If True, use imputed values for
                training models outside the imputation window.
            train_on_partial_fit_order: Order for variable imputation:
                - 'frequency': By frequency level then entity count (default)
                - 'cv': Cross-validation to find easiest variables first,
                  applied regardless of ``train_on_partial_coverage``
            min_cv_train_size: Minimum scorable observations for a variable
                to be cross-validated under ``train_on_partial_fit_order='cv'``;
                below this it falls back to the 'frequency' ordering group.
                Must be >= 2.
            cv_scoring: Scoring for ``train_on_partial_fit_order='cv'``, any
                value accepted by ``sklearn.metrics.check_scoring`` (registry
                string, ``make_scorer`` scorer, or callable). Higher is
                better; the highest-scoring variable is imputed first.
            cv_n_splits: Number of CV folds for ``train_on_partial_fit_order='cv'``.
                Must be >= 2.
            scale_features: If True, divide X_train by the number of
                sub-periods alongside y_train, which is always divided.
            enforce_period_totals: If True (default), rescale the sub-periods
                predicted for one period of a lower-frequency variable so
                that they sum back to the value observed for that period.
                Independently of this flag, an
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
                When set, ``convert_cols_to_index=True`` (fixed by this
                class) converts ``time_col``/``panel_cols`` to X's index
                BEFORE ``fit``/``transform`` ever see it — this conversion
                itself does not depend on the flag's value, only the
                metadata capture used to align ``y`` does (B15). ``y``, if
                provided, must therefore keep the SAME index it had before
                that conversion (typically a plain ``RangeIndex`` matching
                X's original row order) — it is realigned onto X's converted
                index automatically; see ``_align_target_index``.
            panel_cols: List of column names identifying panel entities.
            verbose: If True, print progress messages (cascade stages, fits,
                fallbacks and their reason, dates left unimputed for lack of
                covariates, end-of-transform provenance summary) prefixed
                ``[HighFrequencyImputer]`` via :meth:`_log`. Default is
                False, which keeps the transformer silent (review §5.6).
        """
        # Initialisation du parent
        super().__init__(
            time_col=time_col, panel_cols=panel_cols,
            validate_input=True, strict_validation=True,
            auto_sort=False, convert_cols_to_index=True
        )

        # Validation des paramètres
        # "target_frequency" est seulement VALIDÉ ici (la valeur normalisée
        # renvoyée est jetée) : sklearn.clone() exige get_params()[name] is
        # la valeur reçue par __init__, donc l'attribut stocké ci-dessous
        # doit rester la valeur TELLE QUE REÇUE. La normalisation (clés
        # d'entité en tuples, fréquences en forme de base) est recalculée à
        # chaque fit() et vit dans effective_target_frequency_ (B3/§3.16)
        self._validate_target_frequency_format(target_frequency)
        self._validate_estimator(estimator)
        if additive_transformer is not None:
            self._validate_additive_transformer(additive_transformer)

        if on_frequency_mismatch not in ['error', 'warn']:
            raise ValueError(
                f"on_frequency_mismatch must be 'error' or 'warn', "
                f"got '{on_frequency_mismatch}'"
            )
        if covariate_eligibility not in ('any_entity', 'all_entities'):
            raise ValueError(
                f"covariate_eligibility must be 'any_entity' or 'all_entities', "
                f"got '{covariate_eligibility}'"
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
        if train_on_partial_fit_order not in ('frequency', 'cv'):
            raise ValueError(
                f"train_on_partial_fit_order must be 'frequency' or 'cv', "
                f"got '{train_on_partial_fit_order}'"
            )
        if min_cv_train_size < 2:
            raise ValueError(
                f"min_cv_train_size must be >= 2, got {min_cv_train_size}"
            )
        if cv_n_splits < 2:
            raise ValueError(
                f"cv_n_splits must be >= 2, got {cv_n_splits}"
            )
        # Avertissement : croiser les deux paramètres pour
        # imposer min_cv_train_size >= cv_n_splits couplerait deux options
        # indépendantes. Sous ce seuil, cross_val_score échoue sur chaque
        # pli (trop peu d'observations par pli) et la variable retombe dans
        # le groupe de repli à chaque fois
        if min_cv_train_size < cv_n_splits:
            warnings.warn(
                f"min_cv_train_size ({min_cv_train_size}) < cv_n_splits "
                f"({cv_n_splits}): cross-validation will systematically "
                f"fall back to the 'frequency' ordering for every variable.",
                UserWarning
            )
        # Validation groupée des booléens (B22) : aucun ne l'était, un
        # 'frequency'/1/None passé par erreur se propageait silencieusement
        # jusqu'à un "if" qui l'évalue en vérité/mensonge Python générique
        boolean_params = {
            'cascade_refitting': cascade_refitting,
            'keep_lower_frequencies': keep_lower_frequencies,
            'scale_features': scale_features,
            'enforce_period_totals': enforce_period_totals,
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
        self.additive_transformer = additive_transformer
        self.estimator = estimator
        self.cascade_refitting = cascade_refitting
        self.covariate_eligibility = covariate_eligibility
        self.keep_lower_frequencies = keep_lower_frequencies
        self.on_frequency_mismatch = on_frequency_mismatch
        self.coverage_threshold = coverage_threshold
        self.imputation_scope = imputation_scope
        self.train_on_partial_coverage = train_on_partial_coverage
        self.train_on_partial_fit_order = train_on_partial_fit_order
        self.min_cv_train_size = min_cv_train_size
        self.cv_scoring = cv_scoring
        self.cv_n_splits = cv_n_splits
        self.scale_features = scale_features
        self.enforce_period_totals = enforce_period_totals
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
    # Alignement de la cible
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de résolution du nom de colonne de la cible
    def _resolve_target_column_name(self, y: pd.Series) -> str:
        """Resolve the column name used for ``y`` once merged into a frame.

        Single naming rule shared by ``_fit``, ``_transform`` and
        ``_inverse_transform``. Two independent rules previously let the
        fit-time name (``0``, from an unnamed Series' ``to_frame()``) and
        the transform-time name (``'__target__'``) diverge, so the target
        was never found among ``X_stage.columns`` and silently skipped
        imputation at transform time.

        Args:
            y: Target series, possibly unnamed.

        Returns:
            ``y.name`` if set, else the fallback ``'__target__'``.
        """
        return y.name if y.name is not None else '__target__'

    # Méthode auxiliaire d'alignement de l'index de la cible sur celui de X
    # /!\ Voir si on souhaite pas tout de même tolérer ici ou dans le XYTransformer des index de X et y qui ne coïncident pas (en particulier X en time series peut contenir plus d'observations que y car on align X_t et avec y_t+h)
    def _align_target_index(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Align ``y``'s index onto ``X``'s, tolerating the col->index step.

        When ``time_col``/``panel_cols`` are used, the base transformer
        converts ``X``'s columns to a Multi/DatetimeIndex before ``_fit``/
        ``_transform`` ever see it (``convert_cols_to_index=True``, fixed by
        this class) — but it never touches ``y``, which still carries
        whatever index it had at the call site. If that original index
        matches the one ``conversion_metadata_`` recorded before conversion,
        ``y`` is repositioned onto ``X.index`` (row order is preserved:
        ``auto_sort`` is fixed to False). Otherwise the two indices must
        already match exactly: silently reindexing on mismatched index
        VALUES would grow the working frame with NaN rows instead of
        raising.

        Args:
            X: Working features, already validated/converted.
            y: Target series as received by ``fit``/``transform``.

        Returns:
            ``y`` reindexed onto ``X.index`` when that is safe to do.

        Raises:
            ValueError: If ``X`` and ``y`` indices neither match nor derive
                from the same pre-conversion index.
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

        # Aucun des deux cas : un désaccord d'index est un vrai bug appelant,
        # pas un cas à corriger silencieusement (concat aligne sur la VALEUR
        # de l'index, pas sur la position)
        raise ValueError(
            "X and y have different indices. y must share X's index, or — "
            "for column-based panel/time-series data (time_col/panel_cols) "
            "— the index X had before those columns were converted to the "
            "index."
        )

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
                entities to frequencies). Dict entity keys may be given in
                scalar form for a single-level panel (``{'FR': 'M'}``).

        Returns:
            Normalized target_frequency. For a dict, BOTH sides are
            normalized: the frequency values via ``normalize_frequency`` and
            the entity keys into tuples via ``normalize_entity_key``, so that
            ``{'FR': 'M'}`` becomes ``{('FR',): 'M'}``. This is the only place
            where user-supplied entity keys enter the imputer, and every
            downstream consumer then indexes ``effective_target_frequency_``
            by tuple without any defensive fallback (review §5.4).

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

    # Instance du convertisseur de fréquence
    @property
    def _freq_converter(self) -> FrequencyConverter:
        """Lazy initialization of frequency converter."""
        # Initialisation si l'attribut n'existe pas déjà
        if not hasattr(self, '_freq_converter_cache'):
            self._freq_converter_cache = FrequencyConverter()
        return self._freq_converter_cache

    # -------------------------------------------------------------------------
    # Vues dérivées du plan d'imputation (review §5.3)
    # -------------------------------------------------------------------------
    # Ces quatre attributs étaient autrefois maintenus en parallèle par "_fit".
    # Ils sont désormais dérivés de "imputation_plan_", seul état écrit par le
    # fit : la synchronisation ne peut donc plus diverger. Aucun setter n'est
    # exposé — une affectation lève AttributeError, et la levée d'AttributeError
    # avant le fit préserve le "hasattr" négatif de l'attribut absent d'autrefois
    def _require_plan(self) -> List[ImputationStep]:
        """Return the fitted imputation plan, or raise if there is none.

        Returns:
            The list of :class:`ImputationStep` produced by ``fit``.

        Raises:
            AttributeError: If the imputer has not been fitted yet, so that
                ``hasattr`` on the derived views stays False before ``fit``.
        """
        # Absence de plan : l'imputer n'a pas encore été entraîné
        plan = self.__dict__.get('imputation_plan_')
        if plan is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                f"'imputation_plan_'. Call fit() first."
            )

        return plan

    # Registre des modèles, dérivé du plan
    @property
    def imputation_models_(self) -> "OrderedDict[Tuple, Union[str, Dict[str, Any]]]":
        """Fitted models keyed by cascade stage — derived view, READ-ONLY.

        Rebuilt from ``imputation_plan_`` at each access, in plan order, so
        ``list(imputation_models_) == model_fitting_order_`` always holds.
        Mutating the returned mapping has no effect on the imputer; there is
        no setter either. To use custom estimators, pass them to the
        ``estimator`` parameter before fitting.

        Returns:
            OrderedDict mapping ``(freq_label, group_key)`` to the stage
            entry: the string ``'interpolate_fallback'``, or a dict with
            keys ``model``, ``feature_cols``, ``scale_factor``,
            ``fit_scale_factor``, ``pred_freq`` and ``trained_on_imputed``.

        Raises:
            AttributeError: If the imputer has not been fitted yet.
        """
        return OrderedDict(
            (step.stage_key, step.to_registry_entry())
            for step in self._require_plan()
        )

    # Ordre d'enregistrement des étapes, dérivé du plan
    @property
    def model_fitting_order_(self) -> List[Tuple]:
        """Stage keys in fit order — derived view, READ-ONLY.

        Returns:
            List of the ``(freq_label, group_key)`` stage keys, in the exact
            order in which ``fit`` registered them.

        Raises:
            AttributeError: If the imputer has not been fitted yet.
        """
        return [step.stage_key for step in self._require_plan()]

    # Métadonnées de groupe par étape, dérivées du plan
    @property
    def stage_groups_(self) -> "OrderedDict[Tuple, Dict[str, Any]]":
        """Group metadata of every stage — derived view, READ-ONLY.

        Populated for EVERY registered stage, interpolation fallbacks
        included, unlike ``imputation_models_`` whose fallback entries are
        bare strings.

        Returns:
            OrderedDict mapping each stage key to a dict with keys
            ``var_name``, ``f_var`` and ``entities``.

        Raises:
            AttributeError: If the imputer has not been fitted yet.
        """
        return OrderedDict(
            (step.stage_key, step.group_metadata())
            for step in self._require_plan()
        )

    # Progression de fréquence par variable, dérivée du plan
    @property
    def frequency_progression_(self) -> Dict[str, List[str]]:
        """Cascade stages each variable went through — derived view, READ-ONLY.

        Returns:
            Dict mapping each variable name to the ordered list of stage
            frequency labels at which it was registered, consecutive
            duplicates collapsed.

        Raises:
            AttributeError: If the imputer has not been fitted yet.
        """
        return self._compute_frequency_progression()

    # -------------------------------------------------------------------------
    # Méthodes auxiliaires
    # -------------------------------------------------------------------------

    # Méthode auxiliaire de classification des variables par rapport à une fréquence cible
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
        impute_vars = list(self.variable_categories_['impute'])

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
                _, var_name = split_variable_key(key)
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
        """Determine variable order using cross-validated scoring.

        Variables with the highest CV score (easiest to predict, sklearn's
        "greater is better" convention) are placed first. Falls back to the
        'frequency' ordering group when fewer than ``min_cv_train_size``
        scorable observations are available for training.

        Args:
            X: Working data.
            impute_vars: List of variable keys to order.

        Returns:
            Ordered list, easiest variables first.
        """
        # Vérification qu'il y a au moins deux variables à ordonner
        if len(impute_vars) <= 1:
            return impute_vars

        # Deux groupes de scores, non comparables entre eux : le score de CV
        # (échelle du "scoring" choisi) et l'ordre de fréquence utilisé en
        # repli (entiers, potentiellement grands) ne doivent pas être triés
        # ensemble, sous peine d'envoyer mécaniquement les variables en repli
        # en fin (ou en tête) de liste suivant l'échelle relative des deux. On
        # les trie séparément puis on concatène : d'abord les variables
        # notées par CV, puis les replis
        cv_scored: List[Tuple[Union[str, Tuple], float]] = []
        fallback_scored: List[Tuple[Union[str, Tuple], float]] = []

        # Parcours des variables à imputer
        for var_key in impute_vars:
            # Extraction du nom de la variable de la clé
            _, var_name = split_variable_key(var_key)

            # Préparation des données d'entraînement dans la fenêtre d'imputation
            # stricte : l'ordonnancement doit comparer la qualité de l'imputation des variables sur des données
            # de qualité homogène, sinon le score d'une variable dépend de l'étendue
            # de son extension et l'ordre obtenu n'est plus interprétable
            if hasattr(self, '_imputation_window_calc') and self._imputation_window_calc._is_fitted:
                mask = self._imputation_window_calc.get_imputation_window_mask(
                    X, kind='strict'
                )
            else:
                mask = pd.Series(True, index=X.index)

            # Extraction des colonnes de features : les entités d'un panel sont
            # dans l'index, jamais dans les colonnes, donc aucun filtre sur
            # "panel_cols" n'est nécessaire ici (cf. "_prepare_training_data")
            feature_cols = [c for c in X.columns if c != var_name]

            # Score -inf (pire score possible, convention "greater is
            # better") si la série est univariée : elle ne doit jamais passer
            # en tête
            if not feature_cols:
                cv_scored.append((var_key, -np.inf))
                continue

            # Restriction aux lignes réellement exploitables : la cible doit
            # être observée. "mask" seul laisse passer les sous-périodes
            # NaN d'une variable basse fréquence, sur lesquelles tout fit
            # échoue et le score valait "inf" pour chaque variable — rendant
            # le tri suivant un no-op silencieux. Les NaN résiduels de
            # "X_sub" restent (features), à la charge de l'estimateur
            # (Pipeline avec imputer, ou modèle tolérant les NaN)
            scoring_rows = mask & X[var_name].notna()
            X_sub = X.loc[scoring_rows, feature_cols]
            y_sub = X.loc[scoring_rows, var_name]

            # Fallback si moins de "min_cv_train_size" observations exploitables
            if len(X_sub) < self.min_cv_train_size:
                fallback_scored.append((var_key, get_frequency_order(
                    self.detected_frequencies_[var_key]
                )))
                continue

            # Extraction de l'estimateur
            estimator = self._get_estimator_for_variable(var_name)
            # Score -inf si l'estimateur n'est pas spécifié
            if estimator is None:
                cv_scored.append((var_key, -np.inf))
                continue

            # Validation du "scoring" : lève un ValueError explicite si la
            # valeur n'est pas reconnue par sklearn (registre, make_scorer,
            # appelable de signature scorer(estimator, X, y))
            scorer = check_scoring(estimator, scoring=self.cv_scoring)

            # Initialisation de la KFold : le mélange (shuffle=True) est
            # volontaire ici, l'objectif est de produire un ORDRE de variables
            # à imputer, pas une évaluation honnête d'un modèle de série
            # temporelle.
            # En effet les données ne sont normalement plus des séries temporelles
            # mais sont toutes alignées afin d'être agrégées.
            # "cross_val_score" absorbe la gestion d'erreur par pli
            # (estimateur qui lève sur un pli donné) via "error_score=np.nan"
            scores = cross_val_score(
                estimator, X_sub, y_sub,
                cv=KFold(n_splits=self.cv_n_splits, shuffle=True, random_state=42),
                scoring=scorer,
                error_score=np.nan,
            )
            # Score -inf si TOUS les plis ont échoué, signalé explicitement :
            # sans ce log l'échec systématique resterait silencieux
            if np.all(np.isnan(scores)):
                self._log(
                    f"[fit] CV scoring failed on every fold for '{var_name}': "
                    f"falling back to the lowest possible score."
                )
                score = -np.inf
            else:
                score = float(np.nanmean(scores))
            # Ajout à la liste des scores
            cv_scored.append((var_key, score))

        # Tri décroissant (convention sklearn "greater is better") : le score
        # le plus élevé = variable la plus facile = imputée en premier. Le
        # groupe de repli est trié dans le MÊME sens (reverse=True) pour
        # s'aligner sur le tri par fréquence de "_fit"
        # (sorted(..., key=get_frequency_order, reverse=True), fréquence la
        # plus basse d'abord) — sans quoi les deux chemins ordonneraient les
        # fréquences en sens inverse l'un de l'autre
        cv_scored.sort(key=lambda x: x[1], reverse=True)
        fallback_scored.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in cv_scored] + [v for v, _ in fallback_scored]

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
        ``frequency`` level in the multi-frequency output : 
        every level, including the last (target) one, keeps its
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
        frequency — 12 for a yearly variable predicted monthly.

        The count comes from :meth:`~FrequencyConverter.get_conversion_factor`
        and is exact for nested calendar pairs (M -> Y = 12, Q -> Y = 4,
        M -> Q = 3). Daily and weekly stages instead get the conventional
        counts of the duration table (D -> M = 30, D -> Q = 91, D -> Y = 365,
        W -> Y = 52): there, the factor is a period-invariant average, biased
        by +7.1% in February and -3.2% in January, applying 91 to quarters of
        90, 91 and 92 days alike, and 365 to leap years.
        ``enforce_period_totals=True`` absorbs that bias in the output, never
        in the fitted coefficients, since the factor also feeds
        ``fit_scale_factor``.

        Calendar-exact counting is available as
        :meth:`~FrequencyConverter.count_subperiods_per_period`, but it
        returns one count per period: wiring it in makes the scale factor
        non-scalar, which is the per-row divisor introduced together with
        ``train_on_own_imputations``.

        Args:
            var_key: Variable key (column name, or ``(entity..., column)``).
            pred_freq: Prediction frequency of the stage.

        Returns:
            Number of stage sub-periods per variable period, as a
            period-invariant average for daily and weekly stages.
        """
        # Extraction de la fréquence de prédiction s'il s'agit d'un dictionnaire
        if isinstance(pred_freq, dict):
            if isinstance(var_key, tuple):
                # Décomposition de la clé et accès DIRECT : la variable est
                # imputée à cette étape, donc son entité y figure
                entity, _ = split_variable_key(var_key)
                pf = pred_freq[entity]
            else:
                pf = list(pred_freq.values())[0]
        else:
            pf = pred_freq

        # "get_conversion_factor(haut, bas)" : nombre de périodes hautes dans
        # une période basse — exact sur les paires calendaires emboîtées,
        # moyenne conventionnelle sur les étapes journalières
        return self._freq_converter.get_conversion_factor(
            pf,
            self.detected_frequencies_[var_key],
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
        # Normalisation en tuple : "droplevel" rend des scalaires pour un panel
        # à un seul niveau d'entité
        entity_tuples = [normalize_entity_key(key) for key in entity_index]
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

        Implements the additive constraint : the
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

        # Extraction du nom de la variable
        var_name = stage_group['var_name']
        if var_name not in X_input.columns:
            return values, pd.Series(False, index=values.index)

        # Construction de l'appartenance aux périodes de la fréquence source
        period_membership = self._period_membership(
            X_input.index, stage_group['f_var'], stage_group.get('entities')
        )

        # /!\ J'aimerais rendre optionnelle la cohérence des imputatiohs entre périodes et qu'elle soit pilotée par un argument. Cela peut peut-être être fait en rendant toute la méthode "_apply_period_totals" optionnelle dans le "fit" et pas seulement cette sous-méthode.
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
            mask |= get_entity_mask(X_input, entity)

        return pd.Series(mask, index=X_input.index)

    # Méthode auxiliaire de construction du masque des dates-ancres écrites
    @staticmethod
    def _anchor_mask(
        X_input: pd.DataFrame,
        var_name: str,
        index: pd.Index,
    ) -> pd.Series:
        """Locate the anchor dates among the cells written at a stage.

        An anchor date is a row where the low-frequency variable carries a
        real observation in the untouched input frame. Re-expressing it at
        the stage frequency does not make it a plain model output: it is a
        real observation spread over its period, which is what
        DISAGGREGATED denotes. This holds whether or not the additive
        rescaling ran, hence a mask read off the data rather than off the
        rescaling's return value.

        Args:
            X_input: Untouched input frame of the stage.
            var_name: Column being imputed.
            index: Index of the cells that received a value.

        Returns:
            Boolean Series aligned on ``index``, all False when the column
            is absent from ``X_input``.

        Examples:
            >>> # frame = pd.DataFrame({'yr': [1.0, np.nan, np.nan]}, index=idx)
            >>> # HighFrequencyImputer._anchor_mask(frame, 'yr', idx).tolist()
            >>> # [True, False, False]
        """
        # Colonne absente : aucune ancre à signaler
        if var_name not in X_input.columns:
            return pd.Series(False, index=index)

        return X_input[var_name].reindex(index).notna()

    # Méthode auxiliaire de récupération du modèle déjà entraîné pour une variable
    def _model_for_var(
        self,
        group_key: Union[str, Tuple],
    ) -> Optional[ImputationStep]:
        """Find the step already fitted for a variable, whatever the stage.

        Backs the ``cascade_refitting=False`` regime (review §5.2): a
        variable is fitted once, at the first stage where it is imputable,
        then reused at the following stages. Fallback steps are ignored so
        that a variable which could not be fitted at an earlier stage still
        gets its chance at a later one.

        The plan is walked directly rather than through the derived
        ``imputation_models_`` view, which would rebuild a whole dict at
        every call of the fit loop.

        Args:
            group_key: Registry group key — the variable name, or
                ``(variable name, detected frequency)`` for a panel where the
                frequency differs by entity (see review §2.4).

        Returns:
            The most recently registered step holding a model for the
            variable, or None when none was ever fitted.
        """
        # Parcours du plan sur la composante group_key de la clé d'étape
        found: Optional[ImputationStep] = None
        for step in self.imputation_plan_:
            if step.var_key == group_key and not step.is_fallback:
                found = step

        return found

    # Méthode auxiliaire de prédiction à l'échelle de l'étape
    def _stage_predictions(
        self,
        step: ImputationStep,
        X_features: pd.DataFrame,
    ) -> np.ndarray:
        """Predict at the scale of the stage the step was registered for.

        The scale factor is baked into the model at fit time (``y_train``
        is divided by it), never applied at prediction time. A model reused
        across stages (``cascade_refitting=False``) therefore predicts at
        the scale of the stage it was fitted on, and its output must be
        brought back to the current stage: a yearly variable fitted at the
        quarterly stage predicts quarterly values, three times the monthly
        ones expected at the monthly stage.

        Args:
            step: Plan step of the stage, holding the fitted model, its
                ``fit_scale_factor`` and the stage ``scale_factor``.
            X_features: Features to predict on, at the stage frequency.

        Returns:
            Predictions at the scale of the stage.
        """
        # Prédiction brute, à l'échelle de l'étape d'entraînement
        predictions = step.model.predict(X_features)

        # Report de l'échelle d'entraînement vers celle de l'étape : le rapport
        # vaut 1.0 pour un modèle entraîné pour l'étape courante
        stage_scale = step.scale_factor
        fit_scale = step.fit_scale_factor
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
            # Parcours des variables à imputer (à fréquence plus faible que la fréquence cible)
            for key in self.variable_categories_['impute']:
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

                # Filtrage des étapes sans variable à imputer : une étape n'est
                # utile que s'il existe au moins une variable de fréquence
                # strictement inférieure à imputer à cette étape
                sorted_freqs = [
                    f for f in sorted_freqs
                    if self._classify_variables_at_frequency(f)['impute']
                ]
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
                        if is_higher_frequency(self.effective_target_frequency_[entity], freq) or (normalize_frequency(self.effective_target_frequency_[entity], return_format='base') == freq):
                            freq_dict[entity] = freq
                    # Ajout du dictionnaire à la liste
                    freq_list.append(freq_dict)

                # Ajout de la fréquence cible pour les entités dont elle n'est pas déjà couverte
                sorted_freq_set = set(sorted_freqs)
                freq_dict = {
                    entity: target_freq
                    for entity, target_freq in self.effective_target_frequency_.items()
                    if normalize_frequency(target_freq, return_format='base') not in sorted_freq_set
                }
                if freq_dict:
                    freq_list.append(freq_dict)

                # Filtrage des étapes sans variable à imputer : une étape n'est
                # utile que s'il existe au moins une variable de fréquence
                # strictement inférieure à imputer à cette étape
                freq_list = [
                    f for f in freq_list
                    if self._classify_variables_at_frequency(f)['impute']
                ]
                return freq_list

    # Méthode auxiliaire de construction du frame d'une étape de prédiction
    def _build_stage_frame(
        self,
        X_original: pd.DataFrame,
        imputed_store: Dict[str, pd.Series],
        pred_freq: Union[str, Dict],
        aggregate_keys: Optional[List[Union[str, Tuple]]] = None,
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
            aggregate_keys: Variable keys to aggregate to ``pred_freq``, as
                already computed by the caller via
                :meth:`_classify_variables_at_frequency`. When ``None``
                (default), it is recomputed internally from ``pred_freq`` —
                kept for backward compatibility with callers passing only
                three positional arguments (e.g. the notebook, existing
                tests).

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

        # Agrégation des colonnes plus fréquentes que la fréquence de l'étape :
        # réutilisation de la classification de l'appelant si fournie, pour
        # éviter de la recalculer alors qu'il vient de le faire
        if aggregate_keys is None:
            aggregate_keys = self._classify_variables_at_frequency(pred_freq)['aggregate']
        return self._freq_aligner._aggregate_to_target(
            X_stage, aggregate_keys, pred_freq
        )

    # Méthode auxiliaire de préparation des données d'entraînement du modèle d'imputation
    def _prepare_training_data(
        self,
        X_stage: pd.DataFrame,
        X_original: pd.DataFrame,
        var_key: Union[str, Tuple],
        pred_freq: Union[str, Dict],
    ) -> Tuple[pd.DataFrame, pd.Series, float, Union[pd.Series, pd.DataFrame]]:
        """Prepare X_train, y_train and scale factors for a variable.

        Covariates are **aggregated to the variable's own frequency**
        ``f_var`` (and not merely sampled at the variable's anchor dates):
        the model is fitted on ``f_var``-scale sums of the covariates
        against the ``f_var``-scale target. Predictions, on the other hand,
        are made on covariates carried at the stage frequency ``pred_freq``.
        The gap between both scales is precisely what ``scale_factor`` is
        meant to absorb.

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
        # /!\ Voir si le fait de ne jamais entraîner la prédiction d'une variable y sur ses valeurs imputées lors de cascade précédente est facilement modifiable et paramétrisable. Si ce n'est pas trop compliqué, j'aimerais que l'on puisse choisir si lors de la cascade on entraine également le modèle sur des valeurs prédites à des étapes précédentes (et donc bruitées ou non). Dans le mode de cascade_refitting, j'aimerais qu'on puisse également ajouter grâce à un argument la provenance des différents X (soit sous forme de variable catégorielle car le odèle tolère des variables non numériques, soit sous forme de one-hot).

        # Extraction du nom de la variable
        _, var_name = split_variable_key(var_key)

        # Détermination des colonnes de features
        feature_cols = [
            c for c in X_stage.columns
            if c != var_name
        ]

        # Sans réentraînement en cascade, "imputed_store" reste vide au fit comme
        # au transform : une covariable de fréquence moindre que l'étape n'y est
        # jamais imputée et reste NaN partout sauf sur ses dates-ancres.
        # L'entraînement doit donc se restreindre aux covariables qui seront
        # effectivement disponibles à la prédiction à cette étape
        # /!\ Dans ma compréhension on a également un problème similaire lorsque self.cascade_refitting=True : Lorsque l'on a un jeu de données avec des variables mensuelles, trimestrielles et annuelles par exemple et que l'on veut imputer une variable annuelle à la fréquence trimestrielle alors on ne peut entrainer le modèle servant à imputer la variable annuelle à la fréquence trimestrielle qu'à partir des variables trimstrielles et mensuelles (que l'on peut aggréger à la fréquence trimestrielle pour X pred et à la fréquence annuelle pour X_train) mais on ne peut pas utiliser les autres variables annuelles (sauf celles déjà imputées) pour prédire entrainer le modèle à prédire la valeur de la variable annuelle à la fréquence trimestrielle
        # Réponse : constat confirmé et reproduit — défaut B28, traité par le prompt 23
        # (architecture §3.17). La garde ci-dessous doit disparaître au profit d'un
        # état de matérialisation tenu au fil de la cascade, et le portage des basses
        # fréquences sur la grille de l'étape devient un mode de "covariate_carrying"
        if not self.cascade_refitting:
            # Détermination des variables disponibles à la fréquence de prédiction
            eligible = [
                c for c in feature_cols if self._is_available_at(c, pred_freq)
            ]
            # Cas dégénéré : sans ce journal, le repli par interpolation qui
            # découle du retour anticipé ci-dessous (via "len(X_train) < 2")
            # est incompréhensible
            if feature_cols and not eligible:
                # Logging
                self._log(
                    f"[fit] All covariates filtered out for '{var_name}' at "
                    f"stage {self._stage_frequency_label(pred_freq)} "
                    f"(cascade_refitting=False, covariate_eligibility="
                    f"'{self.covariate_eligibility}')"
                )
            feature_cols = eligible

        # Si la série est univariée, renvoie des éléments vides
        if not feature_cols:
            return pd.DataFrame(), pd.Series(dtype=float), 1.0, pd.Series(dtype=float)

        # Valeurs d'origine de la variable : jamais enrichies de ses propres imputations
        y_source = X_original[var_name]

        # Masque d'entraînement : fenêtre d'entraînement + valeurs non-null pour y
        # /!\ Pourquoi se restreint-on au à l'imputation mask pour les données d'entrainement ? Cela ne me semble a priori pertinent que pour les données de prédiction
        if hasattr(self, '_imputation_window_calc') and self._imputation_window_calc._is_fitted:
            # Extraction du masque de la fenêtre d'entraînement, réglable
            # indépendamment de celle de la prédiction (cf. training_scope du
            # calculateur) : un modèle peut ainsi être ajusté sur une plage plus
            # large que celle sur laquelle il impute
            training_mask = self._imputation_window_calc.get_imputation_window_mask(
                X_stage, kind='training'
            )
            # Hybridation avec la période de disponibilité de y
            training_mask &= y_source.notna()
        else:
            training_mask = y_source.notna()

        # Restriction aux données originales si train_on_partial_coverage=False
        if not self.train_on_partial_coverage:
            if hasattr(self, '_provenance_tracker'):
                # DISAGGREGATED et AGGREGATED sont admis au même titre qu'ORIGINAL : 
                # les dates-ancres d'une variable désagrégée à une étape antérieure 
                # changent de provenance alors que "y_source" reste lu dans les
                # données d'origine — les exclure viderait le masque à l'étape
                # suivante et enverrait tout en repli
                # /!\ Ne serait-il pas logique de rajouter ProvenanceType.AGGREGATED ici car dans le cas où l'on a des données additives, il n'y a pas d'approximation dans l'agrégation ?
                # Réindexation défensive : le masque de provenance est aligné sur
                # provenance_matrix_.index, potentiellement différent de celui de
                # training_mask (X_stage.index) ; un "&" entre index divergents
                # produirait des NaN convertis en objet puis un KeyError au .loc suivant
                original_mask = self._provenance_tracker.get_mask(
                    [ProvenanceType.ORIGINAL, ProvenanceType.DISAGGREGATED],
                    column=var_name
                ).reindex(training_mask.index).fillna(False).astype(bool)
                training_mask = training_mask & original_mask

        # Agrégation des covariables à la fréquence propre de la variable
        f_var = self.detected_frequencies_[var_key]
        # Sélection des covariables strictement plus fréquentes que la variable
        feature_agg_keys = [
            key for key in self._classify_variables_at_frequency(f_var)['aggregate']
            if split_variable_key(key)[1] != var_name
        ]
        X_features = self._freq_aligner._aggregate_to_target(
            X_stage, feature_agg_keys, f_var
        )

        # Extraction des données d'entraînement
        X_train = X_features.loc[training_mask, feature_cols]
        y_train = y_source.loc[training_mask]

        # Calcul du facteur de mise à l'échelle : nombre de sous-périodes de la
        # fréquence de prédiction (haute) dans une période de la fréquence
        # détectée de la variable (basse) — ex. variable annuelle prédite au
        # mensuel : 12
        scale_factor = self._stage_scale_factor(var_key, pred_freq)

        # Diviseurs propres à chaque covariable : chacune doit retrouver
        # l'échelle qu'elle portera au predict, qui n'est ni systématiquement
        # scale_factor ni systématiquement sa propre fréquence
        feature_factors = self._covariate_scaling_divisors(
            X_train, f_var, pred_freq, scale_factor
        )

        return X_train, y_train, scale_factor, feature_factors

    # Méthode auxiliaire de lecture des fréquences détectées d'une colonne
    def _column_frequencies_by_entity(self, column: str) -> Dict[tuple, str]:
        """Detected frequencies of one column, one entry per entity.

        Args:
            column: Bare column name, without the entity part of the key.

        Returns:
            Mapping entity tuple -> detected frequency. The single key is
            ``()`` for a time series. Empty when the column carries no
            detected frequency at all.

        Examples:
            >>> # imputer._column_frequencies_by_entity('ip')
            >>> # {('FR',): 'M', ('DE',): 'Q'}
        """
        # Itération + filtre plutôt qu'une compréhension indexée par le nom nu :
        # un panel peut porter la même colonne à des fréquences différentes
        # selon l'entité, et l'indexer par nom nu ferait gagner la DERNIÈRE
        # entité rencontrée — donc dépendre de l'ordre des colonnes en entrée
        return {
            split_variable_key(key)[0]: freq
            for key, freq in self.detected_frequencies_.items()
            if split_variable_key(key)[1] == column
        }

    # Méthode auxiliaire de disponibilité d'une covariable à la prédiction
    def _is_available_at(
        self,
        column: str,
        pred_freq: Union[str, Dict],
    ) -> bool:
        """Tell whether a covariate will still be observed at prediction time.

        A covariate at least as frequent as the stage is aggregated onto the
        stage grid and is therefore fully observed there. A covariate STRICTLY
        less frequent than the stage is not: without cascade refitting it is
        never imputed, so it stays NaN everywhere but on its own anchor dates —
        training a model on it would fit a coefficient for a column that is
        missing at predict.

        On a panel the column carries one frequency PER ENTITY, so the
        comparison is made entity by entity against that entity's own
        prediction frequency, then aggregated according to
        ``covariate_eligibility``.

        Args:
            column: Bare column name, without the entity part of the key.
            pred_freq: Prediction frequency of the stage: a string for a time
                series, an entity -> frequency dict for a panel.

        Returns:
            True when the column is eligible as a feature for this stage.
            False for a column carrying no detected frequency at all.

        Examples:
            >>> # 'ip' monthly for France, quarterly for Allemagne, stage at 'M'
            >>> # imputer.covariate_eligibility = 'any_entity'
            >>> # imputer._is_available_at('ip', {('France',): 'M',
            >>> #                                  ('Allemagne',): 'M'})
            >>> # True   ('all_entities' would give False)
        """
        # Fréquences détectées de la colonne, une par entité :
        # "_column_frequencies_by_entity" itère et filtre plutôt que d'indexer
        # par le nom nu — un panel peut porter la même colonne à deux fréquences
        # selon l'entité, et l'indexation par nom nu ferait gagner la dernière
        # entité rencontrée, donc dépendre de l'ordre des colonnes en entrée
        frequencies = self._column_frequencies_by_entity(column)

        # Verdict par entité
        verdicts = []
        for entity, f_cov in frequencies.items():
            if isinstance(pred_freq, dict):
                # Entité absente de l'étape : elle n'y prédit rien, elle ne vote pas
                if entity not in pred_freq:
                    continue
                pf = pred_freq[entity]
            else:
                pf = pred_freq
            # Disponible si et seulement si la covariable n'est pas MOINS
            # fréquente que l'étape : elle y est alors agrégée, donc observée
            verdicts.append(not is_higher_frequency(pf, f_cov))

        # Une colonne sans aucune entrée n'est pas éligible
        if not verdicts:
            return False

        # Agrégation sur les entités du panel, gouvernée par le paramètre :
        # "any_entity" laisse les lignes des autres entités en NaN à charge de
        # l'estimateur, "all_entities" écarte la colonne pour tout le monde
        predicate = any if self.covariate_eligibility == 'any_entity' else all

        return predicate(verdicts)

    # Méthode auxiliaire de calcul du diviseur d'une covariable
    def _covariate_divisor(
        self,
        f_col: Optional[str],
        f_var: str,
        pred_freq: str,
        default: float,
    ) -> float:
        """Divisor carrying one covariate from its training to its prediction scale.

        Args:
            f_col: Detected frequency of the covariate, None when unknown.
            f_var: Detected frequency of the variable being imputed.
            pred_freq: Prediction frequency of the stage, for one entity.
            default: Fallback divisor when the frequencies cannot be compared.

        Returns:
            Number of prediction-scale sub-periods the training value totals.
        """
        try:
            # Colonne jamais ré-agrégée vers f_var par "_prepare_training_data" :
            # elle porte déjà au fit l'échelle qu'elle aura au predict
            if not is_higher_frequency(f_col, f_var):
                return 1.0

            # Fréquence réellement portée par la colonne au predict :
            # "_build_stage_frame" n'agrège que ce qui est STRICTEMENT plus fin
            # que l'étape, une covariable plus grossière garde la sienne
            f_stage = pred_freq if is_higher_frequency(f_col, pred_freq) else f_col

            return self._freq_converter.get_conversion_factor(f_stage, f_var)
        except (ValueError, TypeError):
            return default

    # Méthode auxiliaire de calcul des diviseurs de mise à l'échelle des covariables
    def _covariate_scaling_divisors(
        self,
        X_train: pd.DataFrame,
        f_var: str,
        pred_freq: Union[str, Dict],
        default: float,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Compute the divisor carrying each covariate to its prediction scale.

        A covariate is seen at two different scales. At fit time
        :meth:`_prepare_training_data` aggregates by summation to ``f_var``
        every column strictly finer than ``f_var``. At prediction time the
        same column is read raw from the stage frame, where it carries
        ``f_stage``: the stage frequency when :meth:`_build_stage_frame`
        aggregated it (covariate strictly finer than the stage), its own
        detected frequency otherwise.

        The divisor is therefore the number of ``f_stage`` sub-periods in one
        ``f_var`` period, and ``1.0`` for a column that was never
        re-aggregated.

        Frequencies are read  per entity: a panel may carry the same column at
        different frequencies depending on the entity, in which case a single
        divisor per column cannot be right for every row.

        Args:
            X_train: Training frame, already aggregated at ``f_var``.
            f_var: Detected frequency of the variable being imputed.
            pred_freq: Prediction frequency of the stage (str for a time
                series, entity -> frequency dict for a panel).
            default: Fallback divisor for covariates whose frequency is
                unknown, typically the stage scale factor.

        Returns:
            Series of divisors indexed by column name when every entity
            agrees, DataFrame indexed and columned like ``X_train`` otherwise.
        """
        # Entités portées par les fréquences détectées : "()" en série temporelle
        entities = {
            split_variable_key(key)[0] for key in self.detected_frequencies_
        }

        # Diviseur par (entité, colonne)
        per_entity: Dict[tuple, Dict[str, float]] = {
            entity: {} for entity in entities
        }
        # Parcours des colonnes
        for column in X_train.columns:
            # Extraction des fréquences de la colonne par entité
            frequencies = self._column_frequencies_by_entity(column)
            # Parcours des entités
            for entity in entities:
                # Fréquence de prédiction
                pf = (
                    pred_freq[entity] if isinstance(pred_freq, dict) else pred_freq
                )
                # Population du dictionnaire avec le facteur de mise à l'échelle
                per_entity[entity][column] = self._covariate_divisor(
                    frequencies.get(entity), f_var, pf, default
                )

        # Forme compacte quand toutes les entités s'accordent : c'est le cas des
        # séries temporelles et des panels homogènes, de loin le plus fréquent
        rows = list(per_entity.values())
        if not rows:
            return pd.Series(dtype=float)
        if all(row == rows[0] for row in rows[1:]):
            return pd.Series(rows[0], dtype=float)

        # Ventilation ligne à ligne : le diviseur dépend conjointement de
        # l'entité et de la colonne, aucune Series indexée sur l'une des deux
        # dimensions ne peut le porter
        fallback = dict.fromkeys(X_train.columns, default)
        entity_per_row = [
            normalize_entity_key(key) for key in X_train.index.droplevel(-1)
        ]
        return pd.DataFrame(
            [per_entity.get(entity, fallback) for entity in entity_per_row],
            index=X_train.index,
            columns=X_train.columns,
        )

    # Méthode auxiliaire d'application du facteur de mise à l'échelle aux données
    def _apply_frequency_scaling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_factor: float,
        feature_factors: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Scale training data to the prediction-frequency scale.

        The target is always divided by the number of sub-periods so the
        model directly predicts values at the prediction frequency.
        Features are divided too when scale_features=True (i.e. when they
        were aggregated by summation to the variable's low frequency), each
        one by its own divisor when ``feature_factors`` is given.

        Because the model already returns sub-period values, predictions
        must never be multiplied back by the scale factor.

        Args:
            X_train: Training features (aggregated at the variable's frequency).
            y_train: Training target (at the variable's low frequency).
            scale_factor: Number of prediction-frequency sub-periods per
                variable-frequency period (e.g. 12 for yearly -> monthly).
            feature_factors: Per-covariate divisors from
                :meth:`_covariate_scaling_divisors` — a Series indexed by
                column name, or a DataFrame aligned on ``X_train`` when a
                panel carries a column at different frequencies depending on
                the entity. Falling back to ``scale_factor`` for every column
                only holds when all covariates share the prediction frequency,
                which is why the divisors are computed per covariate.

        Returns:
            Tuple of (scaled X_train, scaled y_train).
        """
        # Court-circuit uniquemet si rien n'est à mettre à l'échelle : un
        # scale_factor unitaire n'implique pas des feature_factors unitaires.
        # Une variable trimestrielle prédite au trimestre
        # laisserait sinon ses covariables annuelles intactes, là où une
        # variable de la même étape avec scale_factor = 3 divise les siennes
        scalar_scale = np.isscalar(scale_factor)
        factors_are_unit = feature_factors is None or bool(
            np.all(np.asarray(feature_factors, dtype=float) == 1.0)
        )
        if scalar_scale and scale_factor == 1.0 and factors_are_unit:
            return X_train, y_train

        # La cible est TOUJOURS ramenée à l'échelle d'une sous-période
        y_scaled = y_train / scale_factor

        # Les features ne le sont que si elles ont été agrégées par somme,
        # chacune par le diviseur qui lui rend son échelle de prédiction
        if not self.scale_features:
            return X_train, y_scaled
        if feature_factors is None:
            return X_train / scale_factor, y_scaled

        # Réalignement défensif : le DataFrame porte un diviseur par ligne et
        # par colonne, la Series un diviseur par colonne
        if isinstance(feature_factors, pd.DataFrame):
            divisors = feature_factors.reindex(
                index=X_train.index, columns=X_train.columns
            ).fillna(scale_factor)
        else:
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
        here from ``isna()``. The caller therefore
        intersects the disaggregation scope with the imputation window (see
        :meth:`_prediction_masks`).

        Groups are ordered from the richest covariate pattern to the
        poorest, the empty pattern last: callers skip that final group
        instead of imputing from nothing.

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

    # Méthode auxiliaire d'ajustement d'un calculateur de fenêtre d'imputation
    def _fit_imputation_window(
        self,
        data: pd.DataFrame,
    ) -> Tuple[Optional[ImputationWindowCalculator], Optional[ValueError]]:
        """Fit a fresh imputation-window calculator on the given data.

        The imputation window is a constraint on covariate availability, not
        an estimated parameter: it is a deterministic function of the frame it
        is computed on. ``_fit`` and ``_transform`` therefore
        share this factory and each compute the window on their own data, so
        that ``transform`` on out-of-sample dates keeps predicting instead of
        silently emptying the column. On identical data both calls rebuild the
        same window, which is what preserves
        ``fit_transform(X) == fit(X).transform(X)``.

        Args:
            data: Frame the window is computed on. Must be the frame BEFORE
                the additive transformer, on both paths, or the two windows
                would not coincide.

        Returns:
            Tuple ``(calculator, error)``: the fitted calculator and None, or
            None and the ``ValueError`` raised by the calculator. Callers
            decide how to report the failure — their messages differ.
        """
        # Instanciation avec les hyperparamètres de l'imputeur
        calculator = ImputationWindowCalculator(
            coverage_threshold=self.coverage_threshold,
            imputation_scope=self.imputation_scope,
            min_columns=2,
        )
        # Estilation de la fenêtre
        try:
            calculator.fit(data)
        except ValueError as error:
            return None, error

        return calculator, None

    # Méthode auxiliaire de mise en forme lisible des bornes d'une fenêtre
    @staticmethod
    def _window_bounds_label(
        window_calc: ImputationWindowCalculator,
        max_entities: int = 3,
    ) -> str:
        """Render the bounds of a fitted imputation window as a short string.

        Args:
            window_calc: Fitted calculator whose bounds are rendered.
            max_entities: Maximum number of panel entities listed before the
                rendering is truncated with an ellipsis.

        Returns:
            ``'[start, end]'`` for time series data, ``'France [a, b], ...'``
            for panel data, ``'undefined'`` when no window could be derived.
        """
        # Extraction des dates de début et de fin
        start = window_calc.imputation_window_start_
        end = window_calc.imputation_window_end_

        # Cas des séries temporelles : bornes scalaires
        if not isinstance(start, dict):
            if start is None or end is None:
                return "undefined"
            return f"[{start.date()}, {end.date()}]"

        # Cas des données de panel : bornes par entité, tronquées
        entities = list(start)
        rendered = [
            f"{entity} [{start[entity].date()}, {end[entity].date()}]"
            if start.get(entity) is not None and end.get(entity) is not None
            else f"{entity} undefined"
            for entity in entities[:max_entities]
        ]
        suffix = ", ..." if len(entities) > max_entities else ""
        return ", ".join(rendered) + suffix if rendered else "undefined"

    # Méthode auxiliaire de construction des masques de désagrégation et de prédiction
    def _prediction_masks(
        self,
        X_stage: pd.DataFrame,
        X_input: pd.DataFrame,
        stage_group: Optional[Dict[str, Any]],
        var_name: str,
        window_calc: Optional[ImputationWindowCalculator],
        context: str = '',
    ) -> Tuple[pd.Series, pd.Series]:
        """Build the disaggregation scope and the predictable rows of a stage.

        The scope is the whole grid of the group, anchor dates included :
        the variable is re-expressed at the stage frequency
        over all of it, so no row of the scope may keep a low-frequency
        total. The predictable rows are the scope restricted to the
        imputation window : nothing is ever predicted outside
        the window, where covariate coverage is insufficient by
        construction.

        Callers therefore blank the predictable rows and write back only those
        actually predicted — an anchor left unpredicted inside the window ends
        up NaN rather than carrying its period total. Rows of the scope lying
        outside the window are never touched: nothing is produced there, so
        nothing may be destroyed there either.

        A scope entirely outside the window is reported: it is the signature of
        a ``transform`` on data the window does not cover, which used to empty
        the column in complete silence.

        Args:
            X_stage: Stage frame the masks are aligned on.
            X_input: Untouched input frame of the stage.
            stage_group: Entry of :attr:`stage_groups_` for the stage key.
            var_name: Column being imputed.
            window_calc: Calculator carrying the imputation window, computed
                on the data being imputed (see :meth:`_fit_imputation_window`).
                None or unfitted leaves the whole scope predictable.
            context: Label used in the warning message.

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
        if window_calc is not None and window_calc._is_fitted:
            # Fenêtre de prédiction : "extended_forward" existe précisément pour
            # imputer les fins de série retardées, elle ne doit donc pas être
            # confondue ici avec la fenêtre stricte ni avec celle d'entraînement
            window = window_calc.get_imputation_window_mask(
                X_stage, kind='imputation'
            )
            predictable = scope & window.reindex(X_stage.index).fillna(False).astype(bool)

        # Périmètre non vide entièrement hors fenêtre : distinct du périmètre
        # vide, qui lui est silencieux à juste titre. Sans cet avertissement,
        # une variable ne produisant plus aucune valeur passe inaperçue
        if scope.any() and not predictable.any():
            # Recherche des observations en dehors de la fenêtre
            out_of_window = X_stage.index[scope.to_numpy()]
            suffix = f" for {context}" if context else ""
            # Calcul des bornes de la fenêtre
            bounds = (
                self._window_bounds_label(window_calc)
                if window_calc is not None and window_calc._is_fitted
                else "undefined"
            )
            # Logging
            self._log(
                f"No row in the imputation window{suffix}: "
                f"{len(out_of_window)} date(s) in scope, window {bounds} "
                f"({list(out_of_window[:5])}{'...' if len(out_of_window) > 5 else ''})"
            )
            # Warning
            warnings.warn(
                f"No row of the imputation scope{suffix} falls inside the "
                f"imputation window (bounds: {bounds}): all "
                f"{len(out_of_window)} date(s) in scope (e.g. "
                f"{out_of_window[0]}) are left as they came in, nothing is "
                f"imputed for them.",
                UserWarning
            )

        return scope, predictable

    # Méthode auxiliaire de prédiction d'une variable sur les lignes prédictibles
    def _predict_stage_values(
        self,
        step: ImputationStep,
        X_stage: pd.DataFrame,
        rows_mask: pd.Series,
        context: str = '',
    ) -> pd.Series:
        """Predict a variable on the rows it can actually be predicted on.

        Shared by the intermediate cascade of :meth:`_fit` and the replay of
        :meth:`_transform` so both paths produce identical predictions on
        identical data. One rule: rows are grouped by their pattern of
        available covariates and a group without a single usable covariate is
        left unimputed — nothing is fabricated where nothing was observed.

        The covariates still missing on a partial pattern are handed to the
        estimator AS NaN: filling them here would substitute a hidden
        imputation policy for the user's own. The estimator must therefore
        tolerate NaN in X, or be wrapped in a Pipeline handling them (see the
        ``estimator`` parameter of :meth:`__init__`); one that does not raises,
        and the caller falls back on linear interpolation.

        Args:
            step: Plan step of the stage.
            X_stage: Stage frame holding the covariates.
            rows_mask: Rows the variable may be imputed on, from
                :meth:`_prediction_masks`.
            context: Label used in warning messages.

        Returns:
            Predictions indexed by the rows actually predicted — a subset of
            ``rows_mask``, empty when no row carries any covariate.
        """
        # Extraction des noms des colonnes de features
        feature_cols = list(step.feature_cols)

        # Regroupement des lignes à prédire par motif de covariables disponibles
        prediction_samples = self._determine_prediction_samples(
            X_stage, rows_mask, feature_cols
        )

        # Initialisation de la liste des observations prédites
        predicted: List[pd.Series] = []
        # Initialisation de la liste des observations ignorées
        skipped: List[pd.Index] = []

        # Parcours des groupes, du plus riche en covariables au plus pauvre
        for sample_index, available_cols in prediction_samples:
            # Restriction aux features connues du modèle
            usable = [c for c in feature_cols if c in available_cols]
            # Aucune covariable observée : la date n'est pas imputée
            if not usable:
                skipped.append(sample_index)
                continue

            # Le motif décide SI l'on impute ; les covariables encore
            # manquantes du groupe partent telles quelles à l'estimateur, à qui
            # il revient de les traiter (cf. contrat NaN de "estimator")
            X_pred = X_stage.loc[sample_index, feature_cols]
            predicted.append(
                pd.Series(
                    self._stage_predictions(step, X_pred),
                    index=sample_index,
                )
            )

        # Avertissement agrégé sur les dates laissées non imputées
        if skipped:
            # Extraction des observations ignorées
            skipped_index = skipped[0].append(skipped[1:]) if len(skipped) > 1 else skipped[0]
            suffix = f" for {context}" if context else ""
            # Logging
            self._log(
                f"No covariate available{suffix} on {len(skipped_index)} date(s): "
                f"left unimputed ({list(skipped_index[:5])}{'...' if len(skipped_index) > 5 else ''})"
            )
            # Warning
            warnings.warn(
                f"No covariate available{suffix} on {len(skipped_index)} date(s) "
                f"(e.g. {skipped_index[0]}): they are left unimputed rather than "
                f"predicted from no observed covariate at all.",
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

    # Méthode auxiliaire d'écriture des valeurs produites à une étape
    def _write_stage_values(
        self,
        X_stage: pd.DataFrame,
        X_input: pd.DataFrame,
        var_name: str,
        predictions: pd.Series,
        scope_mask: pd.Series,
        predict_mask: pd.Series,
        tracker: ImputationProvenanceTracker,
        disaggregated_mask: pd.Series,
        trained_on_imputed: bool,
        extra_frames: Sequence[pd.DataFrame] = (),
        context: str = '',
    ) -> bool:
        """Blank the window part of the scope, then write the values back.

        Single point of truth for the "empty then rewrite" discipline shared
        by the cascade of :meth:`_fit` and the replay of :meth:`_transform`.
        Two rules :

        1. blanking is restricted to ``scope ∩ window``. That is the region
           the variable is committed to being re-expressed on, hence the only
           one where keeping a low-frequency total would mix two scales.
           Outside the window nothing is produced, so nothing is destroyed;
        2. nothing is blanked at all when no value was produced — erasing an
           input observation without putting anything in its place is never
           the intended behaviour.

        The provenance follows the data: cells blanked but not rewritten lose
        their original mark, so the matrix never declares observed a cell that
        no longer holds a value.

        The provenance of a written cell depends on its NATURE, never on
        whether the additive rescaling succeeded: an anchor date stays an
        anchor date whether ``enforce_period_totals`` is on or off, so the
        mask handed to :meth:`_mark_imputed_cells` is the union of the
        rescaled cells and of the scope's anchors.

        Args:
            X_stage: Stage frame written in place.
            X_input: Untouched input frame of the stage, holding the
                variable's original low-frequency observations. Only read, to
                locate the anchor dates of the written scope.
            var_name: Column being imputed.
            predictions: Values to write, indexed by the rows they belong to.
            scope_mask: Disaggregation scope, from :meth:`_prediction_masks`.
            predict_mask: Predictable rows, from :meth:`_prediction_masks`.
            tracker: Provenance tracker of the running pass.
            disaggregated_mask: Boolean mask, aligned on ``predictions.index``,
                of the cells produced by a rescaled disaggregation.
            trained_on_imputed: Whether the model saw imputed values at fit
                time (drives MODEL_ON_MIXED vs MODEL_ON_TRUE).
            extra_frames: Further frames to apply the very same blank-then-
                rewrite to, without touching the provenance — the covariate
                mirror of :meth:`_transform` when it is a distinct object
                from the output frame.
            context: Label used in log messages.

        Returns:
            True if values were written, False if the column was left as is.
        """
        # Construction du suffixe
        suffix = f" for {context}" if context else ""

        # Aucune valeur produite : la colonne reste intacte, y compris ses
        # observations d'entrée
        if len(predictions) == 0:
            # Logging
            self._log(
                f"No value produced{suffix}: column '{var_name}' left untouched"
            )
            return False

        # Vidage restreint à la fenêtre, puis réécriture des valeurs produites
        blank_mask = scope_mask & predict_mask
        self._blank_and_write(X_stage, var_name, predictions, blank_mask)

        # Mêmes écritures sur les frames miroirs, sans marquage de provenance :
        # le tracker est unique et n'est écrit qu'une fois, pour le frame de sortie
        for frame in extra_frames:
            if frame is not X_stage:
                self._blank_and_write(frame, var_name, predictions, blank_mask)

        # Marquage de la provenance des cellules effectivement écrites.
        # Une date-ancre reste une date-ancre, que le recalage additif ait eu
        # lieu ou non : le masque de marquage est l'union des cellules recalées
        # et des ancres du périmètre. "disaggregated_mask" conserve ainsi son
        # seul rôle informatif — « cette cellule a été recalée »
        # Réindexation explicite : "_mark_imputed_cells" consomme le masque
        # positionnellement (".to_numpy()" contre "predictions.index"), l'union
        # doit donc être alignée dans cet ordre exact
        anchor_mask = self._anchor_mask(X_input, var_name, predictions.index)
        marked_mask = (
            disaggregated_mask.reindex(predictions.index).fillna(False).astype(bool)
            | anchor_mask
        )
        self._mark_imputed_cells(
            tracker, var_name, predictions.index, marked_mask, trained_on_imputed
        )

        # Cellules vidées sans réécriture : elles ne portent plus de valeur,
        # elles ne peuvent donc plus être déclarées ORIGINAL
        cleared = X_stage.index[blank_mask.to_numpy()].difference(predictions.index)
        if (
            len(cleared) > 0
            and tracker.provenance_matrix_ is not None
            and var_name in tracker.provenance_matrix_.columns
        ):
            cleared = cleared.intersection(tracker.provenance_matrix_.index)
            if len(cleared) > 0:
                tracker.clear_provenance(var_name, cleared)

        return True

    # Méthode auxiliaire de vidage puis réécriture d'une colonne d'un frame d'étape
    @staticmethod
    def _blank_and_write(
        frame: pd.DataFrame,
        var_name: str,
        predictions: pd.Series,
        blank_mask: pd.Series,
    ) -> None:
        """Apply the blank-then-rewrite discipline to one frame, in place.

        Extracted from :meth:`_write_stage_values` so the output frame and
        its covariate mirror receive rigorously the same writes while the
        provenance is marked exactly once.

        Args:
            frame: Frame written in place.
            var_name: Column being imputed.
            predictions: Values to write, indexed by the rows they belong to.
            blank_mask: Rows to blank first (scope restricted to the window).
        """
        # Vidage
        if blank_mask.any():
            frame.loc[blank_mask, var_name] = np.nan
        # Réécriture
        frame.loc[predictions.index, var_name] = predictions

    # Méthode auxiliaire de notation de la provenance des variables agrégées
    def _mark_aggregated_provenance(
        self,
        tracker: ImputationProvenanceTracker,
        X: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]]
    ) -> None:
        """Mark aggregated values in the provenance tracker.

        Only the cells that actually carry an aggregate are marked: an
        aggregation to a lower frequency leaves the sub-period rows empty,
        and marking them too would claim NaN cells were aggregated.
        Cells already marked ORIGINAL are excluded too: since
        aggregation runs at every stage, a column that already held a true
        observation at this index would otherwise have its ORIGINAL
        provenance overwritten by AGGREGATED, even though nothing was
        actually aggregated to obtain it.

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
            grouped = group_keys_by_entity_and_variable(aggregate_keys)
            # Parcours des entités et de leurs colonnes associées
            for entity, cols in grouped.items():
                # Extraction des observations liées à l'entité
                entity_mask = get_entity_mask(X, entity)
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
            columns = extract_column_names(aggregate_keys)
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
        spread out. So is an anchor date re-expressed at the stage
        frequency, whether or not the rescaling ran. Any other imputed cell
        keeps its MODEL_ON_* provenance.

        Args:
            tracker: Provenance tracker to write into.
            column: Column being imputed.
            index: Index of every cell that received a value.
            disaggregated_mask: Boolean mask, aligned on ``index``, of the
                cells to mark DISAGGREGATED — the union of the rescaled
                cells and of the anchor dates, built by
                :meth:`_write_stage_values`.
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

    # Méthode auxiliaire de calcul de la progression de fréquence de chaque variable
    def _compute_frequency_progression(self) -> Dict[str, List[str]]:
        """Compute the sequence of cascade-stage frequencies each variable went through.

        Read off ``imputation_plan_`` (built during PHASE 5 of ``_fit``)
        rather than recomputed independently: the plan already holds the
        exact stage sequence that was actually registered, one step per
        (stage, group) fit — including every panel group, not only the first
        one encountered (former bug, review §4.4).

        Returns:
            Dict mapping each variable name to the ordered list of
            ``freq_label`` stages (as used as the first element of
            ``model_fitting_order_`` entries) at which it was fitted,
            consecutive duplicates collapsed.

        Raises:
            AttributeError: If the imputer has not been fitted yet.
        """
        # Initialisation du dictionnaire résultat
        progression: Dict[str, List[str]] = {}

        # Parcours des étapes effectivement enregistrées, dans l'ordre de fit
        for step in self._require_plan():
            # Ajout du label s'il diffère du dernier enregistré pour cette variable
            stages = progression.setdefault(step.var_name, [])
            if not stages or stages[-1] != step.pred_freq_label:
                stages.append(step.pred_freq_label)

        return progression

    # Méthode auxiliaire de mise en correspondance des bornes (start, end) d'une fenêtre
    @staticmethod
    def _zip_window_bounds(
        start: Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]], None],
        end: Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]], None],
    ) -> Union[Tuple, Dict[tuple, Tuple]]:
        """Pair per-entity (or scalar) start/end bounds into (start, end) tuples.

        Used to populate ``training_window_`` from
        ``ImputationWindowCalculator.imputation_window_start_``/``_end_``,
        which already follow ``imputation_scope`` (review §1.1 follow-up) —
        no need to re-derive bounds from ``imputation_window_mask_`` anymore.

        Args:
            start: Window start(s), as returned by ``ImputationWindowCalculator``
                (scalar for time series, dict keyed by entity tuple for panel).
            end: Window end(s), same shape as ``start``.

        Returns:
            (start, end) tuple, or dict mapping entities to (start, end).
        """
        # Cas des données de panel : appariement par entité
        if isinstance(start, dict):
            return {entity: (start[entity], end[entity]) for entity in start}
        # Cas des séries temporelles
        return (start, end)

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
        PHASE 5: Iterate over frequency prediction list and fit models,
                 producing ``imputation_plan_`` — the ordered list of
                 :class:`ImputationStep` that is the whole fitted state
                 (review §5.3) and that ``_transform`` replays
        PHASE 6: Finalization

        Each cascade stage works on a frame rebuilt from the entry data by
        :meth:`_build_stage_frame` (never from the previous stage's frame),
        so that aggregation artefacts do not accumulate. Intermediate
        predictions feed ``imputed_store``, which enriches the covariates of
        the following stages; ``X_work`` is never modified.

        PHASE 5 falls back to interpolation whenever the training set for a
        variable holds fewer than 2 observations (``len(X_train) < 2`` /
        ``len(X_train_scaled) < 2``). Unlike ``min_cv_train_size`` (which
        only gates whether a variable is scored by cross-validation before
        ``train_on_partial_fit_order='cv'`` orders the cascade), this
        threshold is structural — no estimator can be fit on a single point —
        and stays hardcoded.

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_targets).
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
            # Concaténation de X et y si y spécifié
            # Vérification que X et y ont la même longueur
            if len(X) != len(y):
                raise ValueError("X and y should be of equal length")
            # Alignement de l'index de y sur celui de X : tolère la
            # conversion colonnes -> index (time_col/panel_cols) que "fit"
            # a déjà appliquée à X mais jamais à y
            y = self._align_target_index(X, y)
            y_col_name = self._resolve_target_column_name(y)
            X_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            y_col_name = None
            X_work = X.copy()
        self.target_column_ = y_col_name

        # Identification des entités
        if self.is_panel_ and isinstance(X.index, pd.MultiIndex):
            self.entities_ = get_unique_panel_entities(X)
        else:
            self.entities_ = None

        # Label de fréquence de l'index d'entrée : il identifie, à l'inversion,
        # le niveau à conserver dans une sortie multi-fréquences. Le
        # passage par "_stage_frequency_label" garantit le même format de label
        # que celui porté par le niveau "frequency" de la sortie
        try:
            index_freq = detect_index_frequency(X_work.index, return_format='base')
            self._source_index_frequency_label = self._stage_frequency_label(index_freq)
        except (ValueError, TypeError):
            # Index irrégulier ou trop court : le repli sur le niveau cible suffit
            self._source_index_frequency_label = None

        # Avertissement UNIQUE si aucun estimateur n'est fourni
        if self.estimator is None:
            warnings.warn(
                "No estimator was provided (estimator=None): every variable "
                "will fall back to linear interpolation.",
                UserWarning
            )

        # Normalisation de target_frequency
        normalized_target_frequency = self._validate_target_frequency_format(
            self.target_frequency
        )

        # Expansion de target_frequency en dict si panel + string
        if self.is_panel_ and isinstance(normalized_target_frequency, str) and self.entities_:
            self.effective_target_frequency_ = {
                entity: normalized_target_frequency for entity in self.entities_
            }
        elif isinstance(normalized_target_frequency, dict):
            self.effective_target_frequency_ = normalized_target_frequency.copy()
        else:
            self.effective_target_frequency_ = normalized_target_frequency

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
        self.variable_categories_ = self._classify_variables_at_frequency(
            self.effective_target_frequency_
        )
        self.imputation_order_ = self._determine_imputation_order()

        # =================================================================
        # PHASE 1 — Imputation window
        # =================================================================
        
        # Calcul de la fenêtre d'imputation sur les données du fit. Le
        # calculateur reste porté par l'instance : il définit le jeu
        # d'entraînement (_prepare_training_data, _determine_variable_order_cv).
        # La fenêtre de prédiction, elle, est recalculée sur les données imputées
        window_calc, window_error = self._fit_imputation_window(X_work)

        # Cas d'échec du calcul : le calculateur non entraîné est conservé, les
        # gardes "_is_fitted" des consommateurs le neutralisent d'elles-mêmes
        if window_calc is None:
            # Instanciation du calculateur de fenêtre
            self._imputation_window_calc = ImputationWindowCalculator(
                coverage_threshold=self.coverage_threshold,
                imputation_scope=self.imputation_scope,
                min_columns=2,
            )
            # Warning
            warnings.warn(
                f"Could not calculate imputation window: {window_error}. "
                f"Using all available data.",
                UserWarning
            )
            # On se restreint aux dates minimales et maximales
            if isinstance(X_work.index, pd.MultiIndex):
                time_idx = X_work.index.get_level_values(-1)
            else:
                time_idx = X_work.index
            self.imputation_window_ = (time_idx.min(), time_idx.max())
            self.training_window_ = self.imputation_window_
        else:
            self._imputation_window_calc = window_calc
            # Fenêtre STRICTE (coverage == 1.0), indépendante de imputation_scope
            self.imputation_window_ = self._zip_window_bounds(
                window_calc.imputation_strict_window_start_,
                window_calc.imputation_strict_window_end_,
            )
            # Fenêtre d'entraînement étendue selon imputation_scope
            self.training_window_ = self._zip_window_bounds(
                window_calc.imputation_window_start_,
                window_calc.imputation_window_end_,
            )

            # Avertissement global si aucune fenêtre stricte n'existe (fit "réussi" mais
            # bornes None) : sans cela, tous les entraînements échouent silencieusement
            # un à un et tout finit en interpolate_fallback (cf. PHASE 5)
            start = window_calc.imputation_strict_window_start_
            no_window = (
                start is None if not isinstance(start, dict)
                else all(v is None for v in start.values())
            )
            if no_window:
                # Warning
                warnings.warn(
                    "No strict imputation window found: no model can be trained; "
                    "all imputations will fall back to interpolation.",
                    UserWarning
                )

        # =================================================================
        # PHASE 2 — Additive transformer
        # =================================================================
        # Application du transformer rendant les données additives par période pour passer d'une fréquence à une autre
        if self.additive_transformer is not None:
            # Clone du transformer
            self.additive_transformer_ = clone(self.additive_transformer)
            # Entraînement et application de la transformation
            X_work = self.additive_transformer_.fit_transform(X_work)
            if isinstance(X_work, tuple):
                X_work = X_work[0]
        else:
            self.additive_transformer_ = None

        # =================================================================
        # PHASE 3 — Frequency prediction list
        # =================================================================
        # Construction de la liste des fréquences pour lesquelles il faut imputer les variables
        # Conservation de la liste des étapes : le replay de "_transform" la rejoue à l'identique
        self.freq_prediction_list_ = self._build_frequency_prediction_list()

        # =================================================================
        # PHASE 4 — Provenance initialization
        # =================================================================
        # Instanciation de la classe
        self._provenance_tracker = ImputationProvenanceTracker()
        # Initialisation : Scan du jeu de données initial 
        self._provenance_tracker.initialize(X_work, panel_cols=self.panel_cols)

        # =================================================================
        # PHASE 5 — Iterate over frequency prediction list
        # =================================================================
        # Une étape immuable par couple (fréquence de prédiction, groupe de
        # variables), portant tout ce dont le replay a besoin — modèle,
        # colonnes et statistiques d'entraînement, facteurs d'échelle, et les
        # métadonnées de désagrégation (fréquence source, entités concernées),
        # renseignées y compris sur les replis par interpolation.
        # "imputation_models_", "model_fitting_order_", "stage_groups_" et
        # "frequency_progression_" en sont des vues dérivées en lecture seule
        self.imputation_plan_: List[ImputationStep] = []
        
        # Imputations déjà réalisées, par nom de variable : elles alimentent les
        # covariables des étapes suivantes.
        # Cette table est indexée par NOM NU : le modèle d'un panel est GLOBAL, ses
        # prédictions couvrent toutes les entités de son groupe, et les lignes
        # d'un AUTRE groupe de la même variable proviennent donc de "existing".
        # Aucune valeur n'est écrasée, contrairement à un diviseur d'échelle
        imputed_store: Dict[str, pd.Series] = {}

        # Parcours des fréquences pour lesquelles il faut entraîner un modèle de prédiction
        for pred_freq in self.freq_prediction_list_:
            # 5a. Classification des variables relative à pred_freq
            var_classification = self._classify_variables_at_frequency(pred_freq)
            aggregate_keys = var_classification['aggregate']
            impute_keys = var_classification['impute']

            # Logging
            self._log(
                f"[fit] Stage {self._stage_frequency_label(pred_freq)}: "
                f"{len(impute_keys)} variable(s) to impute: {impute_keys}"
            )

            # 5b. Construction du frame de l'étape, reconstruit depuis les données d'origine
            X_stage = self._build_stage_frame(
                X_original=X_work, imputed_store=imputed_store, pred_freq=pred_freq,
                aggregate_keys=aggregate_keys,
            )
            # Marquage de la provenance sur le frame d'étape (index d'origine)
            self._mark_aggregated_provenance(
                self._provenance_tracker, X_stage, aggregate_keys
            )

            # 5c. Ordonnancement des variables à imputer
            if self.train_on_partial_fit_order == 'cv':
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
            # Le regroupement se fait sur (variable, fréquence détectée) et non la seule 
            # variable : le facteur d'échelle et l'agrégation d'entraînement dépendent 
            # de la fréquence, qui peut différer selon l'entité pour une même variable.
            vars_in_stage: "OrderedDict[Tuple[str, str], List[Union[str, Tuple]]]" = OrderedDict()
            for var_key in ordered_impute_keys:
                _, var_name = split_variable_key(var_key)
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
                # Construction de la clé du groupe (nom de la variable / nom de la variable x fréquence)
                group_key = (
                    var_name if len(freqs_per_var[var_name]) == 1
                    else (var_name, freq_norm)
                )
                # Clé d'origine représentative du groupe (une des clés (entité,
                # variable) qui le composent) : seuls le nom de colonne et la
                # fréquence détectée comptent pour préparer les données
                # d'entraînement, le modèle étant global au panel
                repr_var_key = var_keys[0]

                # Champs d'identité de l'étape, communs à tous les chemins
                # ci-dessous. La clé d'étape qui en découle, (label de
                # fréquence, groupe), fait qu'un même couple (étape, groupe)
                # n'est entraîné qu'une fois et qu'une variable imputée à deux
                # étapes obtient deux étapes distinctes. Les métadonnées de
                # groupe (fréquence source, entités) en font partie : le replay
                # en a besoin même quand l'étape finit en repli
                stage_fields = dict(
                    pred_freq_label=self._freq_label(pred_freq),
                    pred_freq=pred_freq,
                    var_key=group_key,
                    var_name=var_name,
                    source_frequency=freq_norm,
                    entities=to_entity_tuple(
                        [split_variable_key(k)[0] for k in var_keys]
                        if self.is_panel_ and isinstance(var_keys[0], tuple)
                        else None
                    ),
                )
                stage_scale = self._stage_scale_factor(repr_var_key, pred_freq)

                # Étape de repli, prête à l'emploi : les quatre chemins de repli
                # (pas d'estimateur, jeu d'entraînement trop court, jeu vidé par
                # le prétraitement, échec du fit) enregistrent la même étape
                fallback_step = ImputationStep(
                    **stage_fields,
                    model=INTERPOLATE_FALLBACK,
                    feature_cols=(),
                    scale_factor=stage_scale,
                    fit_scale_factor=stage_scale,
                    trained_on_imputed=False,
                )

                # Sans réentraînement, un seul fit par variable.
                # "replace" partage la référence de l'estimateur et conserve "fit_scale_factor"
                if not self.cascade_refitting:
                    # Extraction du modèle
                    base = self._model_for_var(group_key)
                    # Mise à jour de l'étape d'imputation
                    if base is not None:
                        self.imputation_plan_.append(
                            replace(base, **stage_fields, scale_factor=stage_scale)
                        )
                        continue

                # Extraction de l'estimateur
                estimator = self._get_estimator_for_variable(var_name)
                # Cas où l'estimateur n'est pas renseigné
                if estimator is None:
                    # Logging
                    self._log(
                        f"[fit] Fallback interpolate_fallback for '{var_name}': "
                        f"no estimator available"
                    )
                    # Warning : seulement pour un manque PROPRE À CETTE
                    # variable (dict sans '__default__' pour elle) ; le cas
                    # global self.estimator=None a déjà émis un avertissement
                    # unique en PHASE 0, le répéter par variable et
                    # par étape serait purement redondant
                    if self.estimator is not None:
                        warnings.warn(
                            f"No estimator available for variable '{var_name}', "
                            f"using linear interpolation as fallback"
                        )
                    # Fallback sur l'interpolation
                    self.imputation_plan_.append(fallback_step)
                    continue

                # Préparation des données d'entraînement
                X_train, y_train, scale_factor, feature_factors = self._prepare_training_data(
                    X_stage, X_work, repr_var_key, pred_freq
                )

                # Vérification qu'il y a au moins deux observations dans le jeu de données d'entraînement
                if len(X_train) < 2:
                    # Logging
                    self._log(
                        f"[fit] Fallback interpolate_fallback for '{var_name}': "
                        f"insufficient training data ({len(X_train)} observation(s))"
                    )
                    # Warning
                    warnings.warn(
                        f"Not enough training data for variable '{var_name}', "
                        f"using linear interpolation as fallback"
                    )
                    # Repli sur l'interpolation
                    self.imputation_plan_.append(fallback_step)
                    continue

                # 5e. Frequency scaling
                X_train_scaled, y_train_scaled = self._apply_frequency_scaling(
                    X_train, y_train, scale_factor, feature_factors
                )

                # Écartement des covariables entièrement vides.
                # La condition est évaluée sur les DEUX fenêtres : avec une
                # fenêtre d'entraînement plus large que celle de prédiction
                # (cf. "training_scope" du calculateur), une colonne peut être
                # observée à l'entraînement et intégralement NaN au predict, et
                # le modèle apprendrait un coefficient pour une feature
                # systématiquement absente. Élargir la fenêtre d'entraînement
                # doit ajouter des lignes, jamais des colonnes
                usable_cols = X_train_scaled.notna().any()
                if (hasattr(self, '_imputation_window_calc')
                        and self._imputation_window_calc._is_fitted):
                    prediction_window = (
                        self._imputation_window_calc.get_imputation_window_mask(
                            X_stage, kind='imputation'
                        ).reindex(X_stage.index).fillna(False).astype(bool)
                    )
                    if prediction_window.any():
                        seen_at_predict = (
                            X_stage.loc[prediction_window].notna().any()
                            .reindex(X_train_scaled.columns).fillna(False)
                        )
                        usable_cols &= seen_at_predict
                X_train_scaled = X_train_scaled.loc[:, usable_cols]

                # Écartement des lignes d'entraînement sans aucune covariable
                # observée.
                # C'est le pendant, côté entraînement, du groupe au motif vide
                # déjà écarté par "_determine_prediction_samples" côté prédiction
                observed_rows = X_train_scaled.notna().any(axis=1)
                X_train_scaled = X_train_scaled.loc[observed_rows]
                y_train_scaled = y_train_scaled.loc[observed_rows]

                # Suppression des lignes avec y NaN
                valid_mask = y_train_scaled.notna()
                X_train_scaled = X_train_scaled.loc[valid_mask]
                y_train_scaled = y_train_scaled.loc[valid_mask]

                # Repli si le jeu d'entraînement est trop petit ou sans covariable exploitable
                if len(X_train_scaled) < 2 or X_train_scaled.shape[1] == 0:
                    # Logging
                    self._log(
                        f"[fit] Fallback interpolate_fallback for '{var_name}': "
                        f"insufficient training data after preprocessing "
                        f"({len(X_train_scaled)} observation(s), "
                        f"{X_train_scaled.shape[1]} usable covariate(s))"
                    )
                    # Repli sur l'interpolation
                    self.imputation_plan_.append(fallback_step)
                    continue

                # 5f. Fit du modèle + construction de l'étape
                feature_cols = list(X_train_scaled.columns)
                try:
                    # Entraînement de l'estimateur
                    estimator.fit(X_train_scaled, y_train_scaled)
                    # Booléen indiquant si le modèle a été entraîné sur des valeurs imputées
                    # /!\ Je n'arrive pas à voir à quel moment les valeurs imputées ont été ajoutées au jeu de données avant l'entraînement de l'estimateur, peux tu me l'indiquer ?
                    trained_on_imputed = (
                        self.train_on_partial_coverage and bool(imputed_store)
                    )
                    # Paramètres de cette étape d'imputation pour la transformation
                    step = ImputationStep(
                        **stage_fields,
                        model=estimator,
                        feature_cols=tuple(feature_cols),
                        scale_factor=scale_factor,
                        # Facteur cuit dans le modèle : y_train en a été divisé.
                        # Il ne bouge plus, quand "scale_factor" suit l'étape
                        fit_scale_factor=scale_factor,
                        trained_on_imputed=trained_on_imputed,
                    )
                    # Décompte valeurs vraies vs imputées effectivement vues à
                    # l'entraînement, à partir de la provenance au moment du fit
                    # /!\ Ne serait-il pas logique de rajouter ProvenanceType.AGGREGATED ici car dans le cas où l'on a des données additives, il n'y a pas d'approximation dans l'agrégation ?
                    original_mask = self._provenance_tracker.get_mask(
                        [ProvenanceType.ORIGINAL, ProvenanceType.DISAGGREGATED],
                        column=var_name
                    ).reindex(y_train_scaled.index).fillna(False)
                    # Calcul du nombre de vraies valeurs dans le masque original
                    n_true = int(original_mask.sum())
                    # Logging
                    self._log(
                        f"[fit] Fit '{var_name}' at frequency {freq_norm} on "
                        f"{len(y_train_scaled)} observation(s) "
                        f"({n_true} true, {len(y_train_scaled) - n_true} imputed, "
                        f"trained_on_imputed={trained_on_imputed})"
                    )
                # Cas d'erreur
                except Exception as e:
                    # Logging
                    self._log(
                        f"[fit] Fallback interpolate_fallback for '{var_name}': "
                        f"fit failed with {e!r}"
                    )
                    # Warning
                    warnings.warn(
                        f"Failed to fit model for variable '{var_name}': {e}. "
                        f"Using linear interpolation as fallback"
                    )
                    # Repli sur l'interpolation
                    step = fallback_step

                # Enregistrement unique de l'étape : le chemin nominal et le
                # repli sur échec écrivent la même entrée de plan.
                self.imputation_plan_.append(step)

                # 5g. Cascade refitting : application des prédictions intermédiaires,
                # sur l'ensemble des entités
                if self.cascade_refitting:
                    # Cas où l'étape correspond effectivement à un modèle entrainé (et non à l'interpolation par défaut)
                    if not step.is_fallback:
                        # Extraction des méta-données de l'étape
                        stage_group = step.group_metadata()
                        # Extraction du contexte (variable prédite et fréquence du prédiction)
                        stage_context = (
                            f"'{var_name}' at stage "
                            f"{self._stage_frequency_label(pred_freq)}"
                        )
                        # Périmètre de désagrégation (toute la grille du groupe,
                        # dates-ancres comprises) et lignes prédictibles
                        # (restreintes à la fenêtre d'imputation)
                        scope_mask, predict_mask = self._prediction_masks(
                            X_stage, X_work, stage_group, var_name,
                            self._imputation_window_calc, context=stage_context
                        )
                        # Cas où le masque est non vide
                        if scope_mask.any():
                            try:
                                # Les prédictions sont déjà à l'échelle de la fréquence
                                # de prédiction : aucune remise à l'échelle inverse
                                preds = self._predict_stage_values(
                                    step, X_stage, predict_mask,
                                    context=stage_context
                                )
                                # Contrainte additive : la somme des sous-périodes de
                                # chaque période égale la valeur observée
                                preds, disagg_mask = self._apply_period_totals(
                                    preds, X_work, stage_group,
                                    context=stage_context
                                )
                                # Cascade intra-étape : les variables suivantes de
                                # l'étape voient les valeurs imputées. Le vidage du
                                # périmètre et le marquage de la provenance suivent
                                # la discipline commune de "_write_stage_values"
                                written = self._write_stage_values(
                                    X_stage, X_work, var_name, preds,
                                    scope_mask, predict_mask,
                                    self._provenance_tracker, disagg_mask,
                                    step.trained_on_imputed, context=stage_context
                                )
                                # Alimentation du magasin d'imputations pour les étapes
                                # suivantes, par nom de variable. Le receveur du
                                # "combine_first" l'emporte : ce sont donc les
                                # prédictions de l'étape courante qui gagnent, parce
                                # qu'elles sont à l'échelle la plus fine atteinte
                                # jusqu'ici. "existing" ne subsiste que sur les lignes
                                # que "preds" ne couvre pas, c'est-à-dire les entités
                                # d'un autre groupe de la même variable (fréquences
                                # hétérogènes selon l'entité) — l'intention d'origine
                                # est préservée sans figer le magasin sur la première
                                # étape
                                if written:
                                    existing_imputed = imputed_store.get(var_name)
                                    imputed_store[var_name] = (
                                        preds if existing_imputed is None
                                        else preds.combine_first(existing_imputed)
                                    )
                            # Un estimateur ne tolérant pas les NaN lève ici :
                            # les covariables partiellement observées lui
                            # parviennent telles quelles depuis le retrait de
                            # l'imputation à la moyenne
                            except Exception as e:
                                # Warning
                                warnings.warn(
                                    f"Intermediate imputation failed for "
                                    f"'{var_name}': {e}. The estimator must "
                                    f"tolerate NaN in X, or be wrapped in a "
                                    f"Pipeline handling them (e.g. SimpleImputer)."
                                )

        # =================================================================
        # PHASE 6 — Finalisation
        # =================================================================

        # Attribut distinct de celui écrit par "_transform" :
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
        (``freq_prediction_list_``) and, within each stage, the steps of
        ``imputation_plan_``. Each stage frame is rebuilt from the input
        data by :meth:`_build_stage_frame` — the very method used by
        :meth:`_fit` — so fit and transform work on identical stage frames
        for identical data. The input frame itself is never modified.

        The imputation window restricting the predictions is recomputed here,
        on the data being transformed, with the hyperparameters of the fit. 
        Reusing the window fitted on the training data emptied
        the imputed column entirely as soon as the dates fell outside its
        grid. Since the recomputation on the fit data gives back the fit
        window, ``fit_transform(X)`` and ``fit(X).transform(X)`` stay strictly
        identical.

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
        # Vérification que l'entrée est un DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        # Alignement de l'index de y sur celui de X 
        if y is not None:
            y = self._align_target_index(X, y)

        # Mémorisation de l'entrée du DERNIER transform : la transformation
        # inverse y lit l'index source, ses noms de niveaux et, sur demande
        # (restore_original_values), les valeurs d'origine exactes.
        # Cet instantané est donc écrasé à chaque appel de transform
        self._original_X_ = X.copy()
        self._original_y_ = y.copy() if y is not None else None

        # Extraction du nom de y (même règle qu'au fit)
        y_col_name = None
        if y is not None:
            y_col_name = self._resolve_target_column_name(y)
            data_work = pd.concat([X, y.to_frame(name=y_col_name)], axis=1)
        else:
            data_work = X.copy()


        if not isinstance(data_work.index, (pd.DatetimeIndex, pd.MultiIndex)):
            if self.time_col and self.time_col in data_work.columns:
                data_work = data_work.set_index(self.time_col)
            else:
                raise ValueError("Data must have a DatetimeIndex or MultiIndex")

        # Fenêtre d'imputation des données transformée, calculée une seule fois
        # et au même stade qu'au fit (avant le transformer additif), pour que le
        # recalcul sur les données du fit redonne exactement la fenêtre du fit.
        # Réutiliser celle du fit rendait tout entièrement NaN dès que les dates
        # sortaient de sa grille.
        transform_window_calc, window_error = self._fit_imputation_window(data_work)
        if transform_window_calc is None:
            # Logging
            self._log(
                f"[transform] No imputation window could be computed: "
                f"{window_error}. Predicting on the whole scope."
            )
            # Warning
            warnings.warn(
                f"Could not calculate the imputation window of the data being "
                f"transformed: {window_error}. Using all available data.",
                UserWarning
            )

        # 2. Application du transformer additif
        if self.additive_transformer_ is not None:
            # Transformation additive
            data_transformed = self.additive_transformer_.transform(data_work)
            # Extraction du premier élément du tuple si le transformer ne renvoie pas un DataFrame (le premier élément correspond à X)
            if isinstance(data_transformed, tuple):
                data_transformed = data_transformed[0]
        else:
            data_transformed = data_work.copy()

        # Init du tracker de provenance APRÈS le transformateur additif, comme la
        # PHASE 4 du fit qui suit sa PHASE 2. Deux raisons :
        # "ORIGINAL" marque les valeurs présentes à l'entrée de la CASCADE, or
        # celle-ci commence après la transformation additive — un transformateur
        # qui change le motif de NaN (différenciation, log de valeurs négatives)
        # produisait sinon deux masques ORIGINAL différents entre fit et
        # transform ; et la matrice partage ainsi l'index et les colonnes des
        # frames d'étape, tous dérivés de "data_transformed"
        transform_tracker = ImputationProvenanceTracker()
        transform_tracker.initialize(data_transformed, panel_cols=self.panel_cols)

        # 3. Réapplication les étapes du fit dans le même ordre, avec les mêmes frames d'étape
        # Frame d'entrée du replay : jamais modifié, il sert de base à chaque étape
        X_input = data_transformed
        # Imputations déjà réalisées, alimentant les covariables des étapes
        # suivantes. Indexée par NOM NU, sans collision entre entités : voir le
        # commentaire du bloc symétrique de "_fit"
        imputed_store: Dict[str, pd.Series] = {}
        # Frames d'étape APRÈS imputations, par label de fréquence réel (§2.5) :
        # alimenté après la boucle d'imputation de chaque étape, jamais avant
        stage_frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        # Instantané de la matrice de provenance par label de fréquence réel,
        # capturé au même moment que "stage_frames" : "transform_tracker" est
        # unique et continue d'être écrit aux étapes suivantes.
        provenance_frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()

        # Parcours des étapes
        for stage_idx, pred_freq in enumerate(self.freq_prediction_list_):
            # Classification des variables relative à pred_freq
            var_classification = self._classify_variables_at_frequency(pred_freq)
            # Extraction des variables à agréger
            aggregate_keys = var_classification['aggregate']

            # Construction du frame d'étape : même méthode qu'au fit, mêmes frames
            X_stage = self._build_stage_frame(X_input, imputed_store, pred_freq, aggregate_keys)
            # Indication de la provenance
            self._mark_aggregated_provenance(
                transform_tracker, X_stage, aggregate_keys
            )

            # Extraction des étapes du plan associées à cette fréquence
            registry_label = self._freq_label(pred_freq)
            stage_steps = [
                step for step in self.imputation_plan_
                if step.pred_freq_label == registry_label
            ]

            # Label de fréquence réel de l'étape
            freq_label = self._stage_frequency_label(pred_freq)
            # Booléen indiquant s'il s'agit de la dernière étape
            is_final_stage = stage_idx == len(self.freq_prediction_list_) - 1

            # Extraction du nom de la variable imputée dans l'étape
            stage_var_names = [step.var_name for step in stage_steps]
            # Logging
            self._log(
                f"[transform] Stage {freq_label}: replaying "
                f"{len(stage_var_names)} variable(s): {stage_var_names}"
            )

            # Frame des covariables de l'étape, miroir exact du "X_stage" du fit.
            # "X_stage" porte la sortie et reçoit donc toutes les
            # écritures ; le miroir n'accumule que ce que le bloc 5g du fit
            # accumule — les écritures des étapes NON de repli, et seulement sous
            # "cascade_refitting". Frame distinct systématiquement, y compris
            # quand les deux gardes passent partout : un repli non planifié (échec
            # de prédiction au replay) n'est pas connu avant d'être rencontré, et
            # il ne doit pas plus contaminer les covariables qu'un repli déclaré
            X_covariates = X_stage.copy()

            # Parcours des variables imputées à cette étape
            for step in stage_steps:
                # Le plan porte le nom de colonne et les métadonnées de groupe,
                # y compris pour les replis : plus aucun lookup croisé entre
                # registres. Ne pas réutiliser le nom "freq_label" ici : il
                # porte le label lisible de l'étape, utilisé après la boucle
                # pour "stage_frames"
                var_name = step.var_name
                stage_group = step.group_metadata()
                stage_context = f"'{var_name}' at stage {freq_label}"

                # Cas où l'étape est le cas de repli sur l'interpolation
                if step.is_fallback:
                    # Logging
                    self._log(
                        f"[transform] Fallback interpolate_fallback for "
                        f"'{var_name}' at stage {freq_label}: registered as "
                        f"interpolate_fallback at fit time"
                    )
                    # Le repli n'alimente ni imputed_store ni le frame des
                    # covariables : le fit n'en produit aucune valeur (le bloc 5g
                    # écarte les replis), les étapes suivantes ne doivent donc pas
                    # les voir. Seule la sortie les reçoit
                    if var_name in X_stage.columns:
                        # L'ancre sert de point d'appui à l'interpolation, puis le
                        # recalage proportionnel la ramène à l'échelle de la
                        # sous-période. Interpolation lue sur le miroir des
                        # covariables, seul frame dont l'état reproduit celui du fit
                        interpolated = X_covariates[var_name].interpolate(
                            method='linear', limit_direction='both'
                        )
                        # Même restriction que le chemin modèle : rien n'est
                        # produit hors de la fenêtre d'imputation
                        scope_mask, predict_mask = self._prediction_masks(
                            X_stage, X_input, stage_group, var_name,
                            transform_window_calc, context=stage_context
                        )
                        # Application de la contrainte d'agrégation
                        rescaled, disagg_mask = self._apply_period_totals(
                            interpolated.loc[predict_mask], X_input,
                            stage_group, context=stage_context
                        )
                        # Même discipline d'écriture que le chemin modèle
                        self._write_stage_values(
                            X_stage, X_input, var_name, rescaled,
                            scope_mask, predict_mask,
                            transform_tracker, disagg_mask, False,
                            context=stage_context
                        )
                    continue

                # Identification des valeurs manquantes
                if var_name not in X_stage.columns:
                    continue

                # Périmètre de désagrégation (toute la grille du groupe, dates-ancres
                # comprises, pour que la colonne ne mélange pas le total de la période
                # basse fréquence et des valeurs de sous-période) et lignes
                # prédictibles (restreintes à la fenêtre d'imputation)
                scope_mask, predict_mask = self._prediction_masks(
                    X_stage, X_input, stage_group, var_name,
                    transform_window_calc, context=stage_context
                )
                if not scope_mask.any():
                    continue

                try:
                    # Les prédictions sont déjà à l'échelle de la fréquence de
                    # prédiction : aucune remise à l'échelle inverse. Covariables
                    # lues sur le miroir, dont l'état reproduit celui du frame
                    # d'étape du fit au même instant de la cascade
                    predictions = self._predict_stage_values(
                        step, X_covariates, predict_mask, context=stage_context
                    )
                    # Contrainte additive : la somme des sous-périodes de chaque
                    # période égale la valeur observée
                    predictions, disagg_mask = self._apply_period_totals(
                        predictions, X_input, stage_group, context=stage_context
                    )
                    # Cascade intra-étape : le frame de l'étape porte le résultat.
                    # Le vidage du périmètre et le marquage de la provenance
                    # suivent la discipline commune de "_write_stage_values" :
                    # "predictions.index" couvre toutes les entités du groupe.
                    # Le miroir des covariables n'est mis à jour que sous la même
                    # garde que le bloc 5g du fit — étape non de repli et
                    # "cascade_refitting" — pour que les variables suivantes de
                    # l'étape voient exactement ce que le fit leur montrait
                    mirror = (
                        (X_covariates,)
                        if self.cascade_refitting and not step.is_fallback
                        else ()
                    )
                    written = self._write_stage_values(
                        X_stage, X_input, var_name, predictions,
                        scope_mask, predict_mask,
                        transform_tracker, disagg_mask, step.trained_on_imputed,
                        extra_frames=mirror, context=stage_context
                    )
                    # Alimentation des étapes suivantes, symétrique du bloc 5g du
                    # fit : sans réentraînement, les covariables des étapes
                    # suivantes restent celles vues à l'entraînement. Le receveur
                    # du "combine_first" l'emporte : ce sont donc les prédictions
                    # de l'étape courante qui gagnent, à l'échelle la plus fine
                    # atteinte jusqu'ici. "existing" ne subsiste que sur les lignes
                    # que "predictions" ne couvre pas, c'est-à-dire les entités
                    # d'un AUTRE groupe de la même variable (dans le cas où les fréquences
                    # sont hétérogènes selon l'entité)
                    if self.cascade_refitting and written:
                        existing_imputed = imputed_store.get(var_name)
                        imputed_store[var_name] = (
                            predictions if existing_imputed is None
                            else predictions.combine_first(existing_imputed)
                        )
                # Un estimateur ne tolérant pas les NaN lève ici : les
                # covariables partiellement observées lui parviennent telles
                # quelles depuis le retrait de l'imputation à la moyenne, et
                # le repli par interpolation prend alors le relais
                except Exception as e:
                    # Logging
                    self._log(
                        f"[transform] Fallback interpolate_fallback for "
                        f"'{var_name}' at stage {freq_label}: prediction "
                        f"failed with {e!r}"
                    )
                    # Warning
                    warnings.warn(
                        f"Prediction failed for variable '{var_name}': {e}. "
                        f"The estimator must tolerate NaN in X, or be wrapped "
                        f"in a Pipeline handling them (e.g. SimpleImputer). "
                        f"Using interpolation fallback."
                    )
                    # Repli sur l'interpolation, restreint aux lignes prédictibles
                    # et tracé comme toute autre valeur produite : écrire la
                    # colonne entière produirait des valeurs hors fenêtre, sans
                    # provenance, que l'inversion ne saurait pas retirer.
                    # Comme le repli déclaré, il n'alimente que la sortie : le
                    # bloc 5g du fit n'écrit rien quand la prédiction échoue
                    interpolated = X_covariates[var_name].interpolate(
                        method='linear', limit_direction='both'
                    )
                    filled = interpolated.loc[predict_mask].dropna()
                    self._write_stage_values(
                        X_stage, X_input, var_name, filled,
                        scope_mask, predict_mask,
                        transform_tracker, pd.Series(False, index=filled.index),
                        step.trained_on_imputed, context=stage_context
                    )

            # Stockage du frame d'étape APRÈS les imputations de l'étape :
            # un niveau par étape ayant produit au moins un modèle, plus la
            # dernière étape (fréquence cible) même sans imputation propre,
            # nécessaire à la sortie sans MultiIndex ci-dessous
            if stage_steps or is_final_stage:
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
        # "data_result", quand keep_lower_frequencies=True ; sinon
        # la seule matrice au niveau cible
        if self.keep_lower_frequencies and provenance_frames:
            self.imputation_provenance_ = self._build_multifreq_output(provenance_frames)
        else:
            self.imputation_provenance_ = transform_tracker.get_provenance_matrix()

        # Contrôle de cohérence provenance / données
        self._check_provenance_consistency(data_result, self.imputation_provenance_)

        # Résumé de provenance
        overall_stats = transform_tracker.compute_statistics()['overall']
        # Logging
        self._log(
            "Transform summary: " + ", ".join(
                f"{prov_type.value}={overall_stats[f'{prov_type.value}_pct']:.1f}%"
                for prov_type in ProvenanceType
            )
        )

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
    # Méthode auxiliaire de contrôle de cohérence entre provenance et données
    def _check_provenance_consistency(
        self,
        data_result: pd.DataFrame,
        provenance: pd.DataFrame,
    ) -> None:
        """Report cells declared ORIGINAL where the output holds no value.

        The invariant is one-way: a cell may hold a value without a provenance
        (it was never touched by the cascade), but a NaN cell may never be
        declared observed — ``inverse_transform`` reads ORIGINAL as "keep this
        value". A violation is reported rather than silently
        repaired: it signals that some path empties a cell without updating
        the matrix, which is worth seeing.

        The report names the frequency level when the output carries one:
        an intermediate level of a ``keep_lower_frequencies=True`` output
        stacks a stage frame aggregated to a lower frequency against a
        provenance snapshot still taken on the source index, so its dense
        columns read NaN there while their marks stay ORIGINAL. That known
        gap belongs to the multi-frequency output, not to the cascade, and is
        harmless for the inversion, which works on a single level.

        Args:
            data_result: Frame returned by the transform.
            provenance: Provenance matrix built alongside it, sharing its index
                structure.
        """
        # Colonnes communes aux deux frames
        common_cols = [c for c in data_result.columns if c in provenance.columns]
        if not common_cols:
            return

        # Cellules vides déclarées observées
        values = data_result[common_cols]
        marks = provenance[common_cols].reindex(index=values.index)
        inconsistent = values.isna() & (marks == ProvenanceType.ORIGINAL)
        if not inconsistent.to_numpy().any():
            return

        # Ventilation par niveau de fréquence quand la sortie en porte un
        if self._has_frequency_level(data_result):
            levels = data_result.index.get_level_values('frequency')
            detail = ", ".join(
                f"{level}: " + ", ".join(
                    f"{col}={int(count)}"
                    for col, count in inconsistent[levels == level].sum().items()
                    if count > 0
                )
                for level in levels.unique()
                if inconsistent[levels == level].to_numpy().any()
            )
        else:
            detail = ", ".join(
                f"{col}={int(count)}"
                for col, count in inconsistent.sum().items() if count > 0
            )

        total = int(inconsistent.to_numpy().sum())
        # Logging
        self._log(
            f"Provenance inconsistency: {total} empty cell(s) still marked "
            f"ORIGINAL ({detail})"
        )
        # Warning
        warnings.warn(
            f"Provenance inconsistency after transform: {total} cell(s) are NaN "
            f"but still marked ORIGINAL ({detail}). inverse_transform would "
            f"treat them as observed values on the affected level.",
            UserWarning
        )

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
            - Panel: ``(entity..., frequency, date)``
            Every level — including the target-frequency one — keeps its
            real frequency label; no level is named ``'target'``. The
            target level is identified via ``effective_target_frequency_``,
            not via a dedicated label.
        """
        # Étiquette de fréquence de l'étape par ligne, construite directement
        # pour ne pas risquer de collision avec une colonne portant déjà le même nom
        # Initialisation de la liste des jeux de données
        all_frames = []
        # Initialisation de la liste des fréquences
        freq_labels = []

        # Population des liste
        for freq_label, df in stage_frames.items():
            all_frames.append(df)
            freq_labels.append(np.full(len(df), freq_label, dtype=object))

        # Concaténation
        combined = pd.concat(all_frames, ignore_index=False)
        freq_values = np.concatenate(freq_labels)

        # Construction du MultiIndex résultat
        if self.is_panel_ and isinstance(combined.index, pd.MultiIndex):
            # Conservation de l'ensemble des niveaux de l'entité
            n_entity = combined.index.nlevels - 1
            entity_arrays = [
                combined.index.get_level_values(i) for i in range(n_entity)
            ]
            entity_names = [
                combined.index.names[i] if combined.index.names[i] is not None
                else ('entity' if n_entity == 1 else f'entity_{i}')
                for i in range(n_entity)
            ]
            new_index = pd.MultiIndex.from_arrays(
                [*entity_arrays, freq_values, combined.index.get_level_values(-1)],
                names=[*entity_names, 'frequency', 'date'],
            )
        else:
            new_index = pd.MultiIndex.from_arrays(
                [freq_values, combined.index],
                names=['frequency', 'date']
            )

        combined = combined.set_axis(new_index)

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

        The level of the source index (``_source_index_frequency_label``) is the
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
        if self._source_index_frequency_label is not None:
            if self._source_index_frequency_label in available:
                return self._source_index_frequency_label

        # Repli sur le niveau cible, toujours produit par la cascade
        target_label = self._stage_frequency_label(self.effective_target_frequency_)
        if target_label in available:
            return target_label

        # Dernier repli : le dernier niveau empilé, avec avertissement
        warnings.warn(
            f"Neither the source frequency level "
            f"({self._source_index_frequency_label}) nor the target one "
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
        addressed by name, never by position. The index names of the
        remaining levels are restored from the last transform input, in
        case the stacking renamed them (unnamed entity levels become
        ``'entity'``, or ``'entity_0'``, ``'entity_1'``, ... for a
        multi-level panel).

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
            y_col_name = self._resolve_target_column_name(y)
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
