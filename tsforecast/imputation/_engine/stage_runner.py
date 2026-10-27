"""Stage execution of ``HighFrequencyImputer``: fit of one frequency stage, and execution of one
frozen plan step shared by the fit and the replay.

``StageRunner`` runs phase 5 of the fit — stage frame, order of the variables, one fit per
(stage, variable) shared by its source-frequency groups — and executes every plan step, the
replayed ones included, through its single :meth:`StageRunner.execute_step`, on the grids and
window masks of ``grids``. The module also holds the per-step helpers (covariate selection,
estimator resolution, scale factor of a group, pairs of a step).
"""
# Importation des modules
# Modules de base
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Sklearn
from sklearn.base import BaseEstimator, clone

# Utilitaires du package
from ...utils.frequency.utils import get_frequency_order
from .._components.aggregation_constraint import resolve_aggregation_constraint
from .._components.covariate_materializer import CovariateMaterializer
from .._components.imputation_plan import (
    INTERPOLATE_FALLBACK,
    ImputationStep,
    MaterializationWay,
    to_entity_tuple,
)
from .._components.provenance import (
    CELL_ORIGIN_DTYPE,
    CellOrigin,
    ProvenanceType,
    Taint,
    max_cell_origin,
    max_origin,
    origin_to_taint,
)
from .._components.stage_scaler import StageScaler
from .._components.training_set_builder import TrainingSet
from .._components.variable_orderer import VariableSpec
from .frequency_layout import (
    ColumnFrequencies,
    EntityKey,
    FrequencyLayout,
    LogCallback,
    StageFrequency,
    stage_frequency_label,
    stage_frequency_of,
)
from .grids import prediction_grid, restrict_to_entities, stage_mask, unrestricted_grid
from .row_scaling import RowScaler, TrainingRows
from .run_context import _RunContext

# Groupes imputables d'une colonne : (colonne, fréquence source) -> entités
ColumnGroups = Dict[Tuple[str, Optional[str]], Tuple[EntityKey, ...]]

# Origines de cellule admissibles dans "y_train", par modalité de l'axe 2.
# Table plutôt qu'une suite de tests : la modalité n'est
# jamais évaluée comme un booléen ('covariates_only' est truthy)
ELIGIBLE_ORIGINS: Dict[Any, Tuple[CellOrigin, ...]] = {
    False: ('observed',),
    'covariates_only': ('observed',),
    True: ('observed', 'interpolated', 'model'),
}


# =============================================================================
# Auxiliaires d'une étape
# =============================================================================
# Fonction de sélection des colonnes de features
def select_feature_columns(
    materializer: CovariateMaterializer,
    candidates: Sequence[str],
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> Tuple[str, ...]:
    """Select the covariates kept for one stage, per the window rules.

    **Widening ``training_scope`` adds rows, never columns**.
    Feature selection stays governed by availability at prediction
    time, independently of the training window. Two adjustments
    follow from it:

    - a column is kept only if it is non-empty on **both** windows — a
      column observed only on rows the training scope added would train a
      coefficient the prediction grid can never feed;
    - per-entity eligibility is delegated to
      ``CovariateMaterializer.eligible_columns``, which carries
      ``covariate_eligibility``.

    The symmetric row rule — training rows carrying no observed covariate
    at all are dropped — is applied by :func:`drop_empty_training_rows`.

    Args:
        materializer: Materializer of the pass, carrier of
            ``covariate_eligibility``.
        candidates: Candidate covariate columns, in output order.
        training_frame: Frame restricted to the training grid.
        prediction_frame: Frame restricted to the prediction grid.

    Returns:
        Kept columns, in the order of ``candidates``.

    Examples:
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
        >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0) * 3 + 10
        >>> pred_frame = data.assign(q1=np.nan)
        >>> select_feature_columns(CovariateMaterializer(), ['m1', 'q1'], data, pred_frame)
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


# Fonction d'élimination des lignes d'entraînement sans covariable
def drop_empty_training_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop the training rows carrying no observed covariate at all.

    Symmetric to :func:`select_feature_columns`: a row whose every covariate
    is NaN teaches the estimator nothing and, under an estimator that does
    not tolerate NaN, sends the whole fit to the fallback.

    Args:
        X_train: Materialized training features.
        y_train: Raw training target, sharing ``X_train``'s index.

    Returns:
        The pair restricted to the rows carrying at least one observed
        covariate. An empty ``X_train`` (no covariate at all) is returned
        untouched: there is no row to discriminate on.

    Examples:
        >>> X = pd.DataFrame({'m1': [1.0, np.nan, 3.0], 'q1': [np.nan, np.nan, 2.0]})
        >>> X_kept, y_kept = drop_empty_training_rows(X, pd.Series([10.0, 20.0, 30.0]))
        >>> y_kept.tolist()
        [10.0, 30.0]
    """
    # Aucune covariable : rien à filtrer, la garde de taille du fit joue
    if X_train.shape[1] == 0:
        return X_train, y_train

    # Conservation des lignes portant au moins une covariable observée
    kept_rows = X_train.notna().any(axis=1)
    return X_train.loc[kept_rows], y_train.loc[kept_rows]


# Fonction de résolution de l'estimateur d'une colonne
def estimator_for(estimator: Any, column: str) -> Optional[BaseEstimator]:
    """Resolve the estimator of one column, under the dict form too.

    Args:
        estimator: The ``estimator`` parameter of the imputer: an estimator,
            a dict keyed by column (``'__default__'`` optional), or None.
        column: Column being imputed.

    Returns:
        A fresh clone of the estimator covering that column, or None when
        none does — the interpolation fallback then applies.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> estimator_for({'a1': LinearRegression()}, 'a1')
        LinearRegression()
        >>> estimator_for({'a1': LinearRegression()}, 'm1') is None
        True
    """
    # Dictionnaire indexé par colonne : la colonne nommée, puis le défaut explicite
    if isinstance(estimator, dict):
        estimator = estimator.get(column, estimator.get('__default__'))
    # Cas où l'estimateur n'est pas spécifié
    if estimator is None:
        return None
    # Copie indépendante de l'estimateur sklearn
    return clone(estimator)


# Fonction du facteur d'échelle d'un groupe ancré
def group_scale_factor(
    stage_scaler: StageScaler,
    column: str,
    source_frequency: str,
    stage_freq: StageFrequency,
    grid: pd.Index,
) -> Any:
    """Compute the scale factor of one anchored group.

    The target having been scaled row by row, the predictions already come
    out at the pace of the stage: ``scale_factor`` and ``fit_scale_factor``
    coincide and the carry is 1.0. Read by the plan steps of the fit and by
    the steps the replay derives for the pairs the fit never observed.

    Args:
        stage_scaler: Stage scaler of the pass.
        column: Column imputed by the step.
        source_frequency: Detected frequency of the column for the group.
        stage_freq: Frequency of the stage.
        grid: Prediction grid of the group, read by the ``'calendar'``
            scale mode.

    Returns:
        The factor, a float or a per-row ``Series``. A conversion the
        scaler refuses raises: a silent ``1.0`` would mis-scale every
        prediction of the group.

    Examples:
        >>> scaler = StageScaler(scale_features='constant', column_frequencies={'a1': 'Y'})
        >>> group_scale_factor(scaler, 'a1', 'Y', 'M', pd.DatetimeIndex([]))
        12.0
    """
    return stage_scaler.fit_scale_factor(
        column,
        source_freq=source_frequency,
        pred_freq=stage_freq,
        index=grid,
    )


# Fonction des clés de couple d'une étape
def step_pairs(step: ImputationStep) -> List[tuple]:
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
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> from sklearn.linear_model import LinearRegression
        >>> from tsforecast.imputation import HighFrequencyImputer
        >>> idx = pd.MultiIndex.from_product([['FR', 'IT'], dates], names=['entity', 'date'])
        >>> panel = pd.DataFrame({'m1': np.arange(48.0), 'v': np.nan}, index=idx)
        >>> panel.loc[('FR', dates[11::12]), 'v'] = [120.0, 132.0]
        >>> imputer = HighFrequencyImputer(
        ...     target_frequency='M',
        ...     estimator=LinearRegression(),
        ...     impute_unobserved_entities=True,
        ... ).fit(panel)
        >>> step_pairs(imputer.imputation_plan_.steps[-1])
        [('IT', 'v')]
    """
    # Série temporelle : aucune entité, donc aucun couple à nommer
    return [
        (*entity, step.var_name) for entity in (step.entities or ())
    ]


# =============================================================================
# Paramètres, contexte d'ajustement, résultat d'une étape
# =============================================================================
# Paramètres de l'imputeur lus par l'exécution des étapes
@dataclass(frozen=True, kw_only=True)
class StageParams:
    """Parameters of ``HighFrequencyImputer`` the stage execution reads.

    The values are those of the imputer, untouched — the dict forms
    included — so the runner resolves them exactly where the imputer did.
    Every field is required and named: a parameter left out fails at
    construction, instead of silently taking a value the imputer may not
    hold.

    Attributes:
        covariate_strategy: Materialization strategy of the covariates; the
            order of the variables is computed under ``'model'`` only.
        fit_predict_order: ``'frequency'`` or ``'cv'``.
        estimator: The ``estimator`` parameter: an estimator, a dict keyed
            by column, or None.
        impute_intermediate_frequencies: Modality of axis 2, the key of
            :data:`ELIGIBLE_ORIGINS`.
        aggregation_constraint: Aggregation constraint, global or per
            column; under None the coincident cells are all kept.
        impute_unobserved_entities: Whether the pairs never observed are
            imputed from the other entities.
        target_column: Name of the target column in the working frame, None
            without target.
        target_as_covariate: Whether the target may be a covariate of the
            other columns.

    Examples:
        >>> params = StageParams(
        ...     covariate_strategy='model', fit_predict_order='frequency', estimator=None,
        ...     impute_intermediate_frequencies=False, aggregation_constraint='sum',
        ...     impute_unobserved_entities=False, target_column=None, target_as_covariate=False,
        ... )
        >>> params.covariate_strategy, params.aggregation_constraint
        ('model', 'sum')
        >>> StageParams(covariate_strategy='model')
        Traceback (most recent call last):
        ...
        TypeError: StageParams.__init__() missing 7 required keyword-only arguments: ...
    """

    covariate_strategy: str
    fit_predict_order: str
    estimator: Any
    impute_intermediate_frequencies: Any
    aggregation_constraint: Any
    impute_unobserved_entities: bool
    target_column: Optional[str]
    target_as_covariate: bool


# Contexte d'ajustement d'une variable à une étape
@dataclass(frozen=True)
class _VariableFit:
    """Everything one (stage, variable) needs before an estimator is fitted.

    Composed by :meth:`StageRunner.prepare_variable` and consumed
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
        feature_cols: Covariates retained by :func:`select_feature_columns`,
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


# Résultat de l'ajustement d'une étape de fréquence
@dataclass(frozen=True)
class StageOutcome:
    """What the fit of one frequency stage produced, for the imputer to publish.

    Attributes:
        stage_label: Readable label of the stage, key of the plan and of the
            order attributes.
        steps: Plan steps of the stage, in execution order, already
            executed and frozen.
        order: Processing order of the columns, None when no order was
            computed (outside ``covariate_strategy='model'``, or a single
            imputable column).
        cv_scores: Cross-validation scores that decided the order, empty
            under ``fit_predict_order='frequency'``.

    Examples:
        >>> StageOutcome(stage_label='M').steps, StageOutcome(stage_label='M').order
        ((), None)
    """

    stage_label: str
    steps: Tuple[ImputationStep, ...] = ()
    order: Optional[List[str]] = None
    cv_scores: Dict[Any, float] = field(default_factory=dict)


# =============================================================================
# Exécuteur d'étapes
# =============================================================================
# Exécution des étapes d'une passe : fit d'une étape de fréquence, rejeu d'une étape figée
class StageRunner:
    """Run the stages of one pass of ``HighFrequencyImputer``, fit or replay.

    **One single execution of a plan step**, :meth:`execute_step`: the fit
    builds each step then executes it (:meth:`fit_stage`), the replay
    executes the frozen steps through the very same method
    (:meth:`replay_step`). The writes — values, rescaling, provenance, the
    three registers — are therefore common by construction.

    Built once per pass on the execution context of that pass; it reads the
    frequency layout and the parameters of the fit, never the imputer. The
    fit of a stage needs a fit context — a replay composes no training set.

    Args:
        run: Execution context of the pass.
        layout: Frequency layout of the fit.
        params: Parameters of the imputer the stage execution reads.
        log: Logging callback, None outside verbose mode.

    Examples:
        >>> from functools import partial
        >>> from sklearn.linear_model import LinearRegression
        >>> from tsforecast.imputation import HighFrequencyImputer
        >>> from tsforecast.imputation._components.variable_orderer import VariableOrderer
        >>> from tsforecast.imputation._engine.grids import training_mask_at
        >>> from tsforecast.imputation._engine.preparation import prepare_work_frame
        >>> from tsforecast.imputation._engine.run_context import FittedComponents
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
        >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0) * 3 + 10
        >>> imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())
        >>> prepared = prepare_work_frame(imputer, data, fitting=True)
        >>> layout = prepared.fit.layout
        >>> components = FittedComponents.build(
        ...     covariate_strategy='interpolate', covariate_fallback='interpolate',
        ...     covariate_eligibility='any_entity', interpolation_method='linear',
        ...     interpolation_anchor=None, aggregation_constraint='sum',
        ...     scale_features='constant', column_frequencies=layout.frequencies_by_column(),
        ... )
        >>> run = _RunContext.for_fit(
        ...     components,
        ...     window_calculator=prepared.window_calculator,
        ...     provenance_tracker=prepared.provenance_tracker,
        ...     variable_orderer=VariableOrderer(),
        ...     training_mask=partial(
        ...         training_mask_at, prepared.window_calculator, is_panel=False
        ...     ),
        ... )
        >>> params = StageParams(
        ...     covariate_strategy='interpolate', fit_predict_order='frequency',
        ...     estimator=LinearRegression(), impute_intermediate_frequencies=False,
        ...     aggregation_constraint='sum', impute_unobserved_entities=False,
        ...     target_column=None, target_as_covariate=False,
        ... )
        >>> runner = StageRunner(run, layout, params)
        >>> stage = runner.fit_stage(prepared.X_work, 'M')
        >>> [(step.var_name, step.source_frequency, step.n_written) for step in stage.steps]
        [('q1', 'Q', 24)]
    """

    # Initialisation de l'exécuteur
    def __init__(
        self,
        run: _RunContext,
        layout: FrequencyLayout,
        params: StageParams,
        *,
        log: Optional[LogCallback] = None,
    ) -> None:
        self.run = run
        self.layout = layout
        self.params = params
        self.log = log
        # Mise à l'échelle par ligne des jeux mutualisés : sans état, une
        # instance sert toute la passe
        self.row_scaler = RowScaler(run.stage_scaler, layout)

    # Méthode auxiliaire de journalisation
    def _log(self, message: str) -> None:
        """Forward a progress message to the logging callback, if any.

        Args:
            message: Message to log.
        """
        if self.log is not None:
            self.log(message)

    # -------------------------------------------------------------------------
    # Grilles de la passe
    # -------------------------------------------------------------------------
    # Méthode de lecture d'un masque de fenêtre de la passe
    def stage_mask(
        self,
        index: pd.Index,
        stage_freq: StageFrequency,
        kind: Literal['imputation', 'training'],
    ) -> pd.Series:
        """Read one window mask of the pass at the frequency of a stage.

        :func:`stage_mask` on the window calculator of the pass.

        Args:
            index: Index of the working frame, the fallback grid.
            stage_freq: Frequency of the stage, scalar or per entity.
            kind: Window read: ``'imputation'`` or ``'training'``.

        Returns:
            Boolean Series at the stage frequency.
        """
        return stage_mask(
            self.run.window_calculator, index, stage_freq, kind,
            pass_name=self.run.pending_warnings.pass_name, log=self.log,
        )

    # Méthode de la grille de prédiction d'un groupe de la passe
    def prediction_grid(
        self,
        index: pd.Index,
        stage_freq: StageFrequency,
        entities: Optional[Sequence[EntityKey]],
    ) -> pd.Index:
        """Build the prediction grid of a group on the window of the pass.

        :func:`prediction_grid` on the window calculator of the pass.

        Args:
            index: Index of the working frame.
            stage_freq: Frequency of the stage.
            entities: Entities of the group, None for a time series.

        Returns:
            Index of the rows the step writes on.
        """
        return prediction_grid(
            self.run.window_calculator, index, stage_freq, entities,
            pass_name=self.run.pending_warnings.pass_name, log=self.log,
        )

    # -------------------------------------------------------------------------
    # Fit d'une étape de fréquence
    # -------------------------------------------------------------------------
    # Méthode d'exécution d'une étape de fréquence du fit
    def fit_stage(
        self,
        X_work: pd.DataFrame,
        stage_freq: StageFrequency,
    ) -> StageOutcome:
        """Run one frequency stage of the progression, at fit time.

        Three moves, in this exact order: the stage frame, the imputable
        variables and their order, then one pass per variable — one fit
        per (stage, variable), shared by the plan steps of its
        source-frequency groups, each step executed right after it is built.

        Args:
            X_work: Working frame, after the additive transformer.
            stage_freq: Frequency of the stage.

        Returns:
            The :class:`StageOutcome` of the stage: its executed plan steps,
            the order of its columns and the scores that decided it.

        Raises:
            RuntimeError: If the context is that of a replay, which holds no
                training-set builder.
        """
        # Contexte de rejeu : aucun jeu d'entraînement n'y est composable
        if self.run.training_set_builder is None:
            raise RuntimeError(
                "fit_stage needs the execution context of a fit: a replay "
                "context holds no training-set builder."
            )

        # Étiquette lisible de l'étape, clé du plan et des journaux
        stage_label = stage_frequency_label(stage_freq)
        # Détection des fréquences
        freqs_by_column = self.layout.frequencies_by_column(self.run.transform_frequencies)

        # Groupes imputables : un par (colonne, fréquence source)
        groups = self.layout.imputable_groups(
            stage_freq, impute_unobserved_entities=self.params.impute_unobserved_entities
        )
        if not groups:
            self._log(f"[fit] stage {stage_label}: no imputable variable")
            return StageOutcome(stage_label=stage_label)

        # 5a. Frame d'étape : données d'origine, agrégations exactes et miroir
        mask = self.stage_mask(X_work.index, stage_freq, kind='imputation')
        stage_frame = self.run.materializer.stage_frame(
            grid_index=mask.index,
            stage_freq=stage_freq,
            detected_frequencies=freqs_by_column,
            source_data=X_work,
        )

        # 5b. Variables imputables à l'étape, regroupées puis ordonnées
        by_column: Dict[str, ColumnGroups] = {}
        for group_key, entities in groups.items():
            by_column.setdefault(group_key[0], {})[group_key] = entities
        ordered, order, cv_scores = self._order_columns(
            by_column,
            X_work,
            stage_freq=stage_freq,
            stage_frame=stage_frame,
            freqs_by_column=freqs_by_column,
        )

        # 5c. Une variable à la fois, un seul ajustement par (étape, variable)
        steps: List[ImputationStep] = []
        for column in ordered:
            steps.extend(self._fit_variable(
                column=column,
                groups=by_column[column],
                stage_freq=stage_freq,
                stage_label=stage_label,
                X_work=X_work,
                stage_frame=stage_frame,
                freqs_by_column=freqs_by_column,
            ))
        return StageOutcome(
            stage_label=stage_label, steps=tuple(steps), order=order, cv_scores=cv_scores
        )

    # Méthode auxiliaire d'ordonnancement des colonnes d'une étape
    def _order_columns(
        self,
        by_column: Dict[str, ColumnGroups],
        X_work: pd.DataFrame,
        *,
        stage_freq: StageFrequency,
        stage_frame: pd.DataFrame,
        freqs_by_column: ColumnFrequencies,
    ) -> Tuple[List[str], Optional[List[str]], Dict[Any, float]]:
        """Order the imputable columns of one stage.

        The order is computed by :class:`VariableOrderer` only under
        ``covariate_strategy='model'``: it is the only modality where ranks 2
        and 3 of the precedence are reachable, hence the only one where the
        order changes a value. Otherwise the input column
        order is used and no order is reported.

        Under the ``'cv'`` order, each column is scored on the very
        ``(X_train, y_train)`` pair phase 5c will fit it on, composed here by
        :meth:`prepare_variable`: same covariate selection, same
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
            stage_freq: Frequency of the stage.
            stage_frame: Stage frame of 5a.
            freqs_by_column: Detected frequencies, keyed by column.

        Returns:
            Tuple ``(ordered, order, cv_scores)``: the columns to impute in
            processing order; the order to publish, None when none was
            computed; the cross-validation scores that decided it, empty
            when there are none.
        """
        # Ordre d'entrée des colonnes : le défaut, sans effet sur les valeurs
        columns = [column for column in X_work.columns if column in by_column]
        if self.params.covariate_strategy != 'model' or len(columns) <= 1:
            return columns, None, {}

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
                else self.layout.entity_target_frequency(entities[0] if entities else None)
            )
            specs[column] = VariableSpec(
                name=column,
                frequency=lowest,
                entities=entities if self.layout.is_panel else None,
            )

        # Ajustement paresseux de l'ordonnanceur, sous l'ordre 'frequency'
        # seulement — la phase 0 l'ajuste déjà sous 'cv' : l'avertissement
        # croisé qu'il émet ne concerne que la validation croisée et n'aurait
        # ici aucun sens
        variable_orderer = self.run.variable_orderer
        if not hasattr(variable_orderer, 'cv_'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                variable_orderer.fit()

        # Jeux de scoring de l'ordre 'cv' : ceux-là mêmes que la 5c ajustera.
        # Rien n'est préparé sous l'ordre 'frequency', qui n'en lit aucun
        training_sets: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]] = None
        if self.params.fit_predict_order == 'cv':
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
                prepared = self.prepare_variable(
                    column=column,
                    entities=group_entities if self.layout.is_panel else None,
                    stage_freq=stage_freq,
                    X_work=X_work,
                    stage_frame=stage_frame,
                    freqs_by_column=freqs_by_column,
                )
                training_sets[column] = (prepared.X_train, prepared.y_train)

        # Les lignes scorées sont celles du jeu mutualisé, déjà restreintes à la fenêtre 'training'
        ordered = list(variable_orderer.order(
            specs,
            estimator=self.params.estimator,
            log=self.log,
            training_sets=training_sets,
        ))
        # Scores CV qui ont décidé du rang, pour le tracking. Sous l'ordre
        # 'frequency' l'ordonnanceur laisse "scores_" vide
        stage_scores = dict(getattr(variable_orderer, 'scores_', {}) or {})
        return ordered, list(ordered), stage_scores

    # Méthode de composition du contexte d'ajustement d'une variable
    def prepare_variable(
        self,
        *,
        column: str,
        entities: Optional[Sequence[EntityKey]],
        stage_freq: StageFrequency,
        X_work: pd.DataFrame,
        stage_frame: pd.DataFrame,
        freqs_by_column: ColumnFrequencies,
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
        builder = self.run.training_set_builder
        # Matérialiseur des résultats des étapes précédentes
        materializer = self.run.materializer
        # Origines acceptées pour la target et les covariables
        eligible = ELIGIBLE_ORIGINS[self.params.impute_intermediate_frequencies]

        # Départage des cellules coïncidentes, lu par colonne : le
        # paramètre admet une forme dictionnaire, la valeur n'est donc jamais
        # globale. Le constructeur du jeu ne reçoit que ce booléen : il ignore
        # tout de "aggregation_constraint", comme il ignore tout de l'axe 2
        keep_coincident = resolve_aggregation_constraint(
            self.params.aggregation_constraint, column
        ) is None

        # Fréquences remises au constructeur : la colonne cible par entité,
        # couples indétectables explicités, pour qu'aucun bloc n'emprunte la
        # fréquence des autres entités
        training_freqs = self.layout.training_frequencies(column, freqs_by_column)

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
        pred_grid = self.prediction_grid(X_work.index, stage_freq, entities)

        # Vues des deux fenêtres, chacune à sa fréquence : la grille
        # d'entraînement au pas de chaque couche de lignes, la grille de
        # prédiction au pas de l'étape
        train_view = self._training_view(probe, stage_freq, freqs_by_column, X_work)
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
        excluded_target = (
            None if self.params.target_as_covariate else self.params.target_column
        )
        candidates = [
            name for name in X_work.columns
            if name != column and name != excluded_target and name in freqs_by_column
        ]
        feature_cols = select_feature_columns(
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

        # Mise à l'échelle : diviseurs de la cible et des features par ligne,
        # lus couche par couche
        X_train, y_train = training.X, training.y
        if len(training) > 0:
            # Décomposition unique de la grille d'entraînement : la grille
            # réelle que lisent le matérialiseur et le scaler, l'entité de
            # chaque ligne
            rows = TrainingRows.of(training)
            y_train = self.run.stage_scaler.apply(
                y_train,
                self.row_scaler.target_divisors(rows, column=column, stage_freq=stage_freq),
            )
            if feature_cols:
                X_train = self.run.stage_scaler.apply(
                    X_train,
                    self.row_scaler.feature_divisors(
                        rows,
                        feature_cols=feature_cols,
                        freqs_by_column=freqs_by_column,
                        ways=ways,
                        stage_freq=stage_freq,
                    ),
                )

        # Lignes sans aucune covariable observée, puis cible manquante
        X_train, y_train = drop_empty_training_rows(X_train, y_train)
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
        probe: TrainingSet,
        stage_freq: StageFrequency,
        freqs_by_column: ColumnFrequencies,
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
            probe: Covariate-free training set of the variable, carrying the
                rows, their production frequency and the blocks.
            stage_freq: Frequency of the stage, used when there is no row.
            freqs_by_column: Detected frequencies, keyed by column.
            X_work: Working frame.

        Returns:
            The stage frames of every layer, stacked; rows are never written
            back, only read for their non-emptiness.
        """
        # Matérialiseur de la passe, seul producteur des trames d'étape
        materializer = self.run.materializer
        # Grille réelle de la sonde, entité de chaque ligne et couches : le
        # niveau de fréquence n'est jamais lu par le matérialiseur
        rows = TrainingRows.of(probe)

        # Une trame par couche, sur sa sous-grille dédoublonnée : deux cellules
        # coïncidentes partagent par définition une date
        frames: List[pd.DataFrame] = []
        if rows.blocks:
            for layer, in_layer in rows.layers:
                sub_grid = rows.real_index[in_layer]
                sub_grid = sub_grid[~sub_grid.duplicated()]
                frames.append(materializer.stage_frame(
                    grid_index=sub_grid,
                    stage_freq=self.row_scaler.layer_binding(layer),
                    detected_frequencies=freqs_by_column,
                    source_data=X_work,
                ))

        # Aucune ligne : trame vide à la forme de l'étape
        if not frames:
            return materializer.stage_frame(
                grid_index=rows.real_index[:0],
                stage_freq=stage_freq,
                detected_frequencies=freqs_by_column,
                source_data=X_work,
            )
        return pd.concat(frames)

    # Méthode d'ajustement d'une variable à une étape
    def _fit_variable(
        self,
        *,
        column: str,
        groups: ColumnGroups,
        stage_freq: StageFrequency,
        stage_label: str,
        X_work: pd.DataFrame,
        stage_frame: pd.DataFrame,
        freqs_by_column: ColumnFrequencies,
    ) -> List[ImputationStep]:
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

        Returns:
            The plan steps of the column, one per group, executed then
            frozen — degraded into a fallback when their prediction failed.
        """
        # Union des entités des groupes : la grille de prédiction, et à
        # travers elle la sélection des covariables, en dépendent
        group_entities = sorted(
            {entity for entities in groups.values() for entity in entities}, key=repr
        )
        entities = group_entities if self.layout.is_panel else None

        # Contexte d'ajustement. Il est recomposé ici même lorsque la 5b l'a
        # déjà composé pour ordonner : depuis, les variables de rang inférieur
        # ont écrit dans le miroir, et c'est précisément ce que la cascade
        # cherche à exploiter
        prepared = self.prepare_variable(
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
            _X_pred, _ways, pred_origins = self.run.materializer.materialize(
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
            column, X_train, y_train, feature_cols, stage_label,
            announce=writes_something,
        )
        # Logging
        self._log(
            f"[fit] stage {stage_label}, {column!r}: {len(y_train)} pooled "
            f"training rows from blocks {blocks}, {len(groups)} group(s)"
        )

        # Une étape de plan par groupe, partageant le modèle ajusté ci-dessus
        multi_frequency = len(groups) > 1
        steps: List[ImputationStep] = []
        # Parcours des groupes
        for group_key, group in groups.items():
            # Extraction des entités du groupe
            group_entities_or_none = group if self.layout.is_panel else None
            # Construction de l'étape
            step = self._build_step(
                column=column,
                source_frequency=group_key[1],
                entities=group_entities_or_none,
                grid=restrict_to_entities(pred_grid, group_entities_or_none),
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
            degraded, n_written = self.execute_step(
                step, X_work=X_work, stage_freq=stage_freq
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
            steps.append(step)
        return steps

    # Méthode auxiliaire d'ajustement de l'estimateur d'une variable
    def _fit_estimator(
        self,
        column: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: Tuple[str, ...],
        stage_label: str,
        announce: bool = True,
    ) -> Tuple[Any, bool]:
        """Fit the estimator of one variable, or fall back on interpolation.

        Args:
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
            message is accumulated in the pending warnings of the pass, never
            emitted here.
        """
        # Repli d'interpolation, annoncé par l'avertissement agrégé de la phase
        def _fallback(reason: str) -> Tuple[Any, bool]:
            if announce:
                self.run.pending_warnings.add_degraded_step(
                    f"{column!r} at stage {stage_label}: {reason}, "
                    f"interpolation fallback"
                )
            return INTERPOLATE_FALLBACK, True

        # Extraction de l'estimateur
        estimator = estimator_for(self.params.estimator, column)
        # Aucun estimateur pour cette colonne : le repli est la règle, et
        # l'avertissement global de la phase 0 a déjà été émis
        if estimator is None:
            if self.params.estimator is not None:
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
        *,
        column: str,
        source_frequency: Optional[str],
        entities: Optional[Sequence[EntityKey]],
        grid: pd.Index,
        stage_freq: StageFrequency,
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
            scale = group_scale_factor(
                self.run.stage_scaler, column, source_frequency, stage_freq, grid
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
            interpolation_method=self.run.materializer.resolve_method(column),
            interpolation_anchor=self.run.materializer.resolve_anchor(column),
            training_blocks=training_blocks,
            unanchored=unanchored,
            n_training_rows=n_training_rows,
        )

    # -------------------------------------------------------------------------
    # Exécution d'une étape figée : fit et rejeu
    # -------------------------------------------------------------------------
    # Méthode de rejeu d'une étape figée
    def replay_step(
        self,
        step: ImputationStep,
        *,
        X_work: pd.DataFrame,
        stage_freq: StageFrequency,
    ) -> None:
        """Replay one frozen plan step, through the execution of the fit.

        :meth:`execute_step` itself: the replay discards what the fit
        freezes on the step (a degradation, the count of written cells),
        since the plan is the state of the fit and is never rewritten.

        Args:
            step: Frozen step to replay — a plan step, or one the replay
                derived from it for the pairs the fit never observed.
            X_work: Working frame of the replay.
            stage_freq: Frequency binding of the stage.
        """
        # Exécution par la même méthode qu'au fit, résultat abandonné
        self.execute_step(step, X_work=X_work, stage_freq=stage_freq)

    # Méthode unique d'exécution d'une étape du plan
    def execute_step(
        self,
        step: ImputationStep,
        *,
        X_work: pd.DataFrame,
        stage_freq: StageFrequency,
    ) -> Tuple[bool, int]:
        """Execute one frozen plan step.

        The step is already frozen: this method decides nothing and only
        writes — values, rescaling, provenance and the three stores. ``fit``
        builds the step then calls it; ``transform`` replays the plan
        through this very method (:meth:`replay_step`), which is what makes
        the two paths identical by construction.

        An unanchored step departs from the common path on
        three points, and on three only: it is never rescaled to a period
        total, its cells carry ``MODEL_UNANCHORED``, and it has no
        interpolation fallback — there is nothing to interpolate for an
        entity that never observes the column, so a failure leaves its cells
        NaN and unmarked and is reported by the aggregated warning of the
        end of the pass.

        Args:
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
            rescaling, and is 0 whenever nothing was produced. The replay
            discards both.
        """
        # Contexte d'exécution de la passe
        run = self.run
        # Extraction du matérialiseur des étapes précédentes
        materializer = run.materializer
        # Extraction du nom de la colonne
        column = step.var_name
        # Détection des fréquences de la colonne (éventuellement pour différentes entités)
        freqs_by_column = self.layout.frequencies_by_column(run.transform_frequencies)

        # Grille du groupe : la fenêtre d'imputation de ses entités, ancres comprises
        grid = self.prediction_grid(X_work.index, stage_freq, step.entities)

        # Sans ancre, la fenêtre stricte d'une entité est vide par
        # construction : elle est l'intervalle où toutes ses colonnes sont
        # couvertes, et la colonne imputée n'y en couvre aucune. La subordonner
        # à la couverture de la colonne même que l'étape produit serait
        # circulaire ; l'entité est donc laissée sans restriction, exactement
        # comme "stage_mask" traite déjà une entité que le calculateur omet
        if step.unanchored and len(grid) == 0:
            grid = unrestricted_grid(
                run.window_calculator, X_work.index, stage_freq, step.entities,
                pass_name=run.pending_warnings.pass_name, log=self.log,
            )

        if len(grid) == 0:
            return False, 0

        # Production des valeurs : modèle, ou repli d'interpolation
        values: Optional[pd.Series] = None
        if not step.is_fallback:
            values = self._predict_step(step, grid, X_work, freqs_by_column, stage_freq)
        degraded = values is None and not step.is_fallback

        # Absence d'ancre : l'interpolation ne peut pas être le repli, faute
        # d'observation à interpoler. Les couples restent NaN et non marqués,
        # et l'avertissement agrégé de la fin de la passe les nomme
        if step.unanchored and values is None:
            run.pending_warnings.add_unimputed_pairs(step_pairs(step))
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
                restrict_to_entities(X_work.index, step.entities)
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
            run.unanchored_written.update(step_pairs(step))

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

    # Méthode auxiliaire de prédiction d'une étape à modèle
    def _predict_step(
        self,
        step: ImputationStep,
        grid: pd.Index,
        X_work: pd.DataFrame,
        freqs_by_column: ColumnFrequencies,
        stage_freq: StageFrequency,
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
            prediction failed or produced nothing but NaN — the caller then
            falls back on interpolation, or, for an unanchored step, reports
            the pair. A failure is announced by the aggregated warning of the
            degraded steps, an unanchored step excepted: having no fallback,
            it is named by the warning of the unimputed pairs only.
        """
        # Contexte d'exécution de la passe
        run = self.run
        # Covariables produites en mode rejeu des voies de l'étape : c'est le
        # seul chemin autorisé pour X_pred.
        # "record=False" : les registres ne portent que ce qui a été imputé,
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
