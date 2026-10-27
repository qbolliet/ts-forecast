"""Replay of the frozen plan of ``HighFrequencyImputer`` on the data of a transform.

``replay_plan`` replays every step of the plan, stage by stage, through
:meth:`StageRunner.replay_step` — the very execution of the fit — and serves the pairs the fit
never observed by steps derived from the plan (``replay_steps``). The module also holds the checks
of the transform data against the fit (``check_transform_frequencies``,
``warn_rows_outside_window``, ``check_prediction_pass_replayable``) and the binding of the entities
born at transform time (``new_entities``, ``extend_stage_binding``).
"""
# Importation des modules
# Modules de base
import warnings
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# Manipulation de données
import pandas as pd

# Utilitaires du package
from ...panel.utils import get_entity_codes, normalize_entity_key, split_variable_key
from ...utils.frequency.utils import detect_frequency, is_higher_frequency, normalize_frequency
# Composants de l'imputation
from .._components.imputation_plan import ImputationPlan, ImputationStep, to_entity_tuple
from .._components.imputation_window import ImputationWindowCalculator
from .frequency_layout import (
    EntityKey,
    FrequencyLayout,
    PairKey,
    StageFrequency,
    misaligned_anchor_pairs,
    stage_frequency_label,
)
from .output import StageFrames, in_input_order, output_binding, stage_output
from .run_context import _RunContext
from .stage_runner import StageRunner, group_scale_factor
from .warnings import WarningAccumulator


# =============================================================================
# Contrôles des données du transform
# =============================================================================
# Fonction de contrôle des fréquences au transform
def check_transform_frequencies(
    layout: FrequencyLayout,
    X_work: pd.DataFrame,
    optional_columns: Iterable[str] = (),
) -> Tuple[Dict[PairKey, str], Tuple[PairKey, ...]]:
    """Compare the frequencies of the transform with those of the fit.

    The comparison is made **pair by pair** — ``(entity..., column)`` — and
    never column by column: on a panel the same column may legitimately
    carry a different frequency for each entity, and a
    per-column comparison would report a false divergence at every
    transform. A diverging pair warns once, listing every pair, and keeps the
    frequency of the fit; an extra column is ignored.

    Args:
        layout: Frequency layout of the fit.
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

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tsforecast.imputation._engine.frequency_layout import plan_frequencies
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
        >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0)
        >>> layout = plan_frequencies(
        ...     data, 'M', is_panel=False, entities=None,
        ...     impute_intermediate_frequencies=False, on_frequency_mismatch='error',
        ... )
        >>> check_transform_frequencies(layout, data)
        ({}, ())
        >>> check_transform_frequencies(layout, data[['m1']])
        Traceback (most recent call last):
        ...
        ValueError: transform is missing 1 column(s) seen at fit time: ['q1']. Every column of the fit must be present, even entirely empty: it is part of the frozen plan. Extra columns, on the other hand, are ignored.
    """
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
    off_calendar = misaligned_anchor_pairs(X_work, unknown, is_panel=layout.is_panel)
    return (
        {key: freq for key, freq in unknown.items() if key not in off_calendar},
        off_calendar,
    )


# Fonction de l'avertissement unique des lignes hors fenêtre
def warn_rows_outside_window(
    window_calc: ImputationWindowCalculator,
    X_work: pd.DataFrame,
    *,
    is_panel: bool,
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
        is_panel: Whether the data are a panel: the entities of the rows
            left out are then named.

    Examples:
        >>> import warnings
        >>> import numpy as np
        >>> import pandas as pd
        >>> from tsforecast.imputation._components.imputation_window import (
        ...     ImputationWindowCalculator,
        ... )
        >>> dates = pd.date_range('2021-01-31', periods=6, freq='ME')
        >>> data = pd.DataFrame(
        ...     {'m1': np.arange(6.0), 'm2': [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0]},
        ...     index=dates,
        ... )
        >>> calculator = ImputationWindowCalculator().fit(data)
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter('always')
        ...     warn_rows_outside_window(calculator, data, is_panel=False)
        >>> print(caught[0].message)
        2 row(s) of the data being transformed fall outside the imputation window recomputed on them. These rows keep their input values: nothing is ever blanked.
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
    entities = tuple(get_entity_codes(outside)[1]) if is_panel else ()
    warnings.warn(
        f"{len(outside)} row(s) of the data being transformed fall outside "
        f"the imputation window recomputed on them"
        + (f", for entities {entities}" if entities else "")
        + ". These rows keep their input values: nothing is ever blanked.",
        UserWarning,
    )


# Garde de la passe de prédiction : aucun modèle de X ne lit la cible
def check_prediction_pass_replayable(
    plan: Iterable[ImputationStep],
    target_column: Optional[str],
) -> None:
    """Refuse a prediction pass whose models would need the target.

    Under ``target_as_covariate=True`` a model of a column of ``X`` may
    read ``y``, which the prediction pass never has: replaying it would
    feed the model a covariate it was never fitted without.

    Args:
        plan: Steps of the fitted plan.
        target_column: Name of the target in the working frame of the fit.

    Raises:
        ValueError: If a non-fallback step of a column other than the
            target lists the target among its ``feature_cols``.

    Examples:
        >>> check_prediction_pass_replayable([], 'y') is None
        True
    """
    # Identification des cas où la variable dépendante y est utilisée en covariable de l'imputation d'autres variables
    dependent = sorted({
        step.var_name for step in plan
        if not step.is_fallback
        and step.var_name != target_column
        and target_column in step.feature_cols
    })
    # Message d'erreur
    if dependent:
        raise ValueError(
            f"The prediction pass cannot impute {dependent}: their models "
            f"use the target {target_column!r} as a covariate "
            f"(target_as_covariate=True), and transform never reads y once "
            f"the imputer is fitted with it. Refit with "
            f"target_as_covariate=False, or impute the training set with "
            f"fit_transform."
        )


# =============================================================================
# Entités nées au transform
# =============================================================================
# Fonction des entités apparues au transform
def new_entities(
    index: pd.Index,
    known_entities: Optional[Sequence[EntityKey]],
    *,
    is_panel: bool,
) -> Tuple[EntityKey, ...]:
    """List the entities of the transform the fit never saw.

    Args:
        index: Index of the working frame of the transform.
        known_entities: Entities of the fit, None on a time series.
        is_panel: Whether the data are a panel.

    Returns:
        Normalized entity keys, in order of appearance. Empty on a time
        series.

    Examples:
        >>> import pandas as pd
        >>> index = pd.MultiIndex.from_product(
        ...     [['FR', 'ES', 'DE'], pd.date_range('2021-01-31', periods=2, freq='ME')],
        ...     names=['country', 'date'],
        ... )
        >>> new_entities(index, [('FR',), ('DE',)], is_panel=True)
        (('ES',),)
        >>> new_entities(index, None, is_panel=False)
        ()
    """
    # Série temporelle : aucune entité, donc aucune nouveauté possible
    if not is_panel or not known_entities:
        return ()

    # Entités du fit, sous forme normalisée
    known = {normalize_entity_key(entity) for entity in known_entities}
    return tuple(
        entity
        for entity in get_entity_codes(index)[1]
        if entity not in known
    )


# Fonction d'extension de la liaison d'étape aux entités nouvelles
def extend_stage_binding(
    pending_warnings: WarningAccumulator,
    stage_freq: StageFrequency,
    born: Sequence[EntityKey],
    stage_label: str,
) -> StageFrequency:
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
        born: Entities the fit never saw.
        stage_label: Readable label of the stage, for the message.

    Returns:
        The binding extended to the new entities, or the original one.

    Examples:
        >>> from tsforecast.imputation._engine.warnings import WarningAccumulator
        >>> pending = WarningAccumulator('transform')
        >>> extend_stage_binding(pending, {('FR',): 'M', ('DE',): 'M'}, [('ES',)], 'M')
        {('FR',): 'M', ('DE',): 'M', ('ES',): 'M'}
        >>> extend_stage_binding(pending, 'M', [('ES',)], 'M')
        'M'
    """
    # Rien à étendre, ou étape scalaire (série temporelle)
    if not born or not isinstance(stage_freq, dict):
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
            f"{tuple(born)} cannot be bound to it and are skipped"
        )
        return stage_freq

    shared = unique.pop()
    return {**stage_freq, **{entity: shared for entity in born}}


# =============================================================================
# Étapes rejouées
# =============================================================================
# Fonction des étapes rejouées à une étape de fréquence
def replay_steps(
    runner: StageRunner,
    steps: Sequence[ImputationStep],
    born: Sequence[EntityKey],
    binding: StageFrequency,
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
        runner: Stage runner of the replay, on its execution context.
        steps: Plan steps of the stage, in plan order.
        born: Entities the fit never saw.
        binding: Stage binding, already extended (or not) to them.
        X_stage: Working frame, the grid of the anchored groups' scale.

    Returns:
        The steps to execute, in execution order.
    """
    # Paramètre éteint : le plan seul, les couples inconnus du fit intacts
    if not runner.params.impute_unobserved_entities:
        return list(steps)

    # Entités nouvelles effectivement liées à l'étape
    bound = [entity for entity in born if entity in binding]

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
                if not _route_transform_pair(
                    runner.run, entity, column, binding, anchored
                ):
                    remaining.append(entity)
            if remaining:
                composed.append(replace(step, entities=to_entity_tuple(remaining)))

        # Entités nouvelles : ancrées si observées, sans ancre sinon
        for entity in bound:
            if not _route_transform_pair(
                runner.run, entity, column, binding, anchored
            ):
                unanchored.append(entity)

        # Étapes dérivées, juste après celles de la colonne
        representative = column_steps[0]
        for frequency, entities in anchored.items():
            grid = runner.prediction_grid(X_stage.index, binding, entities)
            scale = group_scale_factor(
                runner.run.stage_scaler, column, frequency, binding, grid
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


# Fonction auxiliaire d'aiguillage d'un couple inconnu du fit
def _route_transform_pair(
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


# =============================================================================
# Rejeu du plan
# =============================================================================
# Fonction de rejeu du plan, étape de fréquence par étape de fréquence
def replay_plan(
    runner: StageRunner,
    plan: ImputationPlan,
    X_stage: pd.DataFrame,
    *,
    entities: Optional[Sequence[EntityKey]],
    skip_columns: FrozenSet[str] = frozenset(),
) -> Tuple[StageFrames, StageFrames]:
    """Replay every frozen step of the plan, in the order of the fit.

    Nothing is decided here: the classification, the progression, the
    variable order, the models, the ``feature_cols``, the materialization
    ways and the taints all come from the plan. Only the windows, the stage
    frames, the interpolated values, the predictions, the rescaling and the
    provenance are recomputed. Every step, derived ones included, is
    executed by :meth:`StageRunner.replay_step`, hence by the very
    :meth:`StageRunner.execute_step` of the fit. Once a stage has run, its
    output frame and provenance are assembled by
    :func:`~tsforecast.imputation._engine.output.stage_output`.

    Args:
        runner: Stage runner of the replay, on its execution context, the
            frequency layout of the fit and the parameters of the imputer.
        plan: Fitted plan.
        X_stage: Working frame after the additive transformer. Never
            modified: the imputations travel through the registers of
            :class:`CovariateMaterializer`, exactly as at fit time.
        entities: Entities of the fit, None on a time series.
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
    # Contexte d'exécution du rejeu et plan des fréquences du fit
    run = runner.run
    layout = runner.layout
    # Initialisation des jeux de données et des provenances
    frames: StageFrames = OrderedDict()
    provenances: StageFrames = OrderedDict()

    # Étapes du plan, indexées par label d'étape
    by_stage = plan.by_stage()
    # Entités nées au transform, et couples laissés hors du plan
    born = new_entities(X_stage.index, entities, is_panel=layout.is_panel)
    _note_pairs_left_untouched(run, born, runner.params.impute_unobserved_entities)

    # Report de fréquence : dernière étape traversée par chaque entité
    carried: Dict[EntityKey, str] = {}
    # Parcours des étapes de la progression, rejouée telle quelle
    for stage_freq in layout.progression:
        stage_label = stage_frequency_label(stage_freq)
        # Liaison de fréquence, étendue aux entités nouvelles
        binding = extend_stage_binding(run.pending_warnings, stage_freq, born, stage_label)

        # Étapes du plan de l'étape, complétées pour les couples que le fit
        # n'a jamais observés
        # (étapes des colonnes écartées retirées avant toute dérivation)
        stage_steps = tuple(
            step for step in by_stage.get(stage_label, ())
            if step.var_name not in skip_columns
        )
        steps = replay_steps(runner, stage_steps, born, binding, X_stage)
        # Logging
        if runner.log is not None:
            runner.log(f"[transform] stage {stage_label}: replaying {len(steps)} step(s)")
        # Exécution par la même méthode qu'au fit
        for step in steps:
            runner.replay_step(step, X_work=X_stage, stage_freq=binding)

        # Liaison complète, pour la sortie : une entité que l'étape ne
        # concerne pas garde les lignes de son dernier passage, et une
        # liaison partielle ferait échouer la lecture du masque, qui
        # retomberait alors sans bruit sur tout l'index
        complete = output_binding(binding, carried, X_stage.index, layout)

        # Grille de la frame : l'index du masque, jamais ses seules lignes
        # vraies — les lignes hors fenêtre restent dans la sortie avec
        # leurs valeurs d'entrée. Une ligne d'entrée hors de la grille (une
        # cible plus grossière que l'index) n'y figure pas : elle n'y
        # porterait rien, sa valeur étant agrégée sur la ligne de sa
        # période, et l'inversion rend l'index d'entrée. L'ordre est
        # celui de l'entrée (on impose qu'il soit trié)
        grid = in_input_order(
            runner.stage_mask(X_stage.index, complete, kind='imputation').index,
            X_stage.index,
        )
        # Frame et provenance de l'étape, assemblées après les écritures
        frames[stage_label], provenances[stage_label] = stage_output(
            run, layout, X_stage, grid, complete
        )

        # Report : chaque entité de l'étape y inscrit sa fréquence, lue par
        # les étapes suivantes qui ne la concernent plus
        if isinstance(binding, dict):
            carried.update(binding)

    return frames, provenances


# Fonction auxiliaire des couples que le rejeu laisse tels quels
def _note_pairs_left_untouched(
    run: _RunContext,
    born: Sequence[EntityKey],
    impute_unobserved_entities: bool,
) -> None:
    """Note, for the aggregated warning, the pairs the replay leaves as given.

    Entities born at transform time stay untouched under
    ``impute_unobserved_entities=False``; under ``True``, the pairs unknown to
    the fit observed off the calendar bounds of their frequency are left out
    exactly as the fit leaves its own out.

    Args:
        run: Execution context of the replay.
        born: Entities the fit never saw.
        impute_unobserved_entities: The parameter of the imputer.
    """
    # Entités nouvelles laissées de côté faute de paramètre : une note
    if born and not impute_unobserved_entities:
        run.pending_warnings.add_outside_plan_case(
            f"{len(born)} entity/entities absent from the fit "
            f"{born} are left untouched "
            f"(impute_unobserved_entities=False)"
        )
    # Couples inconnus du fit observés hors des bornes de leur période :
    # écartés comme au fit, et nommés
    if run.transform_misaligned and impute_unobserved_entities:
        run.pending_warnings.add_outside_plan_case(
            f"{len(run.transform_misaligned)} (entity, column) pair(s) "
            f"unknown to the fit are observed off the calendar bounds of "
            f"their frequency and are left untouched: "
            f"{sorted(run.transform_misaligned, key=repr)}"
        )
