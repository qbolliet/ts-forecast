"""Execution state of one ``HighFrequencyImputer`` pass, and the components every pass shares.

``FittedComponents`` holds what the fit configures once and every later pass reads
unchanged: the aggregation constraint, the stage scaler and the configuration of the
covariate materializer. ``_RunContext`` holds the state of ONE pass — fresh materializer
and registers, window calculator, provenance tracker, pending warnings, pairs imputed
without anchor — built by ``_RunContext.for_fit`` or ``_RunContext.for_transform`` and
never stored on the imputer.
"""
# Importation des modules
# Modules de base
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Set, Tuple, Union

# Composants de l'imputation
from .._components.aggregation_constraint import AggregationConstraint
from .._components.covariate_materializer import CovariateMaterializer
from .._components.imputation_window import ImputationWindowCalculator
from .._components.provenance import ImputationProvenanceTracker
from .._components.stage_scaler import StageScaler
from .._components.training_set_builder import (
    LogCallback,
    TrainingMask,
    TrainingSetBuilder,
)
from .._components.variable_orderer import VariableOrderer
from .warnings import WarningAccumulator

# Clé d'un couple : nom de colonne (série) ou (entité..., colonne) (panel)
PairKey = Union[str, Tuple[Any, ...]]

# Contexte nommé par la contrainte d'agrégation dans ses messages
_CONSTRAINT_CONTEXT = 'HighFrequencyImputer'


# Composants configurés au fit, partagés par toutes les passes
@dataclass(frozen=True)
class FittedComponents:
    """Components the fit configures once, then every pass reads unchanged.

    The aggregation constraint and the stage scaler carry no state of their
    own outside a call — the constraint's deferred-warning census lives only
    inside its block — so one instance serves the fit and every replay. The
    materializer, which holds the registers of a pass, is never shared: only
    its configuration is kept, and :meth:`new_materializer` builds a fresh
    one per pass.

    Attributes:
        aggregation_constraint: Constraint rescaling the step predictions,
            and injected into each materializer for its interpolated
            covariates — the same object for both, so one deferred block
            covers both.
        stage_scaler: Scaler of the stages, configured with the frequencies
            detected at fit time.
        materializer_options: Constructor arguments of the materializer,
            the constraint applier excepted.

    Examples:
        >>> components = FittedComponents.build(
        ...     covariate_strategy='interpolate',
        ...     covariate_fallback='interpolate',
        ...     covariate_eligibility='any_entity',
        ...     interpolation_method='linear',
        ...     interpolation_anchor=None,
        ...     aggregation_constraint='sum',
        ...     scale_features='constant',
        ...     column_frequencies={'m1': 'M', 'q1': 'Q'},
        ... )
        >>> materializer = components.new_materializer()
        >>> materializer.aggregation_constraint_applier is components.aggregation_constraint
        True
        >>> materializer is components.new_materializer()
        False
    """

    aggregation_constraint: AggregationConstraint
    stage_scaler: StageScaler
    materializer_options: Mapping[str, Any]

    # Construction des composants à partir des paramètres de l'imputeur
    @classmethod
    def build(
        cls,
        *,
        covariate_strategy: str,
        covariate_fallback: str,
        covariate_eligibility: str,
        interpolation_method: Any,
        interpolation_anchor: Any,
        aggregation_constraint: Any,
        scale_features: Any,
        column_frequencies: Mapping[str, Any],
    ) -> 'FittedComponents':
        """Configure the shared components from the parameters of the imputer.

        Args:
            covariate_strategy: Materialization strategy of the covariates.
            covariate_fallback: Fallback of the ``'model'`` strategy.
            covariate_eligibility: Per-entity eligibility rule of a covariate.
            interpolation_method: Interpolation method, global or per feature.
            interpolation_anchor: Anchor position, global or per feature.
            aggregation_constraint: Aggregation constraint, global or per
                feature.
            scale_features: Scaling mode, global, per feature or per entity.
            column_frequencies: Frequencies detected at fit time, keyed by
                column (a scalar, or one frequency per entity).

        Returns:
            The components, ready to serve every pass. The materializer
            options are validated by each :meth:`new_materializer` call.

        Examples:
            >>> components = FittedComponents.build(
            ...     covariate_strategy='model',
            ...     covariate_fallback='interpolate',
            ...     covariate_eligibility='any_entity',
            ...     interpolation_method='linear',
            ...     interpolation_anchor=None,
            ...     aggregation_constraint=None,
            ...     scale_features='calendar',
            ...     column_frequencies={'q1': 'Q'},
            ... )
            >>> components.stage_scaler.scale_features
            'calendar'
            >>> components.materializer_options['covariate_strategy']
            'model'
        """
        # Contrainte d'agrégation, portée par un objet unique : elle recale les
        # prédictions des étapes ET, injectée dans le matérialiseur, les
        # covariables interpolées
        constraint = AggregationConstraint(
            aggregation_constraint=aggregation_constraint,
            context=_CONSTRAINT_CONTEXT,
        )
        # Metteur à l'échelle des étapes : sans état, il rend un diviseur par
        # appel — jamais un scalaire figé pour tout un jeu
        scaler = StageScaler(
            scale_features=scale_features,
            column_frequencies=column_frequencies,
        )
        return cls(
            aggregation_constraint=constraint,
            stage_scaler=scaler,
            materializer_options={
                'covariate_strategy': covariate_strategy,
                'covariate_fallback': covariate_fallback,
                'covariate_eligibility': covariate_eligibility,
                'interpolation_method': interpolation_method,
                'interpolation_anchor': interpolation_anchor,
                'aggregation_constraint': aggregation_constraint,
            },
        )

    # Matérialiseur neuf d'une passe
    def new_materializer(self) -> CovariateMaterializer:
        """Build the materializer of one pass, its three registers empty.

        Returns:
            A fresh :class:`CovariateMaterializer` configured with
            :attr:`materializer_options`, whose constraint applier is the
            shared :attr:`aggregation_constraint`.

        Raises:
            ValueError: If an option is not admissible (the materializer
                validates its configuration).

        Examples:
            >>> components = FittedComponents.build(
            ...     covariate_strategy='interpolate',
            ...     covariate_fallback='interpolate',
            ...     covariate_eligibility='any_entity',
            ...     interpolation_method='linear',
            ...     interpolation_anchor=None,
            ...     aggregation_constraint='sum',
            ...     scale_features='constant',
            ...     column_frequencies={},
            ... )
            >>> components.new_materializer().imputed_store
            {}
        """
        return CovariateMaterializer(
            **self.materializer_options,
            aggregation_constraint_applier=self.aggregation_constraint,
        )


# État d'exécution d'une passe : fit ou rejeu
@dataclass
class _RunContext:
    """Execution state of one pass of ``HighFrequencyImputer``, a fit or a replay.

    Built at the start of the pass, read and written by the stage execution,
    then dropped: nothing of it survives the pass, except what the imputer
    publishes from it (provenance matrix, ``unanchored_pairs_``, warnings).
    A replay therefore starts from a context of its own, built from the
    fitted state alone, and can neither read nor alter the state of the fit.

    Attributes:
        aggregation_constraint: Shared constraint (see
            :class:`FittedComponents`).
        stage_scaler: Shared stage scaler.
        materializer: Materializer of the pass, sole producer of the
            covariates and holder of the three registers of the pass.
        window_calculator: Window calculator fitted on the data of the pass
            (unfitted when the window could not be computed: its readers then
            restrict nothing).
        provenance_tracker: Provenance tracker of the pass, initialized on
            the frame the pass imputes.
        pending_warnings: Aggregated warnings of the pass, emitted at its end.
        unanchored_written: Pairs actually imputed without anchor.
        transform_frequencies: Frequencies detected on the data of a replay
            for the pairs the fit never observed; empty at fit time.
        transform_misaligned: Those of the pairs the fit never observed that
            a replay observes off the calendar bounds of their frequency.
        training_set_builder: Builder of the mutualized training sets, fit
            only: a replay never composes a training set.
        variable_orderer: Orderer of the variables of a stage, fit only: a
            replay follows the order frozen in the plan.

    Examples:
        >>> components = FittedComponents.build(
        ...     covariate_strategy='interpolate',
        ...     covariate_fallback='interpolate',
        ...     covariate_eligibility='any_entity',
        ...     interpolation_method='linear',
        ...     interpolation_anchor=None,
        ...     aggregation_constraint='sum',
        ...     scale_features='constant',
        ...     column_frequencies={'m1': 'M'},
        ... )
        >>> run = _RunContext.for_transform(
        ...     components,
        ...     window_calculator=ImputationWindowCalculator(),
        ...     provenance_tracker=ImputationProvenanceTracker(),
        ...     transform_frequencies={},
        ...     transform_misaligned=(),
        ... )
        >>> run.stage_scaler is components.stage_scaler, run.training_set_builder
        (True, None)
    """

    # Composants partagés, configurés au fit : lus, jamais réécrits par une passe
    aggregation_constraint: AggregationConstraint
    stage_scaler: StageScaler
    # État propre à la passe
    materializer: CovariateMaterializer
    window_calculator: ImputationWindowCalculator
    provenance_tracker: ImputationProvenanceTracker
    pending_warnings: WarningAccumulator
    unanchored_written: Set[Tuple[Any, ...]] = field(default_factory=set)
    transform_frequencies: Dict[PairKey, str] = field(default_factory=dict)
    transform_misaligned: FrozenSet[PairKey] = frozenset()
    # Propre au fit : le rejeu ne compose aucun jeu d'entraînement et n'ordonne rien
    training_set_builder: Optional[TrainingSetBuilder] = None
    variable_orderer: Optional[VariableOrderer] = None

    # Contexte d'un fit
    @classmethod
    def for_fit(
        cls,
        components: FittedComponents,
        *,
        window_calculator: ImputationWindowCalculator,
        provenance_tracker: ImputationProvenanceTracker,
        variable_orderer: VariableOrderer,
        training_mask: TrainingMask,
        log: Optional[LogCallback] = None,
    ) -> '_RunContext':
        """Build the context of a fit.

        Args:
            components: Components the fit has just configured.
            window_calculator: Window calculator fitted on the data of the fit.
            provenance_tracker: Provenance tracker, initialized on the frame
                the fit imputes.
            variable_orderer: Orderer of the variables of a stage.
            training_mask: Provider of the training-window mask, handed to
                the training-set builder: it never receives the calculator,
                so it cannot read another window.
            log: Logging callback of the builder, or None.

        Returns:
            A context with a fresh materializer, a training-set builder wired
            on that very materializer, and empty accumulators.

        Examples:
            >>> components = FittedComponents.build(
            ...     covariate_strategy='interpolate',
            ...     covariate_fallback='interpolate',
            ...     covariate_eligibility='any_entity',
            ...     interpolation_method='linear',
            ...     interpolation_anchor=None,
            ...     aggregation_constraint='sum',
            ...     scale_features='constant',
            ...     column_frequencies={'m1': 'M'},
            ... )
            >>> run = _RunContext.for_fit(
            ...     components,
            ...     window_calculator=ImputationWindowCalculator(),
            ...     provenance_tracker=ImputationProvenanceTracker(),
            ...     variable_orderer=VariableOrderer(),
            ...     training_mask=lambda frequencies: None,
            ... )
            >>> run.training_set_builder.materializer is run.materializer
            True
            >>> run.pending_warnings.pass_name
            'fit'
        """
        # Matérialiseur de la passe, partagé avec le constructeur du jeu
        # d'entraînement : les covariables d'entraînement lisent le miroir de
        # la passe, et lui seul
        materializer = components.new_materializer()
        return cls(
            aggregation_constraint=components.aggregation_constraint,
            stage_scaler=components.stage_scaler,
            materializer=materializer,
            window_calculator=window_calculator,
            provenance_tracker=provenance_tracker,
            pending_warnings=WarningAccumulator('fit'),
            training_set_builder=TrainingSetBuilder(
                materializer=materializer,
                training_mask=training_mask,
                log=log,
            ),
            variable_orderer=variable_orderer,
        )

    # Contexte d'un rejeu
    @classmethod
    def for_transform(
        cls,
        components: FittedComponents,
        *,
        window_calculator: ImputationWindowCalculator,
        provenance_tracker: ImputationProvenanceTracker,
        transform_frequencies: Mapping[PairKey, str],
        transform_misaligned: Iterable[PairKey],
    ) -> '_RunContext':
        """Build the context of a replay from the fitted components alone.

        Args:
            components: Components configured by the fit, shared as they are.
            window_calculator: Window calculator recomputed on the data being
                transformed.
            provenance_tracker: Provenance tracker of the replay, initialized
                on the frame produced by the additive transformer.
            transform_frequencies: Frequencies detected for the pairs the fit
                never observed, the only ones no fit value can be replayed
                for.
            transform_misaligned: Those of the pairs the fit never observed
                that the replay observes off the calendar bounds of their
                frequency.

        Returns:
            A context with a fresh materializer — registers empty, whatever
            the fit produced —, empty accumulators, copies of the two
            collections, and neither training-set builder nor orderer.

        Examples:
            >>> components = FittedComponents.build(
            ...     covariate_strategy='interpolate',
            ...     covariate_fallback='interpolate',
            ...     covariate_eligibility='any_entity',
            ...     interpolation_method='linear',
            ...     interpolation_anchor=None,
            ...     aggregation_constraint='sum',
            ...     scale_features='constant',
            ...     column_frequencies={'m1': 'M'},
            ... )
            >>> run = _RunContext.for_transform(
            ...     components,
            ...     window_calculator=ImputationWindowCalculator(),
            ...     provenance_tracker=ImputationProvenanceTracker(),
            ...     transform_frequencies={('ES', 'q1'): 'Q'},
            ...     transform_misaligned=[('ES', 'a1')],
            ... )
            >>> run.transform_frequencies, run.transform_misaligned
            ({('ES', 'q1'): 'Q'}, frozenset({('ES', 'a1')}))
            >>> run.variable_orderer is None, run.pending_warnings.pass_name
            (True, 'transform')
        """
        return cls(
            aggregation_constraint=components.aggregation_constraint,
            stage_scaler=components.stage_scaler,
            materializer=components.new_materializer(),
            window_calculator=window_calculator,
            provenance_tracker=provenance_tracker,
            pending_warnings=WarningAccumulator('transform'),
            transform_frequencies=dict(transform_frequencies),
            transform_misaligned=frozenset(transform_misaligned),
        )
