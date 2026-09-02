"""Explicit imputation plan produced by ``HighFrequencyImputer2``.

This module holds the fitted-state description, of the HighFrequencyImputer. 
It follows the "one plan, one executor" principle : the plan is the complete
fitted state and is replayed as-is at ``transform``.
"""
# Importation des modules
# Modules de base
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Mapping, Optional, Tuple, Union

# Manipulation de données
import pandas as pd

# Primitives de souillure et de provenance (lot précédent)
from .provenance import ProvenanceType, Taint, resolve_model_provenance
# Sentinelle de repli, partagée avec la v1 : simple constante, pas une hiérarchie
from .imputation_plan import INTERPOLATE_FALLBACK


# Alias de typage repris de la v1 (redéfinis localement pour éviter tout couplage
# de module à une classe destinée à disparaître)
FrequencyLabel = Union[str, FrozenSet[Tuple[Any, str]]]
GroupKey = Union[str, Tuple[str, str]]
EntityKey = Tuple[str, ...]


# Voies de matérialisation d'une covariable sur la grille d'une étape.
MaterializationWay = Literal[
    'identity',       # rang 1 : f_c == f, la covariable est lue telle quelle
    'aggregate',      # rang 1 : f_c plus fine que f, agrégation exacte sur période complète
    'stage_model',    # rang 2 : covariable imputée à l'étape courante, lue dans le miroir
    'carried_model',  # rang 3 : covariable imputée à une étape antérieure puis REPORTÉE sur f
    'interpolate',    # rang 4 : covariate_fallback='interpolate', interpolation des observations
    'raw_anchors',    # rang 4 : covariate_fallback='tolerate_nan' (ou stratégie 'tolerate_nan')
]

# Ensemble de contrôle des voies admissibles
_MATERIALIZATION_WAYS: FrozenSet[str] = frozenset(
    ('identity', 'aggregate', 'stage_model', 'carried_model', 'interpolate', 'raw_anchors')
)


# Fonction auxiliaire de test de la sentinelle de repli
def _is_interpolate_fallback(model: Any) -> bool:
    """Tell whether ``model`` is the interpolation-fallback sentinel.

    Args:
        model: The ``model`` attribute of a step.

    Returns:
        True when ``model`` equals :data:`INTERPOLATE_FALLBACK`.
    """
    return isinstance(model, str) and model == INTERPOLATE_FALLBACK


# Fonction auxiliaiire de test de l'égalité par ligne
def _scale_equal(left: Union[float, pd.Series], right: Union[float, pd.Series]) -> bool:
    """Compare two scale factors, either scalars or per-row Series.

    Args:
        left: First scale factor.
        right: Second scale factor.

    Returns:
        True when both are the same scalar, or two Series equal element-wise
        and index-wise (``pd.Series.equals``). A scalar and a Series are
        never equal.
    """
    # Extraction du type des entrées
    left_is_series = isinstance(left, pd.Series)
    right_is_series = isinstance(right, pd.Series)
    # Types hétérogènes : jamais égaux
    if left_is_series != right_is_series:
        return False
    # Vérification que les séries sont égales
    if left_is_series:
        return left.equals(right)
    # Vérification de l'égalité des floats
    return left == right


# Étape du plan d'imputation
@dataclass(frozen=True, eq=False)
class ImputationStep:
    """One step of the fitted imputation plan.

    A step is the unit of work of the cascade: one variable group imputed at
    one prediction frequency. It carries everything ``transform`` needs to
    replay that work — the estimator and its training-time metadata, the
    scaling factors, the per-covariate materialization way and the emitted
    taints — so that the replay never has to cross-reference several
    parallel registries.

    Steps are immutable (``frozen=True``): build a variant with
    :func:`dataclasses.replace` rather than mutating one, and grow a plan
    with :func:`append_step`.

    Attributes:
        pred_freq_label: Canonical, hashable label of the stage frequency —
            the frequency string for a time series, a frozenset of the
            entity -> frequency items for a panel. First component of
            :attr:`stage_key`.
        pred_freq: Raw prediction frequency of the stage: a string for a
            time series, a dict entity -> frequency for a panel.
        var_key: Registry group key, second component of :attr:`stage_key`:
            the variable name alone, or ``(variable name, detected
            frequency)`` for a panel whose entities disagree on the
            frequency of that variable.
        var_name: Column being imputed at this step.
        model: Fitted estimator, or the string :data:`INTERPOLATE_FALLBACK`
            when the step falls back on linear interpolation.
        feature_cols: Feature columns the model was fitted on, in fit order.
            Empty for a fallback step.
        scale_factor: Number of stage sub-periods held by one period of the
            variable — 12 for a yearly variable predicted monthly. May be a
            per-row :class:`pandas.Series` when ``y_train`` mixes rows
            produced at different frequencies.
        fit_scale_factor: Scale factor baked into the model at fit time.
            Same per-row form allowed. Never changes once fitted.
        source_frequency: Normalized detected frequency of the variable for
            this group. Defines the periods the predictions are
            disaggregated over when ``aggregation_constraint`` is on.
        entities: Entity tuples covered by the group, None for a time
            series.
        covariate_taint: Worst taint among the covariate cells the model
            actually read, over its effective ``feature_cols``.
        target_taint: Worst taint among the ``y_train`` rows the model fit
            on.
        materialization: Mapping feature column -> :data:`MaterializationWay`,
            one entry per column of :attr:`feature_cols`, replayed as-is at
            ``transform``. Stored as a read-only mapping keyed
            in ``feature_cols`` order.
        is_fallback: Whether the step produces its values by linear
            interpolation instead of a model. A stored field
            rather than a derived property, because a step may be a fallback
            for reasons other than ``model is INTERPOLATE_FALLBACK``; the
            converse implication is still enforced in ``__post_init__``.
        interpolation_method: Interpolation method retained for this
            variable.
        interpolation_anchor: Anchor fraction retained for this variable
            (0.0 / 0.5 / 1.0), or None when not applicable.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> step = ImputationStep(
        ...     pred_freq_label='M',
        ...     pred_freq='M',
        ...     var_key='gdp',
        ...     var_name='gdp',
        ...     model=LinearRegression(),
        ...     feature_cols=('industrial_production',),
        ...     scale_factor=3.0,
        ...     fit_scale_factor=3.0,
        ...     source_frequency='Q',
        ...     entities=None,
        ...     covariate_taint='none',
        ...     target_taint='none',
        ...     materialization={'industrial_production': 'identity'},
        ...     is_fallback=False,
        ...     interpolation_method='linear',
        ...     interpolation_anchor=1.0,
        ... )
        >>> step.stage_key
        ('M', 'gdp')
        >>> step.emitted_provenance
        <ProvenanceType.MODEL_ON_TRUE: 'model_on_true'>
    """
    # Initialisation des attributs
    pred_freq_label: FrequencyLabel
    pred_freq: Union[str, Dict[EntityKey, str]]
    var_key: GroupKey
    var_name: str
    model: Any
    feature_cols: Tuple[str, ...]
    scale_factor: Union[float, pd.Series]
    fit_scale_factor: Union[float, pd.Series]
    source_frequency: str
    entities: Optional[Tuple[EntityKey, ...]]
    covariate_taint: Taint
    target_taint: Taint
    materialization: Mapping[str, MaterializationWay]
    is_fallback: bool
    interpolation_method: str
    interpolation_anchor: Optional[float]

    # Contrôles d'invariants et gel des conteneurs mutables
    def __post_init__(self) -> None:
        """Validate the step and freeze its containers.

        Raises:
            ValueError: If ``materialization`` does not cover exactly
                ``feature_cols``, if it names an unknown way, or if
                ``model is INTERPOLATE_FALLBACK`` while ``is_fallback`` is
                False.
        """
        # Normalisation de feature_cols en tuple (immuabilité, base de la
        # vérification de couverture)
        feature_cols = tuple(self.feature_cols)
        object.__setattr__(self, 'feature_cols', feature_cols)

        # Couverture exacte de feature_cols par materialization
        provided = dict(self.materialization)
        expected_keys = set(feature_cols)
        provided_keys = set(provided)
        # Comparaison des features fournies et attendues
        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys
        # Cas où des features sont en trop ou manquantes
        if missing or extra:
            # Construction du message d'erreur
            details = []
            if missing:
                details.append(f"Columns without a materialization path : {sorted(missing)}")
            if extra:
                details.append(f"Orphaned materialization paths : {sorted(extra)}")
            raise ValueError(
                f"Materialization doit couvrir exactement feature_cols ; " + " ; ".join(details)
            )

        # Voies admissibles
        bad_ways = {
            col: way for col, way in provided.items() if way not in _MATERIALIZATION_WAYS
        }
        if bad_ways:
            raise ValueError(
                f"voies de matérialisation inconnues : {bad_ways} "
                f"(admises : {sorted(_MATERIALIZATION_WAYS)})"
            )

        # Gel de materialization, réordonné selon feature_cols pour un rendu
        # déterministe en debug et dans le diagnostic
        frozen_materialization = MappingProxyType(
            {col: provided[col] for col in feature_cols}
        )
        object.__setattr__(self, 'materialization', frozen_materialization)

        # Cohérence du repli : la sentinelle impose is_fallback
        if _is_interpolate_fallback(self.model) and not self.is_fallback:
            raise ValueError(
                "model is INTERPOLATE_FALLBACK impose is_fallback=True "
                f"(var_name={self.var_name!r}, stage={self.pred_freq_label!r})"
            )

    # Clé d'étape, identifiant du couple (étape de cascade, groupe de variables)
    @property
    def stage_key(self) -> Tuple[FrequencyLabel, GroupKey]:
        """Registry key of the step.

        Returns:
            The ``(pred_freq_label, var_key)`` tuple used as the key of
            ``imputation_models_``.
        """
        return (self.pred_freq_label, self.var_key)

    # Provenance émise par le modèle de l'étape
    @property
    def emitted_provenance(self) -> ProvenanceType:
        """Provenance every cell produced by this step's model carries.

        Returns:
            :attr:`ProvenanceType.INTERPOLATED` for a fallback step, 
             otherwise the MODEL_* provenance resolved from
            :attr:`covariate_taint` and :attr:`target_taint` by
            :func:`~tsforecast.frequency.provenance.resolve_model_provenance`.
        """
        if self.is_fallback:
            return ProvenanceType.INTERPOLATED
        return resolve_model_provenance(self.covariate_taint, self.target_taint)

    # Égalité explicite : l'égalité générée lèverait sur un scale_factor Series
    def __eq__(self, other: object) -> bool:
        """Compare two steps field by field, Series-safe.

        Args:
            other: Object to compare with.

        Returns:
            True when ``other`` is an :class:`ImputationStep` equal on every
            field. ``model`` is compared by identity (estimators have no
            value equality); ``scale_factor`` / ``fit_scale_factor`` are
            compared with :func:`pandas.Series.equals` when they are Series.
        """
        if not isinstance(other, ImputationStep):
            return NotImplemented
        return (
            self.pred_freq_label == other.pred_freq_label
            and self.pred_freq == other.pred_freq
            and self.var_key == other.var_key
            and self.var_name == other.var_name
            and self.model is other.model
            and self.feature_cols == other.feature_cols
            and _scale_equal(self.scale_factor, other.scale_factor)
            and _scale_equal(self.fit_scale_factor, other.fit_scale_factor)
            and self.source_frequency == other.source_frequency
            and self.entities == other.entities
            and self.covariate_taint == other.covariate_taint
            and self.target_taint == other.target_taint
            and dict(self.materialization) == dict(other.materialization)
            and self.is_fallback == other.is_fallback
            and self.interpolation_method == other.interpolation_method
            and self.interpolation_anchor == other.interpolation_anchor
        )

    # Hachage sur un sous-ensemble sûrement hachable et stable
    def __hash__(self) -> int:
        """Hash over the hashable, identity-stable fields of the step.

        Equal steps agree on all these fields, so the eq/hash contract
        holds; unequal steps may collide, which is acceptable.
        """
        return hash((
            self.pred_freq_label,
            self.var_name,
            self.feature_cols,
            self.covariate_taint,
            self.target_taint,
            self.is_fallback,
            self.interpolation_method,
            self.interpolation_anchor,
        ))


# Conteneur immuable du plan complet
@dataclass(frozen=True)
class ImputationPlan:
    """Immutable ordered container of :class:`ImputationStep` (version 2).

    The plan is the complete fitted state replayed as-is at ``transform``.
    It is never mutated in place; :func:`append_step` returns a new plan.

    Attributes:
        steps: The ordered tuple of steps.

    Examples:
        >>> plan = ImputationPlan()
        >>> len(plan)
        0
    """

    steps: Tuple[ImputationStep, ...] = field(default_factory=tuple)

    # Normalisation en tuple (une liste passée reste immuable côté plan)
    def __post_init__(self) -> None:
        """Freeze ``steps`` into a tuple."""
        object.__setattr__(self, 'steps', tuple(self.steps))

    # Itération et indexation en lecture seule
    def __iter__(self):
        """Iterate over the steps in order."""
        return iter(self.steps)

    def __len__(self) -> int:
        """Number of steps in the plan."""
        return len(self.steps)

    def __getitem__(self, index):
        """Index or slice the steps.

        Args:
            index: Integer position or slice.

        Returns:
            The step at ``index``, or a tuple of steps for a slice.
        """
        return self.steps[index]

    # Regroupement des étapes par fréquence d'étape, ordre préservé
    def by_stage(self) -> Dict[FrequencyLabel, Tuple[ImputationStep, ...]]:
        """Group the steps by stage frequency, preserving order.

        Returns:
            Dict mapping each ``pred_freq_label`` to its steps, in the order
            the stages and the steps first appear in the plan.
        """
        grouped: Dict[FrequencyLabel, List[ImputationStep]] = {}
        for step in self.steps:
            grouped.setdefault(step.pred_freq_label, []).append(step)
        return {label: tuple(steps) for label, steps in grouped.items()}

    # Vue registre des modèles
    def models(self) -> Dict[Tuple[FrequencyLabel, GroupKey], Any]:
        """Build the ``imputation_models_`` view.

        Returns:
            Dict mapping each step's :attr:`~ImputationStep.stage_key` to its
            ``model``.
        """
        return {step.stage_key: step.model for step in self.steps}

    # Sérialisation de diagnostic
    def to_diagnostic_frame(self) -> pd.DataFrame:
        """Serialize the plan as a one-row-per-step diagnostic frame.

        Returns:
            DataFrame with columns ``stage``, ``variable``, ``n_features``,
            ``covariate_taint``, ``target_taint``, ``emitted_provenance``
            (via
            :func:`~tsforecast.frequency.provenance.resolve_model_provenance`,
            or :attr:`ProvenanceType.INTERPOLATED` for a fallback step),
            ``is_fallback``, ``interpolation_method``,
            ``interpolation_anchor`` and ``materialization`` (rendered as a
            comma-separated ``col=way`` string).

        Examples:
            >>> ImputationPlan().to_diagnostic_frame().columns.tolist()
            ['stage', 'variable', 'n_features', 'covariate_taint', 'target_taint', 'emitted_provenance', 'is_fallback', 'interpolation_method', 'interpolation_anchor', 'materialization']
        """
        columns = [
            'stage', 'variable', 'n_features', 'covariate_taint', 'target_taint',
            'emitted_provenance', 'is_fallback', 'interpolation_method',
            'interpolation_anchor', 'materialization',
        ]
        rows = []
        for step in self.steps:
            # Rendu lisible de la voie de matérialisation
            materialization_repr = ", ".join(
                f"{col}={way}" for col, way in step.materialization.items()
            )
            rows.append({
                'stage': step.pred_freq_label,
                'variable': step.var_name,
                'n_features': len(step.feature_cols),
                'covariate_taint': step.covariate_taint,
                'target_taint': step.target_taint,
                'emitted_provenance': step.emitted_provenance,
                'is_fallback': step.is_fallback,
                'interpolation_method': step.interpolation_method,
                'interpolation_anchor': step.interpolation_anchor,
                'materialization': materialization_repr,
            })
        return pd.DataFrame(rows, columns=columns)


# Construction incrémentale : un nouveau plan, jamais de mutation en place
def append_step(plan: ImputationPlan, step: ImputationStep) -> ImputationPlan:
    """Return a new plan with ``step`` appended.

    The PHASE 5 build of ``HighFrequencyImputer2`` grows its plan through
    this function; ``plan`` is never mutated.

    Args:
        plan: The current plan.
        step: The step to append.

    Returns:
        A new :class:`ImputationPlan` whose steps are those of ``plan``
        followed by ``step``.

    Examples:
        >>> plan = ImputationPlan()
        >>> append_step(plan, _example_step()).steps  # doctest: +SKIP
        (ImputationStep(...),)
        >>> len(plan)
        0
    """
    return ImputationPlan((*plan.steps, step))


# Fonction de normalisation d'une liste d'entités en tuple immuable
def to_entity_tuple(
    entities: Optional[Iterable[EntityKey]],
) -> Optional[Tuple[EntityKey, ...]]:
    """Freeze an iterable of entity keys for storage in an ImputationStep.

    Args:
        entities: Entity tuples of a group, or None for a time series.

    Returns:
        Tuple of entity keys, or None.

    Examples:
        >>> to_entity_tuple([('France',), ('Italie',)])
        (('France',), ('Italie',))
        >>> to_entity_tuple(None) is None
        True
    """
    # Conservation du None : il signale une série temporelle, pas un groupe vide
    if entities is None:
        return None
    return tuple(entities)
