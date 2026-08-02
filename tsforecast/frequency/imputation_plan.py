"""Explicit imputation plan produced by the mixed-frequency imputer.

This module holds :class:`ImputationStep`, the immutable description of one
step of the cascade fitted by
:class:`tsforecast.frequency.HighFrequencyImputer`. The ordered list of those
steps — ``imputation_plan_`` — is the single source of truth of the fitted
state: ``imputation_models_``, ``model_fitting_order_``, ``stage_groups_`` and
``frequency_progression_`` are all derived, read-only views of it (review
§5.3).
"""
# Importation des modules
# Modules de base
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
# Manipulation de données
import pandas as pd


# Sentinelle de repli : valeur historique portée par "imputation_models_" quand
# aucun modèle n'a pu être entraîné pour l'étape
INTERPOLATE_FALLBACK = 'interpolate_fallback'


# Type aliases
FrequencyLabel = Union[str, FrozenSet[Tuple[Any, str]]]
GroupKey = Union[str, Tuple[str, str]]
EntityKey = Tuple[str, ...]


# Étape du plan d'imputation
# "eq=False" est nécessaire : l'égalité générée comparerait les champs un à un,
# donc "feature_means" (une Series) via un "==" élémentaire dont la valeur de
# vérité est ambiguë. L'identité suffit ici, aucun code ne compare deux étapes
@dataclass(frozen=True, eq=False)
class ImputationStep:
    """One step of the fitted imputation plan.

    A step is the unit of work of the cascade: one variable group imputed at
    one prediction frequency. It carries everything ``transform`` needs to
    replay that work — the estimator and its training-time metadata, the
    scaling factors, and the group metadata driving the disaggregation — so
    that the replay never has to cross-reference several parallel registries
    (review §5.3).

    Steps are immutable: build a variant with :func:`dataclasses.replace`
    rather than mutating one. That is what backs ``cascade_refitting=False``,
    where the very same estimator is registered at several stages with a
    different ``scale_factor``.

    Attributes:
        pred_freq_label: Canonical, hashable label of the stage frequency —
            the frequency string for a time series, a frozenset of the
            entity -> frequency items for a panel. First component of
            :attr:`stage_key`.
        pred_freq: Raw prediction frequency of the stage, as carried by
            ``freq_prediction_list_``: a string for a time series, a dict
            entity -> frequency for a panel.
        var_key: Registry group key, second component of :attr:`stage_key`:
            the variable (column) name alone, or ``(variable name, detected
            frequency)`` for a panel whose entities disagree on the
            frequency of that variable (review §2.4). NEVER an
            ``(entity, variable)`` pair: the model backing a step is global
            to the panel.
        var_name: Column being imputed at this step.
        model: Fitted estimator, or the string :data:`INTERPOLATE_FALLBACK`
            when no model could be trained and the values are produced by
            linear interpolation instead (see :attr:`is_fallback`).
        feature_cols: Feature columns the model was fitted on, in fit order.
            Empty for a fallback step.
        feature_means: Per-covariate means of the TRAINING set, reapplied at
            prediction time to fill the missing covariates (review §2.7) so
            that the result does not depend on the prediction sample. None
            for a fallback step.
        scale_factor: Number of stage sub-periods held by one period of the
            variable — 12 for a yearly variable predicted monthly. Follows
            the stage, so it differs from :attr:`fit_scale_factor` for a
            model reused across stages.
        fit_scale_factor: Scale factor baked into the model at fit time
            (``y_train`` was divided by it). Never changes once fitted:
            :meth:`~tsforecast.frequency.HighFrequencyImputer._stage_predictions`
            uses the ratio with :attr:`scale_factor` to bring a reused
            model's output back to the current stage's scale.
        trained_on_imputed: Whether imputed values entered the training set,
            which drives the MODEL_ON_MIXED vs MODEL_ON_TRUE provenance of
            the cells this step produces.
        source_frequency: Normalized detected frequency of the variable for
            this group. Defines the periods the predictions are
            disaggregated over when ``enforce_period_totals`` is on.
        entities: Entity tuples covered by the group, None for a time
            series. Restricts the disaggregation scope to the entities the
            group was built from.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> step = ImputationStep(
        ...     pred_freq_label='M',
        ...     pred_freq='M',
        ...     var_key='gdp',
        ...     var_name='gdp',
        ...     model=LinearRegression(),
        ...     feature_cols=('industrial_production',),
        ...     feature_means=None,
        ...     scale_factor=3.0,
        ...     fit_scale_factor=3.0,
        ...     trained_on_imputed=False,
        ...     source_frequency='Q',
        ...     entities=None,
        ... )
        >>> step.stage_key
        ('M', 'gdp')
        >>> step.is_fallback
        False
    """

    pred_freq_label: FrequencyLabel
    pred_freq: Union[str, Dict[EntityKey, str]]
    var_key: GroupKey
    var_name: str
    model: Any
    feature_cols: Tuple[str, ...]
    feature_means: Optional[pd.Series]
    scale_factor: float
    fit_scale_factor: float
    trained_on_imputed: bool
    source_frequency: str
    entities: Optional[Tuple[EntityKey, ...]]

    # Clé d'étape, identifiant du couple (étape de cascade, groupe de variables)
    @property
    def stage_key(self) -> Tuple[FrequencyLabel, GroupKey]:
        """Registry key of the step.

        Returns:
            The ``(pred_freq_label, var_key)`` tuple used as the key of
            ``imputation_models_`` and as the entries of
            ``model_fitting_order_``.
        """
        return (self.pred_freq_label, self.var_key)

    # Nature de l'étape : modèle entraîné ou repli par interpolation
    @property
    def is_fallback(self) -> bool:
        """Tell whether the step falls back on linear interpolation.

        Returns:
            True when no estimator could be fitted for the step, its
            :attr:`model` then being :data:`INTERPOLATE_FALLBACK`.
        """
        return isinstance(self.model, str) and self.model == INTERPOLATE_FALLBACK

    # Entrée de registre correspondante, au format historique
    def to_registry_entry(self) -> Union[str, Dict[str, Any]]:
        """Build the ``imputation_models_`` entry of the step.

        Reproduces the historical registry format exactly, so that the
        derived ``imputation_models_`` view stays consumable as before:
        a bare :data:`INTERPOLATE_FALLBACK` string for a fallback step, the
        seven-key metadata dict otherwise.

        Returns:
            The fallback sentinel, or a dict with keys ``model``,
            ``feature_cols``, ``feature_means``, ``scale_factor``,
            ``fit_scale_factor``, ``pred_freq`` and ``trained_on_imputed``.
        """
        # Cas du repli : l'entrée historique est une simple chaîne
        if self.is_fallback:
            return INTERPOLATE_FALLBACK

        return {
            'model': self.model,
            # Reconversion en liste : le tuple sert l'immuabilité de l'étape,
            # mais un tuple passé à un ".loc" de colonnes serait interprété
            # comme une clé unique de MultiIndex
            'feature_cols': list(self.feature_cols),
            'feature_means': self.feature_means,
            'scale_factor': self.scale_factor,
            'fit_scale_factor': self.fit_scale_factor,
            'pred_freq': self.pred_freq,
            'trained_on_imputed': self.trained_on_imputed,
        }

    # Métadonnées de groupe, au format attendu par les auxiliaires de désagrégation
    def group_metadata(self) -> Dict[str, Any]:
        """Build the ``stage_groups_`` entry of the step.

        Format expected by the disaggregation helpers
        (``_prediction_masks``, ``_apply_period_totals``,
        ``_disaggregation_mask``), which read a plain mapping so they stay
        callable with a hand-written dict.

        Returns:
            Dict with keys ``var_name``, ``f_var`` (the group's normalized
            source frequency) and ``entities``.
        """
        return {
            'var_name': self.var_name,
            'f_var': self.source_frequency,
            'entities': list(self.entities) if self.entities is not None else None,
        }


# Fonction de normalisation d'une liste d'entités en tuple immuable
def to_entity_tuple(
    entities: Optional[List[EntityKey]],
) -> Optional[Tuple[EntityKey, ...]]:
    """Freeze a list of entity keys for storage in an ImputationStep.

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
