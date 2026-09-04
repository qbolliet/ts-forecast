"""Materialization of covariates on one imputation-stage grid.

This module holds the component that carries the central invariant of
``HighFrequencyImputer2``:

    A model never sees, at prediction time, a feature-availability pattern more
    degraded than at training time, nor features of a different nature.

Its structural corollary is that there is exactly only one method producing
``X_train`` and ``X_pred`` — :meth:`CovariateMaterializer.materialize` — called
by the fit and by the transform alike.

The component knows nothing of the plan, of the estimator, nor of the
imputation windows: it receives a grid index and some data, and returns
features. Window masks are applied by the caller, upstream, on ``grid_index``.
"""
# Importation des modules
# Modules de base
from typing import (
    Any, Dict, FrozenSet, Iterable, List, Literal, Mapping, Optional,
    Protocol, Sequence, Tuple, Union,
)

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Voies de matérialisation, définies avec l'étape du plan
from .imputation_plan2 import MaterializationWay
# Primitives d'origine de cellule
from .provenance import CellOrigin, max_origin
# Arithmétique et conversion de fréquences
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import is_higher_frequency, normalize_frequency
# Primitives du paramètre de contrainte d'agrégation, partagées avec
# "AggregationConstraint" : une seule validation, une seule résolution
from .aggregation_constraint import (
    AggregationConstraintSetting,
    ConstraintSetting,
    resolve_aggregation_constraint,
    validate_aggregation_constraint,
)
# Utilitaires de panel : découpage et normalisation des clés d'entité
from ..panel.utils import iter_entity_blocks, normalize_entity_key


# Clé de repli des formes dictionnaire par feature.
# Même convention que ``stage_scaler.DEFAULT_SCALE_KEY``, redéfinie localement
# pour ne pas coupler les deux modules.
DEFAULT_MATERIALIZATION_KEY = '__default__'

# Méthode d'interpolation par défaut
DEFAULT_INTERPOLATION_METHOD = 'time'

# Stratégies et modalités admises
_COVARIATE_STRATEGIES: FrozenSet[str] = frozenset(
    ('tolerate_nan', 'interpolate', 'model')
)
_COVARIATE_FALLBACKS: FrozenSet[str] = frozenset(('interpolate', 'tolerate_nan'))
_COVARIATE_ELIGIBILITIES: FrozenSet[str] = frozenset(('any_entity', 'all_entities'))

# Voies réservées aux modèles
_MODEL_WAYS: FrozenSet[str] = frozenset(('stage_model', 'carried_model'))

# Degré de dégradation d'une voie, servant à réduire des verdicts par entité en
# une voie unique par colonne.
_WAY_RANK: Dict[str, int] = {
    'identity': 0,
    'aggregate': 1,
    'interpolate': 2,
    'raw_anchors': 2,
    'stage_model': 3,
    'carried_model': 3,
}

# Origine des cellules produites par chaque voie
_WAY_ORIGIN: Dict[str, CellOrigin] = {
    'identity': 'observed',
    'aggregate': 'observed',
    'raw_anchors': 'observed',
    'interpolate': 'interpolated',
    'stage_model': 'model',
    'carried_model': 'model',
}

# Convertisseur partagé par défaut, pour ne pas en instancier un par appel
_SHARED_CONVERTER = FrequencyConverter()

# Clé d'entité, tuple normalisé (``()`` pour une série temporelle)
EntityKey = Tuple[Any, ...]
# Fréquence d'étape : scalaire, ou une par entité
StageFrequency = Union[str, Mapping[EntityKey, str]]
# Fréquences détectées : par colonne, éventuellement par entité
DetectedFrequencies = Mapping[str, Union[str, Mapping[EntityKey, str]]]
# Réglage par feature : scalaire, ou dictionnaire avec ``'__default__'``
PerFeature = Union[Any, Mapping[str, Any]]


# Contrat du composant de recalage aux totaux de période
class AggregationConstraintApplier(Protocol):
    """Protocol the ``AggregationConstraint`` component must honour.

    The materializer never implements the rescaling itself: it only holds the
    injection point, so that the constraint stays inside the single
    materialization path rather than being re-applied by each caller.
    """

    # Méthode de recalage des sous-périodes sur le total observé
    def rescale(
        self,
        values: pd.Series,
        observations: pd.Series,
        period_freq: str,
        column: Optional[str] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Rescale sub-period values so each complete period matches its total.

        Args:
            values: Values produced on the stage grid, indexed by date.
            observations: Observed low-frequency totals of the column,
                indexed by their own anchor dates.
            period_freq: Frequency of the periods the totals refer to.
            column: Name of the column, so that an applier configured with a
                per-column dict resolves the right constraint. The
                materializer always passes it; it stays optional so an
                applier carrying a single scalar setting can ignore it.

        Returns:
            Tuple ``(rescaled, rescaled_mask)``: the rescaled values, and the
            boolean mask of the cells the constraint actually moved.
        """
        ...


# Fonction auxiliaire de résolution d'un réglage par feature
def _resolve_per_feature(setting: PerFeature, column: Optional[str], default: Any) -> Any:
    """Resolve a scalar-or-dict per-feature setting for one column.

    Args:
        setting: Scalar value, or dict keyed by column name with an optional
            ``'__default__'`` entry.
        column: Column name, or None for the global setting.
        default: Value returned when neither the column nor ``'__default__'``
            is present in the dict form.

    Returns:
        The column's own entry, else the ``'__default__'`` entry, else
        ``default``.

    Examples:
        >>> _resolve_per_feature('linear', 'a1', 'linear')
        'linear'
        >>> _resolve_per_feature({'a1': 'cubic', '__default__': 'time'}, 'a1', 'linear')
        'cubic'
        >>> _resolve_per_feature({'a1': 'cubic', '__default__': 'time'}, 'q1', 'linear')
        'time'
        >>> _resolve_per_feature({'a1': 'cubic'}, 'q1', 'linear')
        'linear'
    """
    # Forme scalaire : la même valeur pour toutes les colonnes
    if not isinstance(setting, Mapping):
        return setting

    # Entrée propre à la colonne
    if column is not None and column in setting:
        return setting[column]
    # Repli explicite
    if DEFAULT_MATERIALIZATION_KEY in setting:
        return setting[DEFAULT_MATERIALIZATION_KEY]
    # Défaut du document
    return default


# Fonction auxiliaire de comparaison de deux fréquences
def _same_frequency(left: Optional[str], right: Optional[str]) -> bool:
    """Tell whether two frequency labels denote the same base frequency.

    Args:
        left: First frequency label ('M', 'ME', 'monthly'...), or None.
        right: Second frequency label, or None.

    Returns:
        True when both normalize to the same base frequency. False as soon as
        one of them is None or unrecognized.

    Examples:
        >>> _same_frequency('ME', 'monthly')
        True
        >>> _same_frequency('Q', 'M')
        False
        >>> _same_frequency(None, 'M')
        False
    """
    # Une fréquence inconnue n'est comparable à rien
    if left is None or right is None:
        return False
    try:
        return normalize_frequency(left) == normalize_frequency(right)
    except (ValueError, TypeError):
        return False


# Composant de matérialisation des covariables
class CovariateMaterializer:
    """Single producer of the feature frames of ``HighFrequencyImputer2``.

    Three registries are held by this component and by it alone:

    - :attr:`imputed_store` — mirror of the values produced, per column;
    - :attr:`imputed_freq_store` — production frequency of each cell, needed
      by the per-row divisors;
    - :attr:`origin_store` — :data:`~tsforecast.frequency.provenance.CellOrigin`
      of each cell.

    Every write goes through ``new.combine_first(existing)``: cells
    the new production does not cover are never destroyed, unlike a plain
    assignment. Presence in :attr:`imputed_store` is therefore not a
    model-imputation signal — observed and interpolated cells sit there too;
    ranks 2 and 3 of the precedence must discriminate on
    :attr:`origin_store` ``== 'model'``.

    The fallback materializes : a value produced by
    fallback feeds the three stores exactly like a model prediction. This is a
    design point, not a detail: ``HighFrequencyImputer``'s
    ``_write_interpolation_fallback`` fed neither ``imputed_store`` nor the
    mirror, and the following steps were left blind to it.

    Attributes:
        imputed_store: Mapping column -> values produced so far.
        imputed_freq_store: Mapping column -> production frequency per cell.
        origin_store: Mapping column -> :data:`CellOrigin` per cell.

    Args:
        covariate_strategy: How a covariate observed at a frequency strictly
            lower than the stage grid is made available to the model.

            - ``'tolerate_nan'``: no materialization at all. The covariate
              carries its values on its own anchor dates only, and NaN
              everywhere else — at fit and at predict. Hard prerequisite of
              this modality: the estimator must tolerate NaN
              (``HistGradientBoostingRegressor``, ``LGBMRegressor``, or a
              ``Pipeline`` including a ``SimpleImputer``). Nothing filters
              those NaN out downstream.
            - ``'interpolate'`` (default): the covariate is interpolated over
              the grid from its observed values, then rescaled to preserve
              the period aggregate named by ``aggregation_constraint`` when
              that parameter is active.
              Looking downstream: linear interpolation between two
              anchors uses the future anchor — in a pseudo-real-time setting
              that is information from the future. It is not a defect for
              historical imputation, but ``imputation_scope='extended_forward'``
              remains the dedicated mechanism for series ends.
            - ``'model'``: covariates are imputed by the same fit/predict
              mechanism as the target variables, in the order given by
              ``fit_predict_order`` — the only mode where that order has an
              effect. The four-rank precedence then applies, stopping at the
              first applicable rank: rank 1 identity or exact aggregation;
              rank 2 the covariate's own imputation at the current stage,
              read from the mirror (``'stage_model'``); rank 3 its imputation
              at an earlier stage ``f'``, carried onto the current grid by its
              own interpolation way and rescaled to the ``f'`` totals
              (``'carried_model'``); rank 4 ``covariate_fallback``.
        covariate_fallback: Secondary approach used at rank 4 of the
            precedence, when the covariate has been materialized by none of
            the higher ranks. Applied at fit and at predict — never full at
            fit and empty at predict.
        covariate_eligibility: Handling of a feature with no observation at
            all for a whole entity. ``'any_entity'`` keeps the column
            as soon as one entity observes it, the empty entities' rows
            staying NaN; ``'all_entities'`` drops the column entirely.
        interpolation_method: Interpolation method, scalar or per-feature dict
            with an optional ``'__default__'`` key. Admissible
            values are those of
            :meth:`FrequencyConverter.interpolate_to_higher_frequency`.
        interpolation_anchor: Position, within its own period, at which an
            observed value is considered reached — ``0.0`` start, ``0.5``
            middle, ``1.0`` end, None attributing the value to the anchor date.
            Scalar or per-feature dict, same rules.
        aggregation_constraint: Constraint the interpolated covariates are
            rescaled to. ``'sum'`` (default) preserves the period totals, and
            None applies no constraint — additivity is the contract of the
            whole class, not a per-column choice, and a
            non-additive column goes through ``additive_transformer`` instead.
            A dict ``{column: setting}`` sets it per column, with an optional
            ``'__default__'`` key — the same convention as
            ``interpolation_method``. Validated by
            :func:`validate_aggregation_constraint`, shared with
            :class:`AggregationConstraint` so the two contracts cannot drift.
            Governs the rescaling of the INTERPOLATED covariates only
            — the rank-1 exact aggregation of a finer covariate
            always sums directly, and never consults this parameter.

        converter: Shared :class:`FrequencyConverter`. A module-level shared
            instance is used when None.
        aggregation_constraint_applier: Object honouring
            :class:`AggregationConstraintApplier` — the ``AggregationConstraint``
            component. Left inert when None, so this lot produces
            un-rescaled interpolations.

    Raises:
        ValueError: If any setting is outside its admissible values, or if a
            per-feature dict is empty.

    Examples:
        >>> materializer = CovariateMaterializer(covariate_strategy='interpolate')
        >>> materializer.resolve_method('a1')
        'time'
        >>> materializer.imputed_store
        {}
    """

    # Initialisation du composant : validation sans transformation
    def __init__(
        self,
        covariate_strategy: Literal['tolerate_nan', 'interpolate', 'model'] = 'interpolate',
        covariate_fallback: Literal['interpolate', 'tolerate_nan'] = 'interpolate',
        covariate_eligibility: Literal['any_entity', 'all_entities'] = 'any_entity',
        interpolation_method: PerFeature = DEFAULT_INTERPOLATION_METHOD,
        interpolation_anchor: PerFeature = None,
        aggregation_constraint: AggregationConstraintSetting = 'sum',
        converter: Optional[FrequencyConverter] = None,
        aggregation_constraint_applier: Optional[AggregationConstraintApplier] = None,
    ) -> None:
        # Validation des littéraux
        self._validate_literal('covariate_strategy', covariate_strategy, _COVARIATE_STRATEGIES)
        self._validate_literal('covariate_fallback', covariate_fallback, _COVARIATE_FALLBACKS)
        self._validate_literal(
            'covariate_eligibility', covariate_eligibility, _COVARIATE_ELIGIBILITIES
        )
        # Validation des formes par feature
        self._validate_interpolation_method(interpolation_method)
        self._validate_interpolation_anchor(interpolation_anchor)
        # Validation de la contrainte d'agrégation
        self._validate_aggregation_constraint(aggregation_constraint)

        # Stockage des paramètres en tant qu'attributs
        self.covariate_strategy = covariate_strategy
        self.covariate_fallback = covariate_fallback
        self.covariate_eligibility = covariate_eligibility
        self.interpolation_method = interpolation_method
        self.interpolation_anchor = interpolation_anchor
        self.aggregation_constraint = aggregation_constraint
        self.converter = converter
        self.aggregation_constraint_applier = aggregation_constraint_applier

        # Initialisation des trois registres tenus par ce composant
        self.imputed_store: Dict[str, pd.Series] = {}
        self.imputed_freq_store: Dict[str, pd.Series] = {}
        self.origin_store: Dict[str, pd.Series] = {}

    # -------------------------------------------------------------------------
    # Validations
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de validation d'un littéral
    @staticmethod
    def _validate_literal(name: str, value: Any, admissible: FrozenSet[str]) -> None:
        """Check that a scalar parameter belongs to its admissible set.

        Args:
            name: Parameter name, used in the message.
            value: Value handed to ``__init__``.
            admissible: Admissible values.

        Raises:
            ValueError: If the value is not admissible.
        """
        if value not in admissible:
            raise ValueError(
                f"{name} must be one of {sorted(admissible)}, got {value!r}"
            )

    # Méthode auxiliaire de validation de la méthode d'interpolation
    @staticmethod
    def _validate_interpolation_method(interpolation_method: PerFeature) -> None:
        """Check the scalar or per-feature form of ``interpolation_method``.

        Args:
            interpolation_method: Value handed to ``__init__``.

        Raises:
            ValueError: If the dict form is empty, or if a value is not a
                non-empty string.
        """
        # Forme dictionnaire : chaque valeur doit être une méthode nommée
        if isinstance(interpolation_method, Mapping):
            if not interpolation_method:
                raise ValueError("interpolation_method dict cannot be empty")
            invalid = {
                key: value for key, value in interpolation_method.items()
                if not isinstance(value, str) or not value
            }
            if invalid:
                raise ValueError(
                    f"interpolation_method values must be non-empty method names, "
                    f"got {invalid}"
                )
            return

        # Forme scalaire
        if not isinstance(interpolation_method, str) or not interpolation_method:
            raise ValueError(
                f"interpolation_method must be a non-empty method name or a dict "
                f"of these, got {interpolation_method!r}"
            )

    # Méthode auxiliaire de validation de l'ancrage
    @staticmethod
    def _validate_interpolation_anchor(interpolation_anchor: PerFeature) -> None:
        """Check the scalar or per-feature form of ``interpolation_anchor``.

        Args:
            interpolation_anchor: Value handed to ``__init__``.

        Raises:
            ValueError: If the dict form is empty, or if a value is neither
                None nor a number within ``[0, 1]``.
        """
        # Contrôle d'une valeur unitaire
        def _check(value: Any, label: str) -> None:
            if value is None:
                return
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{label} must be None or a float in [0, 1], got {value!r}"
                )
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{label} must be None or a float in [0, 1], got {value!r}"
                )

        # Forme dictionnaire
        if isinstance(interpolation_anchor, Mapping):
            if not interpolation_anchor:
                raise ValueError("interpolation_anchor dict cannot be empty")
            for key, value in interpolation_anchor.items():
                _check(value, f"interpolation_anchor[{key!r}]")
            return

        # Forme scalaire
        _check(interpolation_anchor, 'interpolation_anchor')

    # Méthode auxiliaire de validation de la contrainte d'agrégation /!\
    @staticmethod
    def _validate_aggregation_constraint(aggregation_constraint: Any) -> None:
        """Check ``aggregation_constraint`` against its contract.

        Delegates to :func:`validate_aggregation_constraint`, the single
        implementation shared with :class:`AggregationConstraint`: a setting
        accepted by the component applying the constraint must be accepted by
        the component carrying it, and one validation is the only way to keep
        the two from drifting.

        Args:
            aggregation_constraint: Value handed to ``__init__``.

        Raises:
            ValueError: For any value other than ``'sum'``, None, or a dict of
                these values. A
                non-additive column goes through ``additive_transformer``
                instead.
        """
        validate_aggregation_constraint(aggregation_constraint)

    # Méthode de résolution de la contrainte effective d'une colonne
    def resolve_aggregation_constraint(
        self,
        column: Optional[str] = None,
    ) -> ConstraintSetting:
        """Return the aggregation constraint applied to one column.

        Symmetric of :meth:`resolve_method` and :meth:`resolve_anchor` for the
        per-column dict form of ``aggregation_constraint``.

        Args:
            column: Column name, or None for the global setting.

        Returns:
            ``'sum'`` or None: the column's own entry, else the
            ``'__default__'`` entry, else ``'sum'``.

        Examples:
            >>> materializer = CovariateMaterializer(
            ...     aggregation_constraint={'a1': None, '__default__': 'sum'}
            ... )
            >>> materializer.resolve_aggregation_constraint('a1') is None
            True
            >>> materializer.resolve_aggregation_constraint('q1')
            'sum'
        """
        return resolve_aggregation_constraint(self.aggregation_constraint, column)

    # Méthode de contrôle des clés des dictionnaires par feature
    def validate_columns(self, columns: Iterable[str]) -> None:
        """Check the per-column dict keys against the columns actually present.

        Called once by the imputer at ``fit``, with the full set of columns:
        the materialization methods only ever see one stage's subset, against
        which a legitimate key of another variable would look unknown.

        Args:
            columns: Column names present in the data.

        Raises:
            ValueError: If a per-column dict (``interpolation_method``,
                ``interpolation_anchor`` or ``aggregation_constraint``)
                names columns absent from ``columns``, listing the
                offending keys.

        Examples:
            >>> mat = CovariateMaterializer(interpolation_method={'a1': 'cubic'})
            >>> mat.validate_columns(['a1', 'q1'])
            >>> mat.validate_columns(['q1'])
            Traceback (most recent call last):
                ...
            ValueError: interpolation_method names unknown columns : ['a1']
        """
        # Ensemble des colonnes connues
        known = set(columns)
        # Contrôle des trois réglages par colonne : la contrainte d'agrégation
        # partage la clé de repli '__default__' des deux autres
        for name, setting in (
            ('interpolation_method', self.interpolation_method),
            ('interpolation_anchor', self.interpolation_anchor),
            ('aggregation_constraint', self.aggregation_constraint),
        ):
            # Seule la forme dictionnaire porte des clés à vérifier
            if not isinstance(setting, Mapping):
                continue
            # La clé de repli n'est pas un nom de colonne
            unknown = sorted(
                key for key in setting
                if key != DEFAULT_MATERIALIZATION_KEY and key not in known
            )
            if unknown:
                raise ValueError(f"{name} names unknown columns : {unknown}")

    # -------------------------------------------------------------------------
    # Résolution des réglages par feature
    # -------------------------------------------------------------------------
    # Méthode de résolution de la méthode d'interpolation d'une colonne
    def resolve_method(self, column: Optional[str] = None) -> str:
        """Return the interpolation method effectively applied to one column.

        Args:
            column: Column name, or None for the global setting.

        Returns:
            The column's own entry, else the ``'__default__'`` entry, else
            :data:`DEFAULT_INTERPOLATION_METHOD`.

        Examples:
            >>> mat = CovariateMaterializer(
            ...     interpolation_method={'a1': 'cubic', '__default__': 'time'}
            ... )
            >>> mat.resolve_method('a1'), mat.resolve_method('q1')
            ('cubic', 'time')
        """
        return _resolve_per_feature(
            self.interpolation_method, column, DEFAULT_INTERPOLATION_METHOD
        )

    # Méthode de résolution de la position d'ancrage d'une colonne
    def resolve_anchor(self, column: Optional[str] = None) -> Optional[float]:
        """Return the anchor fraction effectively applied to one column.

        Args:
            column: Column name, or None for the global setting.

        Returns:
            The column's own entry, else the ``'__default__'`` entry, else
            None (attribution of the value to the anchor date).

        Examples:
            >>> mat = CovariateMaterializer(
            ...     interpolation_anchor={'a1': 0.5, '__default__': None}
            ... )
            >>> mat.resolve_anchor('a1'), mat.resolve_anchor('q1')
            (0.5, None)
        """
        return _resolve_per_feature(self.interpolation_anchor, column, None)

    # Propriété d'accès au convertisseur effectif
    @property
    def _conv(self) -> FrequencyConverter:
        """Frequency converter actually used.

        Returns:
            The injected converter, or the module-level shared instance.
        """
        return self.converter if self.converter is not None else _SHARED_CONVERTER

    # Propriété d'accès à la voie de l'approche secondaire
    @property
    def _fallback_way(self) -> MaterializationWay:
        """Way rank 4 of the precedence resolves to.

        Returns:
            ``'interpolate'`` when ``covariate_fallback`` is ``'interpolate'``,
            ``'raw_anchors'`` otherwise.
        """
        return 'interpolate' if self.covariate_fallback == 'interpolate' else 'raw_anchors'

    # -------------------------------------------------------------------------
    # Les trois registres
    # -------------------------------------------------------------------------
    # Méthode de purge des trois registres
    def reset(self) -> None:
        """Empty the three stores.

        Called between two passes (fit, then transform): a transform must
        never read the fit's mirror.

        Examples:
            >>> mat = CovariateMaterializer()
            >>> mat.imputed_store['a1'] = pd.Series(dtype=float)
            >>> mat.reset()
            >>> mat.imputed_store
            {}
        """
        self.imputed_store = {}
        self.imputed_freq_store = {}
        self.origin_store = {}

    # Méthode de capture de l'état des registres
    def snapshot(self) -> Dict[str, Dict[str, pd.Series]]:
        """Return a deep copy of the three stores.

        Returns:
            Mapping ``{'imputed': ..., 'imputed_freq': ..., 'origin': ...}``,
            each a fresh dict of fresh Series.

        Examples:
            >>> mat = CovariateMaterializer()
            >>> sorted(mat.snapshot())
            ['imputed', 'imputed_freq', 'origin']
        """
        return {
            'imputed': {col: s.copy() for col, s in self.imputed_store.items()},
            'imputed_freq': {col: s.copy() for col, s in self.imputed_freq_store.items()},
            'origin': {col: s.copy() for col, s in self.origin_store.items()},
        }

    # Méthode d'écriture des trois registres
    def _write_stores(
        self,
        column: str,
        values: pd.Series,
        production_freq: pd.Series,
        origins: pd.Series,
    ) -> None:
        """Write one column's production into the three stores.

        The three series are written together: an origin always describes the
        value that actually sits in the mirror. Each write is a
        ``new.combine_first(existing)``: the new values win on the
        overlap, and cells the new production does not cover survive — a plain
        assignment would drop them.

        Args:
            column: Column being written.
            values: Values produced, NaN cells already dropped.
            production_freq: Frequency each cell was produced at, same index.
            origins: :data:`CellOrigin` of each cell, same index.
        """
        # Aucune cellule produite : rien à écrire, et rien à effacer
        if values.empty:
            return

        # Écriture des trois registres, chacun par combine_first
        for store, produced in (
            (self.imputed_store, values),
            (self.imputed_freq_store, production_freq),
            (self.origin_store, origins),
        ):
            existing = store.get(column)
            store[column] = (
                produced if existing is None else produced.combine_first(existing)
            )

    # Méthode auxiliaire d'indexation du miroir d'une colonne par entité
    def _mirror_blocks(self, column: str) -> Dict[EntityKey, pd.DataFrame]:
        """Assemble the three stores of one column, split by entity.

        Args:
            column: Column whose mirror is read.

        Returns:
            Mapping entity key -> date-indexed frame with columns ``value``,
            ``freq`` and ``origin``. Empty mapping when the column has
            produced nothing yet. A time series yields the single degenerate
            entity ``()``.
        """
        # Colonne jamais produite : aucun bloc
        values = self.imputed_store.get(column)
        if values is None or values.empty:
            return {}

        # Assemblage des trois registres sur l'index du miroir
        frame = pd.DataFrame({
            'value': values,
            'freq': self.imputed_freq_store.get(column, pd.Series(dtype=object)),
            'origin': self.origin_store.get(column, pd.Series(dtype=object)),
        })
        # Découpage par entité, index de date seul dans chaque bloc
        return {
            normalize_entity_key(entity): block
            for entity, _mask, block in iter_entity_blocks(frame)
        }

    # Méthode auxiliaire de sélection des cellules produites à une fréquence
    @staticmethod
    def _stage_cells(block: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
        """Select the mirror cells produced at one given frequency.

        Args:
            block: Entity block returned by :meth:`_mirror_blocks`.
            freq: Production frequency looked for, None matching nothing.

        Returns:
            The sub-frame of the cells whose ``freq`` denotes ``freq``.
        """
        # Fréquence inconnue : aucune cellule ne peut être reconnue
        if freq is None or block.empty:
            return block.iloc[:0]
        # Comparaison par fréquence de base, jamais par étiquette brute
        selected = [_same_frequency(entry, freq) for entry in block['freq']]
        return block[np.asarray(selected, dtype=bool)]

    # Méthode auxiliaire de recherche de la fréquence de report
    @staticmethod
    def _carry_frequency(block: pd.DataFrame, f_stage: Optional[str]) -> Optional[str]:
        """Find the stage a rank-3 carry must start from.

        The production frequency is read in ``imputed_freq_store``, never
        re-detected: the carry must start from the grid the values were
        actually produced on.

        Args:
            block: Entity block returned by :meth:`_mirror_blocks`.
            f_stage: Frequency of the current stage.

        Returns:
            The highest production frequency strictly lower than ``f_stage``
            present in the block — the closest earlier stage — or None when
            the block holds none.
        """
        # Fréquence d'étape inconnue : aucun report définissable
        if f_stage is None or block.empty:
            return None

        # Fréquence la plus haute parmi celles strictement plus basses que f_stage
        best: Optional[str] = None
        for freq in block['freq'].dropna().unique():
            if not is_higher_frequency(f_stage, freq):
                continue
            if best is None or is_higher_frequency(freq, best):
                best = freq
        return best

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de lecture de la fréquence d'une colonne
    @staticmethod
    def _column_frequency(
        detected_frequencies: DetectedFrequencies,
        column: str,
        entity: EntityKey = (),
    ) -> Optional[str]:
        """Read the detected frequency of one column, for one entity.

        Args:
            detected_frequencies: Mapping column -> frequency, or column ->
                (entity -> frequency) for a panel whose entities disagree.
            column: Column name.
            entity: Entity key, ``()`` for a time series.

        Returns:
            The frequency label, or None when the column carries none.
        """
        # Fréquence détectée
        entry = detected_frequencies.get(column)
        # Colonne sans fréquence détectée
        if entry is None:
            return None
        # Forme scalaire : la même fréquence pour toutes les entités
        if not isinstance(entry, Mapping):
            return entry
        # Forme par entité : clé brute puis clé normalisée
        key = normalize_entity_key(entity)
        if entity in entry:
            return entry[entity]
        if key in entry:
            return entry[key]
        # Panel à une seule fréquence déclarée : elle vaut pour tout le monde
        if len(entry) == 1:
            return next(iter(entry.values()))
        return None

    # Méthode auxiliaire de lecture de la fréquence d'étape
    @staticmethod
    def _stage_frequency(stage_freq: StageFrequency, entity: EntityKey = ()) -> Optional[str]:
        """Read the stage frequency for one entity.

        Args:
            stage_freq: Stage frequency, scalar or entity -> frequency.
            entity: Entity key, ``()`` for a time series.

        Returns:
            The stage frequency of that entity, or None when the entity is
            absent from the mapping.
        """
        # Forme scalaire
        if not isinstance(stage_freq, Mapping):
            return stage_freq
        # Forme par entité : clé brute puis clé normalisée
        key = normalize_entity_key(entity)
        if entity in stage_freq:
            return stage_freq[entity]
        if key in stage_freq:
            return stage_freq[key]
        if len(stage_freq) == 1:
            return next(iter(stage_freq.values()))
        return None

    # Méthode de classification d'une covariable face à une grille
    def classify(
        self,
        column: str,
        stage_freq: StageFrequency,
        detected_frequencies: DetectedFrequencies,
        entity: EntityKey = (),
    ) -> MaterializationWay:
        """Classify one covariate against one stage grid.

        The four ranks of the precedence, applied in order, stopping at the
        first applicable one:

        - RANK 1, ``f_c >= f``: ``f_c == f`` -> ``'identity'``, ``f_c`` finer
          than ``f`` -> ``'aggregate'``, cells ``'observed'`` in both cases,
          the aggregation going through
          :meth:`FrequencyConverter.aggregate_to_lower_frequency` with
          ``method='sum'`` and ``full_periods_only=True``. That ``'sum'`` is
          fixed, not read from ``aggregation_constraint``: an exact
          aggregation of a finer covariate is additive by construction,
          and the parameter governs the rescaling of
          interpolated covariates only. An incomplete period yields NaN: that
          is a legitimate source of NaN, identical at fit and at predict, and
          it is never masked;
        - RANK 2, ``c`` already produced at the current stage ``f``
          -> ``'stage_model'``, its values read from the mirror. The trigger is
          the mere presence of cells produced at ``f``, whatever their origin:
          a covariate served by the fallback earlier in the same stage is
          materialized, and is read back from the mirror like any other;
        - RANK 3, ``c`` imputed at an earlier stage ``f'``
          -> ``'carried_model'``, those values carried onto the ``f`` grid;
        - RANK 4, none of the above -> ``covariate_fallback``.

        Ranks 2 and 3 read the stores, and are therefore reachable under
        ``covariate_strategy='model'`` only.

        Args:
            column: Covariate name.
            stage_freq: Frequency of the stage grid.
            detected_frequencies: Detected frequency of each column.
            entity: Entity the classification is made for, ``()`` for a time
                series.

        Returns:
            The materialization way of the covariate on that grid.

        Examples:
            >>> mat = CovariateMaterializer()
            >>> freqs = {'m1': 'M', 'q1': 'Q', 'a1': 'Y'}
            >>> mat.classify('q1', 'Q', freqs)
            'identity'
            >>> mat.classify('m1', 'Q', freqs)
            'aggregate'
            >>> mat.classify('a1', 'Q', freqs)
            'interpolate'
            >>> CovariateMaterializer(covariate_strategy='tolerate_nan').classify(
            ...     'a1', 'Q', freqs)
            'raw_anchors'
        """
        # Extraction de la fréquence de la colonne pour cette entité
        f_col = self._column_frequency(detected_frequencies, column, entity)
        # Extracyion de la fréquence de prédiction à cette étape
        f_stage = self._stage_frequency(stage_freq, entity)

        # Colonne sans fréquence détectée : ses observations telles quelles,
        # aucune conversion n'étant définissable
        if f_col is None or f_stage is None:
            return 'raw_anchors'

        # Rang 1 : identité
        if _same_frequency(f_col, f_stage):
            return 'identity'
        # Rang 1 : agrégation exacte d'une colonne plus fine
        if is_higher_frequency(f_col, f_stage):
            return 'aggregate'

        # Fréquence plus basse que la grille : gouverné par la stratégie.
        # Sortie avant toute lecture de registre sous 'tolerate_nan' et
        # 'interpolate' : les rangs 2 et 3 y sont structurellement
        # inatteignables, ce qui rend l'ordre d'imputation sans effet sur les
        # valeurs produites hors 'model'
        if self.covariate_strategy == 'tolerate_nan':
            return 'raw_anchors'
        if self.covariate_strategy == 'interpolate':
            return 'interpolate'

        # Stratégie 'model' : lecture des registres de l'entité
        block = self._mirror_blocks(column).get(normalize_entity_key(entity))

        # Rang 2 : la covariable a déjà été produite à cette étape. Le critère
        # est la présence de cellules au pas f, quelle que soit leur origine —
        # « le repli matérialise » : une covariable servie par repli plus tôt
        # dans l'étape est ensuite relue dans le miroir
        if block is not None and not self._stage_cells(block, f_stage).empty:
            return 'stage_model'

        # Rang 3 : report d'une imputation d'une étape antérieure
        if block is not None and self._carry_frequency(block, f_stage) is not None:
            return 'carried_model'

        # Rang 4 : approche secondaire, au fit comme au predict
        return self._fallback_way

    # Méthode auxiliaire de repli d'une voie inapplicable à une entité
    def _applicable_way(
        self,
        way: MaterializationWay,
        f_col: Optional[str],
        f_stage: Optional[str],
        mirror_block: Optional[pd.DataFrame] = None,
    ) -> MaterializationWay:
        """Degrade a column-level way to what one entity's state allows.

        The way is a property of the column, but
        a panel may carry the same column at different frequencies depending
        on the entity, and may have imputed it for some entities only. An
        entity that already observes the column at the stage frequency cannot
        be interpolated onto that grid; it falls back on its own rank-1 way.
        An entity whose mirror holds nothing at ``f`` cannot be served by rank
        2; it falls back on rank 3, then on ``covariate_fallback``. The
        invariant is therefore measured per entity.

        Args:
            way: Way retained for the column.
            f_col: Detected frequency of the column for this entity.
            f_stage: Stage frequency for this entity.
            mirror_block: This entity's mirror block, as returned by
                :meth:`_mirror_blocks`. None when the column has produced
                nothing for that entity.

        Returns:
            The way actually applicable to that entity.
        """
        # Voies du modèle : le rang 1 reste prioritaire, et le rang retenu
        # pour la colonne doit exister dans le miroir de cette entité
        if way in _MODEL_WAYS:
            # Rang 1 : la colonne est observée au pas de la grille, ou plus fine
            if _same_frequency(f_col, f_stage):
                return 'identity'
            if f_col is not None and f_stage is not None and is_higher_frequency(f_col, f_stage):
                return 'aggregate'

            # Disponibilité effective du miroir pour cette entité
            block = mirror_block
            has_stage = block is not None and not self._stage_cells(block, f_stage).empty
            carry = None if block is None else self._carry_frequency(block, f_stage)

            # Rang 2, puis rang 3, puis rang 4 ramené aux fréquences de l'entité
            if way == 'stage_model' and has_stage:
                return 'stage_model'
            if carry is not None:
                return 'carried_model'
            if has_stage:
                return 'stage_model'
            return self._applicable_way(self._fallback_way, f_col, f_stage)

        # Seule l'interpolation exige que la grille soit plus fine que la colonne
        if way != 'interpolate':
            return way
        if f_col is None or f_stage is None:
            return 'raw_anchors'
        if is_higher_frequency(f_stage, f_col):
            return 'interpolate'
        # Grille au plus aussi fine que la colonne : rang 1
        return 'identity' if _same_frequency(f_col, f_stage) else 'aggregate'

    # -------------------------------------------------------------------------
    # Interpolation (voie et repli)
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'interpolation d'un bloc d'entité
    def _interpolate_block(
        self,
        column: str,
        observations: pd.Series,
        dates: pd.DatetimeIndex,
        f_col: Optional[str],
        f_stage: Optional[str],
    ) -> pd.Series:
        """Interpolate one entity's observations onto one date grid.

        Args:
            column: Column being interpolated, driving ``resolve_method`` and
                ``resolve_anchor``.
            observations: Observed values of the column, NaN already dropped.
            dates: Target grid dates.
            f_col: Detected frequency of the column.
            f_stage: Frequency of the target grid.

        Returns:
            Values on ``dates``, NaN where the interpolation could produce
            nothing (series edges beyond what ``limit_direction`` allows).
        """
        return self._interpolate_series(column, observations, dates, f_col, f_stage)

    # Méthode auxiliaire d'interpolation d'une série quelconque
    def _interpolate_series(
        self,
        column: str,
        observations: pd.Series,
        dates: pd.DatetimeIndex,
        f_source: Optional[str],
        f_target: Optional[str],
    ) -> pd.Series:
        """Carry one date-indexed series onto a finer grid, by interpolation.

        The interpolation way of ``column`` — its method, its anchor, then the
        rescaling to the totals of ``f_source`` — applied to whatever series
        is handed over: the column's raw observations for the
        ``'interpolate'`` way, its mirror cells produced at the earlier stage
        ``f'`` for the ``'carried_model'`` way. The rescaling therefore bears
        on the totals of the origin stage, never on those of the current one.

        Args:
            column: Column being carried, driving ``resolve_method`` and
                ``resolve_anchor``.
            observations: Values to carry, date-indexed, NaN already dropped.
            dates: Target grid dates.
            f_source: Frequency the values were produced at.
            f_target: Frequency of the target grid.

        Returns:
            Values on ``dates``, NaN where the interpolation could produce
            nothing (series edges beyond what ``limit_direction`` allows).
        """
        # Série vide : la colonne reste NaN sur toute la grille
        if observations.empty:
            return pd.Series(np.nan, index=dates, name=column)

        # Interpolation à la fréquence de l'étape
        interpolated = self._conv.interpolate_to_higher_frequency(
            observations,
            f_target,
            method=self.resolve_method(column),
            source_freq=f_source,
            anchor_fraction=self.resolve_anchor(column),
        )

        # Recalage aux totaux de période de la fréquence D'ORIGINE : inerte
        # tant que l'objet n'est pas fourni
        applier = self.aggregation_constraint_applier
        if applier is not None and self.resolve_aggregation_constraint(column) is not None:
            interpolated, _rescaled_mask = applier.rescale(
                interpolated, observations, f_source, column
            )

        # Réindexation sur la grille demandée : les timestamps décalés par
        # anchor_fraction ne tombent pas sur la grille, seule celle-ci est retenue
        return interpolated.reindex(dates)

    # Méthode publique d'interpolation d'une colonne
    def interpolate_column(
        self,
        column: str,
        grid_index: pd.Index,
        stage_freq: StageFrequency,
        detected_frequencies: DetectedFrequencies,
        source_data: pd.DataFrame,
    ) -> pd.Series:
        """Interpolate one column onto a grid, and record it in the stores.

        This is the interpolation of the class:
        it serves the ``'interpolate'`` strategy, the
        ``covariate_fallback`` and the failure fallback of a
        model imputation, in every strategy.

        The fallback materializes: the values produced here feed the three
        stores exactly like a model prediction would, with origin
        ``'interpolated'``.

        Looking downstream: linear interpolation between two anchors
        uses the future anchor. That is not a defect for historical
        imputation, but it is information from the future in a
        pseudo-real-time setting; ``imputation_scope='extended_forward'``
        remains the dedicated mechanism for series ends.

        Args:
            column: Column to interpolate.
            grid_index: Target grid — a ``DatetimeIndex`` for a time series, a
                panel ``MultiIndex`` (entity levels then date) otherwise.
            stage_freq: Frequency of the grid, scalar or per entity.
            detected_frequencies: Detected frequency of each column.
            source_data: Data holding the column's observations, indexed like
                the input frame. Never modified.

        Returns:
            Values on ``grid_index``, NaN where nothing could be produced.

        Raises:
            KeyError: If ``column`` is absent from ``source_data``.
        """
        # Vérification que la colonne à interpoler est présente dans le jeu de données
        if column not in source_data.columns:
            raise KeyError(f"Column {column!r} missing from source_data")

        # Création du triplet d'interpolation à enregistrer
        values, origins, freqs = self._produce_column(
            column, 'interpolate', grid_index, stage_freq,
            detected_frequencies, source_data,
        )
        # Le repli matérialise : alimentation des trois registres
        self._record(column, values, origins, freqs)
        return values

    # -------------------------------------------------------------------------
    # Production d'une colonne, voie par voie
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'indexation des blocs de la source par entité
    @staticmethod
    def _source_blocks(source_data: pd.DataFrame) -> Dict[EntityKey, pd.DataFrame]:
        """Index the source data by entity, each block indexed by date only.

        Args:
            source_data: Input frame, time series or panel.

        Returns:
            Mapping entity key -> date-indexed block. A time series yields the
            single degenerate entity ``()``.
        """
        return {
            normalize_entity_key(entity): block
            for entity, _mask, block in iter_entity_blocks(source_data)
        }

    # Méthode auxiliaire de production d'une colonne sur toute la grille
    def _produce_column(
        self,
        column: str,
        way: MaterializationWay,
        grid_index: pd.Index,
        stage_freq: StageFrequency,
        detected_frequencies: DetectedFrequencies,
        source_data: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Produce one column on the whole grid, entity block by entity block.

        Args:
            column: Column to produce.
            way: Materialization way retained for the column.
            grid_index: Target grid index.
            stage_freq: Frequency of the grid, scalar or per entity.
            detected_frequencies: Detected frequency of each column.
            source_data: Input frame, never modified.

        Returns:
            Tuple ``(values, origins, production_freq)``, three Series indexed
            on ``grid_index``. ``origins`` and ``production_freq`` are None
            wherever no value could be produced. The production frequency of a
            produced cell is that of the stage, model ways included: the cell
            is materialized on the current stage's grid.
        """
        # Nombre de lignes
        n_rows = len(grid_index)

        # Initialisation du triplet résultat
        values = np.full(n_rows, np.nan, dtype=float)
        origins = np.full(n_rows, None, dtype=object)
        freqs = np.full(n_rows, None, dtype=object)

        # Découpage en blocs (correspondant à des entités)
        blocks = self._source_blocks(source_data)
        # Miroir de la colonne, lu une seule fois, et seulement si la voie
        # retenue peut en avoir besoin
        mirrors = self._mirror_blocks(column) if way in _MODEL_WAYS else {}

        # Parcours des blocs d'entité de la grille : c'est elle qui décide des
        # lignes à produire, la source ne fournissant que les observations
        grid_frame = pd.DataFrame(index=grid_index)
        for entity, mask, grid_block in iter_entity_blocks(grid_frame):
            # Normalisation de l'entité
            key = normalize_entity_key(entity)
            # Extraction des dates
            dates = grid_block.index
            # Extraction des données sources
            source_block = blocks.get(key)
            # Entité absente de la source : ses lignes restent NaN
            if source_block is None or column not in source_block.columns:
                continue
            # Extraction de la fréquence de la colonne
            f_col = self._column_frequency(detected_frequencies, column, key)
            # Extraction de la fréquence de prédiction à cette étape
            f_stage = self._stage_frequency(stage_freq, key)
            # Extraction du miroir de l'entité
            mirror_block = mirrors.get(key)
            # Extraction de la méthode de transformation appliquée
            entity_way = self._applicable_way(way, f_col, f_stage, mirror_block)

            # Création de la colonne transformée our l'entité
            produced, produced_origins = self._produce_block(
                column, entity_way, source_block[column], dates, f_col, f_stage,
                mirror_block,
            )

            # Report du bloc sur les lignes de l'entité dans la grille
            filled = produced.notna().to_numpy()
            values[mask] = produced.to_numpy(dtype=float)
            origins[mask] = produced_origins.to_numpy(dtype=object)
            freqs[mask] = np.where(filled, f_stage, None)

        return (
            pd.Series(values, index=grid_index, name=column),
            pd.Series(origins, index=grid_index, name=column),
            pd.Series(freqs, index=grid_index, name=column),
        )

    # Méthode auxiliaire de production d'un bloc d'entité
    def _produce_block(
        self,
        column: str,
        way: MaterializationWay,
        source: pd.Series,
        dates: pd.DatetimeIndex,
        f_col: Optional[str],
        f_stage: Optional[str],
        mirror_block: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Produce one entity's values for one column, by the retained way.

        Args:
            column: Column being produced.
            way: Way applicable to this entity.
            source: The column's values for this entity, date-indexed, NaN
                included.
            dates: Target grid dates of this entity.
            f_col: Detected frequency of the column for this entity.
            f_stage: Stage frequency for this entity.
            mirror_block: This entity's mirror block, as returned by
                :meth:`_mirror_blocks`. Only the model ways read it.

        Returns:
            Tuple ``(values, origins)``, both indexed by ``dates``, the origin
            being None wherever no value could be produced.
        """
        # Voies du modèle : les valeurs viennent du miroir, jamais de la source
        if way in _MODEL_WAYS:
            values, origins = self._produce_model_block(
                column, way, dates, f_stage, mirror_block
            )
        else:
            # Suppression des valeurs manquantes
            observations = source.dropna()

            # Identité et ancres brutes : les observations portées sur la grille.
            # Mécaniquement identiques, sémantiquement distinctes — 'identity'
            # est un rang 1 (la colonne est à la fréquence de la grille),
            # 'raw_anchors' est le rang 4 de 'tolerate_nan' (les ancres, et NaN
            # partout ailleurs)
            if way in ('identity', 'raw_anchors'):
                values = observations.reindex(dates)
            # Agrégation exacte d'une colonne plus fine, sur périodes complètes.
            # Une période incomplète produit NaN : source légitime, non masquée.
            # "method='sum'" en dur, sans lire "aggregation_constraint" : ce
            # rang suppose l'additivité par construction,
            # ce paramètre ne gouvernant que le recalage des interpolées
            elif way == 'aggregate':
                aggregated = self._conv.aggregate_to_lower_frequency(
                    source, f_stage, method='sum',
                    full_periods_only=True, source_freq=f_col,
                )
                values = aggregated.reindex(dates)
            # Interpolation des seules observations
            else:
                values = self._interpolate_block(
                    column, observations, dates, f_col, f_stage
                )
            # Origine constante, portée par la voie elle-même
            origins = pd.Series(_WAY_ORIGIN[way], index=dates, dtype=object)

        # Aucune origine là où aucune valeur n'a pu être produite
        origins = origins.where(values.notna(), other=None).astype(object)
        return values, origins

    # Méthode auxiliaire de production d'un bloc par une voie de modèle
    def _produce_model_block(
        self,
        column: str,
        way: MaterializationWay,
        dates: pd.DatetimeIndex,
        f_stage: Optional[str],
        mirror_block: Optional[pd.DataFrame],
    ) -> Tuple[pd.Series, pd.Series]:
        """Produce one entity's values from the mirror, ranks 2 and 3.

        Args:
            column: Column being produced.
            way: ``'stage_model'`` (rank 2) or ``'carried_model'`` (rank 3).
            dates: Target grid dates of this entity.
            f_stage: Stage frequency for this entity.
            mirror_block: This entity's mirror block, as returned by
                :meth:`_mirror_blocks`.

        Returns:
            Tuple ``(values, origins)``, both indexed by ``dates``. The origins
            are read in ``origin_store``: an imputation produced by fallback
            reports ``'interpolated'``, and carrying a model value onto a finer
            grid keeps it ``'model'`` — the way never decides the origin.
        """
        # Bloc absent : l'entité n'a rien dans le miroir, ses lignes restent NaN
        empty_values = pd.Series(np.nan, index=dates, name=column)
        empty_origins = pd.Series(None, index=dates, dtype=object)
        if mirror_block is None or mirror_block.empty:
            return empty_values, empty_origins

        # Rang 2 : les valeurs imputées à l'étape courante, telles quelles
        if way == 'stage_model':
            cells = self._stage_cells(mirror_block, f_stage)
            if cells.empty:
                return empty_values, empty_origins
            return (
                cells['value'].reindex(dates).rename(column),
                cells['origin'].reindex(dates).astype(object),
            )

        # Rang 3 : fréquence de production lue dans le registre, jamais redétectée
        f_prime = self._carry_frequency(mirror_block, f_stage)
        if f_prime is None:
            return empty_values, empty_origins

        # Report des cellules de l'étape d'origine sur la grille courante, par
        # la voie d'interpolation de la colonne et avec recalage aux totaux de f'
        cells = self._stage_cells(mirror_block, f_prime)
        values = self._interpolate_series(
            column, cells['value'].dropna(), dates, f_prime, f_stage
        )
        # Origine propagée sans dégradation ni amélioration : l'interpolation
        # d'une valeur de modèle reste de modèle, un repli reste 'interpolated'
        carried_origin = max_origin(cells['origin'].dropna().tolist())
        return values, pd.Series(carried_origin, index=dates, dtype=object)

    # Méthode auxiliaire d'enregistrement d'une colonne dans les registres
    def _record(
        self,
        column: str,
        values: pd.Series,
        origins: pd.Series,
        freqs: pd.Series,
    ) -> None:
        """Record one produced column in the three stores, NaN cells excluded.

        Args:
            column: Column produced.
            values: Values on the grid.
            origins: Origin of each cell, None where nothing was produced.
            freqs: Production frequency of each cell, same convention.
        """
        # Extraction des valeurs non vides
        produced = values.notna()
        # Ecriture dans les registres
        self._write_stores(
            column,
            values[produced],
            freqs[produced],
            origins[produced],
        )

    # -------------------------------------------------------------------------
    # Méthode unique et publique de production des features
    # -------------------------------------------------------------------------
    # Méthode de matérialisation des covariables sur une grille
    def materialize(
        self,
        *,
        columns: Sequence[str],
        grid_index: pd.Index,
        stage_freq: StageFrequency,
        detected_frequencies: DetectedFrequencies,
        source_data: pd.DataFrame,
        materialization: Optional[Mapping[str, MaterializationWay]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, MaterializationWay], Dict[str, CellOrigin]]:
        """Materialize covariates on one grid — the only producer of features.

        Two modes, and the second one is what makes the uniqueness rule of
        enforceable:

        - ``materialization=None``: the component chooses each column's way
          and returns it;
        - ``materialization`` given: the component replays the imposed ways
          without re-deciding anything. The caller chooses the way once, on
          the training grid, then imposes it on the prediction grid, at fit as
          well as at transform.

        No other public method returns a feature frame, and the replay mode is
        the only way to impose a way — under ``'model'`` included. That is
        what carries the uniqueness rule: if a covariate is served by
        ``covariate_fallback`` at predict, the version seen at fit is prepared
        by the same way, even when its anchors would suffice. Otherwise the
        model learns on the exact covariate and predicts on the interpolated
        one — the generalization of the central invariant from the NaN pattern
        to the nature of the values.

        Args:
            columns: Covariate columns to materialize, in output order.
            grid_index: Target grid — a ``DatetimeIndex`` for a time series, a
                panel ``MultiIndex`` (entity levels then date) otherwise.
                Window masks are applied by the caller, upstream, on this
                index.
            stage_freq: Frequency of the grid, scalar or per entity.
            detected_frequencies: Detected frequency of each column, scalar or
                per entity.
            source_data: Input data holding the observations. Never modified.
            materialization: Ways to replay, one entry per column of
                ``columns``. None to let the component choose.

        Returns:
            Tuple ``(features, ways, column_origins)``:

            - ``features``: frame indexed on ``grid_index``, one column per
              entry of ``columns``;
            - ``ways``: the way retained for each column, to be stored in the
              step and replayed as-is;
            - ``column_origins``: the AGGREGATED origin of each column, the
              max of the origins of the cells actually produced — the input of
              the ``covariate_taint`` computation. A column that
              produced nothing reports ``'observed'``.

        Raises:
            ValueError: If ``materialization`` does not cover exactly
                ``columns``.

        Examples:
            >>> dates = pd.date_range('2021-01-31', periods=12, freq='ME')
            >>> data = pd.DataFrame({'m1': range(12), 'a1': np.nan}, index=dates)
            >>> data.loc['2021-12-31', 'a1'] = 120.0
            >>> mat = CovariateMaterializer(covariate_strategy='tolerate_nan')
            >>> features, ways, origins = mat.materialize(
            ...     columns=['m1', 'a1'], grid_index=dates, stage_freq='M',
            ...     detected_frequencies={'m1': 'M', 'a1': 'Y'},
            ...     source_data=data,
            ... )
            >>> ways
            {'m1': 'identity', 'a1': 'raw_anchors'}
            >>> int(features['a1'].notna().sum())
            1
        """
        # Extraction des colonnes
        columns = tuple(columns)

        # Résolution des voies : choix, ou rejeu à l'identique
        ways = self._resolve_ways(
            columns, grid_index, stage_freq, detected_frequencies, materialization
        )

        # Initialisation du dictionnaire des features transformées
        features: Dict[str, pd.Series] = {}
        # Initialisation du dictionnaire des origines des données
        column_origins: Dict[str, CellOrigin] = {}

        # Production colonne par colonne, par la voie retenue
        for column in columns:
            # Production de la colonne
            values, origins, freqs = self._produce_column(
                column, ways[column], grid_index, stage_freq,
                detected_frequencies, source_data,
            )
            # Alimentation des trois registres, quelle que soit la voie : le
            # registre d'origines doit porter 'observed' pour les cellules
            # d'entrée et pour les agrégations exactes
            self._record(column, values, origins, freqs)

            # Ajout des valeurs transformées au jeu de données résultat
            features[column] = values
            # Origine agrégée : max des origines des cellules produites
            produced = origins.dropna().tolist()
            column_origins[column] = max_origin(produced) if produced else 'observed'

        # Construction du jeu de données résultat
        frame = pd.DataFrame(features, index=grid_index, columns=list(columns))
        return frame, ways, column_origins

    # Méthode auxiliaire de résolution des voies d'une matérialisation
    def _resolve_ways(
        self,
        columns: Tuple[str, ...],
        grid_index: pd.Index,
        stage_freq: StageFrequency,
        detected_frequencies: DetectedFrequencies,
        materialization: Optional[Mapping[str, MaterializationWay]],
    ) -> Dict[str, MaterializationWay]:
        """Choose or replay the materialization way of each column.

        The way is a property of the column, not of the entity (one
        entry per column of ``feature_cols``): the per-entity verdicts are
        reduced by taking the most degraded one, and :meth:`_applicable_way`
        then brings it back to what each entity's own frequencies allow.

        Args:
            columns: Columns to materialize.
            grid_index: Target grid index.
            stage_freq: Frequency of the grid.
            detected_frequencies: Detected frequency of each column.
            materialization: Ways to replay, or None to choose.

        Returns:
            Mapping column -> way.

        Raises:
            ValueError: If ``materialization`` does not cover exactly
                ``columns``.
        """
        # Mode rejeu : aucune décision n'est reprise
        if materialization is not None:
            provided = dict(materialization)
            # Vérification que les colonnes attendues sont bien présentes (sans manque et sans excès)
            missing = set(columns) - set(provided)
            extra = set(provided) - set(columns)
            if missing or extra:
                details = []
                if missing:
                    details.append(
                        f"Columns without a materialization path : {sorted(missing)}"
                    )
                if extra:
                    details.append(f"Orphaned materialization paths : {sorted(extra)}")
                raise ValueError(
                    "materialization doit couvrir exactement columns ; "
                    + " ; ".join(details)
                )
            return {column: provided[column] for column in columns}

        # Mode choix : décision déléguée, aucune production de valeurs
        return self.decide_ways(
            columns=columns,
            grid_index=grid_index,
            stage_freq=stage_freq,
            detected_frequencies=detected_frequencies,
        )

    # Méthode de décision de la précédence, isolée de toute production
    def decide_ways(
        self,
        *,
        columns: Sequence[str],
        grid_index: pd.Index,
        stage_freq: StageFrequency,
        detected_frequencies: DetectedFrequencies,
    ) -> Dict[str, MaterializationWay]:
        """Apply the four-rank precedence to each column, and nothing else.

        No value is produced, no store is written. :meth:`materialize` calls it when
        ``materialization`` is None, and replays the imposed ways otherwise.

        The way is a property of the column, not of the entity (one entry per
        column of ``feature_cols``): the per-entity verdicts are reduced by
        taking the most degraded one, and :meth:`_applicable_way` then brings
        it back to what each entity's own frequencies and mirror allow.

        Args:
            columns: Columns to classify.
            grid_index: Target grid — a ``DatetimeIndex`` for a time series, a
                panel ``MultiIndex`` (entity levels then date) otherwise.
            stage_freq: Frequency of the grid, scalar or per entity.
            detected_frequencies: Detected frequency of each column, scalar or
                per entity.

        Returns:
            Mapping column -> way, one entry per column of ``columns``.

        Examples:
            >>> dates = pd.date_range('2021-01-31', periods=12, freq='ME')
            >>> mat = CovariateMaterializer()
            >>> mat.decide_ways(
            ...     columns=['m1', 'q1', 'a1'], grid_index=dates,
            ...     stage_freq='M',
            ...     detected_frequencies={'m1': 'M', 'q1': 'Q', 'a1': 'Y'},
            ... )
            {'m1': 'identity', 'q1': 'interpolate', 'a1': 'interpolate'}
        """
        # Entités de la grille : c'est elle qui décide des verdicts à réduire
        grid_frame = pd.DataFrame(index=grid_index)
        entities = [
            normalize_entity_key(entity)
            for entity, _mask, _block in iter_entity_blocks(grid_frame)
        ]

        # Initalisation du dictionnaire résultat
        ways: Dict[str, MaterializationWay] = {}
        # Parcours des colonnes
        for column in columns:
            # Création des options possibles de transformation de la données en comparant les fréquences source et cible
            candidates = [
                self.classify(column, stage_freq, detected_frequencies, entity)
                for entity in entities
            ]
            # Choix de la méthode la plus conservatrice
            ways[column] = max(candidates, key=lambda way: _WAY_RANK[way])
        return ways

    # -------------------------------------------------------------------------
    # Éligibilité des covariables
    # -------------------------------------------------------------------------
    # Méthode de recensement des entités n'observant jamais une colonne
    def entities_without_column(
        self,
        column: str,
        data: pd.DataFrame,
    ) -> Tuple[EntityKey, ...]:
        """List the entities holding NO observation at all for one column.

        Args:
            column: Column name.
            data: Input data, time series or panel.

        Returns:
            Entity keys with zero non-NaN value for ``column``, in the data's
            entity order. Empty for a time series that observes the column.

        Raises:
            KeyError: If ``column`` is absent from ``data``.

        Examples:
            >>> dates = pd.date_range('2021-01-31', periods=2, freq='ME')
            >>> idx = pd.MultiIndex.from_product([['FR', 'IT'], dates])
            >>> df = pd.DataFrame({'c': [1.0, 2.0, np.nan, np.nan]}, index=idx)
            >>> CovariateMaterializer().entities_without_column('c', df)
            (('IT',),)
        """
        # Vérification que la colonne est bien présente dans le jeu de données
        if column not in data.columns:
            raise KeyError(f"Column {column!r} missing in 'data'")

        # Entités pour lesquelles la colonne est absente
        return tuple(
            normalize_entity_key(entity)
            for entity, _mask, block in iter_entity_blocks(data)
            if block[column].notna().sum() == 0
        )

    # Méthode de sélection des colonnes éligibles comme features
    def eligible_columns(
        self,
        columns: Iterable[str],
        data: pd.DataFrame,
    ) -> Tuple[str, ...]:
        """Select the columns admissible as features, per ``covariate_eligibility``.

        The parameter is recentred on the one case no strategy can repair: a
        feature with no observation at all for a whole entity.

        - ``'any_entity'`` (default): the column is kept as soon as one entity
          observes it. The empty entities' rows stay NaN and fall under the
          estimator's NaN contract — this is the unique source of residual NaN
          under ``covariate_strategy='interpolate'``;
        - ``'all_entities'``: the column is dropped from ``feature_cols`` as
          soon as one entity does not observe it, for estimators that do not
          tolerate NaN.

        The invariant is measured by entity.

        Args:
            columns: Candidate column names.
            data: Input data, time series or panel.

        Returns:
            The retained columns, in the order given. A column observed by no
            entity at all is never retained, whatever the modality.

        Examples:
            >>> dates = pd.date_range('2021-01-31', periods=2, freq='ME')
            >>> idx = pd.MultiIndex.from_product([['FR', 'IT'], dates])
            >>> df = pd.DataFrame({'c': [1.0, 2.0, np.nan, np.nan]}, index=idx)
            >>> CovariateMaterializer().eligible_columns(['c'], df)
            ('c',)
            >>> CovariateMaterializer(
            ...     covariate_eligibility='all_entities'
            ... ).eligible_columns(['c'], df)
            ()
        """
        # Décompte des entités du jeu, pour distinguer "aucune entité ne
        # l'observe" de "une entité ne l'observe pas"
        n_entities = sum(1 for _ in iter_entity_blocks(data))

        retained: List[str] = []
        for column in columns:
            # Colonne absente du frame : jamais retenue
            if column not in data.columns:
                continue
            empty = self.entities_without_column(column, data)
            # Colonne observée par aucune entité : jamais retenue
            if len(empty) == n_entities:
                continue
            # 'all_entities' écarte dès qu'une entité ne l'observe pas
            if empty and self.covariate_eligibility == 'all_entities':
                continue
            retained.append(column)

        return tuple(retained)

    # Représentation lisible du composant
    def __repr__(self) -> str:
        """Return a readable representation of the configuration.

        Returns:
            One-line representation naming the strategy, the fallback and the
            eligibility modality.
        """
        return (
            f"CovariateMaterializer(covariate_strategy={self.covariate_strategy!r}, "
            f"covariate_fallback={self.covariate_fallback!r}, "
            f"covariate_eligibility={self.covariate_eligibility!r})"
        )
