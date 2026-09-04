"""Aggregation constraint of one imputation stage, as an sklearn transformer.

``HighFrequencyImputer2`` predicts the sub-periods of a low-frequency variable
on a finer grid. Left free, those predictions have no reason to agree with the
observation they are supposed to describe: a year observed at 120 may end up
with twelve months summing to 112.5. The aggregation constraint closes that
gap — the sub-periods of a period are rescaled so that their aggregate equals
the observed total, which turns a free-floating prediction into a genuine
disaggregation of the observation.

This module honours the ``AggregationConstraintApplier``
protocol of :mod:`~tsforecast.frequency.covariate_materializer`, whose
injection point is the single place where the constraint enters the
materialization path.

Two returns matter to the caller, and they are different objects:

1. :meth:`AggregationConstraint.rescale` returns the rescaled values and the
   mask of the cells the constraint actually moved;
2. :meth:`AggregationConstraint.anchor_cells_mask` returns the anchor rows
   re-expressed at the stage frequency, i.e. the rows where an observation was
   overwritten by a sub-period value.

Provenance invariance: applying the constraint never changes the provenance of
a cell. A rescaled cell keeps the ``MODEL_*`` or ``INTERPOLATED`` mark it
carried before the rescaling, exactly as rescaling by ``StageScaler`` leaves
provenance untouched. There is no ``DISAGGREGATED`` provenance in ``hfi2``: it
would state nothing the cell's own provenance does not already state, and would
hide whether the value came from a model or from an interpolation. Both masks
above are therefore diagnostic — they say which cells moved and which rows held
an observation — and neither drives a marking.
"""
# Importation des modules
# Modules de base
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Tuple, Union
import warnings

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Normalisation des fréquences : seule la base est acceptée par "to_period"
from ..utils.frequency.utils import normalize_frequency
# Normalisation des clés d'entité du panel
from ..panel.utils import normalize_entity_key


# Contraintes reconnues, et réglage effectif d'une colonne ("None" ne recale rien)
ConstraintKind = Literal['sum', 'mean', 'last']
ConstraintSetting = Optional[ConstraintKind]
# Forme complète du paramètre public
AggregationConstraintSetting = Union[ConstraintSetting, Dict[str, ConstraintSetting]]

# Clé de repli du dictionnaire par colonne
DEFAULT_CONSTRAINT_KEY = '__default__'
# Contrainte retenue pour une colonne non couverte et sans clé de repli
DEFAULT_CONSTRAINT: ConstraintSetting = 'sum'

# Réglages admissibles, base des validations d'__init__
_CONSTRAINT_SETTINGS: Tuple[Any, ...] = ('sum', 'mean', 'last', None)


# Agrégateur de la contrainte 'last'
def _last(values: np.ndarray) -> float:
    """Return the last sub-period of a block.

    Args:
        values: Sub-period values of one period, in grid order.

    Returns:
        The last value of the block, as a float.

    Examples:
        >>> _last(np.array([9.0, 10.0, 12.0]))
        12.0
    """
    return float(values[-1])


# Correspondance contrainte -> agrégateur. Les trois contraintes partagent une
# formule unique — ratio = observé / agrégat, puis toutes les sous-périodes
# multipliées par ce ratio.
_AGGREGATORS: Dict[str, Callable[[np.ndarray], float]] = {
    'sum': lambda values: float(np.sum(values)),
    'mean': lambda values: float(np.mean(values)),
    'last': _last,
}


# Fonction auxiliaire d'appartenance des dates à leur période basse fréquence
def _period_membership(index: pd.Index, period_freq: str) -> pd.Series:
    """Map each row of a grid to the low-frequency period holding it.

    The reference frequency is always the detected frequency of the variable
    being imputed, never the frequency of the cascade stage: a yearly variable
    imputed at the quarter then at the month is rescaled on its yearly total at
    both stages.

    Args:
        index: Grid index — a ``DatetimeIndex`` for a time series, a
            ``MultiIndex`` ``(entity..., date)`` for a panel.
        period_freq: Frequency of the periods the observed totals refer to
            (any representation: ``'Q'``, ``'QS'``, ``'quarterly'``...).

    Returns:
        ``Series`` indexed like ``index`` holding a ``pd.Period`` for a time
        series and an ``(entity_tuple, Period)`` pair for a panel — two
        entities never share a rescaling block.

    Examples:
        >>> idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        >>> [str(period) for period in _period_membership(idx, 'Y')]
        ['2021', '2021']
    """
    # Normalisation en base de fréquence
    base = normalize_frequency(period_freq, return_format='base')

    # Cas des séries temporelles : la période suffit à identifier le bloc
    if not isinstance(index, pd.MultiIndex):
        periods = pd.DatetimeIndex(index).to_period(base)
        return pd.Series(list(periods), index=index, dtype=object)

    # Cas des données de panel : l'entité fait partie de la clé de bloc
    periods = index.get_level_values(-1).to_period(base)
    # Normalisation en tuple : "droplevel" rend des scalaires pour un panel
    # à un seul niveau d'entité
    entity_tuples = [normalize_entity_key(key) for key in index.droplevel(-1)]
    return pd.Series(list(zip(entity_tuples, periods)), index=index, dtype=object)


# Fonction auxiliaire de test d'une clé de bloc utilisable
def _is_usable_key(period_key: Any) -> bool:
    """Tell whether a period membership key designates a rescaling block.

    Args:
        period_key: One value of a ``_period_membership`` Series.

    Returns:
        False for the ``None`` / NaN markers of a row outside the
        disaggregation scope, True otherwise.

    Examples:
        >>> _is_usable_key(pd.Period('2021', 'Y-DEC'))
        True
        >>> _is_usable_key(float('nan'))
        False
    """
    # Le dtype object porte les absences en None ou en float('nan')
    return period_key is not None and not isinstance(period_key, float)


# Fonction de validation du paramètre public
def validate_aggregation_constraint(
    aggregation_constraint: AggregationConstraintSetting,
) -> None:
    """Check that ``aggregation_constraint`` is an admissible setting.

    Exposed at module level because two components carry the parameter —
    :class:`AggregationConstraint` and ``CovariateMaterializer`` — and a
    setting accepted by one must be accepted by the other. A single
    implementation is the only way to keep the two contracts from drifting.

    Args:
        aggregation_constraint: Value handed to ``__init__``: ``'sum'``,
            ``'mean'``, ``'last'``, None, or a dict mapping a column name to
            one of these values with an optional ``'__default__'`` key.

    Raises:
        ValueError: If the value, or one of the values of the dict form, is
            not admissible; or if the dict is empty.

    Examples:
        >>> validate_aggregation_constraint('mean')
        >>> validate_aggregation_constraint({'a1': 'sum', '__default__': None})
        >>> validate_aggregation_constraint('median')
        Traceback (most recent call last):
            ...
        ValueError: aggregation_constraint must be one of ('sum', 'mean', 'last', None), or a dict mapping a column name to one of these values with an optional '__default__' key, got 'median'
    """
    # Forme dictionnaire : chaque valeur doit être admissible
    if isinstance(aggregation_constraint, dict):
        if not aggregation_constraint:
            raise ValueError("aggregation_constraint dict cannot be empty")
        # Enumération des valeurs non-valides
        invalid = {
            key: value
            for key, value in aggregation_constraint.items()
            if value not in _CONSTRAINT_SETTINGS
        }
        if invalid:
            raise ValueError(
                f"aggregation_constraint values must be one of "
                f"{_CONSTRAINT_SETTINGS}, in a dict mapping a column name to "
                f"one of these values with an optional "
                f"{DEFAULT_CONSTRAINT_KEY!r} key, got {invalid}"
            )
        return

    # Forme scalaire
    if aggregation_constraint not in _CONSTRAINT_SETTINGS:
        raise ValueError(
            f"aggregation_constraint must be one of {_CONSTRAINT_SETTINGS}, or "
            f"a dict mapping a column name to one of these values with an "
            f"optional {DEFAULT_CONSTRAINT_KEY!r} key, got "
            f"{aggregation_constraint!r}"
        )


# Fonction de résolution du réglage effectif d'une colonne
def resolve_aggregation_constraint(
    aggregation_constraint: AggregationConstraintSetting,
    column: Optional[str] = None,
) -> ConstraintSetting:
    """Return the constraint effectively applied to one column.

    Module-level for the same reason as
    :func:`validate_aggregation_constraint`: the resolution rules of the dict
    form belong to the parameter, not to one of the components carrying it.

    Args:
        aggregation_constraint: Setting to resolve, scalar or dict.
        column: Column name, or None for the global setting.

    Returns:
        ``'sum'``, ``'mean'``, ``'last'`` or None: the column's own entry, else
        the ``'__default__'`` entry, else ``'sum'``.

    Examples:
        >>> resolve_aggregation_constraint({'a1': 'mean', '__default__': None}, 'a1')
        'mean'
        >>> resolve_aggregation_constraint({'a1': 'mean'}, 'q1')
        'sum'
    """
    # Forme scalaire : la même contrainte pour toutes les colonnes
    if not isinstance(aggregation_constraint, dict):
        return aggregation_constraint

    # Entrée propre à la colonne
    if column is not None and column in aggregation_constraint:
        return aggregation_constraint[column]
    # Repli explicite
    if DEFAULT_CONSTRAINT_KEY in aggregation_constraint:
        return aggregation_constraint[DEFAULT_CONSTRAINT_KEY]
    # Défaut du document
    return DEFAULT_CONSTRAINT


# Fonction de contrôle des clés du dictionnaire contre les colonnes réelles
def validate_constraint_columns(
    aggregation_constraint: AggregationConstraintSetting,
    columns: Iterable[str],
) -> None:
    """Check the dict keys against the columns actually present.

    Args:
        aggregation_constraint: Setting to check, scalar or dict.
        columns: Column names present in the data.

    Raises:
        ValueError: If the dict form names columns absent from ``columns``,
            listing the offending keys.

    Examples:
        >>> validate_constraint_columns({'a1': 'sum'}, ['a1', 'q1'])
        >>> validate_constraint_columns({'zz': 'sum'}, ['a1'])
        Traceback (most recent call last):
            ...
        ValueError: aggregation_constraint names unknown columns : ['zz']
    """
    # Seule la forme dictionnaire porte des clés à vérifier
    if not isinstance(aggregation_constraint, dict):
        return

    # La clé de repli n'est pas un nom de colonne
    known = set(columns)
    unknown = sorted(
        key for key in aggregation_constraint
        if key != DEFAULT_CONSTRAINT_KEY and key not in known
    )
    if unknown:
        raise ValueError(
            f"aggregation_constraint names unknown columns : {unknown}"
        )


# Classe de contrainte d'agrégation
class AggregationConstraint(BaseEstimator, TransformerMixin):
    """Rescale predicted sub-periods so each period matches its observed total.

    The sub-periods predicted for one period of a lower-frequency variable are
    multiplied by ``observed total / predicted aggregate``, so that the column
    carries a genuine disaggregation of the observation instead of a
    free-floating prediction. The aggregate is the one named by the constraint:
    the sum for an additive variable, the mean for a rate, the last sub-period for a stock. All
    three go through the same ratio, so the guards below hold identically for
    the three.

    Four guards:

    - a period only partly predicted (at least one NaN sub-period, typically a
      period straddling the edge of the grid) is left untouched, raw
      predictions kept;
    - a period holding no observation at all (delayed series end) is left
      untouched;
    - a period whose predicted aggregate is zero while its observed total is
      not is left untouched, the ratio being undefined;
    - a period whose predicted aggregate has the opposite sign of its observed
      total is rescaled — the constraint takes precedence — but every
      sub-period then flips sign, and a single aggregated ``UserWarning`` is
      emitted for the whole operation, naming the columns and the number of
      periods concerned. Never one warning per period.

    Anchor disaggregation, not parameterizable:

        A variable imputed at stage ``f`` is predicted over the whole of every
        period it covers, anchors included. The row that held the period total
        receives, like the others, a sub-period value. A column must never mix,
        at one imputation frequency, a period total and sub-period values: that
        heterogeneity makes the column unusable as a covariate, falsifies any
        downstream aggregation, and makes the scale of a row depend on its
        position within the period.

    Two consequences follow, and they are the price of the rule:

    - under ``'sum'``, no information is lost: the sub-periods sum back exactly
      to the observed total;
    - under ``None``, the observed total is overwritten by a free prediction.
      It stays recoverable in two ways, both of them supported: through
      ``inverse_transform`` of the imputer, and through the ``ORIGINAL`` mask
      of the source frequency level in the multi-frequency output
      (``keep_lower_frequencies=True``).

    In both cases the anchor row carries the provenance of the value now
    written there — the ``MODEL_*`` of the step, or ``INTERPOLATED`` — never
    ``ORIGINAL``, the observation no longer being what the cell holds, and
    never a mark of its own. :meth:`anchor_cells_mask` locates the overwritten
    observations, it does not qualify them, and it depends neither on
    ``aggregation_constraint`` nor on whether the rescaling succeeded.

    Provenance invariance:

        Applying the constraint never changes a provenance. A rescaled cell
        keeps the ``MODEL_*`` or ``INTERPOLATED`` mark it carried before the
        rescaling, exactly as it does under ``StageScaler``: the constraint
        moves a value, it does not produce it. There is no ``DISAGGREGATED``
        provenance in ``hfi2`` — it would state nothing the cell's own
        provenance does not already state, and would hide whether the value
        came from a model or from an interpolation.

    The component therefore never marks provenance, and neither of the two
    masks it returns is a provenance mask: they are diagnostic, and say which
    cells the constraint moved and which rows held an observation.

    Both a time series and a panel are served, and several columns at once: a
    ``DatetimeIndex`` or a ``MultiIndex`` ``(entity..., date)`` is accepted
    everywhere, masks come back as ``pd.Series`` on the same index, and two entities never share a rescaling block.

    Attributes:
        anchor_mask_: Anchor rows of the fit grid, per column — a boolean
            ``DataFrame``. Computed at :meth:`fit` because it depends neither
            on the values nor on the constraint.
        rescaled_mask_: Cells the last :meth:`transform` actually moved, per
            column — a boolean ``DataFrame``.
        n_features_in_: Number of columns seen at :meth:`fit`.
        feature_names_in_: Column names seen at :meth:`fit`.

    Args:
        aggregation_constraint: Constraint applied to the predicted
            sub-periods. ``'sum'`` (the default) rescales so their sum equals
            the observed total, ``'mean'`` so their mean does, ``'last'`` so
            the last sub-period does, and ``None`` applies no constraint at
            all. A dict ``{column: setting}`` sets the constraint per column,
            with an optional ``'__default__'`` key covering the rest.
        period_frequencies: Detected frequency of the periods the observed
            totals refer to, for :meth:`fit` / :meth:`transform`: a single
            frequency, or a ``{column: frequency}`` mapping.
        observations: Observed low-frequency values of the columns, at their
            own frequency (anchor rows non-null, the rest NaN), for
            :meth:`fit` / :meth:`transform`. Read from the untouched input
            frame, never from a stage frame.
        context: Label used in the aggregated warning messages, e.g.
            ``"'a1' at stage M"``.

    Raises:
        ValueError: If ``aggregation_constraint`` is not an admissible setting,
            or is an empty dict, or is a dict holding an inadmissible value.

    Examples:
        A year observed at 120 whose three predicted sub-periods sum to 100
        comes back scaled by 1.2, and sums to 120 exactly:

        >>> grid = pd.date_range('2021-01-31', periods=3, freq='ME')
        >>> values = pd.Series([20.0, 30.0, 50.0], index=grid)
        >>> observations = pd.Series([np.nan, np.nan, 120.0], index=grid)
        >>> constraint = AggregationConstraint('sum')
        >>> rescaled, mask = constraint.rescale(values, observations, 'Y')
        >>> rescaled.tolist()
        [24.0, 36.0, 60.0]
        >>> mask.tolist()
        [True, True, True]

        The anchor rows are located whatever the constraint, and carry no
        provenance of their own:

        >>> constraint.anchor_cells_mask(observations, grid).tolist()
        [False, False, True]
        >>> AggregationConstraint(None).anchor_cells_mask(observations, grid).tolist()
        [False, False, True]
    """

    # Initialisation : validation sans transformation
    def __init__(
        self,
        aggregation_constraint: AggregationConstraintSetting = DEFAULT_CONSTRAINT,
        period_frequencies: Optional[Union[str, Mapping[str, str]]] = None,
        observations: Optional[Union[pd.Series, pd.DataFrame]] = None,
        context: str = '',
    ) -> None:
        """Validate the configuration and store it untouched.

        Args:
            aggregation_constraint: Constraint applied (see class docstring).
            period_frequencies: Frequency of the periods of each column.
            observations: Observed low-frequency values of each column.
            context: Label used in the aggregated warning messages.

        Raises:
            ValueError: If ``aggregation_constraint`` is not admissible.
        """
        # Validation du réglage, clés du dict NON vérifiées ici : aucune colonne
        # n'est connue à l'initialisation (voir "validate_columns")
        self._validate_aggregation_constraint(aggregation_constraint)

        # Stockage des paramètres
        self.aggregation_constraint = aggregation_constraint
        self.period_frequencies = period_frequencies
        self.observations = observations
        self.context = context

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de validation du paramètre de contrainte
    @staticmethod
    def _validate_aggregation_constraint(
        aggregation_constraint: AggregationConstraintSetting,
    ) -> None:
        """Check that ``aggregation_constraint`` is an admissible setting.

        Args:
            aggregation_constraint: Value handed to ``__init__``.

        Raises:
            ValueError: If the value, or one of the values of the dict form, is
                not ``'sum'``, ``'mean'``, ``'last'`` or None; or if the dict is
                empty.

        Examples:
            >>> AggregationConstraint._validate_aggregation_constraint('mean')
            >>> AggregationConstraint._validate_aggregation_constraint('median')
            Traceback (most recent call last):
                ...
            ValueError: aggregation_constraint must be one of ('sum', 'mean', 'last', None), or a dict mapping a column name to one of these values with an optional '__default__' key, got 'median'
        """
        # Délégation à la fonction de module : "CovariateMaterializer" porte le
        # même paramètre et doit accepter exactement les mêmes formes
        validate_aggregation_constraint(aggregation_constraint)

    # Méthode de validation des clés du dictionnaire contre les colonnes réelles
    def validate_columns(self, columns: Iterable[str]) -> None:
        """Check the dict keys against the columns actually present.

        Called once by the imputer at ``fit``, with the full set of columns:
        the rescaling methods only ever see one stage's subset, against which a
        legitimate key of another variable would look unknown.

        Args:
            columns: Column names present in the data.

        Raises:
            ValueError: If the dict form names columns absent from ``columns``,
                listing the offending keys.

        Examples:
            >>> constraint = AggregationConstraint({'a1': 'sum'})
            >>> constraint.validate_columns(['a1', 'q1'])
            >>> AggregationConstraint({'zz': 'sum'}).validate_columns(['a1'])
            Traceback (most recent call last):
                ...
            ValueError: aggregation_constraint names unknown columns : ['zz']
        """
        validate_constraint_columns(self.aggregation_constraint, columns)

    # -------------------------------------------------------------------------
    # Résolution de la contrainte et des métadonnées
    # -------------------------------------------------------------------------
    # Méthode de résolution de la contrainte effective d'une colonne
    def resolve_constraint(self, column: Optional[str] = None) -> ConstraintSetting:
        """Return the constraint effectively applied to one column.

        Args:
            column: Column name, or None for the global setting.

        Returns:
            ``'sum'``, ``'mean'``, ``'last'`` or None: the column's own entry,
            else the ``'__default__'`` entry, else ``'sum'``.

        Examples:
            >>> constraint = AggregationConstraint(
            ...     {'a1': 'mean', '__default__': None}
            ... )
            >>> constraint.resolve_constraint('a1')
            'mean'
            >>> constraint.resolve_constraint('q1') is None
            True
        """
        return resolve_aggregation_constraint(self.aggregation_constraint, column)

    # Méthode auxiliaire de résolution de la fréquence de période d'une colonne
    def _period_freq_for(self, column: Optional[str]) -> str:
        """Resolve the period frequency configured for one column.

        Args:
            column: Column name, or None for the global setting.

        Returns:
            The frequency of the periods the column's totals refer to.

        Raises:
            ValueError: If ``period_frequencies`` is not configured, or does
                not cover the column.
        """
        # Dictionnaire des fréquences de chaque colonne
        binding = self.period_frequencies

        # Métadonnée indispensable
        if binding is None:
            raise ValueError(
                "AggregationConstraint requires period_frequencies to be set "
                "before fit"
            )

        # Forme scalaire : une seule fréquence pour toutes les colonnes
        if not isinstance(binding, Mapping):
            return binding

        # Dans le cas où la fréquence n'est pas la même pour l'ensemble des colonnes, on s'assure qu'elle est présente dans le dictionnaire 
        if column is None or column not in binding:
            raise ValueError(
                f"period_frequencies does not cover column {column!r}, got "
                f"keys {sorted(binding)}"
            )
        return binding[column]

    # Méthode auxiliaire d'extraction des observations d'une colonne
    def _observations_for(self, column: Optional[str]) -> pd.Series:
        """Extract the observed low-frequency values of one column.

        Args:
            column: Column name, or None when ``observations`` is a Series.

        Returns:
            The column's observations, anchors non-null and the rest NaN.

        Raises:
            ValueError: If ``observations`` is not configured, or does not
                cover the column.
        """
        # Series/Dataframe des observations pour chaque colonne
        observations = self.observations

        # Métadonnée indispensable
        if observations is None:
            raise ValueError(
                "AggregationConstraint requires observations to be set before fit"
            )

        # DataFrame : une colonne d'observations par variable imputée
        if isinstance(observations, pd.DataFrame):
            # Vérification que la colonne demandée est bien dans le jeu de données
            if column is None or column not in observations.columns:
                raise ValueError(
                    f"observations does not hold column {column!r}, got columns "
                    f"{list(observations.columns)}"
                )
            return observations[column]

        return observations

    # -------------------------------------------------------------------------
    # Recalage
    # -------------------------------------------------------------------------
    # Méthode auxiliaire des totaux observés par période
    @staticmethod
    def _observed_totals(
        observations: pd.Series,
        period_freq: str,
        aggregator: Callable[[np.ndarray], float],
    ) -> Dict[Any, float]:
        """Aggregate the observations of each period into its target total.

        Args:
            observations: Observed low-frequency values, anchors non-null.
            period_freq: Frequency of the periods.
            aggregator: Aggregator of the constraint, applied here too: a
                period carrying several observations is summed under ``'sum'``,
                averaged under ``'mean'``, and read on its last observation
                under ``'last'``. One observation per period is the norm; this
                only settles the degenerate case.

        Returns:
            Target total of each period, keyed like ``_period_membership``.
        """
        # Ancres seules : les lignes vides ne portent aucune contrainte
        observed = observations.dropna()
        if observed.empty:
            return {}

        # Regroupement positionnel des observations par période
        membership = _period_membership(observed.index, period_freq)
        # Initilaisation des blocks
        blocks: Dict[Any, List[float]] = {}
        # Population des blocks avec leur valeur et leur période d'appartenance
        for value, period_key in zip(observed.to_numpy(), membership.to_numpy()):
            if not _is_usable_key(period_key):
                continue
            blocks.setdefault(period_key, []).append(float(value))

        # Agrégation piur chaque période
        return {
            period_key: aggregator(np.asarray(values, dtype=float))
            for period_key, values in blocks.items()
        }

    # Méthode auxiliaire de recalage d'une colonne, sans avertissement
    def _rescale_one(
        self,
        values: pd.Series,
        observations: pd.Series,
        period_freq: str,
        kind: ConstraintKind,
    ) -> Tuple[pd.Series, pd.Series, List[Any], List[Any]]:
        """Rescale one column and report its degenerate periods.

        Split out of :meth:`rescale` so that the warnings can be aggregated
        over several columns: this method never warns, it only reports.

        Args:
            values: Sub-period values produced on the stage grid.
            observations: Observed low-frequency totals of the column.
            period_freq: Frequency of the periods the totals refer to.
            kind: Constraint applied, one of ``'sum'``, ``'mean'``, ``'last'``.

        Returns:
            Tuple ``(rescaled, rescaled_mask, zero_periods, flipped_periods)``:
            the rescaled values, the boolean mask of the cells actually moved,
            the periods whose predicted aggregate was zero against a non-zero
            total, and the periods whose profile was flipped.
        """
        # Extraction de l'agrégateur
        aggregator = _AGGREGATORS[kind]

        # Totaux visés, période par période
        period_totals = self._observed_totals(observations, period_freq, aggregator)

        # Initialisation du résultat et du masque des cellules recalées
        rescaled = values.copy()
        rescaled_mask = pd.Series(False, index=values.index)

        # Regroupement positionnel des lignes prédites par période
        membership = _period_membership(values.index, period_freq)
        rows_by_period: Dict[Any, List[int]] = {}
        for position, period_key in enumerate(membership.to_numpy()):
            if not _is_usable_key(period_key):
                continue
            rows_by_period.setdefault(period_key, []).append(position)

        # Recensement des cas dégénérés pour un avertissement agrégé
        zero_periods: List[Any] = []
        flipped_periods: List[Any] = []

        # Recalage : les prédictions de chaque période basse fréquence sont
        # ajustées pour que leur agrégat égale la valeur observée. La boucle
        # itère sur les totaux observés, ce qui réalise la garde des périodes
        # sans aucune observation : elles n'ont pas de clé ici.
        for period_key, period_value in period_totals.items():
            # Extraction des positions associées à la période
            positions = rows_by_period.get(period_key)
            if not positions:
                continue
            # Extraction des observations associées à ces positions
            block = values.iloc[positions]

            # Périodes partiellement prédites : aucune contrainte imposable
            if not block.notna().all():
                continue

            # Agrégation des observations sur la période
            aggregate = aggregator(block.to_numpy(dtype=float))

            # Agrégat nul : le ratio est indéfini, prédictions brutes conservées
            if aggregate == 0:
                if period_value != 0:
                    zero_periods.append(period_key)
                continue

            # Application du ratio de recalage
            rescaled.iloc[positions] = block * (period_value / aggregate)
            rescaled_mask.iloc[positions] = True

            # Signe opposé : la contrainte est imposée mais le profil est inversé
            if period_value != 0 and np.sign(aggregate) != np.sign(period_value):
                flipped_periods.append(period_key)

        return rescaled, rescaled_mask, zero_periods, flipped_periods

    # Méthode auxiliaire d'émission des avertissements agrégés
    def _emit_warnings(
        self,
        zero_periods: Mapping[str, List[Any]],
        flipped_periods: Mapping[str, List[Any]],
    ) -> None:
        """Emit at most one warning per degenerate case, for the whole operation.

        A period-by-period warning would drown the caller under one message per
        year of the panel: the two messages below name the columns and the
        number of periods concerned, and are emitted once.

        Args:
            zero_periods: Periods whose predicted aggregate was zero against a
                non-zero observed total, per column.
            flipped_periods: Periods whose profile was flipped, per column.
        """
        # Suffixe de contexte,
        suffix = f" for {self.context}" if self.context else ""

        # Recensement des colonnes réellement concernées
        zero_columns = sorted(
            str(column) for column, periods in zero_periods.items() if periods
        )
        flipped_columns = sorted(
            str(column) for column, periods in flipped_periods.items() if periods
        )

        # Warning sur les sommes nulles
        if zero_columns:
            count = sum(len(periods) for periods in zero_periods.values())
            example = next(periods[0] for periods in zero_periods.values() if periods)
            warnings.warn(
                f"Aggregation constraint skipped{suffix}: predictions aggregate "
                f"to zero for {count} period(s) with a non-zero observed total, "
                f"in column(s) {zero_columns} (e.g. {example}). Raw predictions "
                f"kept, their period totals do not match the observations.",
                UserWarning
            )
        # Warning sur les signes inversés
        if flipped_columns:
            count = sum(len(periods) for periods in flipped_periods.values())
            example = next(periods[0] for periods in flipped_periods.values() if periods)
            warnings.warn(
                f"Aggregation constraint flipped the predicted profile{suffix} "
                f"for {count} period(s), in column(s) {flipped_columns} (e.g. "
                f"{example}): the predictions aggregate to the opposite sign of "
                f"the observed total. Period totals are exact but every "
                f"sub-period changed sign.",
                UserWarning
            )

    # Méthode de recalage d'une colonne
    def rescale(
        self,
        values: pd.Series,
        observations: pd.Series,
        period_freq: str,
        column: Optional[str] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Rescale sub-period values so each complete period matches its total.

        Implementation of the ``AggregationConstraintApplier`` protocol of
        :mod:`~tsforecast.frequency.covariate_materializer`. The extra
        ``column`` argument is optional and only selects the constraint under
        the dict form, so the protocol signature is honoured as is.

        The four guards of the class docstring apply. A single aggregated
        ``UserWarning`` is emitted per degenerate case, never one per period.

        Args:
            values: Values produced on the stage grid, indexed by date for a
                time series or by ``(entity..., date)`` for a panel.
            observations: Observed low-frequency totals of the column, indexed
                by their own anchor rows.
            period_freq: Frequency of the periods the totals refer to.
            column: Name of the column, to resolve the constraint under the
                dict form. None reads the global setting.

        Returns:
            Tuple ``(rescaled, rescaled_mask)``: the rescaled values, and the
            boolean mask of the cells the constraint actually moved. That mask
            is diagnostic and drives no marking: moved or not, every cell keeps
            the ``MODEL_*`` or ``INTERPOLATED`` provenance it carried before
            the rescaling. Under a ``None`` constraint the values come back
            untouched and the mask is all-False.

        Examples:
            >>> grid = pd.date_range('2021-01-31', periods=3, freq='ME')
            >>> values = pd.Series([20.0, 30.0, 50.0], index=grid)
            >>> observations = pd.Series([np.nan, np.nan, 120.0], index=grid)
            >>> rescaled, mask = AggregationConstraint('sum').rescale(
            ...     values, observations, 'Y'
            ... )
            >>> float(rescaled.sum())
            120.0
            >>> rescaled, mask = AggregationConstraint(None).rescale(
            ...     values, observations, 'Y'
            ... )
            >>> rescaled.tolist(), bool(mask.any())
            ([20.0, 30.0, 50.0], False)
        """
        # Type d'agrégation pour la colonne
        kind = self.resolve_constraint(column)

        # Contrainte désactivée : valeurs brutes conservées, aucune cellule
        # recalée — la ligne d'ancre porte néanmoins une valeur de sous-période,
        # ce que dit "anchor_cells_mask" et non ce masque-ci
        if kind is None:
            return values, pd.Series(False, index=values.index)

        rescaled, mask, zero_periods, flipped_periods = self._rescale_one(
            values, observations, period_freq, kind
        )

        # Avertissements agrégés sur l'unique colonne traitée
        label = column if column is not None else values.name
        self._emit_warnings({label: zero_periods}, {label: flipped_periods})

        return rescaled, mask

    # -------------------------------------------------------------------------
    # Ancres
    # -------------------------------------------------------------------------
    # Méthode du masque des lignes d'ancre ré-exprimées à la fréquence d'étape
    def anchor_cells_mask(
        self,
        observations: Union[pd.Series, pd.DataFrame],
        index: pd.Index,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Locate the anchor rows re-expressed at the stage frequency.

        An anchor row is a row where the low-frequency variable carries a real
        observation in the untouched input frame. Re-expressed at the stage
        frequency, that row no longer holds the observation but a sub-period
        value: the mask says where an observed total was overwritten, which is
        what ``inverse_transform`` and the diagnostics need to find those
        totals again.

        It is not a provenance mask. The cell carries the provenance of the
        value now written there — the ``MODEL_*`` of the step, or
        ``INTERPOLATED`` — and its being an anchor position changes nothing to
        it. Being read off the data alone, the mask depends neither on
        ``aggregation_constraint`` nor on whether the rescaling succeeded.

        Args:
            observations: Observed low-frequency values, one Series for a
                single column or a DataFrame for several.
            index: Index of the cells that received a value — a
                ``DatetimeIndex`` or a ``MultiIndex`` ``(entity..., date)``.

        Returns:
            Boolean ``Series`` aligned on ``index`` for a Series input, boolean
            ``DataFrame`` indexed like ``index`` for a DataFrame input.

        Examples:
            >>> grid = pd.date_range('2021-01-31', periods=3, freq='ME')
            >>> observations = pd.Series([np.nan, np.nan, 120.0], index=grid)
            >>> AggregationConstraint('sum').anchor_cells_mask(
            ...     observations, grid
            ... ).tolist()
            [False, False, True]
        """
        # Même règle pour une Series et pour une trame : "reindex" ramène les
        # observations sur la grille d'étape, "notna" y désigne les ancres
        return observations.reindex(index).notna()

    # -------------------------------------------------------------------------
    # Protocole sklearn
    # -------------------------------------------------------------------------
    # Méthode d'ajustement : gel du masque des ancres de la grille
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AggregationConstraint':
        """Freeze the anchor mask of the stage grid.

        Nothing is learnt from the values: the anchor mask is a pure function
        of the observations and of the grid ``X`` runs on. The rescaling itself needs the values, so it
        belongs to :meth:`transform`.

        Args:
            X: Values produced on the stage grid, one column per imputed
                variable. Its columns select the variables and its index
                carries the grid.
            y: Ignored, present for the sklearn signature.

        Returns:
            The fitted constraint.

        Raises:
            ValueError: If ``observations`` or ``period_frequencies`` was not
                configured, or does not cover a column of ``X``.

        Examples:
            >>> grid = pd.date_range('2021-01-31', periods=3, freq='ME')
            >>> X = pd.DataFrame({'a1': [20.0, 30.0, 50.0]}, index=grid)
            >>> observations = pd.DataFrame({'a1': [np.nan, np.nan, 120.0]}, index=grid)
            >>> constraint = AggregationConstraint(
            ...     'sum', period_frequencies='Y', observations=observations
            ... ).fit(X)
            >>> constraint.anchor_mask_['a1'].tolist()
            [False, False, True]
        """
        # Extraction des colonnes
        columns = list(X.columns)
        # Contrôle des clés du dictionnaire contre les colonnes réelles
        self.validate_columns(columns)

        # Masque des ancres, colonne par colonne : la résolution passe par
        # "_observations_for", qui porte les messages d'erreur de métadonnée
        self.anchor_mask_ = pd.DataFrame(
            {
                column: self.anchor_cells_mask(self._observations_for(column), X.index)
                for column in columns
            },
            index=X.index,
        )

        # Contrôle anticipé des fréquences : une métadonnée absente est une
        # erreur de branchement, à signaler au fit et non au transform
        for column in columns:
            self._period_freq_for(column)

        # Attributs de convention sklearn
        self.n_features_in_ = len(columns)
        self.feature_names_in_ = np.asarray(columns, dtype=object)

        return self

    # Méthode de transformation : recalage de toutes les colonnes
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rescale every column of a stage frame, warning once for all of them.

        Args:
            X: Values produced on the stage grid, one column per imputed
                variable.

        Returns:
            The rescaled frame. :attr:`rescaled_mask_` holds the boolean mask
            of the cells actually moved, per column.

        Raises:
            NotFittedError: If the constraint was not fitted.

        Examples:
            >>> grid = pd.date_range('2021-01-31', periods=3, freq='ME')
            >>> X = pd.DataFrame({'a1': [20.0, 30.0, 50.0]}, index=grid)
            >>> observations = pd.DataFrame({'a1': [np.nan, np.nan, 120.0]}, index=grid)
            >>> constraint = AggregationConstraint(
            ...     'sum', period_frequencies='Y', observations=observations
            ... ).fit(X)
            >>> constraint.transform(X)['a1'].tolist()
            [24.0, 36.0, 60.0]
            >>> constraint.rescaled_mask_['a1'].tolist()
            [True, True, True]
        """
        # Vérification que le transformer est entraîné
        check_is_fitted(self)

        # Initialisation du résultat et des recensements
        rescaled = X.copy()
        masks: Dict[str, pd.Series] = {}
        zero_periods: Dict[str, List[Any]] = {}
        flipped_periods: Dict[str, List[Any]] = {}

        # Recalage colonne par colonne, sans avertir
        for column in X.columns:
            kind = self.resolve_constraint(column)

            # Contrainte désactivée sur cette colonne : valeurs brutes gardées
            if kind is None:
                masks[column] = pd.Series(False, index=X.index)
                continue

            # Recalage de la colonne
            column_values, mask, zeros, flipped = self._rescale_one(
                X[column],
                self._observations_for(column),
                self._period_freq_for(column),
                kind,
            )
            # Affectation positionnelle : un index dupliqué ne doit déclencher
            # aucun réalignement
            rescaled[column] = column_values.to_numpy()
            masks[column] = mask
            zero_periods[column] = zeros
            flipped_periods[column] = flipped

        # Avertissements agrégés : un seul message pour toute l'opération
        self._emit_warnings(zero_periods, flipped_periods)

        self.rescaled_mask_ = pd.DataFrame(masks, index=X.index)
        return rescaled
