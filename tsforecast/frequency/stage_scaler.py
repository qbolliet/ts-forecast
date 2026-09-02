"""Frequency scaling of one imputation stage, as a reversible sklearn transformer.

``HighFrequencyImputer2`` fits a model whose target is a low-frequency variable
(a yearly total) on a grid running at a higher frequency (months). Both sides of
that model must be expressed at the same scale, and the scale is a pure function
of the frequency pair — never of the data. This module holds that arithmetic.

Three divisors are governed by the scaling mode :

1. :meth:`StageScaler.covariate_divisors` — the covariates, at fit and at predict;
2. :meth:`StageScaler.target_divisor` — ``y``, scalar or per-row ;
3. :meth:`StageScaler.fit_scale_factor` — the factor baked into the model, which
   never moves once the stage is fitted.

The component follows the sklearn transformer protocol: :meth:`StageScaler.fit`
computes and stores those three divisors, :meth:`StageScaler.transform` divides
by them, :meth:`StageScaler.inverse_transform` multiplies back. The three
divisor methods stay usable on their own, without fitting: they are pure
functions of the configuration and of the frequencies handed to them, so a
single instance can serve every stage of a cascade.
"""
# Importation des modules
# Modules de base
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Protocole sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Arithmétique de fréquences : facteur constant et décompte calendaire
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import is_higher_frequency
# Normalisation des clés d'entité du panel
from ..panel.utils import normalize_entity_key


# Modalités de mise à l'échelle
ScaleMode = Literal['constant', 'calendar']
# Réglage effectif d'une colonne : "False" ne vaut que pour les FEATURES /!\ A revoir
ScaleSetting = Union[Literal[False], ScaleMode]
# Forme complète du paramètre public
ScaleFeatures = Union[Literal[False], ScaleMode, Dict[str, ScaleSetting]]

# Un diviseur est un scalaire d'étape ou une valeur par ligne
Divisor = Union[float, pd.Series]
# Clé d'entité normalisée : "()" en série temporelle
EntityKey = Tuple[str, ...]
# Fréquence unique, ou une fréquence par entité
FrequencyBinding = Union[str, Mapping[EntityKey, str]]

# Clé de repli du dictionnaire par feature, même convention que "estimator"
DEFAULT_SCALE_KEY = '__default__'
# Modalité retenue pour une colonne non couverte et sans clé de repli
DEFAULT_SCALE_MODE: ScaleMode = 'constant'

# Réglages admissibles, base des validations d'__init__
_SCALE_SETTINGS: Tuple[Any, ...] = (False, 'constant', 'calendar')

# Convertisseur partagé : sans état, une instance suffit à tout le module
_DEFAULT_CONVERTER = FrequencyConverter()


# Fonction auxiliaire d'extraction des dates d'un index plat ou MultiIndex
def _dates_of(index: pd.Index) -> pd.DatetimeIndex:
    """Extract the date level of a stage index.

    Args:
        index: Stage grid index — a ``DatetimeIndex`` for a time series, a
            ``MultiIndex`` ``(entity..., date)`` for a panel.

    Returns:
        Flat ``DatetimeIndex`` of the dates, positionally aligned on
        ``index``.

    Examples:
        >>> idx = pd.date_range('2021-01-31', periods=2, freq='ME')
        >>> _dates_of(idx).tolist() == idx.tolist()
        True
    """
    # "count_subperiods_per_period" n'accepte qu'un index plat : le niveau de
    # date est toujours le dernier, par convention du package
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(index.get_level_values(-1))
    return pd.DatetimeIndex(index)


# Fonction auxiliaire d'extraction des entités d'un index de panel
def _entities_of(index: pd.Index) -> Optional[List[EntityKey]]:
    """Extract the entity of each row of a stage index.

    Args:
        index: Stage grid index.

    Returns:
        One normalized entity tuple per row, or None when ``index`` carries no
        entity level (time series).
    """
    # Absence de niveau d'entité : série temporelle
    if not isinstance(index, pd.MultiIndex):
        return None
    return [normalize_entity_key(key) for key in index.droplevel(-1)]
    # /!\ Est ce que return index.droplevel(-1).map(normalize_entity_key).tolist() n'est pas une meilleure expression ?


# Classe de mise à l'échelle des données
class StageScaler(BaseEstimator, TransformerMixin):
    """Divide one imputation stage's data by its frequency scale factors.

    A low-frequency variable carries period totals (a yearly value of 120)
    while the stage predicts sub-periods (twelve months): the target is
    divided by the number of stage sub-periods held by one period of the
    variable, so the model directly predicts at the stage scale and its
    output is never multiplied back. Covariates are divided by their own
    divisor, which is neither systematically the stage factor nor
    systematically their own frequency.

    Two modes govern every divisor:

    - ``'constant'`` (default) divides by a constant factor per frequency
      pair, from :meth:`~FrequencyConverter.get_conversion_factor` (M -> Y =
      12, D -> M = 30.0). Suited to seasonally adjusted variables, where the
      average factor smooths what the adjustment already smoothed.
    - ``'calendar'`` divides by the real calendar count, from
      :meth:`~FrequencyConverter.count_subperiods_per_period` (February 28 or
      29, Q1 90 or 91). It produces one divisor per row. Suited to raw,
      non-adjusted variables, where the number of days carries signal.

    ``False`` puts no divisor on the features. It never spares the target:
    ``y`` is always scaled, whatever ``scale_features`` says. That is the
    contract of the current ``False`` of ``HighFrequencyImputer``, not an
    oversight — the model must predict at the stage scale in every mode.

    The transformer protocol is a thin layer over the divisor methods:
    :meth:`fit` stores the three divisors of one stage, :meth:`transform`
    divides, :meth:`inverse_transform` multiplies back. The divisor methods
    themselves need no fitting, and the instance keeps no state beyond its
    configuration: the baked factors belong to the ``ImputationStep``, not to
    the scaler, so one instance can serve every stage of a cascade.

    Attributes:
        feature_divisors_: Covariate divisors learnt by :meth:`fit` — a
            ``Series`` indexed by column name, or a ``DataFrame`` aligned on
            the fit grid. Used by :meth:`transform` on a ``DataFrame``.
        target_divisor_: Target divisor learnt by :meth:`fit`, scalar or
            per-row. Used by :meth:`transform` on a ``Series``.
        fit_scale_factor_: Factor baked into the model at fit time, to be
            frozen in ``ImputationStep.fit_scale_factor``. It differs from
            :attr:`target_divisor_` only when ``y`` mixes rows produced at
            several frequencies. Under ``'calendar'`` it is a
            ``Series`` frozen on the fit grid: a later scale carry-over on a
            different grid must realign it.
        n_features_in_: Number of columns seen at :meth:`fit`.
        feature_names_in_: Column names seen at :meth:`fit`.

    Args:
        scale_features: Scaling mode of the features. ``False``, one of
            ``'constant'`` / ``'calendar'``, or a dict ``{column: mode}`` with
            an optional ``'__default__'`` key (same convention as
            ``estimator``). Dict keys are column names. Columns neither covered
            nor defaulted fall back on ``'constant'``.
        source_freq: Detected frequency of the variable being imputed
            (``f_var``), for :meth:`fit`.
        pred_freq: Prediction frequency of the stage, for :meth:`fit`: a
            string, or an entity -> frequency mapping on a panel.
        column_frequencies: Detected frequency of each covariate, for
            :meth:`fit`: ``{column: frequency}``, each frequency being a
            string or an entity -> frequency mapping.
        target_column: Name of the imputed column, for :meth:`fit`. It selects
            the mode applied to ``y``: the mode of the imputed column, never
            that of the features.
        default_divisor: Divisor retained for a covariate whose frequency
            cannot be compared. ``HighFrequencyImputer`` passes the stage
            scale factor there.
        converter: Frequency converter to use. Defaults to a shared, stateless
            instance.

    Raises:
        ValueError: If ``scale_features`` is not an admissible setting, or is
            an empty dict, or a dict holding an inadmissible value.

    Examples:
        >>> scaler = StageScaler()
        >>> scaler.target_divisor('a1', source_freq='Y', pred_freq='M')
        12.0
        >>> scaler.resolve_mode('a1')
        'constant'
        >>> y = pd.Series([120.0], index=pd.to_datetime(['2021-12-31']))
        >>> scaler.apply(y, 12.0).tolist()
        [10.0]
    """

    # Initialisation : validation sans transformation
    def __init__(
        self,
        scale_features: ScaleFeatures = DEFAULT_SCALE_MODE,
        source_freq: Optional[str] = None,
        pred_freq: Optional[FrequencyBinding] = None,
        column_frequencies: Optional[Mapping[str, FrequencyBinding]] = None,
        target_column: Optional[str] = None,
        default_divisor: float = 1.0,
        converter: Optional[FrequencyConverter] = None,
    ) -> None:
        """Validate the configuration and store it untouched.

        Args:
            scale_features: Scaling mode of the features (see class docstring).
            source_freq: Detected frequency of the imputed variable.
            pred_freq: Prediction frequency of the stage.
            column_frequencies: Detected frequency of each covariate.
            target_column: Name of the imputed column.
            default_divisor: Divisor for covariates of unknown frequency.
            converter: Frequency converter to use.

        Raises:
            ValueError: If ``scale_features`` is not admissible.
        """
        # Validation de la modalité, clés du dict NON vérifiées ici : aucune
        # colonne n'est connue à l'initialisation (voir "validate_columns")
        self._validate_scale_features(scale_features)

        # Stockage des paramètres tels que reçus : "get_params"/"clone" de
        # sklearn exigent l'identité entre argument et attribut
        self.scale_features = scale_features
        self.source_freq = source_freq
        self.pred_freq = pred_freq
        self.column_frequencies = column_frequencies
        self.target_column = target_column
        self.default_divisor = default_divisor
        self.converter = converter

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de validation du paramètre de modalité
    @staticmethod
    def _validate_scale_features(scale_features: ScaleFeatures) -> None:
        """Check that ``scale_features`` is an admissible setting.

        Args:
            scale_features: Value handed to ``__init__``.

        Raises:
            ValueError: If the value, or one of the values of the dict form,
                is not ``False``, ``'constant'`` or ``'calendar'``; or if the
                dict is empty.
        """
        # Forme dictionnaire : chaque valeur doit être admissible
        if isinstance(scale_features, dict):
            if not scale_features:
                raise ValueError("scale_features dict cannot be empty")
            # Enumération des valeurs non-valides
            invalid = {
                key: value
                for key, value in scale_features.items()
                if value not in _SCALE_SETTINGS
            }
            if invalid:
                raise ValueError(
                    f"scale_features values must be one of {_SCALE_SETTINGS}, "
                    f"got {invalid}"
                )
            return

        # Forme scalaire
        if scale_features not in _SCALE_SETTINGS:
            raise ValueError(
                f"scale_features must be one of {_SCALE_SETTINGS} or a dict "
                f"of these values, got {scale_features!r}"
            )

    # Méthode de validation des clés du dictionnaire contre les colonnes réelles
    def validate_columns(self, columns: Iterable[str]) -> None:
        """Check the dict keys against the columns actually present.

        Called once by the imputer at ``fit``, with the full set of columns:
        the divisor methods only ever see one stage's subset, against which a
        legitimate key of another variable would look unknown.

        Args:
            columns: Column names present in the data.

        Raises:
            ValueError: If the dict form names columns absent from
                ``columns``, listing the offending keys.

        Examples:
            >>> scaler = StageScaler(scale_features={'m1': 'calendar'})
            >>> scaler.validate_columns(['m1', 'q1'])
            >>> StageScaler(scale_features={'zz': 'calendar'}).validate_columns(['m1'])
            Traceback (most recent call last):
                ...
            ValueError: scale_features names unknown columns : ['zz']
        """
        # Seule la forme dictionnaire porte des clés à vérifier
        if not isinstance(self.scale_features, dict):
            return

        # La clé de repli n'est pas un nom de colonne
        known = set(columns)
        unknown = sorted(
            key for key in self.scale_features
            if key != DEFAULT_SCALE_KEY and key not in known
        )
        if unknown:
            raise ValueError(
                f"scale_features names unknown columns : {unknown}"
            )

    # -------------------------------------------------------------------------
    # Résolution de la modalité
    # -------------------------------------------------------------------------
    # Méthode de résolution de la modalité effective d'une colonne
    def resolve_mode(self, column: Optional[str] = None) -> ScaleSetting:
        """Return the scaling mode effectively applied to one column.

        Args:
            column: Column name, or None for the global setting.

        Returns:
            ``False``, ``'constant'`` or ``'calendar'``: the column's own
            entry, else the ``'__default__'`` entry, else ``'constant'``.

        Examples:
            >>> scaler = StageScaler(
            ...     scale_features={'m1': 'calendar', '__default__': 'constant'}
            ... )
            >>> scaler.resolve_mode('m1')
            'calendar'
            >>> scaler.resolve_mode('q1')
            'constant'
        """
        setting = self.scale_features

        # Forme scalaire : la même modalité pour toutes les colonnes
        if not isinstance(setting, dict):
            return setting

        # Entrée propre à la colonne
        if column is not None and column in setting:
            return setting[column]
        # Repli explicite
        if DEFAULT_SCALE_KEY in setting:
            return setting[DEFAULT_SCALE_KEY]
        # Défaut du document
        return DEFAULT_SCALE_MODE

    # Méthode auxiliaire de résolution de la modalité de la CIBLE
    def _target_mode(self, column: Optional[str]) -> ScaleMode:
        """Return the mode applied to ``y`` for one imputed column.

        ``False`` disables the divisor of the FEATURES only: the target is
        always scaled, so ``False`` reads as ``'constant'`` here.

        Args:
            column: Name of the imputed column.

        Returns:
            ``'constant'`` or ``'calendar'``.
        """
        mode = self.resolve_mode(column)
        return DEFAULT_SCALE_MODE if mode is False else mode

    # -------------------------------------------------------------------------
    # Arithmétique des diviseurs
    # -------------------------------------------------------------------------
    # Propriété d'accès au convertisseur effectif
    @property
    def _conv(self) -> FrequencyConverter:
        """Frequency converter actually used.

        Returns:
            The injected converter, or the module-level shared instance.
        """
        return self.converter if self.converter is not None else _DEFAULT_CONVERTER

    # Méthode auxiliaire du diviseur d'un couple de fréquences
    def _pair_divisor(
        self,
        f_from: str,
        f_to: str,
        mode: ScaleMode,
        index: Optional[pd.Index] = None,
    ) -> Divisor:
        """Count the ``f_from`` sub-periods held by one ``f_to`` period.

        Args:
            f_from: Higher (finer) frequency being counted.
            f_to: Lower (coarser) frequency holding them.
            mode: ``'constant'`` for the invariant duration ratio,
                ``'calendar'`` for the real per-period count.
            index: Grid the count is spread over. Required under
                ``'calendar'``.

        Returns:
            A float under ``'constant'``; a ``Series`` of floats aligned on
            ``index`` under ``'calendar'``.

        Raises:
            ValueError: If ``'calendar'`` is asked for without an ``index``.

        Examples:
            >>> scaler = StageScaler()
            >>> scaler._pair_divisor('M', 'Y', 'constant')
            12.0
            >>> idx = pd.to_datetime(['2021-02-15', '2024-02-15'])
            >>> scaler._pair_divisor('D', 'M', 'calendar', idx).tolist()
            [28.0, 29.0]
        """
        # Décompte constant : ratio de durée, identique pour toutes les périodes
        if mode != 'calendar':
            return float(self._conv.get_conversion_factor(f_from, f_to))

        # Décompte calendaire : une valeur par ligne, donc un index obligatoire
        if index is None:
            raise ValueError(
                "scale_features='calendar' requires an index to spread the "
                f"per-row divisors over (frequency pair {f_from!r} -> {f_to!r})"
            )
        counts = self._conv.count_subperiods_per_period(
            _dates_of(index), low_freq=f_to, high_freq=f_from
        )
        return pd.Series(counts, index=index, dtype=float)

    # Méthode auxiliaire de résolution d'une fréquence pour une entité
    @staticmethod
    def _freq_for(binding: Optional[FrequencyBinding], entity: EntityKey) -> Optional[str]:
        """Resolve a frequency binding for one entity.

        Args:
            binding: A frequency string, or an entity -> frequency mapping.
            entity: Normalized entity key.

        Returns:
            The frequency of that entity, or None when the mapping does not
            cover it.
        """
        if isinstance(binding, Mapping):
            return binding.get(entity)
        return binding

    # Méthode auxiliaire de collecte des entités portées par les fréquences
    def _binding_entities(
        self,
        bindings: Iterable[Optional[FrequencyBinding]],
    ) -> List[EntityKey]:
        """Collect the entities named by a set of frequency bindings.

        Args:
            bindings: Frequency strings or entity -> frequency mappings.

        Returns:
            Sorted list of normalized entity keys, or ``[()]`` when no binding
            is entity-wise (time series or homogeneous panel).
        """
        # Initialisation de l'ensemble des entités
        entities = set()
        # Parcours des fréquences
        for binding in bindings:
            if isinstance(binding, Mapping):
                entities.update(normalize_entity_key(key) for key in binding)
        # Pseudo-entité unique : aucune ventilation à faire
        return sorted(entities) if entities else [()]

    # Méthode auxiliaire du diviseur d'une covariable
    def _covariate_divisor(
        self,
        f_col: Optional[str],
        f_var: str,
        pred_freq: str,
        mode: ScaleSetting,
        index: Optional[pd.Index],
        default: float,
    ) -> Divisor:
        """Divisor carrying one covariate to its prediction scale.

        A column never re-aggregated towards ``f_var`` already carries at fit
        time the scale it will carry at predict: its anchors keep the scale of
        ``f_col``, and its divisor is ``1.0``. Otherwise the divisor is the
        number of ``pred_freq`` sub-periods in one ``f_var`` period, where
        ``pred_freq`` is the frequency the column really carries on the stage
        grid: the stage frequency when it was aggregated onto it (covariate
        strictly finer than the stage), its own frequency otherwise.

        Args:
            f_col: Detected frequency of the covariate, None when unknown.
            f_var: Detected frequency of the variable being imputed.
            pred_freq: Prediction frequency of the stage, for one entity.
            mode: Scaling mode of that column. ``False`` yields ``1.0``.
            index: Grid the per-row counts are spread over.
            default: Divisor retained when the frequencies cannot be compared.

        Returns:
            The divisor, scalar or per-row.

        Raises:
            ValueError: If ``'calendar'`` is asked for without an ``index``.
        """
        # Modalité "False" : aucun diviseur sur les features
        if mode is False:
            return 1.0

        # Contrôle du prérequis d'index avant le filet d'exception, qui ne doit
        # attraper que l'incomparabilité des fréquences
        if mode == 'calendar' and index is None:
            raise ValueError(
                "scale_features='calendar' requires an index to spread the "
                "per-row covariate divisors over"
            )

        try:
            # Colonne jamais ré-agrégée vers f_var : elle porte déjà au fit
            # l'échelle qu'elle aura au predict
            if not is_higher_frequency(f_col, f_var):
                return 1.0

            # Fréquence réellement portée par la colonne sur la grille d'étape :
            # seul ce qui est strictement plus fin que l'étape y est agrégé
            f_stage = pred_freq if is_higher_frequency(f_col, pred_freq) else f_col

            return self._pair_divisor(f_stage, f_var, mode, index)
        except (ValueError, TypeError):
            # Fréquences incomparables (colonne sans fréquence détectée,
            # littéral inconnu) : repli documenté sur le diviseur par défaut
            return default

    # Méthode de calcul des diviseurs de toutes les covariables
    def covariate_divisors(
        self,
        columns: Sequence[str],
        column_frequencies: Mapping[str, FrequencyBinding],
        source_freq: str,
        pred_freq: FrequencyBinding,
        index: Optional[pd.Index] = None,
        default: Optional[float] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Compute the divisor carrying each covariate to its prediction scale.

        Applies the divisor calculation method column by column and entity by entity: a panel may
        carry the same column at different frequencies depending on the
        entity, in which case a single divisor per column cannot be right for
        every row.

        Args:
            columns: Covariate columns to compute a divisor for.
            column_frequencies: Detected frequency of each column: a string,
                or an entity -> frequency mapping.
            source_freq: Detected frequency of the variable being imputed
                (``f_var``).
            pred_freq: Prediction frequency of the stage: a string, or an
                entity -> frequency mapping.
            index: Stage grid. Required as soon as one column is in
                ``'calendar'`` mode, or the entities disagree on a frequency.
            default: Divisor for a column whose frequency cannot be compared.
                Defaults to ``default_divisor``.

        Returns:
            ``Series`` of floats indexed by column name when every divisor is
            a scalar and every entity agrees; ``DataFrame`` indexed like
            ``index`` and columned like ``columns`` otherwise. Both divide a
            feature frame directly: pandas aligns a ``Series`` on the columns
            and a ``DataFrame`` on both axes.

        Raises:
            ValueError: If per-row divisors are needed and ``index`` is None
                or carries no entity level.

        Examples:
            >>> scaler = StageScaler()
            >>> divisors = scaler.covariate_divisors(
            ...     columns=['m1', 'a2'],
            ...     column_frequencies={'m1': 'M', 'a2': 'Y'},
            ...     source_freq='Y',
            ...     pred_freq='M',
            ... )
            >>> divisors.to_dict()
            {'m1': 12.0, 'a2': 1.0}
        """
        # Diviseur de repli
        fallback = self.default_divisor if default is None else default

        # Entités concernées : celles nommées par une fréquence par entité
        entities = self._binding_entities(
            [pred_freq, *(column_frequencies.get(col) for col in columns)]
        )

        # Modalité par colonne : elle décide seule de la forme du retour
        modes = {column: self.resolve_mode(column) for column in columns}
        calendar_used = any(mode == 'calendar' for mode in modes.values())

        # Diviseurs par (entité, colonne)
        per_entity: Dict[EntityKey, Dict[str, Divisor]] = {}
        for entity in entities:
            pf = self._freq_for(pred_freq, entity)
            per_entity[entity] = {
                column: self._covariate_divisor(
                    self._freq_for(column_frequencies.get(column), entity),
                    source_freq,
                    pf,
                    modes[column],
                    index,
                    fallback,
                )
                for column in columns
            }

        # Forme compacte : tous les diviseurs sont scalaires et les entités
        # s'accordent — cas des séries temporelles et des panels homogènes
        rows = list(per_entity.values())
        if not calendar_used and all(row == rows[0] for row in rows[1:]):
            return pd.Series(rows[0], dtype=float)

        # Forme par ligne : le diviseur dépend de la date, de l'entité, ou des
        # deux, et aucune Series indexée sur une seule dimension ne le porte
        if index is None:
            raise ValueError(
                "Per-row covariate divisors require an index ('calendar' mode "
                "or entities disagreeing on a column frequency)"
            )
        return self._spread(per_entity, columns, index)

    # Méthode auxiliaire de ventilation des diviseurs sur la grille
    def _spread(
        self,
        per_entity: Mapping[EntityKey, Mapping[str, Divisor]],
        columns: Sequence[str],
        index: pd.Index,
    ) -> pd.DataFrame:
        """Spread per-(entity, column) divisors over a stage grid.

        Args:
            per_entity: Divisor of each column, per entity.
            columns: Columns of the output, in order.
            index: Stage grid.

        Returns:
            ``DataFrame`` of floats indexed like ``index``, columned like
            ``columns``.

        Raises:
            ValueError: If several entities are involved and ``index`` carries
                no entity level.
        """
        # Extraction des entités
        entities_per_row = _entities_of(index)
        multi_entity = len(per_entity) > 1

        # Cohérence de l'index avec la ventilation demandée
        if multi_entity and entities_per_row is None:
            raise ValueError(
                "Entity-wise divisors require a MultiIndex (entity..., date), "
                f"got a {type(index).__name__}"
            )

        # Initialisation du DataFrame résultat
        result = pd.DataFrame(index=index, columns=list(columns), dtype=float)
        # Population du DataFrame
        # Parcours des colonnes
        for column in columns:
            # Entité unique : le diviseur couvre toute la grille
            if not multi_entity:
                divisor = next(iter(per_entity.values()))[column]
                result[column] = (
                    divisor if np.isscalar(divisor) else np.asarray(divisor, dtype=float)
                )
                continue

            # Ventilation par entité : comparaison terme à terme plutôt que par
            # un Index de tuples, que pandas diffuserait élément par élément
            for entity, divisors in per_entity.items():
                # Masque correspondant à l'entité
                mask = np.array(
                    [row_entity == entity for row_entity in entities_per_row],
                    dtype=bool,
                )
                if not mask.any():
                    continue
                # Imputation du diviseur correspondant à la colonne pour cette entité
                divisor = divisors[column]
                result.loc[mask, column] = (
                    divisor
                    if np.isscalar(divisor)
                    else np.asarray(divisor, dtype=float)[mask]
                )
        return result

    # Méthode auxiliaire du diviseur d'étape, scalaire ou ventilé par entité
    def _stage_divisor(
        self,
        source_freq: str,
        pred_freq: FrequencyBinding,
        mode: ScaleMode,
        index: Optional[pd.Index],
    ) -> Divisor:
        """Count the stage sub-periods held by one period of the variable.

        Args:
            source_freq: Detected frequency of the variable.
            pred_freq: Prediction frequency of the stage: a string, or an
                entity -> frequency mapping.
            mode: Scaling mode applied to the target.
            index: Stage grid, required under ``'calendar'`` or when the
                entities disagree.

        Returns:
            A float, or a ``Series`` aligned on ``index``.

        Raises:
            ValueError: If per-row divisors are needed and ``index`` is None.
        """
        # Fréquences d'étape distinctes réellement portées
        if isinstance(pred_freq, Mapping):
            distinct = sorted(set(pred_freq.values()))
        else:
            distinct = [pred_freq]

        # Étape homogène : un seul couple de fréquences
        if len(distinct) == 1:
            return self._pair_divisor(distinct[0], source_freq, mode, index)

        # Étape hétérogène : un diviseur par entité, donc par ligne
        if index is None:
            raise ValueError(
                "Entity-wise prediction frequencies require an index to spread "
                f"the per-row divisors over, got frequencies {distinct}"
            )
        entities = self._binding_entities([pred_freq])
        per_entity = {
            entity: {
                '_y': self._pair_divisor(
                    self._freq_for(pred_freq, entity), source_freq, mode, index
                )
            }
            for entity in entities
        }
        return self._spread(per_entity, ['_y'], index)['_y'].rename(None)

    # Méthode du diviseur de la cible
    def target_divisor(
        self,
        column: Optional[str],
        source_freq: str,
        pred_freq: FrequencyBinding,
        index: Optional[pd.Index] = None,
        produced_freq: Optional[pd.Series] = None,
    ) -> Divisor:
        """Compute the divisor of ``y`` for one imputed column.

        The mode applied is that of the imputed column, never that of the
        features, and ``False`` reads as ``'constant'``: the target is always
        scaled.

        When ``produced_freq`` is given, each row of ``y`` is divided by the
        divisor of the frequency AT WHICH IT WAS PRODUCED (read from
        ``imputed_freq_store``), not by the stage scalar: a yearly variable
        already imputed at the quarterly stage carries quarters on those rows,
        and mixing them raw with its yearly anchors would teach the model a
        blend of two scales.

        Args:
            column: Name of the imputed column, for mode resolution.
            source_freq: Detected frequency of the variable.
            pred_freq: Prediction frequency of the stage.
            index: Stage grid, required under ``'calendar'`` or when the
                entities disagree.
            produced_freq: Frequency at which each row of ``y`` was produced,
                indexed like ``y``. When given, the result is ALWAYS a
                ``Series``, even if every row shares the same frequency.

        Returns:
            A float, or a ``Series`` of per-row divisors.

        Raises:
            ValueError: If per-row divisors are needed and no index is
                available.

        Examples:
            Stage ``M``, variable ``a1``, mode ``'constant'``:

            >>> scaler = StageScaler()
            >>> y = pd.Series(
            ...     [120.0, 28.0, 30.0],
            ...     index=pd.to_datetime(['2021-12-31', '2021-03-31', '2021-06-30']),
            ... )
            >>> produced = pd.Series(['Y', 'Q', 'Q'], index=y.index)
            >>> divisors = scaler.target_divisor(
            ...     'a1', source_freq='Y', pred_freq='M', produced_freq=produced
            ... )
            >>> divisors.tolist()
            [12.0, 3.0, 3.0]
            >>> [round(value, 2) for value in scaler.apply(y, divisors)]
            [10.0, 9.33, 10.0]
        """
        # Extraction du mode de mise à l'échelle de la cible
        mode = self._target_mode(column)

        # Toutes les lignes produites à la fréquence de la variable : scalaire
        # d'étape, ou Series calendaire
        if produced_freq is None:
            return self._stage_divisor(source_freq, pred_freq, mode, index)

        # Échelle par ligne : la fréquence de production remplace source_freq
        rows = produced_freq.index
        result = pd.Series(np.nan, index=rows, dtype=float)

        # Fréquence d'étape de chaque ligne : constante, ou lue par entité
        entities_per_row = _entities_of(rows)
        if isinstance(pred_freq, Mapping):
            if entities_per_row is None:
                raise ValueError(
                    "Entity-wise prediction frequencies require a MultiIndex "
                    f"(entity..., date), got a {type(rows).__name__}"
                )
            stage_per_row = [self._freq_for(pred_freq, e) for e in entities_per_row]
        else:
            stage_per_row = [pred_freq] * len(rows)

        # Regroupement par couple (fréquence d'étape, fréquence de production) :
        # un seul appel au convertisseur par couple distinct
        pairs = pd.DataFrame(
            {'stage': stage_per_row, 'produced': produced_freq.to_numpy()},
            index=rows,
        )
        for (f_stage, f_row), group in pairs.groupby(['stage', 'produced'], sort=False):
            divisor = self._pair_divisor(f_stage, f_row, mode, group.index)
            result.loc[group.index] = (
                divisor if np.isscalar(divisor) else np.asarray(divisor, dtype=float)
            )
        return result

    # Méthode du facteur cuit dans le modèle
    def fit_scale_factor(
        self,
        column: Optional[str],
        source_freq: str,
        pred_freq: FrequencyBinding,
        index: Optional[pd.Index] = None,
    ) -> Divisor:
        """Compute the scale factor baked into the model at fit time.

        This is the factor frozen in ``ImputationStep.fit_scale_factor``: it
        never moves once the stage is fitted, and it governs the scale
        carry-over of the predictions. It coincides with
        :meth:`target_divisor` without ``produced_freq`` — the stage scalar —
        and is exposed separately because its LIFETIME differs: the target
        divisor describes one training set, this one describes the model.

        Under ``'calendar'`` the factor is a ``Series`` frozen on the fit
        grid: a carry-over computed on another grid must realign it.

        Args:
            column: Name of the imputed column, for mode resolution.
            source_freq: Detected frequency of the variable.
            pred_freq: Prediction frequency of the stage.
            index: Prediction grid of the fit, required under ``'calendar'``
                or when the entities disagree.

        Returns:
            A float, or a ``Series`` aligned on ``index``.

        Raises:
            ValueError: If per-row divisors are needed and ``index`` is None.

        Examples:
            >>> StageScaler().fit_scale_factor('a1', source_freq='Y', pred_freq='M')
            12.0
        """
        return self._stage_divisor(
            source_freq, pred_freq, self._target_mode(column), index
        )

    # -------------------------------------------------------------------------
    # Application et inversion de l'échelle
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de test du court-circuit
    @staticmethod
    def _is_identity(divisor: Union[Divisor, pd.DataFrame]) -> bool:
        """Tell whether a divisor can be skipped altogether.

        Only a SCALAR strictly equal to ``1.0`` may short-circuit the scaling.
        A ``Series`` or a ``DataFrame`` whose values all happen to be ``1.0``
        must still go through the scaling path: skipping it would leave the
        other divisors of the same stage applied and the scales of two
        variables of one stage would diverge (defect B12). This is the single
        place where the question is asked.

        Args:
            divisor: Divisor to test.

        Returns:
            True only for a scalar equal to ``1.0``.

        Examples:
            >>> StageScaler._is_identity(1.0)
            True
            >>> StageScaler._is_identity(pd.Series([1.0, 1.0]))
            False
        """
        return bool(np.isscalar(divisor)) and float(divisor) == 1.0

    # Méthode d'application de l'échelle
    def apply(
        self,
        values: Union[pd.Series, pd.DataFrame],
        divisor: Union[Divisor, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Divide values by their divisor.

        A ``Series`` divisor is read the way pandas reads it: column-wise
        against a ``DataFrame``, row-wise against a ``Series``.

        Args:
            values: Data to scale — a feature frame or a target series.
            divisor: Scalar, per-row ``Series``, per-column ``Series`` or
                ``DataFrame`` of divisors.

        Returns:
            The scaled data. The input object itself when ``divisor`` is the
            scalar ``1.0``.

        Examples:
            >>> scaler = StageScaler()
            >>> y = pd.Series([120.0, 132.0])
            >>> scaler.apply(y, 12.0).tolist()
            [10.0, 11.0]
        """
        if self._is_identity(divisor):
            return values
        return values / divisor

    # Méthode d'inversion de l'échelle
    def invert(
        self,
        values: Union[pd.Series, pd.DataFrame],
        divisor: Union[Divisor, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Multiply values back by their divisor.

        Args:
            values: Scaled data.
            divisor: The divisor that was applied.

        Returns:
            The unscaled data. The input object itself when ``divisor`` is the
            scalar ``1.0``.

        Examples:
            >>> scaler = StageScaler()
            >>> y = pd.Series([10.0, 11.0])
            >>> scaler.invert(y, 12.0).tolist()
            [120.0, 132.0]
        """
        if self._is_identity(divisor):
            return values
        return values * divisor

    # -------------------------------------------------------------------------
    # Protocole sklearn
    # -------------------------------------------------------------------------
    # Méthode d'ajustement : calcul et gel des trois diviseurs de l'étape
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        produced_freq: Optional[pd.Series] = None,
    ) -> 'StageScaler':
        """Compute the three divisors of one stage.

        Nothing is learnt from the values: the divisors are a pure function of
        the configured frequencies and of the grid ``X`` runs on. Fitting only
        freezes them, so that ``transform`` and ``inverse_transform`` need no
        further metadata and the imputer can bake
        :attr:`fit_scale_factor_` into its ``ImputationStep``.

        Args:
            X: Feature frame of the stage. Its columns select the covariates
                and its index carries the grid the per-row divisors run on.
            y: Target of the stage. Only its index is read, to carry the
                per-row target divisor; defaults to the index of ``X``.
            produced_freq: Frequency at which each row of ``y`` was produced,
                indexed like ``y``. None when every row carries the
                variable's own frequency.

        Returns:
            The fitted scaler.

        Raises:
            ValueError: If ``source_freq`` or ``pred_freq`` was not configured,
                or if per-row divisors are needed without a usable index.

        Examples:
            >>> X = pd.DataFrame(
            ...     {'m1': [100.0, 101.0]},
            ...     index=pd.to_datetime(['2021-11-30', '2021-12-31']),
            ... )
            >>> scaler = StageScaler(
            ...     source_freq='Y', pred_freq='M', column_frequencies={'m1': 'M'},
            ...     target_column='a1',
            ... ).fit(X)
            >>> scaler.fit_scale_factor_
            12.0
            >>> scaler.feature_divisors_.to_dict()
            {'m1': 12.0}
        """
        # Contrôle des métadonnées indispensables : elles sont des paramètres
        # d'initialisation, pas des données, et leur absence est une erreur
        # de branchement, pas un cas limite de données
        if self.source_freq is None or self.pred_freq is None:
            raise ValueError(
                "StageScaler requires source_freq and pred_freq to be set "
                f"before fit, got source_freq={self.source_freq!r} and "
                f"pred_freq={self.pred_freq!r}"
            )

        # Extraction des colonnes
        columns = list(X.columns)
        # Fréquences associées à chaque colonne
        frequencies = self.column_frequencies or {}

        # Diviseurs des covariables, sur la grille de l'étape
        self.feature_divisors_ = self.covariate_divisors(
            columns, frequencies, self.source_freq, self.pred_freq, index=X.index
        )

        # Grille de la cible : celle de y quand elle est fournie
        target_index = y.index if y is not None else X.index

        # Facteur cuit dans le modèle, puis diviseur effectif de y
        self.fit_scale_factor_ = self.fit_scale_factor(
            self.target_column, self.source_freq, self.pred_freq, index=target_index
        )
        self.target_divisor_ = self.target_divisor(
            self.target_column,
            self.source_freq,
            self.pred_freq,
            index=target_index,
            produced_freq=produced_freq,
        )

        # Attributs de convention sklearn
        self.n_features_in_ = len(columns)
        self.feature_names_in_ = np.asarray(columns, dtype=object)

        return self

    # Méthode auxiliaire de sélection du diviseur ajusté
    def _fitted_divisor(
        self,
        values: Union[pd.Series, pd.DataFrame],
    ) -> Union[Divisor, pd.DataFrame]:
        """Select the fitted divisor matching the shape of ``values``.

        Args:
            values: Data to scale.

        Returns:
            :attr:`feature_divisors_` for a ``DataFrame``,
            :attr:`target_divisor_` for a ``Series``.

        Raises:
            TypeError: If ``values`` is neither a ``Series`` nor a
                ``DataFrame``.
        """
        # Une trame est un jeu de covariables, une série est la cible : le
        # composant sert les deux, et seul le type les distingue
        if isinstance(values, pd.DataFrame):
            return self.feature_divisors_
        if isinstance(values, pd.Series):
            return self.target_divisor_
        raise TypeError(
            "StageScaler transforms a DataFrame of covariates or a Series "
            f"target, got {type(values).__name__}"
        )

    # Méthode de transformation : division par les diviseurs ajustés
    def transform(
        self,
        X: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Scale a feature frame or a target series to the stage scale.

        Delegates to :meth:`apply` with the fitted divisor matching the shape
        of ``X``: :attr:`feature_divisors_` for a ``DataFrame``,
        :attr:`target_divisor_` for a ``Series``.

        Args:
            X: Feature frame or target series of the stage.

        Returns:
            The scaled data.

        Raises:
            NotFittedError: If the scaler was not fitted.
            TypeError: If ``X`` is neither a ``Series`` nor a ``DataFrame``.

        Examples:
            >>> X = pd.DataFrame({'m1': [1200.0]}, index=pd.to_datetime(['2021-12-31']))
            >>> scaler = StageScaler(
            ...     source_freq='Y', pred_freq='M', column_frequencies={'m1': 'M'},
            ...     target_column='a1',
            ... ).fit(X)
            >>> scaler.transform(X)['m1'].tolist()
            [100.0]
        """
        check_is_fitted(self)
        return self.apply(X, self._fitted_divisor(X))

    # Méthode d'inversion : multiplication par les diviseurs ajustés
    def inverse_transform(
        self,
        X: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Bring scaled data back to the scale of the variable.

        Args:
            X: Scaled feature frame or target series.

        Returns:
            The unscaled data.

        Raises:
            NotFittedError: If the scaler was not fitted.
            TypeError: If ``X`` is neither a ``Series`` nor a ``DataFrame``.

        Examples:
            >>> X = pd.DataFrame({'m1': [1200.0]}, index=pd.to_datetime(['2021-12-31']))
            >>> scaler = StageScaler(
            ...     source_freq='Y', pred_freq='M', column_frequencies={'m1': 'M'},
            ...     target_column='a1',
            ... ).fit(X)
            >>> scaler.inverse_transform(scaler.transform(X))['m1'].tolist()
            [1200.0]
        """
        check_is_fitted(self)
        return self.invert(X, self._fitted_divisor(X))
