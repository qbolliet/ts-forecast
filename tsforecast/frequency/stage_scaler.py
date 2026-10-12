"""Frequency scaling of one imputation stage, as a reversible sklearn transformer.

``HighFrequencyImputer`` fits a model whose target is a low-frequency variable
(a yearly total) on a grid running at a higher frequency (months). Both sides of
that model must be expressed at the same scale, and the scale is a pure function
of the frequency pair — never of the data. This module holds that arithmetic.

Three divisors are governed by the scaling mode :

1. :meth:`StageScaler.feature_divisors` — the covariates, at fit and at predict,
   each divided by the period its cells actually span on the grid they sit on;
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
from ..utils.frequency.utils import is_higher_frequency, normalize_frequency
# Normalisation des clés d'entité du panel
from ..panel.utils import normalize_entity_key, normalize_entity_mapping


# Modalités de mise à l'échelle
ScaleMode = Literal['constant', 'calendar']
# Réglage effectif d'une colonne, cible comprise : "False" ne divise rien
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
# Modalités nommées, comparées comme chaînes
_SCALE_MODES: Tuple[str, ...] = ('constant', 'calendar')

# Convertisseur partagé : sans état, une instance suffit à tout le module
_DEFAULT_CONVERTER = FrequencyConverter()


# Fonction auxiliaire de test d'admissibilité d'un réglage
def _is_admissible_setting(value: Any) -> bool:
    """Tell whether one value is an admissible scaling setting.

    ``False`` is compared by identity: the resolution reads ``mode is False``,
    so a value merely equal to it (``0``, ``0.0``, ``np.False_``) would pass a
    comparison by equality and then behave as ``'constant'``.

    Args:
        value: Candidate setting.

    Returns:
        True for ``False``, ``'constant'`` and ``'calendar'`` only.

    Examples:
        >>> _is_admissible_setting(False), _is_admissible_setting('calendar')
        (True, True)
        >>> _is_admissible_setting(0), _is_admissible_setting(np.False_)
        (False, False)
    """
    return value is False or (isinstance(value, str) and value in _SCALE_MODES)


# Fonction auxiliaire de la part calendaire d'une période fine dans une période grossière
def _calendar_share(
    dates: pd.DatetimeIndex,
    fine_freq: str,
    coarse_freq: str,
    converter: FrequencyConverter,
) -> np.ndarray:
    """Share of each date's coarse period covered by its fine period, in real time.

    The ``'calendar'`` counterpart of a fractional constant factor: a month
    holds ``31 / 91`` of the first quarter of 2024 and ``29 / 91`` for
    February, where ``'constant'`` gives ``1/3`` to every month. The shares
    of the fine periods of one coarse period sum to 1.

    Args:
        dates: Dates of the rows, each one locating its two periods.
        fine_freq: Finer frequency, the period carried by the row.
        coarse_freq: Coarser frequency, the period of the stage.
        converter: Converter providing the constant fallback.

    Returns:
        Array of shares aligned on ``dates``; the constant factor when the
        frequency bases cannot be expressed as pandas Periods.

    Examples:
        >>> dates = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-31'])
        >>> shares = _calendar_share(dates, 'M', 'Q', FrequencyConverter())
        >>> [round(value * 91) for value in shares]
        [31, 29, 31]
    """
    # Bases de fréquence, sans position ni ancrage
    fine = normalize_frequency(fine_freq, return_format='base')
    coarse = normalize_frequency(coarse_freq, return_format='base')

    try:
        fine_periods = dates.to_period(fine)
        coarse_periods = dates.to_period(coarse)
    except (ValueError, AttributeError):
        # Repli sur le facteur constant si les bases ne sont pas des Periods
        return np.full(len(dates), float(converter.get_conversion_factor(coarse, fine)))

    # Durées exactes en nanosecondes : fin de période incluse (fin = début
    # de la période suivante moins une nanoseconde)
    one = pd.Timedelta(1, unit='ns')
    fine_span = (fine_periods.end_time - fine_periods.start_time + one).asi8
    coarse_span = (coarse_periods.end_time - coarse_periods.start_time + one).asi8
    return fine_span.astype(float) / coarse_span.astype(float)


# Fonction auxiliaire de clé de comparaison d'une fréquence
def _frequency_key(freq: Optional[str]) -> Optional[str]:
    """Comparison key of one frequency, insensitive to position aliases.

    Args:
        freq: Frequency label, or None.

    Returns:
        The base frequency (``'ME'`` and ``'MS'`` both give ``'M'``), the
        label itself when it cannot be normalized, None for None.

    Examples:
        >>> _frequency_key('QE'), _frequency_key('zz'), _frequency_key(None)
        ('Q', 'zz', None)
    """
    if freq is None:
        return None
    try:
        return normalize_frequency(freq, return_format='base')
    except (ValueError, TypeError):
        return freq


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
    return index.droplevel(-1).map(normalize_entity_key).tolist()


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

    When the period carried is FINER than the stage (a monthly block at a
    quarterly stage), the divisor is fractional: ``'constant'``
    gives every month the uniform ratio ``1/3``, ``'calendar'`` its real
    share of days (``31/91`` for January 2024, ``29/91`` for February), the
    shares of one stage period summing to 1.

    The shape of every divisor follows the configuration, never the
    values: per row under ``'calendar'``, for a per-entity ``grid_freq`` or
    ``source_freq`` (a mutualized training set), or when the frequencies
    read resolve differently per entity; compact otherwise.

    ``False`` puts no divisor at all — neither on the covariates it covers,
    nor on ``y`` when it is the setting resolved for the imputed column. The
    target is a column like any other here: an index, a rate or an already
    comparable variable has no reason to be divided, and the caller is trusted
    to keep both sides of one model at a coherent scale. Under ``False`` the
    model therefore predicts at the scale of the variable itself, and the
    frozen :attr:`fit_scale_factor_` is ``1.0``, so the scale carry-over of
    the predictions is a no-op.

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
            (``f_var``), for :meth:`fit`: a string, or an entity -> block
            frequency mapping ``{e: f_block(e)}`` on a mutualized panel
            training set.
        pred_freq: Prediction frequency of the stage, for :meth:`fit`: a
            string, or an entity -> frequency mapping on a panel.
        column_frequencies: Detected frequency of each covariate, for
            :meth:`fit`: ``{column: frequency}``, each frequency being a
            string or an entity -> frequency mapping.
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
        source_freq: Optional[FrequencyBinding] = None,
        pred_freq: Optional[FrequencyBinding] = None,
        column_frequencies: Optional[Mapping[str, FrequencyBinding]] = None,
        materialization: Optional[Mapping[str, str]] = None,
        default_divisor: float = 1.0,
        converter: Optional[FrequencyConverter] = None,
    ) -> None:
        """Validate the configuration and store it untouched.

        Args:
            scale_features: Scaling mode of the features (see class docstring).
            source_freq: Detected frequency of the imputed variable.
            pred_freq: Prediction frequency of the stage.
            column_frequencies: Detected frequency of each covariate.
            materialization: Materialization way of each covariate, read by
                :meth:`fit` alone: only ``'raw_anchors'`` changes a divisor,
                by saying that the cells span a period of the column rather
                than one of the grid. The divisor methods take it as an
                argument, so a single instance still serves every stage.
            default_divisor: Divisor for covariates of unknown frequency.
            converter: Frequency converter to use.

        Raises:
            ValueError: If ``scale_features`` is not admissible.
        """
        # Validation de la modalité, clés du dict non vérifiées ici : aucune
        # colonne n'est connue à l'initialisation (voir "validate_columns")
        self._validate_scale_features(scale_features)

        # Stockage des paramètres tels que reçus : "get_params"/"clone" de
        # sklearn exigent l'identité entre argument et attribut
        self.scale_features = scale_features
        self.source_freq = source_freq
        self.pred_freq = pred_freq
        self.column_frequencies = column_frequencies
        self.materialization = materialization
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
                is not ``False`` (compared by identity: ``0`` or
                ``np.False_`` are rejected), ``'constant'`` or ``'calendar'``;
                or if the dict is empty.
        """
        # Forme dictionnaire : chaque valeur doit être admissible
        if isinstance(scale_features, dict):
            if not scale_features:
                raise ValueError("scale_features dict cannot be empty")
            # Enumération des valeurs non-valides
            invalid = {
                key: value
                for key, value in scale_features.items()
                if not _is_admissible_setting(value)
            }
            if invalid:
                raise ValueError(
                    f"scale_features values must be one of {_SCALE_SETTINGS}, "
                    f"got {invalid}"
                )
            return

        # Forme scalaire
        if not _is_admissible_setting(scale_features):
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

        The setting is validated again on every call: ``set_params``
        (``GridSearchCV``, pipelines) bypasses ``__init__``, and every divisor
        method, ``fit`` included, goes through this resolution.

        Returns:
            ``False``, ``'constant'`` or ``'calendar'``: the column's own
            entry, else the ``'__default__'`` entry, else ``'constant'``.

        Raises:
            ValueError: If ``scale_features`` is not admissible (typically
                after ``set_params``).

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
        # Revalidation : "set_params" contourne la validation d'__init__
        self._validate_scale_features(setting)

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
        """Count the ``f_from`` periods held by one ``f_to`` period.

        ``f_from`` is the stage frequency, ``f_to`` the period actually
        carried. When ``f_to`` is the coarser one, the count is a whole number
        of sub-periods (12 months in a year). When it is the FINER one — a
        monthly block at a quarterly stage — the count is a
        fraction: ``'constant'`` gives the uniform ratio (``1/3`` for every
        month), ``'calendar'`` the real share of time (``31/91``, ``29/91``,
        ``31/91`` over the first quarter of 2024), the shares of one stage
        period summing to 1.

        Args:
            f_from: Stage frequency, whose periods are counted.
            f_to: Frequency of the period carried by the row.
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
            >>> idx = pd.to_datetime(['2024-01-31', '2024-02-29'])
            >>> [round(v * 91) for v in scaler._pair_divisor('Q', 'M', 'calendar', idx)]
            [31, 29]
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
        dates = _dates_of(index)
        # Période portée plus fine que l'étape : part de temps réelle de la
        # période d'étape, fractionnaire. Le décompte du convertisseur
        # compterait sinon la période d'étape qui la recouvre, soit 1
        if is_higher_frequency(f_to, f_from):
            counts = _calendar_share(dates, f_to, f_from, self._conv)
        else:
            counts = self._conv.count_subperiods_per_period(
                dates, low_freq=f_to, high_freq=f_from
            )
        return pd.Series(counts, index=index, dtype=float)

    # Méthode auxiliaire de comparaison d'une colonne à sa grille
    @staticmethod
    def _coarser_than(f_col: Optional[str], f_grid: Optional[str]) -> bool:
        """Tell whether a column is read at its anchors on a grid.

        Args:
            f_col: Frequency of the column for one entity.
            f_grid: Frequency of the grid for that entity.

        Returns:
            True when ``f_col`` is strictly coarser than ``f_grid``, and when
            the two cannot be compared — the cell then keeps the period of the
            column, and an unreadable one falls back on the default divisor.
            False when the column is at least as fine as the grid.

        Examples:
            >>> StageScaler._coarser_than('Y', 'M'), StageScaler._coarser_than('M', 'Q')
            (True, False)
            >>> StageScaler._coarser_than('Q', 'QE')
            False
        """
        # Fréquences absentes ou illisibles : la période de la colonne est
        # conservée, le repli du diviseur s'en charge
        if f_col is None or f_grid is None:
            return True
        try:
            return is_higher_frequency(f_grid, f_col)
        except (ValueError, TypeError):
            return True

    # Méthode auxiliaire de résolution d'une fréquence pour une entité
    @staticmethod
    def _freq_for(binding: Optional[FrequencyBinding], entity: EntityKey) -> Optional[str]:
        """Resolve a frequency binding for one entity.

        The resolution cascade is the one of
        ``CovariateMaterializer._column_frequency``: the entity's own entry —
        the keys being normalized, ``'FR'`` and ``('FR',)`` designate the same
        entity —, then, for a mapping holding a single entry, that entry,
        which stands for the whole panel.

        Args:
            binding: A frequency string, or an entity -> frequency mapping,
                entity keys tuples or, for a single-level panel, scalars.
            entity: Normalized entity key.

        Returns:
            The frequency of that entity, or None when the mapping does not
            cover it.

        Examples:
            >>> StageScaler._freq_for({('FR',): 'Y'}, ('FR',))
            'Y'
            >>> StageScaler._freq_for({('FR',): 'Y'}, ('DE',))
            'Y'
            >>> StageScaler._freq_for({('FR',): 'Y', ('DE',): 'Q'}, ('IT',)) is None
            True
            >>> StageScaler._freq_for({'FR': 'Y', 'DE': 'Q'}, ('DE',))
            'Q'
        """
        # Forme scalaire : la même fréquence pour toutes les entités
        if not isinstance(binding, Mapping):
            return binding

        # Entrée propre, clés normalisées : 'FR' et ('FR',) désignent la même
        # entité, comme pour le matérialiseur
        by_entity = normalize_entity_mapping(binding)
        key = normalize_entity_key(entity)
        if key in by_entity:
            return by_entity[key]
        # Liaison à une seule entrée : elle vaut pour tout le monde
        if len(by_entity) == 1:
            return next(iter(by_entity.values()))
        return None

    # Méthode auxiliaire de collecte des entités portées par une grille
    @staticmethod
    def _grid_entities(index: Optional[pd.Index], parameter: str) -> List[EntityKey]:
        """Collect the distinct entities carried by a stage grid.

        Called as soon as a frequency binding is given per entity and the
        entities to serve are those of the grid, not those the mapping happens
        to name: an entity of the grid left uncovered must be reported, never
        silently skipped.

        Args:
            index: Stage grid.
            parameter: Name of the per-entity parameter, for the messages.

        Returns:
            Sorted list of the distinct normalized entity keys of ``index``.

        Raises:
            ValueError: If ``index`` is None or carries no entity level.

        Examples:
            >>> idx = pd.MultiIndex.from_product(
            ...     [['FR'], pd.date_range('2021-01-31', periods=2, freq='ME')]
            ... )
            >>> StageScaler._grid_entities(idx, 'source_freq')
            [('FR',)]
        """
        # Diviseurs par bloc : une grille est indispensable pour les ventiler
        if index is None:
            raise ValueError(
                f"{parameter} given as an entity -> frequency mapping requires "
                "an index to spread the per-row divisors over, got index=None"
            )
        
        # Extraction des entités de l'index
        entities = _entities_of(index)
        if entities is None:
            raise ValueError(
                f"{parameter} given as an entity -> frequency mapping requires "
                f"a MultiIndex (entity..., date), got a {type(index).__name__}"
            )
        return sorted(set(entities))

    # Méthode auxiliaire de résolution de la fréquence de bloc d'une entité
    def _block_freq(
        self,
        source_freq: FrequencyBinding,
        entity: EntityKey,
        parameter: str = 'source_freq',
    ) -> str:
        """Resolve the block frequency ``f_block(e)`` of one entity.

        Args:
            source_freq: Frequency binding read: a string, or an entity ->
                frequency mapping ``{e: f_block(e)}``.
            entity: Normalized entity key.
            parameter: Name of the parameter the binding came from, for the
                message — ``'source_freq'`` for the target, ``'grid_freq'``
                for the covariates.

        Returns:
            The frequency of the variable for that entity.

        Raises:
            ValueError: If the mapping covers neither the entity nor the whole
                panel — the scale of a block is never guessed.
        """
        # Extraction de la fréquence
        freq = self._freq_for(source_freq, entity)
        # Vérification que la fréquence est spécifiée
        if freq is None:
            # Clés normalisées, triées par représentation
            keys = sorted(normalize_entity_mapping(source_freq), key=repr)
            raise ValueError(
                f"{parameter} does not cover entity {entity!r}: an entity of "
                "the grid must either have its own entry or be covered by a "
                f"single-entry mapping, got keys {keys}"
            )
        return freq

    # Méthode auxiliaire de résolution de la fréquence d'étape d'une entité
    def _stage_freq_for(self, pred_freq: FrequencyBinding, entity: EntityKey) -> str:
        """Resolve the stage frequency of one entity, never guessed.

        A stage binding whose entries all agree is the frequency of the
        stage, and stands for the whole panel — the generalization of the
        single-entry rule. It matters on a mutualized training set: an entity
        whose own progression skips the stage still contributes its block,
        and its rows are brought to the scale of the stage like any other.
        Only a disagreeing binding leaves an uncovered entity without scale.

        Args:
            pred_freq: Stage frequency: a string, or an entity -> frequency
                mapping.
            entity: Normalized entity key.

        Returns:
            The stage frequency of that entity.

        Raises:
            ValueError: If a disagreeing mapping does not cover the entity.

        Examples:
            >>> scaler = StageScaler()
            >>> scaler._stage_freq_for({('FR',): 'M', ('IT',): 'M'}, ('DE',))
            'M'
        """
        if not isinstance(pred_freq, Mapping):
            return pred_freq
        # Entrée propre, clé normalisée, ou liaison à une seule entrée
        freq = self._freq_for(pred_freq, entity)
        if freq is not None:
            return freq
        # Liaison unanime : la fréquence de l'étape vaut pour tout le panel
        if len({_frequency_key(value) for value in pred_freq.values()}) == 1:
            return next(iter(pred_freq.values()))
        # Liaison discordante : l'échelle d'un bloc ne se devine pas
        return self._block_freq(pred_freq, entity, parameter='pred_freq')

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

    # Méthode de calcul des diviseurs de toutes les covariables
    def feature_divisors(
        self,
        *,
        columns: Sequence[str],
        column_frequencies: Mapping[str, FrequencyBinding],
        ways: Mapping[str, str],
        grid_freq: FrequencyBinding,
        stage_freq: FrequencyBinding,
        index: Optional[pd.Index] = None,
        default: Optional[float] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Divide each covariate by the period its cells actually span.

        The divisor of a covariate is read off the period one of its cells
        covers on the grid it sits on, and not off the frequency of the
        imputed variable. As the training grid of the ``HighFrequencyImputer``
        is the mutualized block
        grid, one frequency per entity, and the covariates are
        materialized onto it by the imputation ways. Two cells of the same column
        may then span different periods, and only the way says which:

        - ``'raw_anchors'`` reads a column coarser than the grid where it is
          observed, so the cell spans one period of the column, ``f_col``. A
          column at least as fine as the grid is never read at its anchors:
          the materializer brings it back to identity or exact aggregation
          (rank 1, entity by entity), and its cell spans one period of the
          grid;
        - every other way — identity, exact aggregation, interpolation, mirror
          — produces a cell spanning one period of the grid, ``f_grid``.

        The divisor is then the count of stage sub-periods in that span, the
        very primitive the per-row target divisor uses: the two axes
        share one mechanism, and there are not two scaling plumbings.

        Applied on both grids, it makes the training and prediction features
        comparable by construction a yearly
        aggregate of a monthly covariate being divided by 12 where the same
        covariate read monthly is divided by 1.

        Args:
            columns: Covariate columns to compute a divisor for.
            column_frequencies: Detected frequency of each column: a string,
                or an entity -> frequency mapping.
            ways: Materialization way retained for each column.
            grid_freq: Frequency of the grid the cells sit on: the stage
                frequency on a prediction grid, the block frequency mapping
                ``{e: f_block(e)}`` on a mutualized training grid.
            stage_freq: Frequency of the stage, the common scale everything is
                brought back to.
            index: Grid. Required as soon as one column is in ``'calendar'``
                mode or a binding is given per entity.
            default: Divisor for a column whose frequency cannot be compared.
                Defaults to ``default_divisor``.

        Returns:
            The shape follows the configuration, never the divisor values.
            ``DataFrame`` indexed like ``index`` and columned like
            ``columns`` as soon as one column is in ``'calendar'`` mode,
            ``grid_freq`` is a per-entity mapping (a mutualized training
            set), or the frequencies the divisors read resolve
            differently from one entity to another; ``Series`` of floats
            indexed by column name otherwise (time series, homogeneous
            panels) — even when the divisors of two entities happen to
            coincide, or are all ``1.0``.

        Raises:
            ValueError: If per-row divisors are needed and ``index`` is None
                or carries no entity level, or if an entity of the grid is
                not covered by a per-entity ``grid_freq``, or by a
                disagreeing per-entity ``stage_freq``.

        Examples:
            >>> scaler = StageScaler()
            >>> divisors = scaler.feature_divisors(
            ...     columns=['m1', 'q1'],
            ...     column_frequencies={'m1': 'M', 'q1': 'Q'},
            ...     ways={'m1': 'aggregate', 'q1': 'aggregate'},
            ...     grid_freq='Y', stage_freq='M',
            ... )
            >>> divisors.to_dict()
            {'m1': 12.0, 'q1': 12.0}

            The same two covariates read on the monthly stage grid:

            >>> divisors = scaler.feature_divisors(
            ...     columns=['m1', 'q1'],
            ...     column_frequencies={'m1': 'M', 'q1': 'Q'},
            ...     ways={'m1': 'identity', 'q1': 'interpolate'},
            ...     grid_freq='M', stage_freq='M',
            ... )
            >>> divisors.to_dict()
            {'m1': 1.0, 'q1': 1.0}
        """
        # Diviseur de repli
        fallback = self.default_divisor if default is None else default

        # Modalité par colonne, et liaisons de fréquence réellement lues : la
        # fréquence d'une colonne n'est lue que pour ses ancres brutes
        modes = {column: self.resolve_mode(column) for column in columns}
        calendar_used = any(mode == 'calendar' for mode in modes.values())
        read_columns = [
            column for column in columns
            if modes[column] is not False and ways.get(column) == 'raw_anchors'
        ]
        bindings = [stage_freq, *(column_frequencies.get(col) for col in read_columns)]

        # Entités à servir : celles de la grille dès qu'une liaison est par
        # entité et que la grille porte un niveau d'entité, une entité non
        # couverte étant alors une erreur, jamais un NaN
        if isinstance(grid_freq, Mapping):
            entities = self._grid_entities(index, 'grid_freq')
        elif any(isinstance(binding, Mapping) for binding in bindings) and isinstance(
            index, pd.MultiIndex
        ):
            entities = self._grid_entities(index, 'stage_freq')
        else:
            entities = self._binding_entities(bindings)

        # Diviseurs par (entité, colonne), et signature des fréquences lues
        per_entity: Dict[EntityKey, Dict[str, Divisor]] = {}
        signatures = set()
        # Parcours des entités
        for entity in entities:
            # Fréquence de l'étape
            f_stage = self._stage_freq_for(stage_freq, entity)
            # Fréquence de la grille
            f_grid = (
                self._block_freq(grid_freq, entity, parameter='grid_freq')
                if isinstance(grid_freq, Mapping)
                else grid_freq
            )
            # Dictionnaire des diviseurs
            row: Dict[str, Divisor] = {}
            carried_by_column: Dict[str, Optional[str]] = {}
            # Parcours des colonnes
            for column in columns:
                # Modalité désactivée : la colonne n'est jamais divisée
                if modes[column] is False:
                    row[column] = 1.0
                    continue
                # Période couverte par la cellule : celle de la colonne quand
                # ses ancres sont lues telles quelles, celle de la grille
                # sinon. Une colonne au moins aussi fine que la grille n'est
                # jamais lue à ses ancres : le matérialiseur la ramène à
                # l'identité ou à l'agrégation exacte,
                # et sa cellule couvre alors une période de la grille
                if column in read_columns:
                    f_col = self._freq_for(column_frequencies.get(column), entity)
                    carried = f_col if self._coarser_than(f_col, f_grid) else f_grid
                else:
                    carried = f_grid
                carried_by_column[column] = carried
                try:
                    row[column] = self._pair_divisor(
                        f_stage, carried, modes[column], index
                    )
                except (ValueError, TypeError):
                    # Fréquences incomparables : repli documenté
                    row[column] = fallback
            per_entity[entity] = row
            signatures.add((
                _frequency_key(f_stage),
                tuple(_frequency_key(carried_by_column[c]) for c in columns if c in carried_by_column),
            ))

        # Forme dictée par la configuration seule : par ligne sous 'calendar',
        # sur un jeu mutualisé, ou quand les fréquences lues varient d'une
        # entité à l'autre ; compacte sinon, les diviseurs étant alors
        # identiques par construction
        per_row = calendar_used or isinstance(grid_freq, Mapping) or len(signatures) > 1
        if not per_row:
            return pd.Series(next(iter(per_entity.values())), index=list(columns), dtype=float)

        # Forme par ligne : le diviseur dépend de la date, de l'entité, ou des
        # deux, et aucune Series à une dimension ne le porte
        if index is None:
            raise ValueError(
                "Per-row covariate divisors require an index ('calendar' mode "
                "or entities disagreeing on a frequency)"
            )
        return self._spread(per_entity, columns, index)

    # Méthode auxiliaire de ventilation d'un diviseur unique sur la grille
    def _spread_one(
        self,
        per_entity: Mapping[EntityKey, Divisor],
        index: pd.Index,
    ) -> pd.Series:
        """Spread one per-entity divisor over a stage grid.

        The single place where an entity becomes a row selection: the feature
        divisors and the target divisor both go through it, so a divisor
        landing on the wrong rows is a defect of one routine, not of two.

        Args:
            per_entity: Divisor of one quantity, per entity. A single key
                covers the whole grid, whatever that key is.
            index: Stage grid.

        Returns:
            Unnamed ``Series`` of floats aligned on ``index``.

        Raises:
            ValueError: If several entities are involved and ``index`` carries
                no entity level.
        """
        # Entité unique : le diviseur couvre toute la grille, sans ventilation
        if len(per_entity) == 1:
            divisor = next(iter(per_entity.values()))
            if np.isscalar(divisor):
                return pd.Series(float(divisor), index=index, dtype=float)
            return pd.Series(np.asarray(divisor, dtype=float), index=index)

        # Cohérence de l'index avec la ventilation demandée
        entities_per_row = _entities_of(index)
        if entities_per_row is None:
            raise ValueError(
                "Entity-wise divisors require a MultiIndex (entity..., date), "
                f"got a {type(index).__name__}"
            )

        # Initialisation de la Series résultat
        result = pd.Series(np.nan, index=index, dtype=float)
        # Ventilation par entité : comparaison terme à terme plutôt que par un
        # Index de tuples, que pandas diffuserait élément par élément
        for entity, divisor in per_entity.items():
            # Masque correspondant à l'entité
            mask = np.array(
                [row_entity == entity for row_entity in entities_per_row],
                dtype=bool,
            )
            result.loc[mask] = (
                divisor
                if np.isscalar(divisor)
                else np.asarray(divisor, dtype=float)[mask]
            )
        return result

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
        # Initialisation du DataFrame résultat
        result = pd.DataFrame(index=index, columns=list(columns), dtype=float)
        # Ventilation colonne par colonne, par la routine commune
        for column in columns:
            spread = self._spread_one(
                {
                    entity: divisors[column]
                    for entity, divisors in per_entity.items()
                },
                index,
            )
            # Affectation positionnelle : un index dupliqué ne doit déclencher
            # aucun réalignement
            result[column] = spread.to_numpy()
        return result

    # Méthode auxiliaire du diviseur d'étape, scalaire ou ventilé par entité
    def _stage_divisor(
        self,
        source_freq: FrequencyBinding,
        pred_freq: FrequencyBinding,
        mode: ScaleMode,
        index: Optional[pd.Index],
    ) -> Divisor:
        """Count the stage sub-periods held by one period of the variable.

        Args:
            source_freq: Detected frequency of the variable: a string, or an
                entity -> block frequency mapping ``{e: f_block(e)}``.
            pred_freq: Prediction frequency of the stage: a string, or an
                entity -> frequency mapping.
            mode: Scaling mode applied to the target.
            index: Stage grid, required under ``'calendar'``, when the
                entities disagree, or when ``source_freq`` is a mapping.

        Returns:
            A float, or a ``Series`` aligned on ``index``.

        Raises:
            ValueError: If per-row divisors are needed and ``index`` is None,
                or if an entity of the grid is absent from ``source_freq`` or
                from a disagreeing per-entity ``pred_freq``.
        """
        # Liaison par entité de la fréquence de bloc : un diviseur par bloc,
        # donc par ligne, chaque entité étant ramenée à l'échelle de l'étape
        if isinstance(source_freq, Mapping):
            entities = self._grid_entities(index, 'source_freq')
            return self._spread_one(
                {
                    entity: self._pair_divisor(
                        self._stage_freq_for(pred_freq, entity),
                        self._block_freq(source_freq, entity),
                        mode,
                        index,
                    )
                    for entity in entities
                },
                index,
            )

        # Fréquences d'étape distinctes réellement portées, aux alias de
        # position près ("ME" et "M" sont une seule fréquence)
        if isinstance(pred_freq, Mapping):
            distinct = sorted({_frequency_key(freq) for freq in pred_freq.values()})
        else:
            distinct = [pred_freq]

        # Étape homogène : un seul couple de fréquences, la liaison unanime
        # valant pour toutes les entités de la grille
        if len(distinct) == 1:
            return self._pair_divisor(distinct[0], source_freq, mode, index)

        # Étape hétérogène : un diviseur par entité, donc par ligne
        if index is None:
            raise ValueError(
                "Entity-wise prediction frequencies require an index to spread "
                f"the per-row divisors over, got frequencies {distinct}"
            )
        # Entités de la grille : une entité non couverte lève, jamais un NaN
        entities = (
            self._grid_entities(index, 'pred_freq')
            if isinstance(index, pd.MultiIndex)
            else self._binding_entities([pred_freq])
        )
        return self._spread_one(
            {
                entity: self._pair_divisor(
                    self._stage_freq_for(pred_freq, entity), source_freq, mode, index
                )
                for entity in entities
            },
            index,
        )

    # Méthode du diviseur de la cible
    def target_divisor(
        self,
        column: Optional[str],
        source_freq: FrequencyBinding,
        pred_freq: FrequencyBinding,
        index: Optional[pd.Index] = None,
        produced_freq: Optional[pd.Series] = None,
    ) -> Divisor:
        """Compute the divisor of ``y`` for one imputed column.

        The mode applied is the one resolved for the imputed column, read like
        that of any other column: ``False`` there leaves ``y`` untouched and
        yields the divisor ``1.0``.

        The arithmetic is different between the features and the target,
        which is why this method is not a call to
        :meth:`feature_divisors` on the imputed column. A covariate carried at
        its own frequency is never re-aggregated and divides by ``1.0`` ;
        the target, being produced on the stage grid, carries
        ``pred_freq`` there and divides by the count of ``pred_freq``
        sub-periods in one ``source_freq`` period. The two rules coincide only
        when :meth:`feature_divisors` is handed ``pred_freq`` as the column's
        frequency — and even then the failure modes differ: an incomparable
        frequency falls back on ``default`` for a covariate, but must raise for
        the target, whose scale silently governs what the model learns.

        When ``produced_freq`` is given, each row of ``y`` is divided by the
        divisor of the frequency at which it was produced (read from
        ``imputed_freq_store``), not by the stage scalar: a yearly variable
        already imputed at the quarterly stage carries quarters on those rows,
        and mixing them raw with its yearly anchors would teach the model a
        blend of two scales.

        ``source_freq`` takes the same two forms as ``pred_freq``: a string, or
        an entity -> block frequency mapping ``{e: f_block(e)}`` describing a
        mutualized training set, in which case the divisor is
        per block, hence per row. A block finer than the stage yields a
        fractional divisor (``1/3`` for a monthly block at a quarterly stage),
        a normal case that is neither rejected nor floored at ``1.0``.

        Precedence: when ``produced_freq`` is given, it decides, row by row
        — ``source_freq`` is then of no use and is not read at
        all, whatever its form.

        Args:
            column: Name of the imputed column, for mode resolution.
            source_freq: Detected frequency of the variable: a string, or an
                entity -> block frequency mapping. Ignored when
                ``produced_freq`` is given.
            pred_freq: Prediction frequency of the stage.
            index: Stage grid, required under ``'calendar'``, when the
                entities disagree, or when ``source_freq`` is a mapping.
            produced_freq: Frequency at which each row of ``y`` was produced,
                indexed like ``y``. When given, the result is ALWAYS a
                ``Series``, even if every row shares the same frequency.

        Returns:
            A float, or a ``Series`` of per-row divisors.

        Raises:
            ValueError: If per-row divisors are needed and no index is
                available, or if an entity of the grid is absent from a
                per-entity ``source_freq``.

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
        mode = self.resolve_mode(column)

        # Cible dispensée d'échelle : aucun diviseur, y compris par ligne
        if mode is False:
            return 1.0

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
            stage_per_row = [self._stage_freq_for(pred_freq, e) for e in entities_per_row]
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
        source_freq: FrequencyBinding,
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
        grid: a carry-over computed on another grid must realign it. Under
        ``False`` it is ``1.0``, the target having been left at its own scale.

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
        # Cible dispensée d'échelle : rien n'est cuit dans le modèle
        mode = self.resolve_mode(column)
        if mode is False:
            return 1.0

        return self._stage_divisor(source_freq, pred_freq, mode, index)

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
            The scaled data, with the name of a ``Series`` input kept. The
            input object itself when ``divisor`` is the scalar ``1.0``.

        Examples:
            >>> scaler = StageScaler()
            >>> y = pd.Series([120.0, 132.0])
            >>> scaler.apply(y, 12.0).tolist()
            [10.0, 11.0]
        """
        if self._is_identity(divisor):
            return values
        return self._keep_name(values, values / divisor)

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
            The unscaled data, with the name of a ``Series`` input kept. The
            input object itself when ``divisor`` is the scalar ``1.0``.

        Examples:
            >>> scaler = StageScaler()
            >>> y = pd.Series([10.0, 11.0])
            >>> scaler.invert(y, 12.0).tolist()
            [120.0, 132.0]
        """
        if self._is_identity(divisor):
            return values
        return self._keep_name(values, values * divisor)

    # Méthode auxiliaire de conservation du nom d'une Series
    @staticmethod
    def _keep_name(
        values: Union[pd.Series, pd.DataFrame],
        result: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Give the result of an arithmetic operation the name of its input.

        pandas drops the name of a ``Series`` divided by a differently named
        (or unnamed) ``Series``: a per-row divisor would otherwise erase the
        name of ``y``, which designates the imputed column.

        Args:
            values: Operand being scaled.
            result: Result of the operation.

        Returns:
            ``result``, renamed after ``values`` when both are ``Series``.
        """
        if isinstance(values, pd.Series) and isinstance(result, pd.Series):
            result.name = values.name
        return result

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
            y: Target of the stage. Only its index and its name are read:
                the index carries the per-row target divisor and defaults to
                that of ``X``, the name designates the imputed column whose
                mode applies to ``y``. An unnamed ``y`` falls back on the
                global setting of ``scale_features``.
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
            ... ).fit(X)
            >>> scaler.fit_scale_factor_
            12.0
            >>> scaler.feature_divisors_.to_dict()
            {'m1': 1.0}
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

        # Diviseurs des covariables. La grille de "X" est celle de l'étape :
        # chaque cellule y couvre une période de "pred_freq", sauf celles des
        # colonnes dont "materialization" dit qu'elles sont lues à leurs
        # ancres brutes
        self.feature_divisors_ = self.feature_divisors(
            columns=columns,
            column_frequencies=frequencies,
            ways=self.materialization or {},
            grid_freq=self.pred_freq,
            stage_freq=self.pred_freq,
            index=X.index,
        )

        # Grille de la cible : celle de y quand elle est fournie
        target_index = y.index if y is not None else X.index

        # Colonne imputée : le nom porté par "y", seule source de ce nom
        target_column = y.name if y is not None else None

        # Facteur cuit dans le modèle, puis diviseur effectif de y
        self.fit_scale_factor_ = self.fit_scale_factor(
            target_column, self.source_freq, self.pred_freq, index=target_index
        )
        self.target_divisor_ = self.target_divisor(
            target_column,
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
            ValueError: If a ``DataFrame`` does not hold exactly the columns
                seen at :meth:`fit` (their order is free).
        """
        # Une trame est un jeu de covariables, une série est la cible : le
        # composant sert les deux, et seul le type les distingue
        if isinstance(values, pd.DataFrame):
            # Mêmes colonnes qu'au fit : pandas alignerait sinon les diviseurs
            # par étiquette, en NaN et en union de colonnes, sans erreur
            fitted = list(self.feature_names_in_)
            missing = [column for column in fitted if column not in values.columns]
            unexpected = [column for column in values.columns if column not in fitted]
            if missing or unexpected:
                raise ValueError(
                    "X has different columns than during fit: "
                    f"missing {missing}, unexpected {unexpected}"
                )
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
            ValueError: If a ``DataFrame`` ``X`` does not hold the columns
                seen at :meth:`fit`.

        Examples:
            >>> X = pd.DataFrame({'a2': [1200.0]}, index=pd.to_datetime(['2021-12-31']))
            >>> scaler = StageScaler(
            ...     source_freq='Y', pred_freq='M', column_frequencies={'a2': 'Y'},
            ...     materialization={'a2': 'raw_anchors'},
            ... ).fit(X)
            >>> scaler.transform(X)['a2'].tolist()
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
            ValueError: If a ``DataFrame`` ``X`` does not hold the columns
                seen at :meth:`fit`.

        Examples:
            >>> X = pd.DataFrame({'m1': [1200.0]}, index=pd.to_datetime(['2021-12-31']))
            >>> scaler = StageScaler(
            ...     source_freq='Y', pred_freq='M', column_frequencies={'m1': 'M'},
            ... ).fit(X)
            >>> scaler.inverse_transform(scaler.transform(X))['m1'].tolist()
            [1200.0]
        """
        check_is_fitted(self)
        return self.invert(X, self._fitted_divisor(X))
