"""Frequency conversion utilities for time series data.

This module provides the FrequencyConverter class to handle conversions between
different time frequencies using pandas built-in functionality (asfreq and resample).

FrequencyConverter is the generic conversion engine, consistent with the other
converters of the package (DurationConverter, PeriodPositionConverter): the
output index always carries the target frequency (or the union of the target
indexes when converting columns to mixed frequencies). To convert selected
variables of a dataset while keeping the other columns in place (original
index preserved, or extended by the target dates), see
:class:`tsforecast.frequency.frequency_aligner.FrequencyAligner`, which
delegates the actual conversions to this class.
"""
# Importation des modules
import numbers
import numpy as np
import pandas as pd
from typing import Any, Union, Optional, Literal, Dict, Tuple, List
from pandas.tseries.frequencies import to_offset

# Import de la classe parente
from ..abc.converter import TemporalConverter

# Import des utilitaires de fréquence
from .utils import (
    normalize_frequency,
    is_higher_frequency,
    detect_frequency,
    detect_dataset_frequency,
    detect_index_frequency,
)
from ..validation import validate_temporal_data
from ..parse import ParsedFrequency, parse_frequency, build_frequency_string

# Import des utilitaires de gestion des positions
from ..position.utils import normalize_position
# Import des utilitaires de gestion des durées
from ..duration.utils import get_duration_conversion_factor

# Constantes transverses
from .._constants import BLOCK_START_FREQUENCIES, NON_PERIOD_FREQUENCIES

# Import des utilitaires de panel
from ...panel.utils import is_panel_data

# Types pour les méthodes d'agrégation et d'interpolation
AggregationMethod = Literal['mean', 'sum', 'first', 'last', 'min', 'max', 'median', 'std', 'count', 'all', 'any']
InterpolationMethod = Literal['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']

# Alias de fréquence de fin de période dépréciés par pandas (>= 2.2) au profit
# de leur équivalent explicite ('Y'/'A' -> 'YE', 'Q' -> 'QE', 'M' -> 'ME').
# Ne concerne PAS pd.Period/PeriodIndex, qui utilisent toujours les bases
# nues : seules les fréquences destinées à resample/asfreq/date_range doivent
# être modernisées
_DEPRECATED_PERIOD_END_ALIASES = {'Y': 'YE', 'A': 'YE', 'Q': 'QE', 'M': 'ME'}

# Méthodes d'interpolation acceptées par interpolate_to_higher_frequency
_INTERPOLATION_METHODS = frozenset({
    'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
})

# Fonction de modernisation d'un alias de fréquence déprécié
def _modernize_resample_freq(freq: str) -> str:
    """Replace deprecated bare period-end aliases with their modern form.

    Args:
        freq: Frequency string about to be handed to
            ``resample``/``asfreq``/``date_range`` (e.g. 'Q', 'ME', '2Y').

    Returns:
        ``freq`` unchanged, except its base is replaced when it is one of
        the deprecated bare aliases ('Y', 'A', 'Q', 'M'). Semantically a
        no-op: those bare aliases already meant end-of-period in pandas.

    Examples:
        >>> _modernize_resample_freq('Q')
        'QE'
        >>> _modernize_resample_freq('QS')
        'QS'
        >>> _modernize_resample_freq('D')
        'D'
    """
    # Parsing de la fréquence
    try:
        parsed = parse_frequency(freq)
    except ValueError:
        return freq

    # Seule une base nue (sans position S/E) est concernée
    if parsed.position is not None or parsed.freq not in _DEPRECATED_PERIOD_END_ALIASES:
        return freq
    multiplier = str(parsed.multiplier) if parsed.multiplier != 1 else ''
    anchor = f"-{parsed.suffix}" if parsed.suffix is not None else ''
    return f"{multiplier}{_DEPRECATED_PERIOD_END_ALIASES[parsed.freq]}{anchor}"


# Classe de conversion d'une fréquence dans une autre
class FrequencyConverter(TemporalConverter):
    """Handle conversions between different time frequencies.

    This class manages frequency conversions using pandas built-in functionality,
    primarily asfreq for upsampling and resample for downsampling.

    Examples:
        >>> converter = FrequencyConverter()
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> monthly = converter.convert_frequency(series, 'monthly', method='mean')
        >>> len(monthly)
        1
    """

    # Initialisation
    def __init__(self):
        """Initialize the FrequencyConverter."""

    # Implémentation de la méthode abstraite convert de TemporalConverter
    def convert(self,
                value: Union[pd.Series, pd.DataFrame],
                from_unit: str,
                to_unit: str,
                **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Convert data from one frequency to another.

        Implementation of TemporalConverter.convert() for frequencies.

        Args:
            value: Time series data to convert (Series or DataFrame)
            from_unit: Source frequency (not used, frequency is auto-detected)
            to_unit: Target frequency
            **kwargs: Additional conversion parameters (method, limit, etc.)

        Returns:
            Converted time series data

        Raises:
            ValueError: If conversion parameters are invalid

        Examples:
            >>> converter = FrequencyConverter()
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
            >>> monthly = converter.convert(series, 'daily', 'monthly', method='mean')
            >>> len(monthly)
            1
        """
        # Redirection vers convert_frequency qui contient toute la logique
        return self.convert_frequency(data=value, target_freq=to_unit, **kwargs)

    # Méthode de conversion d'une fréquence en une autre
    def convert_frequency(self,
                         data: Union[pd.Series, pd.DataFrame],
                         target_freq: Union[str, Dict[str, str]],
                         method: Union[AggregationMethod, InterpolationMethod] = 'mean',
                         alignment_method: Literal['ffill', 'bfill', 'nearest', 'none'] = 'ffill',
                         time_col: Optional[str]=None,
                         panel_cols: Optional[List[str]] = None,
                         target_position: Optional[str] = None,
                         full_periods_only: bool = False,
                         limit: Union[int, Literal['default'], None] = 'default',
                         limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
                         limit_area: Optional[Literal['inside', 'outside']] = None) -> Union[pd.Series, pd.DataFrame]:
        """Convert data to target frequency using pandas built-in methods.

        This is the main conversion method that automatically determines whether
        to use upsampling (asfreq) or downsampling (resample) based on the
        frequency relationship. Supports both Series and DataFrame with flexible
        target frequency specification.

        The output index carries the target frequency. Each column is converted
        from its own frequency, detected on its observed (non-NaN) values, and
        the output index is the union of the target indexes of the converted
        columns: the rows of the source grid do not survive. Columns that are
        not converted — absent from a dict ``target_freq``, already at their
        target frequency, or never observed — keep their values at their
        observed dates only (a never observed column contributes no date and
        comes out entirely NaN). When several target frequencies (or
        preserved columns) are mixed, the NaN a converted column gets on the
        union are filled according to ``alignment_method``. The result does
        not depend on the order of the columns.

        Series and DataFrames follow the same rule for degenerate inputs: no
        row raises; rows without any observed value give an empty result (no
        date to keep); observed values without any detectable frequency raise.

        Panels (entities in the leading levels of a MultiIndex) are converted
        entity by entity; an entity that no key of a dict ``target_freq``
        targets is returned unchanged, an entity without any observation of
        the converted columns is left out.

        Args:
            data: Time series data to convert (Series or DataFrame)
            target_freq: Target frequency specification:
                - str: Apply same frequency to all columns
                - Dict[str, str]: Map each column to its target frequency
                  (panels also accept ``(entity...,)`` and
                  ``(entity..., column)`` keys; a panel Series matches column
                  keys on its name)
            method: Aggregation method for downsampling or interpolation method
                for upsampling. The default ``'mean'`` is an aggregation method:
                an upsampling needs an interpolation method (``'linear'``, ...)
            alignment_method: Method to align indexes when mixing frequencies
                ('ffill', 'bfill', 'nearest', 'none')
            time_col: Identifier of time columns to exclude from conversion
            panel_cols: List of panel identifier columns to exclude from
                conversion
            target_position: Optional position for target frequency ('S', 'E',
                'start', 'end'). If None, the position of ``target_freq`` is
                used, otherwise the position of each source (column by column),
                otherwise the default 'E'
            full_periods_only: If True, periods where the number of non-NaN
                observations is less than the calendar count of expected
                sub-periods produce NaN during downsampling.
            limit: Maximum number of consecutive NaN values to fill during
                upsampling interpolation. ``'default'`` uses the frequency
                conversion factor. None applies no limit.
            limit_direction: Direction for filling NaN during upsampling. If
                None, defaults to ``'forward'`` for start-positioned frequencies
                and ``'backward'`` for end-positioned frequencies.
            limit_area: Restriction area for filling NaN during upsampling
                (``'inside'`` or ``'outside'``).

        Returns:
            Converted time series data

        Raises:
            ValueError: If the data is empty, if conversion parameters are
                invalid, if observed values have no detectable frequency (a
                Series, or every column of a DataFrame), or if ``method`` does
                not fit the direction of the conversion.
            NotImplementedError: If ``full_periods_only`` or ``method='all'``
                is combined with a multiplied frequency (``'2MS'``).

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> # Series conversion
            >>> daily_dates = pd.date_range('2023-01-01', periods=31, freq='D')
            >>> daily_series = pd.Series(range(31), index=daily_dates)
            >>> monthly = converter.convert_frequency(daily_series, 'monthly', method='mean')
            >>> isinstance(monthly, pd.Series)
            True
            >>> # DataFrame with string target_freq
            >>> daily_df = pd.DataFrame({'a': range(31), 'b': range(31, 62)}, index=daily_dates)
            >>> monthly_df = converter.convert_frequency(daily_df, 'monthly', method='mean')
            >>> # DataFrame with dict target_freq
            >>> mixed_freq = converter.convert_frequency(daily_df, {'a': 'monthly', 'b': 'weekly'}, method='mean')
        """
        # Validation des paramètres d'entrée
        data = self._validate_conversion_params(data=data, target_freq=target_freq, time_col=time_col, panel_cols=panel_cols)

        # Normalisation de la position cible en code ('S'/'E') : target_position
        # accepte aussi les noms littéraux ('start'/'end') en entrée publique, mais
        # build_frequency_string() n'accepte que des codes
        if target_position is not None:
            target_position = normalize_position(target_position)

        # Cas des données de panel (MultiIndex) : traitement entité par entité
        if is_panel_data(data):
            return self._convert_panel_frequency(
                data=data,
                target_freq=target_freq,
                method=method,
                alignment_method=alignment_method,
                target_position=target_position,
                full_periods_only=full_periods_only,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
            )

        # Cas 1 : Series (la validation garantit une chaîne de caractères pour target_freq hors panel)
        if isinstance(data, pd.Series):
            # Aucune valeur observée : aucune date à conserver (comme un DataFrame)
            if data.isna().all():
                return data.iloc[0:0]

            # Détection de la fréquence actuelle (avec position, anchor et multiplicateur)
            source = detect_frequency(data=data, return_format='components')

            # Cas où aucune fréquence n'a pu être détectée
            if not source:
                raise ValueError("Cannot detect current frequency of the data")

            # Décomposition de la fréquence cible
            target = normalize_frequency(target_freq, return_format='components')

            # Position cible : explicite, sinon celle de la fréquence cible, sinon celle de la source
            target_position = target_position or target.position or source.position

            # Construction des fréquences complètes (base + position + multiplicateur)
            target_freq_with_position = self._with_position(target, target_position)
            source_freq_with_position = self._with_position(source, source.position)

            # Si les fréquences sont identiques, retourner les données telles quelles
            if source_freq_with_position == target_freq_with_position:
                return data

            # Détermination de la direction de conversion
            if is_higher_frequency(target_freq_with_position, source_freq_with_position):
                # Upsampling (la fréquence source vient d'être détectée)
                return self.interpolate_to_higher_frequency(
                    data, target_freq_with_position, method,
                    limit=limit, limit_direction=limit_direction,
                    limit_area=limit_area, source_freq=source_freq_with_position,
                )
            # Downsampling (la fréquence source vient d'être détectée)
            return self.aggregate_to_lower_frequency(
                data, target_freq_with_position, method, full_periods_only,
                source_freq=source_freq_with_position,
            )

        # Cas 2 : DataFrame (colonne par colonne)
        # Construction du frequency_map complet
        frequency_map = self._build_frequency_map(data=data, target_freq=target_freq, target_position=target_position)

        # Groupement des conversions identiques pour optimisation
        grouped_conversions = self._group_conversions_by_operation(frequency_map=frequency_map, method=method)

        # Application des conversions groupées
        return self._apply_grouped_conversions(
            data=data,
            grouped_conversions=grouped_conversions,
            alignment_method=alignment_method,
            full_periods_only=full_periods_only,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area
        )

    # Implémentation de la méthode abstraite get_conversion_factor de TemporalConverter
    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """Get approximate conversion factor between two frequencies.

        Implementation of TemporalConverter.get_conversion_factor() for frequencies.
        The factor represents how many ``from_unit`` periods fit into one
        ``to_unit`` period (when ``to_unit`` is lower frequency) or the
        inverse (when ``to_unit`` is higher frequency).

        Args:
            from_unit: Source frequency (e.g., 'daily', 'D', 'monthly', 'M').
            to_unit: Target frequency (e.g., 'monthly', 'M', 'quarterly', 'Q').

        Returns:
            Approximate conversion factor. When converting from a higher
            frequency to a lower one the factor is >= 1 (e.g., daily→monthly
            ≈ 30). When converting from lower to higher the factor is < 1.

        Raises:
            ValueError: If frequencies are not supported

        Examples:
            >>> converter = FrequencyConverter()
            >>> converter.get_conversion_factor('daily', 'monthly')
            30.0
            >>> converter.get_conversion_factor('monthly', 'quarterly')
            3.0
            >>> converter.get_conversion_factor('quarterly', 'monthly')
            0.3333333333333333
        """
        # Normalisation des fréquences (codes/littéraux) vers leur base durée
        # (ex: 'daily'/'D' -> 'D', 'monthly'/'M' -> 'M'), multiplicateur conservé
        # ('2MS' -> '2M') : c'est la durée de la période qui compte
        from_duration = self._duration_of(from_unit)
        to_duration = self._duration_of(to_unit)

        # Délégation au DurationConverter avec arguments inversés :
        # DurationConverter.get_conversion_factor(a, b) = durée(a) / durée(b)
        # = "combien de b dans un a"
        # On veut : "combien de from_freq dans un to_freq" = durée(to) / durée(from)
        # → on passe (to_freq, from_freq)
        return get_duration_conversion_factor(to_duration, from_duration)

    # Méthode auxiliaire de conversion d'une fréquence en durée (base + multiplicateur)
    @staticmethod
    def _duration_of(frequency: str) -> str:
        """Express a frequency as a duration string, multiplier kept.

        Args:
            frequency: Frequency in any supported format ('2MS', 'QE-DEC', 'daily').

        Returns:
            Duration string made of the base code and the multiplier ('2M', 'Q', 'D').

        Examples:
            >>> FrequencyConverter._duration_of('2MS')
            '2M'
            >>> FrequencyConverter._duration_of('QE-DEC')
            'Q'
        """
        parsed = normalize_frequency(frequency, return_format='components')
        return build_frequency_string(parsed.freq, multiplier=parsed.multiplier)

    # Méthode auxiliaire de construction d'une fréquence avec position
    @staticmethod
    def _with_position(parsed: ParsedFrequency, position: Optional[str]) -> str:
        """Build the frequency string of a decomposed frequency at a given position.

        The anchor is left out: the converters work on the base, the position
        and the multiplier. The string goes to pandas (``resample``): without
        position, a month, quarter, year or semi-month gets its end variant
        (``'ME'``).

        Args:
            parsed: Decomposed frequency.
            position: Position ('S', 'E') or None.

        Returns:
            Pandas frequency alias ('MS', '2QE', 'D').

        Examples:
            >>> FrequencyConverter._with_position(ParsedFrequency('M', None, None, 2), 'S')
            '2MS'
            >>> FrequencyConverter._with_position(ParsedFrequency('M', None, None, 1), None)
            'ME'
        """
        return build_frequency_string(
            parsed.freq, position, multiplier=parsed.multiplier, default_position='E'
        )

    # Méthode auxiliaire de rejet des fréquences multipliées
    @staticmethod
    def _reject_multiplied(operation: str, *frequencies: Optional[str]) -> None:
        """Reject multiplied frequencies for an operation that counts base periods.

        Args:
            operation: Name of the operation, for the error message.
            *frequencies: Frequencies to check (None values are skipped).

        Raises:
            NotImplementedError: If a frequency carries a multiplier. Not a
                ValueError: callers tolerate those and would silently skip
                the operation.

        Examples:
            >>> FrequencyConverter._reject_multiplied('counting', 'MS', 'D')
            >>> FrequencyConverter._reject_multiplied('counting', '2MS')
            Traceback (most recent call last):
                ...
            NotImplementedError: counting does not support the multiplied frequency '2MS'
        """
        # Parcours des fréquences
        for frequency in frequencies:
            # Erreur quand un multiplier est différent de 1
            if frequency and normalize_frequency(frequency, return_format='components').multiplier != 1:
                raise NotImplementedError(
                    f"{operation} does not support the multiplied frequency '{frequency}'"
                )

    # Méthode de comptage des sous-périodes, période cible par période cible
    def count_subperiods_per_period(
        self,
        target_index: pd.DatetimeIndex,
        low_freq: str,
        high_freq: str,
    ) -> np.ndarray:
        """Count the high_freq sub-periods of each period of an index.

        Exploits the concrete periods carried by ``target_index`` to count
        exactly: February holds 28 daily sub-periods and January 31, where a
        constant factor would give ~30.4 for both. Falls back on
        :meth:`get_conversion_factor` (a constant ratio, identical for every
        period) when the frequency bases cannot be expressed as pandas
        Periods.

        Args:
            target_index: Index at the low frequency, one entry per target
                period.
            low_freq: Frequency of ``target_index`` (lower, less granular).
            high_freq: Higher (more granular) frequency to count.

        Returns:
            Array of sub-period counts aligned on ``target_index``.

        Raises:
            ValueError: If either frequency is not supported.

        Examples:
            >>> converter = FrequencyConverter()
            >>> index = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> converter.count_subperiods_per_period(index, 'M', 'D')
            array([31., 28., 31.])
            >>> index = pd.date_range('2023-12-31', periods=1, freq='YE')
            >>> converter.count_subperiods_per_period(index, 'Y', 'M')
            array([12.])
        """
        # Le comptage porte sur des périodes de base : un bloc de n périodes n'a pas de décompte
        self._reject_multiplied('Sub-period counting', low_freq, high_freq)

        # Normalisation des fréquences de base (sans positions S/E ni ancrage)
        low = normalize_frequency(low_freq, return_format='base')
        high = normalize_frequency(high_freq, return_format='base')

        # Comptage exact, période cible par période cible
        try:
            return np.array([
                float(len(pd.period_range(start=p.start_time, end=p.end_time, freq=high)))
                for p in target_index.to_period(low)
            ])
        except (ValueError, AttributeError):
            # Repli sur le comptage constant si les bases ne sont pas des Periods
            # On calcule le ratio de durées : « combien de high dans un low »
            return np.full(len(target_index), self.get_conversion_factor(high, low))

    # Méthode d'agrégation à une fréquence plus faible
    def aggregate_to_lower_frequency(self,
                                   data: Union[pd.Series, pd.DataFrame],
                                   target_freq: str,
                                   method: AggregationMethod = 'mean',
                                   full_periods_only: bool = False,
                                   source_freq: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """Aggregate data to a lower frequency using resample.

        The rows are sorted chronologically first. A period without any
        observation is NaN for every numeric method, ``'sum'`` included
        (``'count'`` gives 0).

        Args:
            data: Time series data to aggregate, on a simple DatetimeIndex
                (panels go through :meth:`convert_frequency`)
            target_freq: Target pandas offset (must be lower than current), with
                optional position ('MS', 'QE', etc.). User labels such as
                ``'quarterly'`` are not offsets and are rejected.
            method: Aggregation method. ``'all'``/``'any'`` reduce boolean
                (or boolean-castable) data: ``'all'`` is True iff every
                sub-period of the target period is present in ``data`` and
                True (a period that is empty, or only partially present at
                the edge of the source index, is always False — this
                full-coverage check is intrinsic to ``'all'`` and applies
                regardless of ``full_periods_only``); ``'any'`` is True iff
                at least one present sub-period is True (an empty period is
                False, matching ``pandas.Series.any()`` on an empty input).
            full_periods_only: If True, periods where the number of non-NaN
                observations is strictly less than the expected sub-period count
                are set to NaN. That count is calendar-based and computed
                period by period (see
                :meth:`count_subperiods_per_period`): a daily-to-monthly
                aggregation expects 28 days in February (29 in a leap year)
                and 31 in January, not a constant ~30. For example, when
                aggregating monthly data to quarterly, a quarter with only 2
                valid months out of 3 will produce NaN. Ignored for
                ``method='all'``/``'any'``, which already encode their own
                (boolean, not NaN) coverage semantics.
            source_freq: Source frequency of the data, used to size the
                expected sub-period count. If None, it is inferred from the
                index. Provide it explicitly when the index frequency differs
                from the variable's own frequency (e.g. a quarterly variable
                carried on a monthly index, or a monthly one carried on a
                daily grid), which is also the case where inference from the
                index is both wrong and silent. Used by
                ``full_periods_only`` and by ``method='all'``. When it is
                neither supplied nor detectable on the index (irregular index,
                fewer than two dates), no expected count exists and both
                coverage guards are skipped.

        Returns:
            Aggregated time series data

        Raises:
            ValueError: If ``data`` is empty, if ``target_freq`` is not a valid
                pandas offset, if ``method`` is unknown, or if a supplied
                ``source_freq`` is not a supported frequency.
            NotImplementedError: If a coverage guard (``full_periods_only``,
                ``method='all'``) meets a multiplied frequency (``'2MS'``).

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> daily_dates = pd.date_range('2023-01-01', periods=31, freq='D')
            >>> daily_series = pd.Series(range(31), index=daily_dates)
            >>> monthly = converter.aggregate_to_lower_frequency(daily_series, 'ME', 'sum')
            >>> len(monthly)
            1
            >>> # Périodes incomplètes remplacées par NaN
            >>> monthly_dates = pd.date_range('2023-01-31', periods=5, freq='ME')
            >>> monthly_series = pd.Series([1, 2, float('nan'), 4, 5], index=monthly_dates)
            >>> quarterly = converter.aggregate_to_lower_frequency(
            ...     monthly_series, 'QE', 'sum', full_periods_only=True
            ... )
            >>> # 'all' : vrai ssi les 12 mois de l'année sont présents et vrais
            >>> monthly_mask = pd.Series(True, index=pd.date_range('2023-01-31', periods=12, freq='ME'))
            >>> converter.aggregate_to_lower_frequency(monthly_mask, 'YE', method='all').tolist()
            [True]
        """
        # Refus des données vides : aucune période à agréger
        self._reject_empty(data, 'aggregate')

        # Modernisation des alias dépréciés ('Y'/'A'/'Q'/'M' -> 'YE'/'YE'/'QE'/'ME')
        # avant tout resample, sans changer la position (S/E) déjà préservée
        target_freq = _modernize_resample_freq(target_freq)

        # Validation que target_freq est un offset pandas valide
        # On ne normalise plus la fréquence pour préserver la position (S/E)
        try:
            to_offset(target_freq)
        except Exception as e:
            raise ValueError(f"Invalid target frequency '{target_freq}': {e}")

        # Validation de la fréquence source fournie par l'appelant : une valeur
        # invalide est une erreur, pas une fréquence « indétectable »
        if source_freq is not None:
            normalize_frequency(source_freq)

        # Tri chronologique des lignes avant resample et détection
        data = self._sorted(data)

        # Resampling à la bonne fréquence (avec position préservée)
        resampled = data.resample(target_freq)

        # Application de la méthode d'agrégation
        if method == 'mean':
            result = resampled.mean()
        elif method == 'sum':
            # Somme d'une période sans observation : NaN (comme 'mean'), et non
            # le 0.0 de pandas (min_count=0), indiscernable d'un vrai zéro
            result = resampled.sum(min_count=1)
        elif method == 'first':
            result = resampled.first()
        elif method == 'last':
            result = resampled.last()
        elif method == 'min':
            result = resampled.min()
        elif method == 'max':
            result = resampled.max()
        elif method == 'median':
            result = resampled.median()
        elif method == 'std':
            result = resampled.std()
        elif method == 'count':
            result = resampled.count()
        elif method == 'all':
            # Vrai ssi toutes les valeurs présentes le sont ; une période sans
            # aucune observation est explicitement fausse (et non vacuously
            # true comme le rendrait `Series.all()` sur une entrée vide)
            result = resampled.agg(lambda s: bool(s.all()) if len(s) > 0 else False)
            # Une période partiellement présente en bord de grille (moins de
            # sous-périodes que n'en compte la fréquence source) ne peut
            # jamais être retenue, même si les valeurs présentes sont toutes
            # vraies : ce garde-fou est intrinsèque à 'all', indépendant de
            # full_periods_only
            result = self._require_full_subperiod_coverage(
                data, target_freq, resampled, result, source_freq=source_freq
            )
        elif method == 'any':
            # Vrai ssi au moins une valeur présente l'est ; une période sans
            # aucune observation est explicitement fausse (déjà le
            # comportement de `Series.any()` sur une entrée vide, rendu
            # explicite ici par symétrie avec 'all')
            result = resampled.agg(lambda s: bool(s.any()) if len(s) > 0 else False)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        # Masquage des périodes incomplètes si demandé ('all'/'any' encodent déjà
        # leur propre sémantique de couverture, en booléen plutôt qu'en NaN)
        if full_periods_only and method not in ('all', 'any'):
            # Nombre attendu de sous-périodes, période cible par période cible
            # (None sans fréquence source fournie ni détectable)
            expected_counts = self._expected_subperiod_counts(
                data, target_freq, result.index, source_freq=source_freq
            )

            # Masquage des périodes incomplètes si le décompte a pu être établi
            if expected_counts is not None:
                valid_counts = resampled.count()
                # Comparaison élément par élément, colonne par colonne pour un DataFrame
                if isinstance(result, pd.DataFrame):
                    fully_covered = valid_counts.ge(expected_counts, axis=0)
                else:
                    fully_covered = valid_counts >= expected_counts
                result = result.where(fully_covered)

        return result

    # Méthode auxiliaire de décompte des sous-périodes attendues par période cible
    def _expected_subperiod_counts(
        self,
        data: Union[pd.Series, pd.DataFrame],
        target_freq: str,
        target_index: pd.DatetimeIndex,
        source_freq: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Count the source sub-periods expected in each target period.

        Single entry point of both coverage guards (``full_periods_only`` and
        ``method='all'``): the source frequency is taken from ``source_freq``
        when the caller knows it, detected on ``data``'s index otherwise, then
        the count is delegated to :meth:`count_subperiods_per_period`, which
        counts calendar period by calendar period (February holds 28 or 29
        daily sub-periods, a quarter 90 to 92, a leap year 366) and falls back
        on a constant ratio only when the frequency bases cannot be expressed
        as pandas Periods.

        Args:
            data: Original data being aggregated, whose index carries the
                source frequency when ``source_freq`` is not supplied.
            target_freq: Target frequency offset string (position and anchor
                allowed, they are stripped before counting).
            target_index: Index of the aggregated result, one entry per
                target period.
            source_freq: Source frequency supplied by the caller. If None, it
                is detected on ``data.index`` — which is only correct when the
                index grid and the variable's own frequency coincide.

        Returns:
            Series of expected sub-period counts aligned on ``target_index``,
            or None when the source frequency is neither supplied nor
            detectable (irregular index, fewer than two dates).

        Raises:
            ValueError: If either frequency cannot be normalized.
            NotImplementedError: If either frequency is multiplied.

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> daily = pd.Series(1.0, index=pd.date_range('2023-01-01', '2023-03-31'))
            >>> monthly_index = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> counts = converter._expected_subperiod_counts(daily, 'ME', monthly_index)
            >>> counts.tolist()
            [31.0, 28.0, 31.0]
        """
        # Fréquence source de l'appelant, sinon détectée sur l'index : sans
        # elle (index irrégulier ou trop court), aucun nombre de sous-périodes
        # attendu n'est définissable
        if source_freq is None:
            try:
                source_freq = detect_index_frequency(index=data.index, return_format='full')
            except ValueError:
                source_freq = None
        if not source_freq:
            return None

        # Le décompte porte sur des périodes de base : les fréquences multipliées sont
        # rejetées avant d'être ramenées à leur base
        self._reject_multiplied('The coverage check (full_periods_only, method=\'all\')', source_freq, target_freq)

        # Extraction des fréquences de base (sans position ni ancrage)
        source_base = normalize_frequency(source_freq, return_format='base')
        target_base = normalize_frequency(target_freq, return_format='base')

        # Décompte calendaire, période cible par période cible (délégué à
        # count_subperiods_per_period, seul dépositaire de cette logique et de
        # son repli sur le facteur constant)
        return pd.Series(
            self.count_subperiods_per_period(target_index, target_base, source_base),
            index=target_index,
        )

    # Méthode auxiliaire de garde-fou de couverture intégrale pour la méthode 'all'
    def _require_full_subperiod_coverage(
        self,
        data: Union[pd.Series, pd.DataFrame],
        target_freq: str,
        resampled: Any,
        result: Union[pd.Series, pd.DataFrame],
        source_freq: Optional[str] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Force False on target periods not fully covered by source sub-periods.

        Used by the ``'all'`` aggregation method: a target period can only be
        True if it contains every sub-period expected at the source
        frequency, not merely the ones actually present in the index.
        Without this check, a period only partially present at the edge of
        the source index (e.g. a yearly bin backed by only 7 of its 12
        months) would evaluate to True whenever the present values happen to
        all be True — exactly the failure mode this method exists to close.

        The expected count is calendar-based and established period by
        period through :meth:`_expected_subperiod_counts`: February expects
        28 daily sub-periods (29 in a leap year), January 31, a quarter 90 to
        92 and a leap year 366. Only when the frequency bases cannot be
        expressed as pandas Periods does the count fall back on a constant
        ratio, identical for every period — a fallback handled inside
        :meth:`count_subperiods_per_period` itself.

        Args:
            data: Original data being aggregated, used to detect the source
                frequency.
            target_freq: Target frequency offset string.
            resampled: The pandas Resampler used to produce ``result``.
            result: Boolean result of the ``'all'`` aggregation, indexed on
                the target frequency.
            source_freq: Source frequency supplied by the caller, forwarded to
                :meth:`_expected_subperiod_counts`. If None, it is detected on
                ``data``'s index.

        Returns:
            ``result`` with any partially covered period forced to False,
            compared element by element (column by column for a DataFrame).
            Unchanged if the source frequency is neither supplied nor
            detectable.
        """
        # Décompte attendu, période cible par période cible : sans fréquence
        # source détectable, aucun garde-fou de couverture n'est applicable
        expected_counts = self._expected_subperiod_counts(
            data, target_freq, result.index, source_freq=source_freq
        )
        if expected_counts is None:
            return result

        valid_counts = resampled.count()

        # Comparaison élément par élément, colonne par colonne pour un DataFrame
        if isinstance(result, pd.DataFrame):
            fully_covered = valid_counts.ge(expected_counts, axis=0)
        else:
            fully_covered = valid_counts >= expected_counts

        return result & fully_covered

    # Méthode d'interpolation à une fréquence plus élevée
    def interpolate_to_higher_frequency(self,
                                      data: Union[pd.Series, pd.DataFrame],
                                      target_freq: str,
                                      method: InterpolationMethod = 'linear',
                                      limit: Union[int, str, None] = None,
                                      limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
                                      limit_area: Optional[Literal['inside', 'outside']] = None,
                                      source_freq: Optional[str] = None,
                                      anchor_fraction: Optional[float] = None) -> Union[pd.Series, pd.DataFrame]:
        """Interpolate data to a higher frequency.

        The rows are sorted chronologically and restricted to the observed
        ones (rows entirely NaN are dropped): a variable carried on a grid
        finer than its own frequency is interpolated from its observations
        only. Each observation is re-stamped on the target period that
        contains it, and the output index is the target grid covering the
        whole source periods of the first and last observations (whatever the
        ratio between the two frequencies, e.g. monthly to weekly). A
        position-less multiplied source (``'2D'``, ``'6h'``) stamps the start
        of its block, a monthly, quarterly, yearly, weekly or semi-monthly one
        without position its end.

        Args:
            data: Time series data to interpolate, on a simple DatetimeIndex
                (panels go through :meth:`convert_frequency`)
            target_freq: Target pandas offset (must be higher than current),
                with optional position ('MS', 'QE', etc.). User labels such as
                ``'daily'`` are not offsets and are rejected.
            method: Interpolation method (same semantics as pandas interpolate)
            limit: Maximum number of consecutive NaN values to fill: an integer
                (numpy integers included). If ``'default'``, uses the frequency
                conversion factor (e.g. 3 for quarterly to monthly), or no
                limit when the source frequency is unknown. If None, no limit
                is applied.
            limit_direction: Direction in which to fill NaN values. If None,
                defaults to ``'forward'`` when target_position is start
                (``'S'``/``'start'``) and ``'backward'`` when target_position is
                end (``'E'``/``'end'``). See
                :meth:`pandas.DataFrame.interpolate` for details.
            limit_area: Restriction on which NaN values to fill. ``'inside'``
                fills only NaN surrounded by valid values, ``'outside'`` fills
                only NaN outside valid values. See
                :meth:`pandas.DataFrame.interpolate` for details.
            source_freq: Source frequency of the data. If None, it is inferred
                from the dates of the observed rows; when it cannot be inferred
                (irregular dates, single observation), the observations are
                placed on the target grid spanning them (``asfreq``), without
                extension to whole periods.
            anchor_fraction: Position, within its own source period, at which
                each observed value is considered reached. ``0.0`` is the start
                of the period, ``0.5`` its middle, ``1.0`` its end. ``None``
                (default) keeps the historical behaviour: the value applies at
                the date it is stamped on (start or end of period, depending on
                the index position), and no re-anchoring beyond
                :meth:`_reanchor_index_to_target` takes place.

                When set, each source timestamp is moved to
                ``period_start + anchor_fraction * period_length`` — the
                **calendar** length of its own source period (365 days in 2021,
                366 in 2024, 28 days in February) — the interpolation runs on
                the **union** of the shifted anchors and the target grid, and
                only the target grid is kept. The shifted timestamps generally
                do not fall on the target grid, so the anchor values themselves
                do not survive in the output: that is the point (at ``0.5`` the
                yearly value no longer claims to hold on 31 December). They are
                never re-injected afterwards.

                Consequences to keep in mind:

                - ``'linear'`` is interpreted by pandas **positionally**, which
                  is meaningless on the irregular union index, so it is treated
                  as ``'time'`` when ``anchor_fraction`` is set (the two agree
                  on the regular grid of the ``None`` path). Every other method
                  already uses the index values and is passed through as is.
                - at the edges of the series, beyond the last shifted anchor,
                  ``limit_direction`` decides, with the defaults resolved by
                  :meth:`_resolve_limit_direction` (``'backward'`` for a target
                  in end position, hence trailing NaN in the example below).
                - ``limit`` counts consecutive NaN on the union index, which is
                  denser than the target grid.
                - if the source frequency base cannot be expressed as a
                  ``pd.Period`` (semi-monthly base), the ``None`` behaviour is
                  silently used instead. A multiplied source frequency
                  (``'2M'``) is rejected: each timestamp would stand for a
                  block of several periods.

                This parameter applies to **interpolation only** — never to
                aggregation, nor to disaggregation by period totals, which stay
                anchored on the exact periods. It composes with a rescaling to
                period totals: the anchor changes the **shape** of the
                interpolation, the rescaling then re-imposes the **total**, in
                that order and without conflict.

        Returns:
            Interpolated time series data; an empty object of the same type
            when no value is observed

        Raises:
            ValueError: If ``data`` has no row, if
                ``target_freq`` is not a valid pandas offset, if ``method`` is
                not a supported interpolation method, if ``limit`` is invalid,
                if ``anchor_fraction`` is neither None nor a real number in
                ``[0, 1]``, or if several observations fall in the same target
                period (the data is not at a lower frequency than the target).
            NotImplementedError: If ``anchor_fraction`` is combined with a
                multiplied source frequency.

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> monthly_dates = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> monthly_series = pd.Series([10, 20, 30], index=monthly_dates)
            >>> daily = converter.interpolate_to_higher_frequency(monthly_series, 'D', 'linear')
            >>> len(daily)
            90

            Yearly series interpolated to quarters. Without an anchor, the
            values are held at the end of each year and the 2022 quarters step
            evenly from 120 to 132:

            >>> yearly = pd.Series(
            ...     [120.0, 132.0],
            ...     index=pd.date_range('2021-12-31', periods=2, freq='YE')
            ... )
            >>> plain = converter.interpolate_to_higher_frequency(yearly, 'Q', method='linear')
            >>> list(plain['2022'].round(2))
            [123.0, 126.0, 129.0, 132.0]

            With ``anchor_fraction=0.5`` the anchors move to the middle of each
            year (2021-07-02 and 2022-07-02), so the first 2022 quarter is
            interpolated between the two shifted points, and the quarters past
            the last anchor are left to ``limit_direction`` (``'backward'``
            here, hence NaN):

            >>> mid = converter.interpolate_to_higher_frequency(
            ...     yearly, 'Q', method='linear', anchor_fraction=0.5
            ... )
            >>> list(mid['2022'].round(2))
            [128.94, 131.93, nan, nan]
        """
        # Validation des paramètres, avant tout calcul : position d'ancrage
        # (sans coercition) et méthode d'interpolation
        if anchor_fraction is not None:
            anchor_fraction = self._validate_anchor_fraction(anchor_fraction)
        if method not in _INTERPOLATION_METHODS:
            raise ValueError(
                f"Unsupported interpolation method: {method}, "
                f"should be in {sorted(_INTERPOLATION_METHODS)}"
            )

        # Refus des données vides : aucune observation à interpoler
        self._reject_empty(data, 'interpolate')

        # Modernisation des alias dépréciés ('Y'/'A'/'Q'/'M' -> 'YE'/'YE'/'QE'/'ME')
        # avant tout resample/asfreq/date_range, sans changer la position (S/E)
        target_freq = _modernize_resample_freq(target_freq)

        # Grille cible à laquelle restreindre le résultat après interpolation sur
        # l'union : renseignée uniquement quand un décalage d'ancre a eu lieu
        final_index: Optional[pd.DatetimeIndex] = None

        # Extraction de la position
        target_position = parse_frequency(frequency_str=target_freq).position
        # Validation que target_freq est un offset pandas valide
        # On ne normalise plus la fréquence pour préserver la position (S/E)
        try:
            to_offset(target_freq)
        except Exception as e:
            raise ValueError(f"Invalid target frequency '{target_freq}': {e}")

        # Restriction aux lignes observées, dans l'ordre chronologique : une
        # variable portée par une grille plus fine que sa propre fréquence (NaN
        # de bourrage) n'est interpolée qu'à partir de ses observations, et
        # l'extension part de la première et de la dernière d'entre elles
        observed = self._sorted(data).dropna(how='all')

        # Aucune valeur observée : aucune date à produire (même règle que
        # convert_frequency, pour une Series comme pour un DataFrame)
        if observed.empty:
            return data.iloc[0:0]

        # Détection de la fréquence source sur les dates observées, sauf si elle
        # est fournie explicitement. La détection peut échouer (une seule
        # observation, dates irrégulières) : repli sur asfreq dans ce cas
        if not source_freq:
            try:
                source_freq = detect_index_frequency(index=observed.index, return_format='full')
            except ValueError:
                source_freq = None

        # Si on peut détecter la fréquence source, on étend l'index pour inclure toutes les périodes
        if source_freq:
            # Extension de l'index pour inclure toutes les périodes intermédiaires
            extended_index = self._extend_index_for_upsampling(
                original_index=observed.index,
                source_freq=source_freq,
                target_freq=target_freq
            )

            # Ré-ancrage de l'index source sur la position cible : sans cela, une
            # source en position 'S' (ex. QS → 2024-01-01) n'intersecte pas une
            # grille cible en position 'E' (ex. ME → 2024-01-31) et la
            # réindexation perdrait toutes les observations
            anchored = observed.copy()
            anchored.index = self._reanchor_index_to_target(
                index=observed.index,
                target_freq=target_freq
            )

            # Deux observations dans une même période cible : la donnée n'est pas
            # d'une fréquence plus basse que la cible
            if anchored.index.has_duplicates:
                raise ValueError(
                    f"Several observations fall in the same '{target_freq}' period: "
                    f"the data is not at a lower frequency than the target"
                )

            if anchor_fraction is None:
                # Réindexation des données sur l'index étendu
                upsampled = anchored.reindex(extended_index)
            else:
                # Décalage des ancres à la fraction demandée de leur période source
                shifted_index = self._shift_index_to_anchor_fraction(
                    index=observed.index,
                    source_freq=source_freq,
                    target_freq=target_freq,
                    anchor_fraction=anchor_fraction
                )

                if shifted_index is None:
                    # Repli sur le comportement anchor_fraction=None quand la base
                    # source ne s'exprime pas en pd.Period
                    upsampled = anchored.reindex(extended_index)
                else:
                    # Les timestamps décalés ne tombent pas sur la grille cible :
                    # l'interpolation se fait sur l'union, la restriction à la
                    # grille cible intervient après
                    anchored.index = shifted_index
                    final_index = extended_index
                    upsampled = anchored.reindex(extended_index.union(shifted_index))

                    # 'linear' est positionnel : sur une union irrégulière il
                    # ignorerait les écarts réels entre timestamps. 'time' est son
                    # équivalent pondéré par le temps, identique sur grille régulière
                    if method == 'linear':
                        method = 'time'
        else:
            # Fallback : utilisation de asfreq si la fréquence source n'est pas détectable
            upsampled = observed.asfreq(target_freq)

        # Arguments d'interpolation : direction et limite résolues (valeurs par défaut)
        interpolate_kwargs = {
            'method': method,
            'limit_direction': self._resolve_limit_direction(
                limit_direction=limit_direction,
                target_position=target_position,
                target_freq=target_freq
            ),
        }
        resolved_limit = self._resolve_interpolation_limit(
            limit=limit,
            source_freq=source_freq,
            target_freq=target_freq
        )
        if resolved_limit is not None:
            interpolate_kwargs['limit'] = resolved_limit
        if limit_area is not None:
            interpolate_kwargs['limit_area'] = limit_area

        # Application de l'interpolation
        result = upsampled.interpolate(**interpolate_kwargs)

        # Restriction à la grille cible : les ancres décalées ne survivent pas
        if final_index is not None:
            result = result.reindex(final_index)

        # Conservation du nom de l'index source (la grille étendue n'en porte pas)
        result.index.name = data.index.name

        return result

    # Méthode auxiliaire de ré-ancrage de l'index source sur la position cible
    def _reanchor_index_to_target(self,
                                  index: pd.DatetimeIndex,
                                  target_freq: str) -> pd.DatetimeIndex:
        """Re-stamp source timestamps on the target frequency position.

        Each source timestamp is mapped to the period of the target base
        frequency that contains it, then re-stamped at the target position
        (start or end). The underlying sub-period is preserved: only the
        start/end convention changes, so a value dated 2024-01-01 on a ``QS``
        index becomes 2024-01-31 when the target is ``ME``.

        This is a no-op when the source and target positions already agree.
        Position-less targets counted in days or finer units (``'D'``, ``'h'``,
        ``'min'``, ...) are stamped at the start of their period, time of day
        kept; semi-monthly targets, which have no ``pd.Period``, roll to the
        ``'SMS'`` anchor at or before each timestamp, or to the ``'SME'``
        anchor at or after it.

        Args:
            index: Source datetime index
            target_freq: Target frequency string, with optional position

        Returns:
            Re-anchored DatetimeIndex

        Examples:
            >>> converter = FrequencyConverter()
            >>> qs_index = pd.date_range('2024-01-01', periods=2, freq='QS')
            >>> converter._reanchor_index_to_target(qs_index, 'ME')
            DatetimeIndex(['2024-01-31', '2024-04-30'], dtype='datetime64[ns]', freq=None)
            >>> converter._reanchor_index_to_target(pd.DatetimeIndex(['2024-01-31']), 'SMS')
            DatetimeIndex(['2024-01-15'], dtype='datetime64[ns]', freq=None)
        """
        # Décomposition de la fréquence cible
        target = normalize_frequency(target_freq, return_format='components')
        target_base = target.freq

        # Base sans position (jours, unités infra-journalières) : la grille cible
        # porte le début de chaque période, à l'heure près — aucune remise à
        # minuit, qui confondrait toutes les heures d'une même journée
        if target_base in BLOCK_START_FREQUENCIES:
            return pd.DatetimeIndex(index.to_period(target_base).to_timestamp(how='start'))

        # Convention du package en l'absence de position explicite : fin de période
        target_pos = target.position if target.position is not None else 'E'

        # Base semi-mensuelle, sans pd.Period : ancre 'SMS' précédente ou ancre
        # 'SME' suivante, selon la position cible
        if target_base in NON_PERIOD_FREQUENCIES:
            offset = to_offset(build_frequency_string(target_base, target_pos))
            roll = offset.rollback if target_pos == 'S' else offset.rollforward
            return pd.DatetimeIndex([roll(timestamp) for timestamp in index]).normalize()

        # Ré-ancrage période par période, au début ou à la fin (jour entier)
        how = 'start' if target_pos == 'S' else 'end'
        return pd.DatetimeIndex(index.to_period(target_base).to_timestamp(how=how).normalize())

    # Méthode auxiliaire de validation de la position d'ancrage
    def _validate_anchor_fraction(self, anchor_fraction: Any) -> float:
        """Validate an anchor position expressed as a fraction of the period.

        Args:
            anchor_fraction: Value to validate, expected to be a real number in
                ``[0, 1]``. ``None`` is handled by the caller and never reaches
                this method.

        Returns:
            The validated value as a float.

        Raises:
            ValueError: If the value is not a real number, or lies outside
                ``[0, 1]``. No silent coercion is performed.

        Examples:
            >>> converter = FrequencyConverter()
            >>> converter._validate_anchor_fraction(0.5)
            0.5
            >>> converter._validate_anchor_fraction(1.5)
            Traceback (most recent call last):
                ...
            ValueError: Invalid anchor_fraction 1.5: expected None or a real number in [0, 1]
        """
        # Rejet des booléens, que isinstance(..., int) accepterait
        if isinstance(anchor_fraction, bool) or not isinstance(
            anchor_fraction, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"Invalid anchor_fraction {anchor_fraction!r}: "
                f"expected None or a real number in [0, 1]"
            )

        # Bornes incluses ; NaN est rejeté par la comparaison
        if not 0.0 <= float(anchor_fraction) <= 1.0:
            raise ValueError(
                f"Invalid anchor_fraction {anchor_fraction}: "
                f"expected None or a real number in [0, 1]"
            )

        return float(anchor_fraction)

    # Méthode auxiliaire de décalage de l'index à une fraction de la période source
    def _shift_index_to_anchor_fraction(self,
                                        index: pd.DatetimeIndex,
                                        source_freq: str,
                                        target_freq: str,
                                        anchor_fraction: float) -> Optional[pd.DatetimeIndex]:
        """Move each timestamp to a fraction of its own source period.

        Each timestamp is mapped to the source period that contains it, then
        re-stamped at ``period_start + anchor_fraction * period_length``. The
        length is the **calendar** length of that very period — 365 days in
        2021, 366 in 2024, 28 days in February — not an average duration, so
        the anchor of a short period is not pushed outside of it.

        The result no longer depends on the position of the source index: a
        yearly value stamped 2021-01-01 (``YS``) and one stamped 2021-12-31
        (``YE``) both anchor at the same fraction of 2021.

        Args:
            index: Source datetime index, one entry per observed period.
            source_freq: Source frequency, with optional position. A multiplied
                frequency (``'2M'``) is rejected: each timestamp would stand
                for a block of several periods.
            target_freq: Target frequency, used to decide whether sub-daily
                precision matters.
            anchor_fraction: Validated fraction in ``[0, 1]``.

        Returns:
            Shifted DatetimeIndex, or None if the source base cannot be
            expressed as a ``pd.Period`` — in which case the caller falls back
            on the ``anchor_fraction=None`` behaviour.

        Examples:
            >>> converter = FrequencyConverter()
            >>> yearly = pd.date_range('2021-12-31', periods=2, freq='YE')
            >>> converter._shift_index_to_anchor_fraction(yearly, 'YE', 'QE', 0.5)
            DatetimeIndex(['2021-07-02', '2022-07-02'], dtype='datetime64[ns]', freq=None)
            >>> converter._shift_index_to_anchor_fraction(yearly, 'YE', 'QE', 0.0)
            DatetimeIndex(['2021-01-01', '2022-01-01'], dtype='datetime64[ns]', freq=None)
            >>> converter._shift_index_to_anchor_fraction(yearly, 'YE', 'QE', 1.0)
            DatetimeIndex(['2021-12-31', '2022-12-31'], dtype='datetime64[ns]', freq=None)
        """
        # Un timestamp d'index multiplié représente un bloc de n périodes, non une période
        self._reject_multiplied('anchor_fraction', source_freq)

        # Normalisation de la base source (sans position S/E)
        source_base = normalize_frequency(source_freq, return_format='base')

        # Base sans pd.Period (semi-mensuelle) : repli de l'appelant sur le
        # comportement anchor_fraction=None
        if source_base in NON_PERIOD_FREQUENCIES:
            return None

        # Périodes source réelles
        periods = index.to_period(source_base)
        starts = periods.start_time
        ends = periods.end_time

        # Durée calendaire pleine de chaque période : end_time est la dernière
        # nanoseconde de la période, d'où le +1 ns. La longueur pleine est de
        # surcroît exactement représentable en float64, ce que n'est pas
        # (end_time - start_time) pour une période annuelle
        lengths = (ends - starts) + pd.Timedelta(1, unit='ns')
        shifted = pd.DatetimeIndex(starts + anchor_fraction * lengths)

        # Bornage à la fin de période : sans lui, anchor_fraction=1.0 tomberait
        # sur le début de la période suivante
        shifted = pd.DatetimeIndex(np.minimum(shifted.values, ends.values))

        # Ramené à minuit tant que la grille cible n'est pas infra-journalière :
        # une cible horaire perdrait sinon toute la position intra-journalière
        target_base = normalize_frequency(target_freq, return_format='base')
        if not is_higher_frequency(target_base, 'D'):
            shifted = shifted.normalize()

        return shifted

    # Méthode auxiliaire de résolution de la valeur par défaut de limit
    def _resolve_interpolation_limit(self,
                                     limit: Union[int, str, None],
                                     source_freq: Optional[str],
                                     target_freq: str) -> Optional[int]:
        """Resolve the interpolation limit value.

        When ``limit`` is ``'default'``, computes the frequency conversion
        factor between the source (low) and target (high) frequency.

        Args:
            limit: Raw limit value (integer — numpy integers included —,
                ``'default'``, or None)
            source_freq: Detected source frequency (may be None)
            target_freq: Target frequency string

        Returns:
            Resolved integer limit, or None for no limit (``None``, or
            ``'default'`` without a known source frequency)

        Raises:
            ValueError: If ``limit`` is neither None, ``'default'`` nor an
                integer (booleans, floats and other strings are rejected).

        Examples:
            >>> converter = FrequencyConverter()
            >>> converter._resolve_interpolation_limit('default', 'QS', 'MS')
            3
            >>> converter._resolve_interpolation_limit(np.int64(2), 'QS', 'MS')
            2
        """
        # Absence de limite
        if limit is None:
            return None

        # Entier Python ou numpy (les booléens, sous-classe d'int, sont refusés)
        if isinstance(limit, numbers.Integral) and not isinstance(limit, (bool, np.bool_)):
            return int(limit)

        # Cas 'default' : facteur de conversion (multiplicateurs compris),
        # « combien de périodes cibles dans une période source » ; sans
        # fréquence source connue, aucune limite
        if isinstance(limit, str) and limit == 'default':
            if not source_freq:
                return None
            return int(round(self.get_conversion_factor(target_freq, source_freq)))

        raise ValueError(
            f"Invalid limit {limit!r}: expected None, 'default' or an integer"
        )

    # Méthode auxiliaire de résolution de la direction d'interpolation
    def _resolve_limit_direction(self,
                                 limit_direction: Optional[str],
                                 target_position: Optional[str],
                                 target_freq: str) -> str:
        """Resolve the default limit_direction based on target position.

        Args:
            limit_direction: Explicit direction or None
            target_position: Target position (``'S'``, ``'E'``, etc.) or None
            target_freq: Target frequency string (used as fallback to extract
                the position)

        Returns:
            ``limit_direction`` when given, otherwise ``'forward'`` for a
            start position and ``'backward'`` for an end position (the
            package convention without explicit position)
        """
        # Valeur explicite : retour immédiat
        if limit_direction is not None:
            return limit_direction

        # Résolution de la position cible ; convention du package en l'absence
        # de position explicite : fin de période
        position = target_position or parse_frequency(target_freq).position or 'E'

        # Conversion en direction
        return 'forward' if normalize_position(position) == 'S' else 'backward'

    # Méthode auxiliaire de validation des paramètres
    def _validate_conversion_params(self,
                                  data: Union[pd.Series, pd.DataFrame],
                                  target_freq: Union[str, Dict[str, str]],
                                  time_col: Optional[str]=None,
                                  panel_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Validate conversion parameters.

        Args:
            data: Input data
            target_freq: Target frequency (str or dict)
            time_col: Time identifier column
            panel_cols: Panel identifier columns

        Returns:
            Validated data, sorted

        Raises:
            ValueError: If the data is empty or if parameters are invalid
        """
        # Vérification du jeu de données
        data = validate_temporal_data(data=data, time_col=time_col, panel_cols=panel_cols, strict=True, sort_data=True, return_metadata=False)

        # Refus des données vides (Series, DataFrame ou panel sans aucune ligne)
        self._reject_empty(data, 'convert')

        # Vérification que la fréquence cible est spécifiée
        if not target_freq:
            raise ValueError("Target frequency cannot be empty")

        # Validation de target_freq selon son type. Seule la base est à valider :
        # parse_frequency ne rend comme position que 'S', 'E' ou None
        if isinstance(target_freq, str):
            try:
                normalize_frequency(parse_frequency(target_freq).freq)
            except ValueError as e:
                raise ValueError(f"Invalid target frequency: {e}")
        elif isinstance(target_freq, dict):
            # Détection de la structure de panel (entités en MultiIndex)
            is_panel = is_panel_data(data)

            # Un dictionnaire n'est autorisé que pour un DataFrame ou un panel (Series MultiIndex)
            if isinstance(data, pd.Series) and not is_panel:
                raise ValueError("Dictionary target_freq is only valid for DataFrame or panel inputs")

            # Colonnes disponibles (hors colonnes de panel)
            if isinstance(data, pd.DataFrame):
                data_cols = set(data.columns)
                if panel_cols:
                    data_cols -= set(panel_cols)
            else:
                data_cols = set()

            # Hors du cas panel : seules des clés de colonnes (str) existantes sont autorisées
            if not is_panel:
                # Rejet des clés tuple (entité / (entité, colonne)) sans structure de panel
                non_column_keys = {k for k in target_freq if not isinstance(k, str)}
                if non_column_keys:
                    raise ValueError(
                        f"Tuple keys in target_freq are only valid for panel inputs: {non_column_keys}"
                    )
                # Vérification que les colonnes désignées existent
                missing_cols = set(target_freq.keys()) - data_cols
                if missing_cols:
                    raise ValueError(f"Columns in target_freq not found in data: {missing_cols}")
            # En panel, les clés peuvent être une colonne (str), une entité (tuple) ou un
            # couple (entité..., colonne) (tuple) : aucune contrainte de sous-ensemble

            # Validation de chaque fréquence cible (clé = colonne, entité ou (entité, colonne))
            for key, freq in target_freq.items():
                try:
                    normalize_frequency(parse_frequency(freq).freq)
                except ValueError as e:
                    raise ValueError(f"Invalid target frequency for key '{key}': {e}")
        else:
            raise ValueError("target_freq must be a string or dictionary")

        # Validation des panel_cols si spécifiés
        if panel_cols:
            # Vérification que les panel_cols ne sont pas dans target_freq si dict
            if isinstance(target_freq, dict):
                overlap = set(panel_cols) & set(target_freq.keys())
                if overlap:
                    raise ValueError(f"Panel columns cannot be in target_freq: {overlap}")

        return data

    # Méthode auxiliaire de conversion des données de panel entité par entité
    def _convert_panel_frequency(self,
                                 data: Union[pd.Series, pd.DataFrame],
                                 target_freq: Union[str, Dict[str, str]],
                                 method: str,
                                 alignment_method: str,
                                 target_position: Optional[str],
                                 full_periods_only: bool,
                                 limit: Union[int, Literal['default'], None],
                                 limit_direction: Optional[str],
                                 limit_area: Optional[str]) -> Union[pd.Series, pd.DataFrame]:
        """Convert panel (MultiIndex) data to target frequency, entity by entity.

        Each entity is converted independently on a simple DatetimeIndex, so the
        conversion direction (up/down sampling) is resolved per ``(entity, column)``
        from each entity's own source frequency. This preserves entity-specific
        target frequencies when ``target_freq`` is an ``{entity: freq}`` mapping.

        Args:
            data: Panel data with a MultiIndex (entity levels + time level).
            target_freq: Target frequency (str, {col: freq}, or {entity: freq}).
            method: Aggregation or interpolation method.
            alignment_method: Method to align indexes for mixed frequencies.
            target_position: Optional target position.
            full_periods_only: If True, incomplete periods produce NaN (downsampling).
            limit: Maximum consecutive NaN to fill (upsampling).
            limit_direction: Direction for NaN filling (upsampling).
            limit_area: Restriction area for NaN filling (upsampling).

        Returns:
            Converted panel data with a MultiIndex combining each entity with its
            converted time index. An entity that no key targets is returned
            unchanged, for a Series as for a DataFrame; an entity without any
            observation of the converted columns is left out.
        """
        # Import local pour éviter les imports circulaires
        from ...panel.utils import get_unique_panel_entities, get_entity_mask

        # Colonnes disponibles (None pour une Series, repérée par son nom)
        columns = list(data.columns) if isinstance(data, pd.DataFrame) else None
        series_name = data.name if isinstance(data, pd.Series) else None
        # Niveaux d'entité (tous sauf le dernier, temporel)
        entity_levels = list(range(data.index.nlevels - 1))
        # Noms des niveaux de l'index d'origine
        index_names = data.index.names

        # Accumulation des résultats par entité
        converted_parts = []

        # Parcours des entités du panel
        for entity in get_unique_panel_entities(data):
            # Masque des observations de l'entité
            entity_mask = get_entity_mask(data, entity)
            # Extraction du sous-jeu de l'entité avec un index temporel simple
            entity_data = data.loc[entity_mask].droplevel(entity_levels)

            # Résolution de la fréquence cible propre à l'entité
            entity_target = self._resolve_panel_target(target_freq, entity, columns, series_name)

            # Cas d'une entité sans aucune colonne ciblée : conservation telle quelle
            if isinstance(entity_target, dict) and not entity_target:
                converted_parts.append(data.loc[entity_mask])
                continue

            # Conversion sur l'index simple (chemin non-panel, direction par colonne)
            converted = self.convert_frequency(
                data=entity_data,
                target_freq=entity_target,
                method=method,
                alignment_method=alignment_method,
                target_position=target_position,
                full_periods_only=full_periods_only,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
            )

            # Entité sans aucune observation des colonnes converties : aucune
            # date à conserver, l'entité disparaît de la sortie
            if converted.empty:
                continue

            # Reconstruction de l'index MultiIndex en réattachant l'entité
            new_index = pd.MultiIndex.from_tuples(
                [(*entity, time) for time in converted.index],
                names=index_names,
            )
            converted = converted.copy()
            converted.index = new_index

            converted_parts.append(converted)

        # Aucune entité observée : aucune ligne
        if not converted_parts:
            return data.iloc[0:0]

        # Concaténation des entités et tri de l'index
        result = pd.concat(converted_parts)
        result = result.sort_index()

        return result

    # Méthode auxiliaire de résolution de la fréquence cible d'une entité de panel
    def _resolve_panel_target(self,
                              target_freq: Union[str, Dict[str, str]],
                              entity: tuple,
                              columns: Optional[List[str]],
                              series_name: Any = None) -> Union[str, Dict[str, str]]:
        """Resolve the target frequency applicable to a given panel entity.

        Supports mixed dict keys: ``(entity..., column)`` tuples, ``(entity...)``
        tuples, or plain column names, with precedence
        ``(entity, column)`` > ``(entity,)`` > ``column`` (see
        :func:`resolve_entity_column_frequencies`). A panel Series is resolved
        as a single column named after the Series.

        Args:
            target_freq: Target specification (str or dict with mixed keys).
            entity: Entity tuple.
            columns: Available data columns (None for Series inputs).
            series_name: Name of the Series, for Series inputs.

        Returns:
            A frequency string (for str inputs or targeted Series entities) or
            a ``{col: freq}`` dict (for DataFrame entities). Columns not covered
            by the mapping are omitted from the dict, and an untargeted Series
            entity gives an empty dict: the caller leaves them unchanged.
        """
        # Import local pour éviter les imports circulaires
        from ...panel.utils import resolve_entity_column_frequencies

        # Cas d'une chaîne : même cible pour toutes les entités et colonnes
        if not isinstance(target_freq, dict):
            return target_freq

        # Cas Series : une colonne unique, repérée par le nom de la Series ;
        # entité non ciblée → dict vide (conservée telle quelle par l'appelant)
        if columns is None:
            resolved = resolve_entity_column_frequencies(entity, [series_name], target_freq)
            return resolved.get(series_name, {})

        # Cas DataFrame : résolution par colonne selon la précédence de spécificité
        return resolve_entity_column_frequencies(entity, columns, target_freq)

    # Méthode auxiliaire de construction d'un mapping associant à chaque colonne la
    def _build_frequency_map(self,
                            data: pd.DataFrame,
                            target_freq: Union[str, Dict[str, str]],
                            target_position: Optional[str]) -> Dict[str, Tuple[str, str]]:
        """Build complete frequency map for DataFrame conversion.

        The target position of each column is, in order: ``target_position``,
        the position carried by its target frequency, the position of the
        column's own source frequency (as for a Series), the default 'E'.

        Args:
            data: Input DataFrame
            target_freq: Target frequency (str or dict)
            target_position: Explicit target position code ('S', 'E') or None

        Returns:
            Dictionary mapping the columns with a detectable frequency to
            (source_freq, target_freq) tuples; undetectable columns and
            columns absent from a dict ``target_freq`` are left out

        Raises:
            ValueError: If the data holds observations but no column has a
                detectable frequency.
        """
        # Détection des fréquences source de chaque colonne (index simple → {col: freq})
        # sous forme décomposée : la position source sert aussi de repli à la cible
        sources = {
            col: parsed
            for col, parsed in detect_dataset_frequency(data, return_format='components').items()
            if parsed
        }

        # Des observations sans aucune fréquence détectable : erreur, comme pour
        # une Series (un cadre sans aucune observation, lui, rend zéro ligne)
        if not sources and data.notna().any().any():
            raise ValueError("Cannot detect current frequency of any column of the data")

        # Fréquence cible de chaque colonne détectée (même cible pour toutes si chaîne)
        if isinstance(target_freq, str):
            targets = {col: target_freq for col in data.columns if col in sources}
        else:
            targets = {col: target for col, target in target_freq.items() if col in sources}

        # Construction du frequency_map : fréquences complètes (base + position +
        # multiplicateur), position explicite prioritaire, puis celle de la cible,
        # puis celle de la source (variables locales par colonne)
        frequency_map = {}
        for col, target in targets.items():
            source = sources[col]
            col_target = normalize_frequency(target, return_format='components')
            col_position = target_position or col_target.position or source.position
            frequency_map[col] = (
                self._with_position(source, source.position),
                self._with_position(col_target, col_position),
            )

        return frequency_map

    # Méthode auxiliaire de groupement des conversions par opération
    def _group_conversions_by_operation(self,
                                       frequency_map: Dict[str, Tuple[str, str]],
                                       method: str) -> Dict[Tuple[str, str, str], List[str]]:
        """Group conversions by identical operations for efficiency.

        Args:
            frequency_map: Dictionary mapping columns to (source_freq, target_freq)
            method: Conversion method

        Returns:
            Dictionary mapping (source_freq, target_freq, method) to list of columns
        """
        # Initialisation du dictionnaire associant une transformation à un ensemble de colonnes
        grouped = {}

        # Parcours du mapping
        for col, (source_freq, target_freq) in frequency_map.items():
            # Ignore les colonnes dont la fréquence ne change pas
            if source_freq == target_freq:
                continue

            # Clé de groupement
            key = (source_freq, target_freq, method)

            # Ajout de la colonne au groupe
            if key not in grouped:
                grouped[key] = []
            # Ajout de la colonne à la clé
            grouped[key].append(col)

        return grouped

    # Méthode auxiliaire d'application des conversions groupées
    def _apply_grouped_conversions(self,
                                  data: pd.DataFrame,
                                  grouped_conversions: Dict[Tuple[str, str, str], List[str]],
                                  alignment_method: str,
                                  full_periods_only: bool = False,
                                  limit: Union[int, str, None] = None,
                                  limit_direction: Optional[str] = None,
                                  limit_area: Optional[str] = None) -> pd.DataFrame:
        """Apply grouped conversions and assemble them on the union of their indexes.

        Each group of columns is converted by
        :meth:`interpolate_to_higher_frequency` or
        :meth:`aggregate_to_lower_frequency`. The output index is the union of
        the target indexes of the converted columns and of the observed dates
        of the columns left unconverted (absent from a dict target, already at
        their target frequency, or without a detectable frequency): the rows
        of the source grid do not survive, and the result does not depend on
        the order of the columns. When nothing is converted, the rows where
        at least one column is observed are returned.

        Args:
            data: Input DataFrame
            grouped_conversions: Dictionary of grouped conversions
            alignment_method: Method filling the NaN a converted column gets on
                the union when frequencies are mixed ('ffill', 'bfill',
                'nearest', 'none')
            full_periods_only: If True, incomplete periods produce NaN
                (downsampling only)
            limit: Maximum number of consecutive NaN to fill (upsampling only)
            limit_direction: Direction for NaN filling (upsampling only)
            limit_area: Restriction area for NaN filling (upsampling only)

        Returns:
            DataFrame with all conversions applied, columns in their original
            order, index named like the source index
        """
        # Conversion de chaque groupe (colonnes de même source, cible et méthode)
        converted_columns: Dict[str, pd.Series] = {}
        for (source_freq, target_freq, conv_method), columns in grouped_conversions.items():
            subset = data[columns]

            # Direction de conversion (bases et multiplicateurs ; positions et
            # ancres sont sans effet sur l'ordre des fréquences) ; la fréquence
            # source du groupe est déjà détectée
            if is_higher_frequency(target_freq, source_freq):
                converted = self.interpolate_to_higher_frequency(
                    subset, target_freq, conv_method,
                    limit=limit, limit_direction=limit_direction,
                    limit_area=limit_area, source_freq=source_freq,
                )
            else:
                converted = self.aggregate_to_lower_frequency(
                    subset, target_freq, conv_method, full_periods_only,
                    source_freq=source_freq,
                )

            for col in columns:
                converted_columns[col] = converted[col]

        # Rien à convertir : lignes où au moins une colonne est observée
        if not converted_columns:
            observed_rows = data.notna().any(axis=1)
            return data if observed_rows.all() else data.loc[observed_rows]

        # Colonnes conservées (hors dict, déjà à la cible ou sans fréquence
        # détectable) : valeurs aux seules dates observées
        preserved_columns = {
            col: data[col].dropna() for col in data.columns if col not in converted_columns
        }

        # Index de sortie : union des index cibles et des dates observées des
        # colonnes conservées, indépendante de l'ordre des colonnes
        unified_index = data.index[:0]
        for series in (*converted_columns.values(), *preserved_columns.values()):
            unified_index = unified_index.union(series.index)

        # Fréquences mélangées : plusieurs cibles, ou des colonnes conservées
        # observées ; seul ce mélange introduit des NaN à combler
        mixed = (
            len({target_freq for (_, target_freq, _) in grouped_conversions}) > 1
            or any(not series.empty for series in preserved_columns.values())
        )

        # Réindexation de chaque colonne sur l'union, dans l'ordre d'origine
        result = pd.DataFrame(index=unified_index)
        for col in data.columns:
            if col in converted_columns:
                series = converted_columns[col].reindex(unified_index)
                result[col] = self._fill_union_gaps(series, alignment_method) if mixed else series
            else:
                result[col] = preserved_columns[col].reindex(unified_index)

        # Nom de l'index source conservé
        result.index.name = data.index.name
        return result

    # Méthode auxiliaire de comblement des NaN introduits par l'union des index
    @staticmethod
    def _fill_union_gaps(series: pd.Series, alignment_method: str) -> pd.Series:
        """Fill the NaN a converted column gets on a mixed-frequency union.

        Args:
            series: Converted column reindexed on the union of indexes.
            alignment_method: 'ffill', 'bfill', 'nearest' (in time, without
                extrapolation) or 'none'.

        Returns:
            The filled column ('none': unchanged).

        Examples:
            >>> s = pd.Series([1.0, None, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
            >>> FrequencyConverter._fill_union_gaps(s, 'ffill').tolist()
            [1.0, 1.0, 3.0]
        """
        if alignment_method == 'ffill':
            return series.ffill()
        if alignment_method == 'bfill':
            return series.bfill()
        if alignment_method == 'nearest':
            return series.interpolate(method='nearest')
        # 'none' : NaN conservés
        return series

    # Méthode auxiliaire de refus des données vides
    @staticmethod
    def _reject_empty(data: Union[pd.Series, pd.DataFrame], operation: str) -> None:
        """Reject data without any row.

        Args:
            data: Series, DataFrame or panel to check.
            operation: Name of the operation, for the error message.

        Raises:
            ValueError: If ``data`` has no row.

        Examples:
            >>> FrequencyConverter._reject_empty(pd.Series([], dtype=float), 'convert')
            Traceback (most recent call last):
                ...
            ValueError: Cannot convert empty data: no row to convert
        """
        if len(data) == 0:
            raise ValueError(f"Cannot {operation} empty data: no row to {operation}")

    # Méthode auxiliaire de tri chronologique
    @staticmethod
    def _sorted(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Sort data by date, unless it is already in increasing order.

        Args:
            data: Series or DataFrame on a DatetimeIndex.

        Returns:
            ``data`` itself when its index is increasing, a sorted copy otherwise.
        """
        return data if data.index.is_monotonic_increasing else data.sort_index()

    # Méthode auxiliaire de bornage des blocs d'une base sans pd.Period
    @staticmethod
    def _offset_block_bounds(start_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             base: str,
                             position: str,
                             multiplier: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Bounds of the source blocks of a base without ``pd.Period`` (semi-monthly).

        The periods are delimited by the pandas offset of the base at the
        given position: a ``'SMS'`` stamp opens its half-month, a ``'SME'``
        stamp closes it.

        Args:
            start_date: First source timestamp.
            end_date: Last source timestamp.
            base: Frequency base (``'SM'``).
            position: ``'S'`` or ``'E'``.
            multiplier: Number of base periods in a block.

        Returns:
            ``(start, end)``: first instant of the first block, last instant of
            the last block.

        Examples:
            >>> FrequencyConverter._offset_block_bounds(
            ...     pd.Timestamp('2024-01-15'), pd.Timestamp('2024-02-29'), 'SM', 'E', 1
            ... )
            (Timestamp('2024-01-01 00:00:00'), Timestamp('2024-02-29 23:59:59.999999999'))
        """
        offset = to_offset(build_frequency_string(base, position))
        one_day = pd.Timedelta(1, unit='D')
        one_ns = pd.Timedelta(1, unit='ns')
        if position == 'S':
            # Début de bloc : de l'ancre qui ouvre le premier bloc à la veille de
            # l'ancre qui suit le dernier
            start = offset.rollback(start_date).normalize()
            end = offset.rollback(end_date).normalize() + multiplier * offset - one_ns
        else:
            # Fin de bloc : du lendemain de l'ancre qui précède le premier bloc à
            # la fin du jour de l'ancre qui ferme le dernier
            start = offset.rollforward(start_date).normalize() - multiplier * offset + one_day
            end = offset.rollforward(end_date).normalize() + one_day - one_ns
        return start, end

    # Méthode auxiliaire d'extension de l'index pour l'upsampling
    def _extend_index_for_upsampling(self,
                                     original_index: pd.DatetimeIndex,
                                     source_freq: str,
                                     target_freq: str) -> pd.DatetimeIndex:
        """Extend the index to the whole source periods when upsampling.

        The returned index is the target grid running from the start of the
        source period (or block, for a multiplied source) of the first
        timestamp to the end of the period of the last one, whatever the
        ratio between the two frequencies (monthly to weekly included). A
        position-less source counted in days or finer units (``'2D'``,
        ``'6h'``) stamps the start of its block, like the labels of
        ``pandas.resample``; the other position-less sources (``'W'``,
        ``'SM'``, bare ``'M'`` / ``'Q'`` / ``'Y'``) stamp its end.
        Semi-monthly sources, which have no ``pd.Period``, are bounded with
        the ``'SMS'`` / ``'SME'`` offsets.

        Args:
            original_index: Sorted source timestamps.
            source_freq: Source frequency, with optional position ('QE', 'QS').
            target_freq: Target frequency, with optional position ('ME', 'MS').

        Returns:
            The target grid covering the source periods, or ``original_index``
            itself when there is nothing to extend: same base and multiplier,
            or a target that is not finer than the source.

        Examples:
            >>> converter = FrequencyConverter()
            >>> qe_index = pd.date_range('2024-03-31', periods=4, freq='QE')
            >>> len(converter._extend_index_for_upsampling(qe_index, 'QE', 'ME'))
            12
            >>> me_index = pd.date_range('2024-01-31', periods=3, freq='ME')
            >>> converter._extend_index_for_upsampling(me_index, 'ME', 'W')[[0, -1]]
            DatetimeIndex(['2024-01-07', '2024-03-31'], dtype='datetime64[ns]', freq=None)
        """
        # Extraction des informations de fréquence, position et multiplicateur
        source = normalize_frequency(source_freq, return_format='components')
        target = normalize_frequency(target_freq, return_format='components')
        source_base, source_multiplier = source.freq, source.multiplier

        # Même fréquence (base et multiplicateur) : pas d'extension nécessaire
        if source_base == target.freq and source_multiplier == target.multiplier:
            return original_index

        # Cible pas plus fine que la source (rapport de durées < 1) : pas
        # d'extension. Le rapport n'a pas à être entier (M → W : 30/7), la
        # grille cible étant construite par date_range sur les bornes des périodes
        ratio = get_duration_conversion_factor(
            build_frequency_string(source_base, multiplier=source_multiplier),
            build_frequency_string(target.freq, multiplier=target.multiplier),
        )
        if ratio < 1:
            return original_index

        # Position de l'horodatage dans son bloc : explicite, sinon début pour les
        # bases comptées en jours ou unités infra-journalières, fin pour les autres
        if source.position is not None:
            source_pos = source.position
        else:
            source_pos = 'S' if source_base in BLOCK_START_FREQUENCIES else 'E'

        # Bornes de la plage : un horodatage multiplié couvre un bloc de n
        # périodes de base, à partir de lui en position début, jusqu'à lui en
        # position fin. pd.Period n'accepte que la base (sans S/E)
        start_date, end_date = original_index[0], original_index[-1]
        if source_base in NON_PERIOD_FREQUENCIES:
            extended_start, extended_end = self._offset_block_bounds(
                start_date, end_date, source_base, source_pos, source_multiplier
            )
        else:
            block_extra = source_multiplier - 1
            first_period = pd.Period(start_date, freq=source_base)
            last_period = pd.Period(end_date, freq=source_base)
            if source_pos == 'S':
                extended_start = first_period.to_timestamp(how='start')
                extended_end = (last_period + block_extra).to_timestamp(how='end')
            else:
                extended_start = (first_period - block_extra).to_timestamp(how='start')
                extended_end = last_period.to_timestamp(how='end')

        # Grille cible complète (position S/E et multiplicateur de la cible inclus)
        return pd.date_range(start=extended_start, end=extended_end, freq=target_freq)
