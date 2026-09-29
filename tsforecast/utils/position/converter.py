"""Period position conversion utilities for time series processing.

This module provides the PeriodPositionConverter class to handle conversions
between start and end period positions for time series data.

Conventions:
    - Only frequencies with a start/end variant in pandas carry a position:
      monthly ('MS' / 'ME'), quarterly ('QS' / 'QE'), yearly ('YS' / 'YE')
      and semi-monthly ('SMS' / 'SME'), with or without a multiplier and an
      anchor ('2MS', 'QS-FEB', 'YE-JUN'). Every other frequency ('D', 'B',
      'W-SUN', 'h', ...) has no position: conversions leave it unchanged.
    - A converted date always lands on the native pandas grid of the target
      offset (midnight of the last day for an end position, like
      ``pd.date_range(freq='ME')``), within the period of its source date.
    - Semi-monthly periods are identified by their rank in the month: the
      'SMS' dates (1st, 15th) pair with the 'SME' dates (15th, month end).
"""
# Importation des modules
import pandas as pd
from typing import Optional, Tuple, Union
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import DateOffset

# Import de la classe parente
from ..abc.converter import TemporalConverter

# Import du normalizer et des types
from .types import PositionType, UserPositionType
from .utils import normalize_position
from ..parse.utils import MONTH_ABBREVIATIONS, ParsedFrequency, parse_frequency, build_frequency_string
# Import des utilitaires de validation
from ..validation import validate_entities_grouped, validate_sorted_within_groups

# Fréquences de base dont les périodes sont décrites par un couple d'offsets début / fin
_PERIOD_FREQUENCIES = ('M', 'Q', 'Y')

# Jour de bascule par défaut des fréquences semi-mensuelles pandas ('SMS' = 1 et 15,
# 'SME' = 15 et fin de mois) : seul jour supporté
_SEMI_MONTH_DAY = 15


# Fonction de décalage d'une ancre mensuelle entre positions début et fin
def _shift_anchor(month: str, from_pos: PositionType, to_pos: PositionType) -> str:
    """Shift a month anchor so that the offset describes the same periods at another position.

    A period starting in month ``m`` ends in month ``m - 1`` of the following
    cycle: ``QS-FEB`` (quarters Feb-Apr, May-Jul, ...) pairs with ``QE-JAN``,
    ``YS-JUL`` (fiscal years July-June) with ``YE-JUN``.

    Args:
        month: Anchor month abbreviation ('JAN' ... 'DEC').
        from_pos: Position the anchor refers to ('S' or 'E').
        to_pos: Target position ('S' or 'E').

    Returns:
        The anchor month for the target position.

    Raises:
        ValueError: If ``month`` is not a pandas month abbreviation.

    Examples:
        >>> _shift_anchor('FEB', 'S', 'E')
        'JAN'
        >>> _shift_anchor('NOV', 'E', 'S')
        'DEC'
        >>> _shift_anchor('DEC', 'E', 'E')
        'DEC'
    """
    if month not in MONTH_ABBREVIATIONS:
        raise ValueError(f"Unsupported anchor month: '{month}'. Expected one of {list(MONTH_ABBREVIATIONS)}")
    if from_pos == to_pos:
        return month
    # Début -> fin : mois précédent ; fin -> début : mois suivant (modulo 12)
    step = -1 if (from_pos, to_pos) == ('S', 'E') else 1
    return MONTH_ABBREVIATIONS[(MONTH_ABBREVIATIONS.index(month) + step) % 12]


# Classe de conversion entre positions de période
class PeriodPositionConverter(TemporalConverter):
    """Handle conversions between period positions (start vs end).

    This class manages conversions between start and end period positions for
    time series data: each date moves to the other bound of its own period
    (e.g. 2024-02-01 in 'MS' <-> 2024-02-29 in 'ME'), honouring multipliers
    and anchors of the frequency. Frequencies without a start/end variant in
    pandas ('D', 'W', 'B', 'h', ...) are left unchanged.

    Examples:
        >>> converter = PeriodPositionConverter()
        >>> dates = pd.date_range('2024-01-01', periods=3, freq='MS')
        >>> series = pd.Series([1, 2, 3], index=dates)
        >>> converter.convert(series, 'start', 'end', freq='M').index.equals(
        ...     pd.date_range('2024-01-31', periods=3, freq='ME'))
        True
    """

    # Initialisation
    def __init__(self):
        """Initialize conversion utilities and normalizer."""

    # Méthode principale de conversion
    def convert(
        self,
        value: Union[pd.Series, pd.DataFrame, pd.DatetimeIndex],
        from_unit: Union[PositionType, UserPositionType],
        to_unit: Union[PositionType, UserPositionType],
        freq: Optional[str] = None,
        **kwargs
    ) -> Union[pd.Series, pd.DataFrame, pd.DatetimeIndex]:
        """Convert time series data from one period position to another.

        Implementation of TemporalConverter.convert() for period positions.

        Args:
            value: Time series data or DatetimeIndex to convert
            from_unit: Source position ('S', 'E', 'start', 'end')
            to_unit: Target position ('S', 'E', 'start', 'end')
            freq: Frequency of the time series (e.g., 'M', 'QS-FEB', '2MS').
                  If None, will attempt to infer from the data (separately
                  for each entity of a panel).
            **kwargs: Additional parameters (unused for position conversion)

        Returns:
            Converted time series data with adjusted index. Values, dtypes
            and column labels are unchanged. If both positions are identical,
            ``value`` itself is returned.

        Raises:
            ValueError: If positions or frequency are invalid, if no frequency
                can be inferred, or if the frequency is coarser than the data
                (two distinct dates of a same entity would be merged into one
                period: aggregate with ``groupby`` / ``resample`` instead)

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> dates = pd.date_range('2024-01-01', periods=3, freq='MS')
            >>> series = pd.Series([1, 2, 3], index=dates)
            >>> [str(d.date()) for d in converter.convert(series, 'start', 'end').index]
            ['2024-01-31', '2024-02-29', '2024-03-31']
        """
        # Import différé du détecteur : évite un import circulaire avec
        # tsforecast.utils.frequency (dont converter.py importe position.utils)
        from ..frequency import detect_index_frequency

        # Normalisation des positions
        from_code = normalize_position(from_unit)
        to_code = normalize_position(to_unit)

        # Si les positions sont identiques, retourner les données telles quelles
        if from_code == to_code:
            return value

        # Conversion selon le type de valeur
        if isinstance(value, pd.DatetimeIndex):
            # Inférence de la fréquence si non fournie (pour DatetimeIndex simple)
            if freq is None:
                freq = detect_index_frequency(index=value, return_format='full')
                if freq is None:
                    raise ValueError("Cannot infer frequency from data. Please provide 'freq' parameter.")
            return self._convert_datetime_index(value, from_code, to_code, freq)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            # Routage selon le type d'index
            if isinstance(value.index, pd.MultiIndex) and value.index.nlevels >= 2 :
                # Pour les panels, l'inférence se fait par groupe dans _convert_panel()
                return self._convert_panel(value, from_code, to_code, freq)
            elif isinstance(value.index, pd.DatetimeIndex):
                # Inférence de la fréquence si non fournie (pour séries temporelles simples)
                if freq is None:
                    freq = detect_index_frequency(index=value.index, return_format='full')
                    if freq is None:
                        raise ValueError("Cannot infer frequency from data. Please provide 'freq' parameter.")
                return self._convert_time_series(value, from_code, to_code, freq)
            else:
                raise ValueError(f"Data must have DatetimeIndex or MultiIndex with datetime at last level")
        else:
            raise ValueError(f"Unsupported value type for conversion: {type(value)}")

    # Méthode de récupération du facteur de conversion
    def get_conversion_factor(
        self,
        from_unit: Union[PositionType, UserPositionType],
        to_unit: Union[PositionType, UserPositionType]
    ) -> float:
        """Get the conversion factor between two positions.

        Implementation of TemporalConverter.get_conversion_factor() for positions.

        Note: For period positions, there is no multiplicative factor. This method
        returns 1.0 if positions are the same, -1.0 if they are different (indicating
        a shift is needed rather than a multiplication).

        Args:
            from_unit: Source position
            to_unit: Target position

        Returns:
            1.0 if same position, -1.0 if different (shift required)

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> converter.get_conversion_factor('start', 'start')
            1.0
            >>> converter.get_conversion_factor('start', 'end')
            -1.0
        """
        # Normalisation des positions
        from_code = normalize_position(from_unit)
        to_code = normalize_position(to_unit)

        # Retour du facteur (1 si identique, -1 si différent pour indiquer un shift)
        return 1.0 if from_code == to_code else -1.0

    # Méthode auxiliaire de résolution des périodes décrites par une fréquence
    def _resolve_period_offsets(
        self,
        freq: str
    ) -> Tuple[str, int, Optional[DateOffset], Optional[DateOffset]]:
        """Resolve the periods described by a frequency string.

        Args:
            freq: Frequency string, possibly multiplied and anchored
                ('2MS', 'QS-FEB', 'YE-JUN', 'SMS', 'W-SUN', 'monthly', ...).

        Returns:
            Tuple ``(kind, n, start_offset, end_offset)``:

            - ``kind='period'`` for monthly / quarterly / yearly frequencies,
              with the single-period start and end offsets describing the
              same periods (``QS-FEB`` -> ``QS-FEB`` / ``QE-JAN``) and the
              multiplier ``n``;
            - ``kind='semi_month'`` for semi-monthly frequencies (offsets
              ``None``);
            - ``kind='none'`` for frequencies without a start/end variant
              ('D', 'B', 'W', 'h', ...), offsets ``None``.

        Raises:
            ValueError: If the frequency is not supported, or if it is a
                multiplied or non-default semi-monthly frequency.

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> kind, n, start, end = converter._resolve_period_offsets('QS-FEB')
            >>> kind, n, start.freqstr, end.freqstr
            ('period', 1, 'QS-FEB', 'QE-JAN')
        """
        # Import différé (voir note en tête de fichier)
        from ..frequency import normalize_frequency

        # Décomposition (multiplicateur, base, position, ancre) ; les noms littéraux
        # qui ne se parsent pas ('business_day') n'ont ni position ni multiplicateur
        try:
            parsed = parse_frequency(freq)
        except ValueError:
            parsed = ParsedFrequency(normalize_frequency(frequency=freq), None, None)
        n, position, suffix = parsed.multiplier, parsed.position, parsed.suffix
        # Normalisation de la fréquence de base
        base_freq = normalize_frequency(frequency=parsed.freq)

        # Cas où la fréquence tolère une position
        if base_freq in _PERIOD_FREQUENCIES:
            if base_freq == 'M':
                return 'period', n, to_offset('MS'), to_offset('ME')
            # Sans position explicite, une ancre pandas désigne le mois de fin ('Q-DEC')
            anchor_pos = position or 'E'
            anchor = suffix or ('JAN' if anchor_pos == 'S' else 'DEC')
            start_month = _shift_anchor(anchor, anchor_pos, 'S')
            end_month = _shift_anchor(start_month, 'S', 'E')
            return (
                'period',
                n,
                to_offset(f"{base_freq}S-{start_month}"),
                to_offset(f"{base_freq}E-{end_month}"),
            )

        # Cas de la fréquence bi-hebdomadaire
        if base_freq == 'SM':
            # Appariement des grilles natives pandas : seul le jour par défaut (15) est supporté
            if n != 1 or suffix not in (None, str(_SEMI_MONTH_DAY)):
                raise ValueError(
                    f"Unsupported semi-monthly frequency for position conversion: '{freq}'. "
                    f"Only 'SMS' / 'SME' (day {_SEMI_MONTH_DAY}, no multiplier) are supported."
                )
            return 'semi_month', n, None, None

        # Fréquence sans notion de position (journalière, hebdomadaire, infra-journalière...)
        return 'none', n, None, None

    # Méthode auxiliaire de conversion d'un DatetimeIndex
    def _convert_datetime_index(
        self,
        index: pd.DatetimeIndex,
        from_pos: PositionType,
        to_pos: PositionType,
        freq: str
    ) -> pd.DatetimeIndex:
        """Convert DatetimeIndex from one position to another.

        Each date is moved to the other bound of its own period, on the native
        pandas grid of the target offset (midnight of the first / last day).

        Args:
            index: DatetimeIndex to convert
            from_pos: Source position code
            to_pos: Target position code
            freq: Frequency string (multiplier and anchor honoured)

        Returns:
            Converted DatetimeIndex (same name and time zone); ``index``
            itself for identical positions or frequencies without position

        Raises:
            ValueError: If two distinct dates would land on the same converted
                date (frequency coarser than the data)

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> dates = pd.date_range('2024-02-01', periods=2, freq='QS-FEB')
            >>> [str(d.date()) for d in converter._convert_datetime_index(dates, 'S', 'E', 'QS-FEB')]
            ['2024-04-30', '2024-07-31']
        """
        # Cas identique, retourner tel quel
        if from_pos == to_pos:
            return index
        # Résolution des offsets de périodes
        kind, n, start_offset, end_offset = self._resolve_period_offsets(freq)

        # Fréquence sans position : aucune borne de période à rejoindre
        if kind == 'none':
            return index

        # Travail au jour calendaire, en heure locale naïve (un jour vaut 23 ou 25 h
        # au changement d'heure d'un index tz-aware) : les bornes pandas sont à minuit
        dates = index.tz_localize(None).normalize() if index.tz is not None else index.normalize()
        one_day = pd.Timedelta(days=1)

        if kind == 'period':
            if from_pos == 'S' and to_pos == 'E':
                # Début de la période contenant chaque date ((d + S) - S : d lui-même s'il est
                # sur l'ancre, l'ancre précédente sinon), puis fin = début + n périodes - 1 jour
                period_start = (dates + start_offset) - start_offset
                converted = period_start + start_offset * n - one_day
            else:
                # Fin de la période contenant chaque date, puis début = fin - n périodes + 1 jour
                period_end = (dates - end_offset) + end_offset
                converted = period_end - end_offset * n + one_day
        else:
            # Semi-mensuel : rang de la demi-période dans le mois (1er <-> 15, 15 <-> fin de mois)
            days = dates.day
            if from_pos == 'S' and to_pos == 'E':
                first_half = days < _SEMI_MONTH_DAY
                converted = dates.where(
                    ~first_half, dates + pd.to_timedelta(_SEMI_MONTH_DAY - days, unit='D')
                )
                converted = converted.where(first_half, dates + pd.offsets.MonthEnd(0))
            else:
                first_half = days <= _SEMI_MONTH_DAY
                converted = dates.where(~first_half, dates - pd.to_timedelta(days - 1, unit='D'))
                converted = converted.where(
                    first_half, dates - pd.to_timedelta(days - _SEMI_MONTH_DAY, unit='D')
                )

        # Conversion en datetime index
        converted = pd.DatetimeIndex(converted, name=index.name)
        if index.tz is not None:
            converted = converted.tz_localize(index.tz)

        # Garde-fou : une conversion de position ne fusionne jamais deux dates distinctes
        # (sinon des observations quitteraient leur période d'origine)
        n_source, n_converted = index.nunique(), converted.nunique()
        if n_converted < n_source:
            raise ValueError(
                f"Frequency '{freq}' is coarser than the data: {n_source} distinct dates would be "
                f"merged into {n_converted} converted dates. Provide the actual frequency of the "
                "data, or aggregate explicitly (groupby / resample) to change frequency."
            )

        return converted

    # Méthode auxiliaire de conversion d'une Series ou DataFrame
    def _convert_time_series(
        self,
        data: Union[pd.Series, pd.DataFrame],
        from_pos: PositionType,
        to_pos: PositionType,
        freq: str
    ) -> Union[pd.Series, pd.DataFrame]:
        """Convert Series or DataFrame index from one position to another.

        Args:
            data: Time series data to convert
            from_pos: Source position code
            to_pos: Target position code
            freq: Frequency string

        Returns:
            Copy of ``data`` with converted index (values, dtypes, column
            labels and ``attrs`` unchanged)

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> dates = pd.date_range('2023-01-01', periods=3, freq='MS')
            >>> series = pd.Series([1, 2, 3], index=dates)
            >>> end_series = converter._convert_time_series(series, 'S', 'E', 'M')
        """
        # Vérification que l'index est un DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex for position conversion")

        # Copie avec le nouvel index : seules les dates changent (dtypes préservés)
        result = data.copy()
        result.index = self._convert_datetime_index(data.index, from_pos, to_pos, freq)
        return result

    # Méthode auxiliaire de conversion d'un panel (Series ou DataFrame avec MultiIndex)
    def _convert_panel(
        self,
        data: Union[pd.Series, pd.DataFrame],
        from_pos: PositionType,
        to_pos: PositionType,
        freq: str = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """Convert panel data (MultiIndex) from one position to another.

        For panel data with mixed frequencies, each entity can have its own
        frequency. The conversion is applied separately to each entity group.

        Args:
            data: Panel data with MultiIndex (time at last level)
            from_pos: Source position code
            to_pos: Target position code
            freq: Optional frequency string. If None, frequency is inferred
                  separately for each entity group (from its dates, then from
                  its columns as a fallback).

        Returns:
            Copy of ``data`` with converted time index at last level (row
            order, values and dtypes unchanged)

        Raises:
            ValueError: If validation fails, structure is invalid, or the
                frequency of an entity cannot be inferred

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> entities = ['A', 'A', 'A', 'B', 'B', 'B']
            >>> dates = list(pd.date_range('2023-01-01', periods=3, freq='MS')) * 2
            >>> idx = pd.MultiIndex.from_arrays([entities, dates])
            >>> series = pd.Series(range(6), index=idx)
            >>> end_series = converter._convert_panel(series, 'S', 'E')
        """
        # Import différé du détecteur : évite un import circulaire avec
        # tsforecast.utils.frequency (dont converter.py importe position.utils)
        from ..frequency import detect_dataset_frequency, detect_index_frequency

        # Vérification que l'index est bien un MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Panel data must have a MultiIndex")

        # Vérification que le dernier niveau est datetime
        last_level = data.index.get_level_values(-1)
        if not isinstance(last_level, pd.DatetimeIndex):
            raise ValueError(
                "Last level of MultiIndex must be DatetimeIndex for position conversion. "
                f"Got {type(last_level).__name__} instead."
            )

        # Validation de la structure du panel : entités groupées
        if not validate_entities_grouped(data):
            raise ValueError(
                "Panel entities must be grouped (contiguous blocks). "
                "Each entity's observations must be adjacent in the data. "
                "Please sort your data by entity then by time."
            )

        # Validation du tri des dates au sein de chaque groupe
        if not validate_sorted_within_groups(data):
            raise ValueError(
                "Dates must be sorted within each entity group. "
                "Please ensure time values are monotonically increasing within each entity."
            )

        # Extraction des niveaux d'entités (tous sauf le dernier)
        n_levels = data.index.nlevels
        entity_levels_indices = list(range(n_levels - 1))
        groupby_levels = 0 if n_levels == 2 else entity_levels_indices

        # Conversion de chaque groupe séparément (pour gérer les fréquences mixtes) ;
        # les dates converties sont replacées à la position de leurs lignes
        grouped = data.groupby(level=groupby_levels, sort=False, dropna=False)
        converted_parts = []
        row_positions = []
        for group_keys, group_data in grouped:
            # Extraction des dates du groupe
            group_dates = group_data.index.get_level_values(-1)

            # Inférence de la fréquence pour ce groupe si non fournie
            group_freq = freq
            if group_freq is None:
                try:
                    group_freq = detect_index_frequency(index=group_dates, return_format='full')
                except ValueError:
                    # Trop peu d'observations pour une détection sur l'index
                    group_freq = None
                if group_freq is None:
                    # Repli : fréquence la plus fine détectée sur les colonnes du groupe,
                    # réduit à son index temporel (une seule entité, NaN écartés par colonne)
                    group_frame = group_data.to_frame() if isinstance(group_data, pd.Series) else group_data
                    group_frame = group_frame.set_axis(group_dates)
                    try:
                        group_freq = detect_dataset_frequency(
                            df=group_frame,
                            return_format='full',
                            check_consistency=True,
                            consistency_mode='highest',
                        )
                    except ValueError:
                        group_freq = None
                if group_freq is None:
                    raise ValueError(
                        f"Cannot infer frequency for entity {group_keys}. "
                        "Please provide 'freq' parameter or ensure data has regular frequency."
                    )

            # Conversion des dates du groupe
            converted_parts.append(self._convert_datetime_index(group_dates, from_pos, to_pos, group_freq))
            row_positions.extend(grouped.indices[group_keys])

        # Reconstitution du niveau temporel dans l'ordre des lignes
        if converted_parts:
            converted_dates = converted_parts[0].append(converted_parts[1:])
            order = pd.Index(row_positions).argsort()
            new_last_level = converted_dates[order]
        else:
            # Panel vide : aucune date à convertir
            new_last_level = last_level

        new_index = pd.MultiIndex.from_arrays(
            [data.index.get_level_values(i) for i in entity_levels_indices] + [new_last_level],
            names=data.index.names
        )

        # Copie avec le nouvel index : seules les dates changent (dtypes préservés)
        result = data.copy()
        result.index = new_index
        return result

    # Méthode de conversion d'un offset pandas complet
    def convert_offset(
        self,
        offset_str: str,
        to_position: Union[PositionType, UserPositionType]
    ) -> str:
        """Convert pandas DateOffset to a different position.

        The returned offset describes the same periods as the source one: a
        leading multiplier is kept, a month anchor is shifted accordingly
        (``QS-FEB`` <-> ``QE-JAN``), and frequencies without a start/end
        variant in pandas ('D', 'W-MON', 'B', 'h', ...) are returned
        unchanged. The result is always a valid pandas offset alias.

        Args:
            offset_str: Source pandas DateOffset (e.g., 'MS', 'QE', '2MS', 'QS-FEB')
            to_position: Target position

        Returns:
            Converted pandas DateOffset string

        Raises:
            ValueError: If the offset or the target position is not supported

        Examples:
            >>> converter = PeriodPositionConverter()
            >>> converter.convert_offset('MS', 'end')
            'ME'
            >>> converter.convert_offset('QE', 'start')
            'QS'
            >>> converter.convert_offset('2MS', 'end')
            '2ME'
            >>> converter.convert_offset('M', 'start')
            'MS'
            >>> converter.convert_offset('QE-NOV', 'start')
            'QS-DEC'
            >>> converter.convert_offset('W-MON', 'end')
            'W-MON'
        """
        # Décomposition de l'offset, multiplicateur éventuel compris (ex: '2MS'),
        # conservé tel quel dans le résultat
        freq, position, suffix, multiplier = parse_frequency(offset_str)

        # Normalisation de la position cible
        to_pos = normalize_position(to_position)

        # Décalage de l'ancre mensuelle (trimestriel, annuel ; mois inconnu rejeté) :
        # sans position explicite, une ancre pandas désigne le mois de fin ('Q-DEC')
        if suffix is not None and freq in ('Q', 'Y'):
            suffix = _shift_anchor(suffix, position or 'E', to_pos)

        # Recombinaison avec la nouvelle position : appliquée même si l'offset
        # d'origine n'en portait pas explicitement une (ex: 'M' -> 'ME'), ignorée
        # pour les fréquences sans variante début / fin (ex: 'W-MON' inchangé)
        return build_frequency_string(freq, to_pos, suffix, multiplier)
