"""Time manipulation utilities for time series processing."""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from typing import NamedTuple, Optional, Union
from datetime import datetime

# Import des utilitaires de fréquence
from ..frequency import normalize_frequency, FrequencyType, UserFrequencyType
from ..parse import parse_frequency

# Fonction de conversion d'une chaîne de caractères en date
def resolve_date(date: Union[str, datetime], format: str = None) -> datetime:
    """Resolve date from provided parameter.

    Args:
        date: Date ('today' or datetime)
        format: String format for dates

    Returns:
        Resolved date

    Raises:
        ValueError: If date is not 'today', a string, or a datetime object, or if
            it cannot be resolved to a valid date (unparseable, empty or ``NaT``)

    Examples:
        >>> from datetime import datetime
        >>> resolve_date('today')  # doctest: +SKIP
        datetime.datetime(2025, 11, 28, 10, 30, 15, 123456)

        >>> resolve_date('2023-06-15')
        datetime.datetime(2023, 6, 15, 0, 0)

        >>> resolve_date('06/15/2023', format='%m/%d/%Y')
        datetime.datetime(2023, 6, 15, 0, 0)

        >>> existing_date = datetime(2023, 6, 15, 14, 30)
        >>> resolve_date(existing_date)
        datetime.datetime(2023, 6, 15, 14, 30)
    """
    # Distinction suivant le type de l'argument
    # Cas où la date est une chaîne de caractères
    if isinstance(date, str):
        # Cas où l'on souhaite obtenir la date d'aujourd'hui
        if date.lower() == 'today':
            return datetime.now()
        else:
            # Utilise le format si spécifié
            if format is not None :
                resolved = pd.to_datetime(date, format=format)
            else:
                resolved = pd.to_datetime(date)
            # Une chaîne vide est convertie en NaT : refus explicite plutôt que propagation
            if pd.isna(resolved):
                raise ValueError(f"Could not resolve a date from {date!r}")
            return resolved.to_pydatetime()
    # Cas où la date est un datetime (NaT en est un : refusé)
    elif isinstance(date, datetime):
        if pd.isna(date):
            raise ValueError("'date' must not be NaT")
        return date
    else:
        raise ValueError("'date' must be 'today', a string or datetime")

# Fonctions de conversion entre timeseries et string
def timeseries_to_string(ts: pd.Series, format: str = "%Y-%m-%d") -> pd.Series:
    """Convert a time series index to string format.

    Args:
        ts: Time series with datetime index
        format: String format for dates (default: "%Y-%m-%d" for year-month-day)

    Returns:
        Series with string-formatted dates as index

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2023-01-01', periods=3, freq='D')
        >>> ts = pd.Series([1, 2, 3], index=dates)
        >>> timeseries_to_string(ts)
        2023-01-01    1
        2023-01-02    2
        2023-01-03    3
        dtype: int64

        >>> timeseries_to_string(ts, format="%m/%d/%Y")
        01/01/2023    1
        01/02/2023    2
        01/03/2023    3
        dtype: int64
    """
    # Conversion de l'index datetime en string selon le format spécifié
    string_index = ts.index.strftime(format)
    return pd.Series(ts.values, index=string_index, name=ts.name)

# Fonction de conversion d'une série avec en index des dates sous forme de chaine de caractères à une série temporelle
def string_to_timeseries(ts: pd.Series, format: str = None) -> pd.Series:
    """Convert a time series with string index to datetime index.

    Args:
        ts: Time series with string index representing dates
        format: String format to parse dates (if None, pandas will infer)

    Returns:
        Series with datetime index

    Examples:
        >>> import pandas as pd
        >>> string_ts = pd.Series([1, 2, 3], index=['2023-01-01', '2023-01-02', '2023-01-03'])
        >>> string_to_timeseries(string_ts)
        2023-01-01    1
        2023-01-02    2
        2023-01-03    3
        dtype: int64

        >>> string_ts_custom = pd.Series([1, 2, 3], index=['01/01/2023', '01/02/2023', '01/03/2023'])
        >>> string_to_timeseries(string_ts_custom, format="%m/%d/%Y")
        2023-01-01    1
        2023-01-02    2
        2023-01-03    3
        dtype: int64
    """
    # Conversion de l'index string en datetime, avec inférence automatique si format non spécifié
    if format is not None:
        datetime_index = pd.to_datetime(ts.index, format=format)
    else:
        datetime_index = pd.to_datetime(ts.index)

    return pd.Series(ts.values, index=datetime_index, name=ts.name)

# Jours de la semaine et mois, dans l'ordre des suffixes d'ancre pandas
_WEEKDAYS = ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')
_MONTHS = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')

# Origine par défaut des grilles à multiplicateur : l'époque Unix (jeudi 1er janvier 1970)
_EPOCH = pd.Timestamp('1970-01-01')
_EPOCH_WEEKDAY = _EPOCH.weekday()
_EPOCH_MONTH_INDEX = 1970 * 12

# Durée en nanosecondes des fréquences infra-journalières
_SUBDAILY_NS = {
    'ns': 1,
    'us': 1_000,
    'ms': 1_000_000,
    's': 1_000_000_000,
    'min': 60 * 1_000_000_000,
    'h': 3_600 * 1_000_000_000,
}

# Nombre de mois par unité des fréquences calendaires à base mensuelle
_MONTHS_PER_UNIT = {'M': 1, 'Q': 3, 'Y': 12}


class _PeriodSpec(NamedTuple):
    """Components of a frequency needed to delimit periods.

    Attributes:
        base: Normalized base frequency ('M', 'W', 'h', ...).
        multiplier: Number of base units per period (1 when absent).
        position: 'S' / 'E' / None; only meaningful with an anchor on 'Q' / 'Y'.
        suffix: Anchor after the dash ('DEC', 'WED', ...) or None.
    """

    base: str
    multiplier: int
    position: Optional[str]
    suffix: Optional[str]


# Résolution d'une fréquence utilisateur en composants de période
def _resolve_spec(frequency: Union[FrequencyType, UserFrequencyType]) -> _PeriodSpec:
    """Split a frequency into base, multiplier, position and anchor.

    Args:
        frequency: Pandas code (possibly multiplied / anchored) or user-friendly name.

    Returns:
        The frequency components.

    Raises:
        ValueError: If the frequency is unsupported, or if its anchor is invalid
            for its base frequency.
    """
    # Validation et base normalisée (lève 'Unsupported frequency' le cas échéant)
    base = normalize_frequency(frequency)

    # Extraction du multiplicateur, de la position et de l'ancre ; les noms littéraux
    # ('minute', 'semi_monthly', ...) ne suivent pas la grammaire pandas : valeurs par défaut
    multiplier, position, suffix = 1, None, None
    try:
        parsed = parse_frequency(frequency)
        if normalize_frequency(parsed.freq) == base:
            multiplier, position, suffix = parsed.multiplier, parsed.position, parsed.suffix
    except ValueError:
        pass

    # Vérification de l'ancre : jour de semaine pour W, mois pour Q / Y, rien ailleurs
    if suffix is not None:
        allowed = {'W': _WEEKDAYS, 'Q': _MONTHS, 'Y': _MONTHS}.get(base)
        if allowed is None:
            raise ValueError(
                f"Unsupported frequency: {frequency}. "
                f"Base frequency '{base}' does not accept an anchor ('-{suffix}')."
            )
        if suffix not in allowed:
            raise ValueError(
                f"Unsupported frequency: {frequency}. "
                f"Invalid anchor '{suffix}' for base frequency '{base}'."
            )
    return _PeriodSpec(base, multiplier, position, suffix)


# Conversion de l'entrée en Timestamp
def _to_timestamp(date: Union[pd.Timestamp, datetime, pd.Period]) -> pd.Timestamp:
    """Convert a date-like input to a ``pd.Timestamp``.

    A ``Period`` is represented by its start.

    Raises:
        TypeError: If ``date`` is not a datetime, Timestamp, datetime64 or Period.
        ValueError: If ``date`` is ``NaT``.
    """
    if isinstance(date, pd.Period):
        timestamp = date.to_timestamp(how='start')
    elif isinstance(date, (datetime, np.datetime64)):
        timestamp = pd.Timestamp(date)
    else:
        raise TypeError(
            f"'date' must be a datetime, Timestamp or Period, got {type(date).__name__}"
        )
    if pd.isna(timestamp):
        raise ValueError("'date' must not be NaT")
    return timestamp


# Alignement de l'origine sur le fuseau de la date
def _align_origin(
    origin: Union[pd.Timestamp, datetime, None], tz
) -> Optional[pd.Timestamp]:
    """Express ``origin`` in the time zone of the reference date.

    A naive origin is read as wall-clock time in ``tz``; an aware origin is
    converted to ``tz`` (or made naive when ``tz`` is None).
    """
    if origin is None:
        return None
    aligned = pd.Timestamp(origin)
    if pd.isna(aligned):
        raise ValueError("'origin' must not be NaT")
    if tz is None:
        return aligned.tz_localize(None) if aligned.tzinfo is not None else aligned
    if aligned.tzinfo is not None:
        return aligned.tz_convert(tz)
    return aligned.tz_localize(tz, ambiguous=True, nonexistent='shift_forward')


# Retour au fuseau d'origine d'une date « murale » (sans fuseau)
def _localize(wall: pd.Timestamp, tz) -> pd.Timestamp:
    """Attach ``tz`` to a wall-clock timestamp (no-op when ``tz`` is None)."""
    if tz is None:
        return wall
    return wall.tz_localize(tz, ambiguous=True, nonexistent='shift_forward')


# Bornes d'une période infra-journalière
def _subdaily_bounds(
    date: pd.Timestamp, spec: _PeriodSpec, origin: Optional[pd.Timestamp]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Boundaries for ns / us / ms / s / min / h periods."""
    length = spec.multiplier * _SUBDAILY_NS[spec.base]
    wall = date.tz_localize(None) if date.tzinfo is not None else date
    origin_wall = 0
    if origin is not None:
        origin_wall = (origin.tz_localize(None) if origin.tzinfo is not None else origin).value

    # Découpage sur l'horloge murale (une heure commence à HH:00 locale, y compris
    # pour les fuseaux à décalage de 30 min), puis retrait de l'écart en temps absolu :
    # l'instant conserve son décalage UTC, y compris dans l'heure ambiguë d'un changement d'heure
    elapsed = (wall.value - origin_wall) % length
    start = date - pd.Timedelta(elapsed, unit='ns')
    return start, start + pd.Timedelta(length, unit='ns')


# Bornes d'une période en jours (D, B, W)
def _day_bounds(
    date: pd.Timestamp, spec: _PeriodSpec, origin: Optional[pd.Timestamp]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Wall-clock boundaries (naive) for D / B / W periods."""
    days_per_unit = 7 if spec.base == 'W' else 1
    length = spec.multiplier * days_per_unit

    if origin is not None:
        origin_day = (origin.tz_localize(None) if origin.tzinfo is not None else origin).normalize()
    elif spec.base == 'W':
        # La semaine `W-X` se termine le jour X : elle commence le lendemain
        start_weekday = (_WEEKDAYS.index(spec.suffix or 'SUN') + 1) % 7
        origin_day = _EPOCH - pd.Timedelta(days=(_EPOCH_WEEKDAY - start_weekday) % 7)
    else:
        origin_day = _EPOCH

    elapsed_days = (date.normalize() - origin_day).days
    start = origin_day + pd.Timedelta(days=(elapsed_days // length) * length)
    return start, start + pd.Timedelta(days=length)


# Bornes d'une période à base mensuelle (SM, M, Q, Y)
def _month_bounds(
    date: pd.Timestamp, spec: _PeriodSpec, origin: Optional[pd.Timestamp]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Wall-clock boundaries (naive) for SM / M / Q / Y periods."""
    month_index = date.year * 12 + date.month - 1

    # Semi-mensuel : indice de quinzaine (1-15 puis 16-fin), grille en quinzaines
    if spec.base == 'SM':
        index = month_index * 2 + (1 if date.day > 15 else 0)
        origin_index = 2 * _EPOCH_MONTH_INDEX
        if origin is not None:
            origin_index = (origin.year * 12 + origin.month - 1) * 2 + (1 if origin.day > 15 else 0)
        start_index = origin_index + ((index - origin_index) // spec.multiplier) * spec.multiplier

        def to_date(fortnight_index: int) -> pd.Timestamp:
            month, half = divmod(fortnight_index, 2)
            return pd.Timestamp(year=month // 12, month=month % 12 + 1, day=16 if half else 1)

        return to_date(start_index), to_date(start_index + spec.multiplier)

    # M / Q / Y : grille en mois ; l'ancre fixe le premier mois de la période
    length = spec.multiplier * _MONTHS_PER_UNIT[spec.base]
    if origin is not None:
        origin_index = origin.year * 12 + origin.month - 1
    elif spec.suffix is None or spec.base == 'M':
        origin_index = _EPOCH_MONTH_INDEX
    else:
        anchor_month = _MONTHS.index(spec.suffix) + 1
        # 'S' : l'ancre est le premier mois ; sinon (E ou absente) le dernier
        first_month = anchor_month if spec.position == 'S' else anchor_month % 12 + 1
        if spec.base == 'Q':
            first_month = (first_month - 1) % 3 + 1
        origin_index = _EPOCH_MONTH_INDEX + first_month - 1
    start_index = origin_index + ((month_index - origin_index) // length) * length

    def to_first_day(index: int) -> pd.Timestamp:
        return pd.Timestamp(year=index // 12, month=index % 12 + 1, day=1)

    return to_first_day(start_index), to_first_day(start_index + length)


# Calcul commun des bornes
def _period_bounds(
    date: Union[pd.Timestamp, datetime, pd.Period],
    frequency: Union[FrequencyType, UserFrequencyType],
    origin: Union[pd.Timestamp, datetime, None],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute ``(start, end)`` of the period containing ``date``."""
    # Résolution et décomposition de la fréquence
    spec = _resolve_spec(frequency)
    # Conversion en timestamp
    timestamp = _to_timestamp(date)
    # Extraction de la timezone
    tz = timestamp.tzinfo
    # Alignement de l'origine de la période sur la timezone
    aligned_origin = _align_origin(origin, tz)

    # Cas des fréquences infrajournalières
    if spec.base in _SUBDAILY_NS:
        return _subdaily_bounds(timestamp, spec, aligned_origin)

    # Fréquences calendaires : découpage sur l'horloge murale, puis retour au fuseau
    wall = timestamp.tz_localize(None) if tz is not None else timestamp
    if spec.base in ('D', 'B', 'W'):
        start, end = _day_bounds(wall, spec, aligned_origin)
    else:
        start, end = _month_bounds(wall, spec, aligned_origin)
    return _localize(start, tz), _localize(end, tz)


# Fonction identifiant la date de début d'une période à partir d'une date et d'une fréquence
def get_period_start(
    date: Union[pd.Timestamp, datetime, pd.Period],
    frequency: Union[FrequencyType, UserFrequencyType],
    origin: Union[pd.Timestamp, datetime, None] = None,
) -> pd.Timestamp:
    """Get the start date of the period containing the given date.

    Periods are half-open intervals ``[start, end)``. Anchors and multipliers of
    pandas frequency strings are honoured:

    - ``W-X``: weeks end on day ``X`` (``W`` = ``W-SUN``: Monday to Sunday);
    - ``Q-X`` / ``QE-X``: quarters end in month ``X``; ``QS-X``: they start in
      month ``X`` (default: calendar quarters). Same for ``Y`` / ``YE`` / ``YS``;
    - position (``MS`` / ``ME``) does not change periods;
    - ``nX`` (``2MS``, ``3D``, ``2h``...): periods of ``n`` base units, aligned on
      ``origin``, or on the Unix epoch (1970-01-01, and its anchor when anchored)
      when ``origin`` is None.

    Calendar frequencies (day and above) are computed on the local wall clock,
    so a day lasts 23 or 25 hours across a DST change. Sub-daily frequencies
    follow the local wall clock (an hour starts at ``HH:00`` local time) and keep
    the UTC offset of ``date``. The time zone of ``date`` is preserved.

    Args:
        date: Reference date. A ``Period`` is represented by its start (the period
            containing a ``Period`` is the one containing its first instant).
        frequency: Period frequency (pandas codes or user-friendly names).
        origin: Optional grid origin for multiplied frequencies. Day and week
            frequencies use its date, month-based ones its month (the anchor of
            the frequency is then ignored), sub-daily ones its instant.

    Returns:
        Start of the period, in the time zone of ``date`` (naive if ``date`` is).

    Raises:
        ValueError: If the frequency or its anchor is unsupported, or if ``date``
            or ``origin`` is ``NaT``.
        TypeError: If ``date`` is not a datetime, Timestamp, datetime64 or Period.

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> get_period_start(pd.Timestamp('2023-06-15'), 'monthly')
        Timestamp('2023-06-01 00:00:00')
        >>> get_period_start(datetime(2023, 6, 15), 'Q-JAN')
        Timestamp('2023-05-01 00:00:00')
        >>> get_period_start(datetime(2023, 6, 15, 14, 35, 22), 'hourly')
        Timestamp('2023-06-15 14:00:00')
        >>> get_period_start(datetime(2023, 6, 15), '2MS')
        Timestamp('2023-05-01 00:00:00')
    """
    return _period_bounds(date, frequency, origin)[0]


# Fonction identifiant la date de fin (exclue) d'une période à partir d'une date et d'une fréquence
def get_period_end(
    date: Union[pd.Timestamp, datetime, pd.Period],
    frequency: Union[FrequencyType, UserFrequencyType],
    origin: Union[pd.Timestamp, datetime, None] = None,
) -> pd.Timestamp:
    """Get the end date of the period containing the given date.

    See :func:`get_period_start` for the handling of anchors, multipliers,
    ``origin``, time zones and ``Period`` inputs.

    Args:
        date: Reference date.
        frequency: Period frequency (pandas codes or user-friendly names).
        origin: Optional grid origin for multiplied frequencies.

    Returns:
        First date outside the period (exclusive boundary), in the time zone of
        ``date``. It is also the start of the next period.

    Raises:
        ValueError: If the frequency or its anchor is unsupported, or if ``date``
            or ``origin`` is ``NaT``.
        TypeError: If ``date`` is not a datetime, Timestamp, datetime64 or Period.

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> get_period_end(pd.Timestamp('2023-06-15'), 'monthly')
        Timestamp('2023-07-01 00:00:00')
        >>> get_period_end(datetime(2023, 6, 15), 'QE-NOV')
        Timestamp('2023-09-01 00:00:00')
        >>> get_period_end(datetime(2023, 6, 15, 14, 35, 22), 'hourly')
        Timestamp('2023-06-15 15:00:00')
        >>> get_period_end(datetime(2023, 6, 15, 1, 2, 3, 456789), 'ns')
        Timestamp('2023-06-15 01:02:03.456789001')
    """
    return _period_bounds(date, frequency, origin)[1]


# Fonction retournant les bornes d'une période
def get_period_boundaries(
    date: Union[pd.Timestamp, datetime, pd.Period],
    frequency: Union[FrequencyType, UserFrequencyType],
    origin: Union[pd.Timestamp, datetime, None] = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get the start and end boundaries of the period containing the given date.

    See :func:`get_period_start` for the handling of anchors, multipliers,
    ``origin``, time zones and ``Period`` inputs.

    Args:
        date: Reference date
        frequency: Period frequency (pandas codes or user-friendly names)
        origin: Optional grid origin for multiplied frequencies.

    Returns:
        Tuple containing (start_date, end_date) where start_date is included
        in the period [start_date, end_date) and end_date is excluded from it.
        Both are ``pd.Timestamp`` (a ``datetime`` subclass) in the time zone of ``date``.

    Raises:
        ValueError: If the frequency or its anchor is unsupported, or if ``date``
            or ``origin`` is ``NaT``.
        TypeError: If ``date`` is not a datetime, Timestamp, datetime64 or Period.

    Examples:
        >>> import pandas as pd
        >>> date = pd.Timestamp('2023-06-15')
        >>> get_period_boundaries(date, 'monthly')
        (Timestamp('2023-06-01 00:00:00'), Timestamp('2023-07-01 00:00:00'))

        >>> get_period_boundaries(date, 'weekly')
        (Timestamp('2023-06-12 00:00:00'), Timestamp('2023-06-19 00:00:00'))

        >>> get_period_boundaries(date, 'W-WED')  # weeks run Thursday to Wednesday
        (Timestamp('2023-06-15 00:00:00'), Timestamp('2023-06-22 00:00:00'))

        >>> get_period_boundaries(date, 'Q')  # Pandas quarterly frequency
        (Timestamp('2023-04-01 00:00:00'), Timestamp('2023-07-01 00:00:00'))
    """
    return _period_bounds(date, frequency, origin)
