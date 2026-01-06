"""Time manipulation utilities for time series processing."""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Literal, Union
from datetime import datetime, timedelta

# Import des utilitaires de fréquence
from ..frequency import normalize_frequency, FrequencyType, UserFrequencyType

# Fonction de conversion d'une chaîne de caractères en date
def resolve_date(date: Union[str, datetime], format: str = None) -> datetime:
    """Resolve date from provided parameter.

    Args:
        date: Date ('today' or datetime)
        format: String format for dates

    Returns:
        Resolved date

    Raises:
        ValueError: If date is not 'today', a string, or a datetime object

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
                return pd.to_datetime(date, format=format).to_pydatetime()
            else:
                return pd.to_datetime(date).to_pydatetime()
    # Cas où la date est un datetime
    elif isinstance(date, datetime):
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

# Fonction identifiant la date de début d'une période à partir d'une date et d'une fréquence
def get_period_start(date: Union[pd.Timestamp, datetime], frequency: Union[FrequencyType, UserFrequencyType]) -> datetime:
    """Get the start date of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (pandas codes or user-friendly names)

    Returns:
        Start date of the period as datetime object

    Raises:
        ValueError: If frequency is not supported

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> get_period_start(pd.Timestamp('2023-06-15'), 'monthly')
        datetime.datetime(2023, 6, 1, 0, 0)
        >>> get_period_start(datetime(2023, 6, 15), 'quarterly')
        datetime.datetime(2023, 4, 1, 0, 0)
        >>> get_period_start(datetime(2023, 6, 15, 14, 35, 22), 'hourly')
        datetime.datetime(2023, 6, 15, 14, 0, 0)
    """
    # Conversion en datetime si nécessaire
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    # Normalisation de la fréquence
    base_freq = normalize_frequency(frequency)

    # Fréquences infra-secondes
    if base_freq == 'ns':
        # Nanoseconde : retourne la date telle quelle (précision datetime limitée)
        return date
    elif base_freq == 'us':
        # Microseconde : début de la microseconde courante
        return datetime(
            date.year, date.month, date.day,
            date.hour, date.minute, date.second,
            date.microsecond
        )
    elif base_freq == 'ms':
        # Milliseconde : début de la milliseconde courante
        millisecond = (date.microsecond // 1000) * 1000
        return datetime(
            date.year, date.month, date.day,
            date.hour, date.minute, date.second,
            millisecond
        )

    # Fréquences infra-journalières
    elif base_freq == 's':
        # Début de la seconde
        return datetime(
            date.year, date.month, date.day,
            date.hour, date.minute, date.second
        )
    elif base_freq == 'min':
        # Début de la minute
        return datetime(
            date.year, date.month, date.day,
            date.hour, date.minute
        )
    elif base_freq == 'h':
        # Début de l'heure
        return datetime(
            date.year, date.month, date.day,
            date.hour
        )

    # Fréquences journalières et supérieures
    elif base_freq == 'D' or base_freq == 'B':
        # Début du jour (minuit)
        return datetime(date.year, date.month, date.day)
    elif base_freq == 'W':
        # Début de la semaine (toujours lundi)
        return datetime(date.year, date.month, date.day) - timedelta(days=date.weekday())
    elif base_freq == 'SM':
        # Semi-mensuel : 1-15 et 16-fin du mois
        if date.day <= 15:
            # Première quinzaine : début le 1er
            return datetime(date.year, date.month, 1)
        else:
            # Deuxième quinzaine : début le 16
            return datetime(date.year, date.month, 16)
    elif base_freq == 'M':
        # Premier jour du mois
        return datetime(date.year, date.month, 1)
    elif base_freq == 'Q':
        # Premier jour du trimestre (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
        quarter = (date.month - 1) // 3 + 1
        quarter_start_month = (quarter - 1) * 3 + 1
        return datetime(date.year, quarter_start_month, 1)
    elif base_freq == 'Y':
        # Premier jour de l'année
        return datetime(date.year, 1, 1)
    else:
        raise ValueError(
            f"Unsupported frequency: {frequency}. "
            f"Base frequency '{base_freq}' is not recognized."
        )

# Fonction identifiant la date de début d'une période à partir d'une date et d'une fréquence
def get_period_end(date: Union[pd.Timestamp, datetime], frequency: Union[FrequencyType, UserFrequencyType]) -> datetime:
    """Get the end date of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (pandas codes or user-friendly names)

    Returns:
        First date outside the period (exclusive boundary) as datetime object

    Raises:
        ValueError: If frequency is not supported

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> get_period_end(pd.Timestamp('2023-06-15'), 'monthly')
        datetime.datetime(2023, 7, 1, 0, 0)
        >>> get_period_end(datetime(2023, 6, 15), 'quarterly')
        datetime.datetime(2023, 7, 1, 0, 0)
        >>> get_period_end(datetime(2023, 6, 15, 14, 35, 22), 'hourly')
        datetime.datetime(2023, 6, 15, 15, 0, 0)
    """
    # Conversion en datetime si nécessaire
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    # Normalisation de la fréquence
    base_freq = normalize_frequency(frequency)

    # Fréquences infra-secondes
    if base_freq == 'ns':
        # Nanoseconde : ajoute une nanoseconde (limitation de datetime)
        # En pratique, datetime a une précision en microsecondes
        return date + timedelta(microseconds=0.001)
    elif base_freq == 'us':
        # Microseconde suivante
        return date + timedelta(microseconds=1)
    elif base_freq == 'ms':
        # Milliseconde suivante
        return date + timedelta(milliseconds=1)

    # Fréquences infra-journalières
    elif base_freq == 's':
        # Seconde suivante
        return date.replace(microsecond=0) + timedelta(seconds=1)
    elif base_freq == 'min':
        # Minute suivante
        return date.replace(second=0, microsecond=0) + timedelta(minutes=1)
    elif base_freq == 'h':
        # Heure suivante
        return date.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Fréquences journalières et supérieures
    elif base_freq == 'D' or base_freq == 'B':
        # Jour suivant (minuit)
        return datetime(date.year, date.month, date.day) + timedelta(days=1)
    elif base_freq == 'W':
        # Lundi de la semaine suivante
        week_start = datetime(date.year, date.month, date.day) - timedelta(days=date.weekday())
        return week_start + timedelta(days=7)
    elif base_freq == 'SM':
        # Semi-mensuel : fin de la période dépend de la quinzaine
        if date.day <= 15:
            # Première quinzaine : fin le 16
            return datetime(date.year, date.month, 16)
        else:
            # Deuxième quinzaine : fin le 1er du mois suivant
            if date.month == 12:
                return datetime(date.year + 1, 1, 1)
            else:
                return datetime(date.year, date.month + 1, 1)
    elif base_freq == 'M':
        # Premier jour du mois suivant
        if date.month == 12:
            return datetime(date.year + 1, 1, 1)
        else:
            return datetime(date.year, date.month + 1, 1)
    elif base_freq == 'Q':
        # Premier jour du trimestre suivant
        quarter = (date.month - 1) // 3 + 1
        if quarter == 4:
            # Q4 -> Q1 de l'année suivante
            return datetime(date.year + 1, 1, 1)
        else:
            # Trimestre suivant dans la même année
            next_quarter_month = quarter * 3 + 1
            return datetime(date.year, next_quarter_month, 1)
    elif base_freq == 'Y':
        # Premier jour de l'année suivante
        return datetime(date.year + 1, 1, 1)
    else:
        raise ValueError(
            f"Unsupported frequency: {frequency}. "
            f"Base frequency '{base_freq}' is not recognized."
        )


# Fonction retournant les bornes d'une période
def get_period_boundaries(date: Union[pd.Timestamp, datetime], frequency: Union[FrequencyType, UserFrequencyType]) -> tuple[datetime, datetime]:
    """Get the start and end boundaries of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (pandas codes or user-friendly names)

    Returns:
        Tuple containing (start_date, end_date) where start_date is included
        in the period [start_date, end_date) and end_date is excluded from it.
        Both dates are datetime objects.

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> date = pd.Timestamp('2023-06-15')
        >>> get_period_boundaries(date, 'monthly')
        (datetime.datetime(2023, 6, 1, 0, 0), datetime.datetime(2023, 7, 1, 0, 0))

        >>> get_period_boundaries(date, 'weekly')
        (datetime.datetime(2023, 6, 12, 0, 0), datetime.datetime(2023, 6, 19, 0, 0))

        >>> get_period_boundaries(date, 'Q')  # Pandas quarterly frequency
        (datetime.datetime(2023, 4, 1, 0, 0), datetime.datetime(2023, 7, 1, 0, 0))
    """
    # Calcul de début de la période
    period_start = get_period_start(date=date, frequency=frequency)
    # Calcul de la fin de la période
    period_end = get_period_end(date=date, frequency=frequency)

    return period_start, period_end
