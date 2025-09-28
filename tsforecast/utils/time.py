"""Time manipulation utilities for time series processing."""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Literal, Union
from datetime import datetime, timedelta

# Import des utilitaires de fréquence
from ..frequency.utils import get_base_frequency, FrequencyType

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
def get_period_start(date: Union[pd.Timestamp, datetime], frequency: FrequencyType) -> datetime:
    """Get the start date of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (pandas codes or user-friendly names)

    Returns:
        Start date of the period as datetime object

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> get_period_start(pd.Timestamp('2023-06-15'), 'monthly')
        datetime.datetime(2023, 6, 1, 0, 0)
        >>> get_period_start(datetime(2023, 6, 15), 'quarterly')
        datetime.datetime(2023, 4, 1, 0, 0)
    """
    # Conversion en datetime si nécessaire
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    # Normalisation de la fréquence
    base_freq = get_base_frequency(frequency)

    # Distinction suivant la fréquence de base avec logique canonique
    if base_freq == 'daily' or base_freq == 'business_daily':
        # Début du jour (minuit)
        return datetime(date.year, date.month, date.day)
    elif base_freq == 'weekly':
        # Début de la semaine (toujours lundi)
        return datetime(date.year, date.month, date.day) - timedelta(days=date.weekday())
    elif base_freq == 'monthly':
        # Premier jour du mois
        return datetime(date.year, date.month, 1)
    elif base_freq == 'quarterly':
        # Premier jour du trimestre (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
        quarter = (date.month - 1) // 3 + 1
        quarter_start_month = (quarter - 1) * 3 + 1
        return datetime(date.year, quarter_start_month, 1)
    elif base_freq == 'annual':
        # Premier jour de l'année
        return datetime(date.year, 1, 1)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Base frequency '{base_freq}' is not recognized.")

# Fonction identifiant la date de début d'une période à partir d'une date et d'une fréquence
def get_period_end(date: Union[pd.Timestamp, datetime], frequency: FrequencyType) -> datetime:
    """Get the end date of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (pandas codes or user-friendly names)

    Returns:
        First date outside the period (exclusive boundary) as datetime object

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> get_period_end(pd.Timestamp('2023-06-15'), 'monthly')
        datetime.datetime(2023, 7, 1, 0, 0)
        >>> get_period_end(datetime(2023, 6, 15), 'quarterly')
        datetime.datetime(2023, 7, 1, 0, 0)
    """
    # Conversion en datetime si nécessaire
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    # Normalisation de la fréquence
    base_freq = get_base_frequency(frequency)

    # Distinction suivant la fréquence de base avec logique canonique
    if base_freq == 'daily' or base_freq == 'business_daily':
        # Jour suivant (minuit)
        return datetime(date.year, date.month, date.day) + timedelta(days=1)
    elif base_freq == 'weekly':
        # Lundi de la semaine suivante
        week_start = datetime(date.year, date.month, date.day) - timedelta(days=date.weekday())
        return week_start + timedelta(days=7)
    elif base_freq == 'monthly':
        # Premier jour du mois suivant
        if date.month == 12:
            return datetime(date.year + 1, 1, 1)
        else:
            return datetime(date.year, date.month + 1, 1)
    elif base_freq == 'quarterly':
        # Premier jour du trimestre suivant
        quarter = (date.month - 1) // 3 + 1
        if quarter == 4:
            # Q4 -> Q1 de l'année suivante
            return datetime(date.year + 1, 1, 1)
        else:
            # Trimestre suivant dans la même année
            next_quarter_month = quarter * 3 + 1
            return datetime(date.year, next_quarter_month, 1)
    elif base_freq == 'annual':
        # Premier jour de l'année suivante
        return datetime(date.year + 1, 1, 1)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Base frequency '{base_freq}' is not recognized.")


# Fonction retournant les bornes d'une période
def get_period_boundaries(date: Union[pd.Timestamp, datetime], frequency: FrequencyType) -> tuple[datetime, datetime]:
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