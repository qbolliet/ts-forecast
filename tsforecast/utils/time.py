"""Time manipulation utilities for time series processing."""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Literal, Union
from datetime import datetime, timedelta

# Import des utilitaires de fréquence
try:
    from ..frequency.utils import normalize_frequency, get_base_frequency, get_reference_point, FrequencyType
except ImportError:
    # Fallback si les modules de fréquence ne sont pas disponibles
    FrequencyType = Union[str, pd.DateOffset]

    def normalize_frequency(freq):
        return str(freq)

    def get_base_frequency(freq):
        return str(freq).split('_')[0]

    def get_reference_point(freq):
        if '_start' in str(freq):
            return 'start'
        elif '_end' in str(freq):
            return 'end'
        return None

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
def get_period_start(date: Union[pd.Timestamp, datetime], frequency: FrequencyType) -> pd.Timestamp:
    """Get the start date of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (supports pandas codes, DateOffsets, and user-friendly names)

    Returns:
        Start date of the period

    Examples:
        >>> import pandas as pd
        >>> get_period_start(pd.Timestamp('2023-06-15'), 'monthly')
        Timestamp('2023-06-01 00:00:00')
        >>> get_period_start(pd.Timestamp('2023-06-15'), 'quarterly_start')
        Timestamp('2023-04-01 00:00:00')
    """
    # Normalisation de la fréquence et extraction des informations
    base_freq = get_base_frequency(frequency)
    reference = get_reference_point(frequency)

    # Distinction suivant la fréquence de base
    if base_freq == 'daily':
        return pd.Timestamp(date.normalize())
    elif base_freq == 'weekly':
        # Début de la semaine (lundi par défaut, dimanche si weekly_end)
        if reference == 'end':
            # Pour weekly_end, le début est le dimanche précédent
            days_to_subtract = (date.weekday() + 1) % 7
            return pd.Timestamp(date - timedelta(days=days_to_subtract))
        else:
            # Pour weekly ou weekly_start, le début est le lundi
            return pd.Timestamp(date - timedelta(days=date.weekday()))
    elif base_freq == 'monthly':
        return pd.Timestamp(year=date.year, month=date.month, day=1)
    elif base_freq == 'quarterly':
        quarter_month = ((date.quarter - 1) * 3) + 1
        return pd.Timestamp(year=date.year, month=quarter_month, day=1)
    elif base_freq == 'annual':
        return pd.Timestamp(year=date.year, month=1, day=1)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Base frequency '{base_freq}' is not recognized.")

# Fonction identifiant la date de début d'une période à partir d'une date et d'une fréquence
def get_period_end(date: Union[pd.Timestamp, datetime], frequency: FrequencyType) -> pd.Timestamp:
    """Get the end date of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (supports pandas codes, DateOffsets, and user-friendly names)

    Returns:
        First date outside the period (exclusive boundary)

    Examples:
        >>> import pandas as pd
        >>> get_period_end(pd.Timestamp('2023-06-15'), 'monthly')
        Timestamp('2023-07-01 00:00:00')
        >>> get_period_end(pd.Timestamp('2023-06-15'), 'quarterly_start')
        Timestamp('2023-07-01 00:00:00')
    """
    # Normalisation de la fréquence et extraction des informations
    base_freq = get_base_frequency(frequency)
    reference = get_reference_point(frequency)

    # Distinction suivant la fréquence de base
    if base_freq == 'daily':
        return pd.Timestamp(date.normalize() + timedelta(days=1))
    elif base_freq == 'weekly':
        # Calcul de la fin de semaine selon le point de référence
        if reference == 'end':
            # Pour weekly_end, la fin est le lundi suivant le dimanche
            days_to_subtract = (date.weekday() + 1) % 7
            week_start = pd.Timestamp(date - timedelta(days=days_to_subtract))
            return week_start + timedelta(days=7)
        else:
            # Pour weekly ou weekly_start, la fin est le lundi suivant
            week_start = pd.Timestamp(date - timedelta(days=date.weekday()))
            return week_start + timedelta(days=7)
    elif base_freq == 'monthly':
        # Premier jour du mois suivant
        if date.month == 12:
            return pd.Timestamp(year=date.year + 1, month=1, day=1)
        else:
            return pd.Timestamp(year=date.year, month=date.month + 1, day=1)
    elif base_freq == 'quarterly':
        # Premier jour du trimestre suivant
        quarter_end_month = date.quarter * 3
        if quarter_end_month == 12:
            return pd.Timestamp(year=date.year + 1, month=1, day=1)
        else:
            return pd.Timestamp(year=date.year, month=quarter_end_month + 1, day=1)
    elif base_freq == 'annual':
        return pd.Timestamp(year=date.year + 1, month=1, day=1)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Base frequency '{base_freq}' is not recognized.")


# Fonction retournant les bornes d'une période
def get_period_boundaries(date: Union[pd.Timestamp, datetime], frequency: FrequencyType) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get the start and end boundaries of the period containing the given date.

    Args:
        date: Reference date
        frequency: Period frequency (supports pandas codes, DateOffsets, and user-friendly names)

    Returns:
        Tuple containing (start_date, end_date) where start_date is included
        in the period [start_date, end_date) and end_date is excluded from it

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> date = pd.Timestamp('2023-06-15')
        >>> get_period_boundaries(date, 'monthly')
        (Timestamp('2023-06-01 00:00:00'), Timestamp('2023-07-01 00:00:00'))

        >>> get_period_boundaries(date, 'weekly_start')
        (Timestamp('2023-06-12 00:00:00'), Timestamp('2023-06-19 00:00:00'))

        >>> get_period_boundaries(date, 'Q')  # Pandas quarterly frequency
        (Timestamp('2023-04-01 00:00:00'), Timestamp('2023-07-01 00:00:00'))
    """
    # Calcul de début de la période
    period_start = get_period_start(date=date, frequency=frequency)
    # Calcul de la fin de la période
    period_end = get_period_end(date=date, frequency=frequency)

    return period_start, period_end