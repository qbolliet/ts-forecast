"""Time manipulation utilities for time series processing."""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Literal
from datetime import timedelta

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
def get_period_start(date: pd.Timestamp, frequency: Literal['daily', 'weekly', 'monthly', 'quarterly', 'annual']) -> pd.Timestamp:
    """Get the start date of the period containing the given date.
    
    Args:
        date: Reference date
        frequency: Period frequency ('daily', 'weekly', 'monthly', 'quarterly', 'annual')
        
    Returns:
        Start date of the period
    """
    # Distinction suivant la fréquence
    if frequency == 'daily':
        return date.normalize()
    elif frequency == 'weekly':
        # Début de la semaine (lundi)
        return date - timedelta(days=date.weekday())
    elif frequency == 'monthly':
        return pd.Timestamp(year=date.year, month=date.month, day=1)
    elif frequency == 'quarterly':
        quarter_month = ((date.quarter - 1) * 3) + 1
        return pd.Timestamp(year=date.year, month=quarter_month, day=1)
    elif frequency == 'annual':
        return pd.Timestamp(year=date.year, month=1, day=1)
    else:
        raise ValueError(f"Unnexpected value for 'frequency' : {frequency}. Should be in ['daily', 'weekly', 'monthly', 'quarterly', 'annual']")

# Fonction identifiant la date de début d'une période à partir d'une date et d'une fréquence
def get_period_end(date: pd.Timestamp, frequency: Literal['daily', 'weekly', 'monthly', 'quarterly', 'annual']) -> pd.Timestamp:
    """Get the end date of the period containing the given date.
    
    Args:
        date: Reference date
        frequency: Period frequency
        
    Returns:
        End date of the period
    """
    # Distinction suivant la fréquence
    if frequency == 'daily':
        return date.normalize() + timedelta(days=1) - timedelta(seconds=1)
    elif frequency == 'weekly':
        # Fin de la semaine (dimanche)
        week_start = date - timedelta(days=date.weekday())
        return week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    elif frequency == 'monthly':
        # Dernier jour du mois
        if date.month == 12:
            next_month = pd.Timestamp(year=date.year + 1, month=1, day=1)
        else:
            next_month = pd.Timestamp(year=date.year, month=date.month + 1, day=1)
        return next_month - timedelta(seconds=1)
    elif frequency == 'quarterly':
        # Dernier jour du trimestre
        quarter_end_month = date.quarter * 3
        if quarter_end_month == 12:
            next_quarter = pd.Timestamp(year=date.year + 1, month=1, day=1)
        else:
            next_quarter = pd.Timestamp(year=date.year, month=quarter_end_month + 1, day=1)
        return next_quarter - timedelta(seconds=1)
    elif frequency == 'annual':
        return pd.Timestamp(year=date.year, month=12, day=31, hour=23, minute=59, second=59)
    else:
        raise ValueError(f"Unnexpected value for 'frequency' : {frequency}. Should be in ['daily', 'weekly', 'monthly', 'quarterly', 'annual']")
