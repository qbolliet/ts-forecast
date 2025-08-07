# utils/time_utils.py
"""Time manipulation utilities for time series processing."""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple
from datetime import datetime, timedelta


class TimeUtils:
    """Utility class for time-related operations."""
    
    @staticmethod
    def get_period_start(date: pd.Timestamp, frequency: str) -> pd.Timestamp:
        """Get the start date of the period containing the given date.
        
        Args:
            date: Reference date
            frequency: Period frequency ('daily', 'weekly', 'monthly', 'quarterly', 'annual')
            
        Returns:
            Start date of the period
        """
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
            return date
    
    @staticmethod
    def get_period_end(date: pd.Timestamp, frequency: str) -> pd.Timestamp:
        """Get the end date of the period containing the given date.
        
        Args:
            date: Reference date
            frequency: Period frequency
            
        Returns:
            End date of the period
        """
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
            return date
    
    @staticmethod
    def generate_date_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate time-based features from dates.
        
        Args:
            dates: DatetimeIndex
            
        Returns:
            DataFrame with date features
        """
        features = pd.DataFrame(index=dates)
        
        # Features temporelles de base
        features['year'] = dates.year
        features['month'] = dates.month
        features['day'] = dates.day
        features['dayofweek'] = dates.dayofweek
        features['dayofyear'] = dates.dayofyear
        features['quarter'] = dates.quarter
        features['weekofyear'] = dates.isocalendar().week
        
        # Features cycliques (sin/cos pour capturer la périodicité)
        features['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
        features['day_sin'] = np.sin(2 * np.pi * dates.day / 31)
        features['day_cos'] = np.cos(2 * np.pi * dates.day / 31)
        features['dayofweek_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
        
        # Indicateurs booléens
        features['is_weekend'] = dates.dayofweek.isin([5, 6]).astype(int)
        features['is_month_start'] = dates.day == 1
        features['is_month_end'] = dates.is_month_end
        features['is_quarter_start'] = dates.is_quarter_start
        features['is_quarter_end'] = dates.is_quarter_end
        features['is_year_start'] = dates.is_year_start
        features['is_year_end'] = dates.is_year_end
        
        return features
    
    @staticmethod
    def create_lag_features(series: pd.Series, 
                           lags: List[int],
                           date_index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """Create lag features for a time series.
        
        Args:
            series: Time series data
            lags: List of lag periods to create
            date_index: Optional datetime index
            
        Returns:
            DataFrame with lag features
        """
        if date_index is not None:
            series = series.copy()
            series.index = date_index
        
        lag_features = pd.DataFrame(index=series.index)
        
        for lag in lags:
            lag_features[f'lag_{lag}'] = series.shift(lag)
        
        return lag_features
    
    @staticmethod
    def create_rolling_features(series: pd.Series,
                               windows: List[int],
                               operations: List[str] = ['mean', 'std', 'min', 'max'],
                               min_periods: Optional[int] = None) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            series: Time series data
            windows: List of window sizes
            operations: List of operations to apply
            min_periods: Minimum observations in window
            
        Returns:
            DataFrame with rolling features
        """
        rolling_features = pd.DataFrame(index=series.index)
        
        for window in windows:
            for op in operations:
                feature_name = f'rolling_{window}_{op}'
                
                if op == 'mean':
                    rolling_features[feature_name] = series.rolling(
                        window=window, min_periods=min_periods
                    ).mean()
                elif op == 'std':
                    rolling_features[feature_name] = series.rolling(
                        window=window, min_periods=min_periods
                    ).std()
                elif op == 'min':
                    rolling_features[feature_name] = series.rolling(
                        window=window, min_periods=min_periods
                    ).min()
                elif op == 'max':
                    rolling_features[feature_name] = series.rolling(
                        window=window, min_periods=min_periods
                    ).max()
                elif op == 'sum':
                    rolling_features[feature_name] = series.rolling(
                        window=window, min_periods=min_periods
                    ).sum()
        
        return rolling_features