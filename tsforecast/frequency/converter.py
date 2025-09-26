"""Frequency conversion utilities for time series data.

This module provides the FrequencyConverter class to handle conversions between
different time frequencies using pandas built-in functionality (asfreq and resample).
"""
# Importation des modules
import pandas as pd
import numpy as np
from typing import Union, Optional, Literal, Dict, Any
from pandas.tseries.frequencies import to_offset

# Import des utilitaires de fréquence
from .utils import normalize_frequency, to_pandas_freq, get_base_frequency, is_higher_frequency


# Types pour les méthodes d'agrégation et d'interpolation
AggregationMethod = Literal['mean', 'sum', 'first', 'last', 'min', 'max', 'median', 'std', 'count']
InterpolationMethod = Literal['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']


class FrequencyConverter:
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

    def __init__(self):
        """Initialize the FrequencyConverter."""
        pass

    def convert_frequency(self,
                         data: Union[pd.Series, pd.DataFrame],
                         target_freq: str,
                         method: Union[AggregationMethod, InterpolationMethod] = 'mean',
                         fill_method: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """Convert data to target frequency using pandas built-in methods.

        This is the main conversion method that automatically determines whether
        to use upsampling (asfreq) or downsampling (resample) based on the
        frequency relationship.

        Args:
            data: Time series data to convert
            target_freq: Target frequency (user-friendly or pandas format)
            method: Aggregation method for downsampling or interpolation method for upsampling
            fill_method: Fill method for missing values ('ffill', 'bfill', None)

        Returns:
            Converted time series data

        Raises:
            ValueError: If conversion parameters are invalid

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> daily_dates = pd.date_range('2023-01-01', periods=31, freq='D')
            >>> daily_series = pd.Series(range(31), index=daily_dates)
            >>> monthly = converter.convert_frequency(daily_series, 'monthly', method='mean')
            >>> isinstance(monthly, pd.Series)
            True
        """
        # Validation des paramètres d'entrée
        self._validate_conversion_params(data, target_freq, method)

        # Détection de la fréquence actuelle
        current_freq = self._detect_current_frequency(data)
        if not current_freq:
            raise ValueError("Cannot detect current frequency of the data")

        # Normalisation de la fréquence cible
        target_freq_normalized = to_pandas_freq(target_freq)

        # Si les fréquences sont identiques, retourner les données telles quelles
        if normalize_frequency(current_freq) == target_freq_normalized:
            return data

        # Détermination de la direction de conversion
        if self._is_upsampling(current_freq, target_freq):
            return self._upsample(data, target_freq_normalized, method, fill_method)
        else:
            return self._downsample(data, target_freq_normalized, method)

    def aggregate_to_lower_frequency(self,
                                   data: Union[pd.Series, pd.DataFrame],
                                   target_freq: str,
                                   method: AggregationMethod = 'mean') -> Union[pd.Series, pd.DataFrame]:
        """Aggregate data to a lower frequency using resample.

        Args:
            data: Time series data to aggregate
            target_freq: Target frequency (must be lower than current)
            method: Aggregation method

        Returns:
            Aggregated time series data

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> daily_dates = pd.date_range('2023-01-01', periods=31, freq='D')
            >>> daily_series = pd.Series(range(31), index=daily_dates)
            >>> monthly = converter.aggregate_to_lower_frequency(daily_series, 'monthly', 'sum')
            >>> len(monthly)
            1
        """
        target_freq_pandas = to_pandas_freq(target_freq)
        resampled = data.resample(target_freq_pandas)

        # Application de la méthode d'agrégation
        if method == 'mean':
            return resampled.mean()
        elif method == 'sum':
            return resampled.sum()
        elif method == 'first':
            return resampled.first()
        elif method == 'last':
            return resampled.last()
        elif method == 'min':
            return resampled.min()
        elif method == 'max':
            return resampled.max()
        elif method == 'median':
            return resampled.median()
        elif method == 'std':
            return resampled.std()
        elif method == 'count':
            return resampled.count()
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    def interpolate_to_higher_frequency(self,
                                      data: Union[pd.Series, pd.DataFrame],
                                      target_freq: str,
                                      method: InterpolationMethod = 'linear',
                                      fill_method: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """Interpolate data to a higher frequency using asfreq.

        Args:
            data: Time series data to interpolate
            target_freq: Target frequency (must be higher than current)
            method: Interpolation method
            fill_method: Fill method for missing values

        Returns:
            Interpolated time series data

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> monthly_dates = pd.date_range('2023-01-01', periods=3, freq='M')
            >>> monthly_series = pd.Series([10, 20, 30], index=monthly_dates)
            >>> daily = converter.interpolate_to_higher_frequency(monthly_series, 'daily', 'linear')
            >>> len(daily) > len(monthly_series)
            True
        """
        target_freq_pandas = to_pandas_freq(target_freq)

        # Utilisation d'asfreq pour créer la nouvelle fréquence
        upsampled = data.asfreq(target_freq_pandas)

        # Application du remplissage si spécifié
        if fill_method == 'ffill':
            upsampled = upsampled.fillna(method='ffill')
        elif fill_method == 'bfill':
            upsampled = upsampled.fillna(method='bfill')

        # Application de l'interpolation
        if method == 'linear':
            return upsampled.interpolate(method='linear')
        elif method == 'time':
            return upsampled.interpolate(method='time')
        elif method == 'index':
            return upsampled.interpolate(method='index')
        elif method == 'values':
            return upsampled.interpolate(method='values')
        elif method == 'nearest':
            return upsampled.interpolate(method='nearest')
        elif method == 'zero':
            return upsampled.interpolate(method='zero')
        elif method == 'slinear':
            return upsampled.interpolate(method='slinear')
        elif method == 'quadratic':
            return upsampled.interpolate(method='quadratic')
        elif method == 'cubic':
            return upsampled.interpolate(method='cubic')
        else:
            return upsampled

    def align_frequencies(self,
                        *datasets: Union[pd.Series, pd.DataFrame],
                        target_freq: Optional[str] = None,
                        method: str = 'mean') -> tuple:
        """Align multiple datasets to the same frequency.

        Args:
            *datasets: Variable number of time series datasets
            target_freq: Target frequency (if None, uses the highest common frequency)
            method: Conversion method to use

        Returns:
            Tuple of aligned datasets

        Examples:
            >>> import pandas as pd
            >>> converter = FrequencyConverter()
            >>> daily_dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> monthly_dates = pd.date_range('2023-01-01', periods=2, freq='M')
            >>> daily_data = pd.Series(range(5), index=daily_dates)
            >>> monthly_data = pd.Series([10, 20], index=monthly_dates)
            >>> aligned = converter.align_frequencies(daily_data, monthly_data, target_freq='monthly')
            >>> len(aligned) == 2
            True
        """
        if not datasets:
            return tuple()

        # Détection des fréquences actuelles
        current_freqs = []
        for dataset in datasets:
            freq = self._detect_current_frequency(dataset)
            if freq:
                current_freqs.append(freq)

        if not current_freqs:
            raise ValueError("Cannot detect frequency for any dataset")

        # Détermination de la fréquence cible si non spécifiée
        if target_freq is None:
            # Utilisation de la fréquence la plus basse (moins granulaire)
            freq_orders = {}
            for freq in set(current_freqs):
                base_freq = get_base_frequency(freq)
                freq_orders[freq] = self._get_frequency_order(base_freq)

            target_freq = max(freq_orders.keys(), key=lambda x: freq_orders[x])

        # Conversion de tous les datasets vers la fréquence cible
        aligned_datasets = []
        for dataset in datasets:
            aligned = self.convert_frequency(dataset, target_freq, method=method)
            aligned_datasets.append(aligned)

        return tuple(aligned_datasets)

    def _validate_conversion_params(self,
                                  data: Union[pd.Series, pd.DataFrame],
                                  target_freq: str,
                                  method: str) -> None:
        """Validate conversion parameters.

        Args:
            data: Input data
            target_freq: Target frequency
            method: Conversion method

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError("Data must be a pandas Series or DataFrame")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")

        if not target_freq:
            raise ValueError("Target frequency cannot be empty")

        try:
            normalize_frequency(target_freq)
        except ValueError as e:
            raise ValueError(f"Invalid target frequency: {e}")

    def _detect_current_frequency(self, data: Union[pd.Series, pd.DataFrame]) -> Optional[str]:
        """Detect the current frequency of the data.

        Args:
            data: Input data

        Returns:
            Detected frequency or None
        """
        try:
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq:
                return inferred_freq
        except Exception:
            pass
        return None

    def _is_upsampling(self, current_freq: str, target_freq: str) -> bool:
        """Determine if conversion is upsampling (to higher frequency).

        Args:
            current_freq: Current frequency
            target_freq: Target frequency

        Returns:
            True if upsampling, False if downsampling
        """
        return is_higher_frequency(target_freq, current_freq)

    def _upsample(self,
                 data: Union[pd.Series, pd.DataFrame],
                 target_freq: str,
                 method: str,
                 fill_method: Optional[str]) -> Union[pd.Series, pd.DataFrame]:
        """Perform upsampling using asfreq and interpolation.

        Args:
            data: Input data
            target_freq: Target frequency (pandas format)
            method: Interpolation method
            fill_method: Fill method for missing values

        Returns:
            Upsampled data
        """
        return self.interpolate_to_higher_frequency(data, target_freq, method, fill_method)

    def _downsample(self,
                   data: Union[pd.Series, pd.DataFrame],
                   target_freq: str,
                   method: str) -> Union[pd.Series, pd.DataFrame]:
        """Perform downsampling using resample and aggregation.

        Args:
            data: Input data
            target_freq: Target frequency (pandas format)
            method: Aggregation method

        Returns:
            Downsampled data
        """
        return self.aggregate_to_lower_frequency(data, target_freq, method)

    def _get_frequency_order(self, base_freq: str) -> int:
        """Get the order of a frequency for comparison.

        Args:
            base_freq: Base frequency name

        Returns:
            Frequency order (higher number = lower frequency)
        """
        frequency_order = {
            'daily': 1,
            'business_daily': 1.5,
            'weekly': 2,
            'monthly': 3,
            'quarterly': 4,
            'annual': 5
        }
        return frequency_order.get(base_freq, 0)