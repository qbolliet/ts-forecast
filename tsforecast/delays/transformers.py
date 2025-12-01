"""Sklearn-compatible transformers for applying publication delays.

This module provides a modular architecture with:
- ShiftTransformer: Pure helper to shift data by N periods
- MaskTransformer: Pure helper to mask N observations per period
- PublicationDelayTransformer: Intelligent orchestrator that handles inference, frequency detection, and panel wrapping
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Literal
from datetime import datetime
import warnings

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# Importation des modules du package
from tsforecast.utils.frequency import normalize_frequency, to_pandas_freq
from tsforecast.utils.time import resolve_date, get_period_boundaries


class ShiftTransformer(BaseEstimator, TransformerMixin):
    """Simple helper to shift time series data by N periods.

    This is a pure operational transformer with no inference logic.
    Panel data handling is done by the orchestrator (PublicationDelayTransformer).

    Parameters:
        n_periods: Number of periods to shift (can be negative for future predictions)
        frequency: Frequency for period arithmetic ('D', 'M', 'Q', 'W', 'h', etc.)

    Attributes:
        n_periods: Stored number of periods to shift
        frequency: Stored frequency code

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='M')
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates, name='GDP')
        >>>
        >>> # Shift forward by 2 monthly periods
        >>> shifter = ShiftTransformer(n_periods=2, frequency='M')
        >>> shifted = shifter.fit_transform(series)
        >>>
        >>> # Inverse shift
        >>> original = shifter.inverse_transform(shifted)
    """

    def __init__(self, n_periods: int, frequency: str):
        """Initialize ShiftTransformer.

        Args:
            n_periods: Number of periods to shift (can be negative)
            frequency: Frequency for period arithmetic ('D', 'M', 'Q', etc.)
        """
        self.n_periods = n_periods
        self.frequency = frequency

    def fit(self, X: pd.Series, y=None):
        """Fit transformer (no-op for ShiftTransformer).

        Args:
            X: Time series to fit
            y: Ignored

        Returns:
            self
        """
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Shift series by n_periods at given frequency.

        Args:
            X: Time series to transform

        Returns:
            Shifted time series

        Raises:
            ValueError: If X is not a pandas Series or doesn't have DatetimeIndex
        """
        # Validation de l'entrée
        if not isinstance(X, pd.Series):
            raise ValueError("X must be a pandas Series")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        return self._shift_by_periods(X, self.n_periods, self.frequency)

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        """Inverse shift by -n_periods.

        Args:
            X: Transformed series

        Returns:
            Series shifted back by -n_periods
        """
        if not isinstance(X, pd.Series):
            raise ValueError("X must be a pandas Series")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        return self._shift_by_periods(X, -self.n_periods, self.frequency)

    def _shift_by_periods(self, series: pd.Series, n_periods: int, frequency: str) -> pd.Series:
        """Core shift logic using PeriodIndex.

        Args:
            series: Series to shift
            n_periods: Number of periods to shift
            frequency: Frequency for period arithmetic

        Returns:
            Shifted series with original index structure
        """
        # Cas où aucun shift n'est nécessaire
        if n_periods == 0:
            return series.copy()

        # Normalisation de la fréquence
        pandas_freq = to_pandas_freq(frequency)

        # Conversion en PeriodIndex pour validation de la fréquence
        # Puis utilisation du shift standard de pandas qui décale les valeurs
        period_index = series.index.to_period(freq=pandas_freq)
        series_period = pd.Series(series.values, index=period_index, name=series.name)

        # Shift standard : décale les valeurs, l'index reste fixe
        shifted_period = series_period.shift(n_periods)

        # Reconversion en DatetimeIndex en préservant les timestamps originaux
        # On map chaque période de l'index original vers son timestamp original
        period_to_timestamp = dict(zip(period_index, series.index))

        # Construction de l'index de sortie
        result_index = pd.DatetimeIndex([
            period_to_timestamp.get(period, pd.NaT)
            for period in shifted_period.index
        ])

        # Création de la série avec l'index datetime restauré
        result = pd.Series(shifted_period.values, index=result_index, name=series.name)

        # Filtrage des NaT (périodes qui n'existent pas dans l'original)
        # et alignement sur l'index original
        return result.reindex(series.index)


class MaskTransformer(BaseEstimator, TransformerMixin):
    """Simple helper to mask N observations per period.

    This is a pure operational transformer with no inference logic.
    Panel data handling is done by the orchestrator (PublicationDelayTransformer).

    Parameters:
        n_obs: Number of observations to mask per period
        mask_frequency: Frequency for period grouping ('D', 'M', 'Q', 'W', 'h', etc.)
        prediction_date: Reference date for masking

    Attributes:
        n_obs: Stored number of observations to mask per period
        mask_frequency: Stored frequency for period grouping
        prediction_date: Stored prediction date
        original_data_: Original data before masking (for inverse_transform)

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> dates = pd.date_range('2024-01-01', periods=90, freq='D')
        >>> series = pd.Series(range(90), index=dates, name='GDP')
        >>>
        >>> # Mask 2 most recent observations per month
        >>> masker = MaskTransformer(
        ...     n_obs=2,
        ...     mask_frequency='M',
        ...     prediction_date=datetime(2024, 3, 31)
        ... )
        >>> masked = masker.fit_transform(series)
        >>>
        >>> # Restore original data
        >>> original = masker.inverse_transform(masked)
    """

    def __init__(self, n_obs: int, mask_frequency: str, prediction_date: datetime):
        """Initialize MaskTransformer.

        Args:
            n_obs: Number of observations to mask per period
            mask_frequency: Frequency for period grouping ('D', 'M', 'Q', etc.)
            prediction_date: Reference date for masking
        """
        self.n_obs = n_obs
        self.mask_frequency = mask_frequency
        self.prediction_date = prediction_date
        self.original_data_ = None

    def fit(self, X: pd.Series, y=None):
        """Fit transformer (no-op for MaskTransformer).

        Args:
            X: Time series to fit
            y: Ignored

        Returns:
            self
        """
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Mask N most recent observations per period.

        Args:
            X: Time series to transform

        Returns:
            Masked time series

        Raises:
            ValueError: If X is not a pandas Series or doesn't have DatetimeIndex
        """
        # Validation de l'entrée
        if not isinstance(X, pd.Series):
            raise ValueError("X must be a pandas Series")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        # Stockage des données originales pour inverse_transform
        self.original_data_ = X.copy()

        return self._mask_n_obs_per_period(X)

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        """Restore masked values from original data.

        Args:
            X: Masked series

        Returns:
            Original series before masking

        Raises:
            ValueError: If transform has not been called yet
        """
        if self.original_data_ is None:
            raise ValueError("Must call transform before inverse_transform")

        return self.original_data_.copy()

    def _mask_n_obs_per_period(self, series: pd.Series) -> pd.Series:
        """Core masking logic: mask N most recent obs per period.

        Args:
            series: Series to mask

        Returns:
            Series with masked observations
        """
        # Cas où aucun masquage n'est nécessaire
        if self.n_obs == 0:
            return series.copy()

        masked_series = series.copy()

        # Génération des périodes
        periods = self._generate_periods(
            start_date=series.index.min(),
            end_date=self.prediction_date,
            frequency=self.mask_frequency
        )

        # Masquage dans chaque période
        for period_start, period_end in periods:
            # Observations dans cette période
            period_mask = (series.index >= period_start) & (series.index < period_end)
            period_obs = series[period_mask]

            if len(period_obs) > 0:
                # Masquer les n_obs plus récentes
                n_to_mask = min(self.n_obs, len(period_obs))
                most_recent_indices = period_obs.index[-n_to_mask:]
                masked_series.loc[most_recent_indices] = np.nan

        return masked_series

    def _generate_periods(
        self,
        start_date: pd.Timestamp,
        end_date: datetime,
        frequency: str
    ) -> List[tuple[datetime, datetime]]:
        """Generate list of (period_start, period_end) tuples.

        Args:
            start_date: Start date for period generation
            end_date: End date for period generation
            frequency: Frequency code for periods

        Returns:
            List of (period_start, period_end) tuples
        """
        # Normalisation de la fréquence
        pandas_freq = to_pandas_freq(frequency)

        # Génération des dates de début de période
        period_starts = pd.date_range(
            start=start_date,
            end=end_date,
            freq=pandas_freq
        )

        # Création des tuples (start, end) pour chaque période
        periods = []
        for period_date in period_starts:
            period_start, period_end = get_period_boundaries(period_date, frequency)
            periods.append((period_start, period_end))

        return periods


class PublicationDelayTransformer(BaseEstimator, TransformerMixin):
    """Intelligent orchestrator for applying publication delays to time series/panel data.

    This transformer handles:
    - Parameter inference from delays DataFrame
    - Frequency detection per column
    - Period-based calculations (not day-based)
    - Automatic panel wrapping with PanelwiseTransformer
    - Warning generation for all-NaN columns

    Parameters:
        delays: Delays specification (Dict or DataFrame)
        strategy: Transformation strategy ('shift' or 'mask')
        delay_unit: Unit of delay ('D', 's', 'h', etc.). If None, inferred from DataFrame
        reference_point: Reference point ('start' or 'end'). If None, inferred from DataFrame
        target_frequency: Target frequency for delay calculation. If None, uses column frequency
        prediction_date: Date of prediction (required for 'mask' strategy)
        time_col: Name of time column (default: 'date')
        panel_cols: Panel column names (None for non-panel data)
        handle_missing_delays: Strategy for missing delays ('ignore', 'warn', 'error')
        default_delay: Default delay value if missing

    Attributes:
        column_transformers_: Dict mapping column names to helper transformers
        inferred_params_: Dict of parameters inferred from delays DataFrame
        detected_frequencies_: Dict of detected frequencies per column

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>>
        >>> # Create delays DataFrame with metadata
        >>> delays_df = pd.DataFrame({
        ...     'variable': ['GDP', 'inflation'],
        ...     'delay': [45.0, 30.0],
        ...     'unit': ['D', 'D'],
        ...     'reference_point': ['end', 'end'],
        ...     'target_frequency': ['M', 'M']
        ... })
        >>>
        >>> # Create transformer (parameters inferred from DataFrame)
        >>> transformer = PublicationDelayTransformer(
        ...     delays=delays_df,
        ...     strategy='shift',
        ...     prediction_date=datetime(2024, 12, 15)
        ... )
        >>>
        >>> # Apply transformation
        >>> X_shifted = transformer.fit_transform(X)
        >>>
        >>> # Reverse transformation
        >>> X_original = transformer.inverse_transform(X_shifted)
    """

    def __init__(
        self,
        delays: Union[Dict[str, float], pd.DataFrame],
        strategy: str = 'shift',
        delay_unit: Optional[str] = None,
        reference_point: Optional[str] = None,
        target_frequency: Optional[str] = None,
        prediction_date: Union[str, datetime] = 'today',
        time_col: str = 'date',
        panel_cols: Optional[List[str]] = None,
        handle_missing_delays: str = 'warn',
        default_delay: Optional[float] = None
    ):
        """Initialize PublicationDelayTransformer.

        Args:
            delays: Dict mapping variable names to delays, or DataFrame with delays
            strategy: 'shift' or 'mask'
            delay_unit: Unit of delay (inferred from DataFrame if None)
            reference_point: 'start' or 'end' (inferred from DataFrame if None)
            target_frequency: Target frequency (inferred from DataFrame if None)
            prediction_date: Prediction date
            time_col: Time column name
            panel_cols: Panel column names
            handle_missing_delays: 'ignore', 'warn', or 'error'
            default_delay: Default delay if missing
        """
        # Validation des paramètres
        if strategy not in ['shift', 'mask']:
            raise ValueError(f"strategy must be 'shift' or 'mask', got '{strategy}'")
        if reference_point is not None and reference_point not in ['start', 'end']:
            raise ValueError(f"reference_point must be 'start' or 'end', got '{reference_point}'")
        if handle_missing_delays not in ['ignore', 'warn', 'error']:
            raise ValueError(f"handle_missing_delays must be 'ignore', 'warn', or 'error', got '{handle_missing_delays}'")

        # Stockage des paramètres
        self.delays = delays
        self.strategy = strategy
        self.delay_unit = delay_unit
        self.reference_point = reference_point
        self.target_frequency = target_frequency
        self.prediction_date = prediction_date
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.handle_missing_delays = handle_missing_delays
        self.default_delay = default_delay

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None):
        """Fit transformer by inferring parameters and preparing helpers.

        Args:
            X: Time series or panel data
            y: Ignored

        Returns:
            self
        """
        # Résolution de la date de prédiction
        self.prediction_date_ = resolve_date(self.prediction_date)

        # Inférence des paramètres depuis delays DataFrame si nécessaire
        self.inferred_params_ = self._infer_parameters_from_delays()

        # Détermination des valeurs finales (explicites > inférées > défaut)
        delay_unit_final = self.delay_unit or self.inferred_params_.get('delay_unit', 'D')
        reference_point_final = self.reference_point or self.inferred_params_.get('reference_point', 'end')

        # Conversion des delays en dictionnaire si DataFrame
        if isinstance(self.delays, pd.DataFrame):
            delays_dict = dict(zip(self.delays['variable'], self.delays['delay']))
        else:
            delays_dict = self.delays

        # Détection du type de données (Series ou DataFrame)
        self.is_series_ = isinstance(X, pd.Series)

        # Détection des fréquences par colonne
        self.detected_frequencies_ = self._detect_frequencies(X)

        # Création des transformers pour chaque colonne
        self.column_transformers_ = {}

        if self.is_series_:
            # Cas d'une Series : un seul transformer
            column_name = X.name or 'series'
            self._fit_column_transformer(
                column_name=column_name,
                series=X,
                delays_dict=delays_dict,
                delay_unit=delay_unit_final,
                reference_point=reference_point_final
            )
        else:
            # Cas d'un DataFrame : un transformer par colonne
            for column_name in X.columns:
                if column_name == self.time_col:
                    continue  # Skip time column
                if self.panel_cols and column_name in self.panel_cols:
                    continue  # Skip panel columns

                self._fit_column_transformer(
                    column_name=column_name,
                    series=X[column_name],
                    delays_dict=delays_dict,
                    delay_unit=delay_unit_final,
                    reference_point=reference_point_final
                )

        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Apply publication delays to data.

        Args:
            X: Time series or panel data

        Returns:
            Transformed data with publication delays applied
        """
        if self.is_series_:
            # Transform series
            column_name = X.name or 'series'
            if column_name in self.column_transformers_:
                result = self.column_transformers_[column_name].transform(X)
                # Check for all-NaN
                if result.isna().all():
                    warnings.warn(
                        f"Column '{column_name}' became all-NaN after applying {self.strategy} strategy"
                    )
                return result
            else:
                return X.copy()
        else:
            # Transform DataFrame column by column
            X_result = X.copy()
            for column_name, transformer in self.column_transformers_.items():
                if column_name in X_result.columns:
                    X_result[column_name] = transformer.transform(X_result[column_name])
                    # Check for all-NaN
                    if X_result[column_name].isna().all():
                        warnings.warn(
                            f"Column '{column_name}' became all-NaN after applying {self.strategy} strategy"
                        )
            return X_result

    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Reverse publication delay transformation.

        Args:
            X: Transformed data

        Returns:
            Data with delays reversed
        """
        if self.is_series_:
            # Inverse transform series
            column_name = X.name or 'series'
            if column_name in self.column_transformers_:
                return self.column_transformers_[column_name].inverse_transform(X)
            else:
                return X.copy()
        else:
            # Inverse transform DataFrame column by column
            X_result = X.copy()
            for column_name, transformer in self.column_transformers_.items():
                if column_name in X_result.columns:
                    X_result[column_name] = transformer.inverse_transform(X_result[column_name])
            return X_result

    def _infer_parameters_from_delays(self) -> Dict[str, any]:
        """Infer delay_unit, reference_point, target_frequency from delays DataFrame.

        Returns:
            Dict of inferred parameters
        """
        inferred = {}

        if isinstance(self.delays, pd.DataFrame):
            # Infer delay_unit from 'unit' column
            if 'unit' in self.delays.columns:
                # Use first value (assume consistent across all variables)
                inferred['delay_unit'] = self.delays['unit'].iloc[0]

            # Infer reference_point from 'reference_point' column
            if 'reference_point' in self.delays.columns:
                inferred['reference_point'] = self.delays['reference_point'].iloc[0]

            # Infer target_frequency from 'target_frequency' column
            if 'target_frequency' in self.delays.columns:
                # Store as dict mapping variable to target_frequency
                inferred['target_frequencies'] = dict(
                    zip(self.delays['variable'], self.delays['target_frequency'])
                )

        return inferred

    def _detect_frequencies(self, X: Union[pd.Series, pd.DataFrame]) -> Dict[str, str]:
        """Detect frequencies for each column.

        Args:
            X: Input data

        Returns:
            Dict mapping column names to detected frequencies
        """
        from ..frequency.detector import detect_frequency

        frequencies = {}

        # Mapping pour convertir pandas 2.2+ codes vers codes compatibles
        freq_mapping = {
            'ME': 'M',    # MonthEnd
            'MS': 'M',    # MonthStart (on traite comme mensuel)
            'QE': 'Q',    # QuarterEnd
            'QS': 'Q',    # QuarterStart
            'YE': 'Y',    # YearEnd
            'YS': 'Y',    # YearStart
            'BME': 'B',   # BusinessMonthEnd
            'BMS': 'B',   # BusinessMonthStart
        }

        if self.is_series_:
            column_name = X.name or 'series'
            freq = detect_frequency(X)
            # Conversion si nécessaire
            freq = freq_mapping.get(freq, freq) if freq else freq
            frequencies[column_name] = freq
        else:
            # Detect frequency per column
            for column_name in X.columns:
                if column_name == self.time_col:
                    continue
                if self.panel_cols and column_name in self.panel_cols:
                    continue

                freq = detect_frequency(X[column_name])
                # Conversion si nécessaire
                freq = freq_mapping.get(freq, freq) if freq else freq
                frequencies[column_name] = freq

        return frequencies

    def _fit_column_transformer(
        self,
        column_name: str,
        series: pd.Series,
        delays_dict: Dict[str, float],
        delay_unit: str,
        reference_point: str
    ):
        """Fit helper transformer for a single column.

        Args:
            column_name: Name of the column
            series: Series data for the column
            delays_dict: Dictionary of delays
            delay_unit: Unit of delay
            reference_point: Reference point
        """
        # Obtention du délai pour cette colonne
        if column_name in delays_dict:
            applicable_delay = delays_dict[column_name]
        elif self.default_delay is not None:
            applicable_delay = self.default_delay
        else:
            if self.handle_missing_delays == 'error':
                raise ValueError(f"No delay found for column '{column_name}'")
            elif self.handle_missing_delays == 'warn':
                warnings.warn(f"No delay found for column '{column_name}', skipping transformation")
            return  # Skip this column

        # Obtention de la fréquence détectée
        column_frequency = self.detected_frequencies_.get(column_name)
        if column_frequency is None:
            warnings.warn(f"Could not detect frequency for column '{column_name}', skipping transformation")
            return

        # Détermination de la target_frequency
        target_freq = None
        if self.target_frequency is not None:
            target_freq = self.target_frequency
        elif 'target_frequencies' in self.inferred_params_ and column_name in self.inferred_params_['target_frequencies']:
            target_freq = self.inferred_params_['target_frequencies'][column_name]
        # Si target_freq reste None, calculate_n_periods_delay utilisera column_frequency

        # Calcul du nombre de périodes
        from .period_utils import calculate_n_periods_delay

        # Utilisation d'une observation typique pour le calcul
        observation_date = series.index[len(series) // 2]

        n_periods = calculate_n_periods_delay(
            applicable_delay=applicable_delay,
            delay_unit=delay_unit,
            prediction_date=self.prediction_date_,
            reference_point=reference_point,
            observation_date=observation_date,
            column_frequency=column_frequency,
            target_frequency=target_freq
        )

        # Création du helper transformer approprié
        if self.strategy == 'shift':
            helper = ShiftTransformer(
                n_periods=n_periods,
                frequency=target_freq or column_frequency
            )
        else:  # mask
            helper = MaskTransformer(
                n_obs=abs(n_periods),  # MaskTransformer expects positive integer
                mask_frequency=target_freq or column_frequency,
                prediction_date=self.prediction_date_
            )

        # Stockage du transformer
        self.column_transformers_[column_name] = helper
