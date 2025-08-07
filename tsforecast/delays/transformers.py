"""Release delay transformer for time series data.

This module provides a scikit-learn compatible transformer to handle
publication delays in time series and panel data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings
from datetime import datetime
import copy

from ..utils.frequency_detection import FrequencyDetector, FrequencyConverter


class ReleaseDelayTransformer(BaseEstimator, TransformerMixin):
    """Transform time series data by applying publication delays.
    
    This transformer shifts time series to handle publication delays, ensuring
    no missing values at the prediction date. It supports both univariate time
    series and panel data.
    
    Parameters:
        prediction_date (Union[str, pd.Timestamp]): Reference date for delay calculation.
            Use 'today' for current date. Default is 'today'.
        time_col (str): Name of the time column in the data. Default is 'date'.
        panel_cols (Optional[List[str]]): Columns identifying panel dimensions.
            None for univariate time series.
        min_observations (int): Minimum observations required for frequency detection.
            Default is 10.
        store_original_index (bool): Whether to store original index for inverse transform.
            Default is True.
            
    Attributes:
        release_delays_ (Dict): Computed publication delays for each series.
        frequency_shifts_ (Dict): Frequency adjustment shifts.
        data_frequency_ (str): Detected dataset frequency.
        is_panel_ (bool): Whether data is panel format.
        original_index_ (pd.Index): Original data index if stored.
        reference_dates_ (Dict): Reference dates for each series.
    """
    
    def __init__(self,
                 prediction_date: Union[str, pd.Timestamp] = 'today',
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 min_observations: int = 10,
                 store_original_index: bool = True):
        """Initialize the ReleaseDelayTransformer."""
        self.prediction_date = prediction_date
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.min_observations = min_observations
        self.store_original_index = store_original_index
        
        # Initialisation des détecteurs de fréquence
        self._freq_detector = FrequencyDetector(min_observations)
        self._freq_converter = FrequencyConverter()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ReleaseDelayTransformer':
        """Compute publication delays for the dataset.
        
        Args:
            X: Input features as a DataFrame with datetime index or time column
            y: Target variable (not used, included for sklearn compatibility)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If data format is invalid or frequencies are inconsistent
        """
        # Validation des données d'entrée
        X = self._validate_input(X)
        
        # Détermination de la date de prédiction effective
        self._set_prediction_date(X)
        
        # Détection de la structure des données (panel ou série temporelle)
        self.is_panel_ = self.panel_cols is not None and len(self.panel_cols) > 0
        
        # Stockage de l'index original si demandé
        if self.store_original_index:
            self.original_index_ = X.index.copy()
        
        # Détection des fréquences pour chaque série
        self.frequency_map_ = self._freq_detector.detect_dataset_frequency(
            X, self.time_col, self.panel_cols
        )
        
        # Validation de la cohérence des fréquences
        is_consistent, common_freq = self._freq_detector.validate_frequency_consistency(
            self.frequency_map_, strict=False
        )
        
        if not is_consistent:
            warnings.warn(
                "Inconsistent frequencies detected across series. "
                "Using most common frequency for reference."
            )
        
        self.data_frequency_ = common_freq
        
        # Calcul des délais de publication
        self.release_delays_ = self._compute_delays(X)
        
        # Calcul des ajustements de fréquence
        self.frequency_shifts_ = self._compute_frequency_adjustments(X)
        
        # Stockage des dates de référence pour chaque série
        self.reference_dates_ = self._compute_reference_dates(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply publication delays to the data.
        
        Args:
            X: Input features to transform
            
        Returns:
            Transformed DataFrame with delays applied
            
        Raises:
            NotFittedError: If transform is called before fit
        """
        # Vérification que le transformer a été ajusté
        check_is_fitted(self, ['release_delays_', 'data_frequency_'])
        
        # Validation des données
        X = self._validate_input(X)
        
        # Copie des données pour éviter les modifications in-place
        X_transformed = X.copy()
        
        # Application des délais selon le type de données
        if self.is_panel_:
            X_transformed = self._apply_delays_panel(X_transformed)
        else:
            X_transformed = self._apply_delays_series(X_transformed)
        
        return X_transformed
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the applied publication delays.
        
        Args:
            X: Transformed data
            
        Returns:
            Original data with delays removed
            
        Raises:
            NotFittedError: If inverse_transform is called before fit
        """
        # Vérification que le transformer a été ajusté
        check_is_fitted(self, ['release_delays_', 'data_frequency_'])
        
        # Validation des données
        X = self._validate_input(X)
        
        # Copie des données
        X_original = X.copy()
        
        # Inversion des délais
        if self.is_panel_:
            X_original = self._reverse_delays_panel(X_original)
        else:
            X_original = self._reverse_delays_series(X_original)
        
        # Restauration de l'index original si disponible
        if hasattr(self, 'original_index_') and len(X_original) == len(self.original_index_):
            X_original.index = self.original_index_
        
        return X_original
    
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If input format is invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Vérification de la présence de la colonne temporelle ou d'un index datetime
        if self.time_col in X.columns:
            # Conversion de la colonne temporelle en datetime si nécessaire
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                X[self.time_col] = pd.to_datetime(X[self.time_col])
        elif isinstance(X.index, pd.DatetimeIndex):
            # L'index est déjà un DatetimeIndex
            pass
        else:
            # Tentative de conversion de l'index en datetime
            try:
                X.index = pd.to_datetime(X.index)
            except:
                raise ValueError(
                    f"Time column '{self.time_col}' not found and index "
                    "cannot be converted to datetime"
                )
        
        # Vérification des colonnes panel si spécifiées
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found: {missing_cols}")
        
        return X
    
    def _set_prediction_date(self, X: pd.DataFrame) -> None:
        """Set the effective prediction date.
        
        Args:
            X: Input data for date reference
        """
        if self.prediction_date == 'today':
            self.prediction_date_ = pd.Timestamp.today()
        else:
            self.prediction_date_ = pd.Timestamp(self.prediction_date)
        
        # Ajustement de la date de prédiction selon la fréquence des données
        if self.time_col in X.columns:
            time_series = X[self.time_col]
        else:
            time_series = X.index
        
        # Vérification que la date de prédiction n'est pas dans le futur des données
        max_date = time_series.max()
        if self.prediction_date_ > max_date:
            warnings.warn(
                f"Prediction date {self.prediction_date_} is beyond data range. "
                f"Using last available date {max_date}"
            )
            self.prediction_date_ = max_date
    
    def _compute_delays(self, X: pd.DataFrame) -> Dict[Union[str, Tuple], int]:
        """Compute publication delays for each series.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary mapping series identifiers to delay counts
        """
        delays = {}
        
        if self.is_panel_:
            # Calcul des délais pour chaque combinaison panel/variable
            for panel_values, group_df in X.groupby(self.panel_cols):
                panel_id = tuple(panel_values) if len(self.panel_cols) > 1 else panel_values
                
                # Pour chaque variable du groupe
                for col in X.columns:
                    if col not in self.panel_cols and col != self.time_col:
                        delay = self._calculate_single_delay(group_df[col], group_df)
                        delays[(panel_id, col)] = delay
        else:
            # Calcul des délais pour chaque colonne
            for col in X.columns:
                if col != self.time_col:
                    delay = self._calculate_single_delay(X[col], X)
                    delays[col] = delay
        
        return delays
    
    def _calculate_single_delay(self, 
                               series: pd.Series, 
                               df: pd.DataFrame) -> int:
        """Calculate delay for a single series.
        
        Args:
            series: Time series to analyze
            df: DataFrame containing time information
            
        Returns:
            Number of periods to shift
        """
        # Récupération de l'index temporel
        if self.time_col in df.columns:
            time_index = df[self.time_col]
        else:
            time_index = df.index
        
        # Tri par date et association avec la série
        sorted_idx = time_index.argsort()
        sorted_series = series.iloc[sorted_idx]
        sorted_time = time_index.iloc[sorted_idx]
        
        # Recherche de la dernière valeur non-NaN avant ou à la date de prédiction
        mask = sorted_time <= self.prediction_date_
        valid_mask = mask & sorted_series.notna()
        
        if not valid_mask.any():
            # Aucune donnée valide avant la date de prédiction
            return 0
        
        # Indice de la dernière observation valide
        last_valid_idx = valid_mask[::-1].idxmax()
        last_valid_date = sorted_time.iloc[last_valid_idx]
        
        # Calcul du nombre de périodes entre la dernière observation et la date de prédiction
        # Basé sur la fréquence détectée de la série
        series_key = self._get_series_key(series, df)
        frequency = self.frequency_map_.get(series_key, self.data_frequency_)
        
        # Calcul du délai en nombre de périodes
        delay_periods = self._count_periods_between(
            last_valid_date, self.prediction_date_, frequency
        )
        
        return delay_periods
    
    def _count_periods_between(self,
                              start_date: pd.Timestamp,
                              end_date: pd.Timestamp,
                              frequency: str) -> int:
        """Count the number of periods between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Frequency of the periods
            
        Returns:
            Number of complete periods
        """
        if frequency == 'daily':
            return (end_date - start_date).days
        elif frequency == 'weekly':
            return (end_date - start_date).days // 7
        elif frequency == 'monthly':
            # Calcul du nombre de mois complets
            months = (end_date.year - start_date.year) * 12
            months += end_date.month - start_date.month
            # Ajustement si le jour du mois n'est pas atteint
            if end_date.day < start_date.day:
                months -= 1
            return max(0, months)
        elif frequency == 'quarterly':
            # Calcul du nombre de trimestres complets
            quarters = (end_date.year - start_date.year) * 4
            quarters += (end_date.quarter - start_date.quarter)
            return max(0, quarters)
        elif frequency == 'annual':
            return max(0, end_date.year - start_date.year)
        else:
            # Fréquence inconnue, utilisation d'une approximation
            warnings.warn(f"Unknown frequency {frequency}, using daily approximation")
            return (end_date - start_date).days
    
    def _get_series_key(self, series: pd.Series, df: pd.DataFrame) -> Union[str, Tuple]:
        """Get the key identifying a series in the frequency map.
        
        Args:
            series: Series to identify
            df: DataFrame containing the series
            
        Returns:
            Key for the series
        """
        if self.is_panel_:
            # Récupération des valeurs du panel pour cette série
            panel_values = []
            for col in self.panel_cols:
                unique_vals = df[col].unique()
                if len(unique_vals) == 1:
                    panel_values.append(unique_vals[0])
            
            panel_id = tuple(panel_values) if len(panel_values) > 1 else panel_values[0]
            return (panel_id, series.name)
        else:
            return series.name
    
    def _compute_frequency_adjustments(self, X: pd.DataFrame) -> Dict[str, int]:
        """Compute frequency adjustment shifts.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary of frequency adjustments
        """
        adjustments = {}
        
        # Pour chaque fréquence détectée différente de la fréquence principale
        for series_key, freq in self.frequency_map_.items():
            if freq != self.data_frequency_:
                # Calcul de l'ajustement nécessaire
                offset = self._freq_converter.get_period_offset(
                    self.prediction_date_,
                    freq,
                    self.data_frequency_
                )
                adjustments[series_key] = offset
        
        return adjustments
    
    def _compute_reference_dates(self, X: pd.DataFrame) -> Dict[Union[str, Tuple], pd.Timestamp]:
        """Compute reference dates for each series.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary mapping series to their reference dates
        """
        reference_dates = {}
        
        # Calcul de la date de référence selon la fréquence principale
        if self.data_frequency_ == 'quarterly':
            # Premier jour du trimestre en cours
            ref_date = pd.Timestamp(
                year=self.prediction_date_.year,
                month=((self.prediction_date_.quarter - 1) * 3) + 1,
                day=1
            )
        elif self.data_frequency_ == 'monthly':
            # Premier jour du mois en cours
            ref_date = pd.Timestamp(
                year=self.prediction_date_.year,
                month=self.prediction_date_.month,
                day=1
            )
        elif self.data_frequency_ == 'annual':
            # Premier jour de l'année en cours
            ref_date = pd.Timestamp(
                year=self.prediction_date_.year,
                month=1,
                day=1
            )
        else:
            # Pour les autres fréquences, utilisation de la date de prédiction
            ref_date = self.prediction_date_
        
        # Attribution de la date de référence à chaque série
        for key in self.release_delays_.keys():
            reference_dates[key] = ref_date
        
        return reference_dates
    
    def _apply_delays_series(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply delays to univariate time series data.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        X_shifted = X.copy()
        
        for col, delay in self.release_delays_.items():
            if col in X.columns and delay > 0:
                # Application du décalage temporel
                X_shifted[col] = X_shifted[col].shift(-delay)
                
                # Ajustement supplémentaire si changement de fréquence
                if col in self.frequency_shifts_:
                    additional_shift = self.frequency_shifts_[col]
                    if additional_shift != 0:
                        X_shifted[col] = X_shifted[col].shift(-additional_shift)
        
        return X_shifted
    
    def _apply_delays_panel(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply delays to panel data.
        
        Args:
            X: Panel data to transform
            
        Returns:
            Transformed panel data
        """
        X_shifted = X.copy()
        
        # Application des délais par groupe panel
        for panel_values, group_df in X.groupby(self.panel_cols):
            panel_id = tuple(panel_values) if len(self.panel_cols) > 1 else panel_values
            group_idx = group_df.index
            
            # Application des délais pour chaque variable du groupe
            for col in X.columns:
                if col not in self.panel_cols and col != self.time_col:
                    key = (panel_id, col)
                    
                    if key in self.release_delays_:
                        delay = self.release_delays_[key]
                        
                        if delay > 0:
                            # Décalage de la série
                            shifted_values = group_df[col].shift(-delay)
                            
                            # Ajustement de fréquence si nécessaire
                            if key in self.frequency_shifts_:
                                additional_shift = self.frequency_shifts_[key]
                                if additional_shift != 0:
                                    shifted_values = shifted_values.shift(-additional_shift)
                            
                            # Mise à jour des valeurs dans le DataFrame principal
                            X_shifted.loc[group_idx, col] = shifted_values
        
        return X_shifted
    
    def _reverse_delays_series(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse delays for univariate time series.
        
        Args:
            X: Transformed data
            
        Returns:
            Original data structure
        """
        X_original = X.copy()
        
        for col, delay in self.release_delays_.items():
            if col in X.columns and delay > 0:
                # Inversion du décalage
                total_shift = delay
                
                # Ajout de l'ajustement de fréquence
                if col in self.frequency_shifts_:
                    total_shift += self.frequency_shifts_[col]
                
                X_original[col] = X_original[col].shift(total_shift)
        
        return X_original
    
    def _reverse_delays_panel(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse delays for panel data.
        
        Args:
            X: Transformed panel data
            
        Returns:
            Original panel data structure
        """
        X_original = X.copy()
        
        # Inversion des délais par groupe panel
        for panel_values, group_df in X.groupby(self.panel_cols):
            panel_id = tuple(panel_values) if len(self.panel_cols) > 1 else panel_values
            group_idx = group_df.index
            
            for col in X.columns:
                if col not in self.panel_cols and col != self.time_col:
                    key = (panel_id, col)
                    
                    if key in self.release_delays_:
                        delay = self.release_delays_[key]
                        
                        if delay > 0:
                            # Calcul du décalage total
                            total_shift = delay
                            if key in self.frequency_shifts_:
                                total_shift += self.frequency_shifts_[key]
                            
                            # Inversion du décalage
                            original_values = group_df[col].shift(total_shift)
                            X_original.loc[group_idx, col] = original_values
        
        return X_original
    
    def get_publication_schedule(self) -> pd.DataFrame:
        """Get a summary of publication delays for all series.
        
        Returns:
            DataFrame with publication delay information
        """
        check_is_fitted(self, ['release_delays_'])
        
        records = []
        for key, delay in self.release_delays_.items():
            if self.is_panel_:
                panel_id, variable = key
                record = {
                    'panel_id': str(panel_id),
                    'variable': variable,
                    'delay_periods': delay,
                    'frequency': self.frequency_map_.get(key, self.data_frequency_),
                    'reference_date': self.reference_dates_.get(key)
                }
            else:
                record = {
                    'variable': key,
                    'delay_periods': delay,
                    'frequency': self.frequency_map_.get(key, self.data_frequency_),
                    'reference_date': self.reference_dates_.get(key)
                }
            records.append(record)
        
        return pd.DataFrame(records)