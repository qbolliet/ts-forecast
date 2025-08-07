"""Frequency detection utilities for time series data.

This module provides functions to detect and validate frequencies in time series
and panel data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from pandas.tseries.frequencies import to_offset


class FrequencyDetector:
    """Detect frequency of time series data.
    
    This class provides methods to detect the frequency of individual series
    and validate frequency consistency across datasets.
    
    Attributes:
        frequency_mapping (Dict[str, str]): Mapping between detected and standard frequencies
        min_observations (int): Minimum observations required for frequency detection
    """
    
    def __init__(self, min_observations: int = 10):
        """Initialize the FrequencyDetector.
        
        Args:
            min_observations: Minimum number of observations required to detect frequency
        """
        self.min_observations = min_observations
        
        # Mapping des fréquences pandas vers des noms standardisés
        self.frequency_mapping = {
            'D': 'daily',
            'B': 'business_daily', 
            'W': 'weekly',
            'M': 'monthly',
            'MS': 'monthly',
            'Q': 'quarterly',
            'QS': 'quarterly',
            'A': 'annual',
            'AS': 'annual',
            'Y': 'annual',
            'YS': 'annual'
        }
        
        # Mapping inverse pour les conversions
        self.standard_to_pandas = {
            'daily': 'D',
            'business_daily': 'B',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'annual': 'A'
        }
    
    def detect_frequency(self, 
                        series: pd.Series, 
                        index_col: Optional[str] = None) -> Optional[str]:
        """Detect the frequency of a time series.
        
        Args:
            series: Time series data
            index_col: Name of the index column if series is part of a DataFrame
            
        Returns:
            Detected frequency as a string, or None if detection fails
            
        Raises:
            ValueError: If series has insufficient non-null observations
        """
        # Suppression des valeurs manquantes pour la détection
        clean_series = series.dropna()
        
        if len(clean_series) < self.min_observations:
            raise ValueError(
                f"Series has only {len(clean_series)} non-null observations, "
                f"minimum required is {self.min_observations}"
            )
        
        # Récupération de l'index temporel
        if isinstance(clean_series.index, pd.DatetimeIndex):
            time_index = clean_series.index
        elif index_col and hasattr(clean_series, index_col):
            time_index = pd.to_datetime(clean_series[index_col])
        else:
            # Tentative de conversion de l'index en datetime
            try:
                time_index = pd.to_datetime(clean_series.index)
            except:
                return None
        
        # Tri de l'index temporel
        time_index = time_index.sort_values()
        
        # Calcul des différences entre observations consécutives
        time_diffs = pd.Series(time_index).diff().dropna()
        
        # Identification de la fréquence modale (la plus fréquente)
        mode_diff = time_diffs.mode()
        
        if len(mode_diff) == 0:
            return None
            
        # Conversion de la différence modale en fréquence pandas
        try:
            inferred_freq = pd.infer_freq(time_index)
            if inferred_freq:
                # Extraction du code de fréquence principal
                freq_code = ''.join(filter(str.isalpha, inferred_freq))
                return self.frequency_mapping.get(freq_code, inferred_freq)
        except:
            pass
        
        # Détection manuelle basée sur les différences si infer_freq échoue
        modal_days = mode_diff[0].days
        
        if modal_days == 1:
            return 'daily'
        elif modal_days == 7:
            return 'weekly'
        elif 28 <= modal_days <= 31:
            return 'monthly'
        elif 89 <= modal_days <= 92:
            return 'quarterly'
        elif 365 <= modal_days <= 366:
            return 'annual'
        
        return None
    
    def detect_dataset_frequency(self, 
                               df: pd.DataFrame,
                               time_col: Optional[str] = None,
                               panel_cols: Optional[List[str]] = None) -> Dict[str, str]:
        """Detect frequencies for all series in a dataset.
        
        Args:
            df: DataFrame containing time series data
            time_col: Name of the time column
            panel_cols: List of columns identifying panel dimensions
            
        Returns:
            Dictionary mapping column names (or (panel_id, column) tuples) to frequencies
            
        Raises:
            ValueError: If frequencies are inconsistent across the dataset
        """
        frequency_map = {}
        
        # Détermination si les données sont en panel
        is_panel = panel_cols is not None and len(panel_cols) > 0
        
        if is_panel:
            # Traitement des données panel
            for panel_values, group_df in df.groupby(panel_cols):
                # Création de l'identifiant du panel
                if len(panel_cols) == 1:
                    panel_id = panel_values
                else:
                    panel_id = tuple(panel_values)
                
                # Détection de la fréquence pour chaque colonne du groupe
                for col in df.columns:
                    if col not in panel_cols and col != time_col:
                        try:
                            freq = self.detect_frequency(group_df[col])
                            if freq:
                                frequency_map[(panel_id, col)] = freq
                        except ValueError:
                            # Pas assez d'observations pour cette série
                            continue
        else:
            # Traitement des séries temporelles simples
            for col in df.columns:
                if col != time_col:
                    try:
                        freq = self.detect_frequency(df[col])
                        if freq:
                            frequency_map[col] = freq
                    except ValueError:
                        continue
        
        return frequency_map
    
    def validate_frequency_consistency(self, 
                                     frequency_map: Dict[str, str],
                                     strict: bool = True) -> Tuple[bool, Optional[str]]:
        """Validate that all series have consistent frequencies.
        
        Args:
            frequency_map: Dictionary of detected frequencies
            strict: If True, all frequencies must be identical
            
        Returns:
            Tuple of (is_consistent, common_frequency)
        """
        if not frequency_map:
            return False, None
        
        unique_frequencies = set(frequency_map.values())
        
        if len(unique_frequencies) == 1:
            return True, list(unique_frequencies)[0]
        
        if strict:
            return False, None
        
        # En mode non-strict, retourner la fréquence la plus commune
        freq_counts = {}
        for freq in frequency_map.values():
            freq_counts[freq] = freq_counts.get(freq, 0) + 1
        
        most_common = max(freq_counts, key=freq_counts.get)
        return True, most_common


class FrequencyConverter:
    """Handle conversions between different time frequencies.
    
    This class manages the relationships between different time frequencies
    and provides methods to convert between them.
    
    Attributes:
        conversion_factors (Dict[Tuple[str, str], int]): Conversion factors between frequencies
        frequency_order (Dict[str, int]): Ordering of frequencies from highest to lowest
    """
    
    def __init__(self):
        """Initialize the FrequencyConverter with standard conversion factors."""
        # Facteurs de conversion entre fréquences
        # Format: (from_freq, to_freq): factor
        self.conversion_factors = {
            ('daily', 'weekly'): 7,
            ('daily', 'monthly'): 30,  # Approximation
            ('daily', 'quarterly'): 91,  # Approximation  
            ('daily', 'annual'): 365,
            ('weekly', 'monthly'): 4,  # Approximation
            ('weekly', 'quarterly'): 13,
            ('weekly', 'annual'): 52,
            ('monthly', 'quarterly'): 3,
            ('monthly', 'annual'): 12,
            ('quarterly', 'annual'): 4,
        }
        
        # Ajout des conversions inverses
        inverse_conversions = {}
        for (from_freq, to_freq), factor in self.conversion_factors.items():
            inverse_conversions[(to_freq, from_freq)] = 1 / factor
        self.conversion_factors.update(inverse_conversions)
        
        # Ordre des fréquences (du plus granulaire au moins granulaire)
        self.frequency_order = {
            'daily': 1,
            'business_daily': 1.5,
            'weekly': 2,
            'monthly': 3,
            'quarterly': 4,
            'annual': 5
        }
    
    def get_conversion_factor(self, from_freq: str, to_freq: str) -> float:
        """Get the conversion factor between two frequencies.
        
        Args:
            from_freq: Source frequency
            to_freq: Target frequency
            
        Returns:
            Conversion factor as a float
            
        Raises:
            ValueError: If conversion is not possible
        """
        if from_freq == to_freq:
            return 1.0
        
        key = (from_freq, to_freq)
        if key in self.conversion_factors:
            return self.conversion_factors[key]
        
        # Tentative de conversion en passant par une fréquence intermédiaire
        for intermediate_freq in self.frequency_order.keys():
            if (from_freq, intermediate_freq) in self.conversion_factors and \
               (intermediate_freq, to_freq) in self.conversion_factors:
                return (self.conversion_factors[(from_freq, intermediate_freq)] * 
                       self.conversion_factors[(intermediate_freq, to_freq)])
        
        raise ValueError(f"No conversion available from {from_freq} to {to_freq}")
    
    def is_higher_frequency(self, freq1: str, freq2: str) -> bool:
        """Check if freq1 is a higher frequency than freq2.
        
        Args:
            freq1: First frequency
            freq2: Second frequency
            
        Returns:
            True if freq1 is higher frequency than freq2
        """
        return self.frequency_order.get(freq1, 0) < self.frequency_order.get(freq2, 0)
    
    def get_period_offset(self, 
                         date: pd.Timestamp, 
                         from_freq: str, 
                         to_freq: str) -> int:
        """Calculate period offset when changing frequencies.
        
        This method calculates how many periods to adjust when converting
        from one frequency to another at a specific date.
        
        Args:
            date: Reference date
            from_freq: Source frequency  
            to_freq: Target frequency
            
        Returns:
            Number of periods to offset
        """
        if from_freq == to_freq:
            return 0
            
        # Conversion de la date vers le début de la période cible
        if to_freq == 'quarterly':
            target_period_start = pd.Timestamp(year=date.year, 
                                             month=((date.quarter - 1) * 3) + 1, 
                                             day=1)
        elif to_freq == 'monthly':
            target_period_start = pd.Timestamp(year=date.year, 
                                             month=date.month, 
                                             day=1)
        elif to_freq == 'annual':
            target_period_start = pd.Timestamp(year=date.year, month=1, day=1)
        else:
            target_period_start = date
        
        # Calcul du décalage en périodes source
        if from_freq == 'monthly' and to_freq == 'quarterly':
            # Nombre de mois depuis le début du trimestre
            months_offset = date.month - target_period_start.month
            return months_offset
        elif from_freq == 'daily' and to_freq == 'monthly':
            # Nombre de jours depuis le début du mois
            days_offset = date.day - 1
            return days_offset
        elif from_freq == 'daily' and to_freq == 'quarterly':
            # Nombre de jours depuis le début du trimestre
            days_offset = (date - target_period_start).days
            return days_offset
        
        return 0
    
    def align_to_frequency(self, 
                          dates: pd.DatetimeIndex, 
                          target_freq: str) -> pd.DatetimeIndex:
        """Align dates to a target frequency.
        
        Args:
            dates: DatetimeIndex to align
            target_freq: Target frequency
            
        Returns:
            Aligned DatetimeIndex
        """
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'MS',  # Month start
            'quarterly': 'QS',  # Quarter start
            'annual': 'AS'  # Year start
        }
        
        if target_freq not in freq_map:
            return dates
            
        # Création d'un index régulier à la fréquence cible
        start = dates.min()
        end = dates.max()
        
        return pd.date_range(start=start, end=end, freq=freq_map[target_freq])