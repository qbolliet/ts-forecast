"""Frequency detection utilities for time series data.

This module provides the FrequencyDetector class to detect and validate frequencies
in time series and panel data, with primary reliance on pandas.infer_freq.
"""
# Importation des modules
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from pandas.tseries.frequencies import to_offset

# Import des utilitaires de fréquence
from .utils import normalize_frequency, to_user_friendly, get_base_frequency


class FrequencyDetector:
    """Detect frequency of time series data.

    This class provides methods to detect the frequency of individual series
    and validate frequency consistency across datasets, with primary reliance
    on pandas.infer_freq and extensions for missing frequencies.

    Attributes:
        min_observations (int): Minimum observations required for frequency detection

    Examples:
        >>> detector = FrequencyDetector()
        >>> dates = pd.date_range('2023-01-01', periods=12, freq='M')
        >>> series = pd.Series(range(12), index=dates)
        >>> detector.detect_frequency(series)
        'monthly'
    """

    def __init__(self, min_observations: int = 2):
        """Initialize the FrequencyDetector.

        Args:
            min_observations: Minimum number of observations required to detect frequency
        """
        self.min_observations = min_observations

    def detect_frequency(self, series: pd.Series) -> Optional[str]:
        """Detect the frequency of a time series.

        This method primarily uses pandas.infer_freq with extensions for frequencies
        not natively supported (like quarterly). It automatically drops NaN values
        before detection.

        Args:
            series: Time series data with datetime index

        Returns:
            Detected frequency as user-friendly string, or None if detection fails

        Raises:
            ValueError: If series has insufficient non-null observations or invalid index

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> series = pd.Series([1, 2, np.nan, 4, 5], index=dates)
            >>> detector = FrequencyDetector()
            >>> detector.detect_frequency(series)
            'daily'
        """
        # Suppression des valeurs manquantes pour la détection
        clean_series = series.dropna()

        if len(clean_series) < self.min_observations:
            raise ValueError(
                f"Series has only {len(clean_series)} non-null observations, "
                f"minimum required is {self.min_observations}"
            )

        # Vérification que l'index est de type datetime
        if not isinstance(clean_series.index, pd.DatetimeIndex):
            # Tentative de conversion de l'index en datetime
            try:
                time_index = pd.to_datetime(clean_series.index)
            except (ValueError, TypeError):
                raise ValueError("Series index cannot be converted to datetime")
        else:
            time_index = clean_series.index

        # Tri de l'index temporel pour assurer la cohérence
        if not time_index.is_monotonic_increasing:
            time_index = time_index.sort_values()

        # Utilisation principale de pandas.infer_freq
        try:
            inferred_freq = pd.infer_freq(time_index)
            if inferred_freq:
                # Conversion vers le format convivial
                return to_user_friendly(inferred_freq)
        except Exception:
            # En cas d'erreur avec infer_freq, continuer avec la détection manuelle
            pass

        # Extension pour les fréquences non supportées par infer_freq
        extended_freq = self._extend_infer_freq(time_index)
        if extended_freq:
            return extended_freq

        return None

    def detect_dataset_frequency(self,
                               df: pd.DataFrame,
                               time_col: Optional[str] = None,
                               panel_cols: Optional[List[str]] = None) -> Dict[str, str]:
        """Detect frequencies for all series in a dataset.

        Args:
            df: DataFrame containing time series data
            time_col: Name of the time column (if None, uses index)
            panel_cols: List of columns identifying panel dimensions

        Returns:
            Dictionary mapping column names (or (panel_id, column) tuples) to frequencies

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
            >>> df = pd.DataFrame({'value1': [1, 2, 3, 4, 5], 'value2': [10, 20, 30, 40, 50]}, index=dates)
            >>> detector = FrequencyDetector()
            >>> freq_map = detector.detect_dataset_frequency(df)
            >>> freq_map
            {'value1': 'daily', 'value2': 'daily'}
        """
        frequency_map = {}

        # Préparation de l'index temporel si spécifié
        if time_col is not None and time_col in df.columns:
            df = df.set_index(time_col)

        # Détermination si les données sont en panel
        is_panel = panel_cols is not None and len(panel_cols) > 0

        if is_panel:
            # Traitement des données panel
            for panel_values, group_df in df.groupby(panel_cols):
                # Création de l'identifiant du panel
                panel_id = panel_values if len(panel_cols) == 1 else tuple(panel_values)

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

        Examples:
            >>> detector = FrequencyDetector()
            >>> freq_map = {'series1': 'daily', 'series2': 'daily'}
            >>> is_consistent, common_freq = detector.validate_frequency_consistency(freq_map)
            >>> is_consistent, common_freq
            (True, 'daily')
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

    def _extend_infer_freq(self, time_index: pd.DatetimeIndex) -> Optional[str]:
        """Extend frequency detection for frequencies not supported by pandas.infer_freq.

        This method provides custom detection for frequencies like quarterly that
        pandas.infer_freq doesn't always detect reliably.

        Args:
            time_index: Sorted datetime index

        Returns:
            Extended frequency detection result or None
        """
        if len(time_index) < self.min_observations:
            return None

        # Calcul des différences entre observations consécutives
        time_diffs = pd.Series(time_index).diff().dropna()

        if len(time_diffs) == 0:
            return None

        # Identification de la différence modale (la plus fréquente)
        mode_diff = time_diffs.mode()

        if len(mode_diff) == 0:
            return None

        modal_days = mode_diff[0].days

        # Détection basée sur le nombre de jours modal
        if modal_days == 1:
            return 'daily'
        elif modal_days == 7:
            return 'weekly'
        elif 28 <= modal_days <= 31:
            return 'monthly'
        elif 89 <= modal_days <= 92:
            # Détection spéciale pour les fréquences trimestrielles
            return self._detect_quarterly_frequency(time_index)
        elif 365 <= modal_days <= 366:
            return 'annual'

        return None

    def _detect_quarterly_frequency(self, time_index: pd.DatetimeIndex) -> Optional[str]:
        """Detect quarterly frequency with reference point.

        Args:
            time_index: Sorted datetime index

        Returns:
            Quarterly frequency with reference point or None
        """
        if len(time_index) < 2:
            return None

        # Vérification des mois pour déterminer le point de référence
        months = time_index.month
        days = time_index.day

        # Si tous les mois correspondent au début de trimestre et jour = 1
        if all(month in [1, 4, 7, 10] for month in months) and all(day == 1 for day in days):
            return 'quarterly_start'

        # Si tous les mois correspondent à la fin de trimestre et jour = dernier du mois
        if all(month in [3, 6, 9, 12] for month in months):
            # Vérification du dernier jour du mois
            last_days = []
            for date in time_index:
                if date.month == 12:
                    last_day = 31
                elif date.month in [4, 6, 9, 11]:
                    last_day = 30
                elif date.month == 2:
                    last_day = 29 if date.year % 4 == 0 else 28
                else:
                    last_day = 31
                last_days.append(last_day)

            if all(date.day == last_day for date, last_day in zip(time_index, last_days)):
                return 'quarterly_end'

        # Par défaut, retourner quarterly sans référence spécifique
        return 'quarterly'